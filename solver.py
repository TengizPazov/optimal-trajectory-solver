import numpy as np
import pickle
import logging
from pathlib import Path
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares, minimize_scalar
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from approximations import T_a, get_initial_adjoint

logger = logging.getLogger(__name__)


class OptimalControlSolver:
    def __init__(self, mu: float, a_max: float, alpha: float, func_type: str):
        self.mu = mu
        self.a_max = a_max
        self.alpha = alpha
        self.func_type = func_type
        self.free_a_max = True
        self.trajectory = None
        self.step_floor_history = []
        self._unbounded_quadratic = False

    def set_boundary_conditions(self, r0, v0, rf, vf, T, *, lambda_value=None):
        self.r0 = np.array(r0, dtype=np.float64)
        self.v0 = np.array(v0, dtype=np.float64)
        self.rf = np.array(rf, dtype=np.float64)
        self.vf = np.array(vf, dtype=np.float64)
        self.T = T
        self.lambda_value = lambda_value
        self.kappa = float(np.linalg.norm(self.rf) / np.linalg.norm(self.r0))

    def initial_guess(self):
        """Аппроксимация для заданной дальности; число оборотов задаётся явно."""
        if self.lambda_value is None:
            raise ValueError('Для автоматического старта задайте lambda_value '
                             'в set_boundary_conditions или передайте p0_guess')
        if (not np.isfinite(self.lambda_value) or self.lambda_value <= 0
                or not np.isfinite(self.kappa) or self.kappa <= 0):
            raise ValueError('lambda_value и отношение радиусов kappa должны быть положительными')
        pr0, pv0 = get_initial_adjoint(self.lambda_value, self.kappa)
        a_raw = max(float(np.linalg.norm(pv0)), 0.1)
        # Проецируем только автоматически выбранную оценку, не меняя p0_guess пользователя.
        a_initial = float(np.clip(a_raw, 1e-4, 1.0))
        self.initial_guess_info = dict(lambda_value=self.lambda_value, kappa=self.kappa,
                                       a_max_raw=a_raw, a_max_initial=a_initial)
        return np.hstack((pr0, pv0, a_initial))

    def optimal_control(self, pv):

        if self._unbounded_quadratic:
            return np.asarray(pv, dtype=float)

        q = float(np.linalg.norm(pv))
        if q < 1e-14:
            return np.zeros(3)

        if self.func_type == 'quadratic':
            alpha = 0.0
        elif self.func_type == 'linear':
            alpha = 1.0
        elif self.func_type == 'mixed':
            alpha = self.alpha
        else:
            raise ValueError(f"unknown func_type: {self.func_type}")

        if alpha >= 1.0 - 1e-12:
            b = self.a_max if q > 1.0 else 0.0
        else:
            b = (q - alpha) / (1.0 - alpha)
            b = min(self.a_max, max(0.0, b))
        return b * pv / q

    def system_equations(self, t, y):
        r = y[0:3]
        v = y[3:6]
        pr = y[6:9]
        pv = y[9:12]

        r_norm = np.linalg.norm(r)
        if r_norm < 1e-12:
            r_norm = 1e-12

        drdt = v
        a = self.optimal_control(pv)
        dvdt = -self.mu * r / r_norm**3 + a
        dot_pr = (self.mu / r_norm**3) * pv - (3 * self.mu / r_norm**5) * np.dot(r, pv) * r
        dot_pv = -pr

        return np.concatenate([drdt, dvdt, dot_pr, dot_pv])

    def integrate_trajectory(self, p0, t_span=None, *, dense_output=False):
        if t_span is None:
            t_span = [0, self.T]
        y0 = np.concatenate([self.r0, self.v0, p0[0:3], p0[3:6]])
        sol = solve_ivp(
            self.system_equations,
            t_span,
            y0,
            method='LSODA',
            rtol=1e-9,
            atol=1e-12,
            dense_output=dense_output,
        )
        return sol

    def residual_vec(self, x):
        """
        Вектор невязки конечных условий.
        x = [pr0_x, pr0_y, pr0_z, pv0_x, pv0_y, pv0_z]
        """
        pr0 = x[0:3]
        pv0 = x[3:6]
        if self.free_a_max:
            self.a_max = x[6]
        a_max = self.a_max

        y0 = np.concatenate([self.r0, self.v0, pr0, pv0])
        sol = solve_ivp(
            self.system_equations,
            [0, self.T],
            y0,
            method='LSODA',
            rtol=1e-9,
            atol=1e-12,
        )

        res = self.trajectory_residual(sol)
        if not np.all(np.isfinite(res)):
            logger.info("solve_ivp не сошёлся")
            return np.ones(6) * 1e3

        res_norm = np.linalg.norm(res)
        logger.info(f"α={self.alpha:.3f} | невязка={res_norm:.6e} | a_max={a_max:.6f}")
        return res

    def trajectory_residual(self, trajectory):
        """Невязка только для конечной, успешно интегрированной траектории."""
        if (trajectory is None or not trajectory.success
                or len(trajectory.t) == 0
                or not np.all(np.isfinite(trajectory.t))
                or not np.all(np.isfinite(trajectory.y))
                or not np.isclose(trajectory.t[-1], self.T, rtol=1e-10, atol=1e-12)):
            return np.full(6, np.inf)
        return np.hstack((self.rf - trajectory.y[0:3, -1],
                          self.vf - trajectory.y[3:6, -1]))

    @staticmethod
    def peak_quadratic_thrust(trajectory):
        """Численный максимум |p_v|, с уточнением между узлами интегратора."""
        peak = float(np.max(np.linalg.norm(trajectory.y[9:12], axis=0)))
        peak_time = float(trajectory.t[np.argmax(np.linalg.norm(trajectory.y[9:12], axis=0))])
        for left, right in zip(trajectory.t[:-1], trajectory.t[1:]):
            candidate = minimize_scalar(
                lambda t: -float(np.linalg.norm(trajectory.sol(t)[9:12])),
                bounds=(left, right), method='bounded',
                options={'xatol': max(1e-14, (right - left) * 1e-10)})
            if candidate.success and -candidate.fun > peak:
                peak, peak_time = float(-candidate.fun), float(candidate.x)
        return peak, peak_time

    def initialize_quadratic(self, p0_guess=None, tol=1e-2):
        """Уточнить неограниченное решение и поставить A на максимум его тяги."""
        seed = self.initial_guess() if p0_guess is None else np.asarray(p0_guess, dtype=float)
        previous_free = self.free_a_max
        self.initial_quadratic_info = {'status': 'failed'}
        self.trajectory = None
        try:
            self.free_a_max = False
            self._unbounded_quadratic = True
            result = least_squares(
                self.residual_vec, seed[:6], method='trf',
                xtol=1e-10, ftol=1e-10, gtol=1e-10, max_nfev=200)
            unbounded = self.integrate_trajectory(result.x, dense_output=True) if result.success else None
            error = float(np.linalg.norm(self.trajectory_residual(unbounded)))
        finally:
            self.free_a_max = previous_free
            self._unbounded_quadratic = False

        if not result.success or not np.isfinite(error) or error > tol:
            result.success = False
            result.message = f'Не принято начальное неограниченное решение: {result.message}; невязка={error}'
        else:
            peak, peak_time = self.peak_quadratic_thrust(unbounded)
            self.initial_quadratic_info.update(unbounded_residual=error,
                                               peak_thrust=peak, peak_time=peak_time)
            if not np.isfinite(peak) or peak > 1.0:
                result.success = False
                result.message = 'Максимум начальной тяги не укладывается в существующую границу A<=1'
            else:
                self.p0 = result.x.copy()
                self.a_max = max(peak, 1e-4)
                bounded = self.integrate_trajectory(self.p0)
                bounded_error = float(np.linalg.norm(self.trajectory_residual(bounded)))
                deviation = (float(np.max(np.linalg.norm(
                    bounded.y[:6] - unbounded.sol(bounded.t)[:6], axis=0)))
                    if np.isfinite(bounded_error) else np.inf)
                self.initial_quadratic_info.update(
                    a_max=self.a_max, bounded_residual=bounded_error,
                    maximum_state_deviation=deviation, verification_tolerance=1e-7)
                if bounded_error <= tol and deviation <= 1e-7:
                    self.trajectory = bounded
                    self.initial_quadratic_info['status'] = 'accepted'
                    result.fun = self.trajectory_residual(bounded)
                    result.cost = float(0.5 * np.dot(result.fun, result.fun))
                else:
                    result.success = False
                    result.message = 'Ограничение изменило начальное квадратичное решение сверх допуска проверки'

        # Сохраняем семикомпонентный интерфейс результата для свободного A.
        result.x = np.r_[result.x, self.a_max]
        result.jac = np.column_stack((result.jac, np.zeros(6)))
        result.grad = np.r_[result.grad, 0.0]
        result.active_mask = np.r_[result.active_mask, 0]
        return result

    def solve(self, p0_guess=None, tol=1e-2):
        if (self.free_a_max and abs(self.alpha) <= 1e-12
                and self.func_type in ('mixed', 'quadratic')):
            return self.initialize_quadratic(p0_guess, tol=tol)
        if p0_guess is None:
            p0_guess = self.initial_guess()

        p0_guess = np.asarray(p0_guess, dtype=float)
        if self.free_a_max:
            if len(p0_guess) == 6:
                p0_guess = np.hstack((p0_guess, self.a_max))
            bounds = ([-np.inf] * 6 + [1e-4], [np.inf] * 6 + [1.0])
        else:
            p0_guess = p0_guess[:6]
            bounds = (-np.inf, np.inf)

        result = least_squares(
            self.residual_vec,
            p0_guess,
            method='trf',
            bounds=bounds,
            # Проверенный профиль: возмущение выше шума интегрирования LSODA.
            jac='2-point',
            diff_step=1e-5,
            x_scale=1.0,
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
            max_nfev=200,
            verbose=0
        )

        if result.success:
            pr0_opt = result.x[0:3]
            pv0_opt = result.x[3:6]
            a_max_opt = result.x[6] if self.free_a_max else self.a_max
            self.p0 = np.hstack((pr0_opt, pv0_opt))
            self.a_max = a_max_opt
            self.trajectory = self.integrate_trajectory(self.p0)
            err = np.linalg.norm(self.trajectory_residual(self.trajectory))
            if not np.isfinite(err) or err > tol:
                result.success = False
                result.message = f"Траектория не прошла проверку: невязка={err}, допуск={tol}"
                self.trajectory = None
                logger.warning(result.message)
            else:
                logger.info(f"Решение принято при α={self.alpha:.3f}, a_max={a_max_opt:.6f}, невязка={err:.6e}")
        else:
            logger.warning(f"Оптимизация не сошлась при α={self.alpha:.3f}")
            self.trajectory = None

        return result

    def continuation_method(self,
                            alpha_start=0.0,
                            alpha_end=1.0,
                            h_init=0.05,
                            tol=1e-2,
                            h_min=1e-4,
                            h_max=0.2,
                            p0_initial=None,
                            h_min_floor=None):
        """
        Адаптивный метод продолжения по параметру alpha (Hamada–Maruta style).
        Возвращает список словарей с ключами 'alpha', 'trajectory', 'a_max'.
        """
        results = []
        alpha = alpha_start
        h = h_init
        current_guess = p0_initial
        hard_floor = h_min * 0.1 if h_min_floor is None else h_min_floor
        if not (0 < hard_floor <= h_min <= h_max and h_init > 0):
            raise ValueError('Требуется 0 < h_min_floor <= h_min <= h_max и h_init > 0')
        local_floor = h_min
        self.step_floor_history = []


        self.alpha_history = []
        self.h_history = []
        self.err_history = []
        self.step_log = []
        self.continuation_status = {
            'state': 'initial_failed', 'target_alpha': alpha_end,
            'reached_alpha': None,
        }
        self.alpha = alpha
        res = self.solve(current_guess, tol=tol)
        err = np.linalg.norm(self.trajectory_residual(self.trajectory))
        accepted = bool(res.success and np.isfinite(err) and err <= tol)
        self.step_log.append((alpha, h, accepted))

        self.alpha_history.append(alpha)
        self.h_history.append(h)
        self.err_history.append(err)

        if not accepted:
            logger.error("Не удалось получить допустимое решение для начального alpha")
            return results

        results.append({
            'alpha': alpha,
            'trajectory': self.trajectory,
            'a_max': self.a_max,
            'p0': self.p0.copy()
        })
        current_guess = self._x_full()
        self.continuation_status.update(state='partial', reached_alpha=alpha)

        while alpha < alpha_end - 1e-12:
            alpha_next = min(alpha + h, alpha_end)

            self.alpha = alpha_next
            res = self.solve(current_guess, tol=tol)

            if self.trajectory is not None and res.success:
                err = np.linalg.norm(self.trajectory_residual(self.trajectory))

                if err <= tol:
                    results.append({
                        'alpha': alpha_next,
                        'trajectory': self.trajectory,
                        'a_max': self.a_max,
                        'p0': self.p0.copy()
                    })
                    alpha = alpha_next
                    self.continuation_status['reached_alpha'] = alpha
                    current_guess = self._x_full()

                    
                    self.alpha_history.append(alpha)
                    self.h_history.append(h)
                    self.err_history.append(err)
                    self.step_log.append((alpha_next, h, True))
                    h = min(h * 1.5, h_max)
                    if local_floor < h_min and h >= h_min:
                        self.step_floor_history.append(dict(alpha=alpha, event='restore',
                                                            floor=h_min))
                        local_floor = h_min

                    logger.info(f"Шаг успешен, новый alpha = {alpha:.4f}, h = {h:.4f}")
                    continue

            # сохраняем неудачный шаг
            self.alpha_history.append(alpha_next)
            self.h_history.append(h)
            self.err_history.append(1e2)
            self.step_log.append((alpha_next, h, False))

            if h <= local_floor * (1 + 1e-12):
                if h <= hard_floor * (1 + 1e-12):
                    logger.warning("Достигнут окончательный минимум шага; продолжение остановлено.")
                    break
                local_floor = max(hard_floor, min(local_floor * 0.1, h * 0.5))
                self.step_floor_history.append(dict(alpha=alpha, event='lower',
                                                    floor=local_floor))
                logger.info(f"Локально уменьшаем нижний предел шага до {local_floor:.6g}")
            h = max(h * 0.5, local_floor)
            logger.info(f"Неудача, уменьшаем шаг до {h:.6f}")

        if abs(alpha - alpha_end) <= 1e-12:
            self.continuation_status['state'] = 'completed'
        return results

    def _x_full(self):
        """Вектор неизвестных для текущего решения (6 или 7 компонент)."""
        return np.hstack((self.p0, self.a_max)) if self.free_a_max else np.array(self.p0, float)

    def compute_thrust_profile(self, trajectory):
        t = trajectory.t
        y = trajectory.y
        thrust = []
        for i in range(len(t)):
            pv = y[9:12, i]
            a = self.optimal_control(pv)
            thrust.append(np.linalg.norm(a))
        return t, np.array(thrust)

    def plot_trajectories_2d(self, results, filename='trajectories.png'):
        plt.figure(figsize=(10, 8))
        if results:
            colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
            for res, color in zip(results, colors):
                traj = res['trajectory']
                if traj is not None and traj.success:
                    r = traj.y[0:3]
                    plt.plot(r[0], r[1], color=color, linewidth=2,
                             alpha=0.7, label=f'α={res["alpha"]:.2f}')
        plt.scatter([self.r0[0]], [self.r0[1]], c='g', s=200, marker='o', label='Начало')
        plt.scatter([self.rf[0]], [self.rf[1]], c='r', s=200, marker='s', label='Цель')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Метод продолжения: траектории при разных α')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        print(f"График сохранён: {filename}")
        plt.close()

    def plot_thrust_profiles(self, results, filename='thrust_profiles.png'):
        plt.figure(figsize=(10, 6))
        for res in results:
            traj = res['trajectory']
            alpha = res['alpha']
            orig_alpha = self.alpha
            orig_a_max = self.a_max
            self.alpha = alpha
            self.a_max = res['a_max']
            try:
                t, thrust = self.compute_thrust_profile(traj)
            finally:
                self.alpha = orig_alpha
                self.a_max = orig_a_max
            plt.plot(t, thrust, label=f'α={alpha:.2f}')
        plt.xlabel('t', fontsize=14, fontweight='bold')
        plt.ylabel('|a(t)|', fontsize=14, fontweight='bold')
        plt.title('Профили тяги при разных α', fontsize=16, fontweight='bold')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        print(f"График сохранён: {filename}")
        plt.close()

    def plot_h_vs_alpha(self, filename='h_vs_alpha_plot.png'):
        """Шаг h от alpha: успешные шаги (линия) и неудачные (красные точки)."""
        log = np.array(self.step_log, dtype=float)
        ok = log[:, 2] > 0
        plt.figure(figsize=(10, 6))
        plt.semilogy(log[ok, 0], log[ok, 1], 'o-', color='b', linewidth=2, label='Успешные шаги (огибающая)')
        if (~ok).any():
            plt.scatter(log[~ok, 0], log[~ok, 1], color='r', s=60, zorder=3, label='Неудачные шаги')
        plt.xlabel('α', fontsize=14, fontweight='bold')
        plt.ylabel('Шаг h (лог масштаб)', fontsize=14, fontweight='bold')
        plt.title('Зависимость шага h от α\n(огибающая + ветки неудачных шагов)', fontsize=16, fontweight='bold')
        plt.grid(True, which='both', linestyle='--', alpha=0.4)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(filename, dpi=200)
        plt.close()
        print(f"График сохранён: {filename}")

    #активый/пассивный
    def plot_passive_earth_mars(self, trajectory, filename="earth_mars_segments.png"):
        output_dir = Path(filename).parent
        r = trajectory.y[0:3]
        pv = trajectory.y[9:12]
        thr = max(1e-4, 1e-3 * self.a_max)

        active_segments, passive_segments = [], []
        cur_x, cur_y, cur_type = [], [], None

        for i in range(r.shape[1]):
            a = self.optimal_control(pv[:, i])
            seg_type = "passive" if np.linalg.norm(a) < thr else "active"
            if cur_type is None:
                cur_type = seg_type
            if seg_type != cur_type:
                (active_segments if cur_type == "active" else passive_segments).append((cur_x, cur_y))
                cur_x, cur_y, cur_type = [], [], seg_type
            cur_x.append(r[0, i])
            cur_y.append(r[1, i])
        if cur_x:
            (active_segments if cur_type == "active" else passive_segments).append((cur_x, cur_y))

        for name, segments in [('active', active_segments), ('passive', passive_segments)]:
            points = (np.vstack([np.column_stack(s) for s in segments])
                      if segments else np.empty((0, 2)))
            np.savetxt(output_dir / f"{name}_segments.txt", points)

        theta = np.linspace(0, 2 * np.pi, 400)
        plt.figure(figsize=(8, 8))
        r_start = np.linalg.norm(self.r0)
        r_target = np.linalg.norm(self.rf)
        plt.plot(r_start * np.cos(theta), r_start * np.sin(theta), '--k', linewidth=2)
        plt.plot(r_target * np.cos(theta), r_target * np.sin(theta), '-.k', linewidth=2)
        for xs, ys in active_segments:
            plt.plot(xs, ys, color='#ff007f', linewidth=3, alpha=0.8)
        for xs, ys in passive_segments:
            plt.plot(xs, ys, color='#00cfd1', linewidth=3, alpha=0.9)
        plt.scatter(r[0, 0], r[1, 0], c='blue', s=120, edgecolors='black', zorder=5)
        plt.scatter(r[0, -1], r[1, -1], c='red', s=120, edgecolors='black', zorder=5)
        plt.legend(handles=[mlines.Line2D([], [], color='#ff007f', linewidth=3, label='Активный участок'),
                            mlines.Line2D([], [], color='#00cfd1', linewidth=3, label='Пассивный участок')],
                   loc='upper left', fontsize=14)
        plt.xlabel("x, а.е.", fontsize=18)
        plt.ylabel("y, а.е.", fontsize=18)
        plt.title("Активные и пассивные участки траектории", fontsize=20)
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.axis('equal')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"График сохранён: {filename}")
        print(f"Активных сегментов: {len(active_segments)}")
        print(f"Пассивных сегментов: {len(passive_segments)}")

    #зшслду
    def save_alpha1(self, filepath="alpha1.pkl"):
        if abs(self.alpha - 1.0) > 1e-12:
            raise ValueError("save_alpha1 требует alpha=1; используйте save_solution для частичного решения")
        self.save_solution(filepath)

    def save_solution(self, filepath):
        data = {
            "mu": self.mu, "func_type": self.func_type, "alpha": self.alpha, "a_max": self.a_max,
            "free_a_max": self.free_a_max,
            "lambda_value": self.lambda_value,
            "p0": self.p0.copy(), "r0": self.r0.copy(), "v0": self.v0.copy(),
            "rf": self.rf.copy(), "vf": self.vf.copy(), "T": self.T,
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)

    @staticmethod
    def load_and_integrate(filepath="alpha1.pkl"):
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        solver = OptimalControlSolver(mu=data["mu"], a_max=data["a_max"],
                                      alpha=data["alpha"], func_type=data["func_type"])
        solver.free_a_max = data.get("free_a_max", True)
        solver.set_boundary_conditions(data["r0"], data["v0"], data["rf"], data["vf"], data["T"],
                                       lambda_value=data.get('lambda_value'))
        solver.p0 = data["p0"]
        traj = solver.integrate_trajectory(solver.p0)
        if not np.all(np.isfinite(solver.trajectory_residual(traj))):
            raise RuntimeError("Не удалось интегрировать сохранённое решение до конечного времени")
        solver.trajectory = traj
        return solver, traj
