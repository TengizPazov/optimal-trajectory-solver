import numpy as np
import pickle
import logging
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from approximations import T_a, get_initial_adjoint

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class OptimalControlSolver:
    def __init__(self, mu: float, a_max: float, alpha: float, func_type: str):
        self.mu = mu
        self.a_max = a_max
        self.alpha = alpha
        self.func_type = func_type
        self.free_a_max = True

    def set_boundary_conditions(self, r0, v0, rf, vf, T):
        self.r0 = np.array(r0, dtype=np.float64)
        self.v0 = np.array(v0, dtype=np.float64)
        self.rf = np.array(rf, dtype=np.float64)
        self.vf = np.array(vf, dtype=np.float64)
        self.T = T

    def optimal_control(self, pv):

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

    def integrate_trajectory(self, p0, t_span=None):
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

        if not sol.success:
            logger.info("solve_ivp не сошёлся")
            return np.ones(6) * 1e3

        rf = sol.y[0:3, -1]
        vf = sol.y[3:6, -1]
        r_res = self.rf - rf
        v_res = self.vf - vf
        res = np.hstack((r_res, v_res))

        res_norm = np.linalg.norm(res)
        logger.info(f"α={self.alpha:.3f} | невязка={res_norm:.6e} | a_max={a_max:.6f}")
        return res

    def solve(self, p0_guess=None):
        if p0_guess is None:
            pr0, pv0 = get_initial_adjoint(0.5, 1.2)
            a_max_0 = max(np.linalg.norm(pv0), 0.1)
            p0_guess = np.hstack((pr0, pv0, a_max_0))

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
            logger.info(f"Успешная оптимизация при α={self.alpha:.3f}, a_max={a_max_opt:.6f}")
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
                            p0_initial=None):
        """
        Адаптивный метод продолжения по параметру alpha (Hamada–Maruta style).
        Возвращает список словарей с ключами 'alpha', 'trajectory', 'a_max'.
        """
        results = []
        alpha = alpha_start
        h = h_init
        current_guess = p0_initial


        self.alpha_history = []
        self.h_history = []
        self.err_history = []
        self.alpha = alpha
        res = self.solve(current_guess)
        if self.trajectory is None:
            logger.error("Не удалось получить решение для начального alpha")
            return results

        err = np.linalg.norm(self.residual_vec(self._x_full()))
        self.step_log = [(alpha, h, True)]

        self.alpha_history.append(alpha)
        self.h_history.append(h)
        self.err_history.append(err)

        if err > tol:
            logger.warning(f"Начальное решение имеет большую невязку {err:.6e} > {tol}")

        results.append({
            'alpha': alpha,
            'trajectory': self.trajectory,
            'a_max': self.a_max,
            'p0': self.p0.copy()
        })
        current_guess = self._x_full()

        while alpha < alpha_end - 1e-12:
            alpha_next = min(alpha + h, alpha_end)

            self.alpha = alpha_next
            res = self.solve(current_guess)

            if self.trajectory is not None and res.success:
                err = np.linalg.norm(self.residual_vec(self._x_full()))

                if err <= tol:
                    results.append({
                        'alpha': alpha_next,
                        'trajectory': self.trajectory,
                        'a_max': self.a_max,
                        'p0': self.p0.copy()
                    })
                    alpha = alpha_next
                    current_guess = self._x_full()

                    
                    self.alpha_history.append(alpha)
                    self.h_history.append(h)
                    self.err_history.append(err)
                    self.step_log.append((alpha_next, h, True))
                    h = min(h * 1.5, h_max)

                    logger.info(f"Шаг успешен, новый alpha = {alpha:.4f}, h = {h:.4f}")
                    continue

            # сохраняем неудачный шаг
            self.alpha_history.append(alpha_next)
            self.h_history.append(h)
            self.err_history.append(1e2)
            self.step_log.append((alpha_next, h, False))

            if h <= h_min * (1 + 1e-12):
                logger.warning("Шаг уже минимальный и не сходится, метод продолжения остановлен.")
                break
            h = max(h * 0.5, h_min)
            logger.info(f"Неудача, уменьшаем шаг до {h:.6f}")

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
            self.alpha = alpha
            t, thrust = self.compute_thrust_profile(traj)
            self.alpha = orig_alpha
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

        if active_segments:
            np.savetxt("active_segments.txt", np.vstack([np.column_stack(s) for s in active_segments]))
        if passive_segments:
            np.savetxt("passive_segments.txt", np.vstack([np.column_stack(s) for s in passive_segments]))

        theta = np.linspace(0, 2 * np.pi, 400)
        plt.figure(figsize=(8, 8))
        plt.plot(np.cos(theta), np.sin(theta), '--k', linewidth=2)
        plt.plot(1.52 * np.cos(theta), 1.52 * np.sin(theta), '-.k', linewidth=2)
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
        data = {
            "mu": self.mu, "func_type": self.func_type, "alpha": self.alpha, "a_max": self.a_max,
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
        solver.set_boundary_conditions(data["r0"], data["v0"], data["rf"], data["vf"], data["T"])
        solver.p0 = data["p0"]
        traj = solver.integrate_trajectory(solver.p0)
        return solver, traj