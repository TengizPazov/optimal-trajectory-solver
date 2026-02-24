import numpy as np
import logging
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from approximations import T_a, get_initial_adjoint

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class OptimalControlSolver:
    def __init__(self, mu: float, a_max: float, alpha: float, func_type: str):
        self.mu = mu
        self.a_max = a_max
        self.alpha = alpha
        self.func_type = func_type

    def set_boundary_conditions(self, r0, v0, rf, vf, T):
        self.r0 = np.array(r0, dtype=np.float64)
        self.v0 = np.array(v0, dtype=np.float64)
        self.rf = np.array(rf, dtype=np.float64)
        self.vf = np.array(vf, dtype=np.float64)
        self.T = T

    def optimal_control(self, pv):
        pv_norm = np.linalg.norm(pv)
        eps = 1e-8

        if self.func_type == 'quadratic':
            return pv

        elif self.func_type == 'linear':
            if pv_norm < eps:
                return np.zeros(3)
            return self.a_max * pv / pv_norm

        elif self.func_type == 'mixed':
            if pv_norm < eps:
                return np.zeros(3)
            
            if abs(self.alpha) < 1e-12:
                return pv

            if abs(self.alpha - 1.0) < 1e-12:
                smooth_eps = 1e-3
                x = pv_norm - (1 - smooth_eps)
                smooth = 0.5 * (x + np.sqrt(x**2 + smooth_eps**2))
                scale = np.clip(smooth / smooth_eps, 0.0, 1.0)
                a = scale * (pv / pv_norm)
                # Ограничение по a_max
                a_norm = np.linalg.norm(a)
                if a_norm > self.a_max:
                    a = self.a_max * a / a_norm
                return a

            x = pv_norm - self.alpha
            smooth = 0.5 * (x + np.sqrt(x**2 + 1e-6))
            scale = smooth / (1 - self.alpha)
            scale = np.clip(scale, 0.0, 1.0)

            a = scale * (pv / pv_norm)
            a_norm = np.linalg.norm(a)
            if a_norm > self.a_max:
                a = self.a_max * a / a_norm

            return a

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
        x = [pr0_x, pr0_y, pr0_z, pv0_x, pv0_y, pv0_z, a_max]
        """
        pr0 = x[0:3]
        pv0 = x[3:6]
        a_max = x[6]
        self.a_max = a_max

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

        # Границы: pr0, pv0 не ограничены; a_max ∈ [1e-4, 1.0]
        bounds = ([-np.inf] * 6 + [1e-4], [np.inf] * 6 + [1.0])

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
            a_max_opt = result.x[6]
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

        # Первый шаг (alpha = alpha_start)
        self.alpha = alpha
        res = self.solve(current_guess)
        if self.trajectory is None:
            logger.error("Не удалось получить решение для начального alpha")
            return results

        # Вычисляем невязку
        x_full = np.hstack((self.p0, self.a_max))
        err = np.linalg.norm(self.residual_vec(x_full))
        if err > tol:
            logger.warning(f"Начальное решение имеет большую невязку {err:.6e} > {tol}")
            # Можно попробовать уменьшить допуск или выйти
            # Но продолжим, возможно, дальше подправится

        results.append({
            'alpha': alpha,
            'trajectory': self.trajectory,
            'a_max': self.a_max
        })
        current_guess = np.hstack((self.p0, self.a_max))

        while alpha < alpha_end - 1e-12:
            # Предлагаем новый alpha
            alpha_next = min(alpha + h, alpha_end)

            # Пробуем решить для alpha_next
            self.alpha = alpha_next
            res = self.solve(current_guess)

            if self.trajectory is not None and res.success:
                # Проверяем невязку
                x_full = np.hstack((self.p0, self.a_max))
                err = np.linalg.norm(self.residual_vec(x_full))
                if err <= tol:
                    #сохраняем и увеличиваем шаг
                    results.append({
                        'alpha': alpha_next,
                        'trajectory': self.trajectory,
                        'a_max': self.a_max
                    })
                    alpha = alpha_next
                    current_guess = np.hstack((self.p0, self.a_max))
                    h = min(h * 1.5, h_max)
                    logger.info(f"Шаг успешен, новый alpha = {alpha:.4f}, h = {h:.4f}")
                    continue

            # Если не получилось – уменьшаем шаг и пробуем снова для того же alpha
            h = max(h * 0.5, h_min)
            logger.info(f"Неудача, уменьшаем шаг до {h:.6f}")

            if h < h_min - 1e-12:
                logger.warning("Шаг стал слишком мал, метод продолжения остановлен.")
                break

        return results

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
        plt.xlabel('t')
        plt.ylabel('|a(t)|')
        plt.title('Профили тяги при разных α')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        print(f"График сохранён: {filename}")
        plt.close()


#пример
if __name__ == "__main__":
    solver = OptimalControlSolver(
        mu=1.0,
        a_max=1.0,
        alpha=0.0,
        func_type='mixed'
    )

    r0 = np.array([1.0, 0.0, 0.0])
    v0 = np.array([0.0, 1.0, 0.0])

    rf = np.array([-1.2, 0.0, 0.0])
    vf = np.array([0.0, -1/np.sqrt(1.2), 0.0])

    T = 2 * np.pi * T_a(0.5, 1.2)
    solver.set_boundary_conditions(r0, v0, rf, vf, T)

    results = solver.continuation_method(
        alpha_start=0.0,
        alpha_end=1.0,
        h_init=0.05,
        tol=1e-2,
        h_min=1e-4,
        h_max=0.2,
        p0_initial=None
    )

    print(f"\nПолучено решений: {len(results)}")
    for res in results:
        print(f"α = {res['alpha']:.3f}, a_max = {res['a_max']:.6f}")

    if results:
        step = max(1, len(results) // 3)
        plot_results = results[::step]
        solver.plot_trajectories_2d(plot_results, 'continuation_trajectories.png')
        solver.plot_thrust_profiles(plot_results, 'thrust_profiles.png')

        traj0 = results[0]['trajectory']
        pv = traj0.y[9:12]
        pv_norm = np.linalg.norm(pv, axis=0)
        print(f"Минимальное |p_v| при alpha=0: {np.min(pv_norm):.6f}")
        print(f"Максимальное |p_v| при alpha=0: {np.max(pv_norm):.6f}")