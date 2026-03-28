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
                a = pv
            elif abs(self.alpha - 1.0) < 1e-12:
                a = self.a_max * pv / pv_norm if pv_norm > 1e-12 else np.zeros(3)
            else:
                if pv_norm <= self.alpha:
                    a = np.zeros(3)
                else:
                    a = ((pv_norm - self.alpha) / (1 - self.alpha)) * (pv / pv_norm)
                    a_norm = np.linalg.norm(a)
                    if a_norm > self.a_max:
                        a = (self.a_max / a_norm) * a

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

        # --- история ---
        self.alpha_history = []
        self.h_history = []
        self.err_history = []

        # Первый шаг (alpha = alpha_start)
        self.alpha = alpha
        res = self.solve(current_guess)
        if self.trajectory is None:
            logger.error("Не удалось получить решение для начального alpha")
            return results

        # Вычисляем невязку
        x_full = np.hstack((self.p0, self.a_max))
        err = np.linalg.norm(self.residual_vec(x_full))

        # сохраняем
        self.alpha_history.append(alpha)
        self.h_history.append(h)
        self.err_history.append(err)

        if err > tol:
            logger.warning(f"Начальное решение имеет большую невязку {err:.6e} > {tol}")

        results.append({
            'alpha': alpha,
            'trajectory': self.trajectory,
            'a_max': self.a_max
        })
        current_guess = np.hstack((self.p0, self.a_max))

        while alpha < alpha_end - 1e-12:
            alpha_next = min(alpha + h, alpha_end)

            self.alpha = alpha_next
            res = self.solve(current_guess)

            if self.trajectory is not None and res.success:
                x_full = np.hstack((self.p0, self.a_max))
                err = np.linalg.norm(self.residual_vec(x_full))

                if err <= tol:
                    results.append({
                        'alpha': alpha_next,
                        'trajectory': self.trajectory,
                        'a_max': self.a_max
                    })
                    alpha = alpha_next
                    current_guess = np.hstack((self.p0, self.a_max))
                    h = min(h * 1.5, h_max)

                    # сохраняем
                    self.alpha_history.append(alpha)
                    self.h_history.append(h)
                    self.err_history.append(err)

                    logger.info(f"Шаг успешен, новый alpha = {alpha:.4f}, h = {h:.4f}")
                    continue

            h = max(h * 0.5, h_min)
            logger.info(f"Неудача, уменьшаем шаг до {h:.6f}")

            if h < h_min - 1e-12:
                logger.warning("Шаг стал слишком мал, метод продолжения остановлен.")
                break

            # сохраняем неудачный шаг
            self.alpha_history.append(alpha_next)
            self.h_history.append(h)
            self.err_history.append(1e2)

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

    def plot_earth_mars_presentation(self, trajectory, optimal_control_func, filename="earth_mars_presentation.png"):
        r = trajectory.y[0:3]
        pv = trajectory.y[9:12]

        active_x, active_y = [], []
        passive_x, passive_y = [], []

        # --- разделение участков ---
        for i in range(r.shape[1]):
            a = optimal_control_func(pv[:, i])
            a_norm = np.linalg.norm(a)

            if a_norm < 1e-4:
                passive_x.append(r[0, i])
                passive_y.append(r[1, i])
            else:
                active_x.append(r[0, i])
                active_y.append(r[1, i])

        # --- орбиты ---
        theta = np.linspace(0, 2*np.pi, 1000)
        r_earth = 1.0
        r_mars = 1.52

        x_earth = r_earth * np.cos(theta)
        y_earth = r_earth * np.sin(theta)

        x_mars = r_mars * np.cos(theta)
        y_mars = r_mars * np.sin(theta)

        # --- оформление ---
        plt.figure(figsize=(8, 8))

        # орбиты
        plt.plot(x_earth, y_earth, linestyle='--', color='black', linewidth=2, label='Орбита Земли')
        plt.plot(x_mars, y_mars, linestyle='-.', color='black', linewidth=2, label='Орбита Марса')

        # траектория
        plt.plot(passive_x, passive_y, color='#00cfd1', linewidth=3, label='Пассивный участок')
        plt.plot(active_x, active_y, color='#ff00aa', linewidth=3, label='Активный участок')

        # старт и финиш
        plt.scatter(r[0, 0], r[1, 0], color='blue', s=120, edgecolors='black', zorder=5)
        plt.scatter(r[0, -1], r[1, -1], color='red', s=120, edgecolors='black', zorder=5)

        # подписи планет
        plt.text(1.05, 0.05, "Земля", fontsize=12)
        plt.text(1.57, 0.05, "Марс", fontsize=12)

        # оси
        plt.xlabel("x, а.е.", fontsize=14)
        plt.ylabel("y, а.е.", fontsize=14)

        # сетка
        plt.grid(True, linestyle='--', alpha=0.5)

        # легенда
        plt.legend(fontsize=12, loc='upper left')

        # пропорции
        plt.axis('equal')

        # заголовок
        plt.title("Пример перелёта Земля–Марс", fontsize=16)

        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        print(f"График сохранён: {filename}")
        plt.close()

    # --- график шага h(α) ---
    def plot_step_vs_alpha(self, filename="step_vs_alpha.png", datafile="step_vs_alpha_data.csv"):
        alpha = np.array(self.alpha_history)
        h = np.array(self.h_history)
        err = np.array(self.err_history)

        # успешные шаги (огибающая)
        mask_success = err < 1e2
        alpha_success = alpha[mask_success]
        h_success = h[mask_success]

        # неудачные шаги (ветки)
        mask_fail = err >= 1e2
        alpha_fail = alpha[mask_fail]
        h_fail = h[mask_fail]

        # --- Сохранение данных в CSV ---
        with open(datafile, "w") as f:
            f.write("alpha,h,success\n")
            for a, hh, ok in zip(alpha, h, mask_success):
                f.write(f"{a},{hh},{int(ok)}\n")

        print(f"Данные графика сохранены в файл: {datafile}")

        # --- Построение графика ---
        plt.figure(figsize=(8, 5))

        # огибающая — линия
        plt.plot(alpha_success, h_success, '-o', color='blue', label='Успешные шаги (огибающая)')

        # неудачные шаги — точки
        plt.scatter(alpha_fail, h_fail, color='red', s=40, label='Неудачные шаги')

        plt.yscale('log')
        plt.xlabel("α")
        plt.ylabel("Шаг h (лог масштаб)")
        plt.grid(True)
        plt.title("Зависимость шага h от α (огибающая + неудачные ветки)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename, dpi=200)
        plt.close()
        print(f"График сохранён: {filename}")



#пример
if __name__ == "__main__":
    solver = OptimalControlSolver(
        mu=1.0,
        a_max=1.0,
        alpha=0.0,
        func_type='mixed'
    )
    kappa = 1.5
    r0 = np.array([1.0, 0.0, 0.0])
    v0 = np.array([0.0, 1.0, 0.0])

    rf = np.array([-kappa, 0.0, 0.0])
    vf = np.array([0.0, -1/np.sqrt(kappa), 0.0])

    T = 2 * np.pi * T_a(0.5, kappa)
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

    # --- построение новых графиков ---
    solver.plot_step_vs_alpha()
    solver.plot_error_vs_alpha()

    print(f"\nПолучено решений: {len(results)}")
    for res in results:
        print(f"α = {res['alpha']:.3f}, a_max = {res['a_max']:.6f}")

    if results:
        plot_results = [results[0], results[-2]]
        solver.plot_trajectories_2d(plot_results, 'continuation_trajectories.png')
        solver.plot_thrust_profiles(plot_results, 'thrust_profiles.png')

        traj0 = results[0]['trajectory']
        pv = traj0.y[9:12]
        pv_norm = np.linalg.norm(pv, axis=0)
        print(f"Минимальное |p_v| при alpha=0: {np.min(pv_norm):.6f}")
        print(f"Максимальное |p_v| при alpha=0: {np.max(pv_norm):.6f}")

        solver.plot_earth_mars_presentation(
            trajectory=traj0,
            optimal_control_func=solver.optimal_control,
            filename="earth_mars_presentation.png"
        )