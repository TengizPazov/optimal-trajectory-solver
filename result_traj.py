import numpy as np
import math
import logging
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, least_squares
import matplotlib.pyplot as plt
from approximations import T_a, get_initial_adjoint

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class OptimalControlSolver:
    def __init__(self, mu: float, a_max: float, alpha: float, func_type):
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

            #сглаживание
            if abs(self.alpha - 1.0) < 1e-12:
                smooth_eps = 1e-3
                x = pv_norm - (1 - smooth_eps)
                smooth = 0.5 * (x + np.sqrt(x**2 + smooth_eps**2))
                scale = np.clip(smooth / smooth_eps, 0.0, 1.0)
                a = scale * (pv / pv_norm)

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

    def loss(self, x):
        pr0 = x[0:3]
        pv0 = x[3:6]
        a_max = np.clip(x[6], 1e-4, 1.0)

        self.a_max = a_max

        y0 = np.concatenate([self.r0, self.v0, pr0, pv0])

        sol = solve_ivp(
            self.system_equations,
            [0, self.T],
            y0,
            method='LSODA',
            rtol=1e-8,
            atol=1e-10,
        )

        if sol.success:
            rf = sol.y[0:3, -1]
            vf = sol.y[3:6, -1]
            r_res = self.rf - rf
            v_res = self.vf - vf
            residual = np.linalg.norm(np.hstack((r_res, v_res)))

            logger.info(f"Невязка: {residual:.6f}, pr_0: {pr0}, pv_0: {pv0}, a_max: {a_max:.6f}")
            return residual
        else:
            return 1e6

    def residual_vec(self, x):
        pr0 = x[0:3]
        pv0 = x[3:6]
        a_max = np.clip(x[6], 1e-4, 1.0)
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
        logger.info(f"Невязка: {res_norm:.6e}, pr0={pr0}, pv0={pv0}, a_max={a_max:.6f}")

        return res

    def solve(self, p0_guess=None, method='SLSQP', options=None):
        if abs(self.alpha) < 1e-12:
            self.a_max = 0.01
            logger.info("α = 0 a_max фиксирован = 0.01")

            if p0_guess is None:
                pr0, pv0 = get_initial_adjoint(0.5, 1.2)
                p0_guess = np.hstack((pr0, pv0))

            def loss_alpha0(x):
                pr0 = x[0:3]
                pv0 = x[3:6]

                y0 = np.concatenate([self.r0, self.v0, pr0, pv0])
                sol = solve_ivp(
                    self.system_equations,
                    [0, self.T],
                    y0,
                    method='LSODA',
                    rtol=1e-9,
                    atol=1e-12,
                )

                rf = sol.y[0:3, -1]
                vf = sol.y[3:6, -1]
                res = np.hstack((rf - self.rf, vf - self.vf))
                res_norm = np.linalg.norm(res)

                logger.info(f"[α=0] Невязка: {res_norm:.6e}, pr0={pr0}, pv0={pv0}")
                return res_norm

            result = minimize(
                loss_alpha0,
                p0_guess,
                method=method,
                options={'disp': True, 'maxiter': 200}
            )

            pr0 = result.x[0:3]
            pv0 = result.x[3:6]
            self.p0 = np.hstack((pr0, pv0))
            self.trajectory = self.integrate_trajectory(self.p0)

            logger.info(f"[α=0] Итоговое pr0={pr0}, pv0={pv0}, a_max=0.01\n")
            return result

        if p0_guess is None:
            pr_0, pv_0 = get_initial_adjoint(0.5, 1.2)
            a_max_0 = np.linalg.norm(pv_0)
            p0_guess = np.hstack((pr_0, pv_0, a_max_0))

        logger.info(f"\n=== ШАГ α = {self.alpha:.3f} оптимизируем pr0, pv0, a_max ===")

        result = least_squares(
            self.residual_vec,
            p0_guess,
            method='trf',
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
            max_nfev=200,
        )

        pr0 = result.x[0:3]
        pv0 = result.x[3:6]
        a_max = np.clip(result.x[6], 1e-4, 1.0)

        self.a_max = a_max
        self.p0 = np.hstack((pr0, pv0))
        self.trajectory = self.integrate_trajectory(self.p0)

        logger.info(f"[α={self.alpha:.3f}] Итоговое pr0={pr0}, pv0={pv0}, a_max={a_max:.6f}\n")

        return result

    def continuation_method(self, alpha_values, p0_initial=None, a_max_fixed=None):
        results = []
        current_p0 = p0_initial

        targets = list(alpha_values)
        if not targets:
            return results

        self.alpha = targets[0]
        result0 = self.solve(current_p0)
        x_full = np.hstack((self.p0, self.a_max))
        err0 = np.linalg.norm(self.residual_vec(x_full))

        current_p0 = np.hstack((self.p0, self.a_max))
        results.append({
            'alpha': self.alpha,
            'trajectory': self.trajectory,
            'a_max': self.a_max
        })

        alpha_prev = targets[0]

        for alpha_target in targets[1:]:
            alpha_curr = alpha_prev
            max_refine = 20
            refine_count = 0
            while True:
                step = alpha_target - alpha_curr
                if abs(step) < 1e-6 or refine_count >= max_refine:
                    self.alpha = alpha_target
                    result = self.solve(current_p0)
                    x_full = np.hstack((self.p0, self.a_max))
                    err = np.linalg.norm(self.residual_vec(x_full))
                    current_p0 = np.hstack((self.p0, self.a_max))
                    results.append({
                        'alpha': self.alpha,
                        'trajectory': self.trajectory,
                        'a_max': self.a_max
                    })
                    alpha_prev = alpha_target
                    break

                self.alpha = alpha_curr + step
                result = self.solve(current_p0)
                x_full = np.hstack((self.p0, self.a_max))
                err = np.linalg.norm(self.residual_vec(x_full))

                if err > 1e-2 and abs(step) > 1e-3:
                    alpha_curr = alpha_curr + 0.5 * step
                    refine_count += 1
                    continue
                else:
                    current_p0 = np.hstack((self.p0, self.a_max))
                    alpha_curr = self.alpha
                    continue

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
        plt.title('Метод продолжения: множество траекторий при разных α')
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

            self.alpha = alpha
            t, thrust = self.compute_thrust_profile(traj)
            plt.plot(t, thrust, label=f'α={alpha:.2f}')

        plt.xlabel('t')
        plt.ylabel('|a(t)|')
        plt.title('Профиль тяги при разных α')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        print(f"График сохранён: {filename}")
        plt.close()


# пример
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

    T = 2*np.pi*T_a(0.5, 1.2)

    solver.set_boundary_conditions(r0, v0, rf, vf, T)

    alpha_values = np.linspace(0, 1, 100)

    results = solver.continuation_method(alpha_values)
    results_sparse = [results[0], results[-1]]
    solver.plot_trajectories_2d(results_sparse, 'continuation_trajectories.png')
    solver.plot_thrust_profiles(results_sparse, 'thrust_profiles.png')

    traj0 = results[0]['trajectory']
    pv = traj0.y[9:12]
    pv_norm = np.linalg.norm(pv, axis=0)

    print("Минимальное значение |p_v| при alpha=0:", np.min(pv_norm))
    print("Максимальное значение |p_v| при alpha=0:", np.max(pv_norm))
