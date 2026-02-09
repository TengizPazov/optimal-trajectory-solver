import numpy as np
import math
import logging
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from approximations import T_a, get_initial_adjoint

# Настройка логгирования
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class OptimalControlSolver:
    '''
    Сlass for solving the optimal control problem using the sighting method
    '''
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

        if self.func_type == 'quadratic':
            a = pv

        elif self.func_type == 'linear':
            a = self.a_max * pv / pv_norm if pv_norm > 1e-15 else np.zeros(3)

        elif self.func_type == 'mixed':

            if self.alpha > 0.95: 
                return self.a_max * pv / pv_norm if pv_norm > 1e-12 else np.zeros(3)

            if abs(self.alpha) < 1e-10:
                a = pv
            elif abs(self.alpha - 1.0) < 1e-10:
                a = self.a_max * pv / pv_norm if pv_norm > 1e-10 else np.zeros(3)
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
            method='RK45',
            rtol=1e-9,
            atol=1e-12,
            max_step = 1e-3
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
            method='RK45',
            rtol=1e-8,
            atol=1e-10,
            max_step=1e-3
        )

        if sol.success:
            rf = sol.y[0:3, -1]
            vf = sol.y[3:6, -1]
            r_res = self.rf - rf
            v_res = self.vf - vf
            residual = np.linalg.norm(np.hstack((r_res, v_res)))
            
            # Логгирование значений на каждом шаге
            logger.info(f"Невязка: {residual:.6f}, pr_0: [{pr0[0]:.6f}, {pr0[1]:.6f}, {pr0[2]:.6f}], "
                        f"pv_0: [{pv0[0]:.6f}, {pv0[1]:.6f}, {pv0[2]:.6f}], a_max: {a_max:.6f}")
            
            return residual
        else:
            return 1e6


    def solve(self, p0_guess=None, method='Nelder-Mead', options=None):
        if p0_guess is None:
            pr_0, pv_0 = get_initial_adjoint(0.5, 1.2)
            a_max_0 = np.linalg.norm(pv_0)
            p0_guess = np.hstack((pr_0, pv_0, a_max_0))  # добавили a_max

        if options is None:
            options = {'disp': False, 'maxiter': 80}

        result = minimize(
            self.loss,
            p0_guess,
            method=method,
            options=options
        )

        # читаем оптимальные параметры
        pr0 = result.x[0:3]
        pv0 = result.x[3:6]
        a_max = np.clip(result.x[6], 1e-4, 1.0)

        self.a_max = a_max
        self.p0 = np.hstack((pr0, pv0))

        self.trajectory = self.integrate_trajectory(self.p0)
        return result


    def continuation_method(self, alpha_values, p0_initial=None, a_max_fixed=None):
        results = []
        current_p0 = p0_initial

        for alpha in alpha_values:
            self.alpha = alpha
            print(f"α = {alpha:.3f}")

            result = self.solve(current_p0)

            print(f"Найдено a_max = {self.a_max:.6f}")

            current_p0 = np.hstack((self.p0, self.a_max))

            results.append({
                'alpha': alpha,
                'trajectory': self.trajectory,
                'a_max': self.a_max
            })


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


    T = 2*np.pi*T_a(0.5,1.2)

    solver.set_boundary_conditions(r0, v0, rf, vf, T)
    #alpha_values = np.linspace(0, 0.9, 20)
    alpha_values = np.linspace(0, 1.0, 20)

    results = solver.continuation_method(alpha_values)
    results_sparse = [results[0], results[-1]]
    solver.plot_trajectories_2d(results_sparse, 'continuation_trajectories.png')
    solver.plot_thrust_profiles(results_sparse, 'thrust_profiles.png')
    # Берём решение для alpha = 0
    traj0 = results[0]['trajectory']

    pv = traj0.y[9:12]
    pv_norm = np.linalg.norm(pv, axis=0)

    print("Минимальное значение |p_v| при alpha=0:", np.min(pv_norm))
    print("Максимальное значение |p_v| при alpha=0:", np.max(pv_norm))