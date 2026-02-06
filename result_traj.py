import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt


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
            method='LSODA',
            rtol=1e-8,
            atol=1e-10
        )
        return sol

    def loss(self, x):
        pr0 = x[0:3]
        pv0 = x[3:6]

        y0 = np.concatenate([self.r0, self.v0, pr0, pv0])
        sol = solve_ivp(
            self.system_equations,
            [0, self.T],
            y0,
            method='LSODA',
            rtol=1e-8,
            atol=1e-10
        )

        if sol.success:
            rf = sol.y[0:3, -1]
            vf = sol.y[3:6, -1]
            r_res = self.rf - rf
            v_res = self.vf - vf
            return np.linalg.norm(np.hstack((r_res, v_res)))
        else:
            return 1e6

    def solve(self, p0_guess=None, method='Nelder-Mead', options=None):
        if p0_guess is None:
            p0_guess = np.array([0.1, 0.1, 0.0, -0.1, 0.05, 0.0])

        if options is None:
            options = {'disp': False, 'maxiter': 40}

        result = minimize(
            self.loss,
            p0_guess,
            method=method,
            options=options
        )

        self.p0 = result.x
        self.trajectory = self.integrate_trajectory(self.p0)

        return result

    def continuation_method(self, alpha_values, p0_initial=None, a_max_fixed=0.3):
        self.a_max = a_max_fixed

        results = []
        current_p0 = p0_initial

        for alpha in alpha_values:
            self.alpha = alpha
            result = self.solve(current_p0)

            current_p0 = self.p0.copy()

            results.append({
                'alpha': alpha,
                'trajectory': self.trajectory
            })

        return results

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


# пример
if __name__ == "__main__":
    solver = OptimalControlSolver(
        mu=1.0,
        a_max=1.0,
        alpha=0.0,
        func_type='mixed'
    )

    r0 = np.array([1.0, 0.0, 0.0])
    v0 = np.array([0.0, 0.8, 0.0])

    rf = np.array([-1.2, 0.0, 0.0])
    vf = np.array([0.0, -0.7, 0.0])

    T = 8.0

    solver.set_boundary_conditions(r0, v0, rf, vf, T)
    alpha_values = np.linspace(0, 1, 10)

    results = solver.continuation_method(alpha_values, a_max_fixed=0.5)

    solver.plot_trajectories_2d(results, 'continuation_trajectories.png')
