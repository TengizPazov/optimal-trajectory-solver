import numpy as np
import pickle
import logging
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from approximations import T_a, get_initial_adjoint
import matplotlib.lines as mlines

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

    # ------------------ УПРАВЛЕНИЕ ------------------
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
                a = self.a_max * pv / pv_norm
            else:
                if pv_norm <= self.alpha:
                    a = np.zeros(3)
                else:
                    a = ((pv_norm - self.alpha) / (1 - self.alpha)) * (pv / pv_norm)
                    a_norm = np.linalg.norm(a)
                    if a_norm > self.a_max:
                        a = (self.a_max / a_norm) * a
            return a

    # ------------------ ДИНАМИКА ------------------
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

    # ------------------ ИНТЕГРИРОВАНИЕ ------------------
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

    # ------------------ НЕВЯЗКА ------------------
    def residual_vec(self, x):
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
            return np.ones(6) * 1e3

        rf = sol.y[0:3, -1]
        vf = sol.y[3:6, -1]
        res = np.hstack((self.rf - rf, self.vf - vf))
        return res

    # ------------------ РЕШЕНИЕ ------------------
    def solve(self, p0_guess=None):
        if p0_guess is None:
            pr0, pv0 = get_initial_adjoint(0.5, 1.2)
            a_max_0 = max(np.linalg.norm(pv0), 0.1)
            p0_guess = np.hstack((pr0, pv0, a_max_0))

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
            self.p0 = result.x[0:6]
            self.a_max = result.x[6]
            self.trajectory = self.integrate_trajectory(self.p0)
        else:
            self.trajectory = None

        return result

    # ------------------ МЕТОД ПРОДОЛЖЕНИЯ ------------------
    def continuation_method(self, alpha_start, alpha_end, h_init, tol, h_min, h_max):
        results = []
        alpha = alpha_start
        h = h_init
        current_guess = None

        while alpha < alpha_end + 1e-12:

            print(f"\n=== Пытаемся шаг: α = {alpha:.4f} ===")

            self.alpha = alpha
            res = self.solve(current_guess)

            if res.success:
                # вычисляем невязку
                x_full = np.hstack((self.p0, self.a_max))
                err = np.linalg.norm(self.residual_vec(x_full))

                print(f"Невязка после solve: {err:.6e}")

                if err < tol:
                    print(f"Успех: α = {alpha:.4f}, h = {h:.4f}")

                    # 🔥 ВСЕГДА сохраняем p0
                    results.append({
                        "alpha": alpha,
                        "trajectory": self.trajectory,
                        "a_max": self.a_max,
                        "p0": self.p0.copy()
                    })

                    current_guess = np.hstack((self.p0, self.a_max))
                    alpha += h
                    h = min(h * 1.5, h_max)
                    continue

            # если не успех
            h *= 0.5
            print(f"Неудача: уменьшаем шаг → h = {h:.4f}")

            if h < h_min:
                print("Шаг стал слишком мал — остановка метода продолжения")
                break

        return results


    # ------------------ ПОСТРОЕНИЕ ТОЛЬКО ПАССИВНОГО УЧАСТКА ------------------
    def plot_passive_earth_mars(self, trajectory, filename="earth_mars_segments.png"):

        r = trajectory.y[0:3]
        pv = trajectory.y[9:12]

        thr = max(1e-4, 1e-3 * self.a_max)

        # --- списки сегментов ---
        active_segments = []
        passive_segments = []

        current_seg_x = []
        current_seg_y = []
        current_type = None  # "active" или "passive"

        # --- классификация точек ---
        for i in range(r.shape[1]):
            a = self.optimal_control(pv[:, i])
            is_passive = np.linalg.norm(a) < thr
            seg_type = "passive" if is_passive else "active"

            if current_type is None:
                current_type = seg_type

            # если тип сегмента сменился → закрываем предыдущий
            if seg_type != current_type:
                if current_type == "active":
                    active_segments.append((current_seg_x, current_seg_y))
                else:
                    passive_segments.append((current_seg_x, current_seg_y))

                current_seg_x = []
                current_seg_y = []
                current_type = seg_type

            current_seg_x.append(r[0, i])
            current_seg_y.append(r[1, i])

        # --- добавляем последний сегмент ---
        if current_seg_x:
            if current_type == "active":
                active_segments.append((current_seg_x, current_seg_y))
            else:
                passive_segments.append((current_seg_x, current_seg_y))

        # --- сохранение массивов ---
        if active_segments:
            np.savetxt(
                "active_segments.txt",
                np.vstack([np.column_stack(seg) for seg in active_segments])
            )

        if passive_segments:
            np.savetxt(
                "passive_segments.txt",
                np.vstack([np.column_stack(seg) for seg in passive_segments])
            )

        # --- орбиты ---
        theta = np.linspace(0, 2*np.pi, 400)
        x_earth = np.cos(theta)
        y_earth = np.sin(theta)
        x_mars = 1.52 * np.cos(theta)
        y_mars = 1.52 * np.sin(theta)

        plt.figure(figsize=(8, 8))

        # орбиты
        plt.plot(x_earth, y_earth, '--k', linewidth=2)
        plt.plot(x_mars, y_mars, '-.k', linewidth=2)

        # --- активные сегменты ---
        for xs, ys in active_segments:
            plt.plot(xs, ys, color='#ff007f', linewidth=3, alpha=0.8)

        # --- пассивные сегменты (сверху) ---
        for xs, ys in passive_segments:
            plt.plot(xs, ys, color='#00cfd1', linewidth=3, alpha=0.9)

        # --- старт и финиш ---
        plt.scatter(r[0, 0], r[1, 0], c='blue', s=120, edgecolors='black', zorder=5)
        plt.scatter(r[0, -1], r[1, -1], c='red', s=120, edgecolors='black', zorder=5)

        # --- легенда ---
        active_line = mlines.Line2D([], [], color='#ff007f', linewidth=3, label='Активный участок')
        passive_line = mlines.Line2D([], [], color='#00cfd1', linewidth=3, label='Пассивный участок')

        plt.legend(handles=[active_line, passive_line],
                loc='upper left',
                fontsize=14)

        # --- оформление ---
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



    # ------------------ СОХРАНЕНИЕ В PICKLE ------------------
    def save_alpha1(self, filepath="alpha1.pkl"):
        data = {
            "mu": self.mu,
            "func_type": self.func_type,
            "alpha": self.alpha,
            "a_max": self.a_max,
            "p0": self.p0.copy(),
            "r0": self.r0.copy(),
            "v0": self.v0.copy(),
            "rf": self.rf.copy(),
            "vf": self.vf.copy(),
            "T": self.T
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)

    @staticmethod
    def load_and_integrate(filepath="alpha1.pkl"):
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        solver = OptimalControlSolver(
            mu=data["mu"],
            a_max=data["a_max"],
            alpha=data["alpha"],
            func_type=data["func_type"]
        )
        solver.set_boundary_conditions(
            data["r0"], data["v0"], data["rf"], data["vf"], data["T"]
        )
        solver.p0 = data["p0"]
        traj = solver.integrate_trajectory(solver.p0)
        return solver, traj
