from solver import OptimalControlSolver
import numpy as np
from approximations import T_a

solver = OptimalControlSolver(mu=1.0, a_max=0.01, alpha=0.0, func_type='mixed')

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
    h_max=0.2
)

# берём alpha=1
res1 = min(results, key=lambda r: abs(r["alpha"] - 1.0))

solver.alpha = res1["alpha"]
solver.a_max = res1["a_max"]
solver.p0 = res1["p0"]
solver.trajectory = res1["trajectory"]

solver.save_alpha1("alpha1.pkl")
solver.plot_passive_earth_mars(solver.trajectory, "earth_mars_passive_alpha1.png")

print("Готово: сохранены pickle и массивы актив/пассив")
