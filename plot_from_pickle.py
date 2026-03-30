from solver import OptimalControlSolver

solver, traj = OptimalControlSolver.load_and_integrate("alpha1.pkl")
solver.plot_passive_earth_mars(traj, "earth_mars_passive_loaded.png")

print("График построен из pickle")
