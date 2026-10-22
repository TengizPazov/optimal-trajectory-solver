"""Independent integration check of the base quadratic initialization."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from approximations import T_a
from solver import OptimalControlSolver


def main():
    output = Path(sys.argv[1])
    output.mkdir(parents=True, exist_ok=False)
    solver = OptimalControlSolver(1., .01, 0., 'mixed')
    solver.set_boundary_conditions(
        [1., 0., 0.], [0., 1., 0.], [-1.5, 0., 0.],
        [0., -1. / np.sqrt(1.5), 0.], 2. * np.pi * T_a(.5, 1.5), lambda_value=.5)
    result = solver.solve()
    if not result.success:
        raise RuntimeError(result.message)
    y0 = np.r_[solver.r0, solver.v0, solver.p0]
    kwargs = dict(method='DOP853', rtol=1e-12, atol=1e-14, dense_output=True)
    bounded = solve_ivp(solver.system_equations, [0., solver.T], y0, **kwargs)
    try:
        solver._unbounded_quadratic = True
        unbounded = solve_ivp(solver.system_equations, [0., solver.T], y0, **kwargs)
    finally:
        solver._unbounded_quadratic = False
    time = np.linspace(0., solver.T, 10001)
    peak, peak_time = solver.peak_quadratic_thrust(unbounded)
    info = dict(initialization=solver.initial_quadratic_info,
                dop853_bounded_residual=float(np.linalg.norm(solver.trajectory_residual(bounded))),
                dop853_unbounded_residual=float(np.linalg.norm(solver.trajectory_residual(unbounded))),
                dop853_peak=peak, dop853_peak_time=peak_time,
                dop853_maximum_state_deviation=float(np.max(np.linalg.norm(
                    bounded.sol(time)[:6] - unbounded.sol(time)[:6], axis=0))),
                source_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                               for name in ('solver.py', 'approximations.py', 'run_continuation.py')})
    (output / 'verification.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    assert bounded.success and unbounded.success
    assert abs(peak - solver.a_max) < 1e-9
    assert info['dop853_bounded_residual'] < 1e-7
    assert info['dop853_maximum_state_deviation'] < 1e-7
    print(json.dumps(info, indent=2))


if __name__ == '__main__':
    main()
