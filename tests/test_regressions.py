import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares as scipy_least_squares

import run_continuation
from solver import OptimalControlSolver


def make_solver():
    solver = OptimalControlSolver(1.0, 0.2, 0.2, 'mixed')
    solver.set_boundary_conditions([2, 0, 0], [0, 1, 0], [-3, 0, 0], [0, -1, 0], 1.0)
    solver.p0 = np.zeros(6)
    return solver


def trajectory(solver, error=0.0):
    y = np.zeros((12, 2))
    y[:3, 0], y[3:6, 0] = solver.r0, solver.v0
    y[:3, -1], y[3:6, -1] = solver.rf, solver.vf
    y[0, -1] += error
    return SimpleNamespace(success=True, t=np.array([0.0, solver.T]), y=y)


class SolverTests(unittest.TestCase):
    def tearDown(self):
        plt.close('all')

    def test_solve_rejects_invalid_reintegration(self):
        for failure in ('failed', 'early', 'nan', 'infinite_time', 'residual'):
            with self.subTest(failure=failure):
                s = make_solver()
                traj = trajectory(s)
                if failure == 'failed':
                    traj.success = False
                elif failure == 'early':
                    traj.t[-1] = 0.5
                elif failure == 'nan':
                    traj.y[0, -1] = np.nan
                elif failure == 'infinite_time':
                    traj.t[-1] = np.inf
                else:
                    traj.y[0, -1] += 1.0
                opt = SimpleNamespace(success=True, x=np.r_[np.zeros(6), 0.2])
                with patch('solver.least_squares', return_value=opt), patch.object(
                        s, 'integrate_trajectory', return_value=traj):
                    result = s.solve(np.r_[np.zeros(6), 0.2], tol=0.01)
                self.assertFalse(result.success)
                self.assertIsNone(s.trajectory)

    def test_solve_accepts_valid_trajectory(self):
        s = make_solver()
        opt = SimpleNamespace(success=True, x=np.r_[np.zeros(6), 0.2])
        with patch('solver.least_squares', return_value=opt) as corrector, patch.object(
                s, 'integrate_trajectory', return_value=trajectory(s, 1e-6)):
            self.assertTrue(s.solve(np.r_[np.zeros(6), 0.2]).success)
        self.assertEqual(corrector.call_args.kwargs['diff_step'], 1e-5)
        self.assertEqual(corrector.call_args.kwargs['jac'], '2-point')
        self.assertEqual(corrector.call_args.kwargs['x_scale'], 1.0)

    def test_initial_optimizer_failure_is_logged(self):
        s = make_solver()
        with patch.object(s, 'solve', return_value=SimpleNamespace(success=False)):
            self.assertEqual(s.continuation_method(), [])
        self.assertEqual(s.continuation_status['state'], 'initial_failed')
        self.assertEqual(s.step_log, [(0.0, 0.05, False)])

    def test_bad_initial_residual_is_not_accepted(self):
        s = make_solver()
        s.trajectory = trajectory(s, 1.0)
        with patch.object(s, 'solve', return_value=SimpleNamespace(success=True)):
            self.assertEqual(s.continuation_method(), [])
        self.assertFalse(s.step_log[0][2])
        self.assertIsNone(s.continuation_status['reached_alpha'])

    def test_partial_continuation_preserves_last_accepted_alpha(self):
        s = make_solver()
        def solve(*args, **kwargs):
            ok = s.alpha == 0.0
            s.trajectory = trajectory(s) if ok else None
            return SimpleNamespace(success=ok)
        with patch.object(s, 'solve', side_effect=solve):
            results = s.continuation_method(h_init=0.05, h_min=0.05, h_min_floor=0.05)
        self.assertEqual(len(results), 1)
        self.assertEqual(s.continuation_status, {
            'state': 'partial', 'target_alpha': 1.0, 'reached_alpha': 0.0})
        self.assertEqual([r[2] for r in s.step_log], [True, False])

    def test_completed_continuation(self):
        s = make_solver()
        def solve(*args, **kwargs):
            s.trajectory = trajectory(s)
            return SimpleNamespace(success=True)
        with patch.object(s, 'solve', side_effect=solve):
            results = s.continuation_method(alpha_end=0.1)
        self.assertEqual(results[-1]['alpha'], 0.1)
        self.assertEqual(s.continuation_status['state'], 'completed')

    def test_initial_guess_uses_explicit_turns_and_actual_radius_ratio(self):
        s = make_solver()
        s.set_boundary_conditions([1, 0, 0], [0, 1, 0], [-.72, 0, 0], [0, -1, 0], 4,
                                  lambda_value=1.3)
        with patch('solver.get_initial_adjoint', return_value=(np.zeros(3), np.ones(3))) as guess:
            seed = s.initial_guess()
        guess.assert_called_once_with(1.3, .72)
        self.assertEqual(seed[-1], 1.0)
        self.assertGreater(s.initial_guess_info['a_max_raw'], 1.)

    def test_automatic_guess_requires_explicit_turn_count(self):
        s = make_solver()
        with self.assertRaisesRegex(ValueError, 'lambda_value'):
            s.solve()

    def test_explicit_guess_takes_precedence(self):
        s = make_solver()
        opt = SimpleNamespace(success=True, x=np.r_[np.zeros(6), .2])
        with patch('solver.get_initial_adjoint') as guess, patch(
                'solver.least_squares', return_value=opt), patch.object(
                s, 'integrate_trajectory', return_value=trajectory(s)):
            s.solve(np.r_[np.zeros(6), .2])
        guess.assert_not_called()

    def test_local_floor_reduction_and_recovery(self):
        s = make_solver()
        failed_once = False
        def solve(*args, **kwargs):
            nonlocal failed_once
            ok = s.alpha == 0. or failed_once
            if not ok:
                failed_once = True
            s.trajectory = trajectory(s) if ok else None
            return SimpleNamespace(success=ok)
        with patch.object(s, 'solve', side_effect=solve):
            results = s.continuation_method(alpha_end=.001, h_init=1e-4,
                                            h_min=1e-4, h_min_floor=1e-5)
        self.assertEqual(results[-1]['alpha'], .001)
        self.assertEqual([e['event'] for e in s.step_floor_history], ['lower', 'restore'])
        self.assertAlmostEqual(s.step_log[2][1], 5e-5)

    def test_local_floor_has_a_finite_stop(self):
        s = make_solver()
        def solve(*args, **kwargs):
            ok = s.alpha == 0.
            s.trajectory = trajectory(s) if ok else None
            return SimpleNamespace(success=ok)
        with patch.object(s, 'solve', side_effect=solve):
            s.continuation_method(h_init=.05, h_min=.05, h_min_floor=.005)
        self.assertEqual(s.continuation_status['state'], 'partial')
        self.assertAlmostEqual(s.step_log[-1][1], .005)
        self.assertLess(len(s.step_log), 10)

    def test_invalid_floor_is_rejected_before_solving(self):
        s = make_solver()
        with patch.object(s, 'solve') as solve:
            with self.assertRaises(ValueError):
                s.continuation_method(h_min=1e-4, h_min_floor=1e-3)
        solve.assert_not_called()

    def test_turn_count_survives_save_load(self):
        s = make_solver()
        s.lambda_value = 1.3
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / 'solution.pkl'
            s.save_solution(target)
            with patch.object(OptimalControlSolver, 'integrate_trajectory', return_value=trajectory(s)):
                loaded, _ = OptimalControlSolver.load_and_integrate(target)
        self.assertEqual(loaded.lambda_value, 1.3)

    def test_initial_quadratic_cap_matches_analytic_double_integrator(self):
        # μ=0: a(t)=0.06-0.012t для перелёта 1 -> 2 за T=10 с нулевыми скоростями.
        s = OptimalControlSolver(0., .001, 0., 'mixed')
        s.set_boundary_conditions([1, 0, 0], [0, 0, 0], [2, 0, 0], [0, 0, 0], 10.)
        with patch('solver.least_squares', wraps=scipy_least_squares) as corrector:
            result = s.solve(np.array([.011, 0, 0, .055, 0, 0, .001]))
        self.assertNotIn('diff_step', corrector.call_args.kwargs)
        self.assertTrue(result.success)
        np.testing.assert_allclose(s.p0, [.012, 0, 0, .06, 0, 0], atol=1e-8)
        self.assertAlmostEqual(s.a_max, .06, places=8)
        self.assertLess(np.linalg.norm(s.trajectory_residual(s.trajectory)), 1e-8)
        self.assertLess(s.initial_quadratic_info['maximum_state_deviation'], 1e-7)
        self.assertEqual(result.x.shape, (7,))
        self.assertEqual(result.jac.shape, (6, 7))

    def test_peak_search_includes_interior_of_integration_intervals(self):
        y = np.zeros((12, 2))
        y[9] = 1.
        def dense(t):
            value = np.zeros(12)
            value[9] = 1. + 4. * t * (1. - t)
            return value
        traj = SimpleNamespace(t=np.array([0., 1.]), y=y, sol=dense)
        peak, time = OptimalControlSolver.peak_quadratic_thrust(traj)
        self.assertAlmostEqual(peak, 2., places=10)
        self.assertAlmostEqual(time, .5, places=6)

    def test_initial_peak_above_upper_bound_is_rejected_without_clipping(self):
        # The same rest-to-rest transfer in T=1 needs peak thrust 6 > 1.
        s = OptimalControlSolver(0., .2, 0., 'mixed')
        s.set_boundary_conditions([1, 0, 0], [0, 0, 0], [2, 0, 0], [0, 0, 0], 1.)
        result = s.solve(np.array([12., 0, 0, 6., 0, 0, .2]))
        self.assertFalse(result.success)
        self.assertGreater(s.initial_quadratic_info['peak_thrust'], 1.)
        self.assertEqual(s.a_max, .2)
        self.assertIsNone(s.trajectory)
        self.assertFalse(s._unbounded_quadratic)
        self.assertTrue(s.free_a_max)

    def test_initialization_restores_mode_on_failure(self):
        s = make_solver()
        s.alpha = 0.
        with patch('solver.least_squares', side_effect=RuntimeError('test')):
            with self.assertRaises(RuntimeError):
                s.solve(np.r_[np.zeros(6), .2])
        self.assertTrue(s.free_a_max)
        self.assertFalse(s._unbounded_quadratic)

    def test_fixed_cap_is_not_replaced_by_initialization(self):
        s = make_solver()
        s.alpha = 0.
        s.free_a_max = False
        opt = SimpleNamespace(success=True, x=np.zeros(6))
        with patch.object(s, 'initialize_quadratic') as initialize, patch(
                'solver.least_squares', return_value=opt), patch.object(
                s, 'integrate_trajectory', return_value=trajectory(s)):
            s.solve(np.zeros(6))
        initialize.assert_not_called()
        self.assertEqual(s.a_max, .2)

    def test_profiles_use_individual_caps_and_restore_state(self):
        s = make_solver()
        s.alpha, s.a_max = 0.4, 0.9
        traj = trajectory(s)
        traj.y[9] = 2.0
        results = [dict(alpha=0., a_max=.2, trajectory=traj),
                   dict(alpha=1., a_max=.7, trajectory=traj)]
        with patch('solver.plt.plot') as plot, patch('solver.plt.savefig'), patch('solver.plt.legend'):
            s.plot_thrust_profiles(results)
            np.testing.assert_allclose(plot.call_args_list[0].args[1], [.2, .2])
            np.testing.assert_allclose(plot.call_args_list[1].args[1], [.7, .7])
        self.assertEqual((s.alpha, s.a_max), (.4, .9))
        with patch.object(s, 'compute_thrust_profile', side_effect=RuntimeError('test')):
            with self.assertRaises(RuntimeError):
                s.plot_thrust_profiles(results)
        self.assertEqual((s.alpha, s.a_max), (.4, .9))

    def test_orbits_follow_boundary_radii_and_exports_stay_beside_plot(self):
        s = make_solver()
        with tempfile.TemporaryDirectory() as tmp, patch('solver.plt.plot') as plot, patch('solver.plt.savefig'):
            s.plot_passive_earth_mars(trajectory(s), Path(tmp) / 'orbit.png')
            for call, radius in zip(plot.call_args_list[:2], (2., 3.)):
                np.testing.assert_allclose(np.hypot(call.args[0], call.args[1]), radius)
            self.assertTrue((Path(tmp) / 'active_segments.txt').exists())
            self.assertTrue((Path(tmp) / 'passive_segments.txt').exists())

    def test_alpha1_writer_refuses_partial_solution(self):
        s = make_solver()
        s.alpha = .6
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / 'alpha1.pkl'
            with self.assertRaises(ValueError):
                s.save_alpha1(target)
            self.assertFalse(target.exists())

    def test_fixed_cap_mode_survives_save_load(self):
        s = make_solver()
        s.free_a_max = False
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / 'solution.pkl'
            s.save_solution(target)
            with patch.object(OptimalControlSolver, 'integrate_trajectory', return_value=trajectory(s)):
                loaded, traj = OptimalControlSolver.load_and_integrate(target)
            self.assertFalse(loaded.free_a_max)
            self.assertIs(loaded.trajectory, traj)


class RunnerTests(unittest.TestCase):
    def test_initial_failure_writes_diagnostics_without_solution(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()), patch.object(
                OptimalControlSolver, 'solve', return_value=SimpleNamespace(success=False)):
            output = Path(tmp) / 'run'
            self.assertEqual(run_continuation.run(output), 1)
            summary = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
            self.assertEqual(summary['status'], 'initial_failed')
            self.assertEqual(summary['solutions_count'], 0)
            self.assertTrue((output / 'step_vs_alpha_data.csv').exists())
            self.assertFalse(list(output.glob('*.pkl')))

    def test_partial_and_completed_outputs_are_distinct(self):
        for state, alpha, exit_code, stem in [
                ('partial', .6, 2, 'partial_solution'), ('completed', 1., 0, 'alpha1')]:
            with self.subTest(state=state), tempfile.TemporaryDirectory() as tmp:
                def continuation(s, **kwargs):
                    s.p0 = np.zeros(6)
                    s.step_log = [(0., .05, True), (alpha, .05, True)]
                    s.continuation_status = dict(state=state, target_alpha=1., reached_alpha=alpha)
                    return [dict(alpha=alpha, a_max=.2, p0=s.p0, trajectory=trajectory(s))]
                output = Path(tmp) / 'run'
                # Windows pipes may use cp1251, which cannot encode Greek alpha.
                with contextlib.redirect_stdout(io.TextIOWrapper(io.BytesIO(), encoding='cp1251')), patch.object(
                        OptimalControlSolver, 'continuation_method', continuation), patch.object(
                        OptimalControlSolver, 'plot_passive_earth_mars'), patch.object(
                        OptimalControlSolver, 'plot_thrust_profiles'), patch.object(
                        OptimalControlSolver, 'plot_h_vs_alpha'):
                    self.assertEqual(run_continuation.run(output), exit_code)
                summary = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
                self.assertEqual(summary['status'], state)
                self.assertTrue((output / f'{stem}.pkl').exists())
                if state == 'partial':
                    self.assertFalse((output / 'alpha1.pkl').exists())
                with np.load(output / 'solutions.npz') as data:
                    self.assertEqual(data['alpha'][-1], alpha)
                    self.assertIn('y_0', data.files)

    def test_existing_directory_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / 'marker.txt'
            marker.write_text('keep', encoding='utf-8')
            with self.assertRaises(FileExistsError):
                run_continuation.run(tmp)
            self.assertEqual(marker.read_text(encoding='utf-8'), 'keep')

    def test_unexpected_failure_writes_summary(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()), patch.object(
                OptimalControlSolver, 'continuation_method', side_effect=RuntimeError('test')):
            output = Path(tmp) / 'run'
            self.assertEqual(run_continuation.run(output), 1)
            summary = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
            self.assertEqual(summary['status'], 'failed')
            self.assertIn('RuntimeError', summary['error'])


if __name__ == '__main__':
    unittest.main()
