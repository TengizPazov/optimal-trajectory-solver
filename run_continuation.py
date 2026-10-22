"""Штатный пример: python run_continuation.py [--output НОВАЯ_ПАПКА]."""
import argparse
from datetime import datetime
import hashlib
import json
import logging
from pathlib import Path
import platform
import subprocess
import time

import matplotlib
import numpy as np
import pandas as pd
import scipy

from approximations import T_a
from solver import OptimalControlSolver

ROOT = Path(__file__).resolve().parent


def source_info():
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(ROOT.glob('*.py'))}
    try:
        revision = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    return {'git_revision': revision, 'source_sha256': hashes}


def run(output_dir):
    """Выполнить пример в новой папке; вернуть 0, 1 (отказ) или 2 (частично)."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    matplotlib.use('Agg', force=True)
    log_handler = logging.FileHandler(output_dir / 'run.log', encoding='utf-8')
    log_handler.setFormatter(logging.Formatter('%(message)s'))
    logger = logging.getLogger('solver')
    previous_level = logger.level
    logger.setLevel(logging.INFO)
    logger.addHandler(log_handler)
    start = time.perf_counter()
    config = dict(alpha_start=0.0, alpha_end=1.0, h_init=0.05,
                  tol=1e-2, h_min=1e-4, h_max=0.2, h_min_floor=1e-5)
    summary = {
        'status': 'failed', 'config': config,
        'corrector': {'profile': 'diff5_tight', 'method': 'trf', 'jac': '2-point',
                      'diff_step': 1e-5, 'x_scale': 1.0, 'xtol': 1e-10,
                      'ftol': 1e-10, 'gtol': 1e-10, 'max_nfev': 200,
                      'applies_to': 'bounded_correction'},
        'model': {'mu': 1.0, 'lambda_value': 0.5, 'kappa': 1.5, 'initial_a_max': 0.01,
                  'func_type': 'mixed', 'free_a_max': True},
        'environment': {'python': platform.python_version(),
                        'numpy': np.__version__, 'scipy': scipy.__version__,
                        'matplotlib': matplotlib.__version__, 'pandas': pd.__version__},
        **source_info(),
    }
    try:
        solver = OptimalControlSolver(mu=1.0, a_max=0.01, alpha=0.0, func_type='mixed')
        kappa = 1.5
        solver.set_boundary_conditions(
            [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-kappa, 0.0, 0.0],
            [0.0, -1 / np.sqrt(kappa), 0.0], 2 * np.pi * T_a(0.5, kappa), lambda_value=0.5)
        summary['boundary_conditions'] = {
            key: getattr(solver, key).tolist() for key in ('r0', 'v0', 'rf', 'vf')}
        summary['model']['T'] = solver.T
        print(f'Расчёт запущен. Результаты: {output_dir}', flush=True)
        results = solver.continuation_method(**config)
        summary.update({key: value for key, value in solver.continuation_status.items()
                        if key != 'state'})
        summary['status'] = solver.continuation_status['state']
        summary['solutions_count'] = len(results)
        summary['attempts'] = len(solver.step_log)
        summary['failed_attempts'] = sum(not row[2] for row in solver.step_log)
        summary['step_floor_history'] = solver.step_floor_history
        summary['initial_guess'] = getattr(solver, 'initial_guess_info', None)
        summary['initial_quadratic'] = getattr(solver, 'initial_quadratic_info', None)
        pd.DataFrame(solver.step_log, columns=['alpha', 'h', 'success']).assign(
            success=lambda df: df.success.astype(int)).to_csv(
                output_dir / 'step_vs_alpha_data.csv', index=False)
        if not results:
            print('Начальное решение не получено; подробности в run.log.', flush=True)
            return 1

        final = results[-1]
        solver.alpha, solver.a_max = final['alpha'], final['a_max']
        solver.p0, solver.trajectory = final['p0'], final['trajectory']
        completed = (summary['status'] == 'completed'
                     and abs(solver.alpha - config['alpha_end']) <= 1e-12)
        if not completed:
            summary['status'] = 'partial'
        stem = 'alpha1' if completed else 'partial_solution'
        solver.save_solution(output_dir / f'{stem}.pkl')
        arrays = {'alpha': np.array([r['alpha'] for r in results]),
                  'a_max': np.array([r['a_max'] for r in results]),
                  'p0': np.array([r['p0'] for r in results])}
        for i, result in enumerate(results):
            arrays[f't_{i}'] = result['trajectory'].t
            arrays[f'y_{i}'] = result['trajectory'].y
        np.savez_compressed(output_dir / 'solutions.npz', **arrays)
        summary['final_a_max'] = solver.a_max
        summary['final_residual_norm'] = float(np.linalg.norm(
            solver.trajectory_residual(solver.trajectory)))
        solver.plot_passive_earth_mars(
            solver.trajectory, output_dir / f'earth_mars_{stem}.png')
        solver.plot_thrust_profiles(results, output_dir / 'thrust_profiles.png')
        solver.plot_h_vs_alpha(output_dir / 'h_vs_alpha_plot.png')
        print(f"Статус: {summary['status']}; решений: {len(results)}; alpha={solver.alpha:.8f}")
        return 0 if completed else 2
    except Exception as exc:
        summary['status'] = 'failed'
        summary['error'] = f'{type(exc).__name__}: {exc}'
        logger.exception('Расчёт завершился с ошибкой')
        print(f"Ошибка: {summary['error']}. Подробности в run.log.", flush=True)
        return 1
    finally:
        summary['elapsed_seconds'] = time.perf_counter() - start
        (output_dir / 'summary.json').write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        logger.removeHandler(log_handler)
        logger.setLevel(previous_level)
        log_handler.close()
        print(f"Время: {summary['elapsed_seconds']:.2f} с", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Новая папка для результатов')
    args = parser.parse_args(argv)
    output_dir = args.output or ROOT / 'outputs' / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    try:
        return run(output_dir)
    except FileExistsError:
        parser.error(f'Папка уже существует: {output_dir}. Укажите новую папку.')


if __name__ == '__main__':
    raise SystemExit(main())
