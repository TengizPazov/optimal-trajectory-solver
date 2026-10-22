"""Повторно интегрировать сохранённое решение и построить его траекторию."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from solver import OptimalControlSolver


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('solution', type=Path, help='Файл .pkl выбранного запуска')
    parser.add_argument('--output', type=Path, help='По умолчанию рядом с решением')
    args = parser.parse_args(argv)
    solver, traj = OptimalControlSolver.load_and_integrate(args.solution)
    output = args.output or args.solution.parent / f'{args.solution.stem}_loaded.png'
    solver.plot_passive_earth_mars(traj, output)
    print(f'График построен для alpha={solver.alpha:.8f}, a_max={solver.a_max:.8f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
