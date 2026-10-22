"""Compare saved correction traces and independently verify accepted states."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from approximations import T_a
from solver import OptimalControlSolver


def analyze(folder):
    rows = json.loads((folder / 'attempts.json').read_text(encoding='utf-8'))
    failures = [r for r in rows if not r['accepted']]
    limited = [r for r in failures if r.get('raw', {}).get('status') == 0]
    def final_error(r):
        return float(np.linalg.norm(r['raw']['fun']))
    bins = []
    for low, high in ((0., .2), (.2, .5), (.5, .6), (.6, .7), (.7, .8), (.8, 1.000001)):
        subset = [r for r in rows if low <= r['alpha'] < high]
        bins.append(dict(alpha_range=[low, min(high, 1.)], attempts=len(subset),
            failed=sum(not r['accepted'] for r in subset),
            evaluations=sum(r['evaluation_count'] for r in subset),
            seconds=sum(r['seconds'] for r in subset)))
    verification = []
    s = OptimalControlSolver(1., .01, 0., 'mixed')
    s.set_boundary_conditions([1.,0.,0.], [0.,1.,0.], [-1.5,0.,0.],
        [0.,-1/np.sqrt(1.5),0.], 2*np.pi*T_a(.5,1.5), lambda_value=.5)
    with np.load(folder / 'accepted.npz') as data:
        for i, alpha in enumerate(data['alpha']):
            s.alpha, s.a_max = float(alpha), float(data['a_max'][i])
            sol = solve_ivp(s.system_equations, [0.,s.T],
                np.r_[s.r0,s.v0,data['p0'][i]], method='DOP853',rtol=1e-11,atol=1e-13,
                max_step=s.T/200)
            verification.append(dict(alpha=s.alpha, a_max=s.a_max, success=bool(sol.success),
                residual=float(np.linalg.norm(s.trajectory_residual(sol)))))
    return dict(summary=json.loads((folder/'summary.json').read_text(encoding='utf-8')),
        evaluations=sum(r['evaluation_count'] for r in rows),
        correction_seconds=sum(r['seconds'] for r in rows),
        failed_count=len(failures), max_nfev_failures=len(limited),
        max_nfev_failures_with_final_residual_under_tol=sum(final_error(r)<=.01 for r in limited),
        max_nfev_failed_seconds=sum(r['seconds'] for r in limited),
        failed_seconds=sum(r['seconds'] for r in failures), bins=bins,
        largest_attempts=[dict(index=r['index'],alpha=r['alpha'],accepted=r['accepted'],
            seconds=r['seconds'], evaluations=r['evaluation_count'],
            initial_residual=r['initial_residual'],final_residual=final_error(r),
            best_trial_residual=r['best_trial_residual'],status=r['raw']['status'],
            nfev=r['raw']['nfev']) for r in sorted(rows,key=lambda r:r['seconds'],reverse=True)[:10]],
        verification=verification)


def main():
    current = Path(sys.argv[1])
    old = ROOT/'outputs/base-start-trace-2026-10-22/legacy'
    result = dict(current=analyze(current), legacy=analyze(old))
    (current/'comparison.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    compact = {k:{name:value for name,value in v.items() if name!='verification'}
               for k,v in result.items()}
    print(json.dumps(compact,indent=2))
    for name, data in result.items():
        print(name, 'DOP853 max residual', max(r['residual'] for r in data['verification']),
              'final',data['verification'][-1])
    with np.load(current/'accepted.npz') as data:
        if data['alpha'][-1] == 1.:
            s = OptimalControlSolver(1.,float(data['a_max'][-1]),1.,'mixed')
            s.set_boundary_conditions([1.,0.,0.], [0.,1.,0.], [-1.5,0.,0.],
                [0.,-1/np.sqrt(1.5),0.], 2*np.pi*T_a(.5,1.5), lambda_value=.5)
            s.p0 = data['p0'][-1].copy()
            s.save_alpha1(current/'alpha1.pkl')


if __name__ == '__main__':
    main()
