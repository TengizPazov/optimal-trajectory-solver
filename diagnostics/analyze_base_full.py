"""Compare saved correction traces and independently verify accepted states."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from diagnostics.common import make_solver, verify_solutions, write_json


def analyze(folder):
    rows = json.loads((folder / 'attempts.json').read_text(encoding='utf-8'))
    failures = [r for r in rows if not r['accepted']]
    limited = [r for r in failures if (r.get('raw') or {}).get('status') == 0]
    def final_error(r):
        raw=r.get('raw') or {}
        return float(np.linalg.norm(raw['fun'])) if 'fun' in raw else None
    bins = []
    for low, high in ((0., .2), (.2, .5), (.5, .6), (.6, .7), (.7, .8), (.8, 1.000001)):
        subset = [r for r in rows if low <= r['alpha'] < high]
        bins.append(dict(alpha_range=[low, min(high, 1.)], attempts=len(subset),
            failed=sum(not r['accepted'] for r in subset),
            evaluations=sum(r['evaluation_count'] for r in subset),
            seconds=sum(r['seconds'] for r in subset)))
    s=make_solver({'lambda':.5,'kappa':1.5,'snap_phase':True})
    verification=verify_solutions(s,folder/'accepted.npz')
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
            best_trial_residual=r['best_trial_residual'],status=(r.get('raw') or {}).get('status'),
            nfev=(r.get('raw') or {}).get('nfev')) for r in sorted(rows,key=lambda r:r['seconds'],reverse=True)[:10]],
        verification=verification)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('current',type=Path)
    parser.add_argument('--reference',type=Path,help='Optional saved base trace to compare')
    args=parser.parse_args()
    current=args.current
    result=dict(current=analyze(current))
    if args.reference:
        result['reference']=analyze(args.reference)
    write_json(current/'comparison.json',result)
    compact = {k:{name:value for name,value in v.items() if name!='verification'}
               for k,v in result.items()}
    print(json.dumps(compact,indent=2))
    for name, data in result.items():
        print(name, 'DOP853 max residual', max(r['residual'] for r in data['verification']),
              'final',data['verification'][-1])
    with np.load(current/'accepted.npz') as data:
        if data['alpha'][-1] == 1.:
            s=make_solver({'lambda':.5,'kappa':1.5,'snap_phase':True})
            s.alpha,s.a_max=1.,float(data['a_max'][-1])
            s.p0 = data['p0'][-1].copy()
            s.save_alpha1(current/'alpha1.pkl')


if __name__ == '__main__':
    main()
