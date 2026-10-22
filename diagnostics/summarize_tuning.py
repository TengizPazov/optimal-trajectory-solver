"""Collect hyperparameter trials without conflating evaluation budgets with failures."""
import argparse
import csv
import json
from pathlib import Path
import statistics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('folders',type=Path,nargs='+')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    rows=[]
    for folder in args.folders:
        for result in sorted(folder.glob('*/result.json')):
            r=json.loads(result.read_text(encoding='utf-8'))
            case=r['case']; profile=r.get('profile',case['profile'])
            attempts_path=result.parent/'corrector_attempts.json'
            attempts=json.loads(attempts_path.read_text()) if attempts_path.exists() else []
            rows.append(dict(series=folder.name,id=case['id'],geometry=case['geometry'],
                lambda_value=case['lambda'],kappa=case['kappa'],tol=case.get('tol',.01),
                profile=profile['id'],status=r['status'],evaluations=r['tuning_evaluations'],
                seconds=r['elapsed_seconds'],attempts=r.get('attempts'),
                accepted=r.get('solutions_count'),
                last_alpha=r.get('last_alpha',r.get('last_accepted',{}).get('alpha')),
                max_independent_residual=r.get('maximum_independent_residual'),
                final_independent_residual=(r.get('all_checkpoint_checks') or [{}])[-1].get('residual'),
                final_a_max=(r.get('solutions') or [{}])[-1].get('a_max'),
                final_L=(r.get('solutions') or [{}])[-1].get('L'),
                max_nfev_stops=sum(a['status']==0 for a in attempts),
                evaluation_budget=case.get('evaluation_budget'),path=str(result)))
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'trials.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    groups=[]
    for name in sorted({r['profile'] for r in rows}):
        subset=[r for r in rows if r['profile']==name]
        good=[r for r in subset if r['status']=='completed']
        groups.append(dict(profile=name,trials=len(subset),completed=len(good),
            statuses={status:sum(r['status']==status for r in subset)
                      for status in sorted({r['status'] for r in subset})},
            total_evaluations=sum(r['evaluations'] for r in subset),
            median_completed_evaluations=statistics.median(r['evaluations'] for r in good) if good else None))
    (args.output/'tables.json').write_text(json.dumps(dict(groups=groups,trials=rows),indent=2),encoding='utf-8')
    print(json.dumps(groups,indent=2))


if __name__=='__main__':main()
