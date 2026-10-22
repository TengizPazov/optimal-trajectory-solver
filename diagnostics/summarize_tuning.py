"""Collect hyperparameter trials without conflating evaluation budgets with failures."""
import argparse
import csv
import json
from pathlib import Path
import statistics
from itertools import combinations


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('folders',type=Path,nargs='+')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    rows=[]
    for folder in args.folders:
        for result in sorted(folder.glob('*/result.json')):
            r=json.loads(result.read_text(encoding='utf-8'))
            case=r['case']; profile=r.get('profile',case.get('profile',{}))
            verification=r.get('verification_status',
                'passed' if r.get('independently_valid') else 'not_recorded')
            status=r['status']
            if status=='completed' and verification!='passed':
                status='unverified'
            attempts_path=result.parent/'corrector_attempts.json'
            attempts=json.loads(attempts_path.read_text()) if attempts_path.exists() else []
            rows.append(dict(series=folder.name,id=case['id'],geometry=case['geometry'],
                lambda_value=case['lambda'],kappa=case['kappa'],tol=case.get('tol',.01),
                profile=profile.get('id','unknown'),status=status,verification_status=verification,
                evaluations=r.get('tuning_evaluations',r.get('evaluations')),
                seconds=r.get('elapsed_seconds'),attempts=r.get('attempts'),
                accepted=r.get('solutions_count'),
                last_alpha=r.get('last_alpha',(r.get('last_accepted') or {}).get('alpha')),
                max_independent_residual=r.get('maximum_independent_residual'),
                final_independent_residual=(r.get('all_checkpoint_checks') or [{}])[-1].get('residual'),
                final_a_max=(r.get('solutions') or [{}])[-1].get('a_max'),
                final_L=(r.get('solutions') or [{}])[-1].get('L'),
                max_nfev_stops=sum(a['status']==0 for a in attempts),
                evaluation_budget=case.get('evaluation_budget'),path=str(result),
                task_key=json.dumps({key:case.get(key,default) for key,default in
                    [('lambda',None),('kappa',None),('tol',.01),('phase','full'),
                     ('snap_phase',False),('alpha_start',0.),('alpha_end',1.),('seed','auto'),
                     ('seed_from',None)]},sort_keys=True)))
    if not rows:
        p.error('No per-case result.json files found')
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'trials.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    groups=[]
    for series,name in sorted({(r['series'],r['profile']) for r in rows}):
        subset=[r for r in rows if r['profile']==name and r['series']==series]
        good=[r for r in subset if r['status']=='completed']
        costs=[r['evaluations'] for r in good if r['evaluations'] is not None]
        groups.append(dict(series=series,profile=name,trials=len(subset),completed=len(good),
            statuses={status:sum(r['status']==status for r in subset)
                      for status in sorted({r['status'] for r in subset})},
            total_known_evaluations=sum(r['evaluations'] or 0 for r in subset),
            median_completed_evaluations=statistics.median(costs) if costs else None))
    paired=[]
    for left,right in combinations(sorted({r['profile'] for r in rows}),2):
        by_profile={name:{} for name in (left,right)}
        for r in rows:
            if r['profile'] in by_profile:
                by_profile[r['profile']].setdefault(r['task_key'],[]).append(r)
        shared=by_profile[left].keys() & by_profile[right].keys()
        values=[]
        for key in shared:
            runs=[by_profile[name][key] for name in (left,right)]
            if all(all(r['status']=='completed' and r['evaluations'] is not None for r in group)
                   for group in runs):
                values.append([statistics.median(r['evaluations'] for r in group) for group in runs])
        paired.append(dict(left=left,right=right,shared_tasks=len(shared),
            paired_completed_tasks=len(values),excluded_tasks=len(shared)-len(values),
            left_evaluations=sum(v[0] for v in values),right_evaluations=sum(v[1] for v in values)))
    (args.output/'tables.json').write_text(json.dumps(dict(groups=groups,trials=rows,
        paired_completed_comparisons=paired),indent=2),encoding='utf-8')
    print(json.dumps(groups,indent=2))


if __name__=='__main__':main()
