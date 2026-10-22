"""Collect targeted runs into a CSV; timeouts remain distinct from failures."""
import argparse
import csv
import json
from pathlib import Path


def collect(inputs):
    rows=[]
    for directory in inputs:
        for source in sorted(directory.glob('*/result.json')):
            r=json.loads(source.read_text(encoding='utf-8')); c=r['case']
            final=(r.get('solutions') or [{}])[-1]
            progress=r.get('last_progress') or {}
            accepted_path=source.parent/'last_accepted.json'
            accepted=json.loads(accepted_path.read_text(encoding='utf-8')) if accepted_path.exists() else {}
            rows.append(dict(stage=directory.name,case=c['id'],phase=c['phase'],
                lambda_value=c['lambda'],kappa=c['kappa'],seed=c.get('seed','default'),
                tol=c.get('tol',.01),h_init=c.get('step',.05),h_max=c.get('h_max',.2),
                status=r['status'],seconds=r.get('elapsed_seconds'),
                attempts=r.get('attempts',progress.get('attempts')),
                evaluations=r.get('evaluations',progress.get('evaluations')),
                accepted_solutions=r.get('solutions_count'),
                reached_alpha=r.get('last_alpha',accepted.get('alpha')),
                last_attempted_alpha=progress.get('alpha'),
                final_residual=final.get('residual'),
                dop853_residual=r.get('independent_check',{}).get('residual'),
                final_a_max=final.get('a_max'),actual_turns=final.get('turns'),
                active_arcs=final.get('active_arcs'),minimum_radius=final.get('minimum_radius'),
                error=r.get('error',r.get('message','')),result_file=str(source.resolve())))
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs',type=Path,nargs='+')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    rows=collect(args.inputs)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    print(f'{len(rows)} cases -> {args.output}')


if __name__=='__main__':main()
