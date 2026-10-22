"""Reproducible, process-isolated hyperparameter trials; production defaults unchanged."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stability_sweep import write_json


def worker(path):
    os.environ['MPLBACKEND'] = 'Agg'
    import numpy as np
    from scipy.integrate import solve_ivp
    import solver as module
    import stability_sweep
    case = json.loads(path.read_text(encoding='utf-8'))
    profile = case['profile']
    budget = case.get('evaluation_budget')
    original = module.least_squares
    calls, attempts = 0, []
    class EvaluationBudget(Exception):
        pass
    def wrapped(fun, x0, *args, **kwargs):
        # The unbounded quadratic initialization is identical for all profiles.
        bounded = 'bounds' in kwargs
        if bounded:
            # Freeze the pre-tuning FD baseline even after production adopts diff5.
            kwargs['diff_step'] = None
            kwargs.update(profile.get('corrector', {}))
        start = time.perf_counter()
        before = calls
        row = dict(index=len(attempts), initialization=not bounded,
                   settings={k: v for k, v in kwargs.items() if k != 'bounds'})
        progress_path = path.parent/'progress.json'
        if progress_path.exists():
            row['alpha'] = json.loads(progress_path.read_text(encoding='utf-8'))['alpha']
        def counted(x):
            nonlocal calls
            if budget is not None and calls >= budget:
                raise EvaluationBudget('screening evaluation budget exhausted')
            calls += 1
            return fun(x)
        try:
            result = original(counted, x0, *args, **kwargs)
            row.update(status=int(result.status), success=bool(result.success),
                nfev=int(result.nfev), residual=float(np.linalg.norm(result.fun)),
                optimality=float(result.optimality))
            return result
        except EvaluationBudget:
            row['status'] = 'evaluation_budget'
            raise
        finally:
            row.update(seconds=time.perf_counter()-start,evaluations=calls-before)
            attempts.append(row)
            write_json(path.parent/'corrector_attempts.json',attempts)
    module.least_squares = wrapped
    effective = dict(case)
    effective.update(profile.get('continuation', {}))
    stability_sweep.worker(effective,path.parent)
    result_path = path.parent/'result.json'
    record = json.loads(result_path.read_text(encoding='utf-8'))
    if record.get('error','').startswith('EvaluationBudget:'):
        record['status'] = 'evaluation_budget'
        checkpoint = path.parent/'last_accepted.json'
        record['last_accepted'] = json.loads(checkpoint.read_text()) if checkpoint.exists() else None
    record['tuning_evaluations'] = calls
    record['profile'] = profile
    record['initialization_unchanged'] = True
    record['source_sha256'] = {n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest()
                              for n in ('solver.py','approximations.py')}
    # Verify every accepted checkpoint after a completed trial, not only its endpoint.
    data_path = path.parent/'solutions.npz'
    checks = []
    if data_path.exists():
        with np.load(data_path) as data:
            s = module.OptimalControlSolver(1.,.01,0.,'mixed')
            b = record['boundary_conditions']
            s.set_boundary_conditions(b['r0'],b['v0'],b['rf'],b['vf'],record['T'],
                                      lambda_value=case['lambda'])
            for i,alpha in enumerate(data['alpha']):
                s.alpha, s.a_max = float(alpha),float(data['a_max'][i])
                sol=solve_ivp(s.system_equations,[0.,s.T],np.r_[s.r0,s.v0,data['p0'][i]],
                    method='DOP853',rtol=1e-11,atol=1e-13,max_step=s.T/200)
                residual=float(np.linalg.norm(s.trajectory_residual(sol)))
                checks.append(dict(alpha=s.alpha,success=bool(sol.success),residual=residual))
        record['all_checkpoint_checks'] = checks
        record['independently_valid'] = all(c['success'] and c['residual']<=case.get('tol',.01)
                                            for c in checks)
        record['maximum_independent_residual'] = max(c['residual'] for c in checks)
        if record['status']=='completed' and not record['independently_valid']:
            record['status']='verification_failed'
    write_json(result_path,record)


def run(case, output):
    folder=output/case['id']
    folder.mkdir()
    write_json(folder/'case.json',case)
    env=os.environ.copy()
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        env[name]='1'
    with (folder/'run.log').open('w',encoding='utf-8') as log:
        process=subprocess.run([sys.executable,'-X','utf8','-B',str(Path(__file__).resolve()),
            '--worker',str(folder/'case.json')],env=env,stdout=log,stderr=log)
    result=folder/'result.json'
    if process.returncode != 0:
        return dict(case=case,status='process_error',exit_code=process.returncode)
    return json.loads(result.read_text(encoding='utf-8'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',type=Path)
    parser.add_argument('--plan',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--profile',type=Path,help='Override profiles in the plan with one saved profile')
    parser.add_argument('--jobs',type=int,default=3)
    args=parser.parse_args()
    if args.worker:
        worker(args.worker)
        return
    cases=json.loads(args.plan.read_text(encoding='utf-8'))
    if args.profile:
        profile=json.loads(args.profile.read_text(encoding='utf-8'))
        for case in cases:
            case['profile']=profile
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'plan.json',cases)
    records=[]
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures=[pool.submit(run,c,args.output) for c in cases]
        for future in as_completed(futures):
            r=future.result(); records.append(r)
            write_json(args.output/'results.json',records)
            print(r['case']['id'],r['status'],r.get('tuning_evaluations'),
                  round(r.get('elapsed_seconds',0),2),flush=True)


if __name__=='__main__':
    main()
