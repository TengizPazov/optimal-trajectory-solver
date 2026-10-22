"""Process-isolated trials using explicit corrector profiles and shared checks."""
import argparse
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from diagnostics.common import read_json, read_plan, run_plan, source_hashes, write_json
from diagnostics import stability_sweep
from solver_settings import DEFAULT_PROFILE, load_profile


class EvaluationBudget(Exception):
    pass


class TrialObserver:
    def __init__(self, folder, budget):
        self.folder, self.budget = folder, budget
        self.calls, self.attempts = 0, []

    def before_evaluation(self, solver, x):
        if self.budget is not None and self.calls >= self.budget:
            raise EvaluationBudget('screening evaluation budget exhausted')
        self.calls += 1

    def before_solve(self, solver):
        initialization = (solver.free_a_max and abs(solver.alpha)<=1e-12
                          and solver.func_type in ('mixed','quadratic'))
        self.before = self.calls
        self.row = dict(index=len(self.attempts),alpha=solver.alpha,initialization=initialization,
                        settings=dict(solver.initial_corrector_options if initialization
                                      else solver.corrector_options))

    def after_solve(self, solver, result, error, seconds):
        import numpy as np
        if result is not None:
            self.row.update(status=int(result.status),optimizer_success=result.status>0,
                accepted=bool(result.success),nfev=int(result.nfev),
                residual=float(np.linalg.norm(result.fun)),optimality=float(result.optimality))
        else:
            self.row.update(status='evaluation_budget' if isinstance(error,EvaluationBudget) else 'error',
                            error=f'{type(error).__name__}: {error}')
        self.row.update(seconds=seconds,evaluations=self.calls-self.before)
        self.attempts.append(self.row)
        write_json(self.folder/'corrector_attempts.json',self.attempts)


def worker(path):
    os.environ['MPLBACKEND']='Agg'
    case=read_json(path)
    profile=case['profile']
    observer=TrialObserver(path.parent,case.get('evaluation_budget'))
    record=dict(case=case,status='running',stage='starting',verification_status='not_run')
    write_json(path.parent/'result.json',record)
    try:
        # Explicit frozen baseline for old plans; production defaults remain untouched.
        options=dict(DEFAULT_PROFILE['corrector'],diff_step=None)
        options.update(profile.get('corrector',{}))
        effective=dict(case)
        effective.update(profile.get('continuation',{}))
        record=stability_sweep.worker(effective,path.parent,observer=observer,
                                      corrector_options=options,verify_all=True,publish=False)
        if record.get('error','').startswith('EvaluationBudget:'):
            record['status']='evaluation_budget'
            checkpoint=path.parent/'last_accepted.json'
            record['last_accepted']=read_json(checkpoint) if checkpoint.exists() else None
    except Exception as exc:
        record.update(status='error',error=f'{type(exc).__name__}: {exc}',
                      traceback=traceback.format_exc())
    finally:
        record.update(case=case,profile=profile,tuning_evaluations=observer.calls,
                      initialization_unchanged=True,
                      source_sha256=source_hashes())
        write_json(path.parent/'result.json',record)
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',type=Path)
    parser.add_argument('--plan',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--profile',type=Path,help='Override profiles using a saved profile or reference')
    parser.add_argument('--jobs',type=int,default=3)
    args=parser.parse_args()
    if args.worker:
        worker(args.worker)
        return
    cases=read_plan(args.plan)
    if args.profile:
        profile=load_profile(args.profile)
        for case in cases:
            case['profile']=profile
    run_plan(cases,args.output,Path(__file__).resolve(),jobs=args.jobs)


if __name__=='__main__':
    main()
