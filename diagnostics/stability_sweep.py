"""Targeted stability diagnostics; does not change the production solver."""
import argparse
import logging
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from diagnostics.common import make_solver, read_json, read_plan, run_plan, save_solutions, verify_solutions, write_json


def worker(case, folder, *, observer=None, corrector_options=None, verify_all=False, publish=True):
    os.environ['MPLBACKEND'] = 'Agg'
    import numpy as np
    from approximations import get_initial_adjoint
    from solver import OptimalControlSolver
    logging.getLogger('solver').setLevel(logging.WARNING)
    start = time.perf_counter()
    record = dict(case=case, status='running', stage='solve', verification_status='not_run')
    write_json(folder/'result.json',record)
    accepted = []
    class Probe(OptimalControlSolver):
        evaluations = 0
        attempts = 0
        def residual_vec(self, x):
            if observer is not None:
                observer.before_evaluation(self,x)
            self.evaluations += 1
            return super().residual_vec(x)
        def solve(self, *args, **kwargs):
            self.attempts += 1
            write_json(folder/'progress.json', dict(alpha=self.alpha, attempts=self.attempts,
                       step_floor_history=getattr(self,'step_floor_history',[]),
                       evaluations=self.evaluations, elapsed_seconds=time.perf_counter()-start))
            if observer is not None:
                observer.before_solve(self)
            solve_start = time.perf_counter()
            result, error = None, None
            try:
                result = super().solve(*args, **kwargs)
            except Exception as exc:
                error = exc
                raise
            finally:
                if observer is not None:
                    observer.after_solve(self,result,error,time.perf_counter()-solve_start)
            if result.success and self.trajectory is not None:
                accepted.append(dict(alpha=float(self.alpha),a_max=float(self.a_max),
                                     p0=self.p0.copy(),trajectory=self.trajectory))
                write_json(folder/'last_accepted.json',dict(alpha=float(self.alpha),
                    a_max=float(self.a_max),residual=float(np.linalg.norm(self.trajectory_residual(self.trajectory)))))
                np.savez_compressed(folder/'last_accepted.npz',p0=self.p0,
                    t=self.trajectory.t,y=self.trajectory.y)
            write_json(folder/'progress.json', dict(alpha=self.alpha, attempts=self.attempts,
                       step_floor_history=getattr(self,'step_floor_history',[]),
                       evaluations=self.evaluations, success=bool(result.success),
                       elapsed_seconds=time.perf_counter()-start))
            return result
    s = make_solver(case,Probe)
    if corrector_options is not None:
        s.corrector_options = dict(corrector_options)
    try:
        lam, kap = case['lambda'], case['kappa']
        record['boundary_conditions']={k:getattr(s,k).tolist() for k in ['r0','v0','rf','vf']}
        record['T'] = s.T
        record['corrector'] = s.corrector_configuration()
        record['a_max_constructor'] = s.a_max_constructor
        seed = None
        if case.get('seed') == 'default':
            # Исторический контроль D1: старое приближение независимо от геометрии.
            pr,pv=get_initial_adjoint(.5,1.2)
            seed=np.r_[pr,pv,max(float(np.linalg.norm(pv)),.1)]
        if case.get('seed') in ('matched', 'matched_capped'):
            pr, pv = get_initial_adjoint(lam, kap)
            seed = np.r_[pr,pv,max(float(np.linalg.norm(pv)),.1)]
            if case['seed'] == 'matched_capped': seed[-1]=min(seed[-1],1.)
        if case.get('seed_from'):
            seed_path=Path(case['seed_from'])
            if not seed_path.is_absolute(): seed_path=ROOT/seed_path
            with np.load(seed_path) as saved:
                seed=np.r_[saved['p0'][-1],saved['a_max'][-1]]
        record['initial_seed'] = None if seed is None else seed.tolist()
        if case['phase'] == 'initial':
            result = s.solve(seed, tol=case.get('tol', .01))
            record.update(solve_status='accepted' if result.success else 'initial_failed',
                          message=str(result.message), nfev=int(result.nfev))
            results = [] if s.trajectory is None else [dict(alpha=0., a_max=s.a_max,
                       p0=s.p0.copy(), trajectory=s.trajectory)]
        else:
            step = case.get('step', .05)
            results = s.continuation_method(p0_initial=seed, h_init=step,
                        h_max=case.get('h_max', .2), h_min=case.get('h_min',1e-4),
                        h_min_floor=case.get('h_min_floor'),
                        alpha_start=case.get('alpha_start',0.),alpha_end=case.get('alpha_end',1.),
                        tol=case.get('tol', .01))
            record.update(solve_status=s.continuation_status['state'],
                          continuation=s.continuation_status,
                          steps=[[float(a),float(h),bool(ok)] for a,h,ok in s.step_log])
            record['step_floor_history']=s.step_floor_history
        record['initial_guess_info']=getattr(s,'initial_guess_info',None)
        record['initial_quadratic']=getattr(s,'initial_quadratic_info',None)
        record.update(evaluations=s.evaluations, attempts=s.attempts, solutions_count=len(results))
        summaries=[]
        for i,r in enumerate(results):
            tr=r['trajectory']; s.alpha=r['alpha']; s.a_max=r['a_max']
            acceleration=np.array([s.optimal_control(pv) for pv in tr.y[9:12].T])
            thrust=np.linalg.norm(acceleration,axis=1)
            dr=s.rf-tr.y[:3,-1]; dv=s.vf-tr.y[3:6,-1]
            radius=np.linalg.norm(tr.y[:3],axis=0)
            H=np.sum(tr.y[6:9]*tr.y[3:6],axis=0)+np.sum(tr.y[9:12]*(-tr.y[:3]/radius**3+acceleration.T),axis=0)-.5*(1-s.alpha)*thrust**2-s.alpha*thrust
            active=thrust>=max(1e-4,1e-3*s.a_max)
            summaries.append(dict(alpha=float(s.alpha),a_max=float(s.a_max),
                residual=float(np.linalg.norm(np.r_[dr,dv])),minimum_radius=float(radius.min()),
                turns=float((np.unwrap(np.arctan2(tr.y[1],tr.y[0]))[-1])/(2*np.pi)),
                active_arcs=int(active[0])+int(np.sum(active[1:] & ~active[:-1])),
                Q=float(.5*np.trapezoid(thrust**2,tr.t)),L=float(np.trapezoid(thrust,tr.t)),
                hamiltonian_span=float(np.ptp(H)),points=len(tr.t)))
        record['solutions']=summaries
        if results:
            r=results[-1]; s.alpha=r['alpha'];s.a_max=r['a_max'];s.p0=r['p0']
            record['last_alpha']=float(s.alpha)
            save_solutions(folder/'solutions.npz',results)
            record['stage']='verification'
            write_json(folder/'result.json',record)
            checks=verify_solutions(s,folder/'solutions.npz',all_points=verify_all,
                rtol=1e-11 if verify_all else 1e-10,atol=1e-13 if verify_all else 1e-12)
            record['independent_check']=checks[-1]
            record['all_checkpoint_checks']=checks
            record['maximum_independent_residual']=max(c['residual'] for c in checks)
            record['independently_valid']=all(c['success'] and c['residual']<=case.get('tol',.01) for c in checks)
            record['verification_status']='passed' if record['independently_valid'] else 'failed'
        record['status'] = record['solve_status']
        if record.get('independently_valid') is False:
            record['status']='verification_failed'
        record['stage']='done'
    except Exception as exc:
        record.update(status='error',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
        if record['stage']=='verification':
            record['verification_status']='error'
        if accepted and not (folder/'solutions.npz').exists():
            save_solutions(folder/'solutions.npz',accepted)
        record.update(evaluations=s.evaluations,attempts=s.attempts)
        record.setdefault('solutions_count',len(accepted))
    record['elapsed_seconds']=time.perf_counter()-start
    if publish:
        write_json(folder/'result.json',record)
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',type=Path)
    parser.add_argument('--plan',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--jobs',type=int,default=4)
    parser.add_argument('--timeout',type=float,default=180)
    args=parser.parse_args()
    if args.worker:
        worker(read_json(args.worker),args.worker.parent)
        return
    run_plan(read_plan(args.plan),args.output,Path(__file__).resolve(),jobs=args.jobs,timeout=args.timeout)


if __name__=='__main__':main()
