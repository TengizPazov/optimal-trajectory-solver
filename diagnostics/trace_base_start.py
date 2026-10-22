"""Paired trace of the existing corrector for one exact base problem."""
import argparse,logging,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ['MPLBACKEND']='Agg'
sys.path.insert(0,str(ROOT))
import numpy as np
from approximations import get_initial_adjoint
import solver as module
from solver import OptimalControlSolver
from diagnostics.common import make_solver, save_solutions, source_hashes, write_json as write


def main():
    p=argparse.ArgumentParser();p.add_argument('seed',choices=['legacy','actual']);p.add_argument('output',type=Path);p.add_argument('--seconds',type=float,default=None)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    logging.getLogger('solver').setLevel(logging.WARNING)
    started=time.perf_counter();attempts=[];accepted=[]
    original=module.least_squares
    class BudgetStop(Exception):pass
    class Trace(OptimalControlSolver):
        def residual_vec(self,x):
            if args.seconds is not None and time.perf_counter()-started>args.seconds:raise BudgetStop()
            value=super().residual_vec(x)
            self.evaluations.append(dict(x=x.copy(),residual=float(np.linalg.norm(value))))
            return value
        def solve(self,p0_guess=None,tol=.01):
            self.evaluations=[];self.raw_result=None
            initial=self.initial_guess() if p0_guess is None else np.asarray(p0_guess).copy()
            t=time.perf_counter()
            row=dict(index=len(attempts),alpha=self.alpha,x_initial=initial,
                     elapsed_start=t-started)
            def wrapped(*a,**kw):
                result=original(*a,**kw)
                self.raw_result=dict(success=result.success,message=str(result.message),
                    status=result.status,nfev=result.nfev,njev=result.njev,
                    optimality=result.optimality,x=result.x.copy(),fun=result.fun.copy(),
                    jac=result.jac.copy(),jac_column_norm=np.linalg.norm(result.jac,axis=0),
                    jac_singular_values=np.linalg.svd(result.jac,compute_uv=False))
                return result
            module.least_squares=wrapped
            try:
                res=super().solve(initial,tol=tol)
                row.update(accepted=bool(res.success),message=str(res.message),raw=self.raw_result)
                if self.trajectory is not None:
                    y=self.trajectory.y;q=np.linalg.norm(y[9:12],axis=0)
                    row.update(A=self.a_max,p0=self.p0.copy(),residual=float(np.linalg.norm(self.trajectory_residual(self.trajectory))),
                        q_max=float(q.max()),q_initial=float(q[0]),q_min=float(q.min()))
                    accepted.append(dict(alpha=self.alpha,a_max=self.a_max,p0=self.p0.copy(),
                                         trajectory=self.trajectory))
                    save_solutions(args.output/'accepted.npz',accepted)
                return res
            except BudgetStop:
                row.update(accepted=False,budget_stop=True)
                raise
            finally:
                module.least_squares=original
                row.update(seconds=time.perf_counter()-t,evaluation_count=len(self.evaluations),
                    initial_residual=self.evaluations[0]['residual'] if self.evaluations else None,
                    best_trial_residual=min((v['residual'] for v in self.evaluations),default=None))
                write(args.output/f'attempt_{row["index"]:03}.json',dict(summary=row,evaluations=self.evaluations))
                attempts.append(row);write(args.output/'attempts.json',attempts)
                print(args.seed,row['index'],f'alpha={self.alpha:.8f}',row.get('accepted'),row.get('raw',{}).get('status'),round(row['seconds'],2),flush=True)
    s=make_solver({'lambda':.5,'kappa':1.5,'snap_phase':True},Trace)
    T=s.T
    seed=None
    if args.seed=='legacy':
        pr,pv=get_initial_adjoint(.5,1.2);seed=np.r_[pr,pv,max(np.linalg.norm(pv),.1)]
    try:
        s.continuation_method(p0_initial=seed,h_min_floor=1e-5)
        state=s.continuation_status['state']
    except BudgetStop:state='budget_stop'
    write(args.output/'summary.json',dict(seed=args.seed,status=state,seconds=time.perf_counter()-started,
        accepted_count=len(accepted),accepted_alpha=accepted[-1]['alpha'] if accepted else None,
        attempts=len(attempts),floor_history=s.step_floor_history,T=T,
        boundary_conditions={name:getattr(s,name) for name in ('r0','v0','rf','vf')},
        seconds_limit=args.seconds,initial_quadratic=getattr(s,'initial_quadratic_info',None),
        source_sha256=source_hashes()))


if __name__=='__main__':main()
