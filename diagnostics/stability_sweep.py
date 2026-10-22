"""Targeted stability diagnostics; does not change the production solver."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    import math
    def clean(x):
        if isinstance(x, dict): return {k: clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)): return [clean(v) for v in x]
        if isinstance(x, float) and not math.isfinite(x): return None
        return x
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2), encoding='utf-8')


def worker(case, folder):
    os.environ['MPLBACKEND'] = 'Agg'
    sys.path.insert(0, str(ROOT))
    import numpy as np
    from scipy.integrate import solve_ivp
    from approximations import T_a, get_initial_adjoint
    from solver import OptimalControlSolver
    logging.getLogger('solver').setLevel(logging.WARNING)
    start = time.perf_counter()
    record = dict(case=case, status='error')
    class Probe(OptimalControlSolver):
        evaluations = 0
        attempts = 0
        def residual_vec(self, x):
            self.evaluations += 1
            return super().residual_vec(x)
        def solve(self, *args, **kwargs):
            self.attempts += 1
            write_json(folder/'progress.json', dict(alpha=self.alpha, attempts=self.attempts,
                       step_floor_history=getattr(self,'step_floor_history',[]),
                       evaluations=self.evaluations, elapsed_seconds=time.perf_counter()-start))
            result = super().solve(*args, **kwargs)
            if result.success and self.trajectory is not None:
                write_json(folder/'last_accepted.json',dict(alpha=float(self.alpha),
                    a_max=float(self.a_max),residual=float(np.linalg.norm(self.trajectory_residual(self.trajectory)))))
                np.savez_compressed(folder/'last_accepted.npz',p0=self.p0,
                    t=self.trajectory.t,y=self.trajectory.y)
            write_json(folder/'progress.json', dict(alpha=self.alpha, attempts=self.attempts,
                       step_floor_history=getattr(self,'step_floor_history',[]),
                       evaluations=self.evaluations, success=bool(result.success),
                       elapsed_seconds=time.perf_counter()-start))
            return result
    s = Probe(1., .01, 0., 'mixed')
    try:
        lam, kap = case['lambda'], case['kappa']
        theta = 2*np.pi*lam
        sine, cosine = np.sin(theta), np.cos(theta)
        if case.get('snap_phase'):
            if abs(sine) < 1e-14: sine = 0.0
            if abs(cosine) < 1e-14: cosine = 0.0
        T = float(2*np.pi*T_a(lam, kap))
        s.set_boundary_conditions([1.,0.,0.],[0.,1.,0.],
            [kap*cosine,kap*sine,0.],
            [-sine/np.sqrt(kap),cosine/np.sqrt(kap),0.], T,lambda_value=lam)
        record['boundary_conditions']={k:getattr(s,k).tolist() for k in ['r0','v0','rf','vf']}
        record['T'] = T
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
            record.update(status='accepted' if result.success else 'initial_failed',
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
            record.update(status=s.continuation_status['state'],
                          continuation=s.continuation_status,
                          steps=[[float(a),float(h),bool(ok)] for a,h,ok in s.step_log])
            record['step_floor_history']=s.step_floor_history
        record['initial_guess_info']=getattr(s,'initial_guess_info',None)
        record.update(evaluations=s.evaluations, attempts=s.attempts, solutions_count=len(results))
        summaries=[]
        arrays={}
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
            arrays[f't_{i}']=tr.t; arrays[f'y_{i}']=tr.y
        record['solutions']=summaries
        if results:
            r=results[-1]; s.alpha=r['alpha'];s.a_max=r['a_max'];s.p0=r['p0']
            record['last_alpha']=float(s.alpha)
            # Independent forward check only: same shooting parameters, no correction.
            check=solve_ivp(s.system_equations,[0,T],np.r_[s.r0,s.v0,s.p0],
                method='DOP853',rtol=1e-10,atol=1e-12,max_step=T/200)
            record['independent_check']=dict(success=bool(check.success),
                residual=float(np.linalg.norm(s.trajectory_residual(check))),points=len(check.t))
            arrays['alpha']=np.array([r['alpha'] for r in results])
            arrays['a_max']=np.array([r['a_max'] for r in results])
            arrays['p0']=np.array([r['p0'] for r in results])
            np.savez_compressed(folder/'solutions.npz',**arrays)
    except Exception as exc:
        record.update(status='error',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
    record['elapsed_seconds']=time.perf_counter()-start
    write_json(folder/'result.json',record)


def run_case(case, output, timeout):
    folder=output/case['id'];folder.mkdir()
    write_json(folder/'case.json',case)
    env=os.environ.copy()
    for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
        env[name]='1'
    start=time.perf_counter()
    with (folder/'run.log').open('w',encoding='utf-8') as log:
        try:
            completed=subprocess.run([sys.executable,'-X','utf8','-B',str(Path(__file__).resolve()),
                '--worker',str(folder/'case.json')],stdout=log,stderr=log,env=env,timeout=timeout)
            if not (folder/'result.json').exists():
                write_json(folder/'result.json',dict(case=case,status='process_error',exit_code=completed.returncode))
        except subprocess.TimeoutExpired:
            progress=json.loads((folder/'progress.json').read_text(encoding='utf-8')) if (folder/'progress.json').exists() else None
            write_json(folder/'result.json',dict(case=case,status='timeout',budget_seconds=timeout,
                elapsed_seconds=time.perf_counter()-start,last_progress=progress))
    result=json.loads((folder/'result.json').read_text(encoding='utf-8'))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',type=Path)
    parser.add_argument('--plan',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--jobs',type=int,default=4)
    parser.add_argument('--timeout',type=float,default=180)
    args=parser.parse_args()
    if args.worker:
        worker(json.loads(args.worker.read_text(encoding='utf-8-sig')),args.worker.parent)
        return
    cases=json.loads(args.plan.read_text(encoding='utf-8-sig'))
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'plan.json',cases)
    records=[]
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        tasks=[pool.submit(run_case,c,args.output,args.timeout) for c in cases]
        for task in as_completed(tasks):
            r=task.result();records.append(r)
            write_json(args.output/'results.json',records)
            print(r['case']['id'],r['status'],round(r.get('elapsed_seconds',0),2),flush=True)


if __name__=='__main__':main()
