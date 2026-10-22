"""Shared result protocol, process runner and numerical checks for diagnostics."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import hashlib
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def source_hashes():
    names=('solver.py','approximations.py','solver_settings.py','solver_profile.json','run_continuation.py')
    return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def read_plan(path):
    from solver_settings import load_profile
    path = Path(path)
    cases = read_json(path)
    for case in cases:
        profile = case.get('profile',{})
        if 'profile_file' in profile:
            case['profile'] = load_profile(path.parent/profile['profile_file'])
    return cases


def write_json(path, value):
    def clean(x):
        if hasattr(x, 'tolist'):
            return clean(x.tolist())
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, float) and not math.isfinite(x):
            return None
        return x
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def make_solver(case, solver_type=None):
    import numpy as np
    from approximations import T_a
    from solver import OptimalControlSolver
    lam, kap = case['lambda'], case['kappa']
    sine, cosine = np.sin(2*np.pi*lam), np.cos(2*np.pi*lam)
    if case.get('snap_phase'):
        if abs(sine) < 1e-14:
            sine = 0.
        if abs(cosine) < 1e-14:
            cosine = 0.
    solver_type = solver_type or OptimalControlSolver
    s = solver_type(1., .01, 0., 'mixed')
    s.set_boundary_conditions([1.,0.,0.], [0.,1.,0.],
        [kap*cosine,kap*sine,0.], [-sine/np.sqrt(kap),cosine/np.sqrt(kap),0.],
        float(2*np.pi*T_a(lam,kap)), lambda_value=lam)
    return s


def save_solutions(path, results):
    import numpy as np
    arrays = {name: np.array([r[name] for r in results]) for name in ('alpha','a_max','p0')}
    for i, result in enumerate(results):
        arrays[f't_{i}'] = result['trajectory'].t
        arrays[f'y_{i}'] = result['trajectory'].y
    np.savez_compressed(path, **arrays)


def verify_solutions(solver, path, *, all_points=True, rtol=1e-11, atol=1e-13):
    """Forward checks only; restore the solver state even when integration raises."""
    import numpy as np
    from scipy.integrate import solve_ivp
    original = solver.alpha, solver.a_max
    checks = []
    try:
        with np.load(path) as data:
            indices = range(len(data['alpha'])) if all_points else [len(data['alpha'])-1]
            for i in indices:
                solver.alpha, solver.a_max = float(data['alpha'][i]), float(data['a_max'][i])
                trajectory = solve_ivp(solver.system_equations, [0.,solver.T],
                    np.r_[solver.r0,solver.v0,data['p0'][i]], method='DOP853',
                    rtol=rtol, atol=atol, max_step=solver.T/200)
                checks.append(dict(alpha=solver.alpha, a_max=solver.a_max,
                    success=bool(trajectory.success), points=len(trajectory.t),
                    residual=float(np.linalg.norm(solver.trajectory_residual(trajectory)))))
    finally:
        solver.alpha, solver.a_max = original
    return checks


def run_case(case, output, script, timeout=None):
    """The per-case file is authoritative, including crashes and timeouts."""
    folder = Path(output)/case['id']
    folder.mkdir()
    write_json(folder/'case.json',case)
    result_path = folder/'result.json'
    write_json(result_path,dict(case=case,status='running',stage='starting',
                               verification_status='not_run'))
    env = os.environ.copy()
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        env[name] = '1'
    start = time.perf_counter()
    failure = None
    with (folder/'run.log').open('w',encoding='utf-8') as log:
        try:
            process = subprocess.run([sys.executable,'-X','utf8','-B',str(script),
                '--worker',str(folder/'case.json')],env=env,stdout=log,stderr=log,timeout=timeout)
            if process.returncode:
                failure = dict(status='process_error',exit_code=process.returncode)
        except subprocess.TimeoutExpired:
            failure = dict(status='timeout',budget_seconds=timeout)
        except OSError as exc:
            failure = dict(status='process_error',error=f'{type(exc).__name__}: {exc}')
    try:
        record = read_json(result_path)
    except (OSError, ValueError):
        record = dict(case=case,status='running',stage='unknown')
    if failure is None and record['status'] == 'running':
        failure = dict(status='process_error',error='Worker exited without a final result')
    if (failure is None and record['status'] in ('completed','accepted')
            and record.get('verification_status') != 'passed'):
        failure = dict(status='process_error',error='Worker reported completion without verification')
    if failure:
        record.update(failure,elapsed_seconds=time.perf_counter()-start)
        for name in ('progress','last_accepted'):
            if (folder/f'{name}.json').exists():
                record['last_progress' if name=='progress' else name] = read_json(folder/f'{name}.json')
        # A pre-existing completion cannot survive a process failure.
        if record.get('stage') == 'verification' or record.get('verification_status') == 'passed':
            record['verification_status'] = 'interrupted'
    write_json(result_path,record)
    return record


def run_plan(cases, output, script, *, jobs=3, timeout=None):
    ids = [case['id'] for case in cases]
    if len(ids) != len(set(ids)) or any(Path(i).name != i or i in ('.','..') for i in ids):
        raise ValueError('Case ids must be unique directory names')
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'plan.json',cases)
    records = []
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(run_case,c,output,script,timeout) for c in cases]
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            write_json(output/'results.json',records)
            print(record['case']['id'],record['status'],
                  round(record.get('elapsed_seconds',0),2),flush=True)
    return records
