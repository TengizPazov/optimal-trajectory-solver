"""Compare finite differences at one fixed base checkpoint with tighter integration."""
import json
from pathlib import Path
import sys

import numpy as np
from scipy.integrate import solve_ivp

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from approximations import T_a
from solver import OptimalControlSolver


def main():
    source=ROOT/'outputs/base-full-quadratic-2026-10-22/accepted.npz'
    with np.load(source) as data:
        index=int(np.argmin(abs(data['alpha']-.614278268814087)))
        alpha=float(data['alpha'][index])
        x=np.r_[data['p0'][index],data['a_max'][index]]
    s=OptimalControlSolver(1.,float(x[-1]),alpha,'mixed')
    s.set_boundary_conditions([1.,0.,0.],[0.,1.,0.],[-1.5,0.,0.],
        [0.,-1/np.sqrt(1.5),0.],2*np.pi*T_a(.5,1.5),lambda_value=.5)
    def residual(value,method):
        s.a_max=float(value[-1])
        if method=='LSODA': return s.residual_vec(value)
        tr=solve_ivp(s.system_equations,[0.,s.T],np.r_[s.r0,s.v0,value[:6]],
            method='DOP853',rtol=1e-12,atol=1e-14,max_step=s.T/200)
        return s.trajectory_residual(tr)
    def jacobian(relative,method,central):
        base=residual(x,method)
        columns=[]
        for j in range(7):
            h=relative*abs(x[j]) if x[j] else relative
            plus=x.copy();plus[j]+=h
            if central:
                minus=x.copy();minus[j]-=h
                columns.append((residual(plus,method)-residual(minus,method))/(2*h))
            else: columns.append((residual(plus,method)-base)/h)
        return np.column_stack(columns)
    # Compare in-plane columns only: planar coordinates equal zero have special FD fallbacks.
    columns=[0,1,3,4,6]
    ref=jacobian(1e-4,'DOP853',True)[:,columns]
    check=jacobian(1e-5,'DOP853',True)[:,columns]
    rows=[]
    for step in (np.sqrt(np.finfo(float).eps),1e-6,1e-5,1e-4):
        jac=jacobian(step,'LSODA',False)[:,columns]
        rows.append(dict(relative_step=float(step),
            relative_difference_to_dop853=float(np.linalg.norm(jac-ref)/np.linalg.norm(ref))))
    info=dict(alpha=alpha,reference_central_steps=[1e-4,1e-5],
        reference_relative_difference=float(np.linalg.norm(check-ref)/np.linalg.norm(ref)),
        columns=columns,rows=rows)
    target=Path(sys.argv[1])
    target.write_text(json.dumps(info,indent=2),encoding='utf-8')
    print(json.dumps(info,indent=2))


if __name__=='__main__':main()
