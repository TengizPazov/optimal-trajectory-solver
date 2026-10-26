"""Прощупать «дно»: для каждой дальности найти kappa*, где q_min(kappa) минимально, и значение там.
Стартует от сетки map.csv (локальный минимум по kappa) и уточняет bounded-минимизацией.
    python valley_track.py zoom_v2/map.csv --out zoom_v2/valley.csv [--kmin 1.2 --kmax 2.0] [--procs 2]
Результат: valley.csv (lam, kappa_star, q_min_star, A_Q) и valley.png.
Если q_min_star ~ 1e-5 и ниже, p_v(t) проходит через ноль (предел точности решения ~1e-5..1e-6).
"""
import argparse
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt


def f(args):
    lam, lo, hi = args
    from scipy.optimize import minimize_scalar
    from branching_scan import make_solver, quadratic_start, analyse
    cache = {}

    def q(k):
        try:
            s = make_solver(k, lam); AQ, p0 = quadratic_start(s)
            r = analyse(s, p0, 0.0, AQ * 1.1, n=20000)
            cache[k] = (r['q_min'], AQ); return r['q_min']
        except Exception:
            return 1.0
    res = minimize_scalar(q, bounds=(lo, hi), method='bounded', options=dict(xatol=2e-4, maxiter=40))
    k = float(res.x); qm, AQ = cache.get(k, (float(res.fun), np.nan))
    return dict(lam=lam, kappa_star=k, q_min_star=qm, A_Q=AQ, evals=len(cache))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('csv', type=Path); ap.add_argument('--out', type=Path, default=None)
    ap.add_argument('--kmin', type=float, default=None); ap.add_argument('--kmax', type=float, default=None)
    ap.add_argument('--procs', type=int, default=2)
    a = ap.parse_args()
    d = pd.read_csv(a.csv).dropna(subset=['q_min'])
    jobs = []
    for lam, g in d.groupby('lam'):
        g = g.sort_values('kappa')
        if a.kmin is not None: g = g[g.kappa >= a.kmin]
        if a.kmax is not None: g = g[g.kappa <= a.kmax]
        if len(g) < 3: continue
        i = int(np.argmin(g.q_min.values)); ks = g.kappa.values
        lo, hi = ks[max(i - 1, 0)], ks[min(i + 1, len(ks) - 1)]
        jobs.append((float(lam), float(lo), float(hi)))
    with Pool(a.procs) as p:
        rows = []
        for r in p.imap(f, jobs):
            rows.append(r); print(r, flush=True)
    v = pd.DataFrame(rows); out = a.out or a.csv.parent / 'valley.csv'
    v.to_csv(out, index=False)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    axs[0].plot(v.lam, v.kappa_star, 'o-'); axs[0].set_xlabel('угловая дальность'); axs[0].set_ylabel('kappa* (дно долины)'); axs[0].grid(alpha=.3)
    axs[1].semilogy(v.lam, np.clip(v.q_min_star, 1e-8, None), 'o-'); axs[1].axhline(1e-5, color='r', ls='--', lw=.8)
    axs[1].set_xlabel('угловая дальность'); axs[1].set_ylabel('min q_min по kappa'); axs[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out.with_suffix('.png'), dpi=150)
