"""Карта режима «без пассивных участков» по (kappa, угловая дальность).

Для каждой пары решает квадратичную задачу (alpha=0), берёт q_min=min|p_v(t)|:
пассивной дуги нет, пока alpha < q_min (приближённо alpha*_C ~ q_min), а также A_Q (пик тяги).
    python map_no_passive.py --out map_np
"""
import argparse, itertools, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from branching_scan import make_solver, quadratic_start, analyse

ap = argparse.ArgumentParser()
ap.add_argument('--kappas', type=float, nargs='+', default=[1.1, 1.2, 1.35, 1.5, 1.7, 1.9])
ap.add_argument('--lams', type=float, nargs='+', default=[0.4, 0.5, 0.6, 0.75, 0.9, 1.0])
ap.add_argument('--out', type=Path, default=Path('map_np'))
a = ap.parse_args()
a.out.mkdir(exist_ok=True)
rows = []
for k, l in itertools.product(a.kappas, a.lams):
    try:
        s = make_solver(k, l)
        AQ, p0 = quadratic_start(s)
        r = analyse(s, p0, 0.0, AQ * 1.1)
        rows.append(dict(kappa=k, lam=l, T=s.T, A_Q=AQ, q_min=r['q_min'], q_max=r['q_max'], status='ok'))
    except Exception as e:
        rows.append(dict(kappa=k, lam=l, T=np.nan, A_Q=np.nan, q_min=np.nan, q_max=np.nan, status=str(e)[:60]))
    print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(a.out / 'map.csv', index=False)
df = pd.DataFrame(rows)
fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, col, ttl in zip(axs, ['q_min', 'A_Q'], ['q_min ~ alpha*_C (граница режима без C)', 'A_Q (пик тяги)']):
    P = df.pivot(index='kappa', columns='lam', values=col)
    im = ax.imshow(P.values, origin='lower', aspect='auto', cmap='viridis')
    ax.set_xticks(range(len(P.columns))); ax.set_xticklabels(P.columns)
    ax.set_yticks(range(len(P.index))); ax.set_yticklabels(P.index)
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            v = P.values[i, j]
            ax.text(j, i, 'нет' if np.isnan(v) else f'{v:.3g}', ha='center', va='center', color='w', fontsize=8)
    ax.set_xlabel('угловая дальность, обороты'); ax.set_ylabel('kappa'); ax.set_title(ttl, fontsize=10)
    fig.colorbar(im, ax=ax)
fig.tight_layout(); fig.savefig(a.out / 'map_no_passive.png', dpi=150)
