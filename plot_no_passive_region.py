"""Где НЕТ пассивных участков: область под кривой alpha < q_min(kappa, дальность).
    python plot_no_passive_region.py map_np/map.csv
Рисует (1) alpha vs дальность для каждого kappa, (2) alpha vs kappa для каждой дальности.
Закрашено — режим без пассивной дуги (слово I или S-I-S), выше кривой пассивная дуга есть.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

p = Path(sys.argv[1] if len(sys.argv) > 1 else 'map_np/map.csv')
d = pd.read_csv(p).dropna(subset=['q_min'])
fig, axs = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
for ax, key, other, xl in [(axs[0], 'kappa', 'lam', 'угловая дальность, обороты'),
                           (axs[1], 'lam', 'kappa', 'kappa')]:
    vals = sorted(d[key].unique())
    cm = plt.cm.viridis(np.linspace(0, 0.9, len(vals)))
    for c, v in zip(cm, vals):
        g = d[d[key] == v].sort_values(other)
        ax.plot(g[other], g.q_min, 'o-', color=c, label=f'{key}={v:g}')
        ax.fill_between(g[other], 0, g.q_min, color=c, alpha=0.12)
    ax.set_xlabel(xl); ax.set_yscale('symlog', linthresh=1e-4)
    ax.set_ylim(0, 0.02); ax.grid(alpha=0.3); ax.legend(fontsize=7, ncol=2)
axs[0].set_ylabel('alpha')
fig.suptitle('Область под кривой — пассивных участков нет; выше кривой есть (кривая: alpha*_C ≈ min|p_v|)', fontsize=10)
fig.tight_layout(); out = p.parent / 'no_passive_region.png'
fig.savefig(out, dpi=150); print(out)
