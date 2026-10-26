"""Где НЕТ пассивных участков. Двумерная карта по (kappa, угловая дальность) + срезы.
    python plot_no_passive_region.py map_fine/map.csv [--alpha 0.002]
Рисует в папке map.csv:
  no_passive_2d.png    цвет = log10 q_min ~ alpha*_C (под ним, при alpha<q_min, пассивной дуги нет);
                       чёрные линии уровня q_min = 1e-3 и 5e-3, красная кривая — глубокий минимум q_min
                       (там p_v проходит через ноль и пассивная дуга рождается при alpha -> 0);
                       серые клетки — решение не найдено.
  no_passive_alpha.png бинарная карта: для заданного --alpha закрашено, где пассивных участков НЕТ.
  no_passive_slices.png срезы alpha*_C(дальность) для разных kappa и alpha*_C(kappa) для разных дальностей.
  no_passive_AQ.png    пик тяги A_Q на той же сетке.
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument('csv', type=Path)
ap.add_argument('--alpha', type=float, nargs='+', default=[0.001, 0.003, 0.006])
a = ap.parse_args()
out = a.csv.parent
d = pd.read_csv(a.csv).drop_duplicates(['kappa', 'lam'])
K = np.sort(d.kappa.unique()); L = np.sort(d.lam.unique())


def grid(col):
    G = np.full((len(K), len(L)), np.nan)
    ik = {round(v, 6): i for i, v in enumerate(K)}; il = {round(v, 6): i for i, v in enumerate(L)}
    for r in d.itertuples():
        G[ik[round(r.kappa, 6)], il[round(r.lam, 6)]] = getattr(r, col)
    return G


Q = grid('q_min'); AQ = grid('A_Q')
ext = [L[0], L[-1], K[0], K[-1]]
dl = (L[1] - L[0]) / 2 if len(L) > 1 else 0.5; dk = (K[1] - K[0]) / 2 if len(K) > 1 else 0.5
ext = [L[0] - dl, L[-1] + dl, K[0] - dk, K[-1] + dk]
cmap = plt.cm.viridis.copy(); cmap.set_bad('#cccccc')

# ---- 1. 2D карта q_min
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(np.log10(np.clip(Q, 1e-6, None)), origin='lower', extent=ext, aspect='auto',
               cmap=cmap, vmin=-5, vmax=-1.9)
cs = ax.contour(L, K, Q, levels=[1e-3, 5e-3], colors='k', linewidths=0.9)
ax.clabel(cs, fmt=lambda v: f'{v:g}', fontsize=8)
# кривая глубокого минимума: для каждой дальности — kappa с наименьшим q_min, если он мал
for j, l in enumerate(L):
    col = Q[:, j]
    if np.all(np.isnan(col)): continue
    i = np.nanargmin(col)
    if col[i] < 2e-4 and 0 < i < len(K) - 1:
        ax.plot(l, K[i], 'o', color='red', ms=4)
nan = np.isnan(Q)
ax.set_xlabel('угловая дальность, обороты'); ax.set_ylabel('kappa')
ax.set_title('log10 q_min ≈ log10 alpha*_C: при alpha < q_min пассивных участков нет\n'
             '(красные точки — q_min < 2e-4, серое — нет решения)', fontsize=10)
fig.colorbar(im, ax=ax, label='log10 q_min')
fig.tight_layout(); fig.savefig(out / 'no_passive_2d.png', dpi=160); plt.close(fig)

# ---- 2. бинарные карты для заданных alpha
fig, axs = plt.subplots(1, len(a.alpha), figsize=(4.6 * len(a.alpha), 4.4), squeeze=False)
for ax, al in zip(axs[0], a.alpha):
    B = np.where(np.isnan(Q), np.nan, (Q > al).astype(float))
    ax.imshow(B, origin='lower', extent=ext, aspect='auto',
              cmap=matplotlib.colors.ListedColormap(['#e8b4b4', '#9ed6a0']).with_extremes(bad='#cccccc'), vmin=0, vmax=1)
    ax.set_title(f'alpha = {al:g}\nзелёное — пассивных участков НЕТ', fontsize=9)
    ax.set_xlabel('угловая дальность'); ax.set_ylabel('kappa')
fig.tight_layout(); fig.savefig(out / 'no_passive_alpha.png', dpi=150); plt.close(fig)

# ---- 3. срезы
fig, axs = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
for ax, vals, key, xs, xl in [(axs[0], K[::max(1, len(K) // 6)], 'kappa', L, 'угловая дальность'),
                              (axs[1], L[::max(1, len(L) // 6)], 'lam', K, 'kappa')]:
    cm = plt.cm.viridis(np.linspace(0, 0.9, len(vals)))
    for c, v in zip(cm, vals):
        y = Q[np.argmin(abs(K - v)), :] if key == 'kappa' else Q[:, np.argmin(abs(L - v))]
        ax.plot(xs, y, '-o', ms=3, color=c, label=f'{key}={v:.3g}'); ax.fill_between(xs, 0, y, color=c, alpha=0.08)
    ax.set_yscale('symlog', linthresh=1e-4); ax.set_ylim(0, 0.02); ax.grid(alpha=0.3)
    ax.set_xlabel(xl); ax.legend(fontsize=7, ncol=2)
axs[0].set_ylabel('alpha*_C ≈ q_min  (под кривой пассивных участков нет)')
fig.tight_layout(); fig.savefig(out / 'no_passive_slices.png', dpi=150); plt.close(fig)

# ---- 4. A_Q
fig, ax = plt.subplots(figsize=(7, 5.4))
im = ax.imshow(AQ, origin='lower', extent=ext, aspect='auto', cmap=cmap)
ax.set_xlabel('угловая дальность'); ax.set_ylabel('kappa'); ax.set_title('A_Q — пик тяги квадратичного решения')
fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(out / 'no_passive_AQ.png', dpi=150); plt.close(fig)
print('готово:', *(p.name for p in sorted(out.glob('no_passive_*.png'))))
print(f'точек: {len(d)}, без решения: {int(np.isnan(Q).sum())}')
