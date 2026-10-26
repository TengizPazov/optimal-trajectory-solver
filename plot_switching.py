"""Наглядные зоны переключения управления по результатам branching_scan.py.

    python plot_switching.py scan_out                      # все rho, автоматический выбор alpha
    python plot_switching.py scan_out --rho 1.1 --alphas 0 0.01 0.05 0.2 0.5 0.9

Для каждого rho рисует (папка scan_out/switch_plots):
  zones_rho<rho>.png      карта «alpha — время»: зоны C (серая), I (синяя), S (красная);
                          чёрные линии — моменты переключений t_k(alpha); вертикальные
                          пунктиры — структурные события (смена слова дуг) с подписью alpha*.
  q_profiles_rho<rho>.png функция переключения q(t) = |p_v(t)| для выбранных alpha: пороги
                          (q = alpha и q = alpha+(1-alpha)A), цветные зоны дуг, точки переключений.
  trajectories_rho<rho>.png  траектории в плоскости (цвет — тип дуги), там же точки переключений.
Нужны solutions.csv и events.csv из branching_scan.py (в solutions.csv должны быть столбцы p0_*).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from branching_scan import make_solver, thresholds

COL = {'C': '#c9ccd1', 'I': '#4c8fd6', 'S': '#e05353'}


def parse_switches(s):
    if not isinstance(s, str) or not s:
        return []
    return [float(x) for x in s.split(';')]


def intervals(word, switches, T):
    labels = word.split('-')
    bounds = [0.0] + list(switches) + [T]
    return [(labels[i], bounds[i], bounds[i + 1]) for i in range(len(labels))]


# ------------------------------------------------------------------ 1. карта зон
def plot_zones(df, ev, T, out, rho):
    d = df.sort_values('alpha')
    x = d.alpha.values
    if len(x) < 2:
        return
    edges = np.r_[x[0], 0.5 * (x[1:] + x[:-1]), x[-1]]
    fig, ax = plt.subplots(figsize=(11, 5.2))
    for i, (_, r) in enumerate(d.iterrows()):
        for lab, t0, t1 in intervals(r.word, parse_switches(r.switches), T):
            ax.fill_between([edges[i], edges[i + 1]], t0, t1, color=COL[lab], lw=0)
    nsw = d.switches.map(lambda s: len(parse_switches(s))).max()
    for k in range(int(nsw)):
        px = [a for a, s in zip(d.alpha, d.switches) if len(parse_switches(s)) > k]
        py = [parse_switches(s)[k] for s in d.switches if len(parse_switches(s)) > k]
        ax.plot(px, py, 'k-', lw=0.9)
    if ev is not None and len(ev):
        for _, e in ev.iterrows():
            ax.axvline(e.alpha_star, color='k', ls=':', lw=1)
            ax.text(e.alpha_star, T * 1.012, f'{e.alpha_star:.4f}', rotation=90, fontsize=7,
                    ha='center', va='bottom')
    ax.set_xlim(edges[0], edges[-1])
    ax.set_ylim(0, T)
    ax.set_xlabel('alpha')
    ax.set_ylabel('время t')
    ax.legend(handles=[Patch(color=COL['C'], label='C: пассивная (b=0)'),
                       Patch(color=COL['I'], label='I: промежуточная'),
                       Patch(color=COL['S'], label='S: насыщенная (b=A)')],
              loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
    ax.set_title(f'Зоны управления вдоль продолжения, rho={rho:g}   (линии — моменты переключений, пунктир — события)',
                 pad=22, fontsize=10)
    fig.tight_layout()
    fig.savefig(out / f'zones_rho{rho:g}.png', dpi=160)
    plt.close(fig)


# ------------------------------------------------------------------ 2. профили q(t) и 3. траектории
def solve_profile(s, p0, alpha, A, n=4000):
    s.alpha, s.a_max = alpha, A
    sol = s.integrate_trajectory(p0, dense_output=True)
    tt = np.linspace(0, s.T, n)
    Y = sol.sol(tt)
    return tt, Y


def pick_alphas(d, k=6):
    a = np.sort(d.alpha.values)
    return list(a[np.unique(np.linspace(0, len(a) - 1, k).round().astype(int))])


def plot_profiles(df, s, A, out, rho, alphas):
    T = s.T
    nc = min(3, len(alphas))
    nr = int(np.ceil(len(alphas) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(4.8 * nc, 3.4 * nr), squeeze=False)
    fig2, axs2 = plt.subplots(nr, nc, figsize=(4.2 * nc, 4.2 * nr), squeeze=False)
    for ax in list(axs.ravel()) + list(axs2.ravel()):
        ax.set_visible(False)
    for ax, ax2, a in zip(axs.ravel(), axs2.ravel(), alphas):
        r = df.iloc[(df.alpha - a).abs().argmin()]
        p0 = np.array([r[f'p0_{i}'] for i in range(6)])
        tt, Y = solve_profile(s, p0, r.alpha, A)
        q = np.linalg.norm(Y[9:12], axis=0)
        lo, hi = thresholds(r.alpha, A)
        sw = parse_switches(r.switches)
        ax.set_visible(True)
        for lab, t0, t1 in intervals(r.word, sw, T):
            ax.axvspan(t0, t1, color=COL[lab], alpha=0.35, lw=0)
        ax.plot(tt, q, 'k', lw=1.8)
        ax.axhline(lo, color='C1', ls='--', lw=1, label='q = alpha')
        if hi != lo:
            ax.axhline(hi, color='C3', ls='--', lw=1, label='q = alpha+(1-alpha)A')
        for ts in sw:
            ax.plot(ts, float(np.interp(ts, tt, q)), 'o', color='k', ms=5)
        ax.set_title(f'alpha={r.alpha:.4g}   {r.word}', fontsize=9)
        ax.set_xlabel('t')
        ax.set_ylabel('q(t)')
        ax.grid(alpha=0.3)
        # траектория в плоскости
        ax2.set_visible(True)
        th = np.linspace(0, 2 * np.pi, 300)
        ax2.plot(np.cos(th), np.sin(th), '--', color='#999', lw=0.8)
        ax2.plot(s.kappa * np.cos(th), s.kappa * np.sin(th), '-.', color='#999', lw=0.8)
        for lab, t0, t1 in intervals(r.word, sw, T):
            m = (tt >= t0) & (tt <= t1)
            ax2.plot(Y[0, m], Y[1, m], color=COL[lab] if lab != 'C' else '#7d828a', lw=3)
        for ts in sw:
            i = np.searchsorted(tt, ts)
            ax2.plot(Y[0, min(i, len(tt) - 1)], Y[1, min(i, len(tt) - 1)], 'ko', ms=4)
        ax2.plot(*Y[0:2, 0], 'go', ms=7)
        ax2.plot(*Y[0:2, -1], 'ks', ms=7)
        ax2.set_aspect('equal')
        ax2.grid(alpha=0.3)
        ax2.set_title(f'alpha={r.alpha:.4g}   {r.word}', fontsize=9)
    h, l = axs.ravel()[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=2, frameon=False)
    fig.suptitle(f'Функция переключения q(t) и зоны дуг, rho={rho:g} (серая C, синяя I, красная S; точки — переключения)',
                 fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(out / f'q_profiles_rho{rho:g}.png', dpi=150)
    fig2.suptitle(f'Траектории, rho={rho:g} (серая C, синяя I, красная S; ● старт, ■ финиш)', fontsize=10)
    fig2.tight_layout(rect=(0, 0, 1, 0.97))
    fig2.savefig(out / f'trajectories_rho{rho:g}.png', dpi=150)
    plt.close(fig)
    plt.close(fig2)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('scan_dir', type=Path)
    ap.add_argument('--rho', type=float, nargs='+', default=None)
    ap.add_argument('--alphas', type=float, nargs='+', default=None,
                    help='значения alpha для профилей q(t) (берётся ближайшее решение)')
    ap.add_argument('--nprof', type=int, default=6, help='сколько профилей, если --alphas не задан')
    args = ap.parse_args(argv)

    df = pd.read_csv(args.scan_dir / 'solutions.csv')
    ev_path = args.scan_dir / 'events.csv'
    ev = pd.read_csv(ev_path) if ev_path.exists() and ev_path.stat().st_size > 5 else None
    meta = json.loads((args.scan_dir / 'summary.json').read_text())
    s = make_solver(meta['kappa'], meta['lam'])
    out = args.scan_dir / 'switch_plots'
    out.mkdir(exist_ok=True)
    have_p0 = 'p0_0' in df.columns
    for rho in (args.rho or sorted(df.rho.unique())):
        d = df[np.isclose(df.rho, rho)]
        if d.empty:
            print(f'rho={rho}: нет данных')
            continue
        e = ev[np.isclose(ev.rho, rho)] if ev is not None else None
        A = float(d.A.iloc[0])
        plot_zones(d, e, s.T, out, rho)
        if have_p0:
            plot_profiles(d, s, A, out, rho, args.alphas or pick_alphas(d, args.nprof))
        else:
            print('в solutions.csv нет столбцов p0_*: профили q(t) и траектории пропущены '
                  '(перезапустите branching_scan.py новой версией)')
        print(f'rho={rho:g}: графики в {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
