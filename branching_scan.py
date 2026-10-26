"""Карта ветвления: слово дуг (C/I/S) и число активных участков вдоль продолжения по alpha
для нескольких значений тяги A = rho * A_Q  (A_Q — максимум тяги квадратичного решения).

Использует ваш OptimalControlSolver с ФИКСИРОВАННЫМ a_max (free_a_max=False).

    python branching_scan.py --rhos 0.8 1.0 1.1 1.3 1.6 --out scan_kappa1.5
    python branching_scan.py --rhos 1.1 --alpha-end 0.5 --out scan_test --no-refine

Результаты в папке --out:
    solutions.csv   каждое принятое решение: rho, alpha, слово, n_act, моменты переключений, q_min/q_max, запасы, Q, L, J, невязка
    events.csv      смены слова дуг: между какими alpha, уточнённое alpha*, слово до/после
    steps_rho*.csv  шаги продолжения (alpha, h, success) для каждого rho
    phase_diagram.png  карта: по x alpha, по y rho, цвет = слово дуг
    summary.json    статус каждого rho
"""
import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.optimize import brentq

_trapz = getattr(np, 'trapezoid', None) or np.trapz   # в numpy 2.x trapz удалён

from approximations import T_a
from solver import OptimalControlSolver

logging.getLogger('solver').setLevel(logging.WARNING)


# ---------------------------------------------------------------- задача
def make_solver(kappa, lam, a_ang_time=None):
    s = OptimalControlSolver(mu=1.0, a_max=0.01, alpha=0.0, func_type='mixed')
    T = 2 * np.pi * T_a(lam, kappa)
    s.set_boundary_conditions([1, 0, 0], [0, 1, 0], [-kappa, 0, 0],
                              [0, -1 / np.sqrt(kappa), 0], T, lambda_value=lam)
    return s


def quadratic_start(s, tol=1e-6):
    """Неограниченное квадратичное решение: A_Q (максимум тяги) и начальные сопряжённые."""
    s.free_a_max = True
    s.alpha = 0.0
    res = s.solve(None, tol=tol)
    if not res.success:
        raise RuntimeError('не удалось получить квадратичное решение: ' + str(res.message))
    return float(s.a_max), s.p0.copy()


# ---------------------------------------------------------------- классификация дуг
def thresholds(alpha, A):
    """Пороги по q = |p_v|: ниже lo — пассивная дуга C, выше hi — насыщенная S."""
    if alpha >= 1 - 1e-12:
        return 1.0, 1.0
    return alpha, alpha + (1 - alpha) * A


def analyse(s, p0, alpha, A, n=6000):
    """Интегрирует решение (плотный вывод) и возвращает слово дуг, переключения, Q, L, J, запасы."""
    s.alpha, s.a_max = alpha, A
    sol = s.integrate_trajectory(p0, dense_output=True)
    T = s.T
    tt = np.linspace(0.0, T, n)
    Y = sol.sol(tt)
    q = np.linalg.norm(Y[9:12], axis=0)
    lo, hi = thresholds(alpha, A)

    def label(qv):
        return 0 if qv <= lo else (2 if qv >= hi else 1)       # 0=C 1=I 2=S

    lab = np.array([label(v) for v in q])
    # моменты переключений: корни g(t)=q(t)-порог внутри интервалов, где метка сменилась
    qf = lambda t: float(np.linalg.norm(sol.sol(t)[9:12]))
    word, sw = [lab[0]], []
    for i in range(1, n):
        if lab[i] != lab[i - 1]:
            thr = (lo if min(lab[i], lab[i - 1]) == 0 else hi) if lo != hi else lo
            if lo != hi and {lab[i], lab[i - 1]} == {0, 2}:    # перескок через I за один шаг сетки
                thr = lo
            try:
                ts = brentq(lambda t: qf(t) - thr, tt[i - 1], tt[i], xtol=1e-12)
            except ValueError:
                ts = 0.5 * (tt[i - 1] + tt[i])
            sw.append(ts)
            word.append(lab[i])
    names = 'CIS'
    word_s = '-'.join(names[w] for w in word)
    active = [w != 0 for w in word]
    n_act = int(active[0]) + sum(1 for k in range(1, len(active)) if active[k] and not active[k - 1])
    # тяга и функционалы
    b = np.where(q <= lo, 0.0, np.where(q >= hi, A, (q - alpha) / max(1 - alpha, 1e-14)))
    b = np.minimum(b, A)
    Q = float(_trapz(0.5 * b ** 2, tt))
    L = float(_trapz(b, tt))
    rf = np.hstack((s.rf - Y[0:3, -1], s.vf - Y[3:6, -1]))
    return dict(word=word_s, n_act=n_act, switches=sw, q_min=float(q.min()), q_max=float(q.max()),
                gC_margin=float(np.min(np.abs(q - lo))), gS_margin=float(np.min(np.abs(q - hi))),
                Q=Q, L=L, J=(1 - alpha) * Q + alpha * L, residual=float(np.linalg.norm(rf)),
                shortest_arc=float(np.min(np.diff([0.0] + sw + [T]))))


# ---------------------------------------------------------------- старт при данном A
def start_at_A(s, A_Q, z_Q, rho, tol):
    """Решение при alpha=0 и A=rho*A_Q (для rho<1 — продолжением по A вниз от rho=1)."""
    s.free_a_max = False
    s.alpha = 0.0
    z = z_Q.copy()
    path = [1.0] if rho >= 1 else list(np.round(np.arange(1.0, rho - 1e-9, -0.05), 10)) + [rho]
    for r in path:
        s.a_max = r * A_Q
        res = s.solve(z, tol=tol)
        if not res.success:
            raise RuntimeError(f'нет решения при alpha=0, rho={r:.3f}')
        z = s.p0.copy()
    s.a_max = rho * A_Q
    return z


# ---------------------------------------------------------------- уточнение события
def refine_event(s, lo_rec, hi_rec, A, tol, iters):
    """Бисекция по alpha между двумя решениями с разным словом; каждое приближение —
    коррекция из ближайшего известного решения (слева).
    Возвращает (a_lo, a_hi, слово_справа, z_справа, n_act_справа): первая найденная смена слова."""
    a_lo, a_hi = lo_rec['alpha'], hi_rec['alpha']
    z_lo, w_lo = lo_rec['p0'].copy(), lo_rec['word']
    w_hi, z_hi, n_hi = hi_rec['word'], hi_rec['p0'].copy(), hi_rec['n_act']
    s.free_a_max = False
    for _ in range(iters):
        mid = 0.5 * (a_lo + a_hi)
        s.alpha, s.a_max = mid, A
        # старт из левого решения; если не вышло — из правого, затем с ослабленным допуском
        ok = False
        for guess, tl in ((z_lo, tol), (z_hi, tol), (z_lo, 10 * tol), (z_hi, 10 * tol)):
            if s.solve(guess, tol=tl).success:
                ok = True
                break
        if not ok:
            break
        info = analyse(s, s.p0, mid, A, n=3000)
        if info['word'] == w_lo:
            a_lo, z_lo = mid, s.p0.copy()
        else:
            a_hi, w_hi, z_hi, n_hi = mid, info['word'], s.p0.copy(), info['n_act']
    return a_lo, a_hi, w_hi, z_hi, n_hi


# ---------------------------------------------------------------- основной сценарий
def run_rho(s, A_Q, z_Q, rho, args, out, rows, events, summary):
    A = rho * A_Q
    t0 = time.time()
    st = dict(rho=rho, A=A, status='failed')
    try:
        z = start_at_A(s, A_Q, z_Q, rho, args.tol)
        s.free_a_max = False
        s.a_max = A
        res = s.continuation_method(alpha_start=0.0, alpha_end=args.alpha_end, tol=args.tol,
                                    h_init=args.h_init, h_min=args.h_min, h_max=args.h_max,
                                    p0_initial=z)
    except RuntimeError as e:
        st['error'] = str(e)
        summary.append(st)
        print(f'rho={rho}: {e}', flush=True)
        return
    pd.DataFrame(s.step_log, columns=['alpha', 'h', 'success']).assign(
        success=lambda d: d.success.astype(int)).to_csv(out / f'steps_rho{rho:g}.csv', index=False)
    recs = []
    for r in res:
        info = analyse(s, r['p0'], r['alpha'], A)
        recs.append(dict(rho=rho, A=A, alpha=r['alpha'], p0=r['p0'].copy(), **info))
    for k in range(1, len(recs)):
        lo, hi = recs[k - 1], recs[k]
        # между двумя соседними решениями может лежать несколько событий: уточняем по очереди
        for _ in range(6):
            if lo['word'] == hi['word']:
                break
            ev = dict(rho=rho, alpha_lo=lo['alpha'], alpha_hi=hi['alpha'],
                      word_before=lo['word'], word_after=hi['word'],
                      n_act_before=lo['n_act'], n_act_after=hi['n_act'])
            if args.refine <= 0:
                ev['alpha_star'] = 0.5 * (ev['alpha_lo'] + ev['alpha_hi'])
                events.append(ev)
                break
            a_lo, a_hi, w_after, z_hi, n_hi = refine_event(s, lo, hi, A, args.tol, args.refine)
            ev.update(alpha_lo=a_lo, alpha_hi=a_hi, alpha_star=0.5 * (a_lo + a_hi),
                      word_after=w_after, n_act_after=n_hi,
                      bracket_width=a_hi - a_lo, refined=bool(a_hi - a_lo < 0.1 * (hi['alpha'] - lo['alpha'])))
            events.append(ev)
            if a_hi >= hi['alpha'] - 1e-15:      # бисекция не сузила интервал — дальше не продвинуться
                break
            lo = dict(alpha=a_hi, p0=z_hi, word=w_after, n_act=n_hi)

    for r in recs:
        rows.append({k: v for k, v in r.items() if k != 'p0'} |
                    dict(switches=';'.join(f'{t:.6f}' for t in r['switches']),
                         **{f'p0_{i}': float(v) for i, v in enumerate(r['p0'])}))
    st.update(status=s.continuation_status['state'], reached_alpha=s.continuation_status['reached_alpha'],
              solutions=len(recs), attempts=len(s.step_log),
              failed_attempts=int(sum(not x[2] for x in s.step_log)),
              seconds=time.time() - t0)
    summary.append(st)
    print(f"rho={rho}: {st['status']}, alpha до {st['reached_alpha']}, решений {len(recs)}, "
          f"отказов {st['failed_attempts']}, {st['seconds']:.0f} с", flush=True)


def phase_diagram(df, out):
    words = sorted(df.word.unique(), key=lambda w: (w.count('-'), w))
    cmap = plt.get_cmap('tab20')
    col = {w: cmap(i % 20) for i, w in enumerate(words)}
    rhos = sorted(df.rho.unique())
    fig, ax = plt.subplots(figsize=(11, 1.1 * len(rhos) + 2.2))
    for y, rho in enumerate(rhos):
        d = df[df.rho == rho].sort_values('alpha')
        x = d.alpha.values
        edges = np.r_[x[0], 0.5 * (x[1:] + x[:-1]), x[-1]] if len(x) > 1 else np.r_[x[0] - .01, x[0] + .01]
        for i, w in enumerate(d.word.values):
            ax.fill_betweenx([y - .4, y + .4], edges[i], edges[i + 1], color=col[w], lw=0)
        ax.text(x[-1] + 0.005, y, f'{d.n_act.iloc[-1]} акт.', va='center', fontsize=8)
    ax.set_yticks(range(len(rhos)))
    ax.set_yticklabels([f'{r:g}' for r in rhos])
    ax.set_xlabel('alpha')
    ax.set_ylabel('rho = A / A_Q')
    ax.set_xlim(0, df.alpha.max() * 1.08)
    ax.legend(handles=[Patch(color=col[w], label=w) for w in words], title='слово дуг',
              bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
    ax.set_title('Структура дуг: C — пассивная, I — промежуточная, S — насыщенная')
    fig.tight_layout()
    fig.savefig(out / 'phase_diagram.png', dpi=170)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--rhos', type=float, nargs='+', default=[1.0, 1.1, 1.3, 1.6])
    ap.add_argument('--kappa', type=float, default=1.5)
    ap.add_argument('--lam', type=float, default=0.5, help='угловая дальность (lambda_value), как в run_continuation')
    ap.add_argument('--alpha-end', type=float, default=1.0)
    ap.add_argument('--tol', type=float, default=1e-4, help='допуск невязки (в run_continuation было 1e-2)')
    ap.add_argument('--h-init', type=float, default=0.05)
    ap.add_argument('--h-min', type=float, default=1e-3)
    ap.add_argument('--h-max', type=float, default=0.1)
    ap.add_argument('--refine', type=int, default=6, help='шагов бисекции для уточнения alpha* (0 — не уточнять)')
    ap.add_argument('--no-refine', action='store_true')
    ap.add_argument('--out', type=Path, default=Path('scan_out'))
    args = ap.parse_args(argv)
    if args.no_refine:
        args.refine = 0
    args.out.mkdir(parents=True, exist_ok=True)

    s = make_solver(args.kappa, args.lam)
    A_Q, z_Q = quadratic_start(s)
    print(f'A_Q = {A_Q:.10f}', flush=True)
    rows, events, summary = [], [], []
    for rho in args.rhos:
        run_rho(s, A_Q, z_Q, rho, args, args.out, rows, events, summary)
        if rows:   # промежуточное сохранение после каждого rho
            df = pd.DataFrame(rows)
            df.to_csv(args.out / 'solutions.csv', index=False)
            pd.DataFrame(events).to_csv(args.out / 'events.csv', index=False)
            phase_diagram(df, args.out)
        (args.out / 'summary.json').write_text(
            json.dumps(dict(A_Q=A_Q, kappa=args.kappa, lam=args.lam, runs=summary), ensure_ascii=False, indent=2))
    if events:
        print('\nСобытия (смена слова дуг):')
        print(pd.DataFrame(events).round(5).to_string(index=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
