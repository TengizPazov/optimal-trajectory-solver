"""Карта режима «без пассивных участков» на мелкой двумерной сетке (kappa, угловая дальность).

Для каждой точки решается квадратичная задача (alpha=0) и считается q_min = min_t |p_v(t)|:
пассивной дуги нет, пока alpha < q_min (приближённо alpha*_C ~ q_min). Также пишутся A_Q, T.
Считает параллельно и ДОПИСЫВАЕТ в map.csv (можно прервать и запустить снова).

    python map_no_passive.py --out map_fine                       # сетка по умолчанию 19x25
    python map_no_passive.py --kmin 1.5 --kmax 2.0 --nk 21 --lmin 0.6 --lmax 0.9 --nl 31 --out zoom
    python map_no_passive.py --procs 4 --out map_fine
Затем:  python plot_no_passive_region.py map_fine/map.csv
"""
import argparse
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd


def one(args):
    k, l = args
    from branching_scan import make_solver, quadratic_start, analyse
    try:
        s = make_solver(k, l)
        AQ, p0 = quadratic_start(s)
        r = analyse(s, p0, 0.0, AQ * 1.1, n=20000)
        return dict(kappa=k, lam=l, T=float(s.T), A_Q=AQ, q_min=r['q_min'], q_max=r['q_max'],
                    word=r['word'], status='ok')
    except Exception as e:
        return dict(kappa=k, lam=l, T=np.nan, A_Q=np.nan, q_min=np.nan, q_max=np.nan, word='',
                    status=str(e)[:80].replace(',', ';').replace('\n', ' '))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--kmin', type=float, default=1.1); ap.add_argument('--kmax', type=float, default=2.0)
    ap.add_argument('--nk', type=int, default=19)
    ap.add_argument('--lmin', type=float, default=0.4); ap.add_argument('--lmax', type=float, default=1.0)
    ap.add_argument('--nl', type=int, default=25)
    ap.add_argument('--procs', type=int, default=2)
    ap.add_argument('--out', type=Path, default=Path('map_fine'))
    a = ap.parse_args()
    a.out.mkdir(exist_ok=True)
    f = a.out / 'map.csv'
    done = pd.read_csv(f) if f.exists() else pd.DataFrame(columns=['kappa', 'lam'])
    have = {(round(r.kappa, 6), round(r.lam, 6)) for r in done.itertuples()}
    todo = [(round(k, 6), round(l, 6)) for k in np.linspace(a.kmin, a.kmax, a.nk)
            for l in np.linspace(a.lmin, a.lmax, a.nl) if (round(k, 6), round(l, 6)) not in have]
    print(f'к расчёту {len(todo)} точек (уже есть {len(have)})', flush=True)
    rows = done.to_dict('records')
    with Pool(a.procs) as pool:
        for i, r in enumerate(pool.imap_unordered(one, todo), 1):
            rows.append(r)
            if i % 10 == 0 or i == len(todo):
                pd.DataFrame(rows).to_csv(f, index=False)
                print(f'{i}/{len(todo)}', flush=True)
    pd.DataFrame(rows).to_csv(f, index=False)


if __name__ == '__main__':
    main()
