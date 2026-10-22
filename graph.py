"""Построить график шага из CSV выбранного запуска."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('datafile', type=Path, help='step_vs_alpha_data.csv нужного запуска')
    parser.add_argument('--output', type=Path, help='По умолчанию рядом с CSV')
    args = parser.parse_args(argv)
    df = pd.read_csv(args.datafile)
    ok = df['success'] == 1
    plt.figure(figsize=(10, 6))
    plt.plot(df.loc[ok, 'alpha'], df.loc[ok, 'h'], '-o', color='blue',
             label='Успешные шаги (огибающая)', linewidth=2, markersize=7)
    plt.scatter(df.loc[~ok, 'alpha'], df.loc[~ok, 'h'], color='red', s=60,
                label='Неудачные шаги')
    plt.yscale('log')
    plt.xlabel('α', fontsize=14, fontweight='bold')
    plt.ylabel('Шаг h', fontsize=14, fontweight='bold')
    plt.title('Зависимость шага h от α\n(огибающая + ветки неудачных шагов)',
              fontsize=16, fontweight='bold')
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()
    output = args.output or args.datafile.parent / 'h_vs_alpha_plot.png'
    plt.savefig(output, dpi=200)
    plt.close()
    print(f'График сохранён: {output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
