"""Запуск из проекта: python passive_boundary_mlp.py
Повторное обучение: python passive_boundary_mlp.py --dataset boundary_run/dataset.csv
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.compose import TransformedTargetRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


TRAIN = (1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6)
TEST = 1.25
TOL = 1e-4


def make_solver():
    from solver import OptimalControlSolver
    from approximations import T_a

    kappa, lam = 1.5, 0.5
    solver = OptimalControlSolver(mu=1., a_max=0.01, alpha=0., func_type="mixed")
    solver.set_boundary_conditions(
        [1, 0, 0], [0, 1, 0], [-kappa, 0, 0], [0, -1 / np.sqrt(kappa), 0],
        2 * np.pi * T_a(lam, kappa), lambda_value=lam)
    return solver


def minimum_q(trajectory):
    # Проверяем концы интервала и уточняем внутренние минимумы плотного решения.
    t = np.unique(np.r_[trajectory.t, np.linspace(0, trajectory.t[-1], 2001)])
    q = np.linalg.norm(trajectory.sol(t)[9:12], axis=0)
    best = q.min()
    candidates = np.flatnonzero((q[1:-1] <= q[:-2]) & (q[1:-1] < q[2:])) + 1
    for i in candidates:
        result = minimize_scalar(lambda x: np.linalg.norm(trajectory.sol(x)[9:12]),
                                 bounds=(t[i-1], t[i+1]), method="bounded")
        if result.success:
            best = min(best, result.fun)
    return best


def collect(out):
    start = make_solver()
    start.free_a_max = True
    result = start.solve(None, tol=1e-6)
    if not result.success:
        raise RuntimeError(f"Квадратичный старт: {result.message}")
    rows = []
    for rho in (*TRAIN, TEST):
        solver = make_solver()
        solver.free_a_max = False
        solver.a_max = rho * start.a_max
        solutions = solver.continuation_method(
            alpha_start=0., alpha_end=1., p0_initial=start.p0.copy(),
            tol=TOL, h_init=0.025, h_max=0.025, h_min=0.0001)
        count = 0
        for item in solutions:
            alpha = item["alpha"]
            solver.alpha = alpha
            trajectory = solver.integrate_trajectory(item["p0"], dense_output=True)
            error = np.linalg.norm(solver.trajectory_residual(trajectory))
            if not np.isfinite(error) or error > TOL:
                continue
            q_min = minimum_q(trajectory)
            rows.append(dict(rho=rho, alpha=alpha, A=solver.a_max, q_min=q_min,
                             margin=q_min-alpha, residual=error,
                             split="test" if rho == TEST else "train",
                             **{f"p0_{i}": v for i, v in enumerate(item["p0"][:6])}))
            count += 1
        print(f"rho={rho:g}, {solver.continuation_status}, принято в датасет: {count}",
              flush=True)
        if rows:
            pd.DataFrame(rows).to_csv(out / "dataset.csv", index=False)
    return pd.DataFrame(rows)


def train_plot(data, out, show=False):
    if data.empty:
        raise ValueError("Нет принятых решений.")
    train = data[data.split == "train"]
    test = data[data.split == "test"].sort_values("alpha")
    if (train.rho.nunique() < 2 or test.rho.nunique() != 1 or len(test) < 2
            or set(train.rho) & set(test.rho)):
        raise ValueError("Нужны разные обучающие rho и отдельная тестовая серия.")
    if not (train.margin.min() < 0 < train.margin.max()):
        print("В обучающих данных нет обоих знаков. Граница не подтверждена.")
    features = ["rho", "alpha"]
    model = TransformedTargetRegressor(
        regressor=make_pipeline(StandardScaler(), MLPRegressor(
            hidden_layer_sizes=(16, 16), activation="tanh", solver="lbfgs",
            alpha=1e-3, max_iter=3000, random_state=42)),
        transformer=StandardScaler())
    model.fit(train[features], train.margin)
    predicted = model.predict(test[features])
    mae = np.mean(np.abs(predicted - test.margin))
    print(f"Тест: {len(test)} решений, MAE запаса = {mae:.5g}")
    rho = test.rho.iloc[0]
    grid = pd.DataFrame({"alpha": np.linspace(test.alpha.min(), test.alpha.max(), 501),
                         "rho": rho})
    # Сравниваем только в рассчитанных точках: ширина промежутка между
    # соседними alpha не означает, что режим внутри него известен.
    from matplotlib.lines import Line2D
    actual_sign = np.sign(test.margin.to_numpy())
    model_sign = np.sign(predicted)
    colors = {-1: "#157d73", 0: "#efb84c", 1: "#b9c9de"}
    labels = {-1: "Есть пассивная дуга", 0: "На пороге", 1: "Без пассивной дуги"}
    overview, panel = plt.subplots(figsize=(10, 3.8), constrained_layout=True)
    for level, signs in [(1, actual_sign), (0, model_sign)]:
        panel.scatter(test.alpha, np.full(len(test), level), marker="s", s=82,
                      c=[colors[v] for v in signs], edgecolors="white", linewidths=.5)
    wrong = actual_sign != model_sign
    panel.scatter(test.alpha[wrong], np.zeros(wrong.sum()), marker="x",
                  color="#9b2144", s=45, linewidths=1.6)
    panel.set(yticks=[0, 1], yticklabels=["Нейросеть", "Решатель"], ylim=(-.55, 1.65),
              xlabel="Вес функционала α   ·   каждый столбец — отдельный перелёт",
              title=f"Появляется ли пассивная дуга?     ρ = {rho:g}     |     Совпадений: {(~wrong).sum()}/{len(test)}")
    legend = [Line2D([], [], marker="s", linestyle="", color=colors[k], label=labels[k])
              for k in (1, -1, 0) if k in actual_sign or k in model_sign]
    legend.append(Line2D([], [], marker="x", linestyle="", color="#9b2144", label="Расхождение"))
    panel.legend(handles=legend, loc="upper center", bbox_to_anchor=(.5, -.32),
                 ncol=len(legend), frameon=False)
    panel.spines[["top", "right", "left"]].set_visible(False)
    overview.savefig(out / "test_regimes.png", dpi=180)
    fig, ax = plt.subplots(figsize=(8, 4.7))
    ax.axhline(0, color="0.4", lw=1, ls="--", label="Порог выключения")
    ax.plot(grid.alpha, model.predict(grid[features]), label="MLP")
    ax.scatter(test.alpha, test.margin, s=22, color="black", label="Решатель", zorder=3)
    ax.set(xlabel="Вес функционала α", ylabel="min ‖pᵥ(t)‖ − α",
           title=f"Отложенная серия ρ = {rho:g}     |     MAE = {mae:.3g}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "test_rho.png", dpi=180)
    test.assign(predicted_margin=predicted).to_csv(out / "test_predictions.csv", index=False)
    if show:
        plt.show()
    plt.close(fig)
    plt.close(overview)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("boundary_run"))
    parser.add_argument("--dataset", type=Path, help="Повторно использовать готовый CSV")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.dataset:
        data = pd.read_csv(args.dataset)
    else:
        if (args.out / "dataset.csv").exists():
            parser.error("Датасет уже существует. Используй --dataset или новый --out.")
        data = collect(args.out)
    train_plot(data, args.out, args.show)


if __name__ == "__main__":
    main()
