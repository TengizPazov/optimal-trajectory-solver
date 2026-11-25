"""Сбор траекторий, обучение двух MLP и независимый тест.
"""
import argparse
import copy
import hashlib
import json
import logging
import os
import platform
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.spatial.transform import Rotation
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


# Начальный диапазон задачи. Расширять после проверки сходимости решателя.
RANGES = {
    "mu": (0.9, 1.1),
    "radius": (0.95, 1.05),
    "kappa": (1.35, 1.65),
    "lambda_value": (0.46, 0.54),
    "time_factor": (0.95, 1.05),
    "rho": (1.05, 1.6),
    "v0_tangent_factor": (0.98, 1.02),
    "vf_tangent_factor": (0.98, 1.02),
    "v0_radial_factor": (-0.01, 0.01),
    "vf_radial_factor": (-0.01, 0.01),
    "relative_inclination_deg": (-3.0, 3.0),
}
STATE_COLUMNS = [f"{v}_{axis}" for v in ("r0", "v0", "rf", "vf")
                 for axis in ("x", "y", "z")]
FEATURES = ["mu", "T", "A", "alpha", "lambda_value"] + STATE_COLUMNS
P0_COLUMNS = [f"p0_{i}" for i in range(6)]
MODE_NAMES = ["C — выключен", "I — промежуточная тяга", "S — максимальная тяга"]
COLORS = ["#277e77", "#d8a843", "#476eb5"]


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def save_npz(path, **arrays):
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def make_cases(count, seed):
    from approximations import T_a, get_initial_adjoint

    rng = np.random.default_rng(seed)
    order = rng.permutation(count)
    splits = np.full(count, "train", dtype=object)
    held = max(2, int(round(count * 0.15)))
    splits[order[:held]] = "test"
    splits[order[held:2 * held]] = "validation"
    rows = []
    for case_id in range(count):
        c = {name: float(rng.uniform(*bounds)) for name, bounds in RANGES.items()}
        radius, mu, kappa = c["radius"], c["mu"], c["kappa"]
        angle = 2 * np.pi * c["lambda_value"]
        tilt = np.deg2rad(c["relative_inclination_deg"])
        rotation = Rotation.random(random_state=rng).as_matrix()
        radial = np.array([np.cos(angle), np.sin(angle) * np.cos(tilt),
                           np.sin(angle) * np.sin(tilt)])
        tangent = np.array([-np.sin(angle), np.cos(angle) * np.cos(tilt),
                            np.cos(angle) * np.sin(tilt)])
        r0 = rotation @ np.array([radius, 0, 0])
        v0 = rotation @ (np.sqrt(mu / radius) * np.array([
            c["v0_radial_factor"], c["v0_tangent_factor"], 0]))
        rf = rotation @ (radius * kappa * radial)
        vf = rotation @ (np.sqrt(mu / (radius * kappa)) * (
            c["vf_radial_factor"] * radial + c["vf_tangent_factor"] * tangent))
        time_scale = np.sqrt(radius ** 3 / mu)
        c["T"] = float(2 * np.pi * T_a(c["lambda_value"], kappa)
                       * time_scale * c["time_factor"])
        # Масштабирование квадратичного старта: pv имеет масштаб ускорения,
        # pr — ускорения / времени. Небольшие изменения геометрии уточняет solve.
        pr, pv = get_initial_adjoint(c["lambda_value"], kappa)
        acceleration_scale = mu / radius ** 2
        guess = np.r_[rotation @ pr * acceleration_scale / time_scale,
                      rotation @ pv * acceleration_scale]
        c.update(dict(zip(STATE_COLUMNS, np.r_[r0, v0, rf, vf])))
        c.update({f"guess_{j}": float(v) for j, v in enumerate(guess)})
        c.update(case_id=case_id, split=splits[case_id])
        rows.append(c)
    return pd.DataFrame(rows)


def make_solver(case, A=0.1, alpha=0.0):
    from solver import OptimalControlSolver

    solver = OptimalControlSolver(mu=float(case["mu"]), a_max=float(A),
                                  alpha=float(alpha), func_type="mixed")
    states = [np.array([case[f"{v}_{axis}"] for axis in ("x", "y", "z")])
              for v in ("r0", "v0", "rf", "vf")]
    solver.set_boundary_conditions(*states, float(case["T"]),
                                   lambda_value=float(case["lambda_value"]))
    return solver


def collect_case(case, args, path):
    solver = make_solver(case)
    seed = np.array([case[f"guess_{j}"] for j in range(6)])
    initial = solver.solve(seed, tol=args.tol)
    if not initial.success:
        raise RuntimeError(f"Начальное решение отклонено. {initial.message}")
    A_Q = float(solver.a_max)
    A = float(case["rho"] * A_Q)
    solver.free_a_max = False
    solver.a_max = A
    solutions = solver.continuation_method(
        alpha_start=0.0, alpha_end=args.alpha_end, p0_initial=solver.p0.copy(),
        tol=args.tol, h_init=0.001, h_min=0.0001, h_max=args.h_max)
    tau = np.linspace(0, 1, args.time_points)
    records, qs, p0s, errors = [], [], [], []
    for item in solutions:
        solver.alpha = float(item["alpha"])
        trajectory = solver.integrate_trajectory(item["p0"], dense_output=True)
        error = float(np.linalg.norm(solver.trajectory_residual(trajectory)))
        if not np.isfinite(error) or error > args.tol:
            continue
        q = np.linalg.norm(trajectory.sol(tau * solver.T)[9:12], axis=0)
        p0 = np.asarray(item["p0"][:6])
        if not np.all(np.isfinite(q)) or not np.all(np.isfinite(p0)):
            continue
        record = dict(case, A=A, alpha=solver.alpha)
        records.append([record[name] for name in FEATURES])
        qs.append(q)
        p0s.append(p0)
        errors.append(error)
    if not records:
        raise RuntimeError("Нет решений, прошедших повторное интегрирование.")
    save_npz(path, X=np.asarray(records), p0=np.asarray(p0s), q=np.asarray(qs),
             tau=tau, residual=np.asarray(errors), A_Q=np.array(A_Q))
    return {"state": solver.continuation_status["state"],
            "reached_alpha": float(max(row[FEATURES.index("alpha")] for row in records)),
            "accepted": len(records), "A_Q": A_Q, "A": A}


def source_hashes():
    root = Path(__file__).resolve().parent
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ("solver.py", "approximations.py", "solver_settings.py",
                         "solver_profile.json")}


def collect(args):
    config = {"format": 1, "n_cases": args.n_cases, "seed": args.seed,
              "ranges": {k: list(v) for k, v in RANGES.items()}, "tol": args.tol,
              "alpha_end": args.alpha_end, "time_points": args.time_points,
              "h_max": args.h_max, "source_hashes": source_hashes()}
    config_path = args.out / "collection_config.json"
    if config_path.exists():
        if not args.resume:
            raise ValueError("Расчёт уже существует. Добавь --resume или выбери новый --out.")
        if json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise ValueError("Параметры или исходники изменились. Выбери новую папку --out.")
        cases = pd.read_csv(args.out / "cases.csv")
    else:
        cases = make_cases(args.n_cases, args.seed)
        cases.to_csv(args.out / "cases.csv", index=False)
        write_json(config_path, config)
    (args.out / "cases").mkdir(exist_ok=True)
    status_path = args.out / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
    for case in cases.to_dict("records"):
        case_id = int(case["case_id"])
        key = str(case_id)
        path = args.out / "cases" / f"case_{case_id:05d}.npz"
        if path.exists():
            continue
        if key in status and not args.retry_failed:
            continue
        started = time.perf_counter()
        try:
            entry = collect_case(case, args, path)
        except Exception as exc:
            entry = {"state": "failed", "accepted": 0,
                     "error": f"{type(exc).__name__}: {exc}"}
        entry["seconds"] = time.perf_counter() - started
        status[key] = entry
        write_json(status_path, status)
        print(f"Задача {case_id + 1}/{len(cases)} [{case['split']}]  {entry}", flush=True)


def load_dataset(out):
    cases = pd.read_csv(out / "cases.csv")
    parts = {name: [] for name in ("X", "p0", "q", "case_id", "split", "residual")}
    tau = None
    for case in cases.to_dict("records"):
        path = out / "cases" / f"case_{int(case['case_id']):05d}.npz"
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as arrays:
            current_tau = arrays["tau"]
            if tau is not None and not np.array_equal(tau, current_tau):
                raise ValueError("Временные сетки задач различаются.")
            tau = current_tau.copy()
            count = len(arrays["X"])
            for name in ("X", "p0", "q", "residual"):
                parts[name].append(arrays[name].copy())
            parts["case_id"].append(np.full(count, int(case["case_id"])))
            parts["split"].append(np.full(count, case["split"]))
    if not parts["X"]:
        raise ValueError("Нет принятых траекторий. Причины записаны в status.json.")
    data = {name: np.concatenate(values) for name, values in parts.items()}
    data["tau"] = tau
    for split in ("train", "validation", "test"):
        if not np.any(data["split"] == split):
            raise ValueError(f"Нет принятых задач в {split}. Увеличь выборку или проверь status.json.")
    summary = pd.DataFrame(data["X"], columns=FEATURES)
    summary[P0_COLUMNS] = data["p0"]
    summary["case_id"] = data["case_id"]
    summary["split"] = data["split"]
    summary["residual"] = data["residual"]
    summary["sampled_q_min"] = data["q"].min(axis=1)
    summary.to_csv(out / "dataset.csv", index=False)
    return data, cases


def time_features(X, tau):
    return np.column_stack((np.repeat(X, len(tau), axis=0), np.tile(tau, len(X))))


def fit_network(X, y, X_val, y_val, layers, args, name):
    input_scaler = StandardScaler().fit(X)
    target_scaler = StandardScaler().fit(y)
    X_scaled, y_scaled = input_scaler.transform(X), target_scaler.transform(y)
    validation_X = input_scaler.transform(X_val)
    validation_y = target_scaler.transform(y_val)
    model = MLPRegressor(hidden_layer_sizes=layers, activation="tanh", solver="adam",
                         alpha=1e-4, learning_rate_init=1e-3,
                         batch_size=min(256, len(X)), random_state=args.seed)
    target = y_scaled.ravel() if y.shape[1] == 1 else y_scaled
    best_loss, best_model, stale, history = np.inf, None, 0, []
    for epoch in range(1, args.epochs + 1):
        model.partial_fit(X_scaled, target)
        predicted = model.predict(validation_X).reshape(validation_y.shape)
        loss = float(np.mean((predicted - validation_y) ** 2))
        if not np.isfinite(loss):
            raise RuntimeError(f"{name}: нечисловая ошибка обучения.")
        history.append({"epoch": epoch, "train_loss": float(model.loss_), "val_mse": loss})
        if loss < best_loss:
            best_loss, best_model, stale = loss, copy.deepcopy(model), 0
        else:
            stale += 1
        if epoch == 1 or epoch % 20 == 0:
            print(f"{name}: эпоха {epoch}, validation MSE = {loss:.6g}", flush=True)
        if stale >= args.patience:
            break
    pd.DataFrame(history).to_csv(args.out / f"history_{name}.csv", index=False)
    return {"network": best_model, "input_scaler": input_scaler,
            "target_scaler": target_scaler, "layers": layers,
            "best_validation_mse": best_loss}


def predict(bundle, X):
    scaled = bundle["input_scaler"].transform(X)
    output = bundle["network"].predict(scaled)
    output = np.asarray(output).reshape(len(X), -1)
    return bundle["target_scaler"].inverse_transform(output)


def control(q, X):
    alpha = X[:, FEATURES.index("alpha"), None]
    A = X[:, FEATURES.index("A"), None]
    lower = alpha
    upper = alpha + (1 - alpha) * A
    mode = np.where(q <= lower, 0, np.where(q >= upper, 2, 1))
    linear = alpha >= 1 - 1e-12
    mode = np.where(linear, np.where(q > 1, 2, 0), mode)
    fraction = np.clip((q - alpha) / np.maximum((1 - alpha) * A, 1e-14), 0, 1)
    fraction = np.where(linear, (q > 1).astype(float), fraction)
    distance = np.minimum(np.abs(q - lower), np.abs(q - upper))
    return mode, fraction, distance


def regression_metrics(actual, predicted):
    error = predicted - actual
    return {"mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error ** 2)))}


def classification_metrics(actual, predicted):
    actual, predicted = np.asarray(actual).ravel(), np.asarray(predicted).ravel()
    if not len(actual):
        return {"samples": 0}
    report = classification_report(actual, predicted, labels=[0, 1, 2],
                                   target_names=MODE_NAMES, output_dict=True,
                                   zero_division=0)
    supported = np.unique(actual)
    recalls = [report[MODE_NAMES[int(label)]]["recall"] for label in supported]
    return {"samples": int(len(actual)),
            "accuracy": float(accuracy_score(actual, predicted)),
            "macro_f1_three_modes": float(f1_score(actual, predicted, labels=[0, 1, 2],
                                                    average="macro", zero_division=0)),
            "macro_f1_present_modes": float(f1_score(actual, predicted, labels=supported,
                                                      average="macro", zero_division=0)),
            "balanced_accuracy_present_modes": float(np.mean(recalls)),
            "confusion_matrix_C_I_S": confusion_matrix(actual, predicted, labels=[0, 1, 2]).tolist(),
            "per_class": {name: report[name] for name in MODE_NAMES}}


def shooting_checks(data, p0_pred, cases, args):
    test_indices = np.flatnonzero(data["split"] == "test")
    X = data["X"][test_indices]
    ids = data["case_id"][test_indices]
    rows = []
    for case_id in np.unique(ids):
        candidates = np.flatnonzero((ids == case_id) & np.isclose(X[:, 3], 1, atol=1e-12, rtol=0))
        if not len(candidates) or len(rows) >= args.shooting_checks:
            continue
        i = int(candidates[-1])
        case = cases[cases.case_id == case_id].iloc[0].to_dict()
        entry = {"case_id": int(case_id), "alpha": 1.0}
        started = time.perf_counter()
        try:
            solver = make_solver(case, X[i, 2], 1.0)
            solver.free_a_max = False
            raw = solver.integrate_trajectory(p0_pred[i])
            before = float(np.linalg.norm(solver.trajectory_residual(raw)))
            entry["residual_before"] = before if np.isfinite(before) else None
            result = solver.solve(p0_pred[i], tol=args.tol)
            after = float(np.linalg.norm(solver.trajectory_residual(solver.trajectory)))
            entry.update(accepted=bool(result.success and np.isfinite(after) and after <= args.tol),
                         residual_after=after if np.isfinite(after) else None,
                         nfev=int(result.nfev))
        except Exception as exc:
            entry.update(accepted=False, error=f"{type(exc).__name__}: {exc}")
        entry["seconds"] = time.perf_counter() - started
        rows.append(entry)
        print(f"Стрельба из прогноза p0: {entry}", flush=True)
    return {"tested": len(rows), "success_rate":
            float(np.mean([r["accepted"] for r in rows])) if rows else None,
            "cases": rows}


def plot_results(X, actual_q, predicted_q, actual_p0, predicted_p0, tau, ids, out, show):
    actual_mode, actual_b, _ = control(actual_q, X)
    predicted_mode, predicted_b, _ = control(predicted_q, X)
    # Выбираем задачу по id и alpha, независимо от величины ошибки модели.
    case_id = int(np.min(ids))
    indices = np.flatnonzero(ids == case_id)
    chosen = list(dict.fromkeys(int(indices[np.argmin(np.abs(X[indices, 3] - a))])
                               for a in (0.0, 0.5, 1.0)))
    fig, axes = plt.subplots(len(chosen), 2, figsize=(12, 3.1 * len(chosen)),
                             squeeze=False, constrained_layout=True)
    for row, i in enumerate(chosen):
        ax = axes[row, 0]
        drawstyle = "steps-mid" if X[i, 3] >= 1 - 1e-12 else "default"
        ax.plot(tau, actual_b[i], color="black", label="Решатель", lw=2, drawstyle=drawstyle)
        ax.plot(tau, predicted_b[i], color="#d17330", ls="--", label="Нейросеть", lw=1.6,
                drawstyle=drawstyle)
        ax.set(xlabel="Доля времени перелёта t / T", ylabel="Тяга / максимальная тяга",
               ylim=(-0.06, 1.06), title=f"Тестовая задача {case_id}, α = {X[i, 3]:.4g}")
        ax.legend(loc="best")
        matrix = np.vstack((actual_mode[i], predicted_mode[i]))
        axes[row, 1].imshow(matrix, aspect="auto", interpolation="nearest", vmin=0, vmax=2,
                            cmap=ListedColormap(COLORS), extent=[0, 1, 1.5, -.5])
        axes[row, 1].set(yticks=[0, 1], yticklabels=["Решатель", "Нейросеть"],
                        xlabel="Доля времени перелёта t / T", title="Режим в каждом временном узле")
    handles = [Patch(color=c, label=n) for c, n in zip(COLORS, MODE_NAMES)]
    axes[0, 1].legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, 1.45),
                      ncol=3, fontsize=8)
    fig.savefig(out / "test_thrust.png", dpi=170)
    cm = confusion_matrix(actual_mode.ravel(), predicted_mode.ravel(), labels=[0, 1, 2])
    counts = cm.sum(axis=1, keepdims=True)
    normalized = np.divide(cm, counts, out=np.zeros_like(cm, dtype=float), where=counts != 0)
    fig2, ax = plt.subplots(figsize=(6.5, 5.5), constrained_layout=True)
    ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
    for i in range(3):
        for j in range(3):
            label = f"{cm[i, j]}\n{normalized[i, j]:.1%}" if counts[i, 0] else "0\nнет примеров"
            ax.text(j, i, label, ha="center", va="center",
                    color="white" if normalized[i, j] > .5 else "black")
    ax.set(xticks=range(3), yticks=range(3), xticklabels=["C", "I", "S"],
           yticklabels=["C", "I", "S"], xlabel="Прогноз", ylabel="Решатель",
           title="Режимы управления на всех тестовых задачах")
    fig2.savefig(out / "confusion_matrix.png", dpi=170)
    fig3, axes3 = plt.subplots(2, 3, figsize=(11, 6), constrained_layout=True)
    for j, ax in enumerate(axes3.flat):
        low = min(actual_p0[:, j].min(), predicted_p0[:, j].min())
        high = max(actual_p0[:, j].max(), predicted_p0[:, j].max())
        ax.scatter(actual_p0[:, j], predicted_p0[:, j], s=8, alpha=.5)
        ax.plot([low, high], [low, high], color="black", ls="--")
        ax.set(xlabel="Решатель", ylabel="Прогноз", title=P0_COLUMNS[j])
    fig3.savefig(out / "test_p0.png", dpi=170)
    if show:
        plt.show()
    for figure in (fig, fig2, fig3):
        plt.close(figure)


def train_and_evaluate(args):
    data, cases = load_dataset(args.out)
    train = data["split"] == "train"
    validation = data["split"] == "validation"
    test = data["split"] == "test"
    rng = np.random.default_rng(args.seed)
    Xq_train = time_features(data["X"][train], data["tau"])
    yq_train = data["q"][train].reshape(-1, 1)
    # Ограничение памяти и длительности обучения для больших наборов.
    if len(Xq_train) > args.max_train_points:
        sample = rng.choice(len(Xq_train), args.max_train_points, replace=False)
        Xq_train, yq_train = Xq_train[sample], yq_train[sample]
    q_model = fit_network(Xq_train, yq_train,
                          time_features(data["X"][validation], data["tau"]),
                          data["q"][validation].reshape(-1, 1), (64, 64, 32), args, "q")
    p0_model = fit_network(data["X"][train], data["p0"][train],
                           data["X"][validation], data["p0"][validation],
                           (64, 64), args, "p0")
    joblib.dump({"features": FEATURES, "q_features": FEATURES + ["tau"],
                 "q_model": q_model, "p0_model": p0_model, "q_clip_min": 0.0,
                 "seed": args.seed, "versions": {"python": platform.python_version(),
                 "numpy": np.__version__, "scipy": scipy.__version__,
                 "sklearn": sklearn.__version__}}, args.out / "models.joblib")
    X = data["X"][test]
    q = data["q"][test]
    predicted_q = np.maximum(0, predict(q_model, time_features(X, data["tau"]))).reshape(q.shape)
    predicted_p0 = predict(p0_model, X)
    actual_mode, actual_b, distance = control(q, X)
    predicted_mode, predicted_b, _ = control(predicted_q, X)
    stable = distance > args.switch_band
    ids = data["case_id"][test]
    per_case = []
    for case_id in np.unique(ids):
        selection = ids == case_id
        per_case.append({"case_id": int(case_id),
                         "accuracy": float(np.mean(actual_mode[selection] == predicted_mode[selection]))})
    pd.DataFrame(per_case).to_csv(args.out / "test_cases_metrics.csv", index=False)
    metrics = {"coverage": {}, "q": regression_metrics(q, predicted_q),
               "thrust_fraction": regression_metrics(actual_b, predicted_b),
               "regimes_all_nodes": classification_metrics(actual_mode, predicted_mode),
               "regimes_away_from_switches": classification_metrics(actual_mode[stable], predicted_mode[stable]),
               "switch_band": args.switch_band,
               "mean_case_accuracy": float(np.mean([r["accuracy"] for r in per_case])),
               "p0": regression_metrics(data["p0"][test], predicted_p0),
               "p0_per_component": {name: regression_metrics(data["p0"][test, j], predicted_p0[:, j])
                                    for j, name in enumerate(P0_COLUMNS)},
               "validation_mse_scaled": {"q": q_model["best_validation_mse"],
                                         "p0": p0_model["best_validation_mse"]}}
    actual_passive = np.any(q < X[:, 3, None], axis=1)
    predicted_passive = np.any(predicted_q < X[:, 3, None], axis=1)
    metrics["passive_presence_sampled"] = {
        "accuracy": float(accuracy_score(actual_passive, predicted_passive)),
        "f1_passive": float(f1_score(actual_passive, predicted_passive, zero_division=0)),
        "support_passive": int(actual_passive.sum()),
        "trajectories": int(len(X)),
        "confusion_matrix_absent_present": confusion_matrix(
            actual_passive, predicted_passive, labels=[False, True]).tolist()}
    for split in ("train", "validation", "test"):
        selected = data["split"] == split
        metrics["coverage"][split] = {
            "requested_cases": int((cases.split == split).sum()),
            "accepted_cases": int(len(np.unique(data["case_id"][selected]))),
            "trajectories": int(selected.sum()),
            "alpha1_trajectories": int(np.sum(np.isclose(data["X"][selected, 3], 1, rtol=0, atol=1e-12)))}
    endpoint = np.isclose(X[:, 3], 1, atol=1e-12, rtol=0)
    if np.any(endpoint):
        metrics["p0_alpha1"] = regression_metrics(data["p0"][test][endpoint], predicted_p0[endpoint])
        metrics["regimes_alpha1"] = classification_metrics(actual_mode[endpoint], predicted_mode[endpoint])
    metrics["shooting_from_predicted_p0"] = shooting_checks(data, predicted_p0, cases, args)
    write_json(args.out / "metrics.json", metrics)
    save_npz(args.out / "test_predictions.npz", X=X, tau=data["tau"], case_id=ids,
             q_true=q, q_pred=predicted_q, p0_true=data["p0"][test], p0_pred=predicted_p0,
             modes_true=actual_mode, modes_pred=predicted_mode)
    print("\nПокрытие выборок")
    print(pd.DataFrame(metrics["coverage"]).T.to_string())
    regimes = metrics["regimes_all_nodes"]
    print(f"\nAccuracy режимов       {regimes['accuracy']:.4f}")
    print(f"Balanced accuracy      {regimes['balanced_accuracy_present_modes']:.4f}")
    print(f"Macro F1, три режима   {regimes['macro_f1_three_modes']:.4f}")
    print(f"MAE q(t)               {metrics['q']['mae']:.6g}")
    print(f"RMSE q(t)              {metrics['q']['rmse']:.6g}")
    print(f"MAE p0                 {metrics['p0']['mae']:.6g}")
    print(f"RMSE p0                {metrics['p0']['rmse']:.6g}")
    print("\nМетрики каждого режима")
    print(pd.DataFrame(regimes["per_class"]).T.to_string(float_format=lambda x: f"{x:.4f}"))
    checks = metrics["shooting_from_predicted_p0"]
    if checks["tested"]:
        print(f"\nУспешная стрельба из p0 сети   {checks['success_rate']:.1%} из {checks['tested']}")
    print("Полные метрики записаны в metrics.json", flush=True)
    plot_results(X, q, predicted_q, data["p0"][test], predicted_p0,
                 data["tau"], ids, args.out, args.show)
    print(f"Результаты сохранены в {args.out.resolve()}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("trajectory_ml_run"))
    parser.add_argument("--stage", choices=["all", "collect", "train"], default="all")
    parser.add_argument("--n-cases", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--time-points", type=int, default=96)
    parser.add_argument("--alpha-end", type=float, default=1.0)
    parser.add_argument("--h-max", type=float, default=0.025)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--max-train-points", type=int, default=100000)
    parser.add_argument("--switch-band", type=float, default=1e-4,
                        help="Полоса около порогов в единицах q для отдельной метрики")
    parser.add_argument("--shooting-checks", type=int, default=5,
                        help="Число тестовых задач alpha=1 для стрельбы из прогноза p0; 0 отключает")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    if args.n_cases < 10 or args.time_points < 8:
        parser.error("Нужно n-cases >= 10 и time-points >= 8.")
    if not (0 < args.alpha_end <= 1 and 0.001 <= args.h_max <= 1 and args.tol > 0):
        parser.error("Проверь alpha-end, h-max и tol.")
    if min(args.epochs, args.patience, args.max_train_points, args.threads) < 1:
        parser.error("epochs, patience, max-train-points и threads должны быть положительными.")
    if args.shooting_checks < 0 or args.switch_band < 0:
        parser.error("shooting-checks и switch-band должны быть неотрицательными.")
    args.out.mkdir(parents=True, exist_ok=True)
    logging.getLogger("solver").setLevel(logging.ERROR)
    with threadpool_limits(limits=args.threads):
        if args.stage in ("all", "collect"):
            collect(args)
        if args.stage in ("all", "train"):
            train_and_evaluate(args)


if __name__ == "__main__":
    main()
