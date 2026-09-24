"""Finite-sample decision comparison for the corruption-risk interface.

This analysis reuses the locked fixed-feature heads and disjoint calibration/test
pools from confirmatory_v1. It varies only the number of clean calibration
anchors used by each decision rule. No test quantity is used to choose a
budget, method, or threshold.
"""

from __future__ import annotations

import csv
import json
import pathlib
from collections import defaultdict
from typing import Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "decision_results"
CONF = ROOT / "confirmatory_v1"
SEEDS = list(range(926301, 926333))
BUDGETS = [50, 100, 200, 400, 800]
REPETITIONS = 200
METHODS = ["direct", "full", "quadratic", "rate_only"]
COLORS = {
    "direct": "#0072B2",
    "full": "#D55E00",
    "quadratic": "#009E73",
    "rate_only": "#666666",
}


def corrupt(y: np.ndarray, cue: np.ndarray, eta: float, mechanism: str, seed: int) -> np.ndarray:
    result = y.copy()
    chosen = []
    rng = np.random.default_rng(seed)
    for label in (-1, 1):
        indices = np.flatnonzero(y == label)
        count = int(round(len(indices) * eta))
        if mechanism == "reinforce":
            indices = indices[cue[indices] != y[indices]]
        if mechanism == "oppose":
            indices = indices[cue[indices] == y[indices]]
        if len(indices) < count:
            raise ValueError("requested mechanism has too few eligible rows")
        chosen.extend(rng.permutation(indices)[:count])
    result[chosen] *= -1
    return result


def moment(X: np.ndarray, y: np.ndarray, agreement: float):
    weights = np.where(np.asarray([-1, 1])[None, :] == y[:, None], agreement, 1.0 - agreement)
    flat = X.reshape(-1, X.shape[-1])
    flat_weights = weights.reshape(-1) / len(y)
    yy = np.repeat(y, 2)
    return (flat.T * flat_weights) @ flat, flat.T @ (flat_weights * yy)


def risk(X: np.ndarray, y: np.ndarray, w: np.ndarray, agreement: float = 0.05) -> float:
    weights = np.where(np.asarray([-1, 1])[None, :] == y[:, None], agreement, 1.0 - agreement)
    return float(np.mean(np.sum(weights * (X @ w - y[:, None]) ** 2, axis=1)))


def load_features(name: str):
    value = np.load(CONF / f"features_{name}.npz")
    raw = value["X"].astype(float)
    projection = np.load(CONF / "fixed_projection.npz")
    projected = (raw - projection["center"]) @ projection["basis"] / projection["scale"]
    projected = np.concatenate(
        [np.ones((*projected.shape[:2], 1), dtype=float), projected], axis=-1
    )
    return projected, value["y"].astype(int)


def class_stratified_indices(y: np.ndarray, m: int, rng: np.random.Generator) -> np.ndarray:
    if m % 2:
        raise ValueError("all registered calibration budgets must be even")
    parts = []
    for label in (-1, 1):
        pool = np.flatnonzero(y == label)
        parts.append(rng.choice(pool, size=m // 2, replace=True))
    return np.concatenate(parts)


def calibration_state(C: np.ndarray, cy: np.ndarray) -> Dict[str, np.ndarray]:
    S, b = moment(C, cy, 0.5)
    Se, be = moment(C, cy, 0.05)
    Sf, bf = moment(C, cy, 0.95)
    Sinv = np.linalg.inv(S)
    u = Sinv @ b
    wf = np.linalg.solve(Sf, bf)
    G = risk(C, cy, wf) - risk(C, cy, u)
    return {"Sinv": Sinv, "Se": Se, "u": u, "be": be, "G": G}


def calibration_predictions(
    C: np.ndarray,
    cy: np.ndarray,
    state: Dict[str, np.ndarray],
    row: Dict,
    rng_seed: int,
) -> Dict[str, float]:
    """Compute full and quadratic predictions on one calibration resample."""
    Sinv, Se, u, be, G = (
        state["Sinv"],
        state["Se"],
        state["u"],
        state["be"],
        state["G"],
    )
    flat = C.reshape(-1, C.shape[-1])
    yy = np.repeat(cy, 2)
    cue = np.tile([-1, 1], len(cy))
    target = corrupt(yy, cue, float(row["eta"]), row["mechanism"], rng_seed)
    d = flat.T @ (target - yy) / len(yy)
    move = Sinv @ d
    quadratic = -G + move @ Se @ move
    alignment = 2.0 * move @ (Se @ u - be)
    return {"full": float(quadratic + alignment), "quadratic": float(quadratic)}


def collect() -> List[Dict]:
    C, cy = load_features("calibration")
    heads = np.load(CONF / "fixed_heads.npz")
    fixed = json.loads((CONF / "fixed_metrics.json").read_text(encoding="utf-8"))
    rows = fixed["fixed"]["rows"]
    if len(rows) != len(heads["heads"]):
        raise RuntimeError("fixed prediction rows and heads are not aligned")
    factual_by_seed = {seed: heads["factual"][i] for i, seed in enumerate(SEEDS)}
    output: List[Dict] = []
    for m in BUDGETS:
        for repetition in range(REPETITIONS):
            rng = np.random.default_rng(941000 + 1000 * m + repetition)
            ix = class_stratified_indices(cy, m, rng)
            Cb, yb = C[ix], cy[ix]
            state = calibration_state(Cb, yb)
            pred_cache: Dict[tuple, Dict[str, float]] = {}
            for row_index, row in enumerate(rows):
                if float(row["eta"]) <= 0.0:
                    continue
                key = (int(row["seed"]), row["mechanism"], float(row["eta"]))
                pred_cache[key] = calibration_predictions(
                    Cb, yb, state, row, 942000 + row_index
                )
            for row_index, row in enumerate(rows):
                if float(row["eta"]) <= 0.0:
                    continue
                seed = int(row["seed"])
                key = (seed, row["mechanism"], float(row["eta"]))
                direct = risk(Cb, yb, heads["heads"][row_index]) - risk(
                    Cb, yb, factual_by_seed[seed]
                )
                p = pred_cache[key]
                same_rate = [
                    pred_cache[(seed, mechanism, float(row["eta"]))]["full"]
                    for mechanism in ("random", "reinforce", "oppose")
                ]
                output.append(
                    {
                        "budget": m,
                        "repetition": repetition,
                        "row_index": row_index,
                        "seed": seed,
                        "mechanism": row["mechanism"],
                        "eta": float(row["eta"]),
                        "true_gap": float(row["observed"]),
                        "direct": float(direct),
                        "full": p["full"],
                        "quadratic": p["quadratic"],
                        "rate_only": float(np.mean(same_rate)),
                    }
                )
    return output


def summarize(rows: Iterable[Dict]) -> List[Dict]:
    grouped = defaultdict(list)
    for row in rows:
        for method in METHODS:
            grouped[(int(row["budget"]), method)].append(
                (float(row[method]), float(row["true_gap"]))
            )
    output = []
    for (budget, method), values in sorted(grouped.items()):
        pred = np.asarray([v[0] for v in values])
        truth = np.asarray([v[1] for v in values])
        output.append(
            {
                "budget": budget,
                "method": method,
                "mae": float(np.mean(np.abs(pred - truth))),
                "sign_accuracy": float(np.mean(np.sign(pred) == np.sign(truth))),
                "false_benefit": float(np.mean((pred < 0) & (truth >= 0))),
                "false_harm": float(np.mean((pred > 0) & (truth <= 0))),
                "n": int(len(values)),
            }
        )
    return output


def write_outputs(raw: List[Dict], summary: List[Dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "decision_raw.json").write_text(
        json.dumps(raw, indent=2, allow_nan=False), encoding="utf-8"
    )
    with (OUT / "decision_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    lines = [
        "% Generated by decision_protocol.py from locked confirmatory_v1 outputs.",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"$m$ & Method & MAE & Sign acc. & False benefit & False harm \\",
        r"\midrule",
    ]
    labels = {
        "direct": "Direct validation",
        "full": "Full directional",
        "quadratic": "Quadratic-only",
        "rate_only": "Rate-only",
    }
    for row in summary:
        lines.append(
            f"{row['budget']} & {labels[row['method']]} & {row['mae']:.4f} & "
            f"{row['sign_accuracy']:.3f} & {row['false_benefit']:.3f} & "
            f"{row['false_harm']:.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "decision_table_generated.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def plot(summary: List[Dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8), constrained_layout=True)
    for method in METHODS:
        selected = [r for r in summary if r["method"] == method]
        selected.sort(key=lambda r: r["budget"])
        x = [r["budget"] for r in selected]
        axes[0].plot(
            x,
            [r["mae"] for r in selected],
            marker="o",
            linewidth=1.8,
            markersize=4,
            color=COLORS[method],
            label=method.replace("_", " "),
        )
        axes[1].plot(
            x,
            [r["sign_accuracy"] for r in selected],
            marker="o",
            linewidth=1.8,
            markersize=4,
            color=COLORS[method],
            label=method.replace("_", " "),
        )
    for ax, ylabel, title in [
        (axes[0], "Absolute error in test risk gap", "Magnitude prediction"),
        (axes[1], "Sign accuracy", "Benefit/harm direction"),
    ]:
        ax.set_xscale("log", basex=2)
        ax.set_xticks(BUDGETS)
        ax.set_xticklabels([str(x) for x in BUDGETS])
        ax.set_xlabel("Clean calibration anchors $m$")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", weight="bold", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="#dddddd", linewidth=0.55, alpha=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=7.5)
    axes[0].legend(frameon=False, fontsize=6.7, loc="upper right")
    fig.suptitle(
        "Finite-sample decisions under a matched clean-label budget",
        fontsize=10,
        weight="bold",
    )
    fig.savefig(OUT / "fig_decision_budget.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(OUT / "fig_decision_budget.png", dpi=220, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main() -> None:
    raw = collect()
    summary = summarize(raw)
    write_outputs(raw, summary)
    plot(summary)
    print(json.dumps({"raw_rows": len(raw), "summary_rows": len(summary)}, indent=2))


if __name__ == "__main__":
    main()
