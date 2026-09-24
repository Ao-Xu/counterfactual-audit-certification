"""Compute paired utility quantities from the locked Qwen/MultiNLI predictions.

The rewrite arm is automatically screened, so epsilon_gen is intentionally
not set to zero.  The output is an operational audit of paired quantities;
it is not reported as a theorem-valid semantic certificate.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1] / "llm_cfpt" / "crossed_consistency_v4"
OUT = Path(__file__).resolve().parent / "qwen_paired_results"
CONDITIONS = ["Wlow", "Wstar"]
POOLS = range(8)
B = 1.0
K = 4
RHO = 0.04
DELTA = 0.05


def read_probs(split: str, pool: int, condition: str) -> np.ndarray:
    path = ROOT / "predictions" / split / f"confirm{pool}_{condition}.json"
    return np.asarray(json.loads(path.read_text(encoding="utf-8"))["probabilities"], dtype=float)


def labels(split: str) -> np.ndarray:
    rows = json.loads((ROOT / "splits" / f"{split}.json").read_text(encoding="utf-8"))
    return np.asarray([[row["y"] for row in anchor] for anchor in rows], dtype=int)


def loss(prob: np.ndarray, y: np.ndarray) -> np.ndarray:
    # Labels are -1/1 and probabilities use columns 0/1.
    col = (y == 1).astype(int)
    return 1.0 - np.take_along_axis(prob, col[..., None], axis=-1)[..., 0]


def radius(n: int, k: int, delta: float = DELTA) -> float:
    t = np.log(8.0 / delta)
    return (
        B * np.sqrt(2.0 * t / n)
        + 2.0 * B * t / (3.0 * n)
        + B * np.sqrt(2.0 * t / (n * k))
        + 2.0 * B * t / (3.0 * n * k)
    )


def one(split: str, pool: int, condition: str, y: np.ndarray) -> dict:
    factual = loss(read_probs(split, pool, "W0"), y)
    candidate = loss(read_probs(split, pool, condition), y)
    z = candidate - factual
    anchor_mean = z.mean(axis=1)
    delta_hat = float(z.mean())
    sigma_a = float(anchor_mean.var(ddof=1))
    sigma_s = float(z.var(axis=1, ddof=1).mean())
    return {
        "split": split,
        "pool": pool,
        "condition": condition,
        "n": int(z.shape[0]),
        "k": int(z.shape[1]),
        "delta_hat": delta_hat,
        "sigma_a_hat": sigma_a,
        "sigma_s_hat": sigma_s,
        "gamma_bd": radius(z.shape[0], z.shape[1]),
        "unadjusted_U_rho": delta_hat + radius(z.shape[0], z.shape[1]) + B * np.sqrt(RHO),
        "semantic_correction": "unknown_automatic_screen",
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [
        one(split, pool, condition, labels(split))
        for split in ["calibration", "test"]
        for pool in POOLS
        for condition in CONDITIONS
    ]
    (OUT / "qwen_paired_raw.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with (OUT / "qwen_paired_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for condition in CONDITIONS:
        cr = [r for r in rows if r["split"] == "calibration" and r["condition"] == condition]
        tr = [r for r in rows if r["split"] == "test" and r["condition"] == condition]
        summary[condition] = {
            "calibration_delta_mean": float(np.mean([r["delta_hat"] for r in cr])),
            "calibration_U_mean": float(np.mean([r["unadjusted_U_rho"] for r in cr])),
            "test_delta_mean": float(np.mean([r["delta_hat"] for r in tr])),
            "test_sigma_a_mean": float(np.mean([r["sigma_a_hat"] for r in tr])),
            "test_sigma_s_mean": float(np.mean([r["sigma_s_hat"] for r in tr])),
        }
    (OUT / "qwen_paired_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot(rows)


def plot(rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.55), constrained_layout=True)
    colors = {"Wlow": "#E69F00", "Wstar": "#0072B2"}
    for condition, color in colors.items():
        cr = [r for r in rows if r["split"] == "calibration" and r["condition"] == condition]
        tr = [r for r in rows if r["split"] == "test" and r["condition"] == condition]
        axes[0].scatter(
            [r["unadjusted_U_rho"] for r in cr],
            [r["delta_hat"] for r in tr],
            color=color,
            label=condition,
            s=24,
            alpha=0.85,
        )
        axes[1].plot(
            range(8),
            [r["sigma_a_hat"] for r in tr],
            marker="o",
            color=color,
            label=condition,
            linewidth=1.4,
            markersize=3,
        )
    axes[0].axvline(0, color="#222222", linestyle="--", linewidth=.8)
    axes[0].axhline(0, color="#222222", linestyle=":", linewidth=.8)
    axes[0].set_xlabel("calibration $U_h^{\\rm bd}$ (unadjusted)")
    axes[0].set_ylabel("independent-test $\\hat\\Delta_h$")
    axes[0].set_title("(a) Audit versus test gap", loc="left", weight="bold")
    axes[1].set_xlabel("training pool")
    axes[1].set_ylabel("test anchor variance $\\hat\\sigma_A^2$")
    axes[1].set_title("(b) Paired anchor component", loc="left", weight="bold")
    axes[1].set_xticks(range(8))
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="#dddddd", linewidth=.55)
        ax.tick_params(labelsize=7)
    axes[0].legend(frameon=False, fontsize=6)
    fig.savefig(OUT / "fig_qwen_paired_audit.pdf", bbox_inches="tight", pad_inches=.03)
    fig.savefig(OUT / "fig_qwen_paired_audit.png", dpi=240, bbox_inches="tight", pad_inches=.03)
    plt.close(fig)


if __name__ == "__main__":
    main()
