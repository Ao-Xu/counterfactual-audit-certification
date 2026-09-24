"""Failure-boundary audit for semantic corruption and support escape.

The first panel is computed from the locked natural-text Qwen predictions.  The
second panel is an exact bounded witness: a reference law has no mass on a new
style, so a reference-only certificate cannot identify the loss response on
that style.  The witness is intentionally a boundary illustration, not a new
claim about the Qwen model.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1] / "llm_cfpt" / "verified_results"
OUT = Path(__file__).resolve().parent / "boundary_outputs"
SIZES = ["0p5b", "1p5b"]
TASKS = ["attributes", "relations"]
SEEDS = [2027, 2028, 2029]
CONDITIONS = [("factual", 0.0), ("cf_0", 0.0), ("cf_20", 0.2), ("cf_40", 0.4)]


def risk(result):
    ev = result["evaluation"]
    mask = np.asarray(ev["splits"]) == "test"
    p = np.asarray(ev["probabilities"], dtype=float)[mask]
    y = np.asarray(ev["targets"], dtype=int)[mask]
    return float(np.mean(1.0 - p[np.arange(len(y)), y]))


def qwen_records():
    rows = []
    for size in SIZES:
        for task in TASKS:
            factual = {seed: risk(json.loads((ROOT / f"{size}_{task}_{seed}_factual.json").read_text(encoding="utf-8"))) for seed in SEEDS}
            for condition, eta in CONDITIONS:
                for seed in SEEDS:
                    value = risk(json.loads((ROOT / f"{size}_{task}_{seed}_{condition}.json").read_text(encoding="utf-8")))
                    rows.append({
                        "size": size, "task": task, "seed": seed, "condition": condition,
                        "eta": eta, "delta_vs_factual": value - factual[seed],
                    })
    return rows


def support_escape():
    # Reference tags have a response difference of zero, while the unseen tag
    # can take any value inside the bounded loss-difference envelope.
    rows = []
    for gamma in np.linspace(0.0, 1.0, 11):
        for response in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            rows.append({"gamma": float(gamma), "off_support_response": response,
                         "reference_certificate": 0.0, "actual_gap": float(gamma * response)})
    return rows


def write_csv(path, rows):
    rows = list(rows)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def plot(qrows, srows):
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.titlesize": 9, "axes.labelsize": 8})
    fig, ax = plt.subplots(1, 2, figsize=(7.15, 2.55), constrained_layout=True)
    colors = {"0p5b": "#0072B2", "1p5b": "#D55E00"}
    marks = {"attributes": "o", "relations": "s"}
    for size in SIZES:
        for task in TASKS:
            rr = [r for r in qrows if r["size"] == size and r["task"] == task]
            xs, means, lows, highs = [], [], [], []
            for eta in [0.0, 0.2, 0.4]:
                vals = np.asarray([r["delta_vs_factual"] for r in rr if r["eta"] == eta and r["condition"] != "factual"], dtype=float)
                if eta == 0.0:
                    vals = np.asarray([r["delta_vs_factual"] for r in rr if r["condition"] == "cf_0"], dtype=float)
                xs.append(eta); means.append(float(vals.mean()));
                if len(vals) > 1:
                    boot = np.mean(vals[np.random.default_rng(100 + len(vals)).integers(len(vals), size=(2000, len(vals)))], axis=1)
                    lows.append(float(np.quantile(boot, .025))); highs.append(float(np.quantile(boot, .975)))
                else: lows.append(float(vals[0])); highs.append(float(vals[0]))
            ax[0].errorbar(xs, means, yerr=[np.asarray(means)-np.asarray(lows), np.asarray(highs)-np.asarray(means)],
                           marker=marks[task], color=colors[size], linestyle="-" if task == "attributes" else "--",
                           linewidth=1.25, capsize=2, label=f"{size}, {task}")
    ax[0].axhline(0, color="#555555", linestyle=":", linewidth=.8)
    ax[0].set(xlabel="Screened rewrite corruption rate $\\eta$", ylabel="Test loss change vs. factual",
              title="(a) Semantic corruption boundary")
    ax[0].legend(frameon=False, fontsize=5.8, ncol=2, loc="upper left")
    by_response = {}
    for row in srows: by_response.setdefault(row["off_support_response"], []).append(row)
    for response in [-1.0, -0.5, 0.0, 0.5, 1.0]:
        rr = by_response[response]
        ax[1].plot([r["gamma"] for r in rr], [r["actual_gap"] for r in rr], marker="o", markersize=3,
                   linewidth=1.15, label=f"unseen response={response:g}")
    ax[1].axhline(0, color="#555555", linestyle=":", linewidth=.8)
    ax[1].plot([0, 1], [0, 0], color="#222222", linestyle="--", linewidth=1.2, label="reference-only certificate")
    ax[1].set(xlabel="Off-support mass $\\gamma$", ylabel="Risk gap", title="(b) Support escape")
    ax[1].legend(frameon=False, fontsize=5.6, ncol=2, loc="upper left")
    for a in ax:
        a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
        a.grid(axis="y", color="#dddddd", linewidth=.55); a.set_axisbelow(True)
        a.tick_params(labelsize=7)
    path = OUT / "fig_failure_boundaries"
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=.03)
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=.03)
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight", pad_inches=.03)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    qrows, srows = qwen_records(), support_escape()
    write_csv(OUT / "qwen_corruption_boundary.csv", qrows)
    write_csv(OUT / "support_escape_witness.csv", srows)
    manifest = {
        "qwen_source": str(ROOT), "qwen_conditions": CONDITIONS,
        "support_witness": "reference tags have zero response; unseen tag response in [-1,1]",
        "claim_scope": "negative-boundary audit; no claim of semantic preservation for automatically screened rewrites",
    }
    (OUT / "boundary_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    plot(qrows, srows)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
