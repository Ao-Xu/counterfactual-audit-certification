"""Controlled finite-state separation experiment for shift-class granularity.

The population is deliberately finite and fully known.  It separates three
response geometries: a nuisance response shared globally, a response shared
only within observed groups, and a response varying by task anchor.  We then
estimate the corresponding chi-square worst-case loss difference from noisy
audit observations.  The experiment reports an empirical diagnostic audit,
not a claim that Qwen training satisfies the finite-state generator.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).resolve().parent / "controlled_outputs"
SEED = 271828
N_GROUPS = 8
ANCHORS_PER_GROUP = 32
N_UNITS = N_GROUPS * ANCHORS_PER_GROUP
N_STYLES = 4
RHO = 0.25
BUDGETS = [32, 64, 128, 256]
# The original design was 1200 x 160.  The audit is unchanged in its cells;
# these values keep the repeated finite-sample check reproducible on the
# workstation while avoiding a Python-loop bottleneck in the bootstrap.
REPS = 600
BOOT = 60
SCENARIOS = ["global", "group", "task"]
CLASSES = ["global", "group", "task"]


def class_certificate(values: np.ndarray, groups: np.ndarray, cls: str, rho: float = RHO) -> float:
    values = np.asarray(values, dtype=float)
    if cls == "global":
        cells = [values.mean(axis=0)]
    elif cls == "group":
        cells = [values[groups == g].mean(axis=0) for g in sorted(set(groups.tolist()))]
    elif cls == "task":
        cells = [row for row in values]
    else:
        raise ValueError(cls)
    out = []
    q = np.full(values.shape[1], 1.0 / values.shape[1])
    for cell in cells:
        mu = float(q @ cell)
        var = float(q @ (cell - mu) ** 2)
        out.append(mu + np.sqrt(max(0.0, rho * var)))
    return float(np.mean(out))


def reference_mean(values: np.ndarray, groups: np.ndarray, cls: str) -> float:
    if cls == "global":
        return float(values.mean())
    if cls == "group":
        return float(np.mean([values[groups == g].mean() for g in sorted(set(groups.tolist()))]))
    if cls == "task":
        return float(values.mean())
    raise ValueError(cls)


def make_population(rng: np.random.Generator, scenario: str):
    groups = np.repeat(np.arange(N_GROUPS), ANCHORS_PER_GROUP)
    styles = np.arange(N_STYLES)
    # A small negative reference effect represents a clean candidate benefit;
    # the shift-sensitive component is large enough to expose class mismatch.
    if scenario == "global":
        style = np.asarray([0.075, -0.025, -0.025, -0.025])
        values = np.tile(style, (N_UNITS, 1)) - 0.03
    elif scenario == "group":
        group_style = rng.normal(0.0, 0.075, size=(N_GROUPS, N_STYLES))
        group_style -= group_style.mean(axis=1, keepdims=True)
        values = group_style[groups] - 0.03
    elif scenario == "task":
        task_style = rng.normal(0.0, 0.075, size=(N_UNITS, N_STYLES))
        task_style -= task_style.mean(axis=1, keepdims=True)
        values = task_style - 0.03
    else:
        raise ValueError(scenario)
    return np.clip(values, -0.25, 0.25), groups


def bootstrap_radius(observed: np.ndarray, groups: np.ndarray, cls: str,
                     rng: np.random.Generator) -> float:
    n = len(observed)
    vals = []
    for _ in range(BOOT):
        if cls == "global" or cls == "task":
            ix = rng.integers(0, n, size=n)
            vals.append(class_certificate(observed[ix], groups[ix], cls))
        else:
            ix_parts = []
            for g in sorted(set(groups.tolist())):
                loc = np.flatnonzero(groups == g)
                ix_parts.append(loc[rng.integers(0, len(loc), size=len(loc))])
            ix = np.concatenate(ix_parts)
            vals.append(class_certificate(observed[ix], groups[ix], cls))
    point = class_certificate(observed, groups, cls)
    return float(np.quantile(np.abs(np.asarray(vals) - point), 0.95))


def one_population(scenario: str, pop_seed: int):
    rng = np.random.default_rng(pop_seed)
    values, groups = make_population(rng, scenario)
    truth = {cls: class_certificate(values, groups, cls) for cls in CLASSES}
    return values, groups, truth


def run() -> List[Dict]:
    rows = []
    for scenario_index, scenario in enumerate(SCENARIOS):
        values, groups, truth = one_population(scenario, SEED + scenario_index)
        for m in BUDGETS:
            for rep in range(REPS):
                rng = np.random.default_rng(700000 + scenario_index * 100000 + m * 1000 + rep)
                ix = rng.choice(N_UNITS, size=m, replace=False)
                # Bounded audit noise emulates a finite test/sample estimate of
                # the pairwise loss difference; it is independent by unit.
                observed = np.clip(values[ix] + rng.normal(0.0, 0.035, size=(m, N_STYLES)), -1.0, 1.0)
                sample_groups = groups[ix]
                for cls in CLASSES:
                    point = class_certificate(observed, sample_groups, cls)
                    radius = bootstrap_radius(observed, sample_groups, cls, np.random.default_rng(rng.integers(2**31)))
                    upper = point + radius
                    target = truth[scenario]
                    rows.append({
                        "scenario": scenario, "budget": m, "rep": rep, "audit_class": cls,
                        "point": point, "radius": radius, "upper": upper,
                        "truth_matching": target, "covers": bool(upper >= target),
                        "undercoverage": bool(upper < target),
                        "reference": reference_mean(observed, sample_groups, cls),
                        "robust_excess": point - reference_mean(observed, sample_groups, cls),
                    })
    return rows


def write_csv(path: Path, rows: Iterable[Dict]) -> None:
    rows = list(rows)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def summarize(rows: List[Dict]) -> List[Dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["scenario"], row["budget"], row["audit_class"])].append(row)
    out = []
    for key, vals in sorted(groups.items()):
        out.append({
            "scenario": key[0], "budget": key[1], "audit_class": key[2], "n": len(vals),
            "coverage": float(np.mean([x["covers"] for x in vals])),
            "undercoverage": float(np.mean([x["undercoverage"] for x in vals])),
            "mean_point": float(np.mean([x["point"] for x in vals])),
            "mean_radius": float(np.mean([x["radius"] for x in vals])),
            "mean_upper": float(np.mean([x["upper"] for x in vals])),
            "mean_truth": float(np.mean([x["truth_matching"] for x in vals])),
        })
    return out


def plot(summary: List[Dict]) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.titlesize": 9, "axes.labelsize": 8})
    colors = {"global": "#0072B2", "group": "#D55E00", "task": "#009E73"}
    labels = {"global": "Global", "group": "Group", "task": "Task"}
    fig, ax = plt.subplots(1, 3, figsize=(7.15, 2.55), constrained_layout=True)
    # Compact heatmap: rows are the true generator class and columns are the
    # audit class.  The fixed budget makes the mismatch pattern immediately
    # legible without a large multi-line legend.
    heat = np.asarray([[next(x["undercoverage"] for x in summary
                              if x["scenario"] == s and x["audit_class"] == c and x["budget"] == 64)
                       for c in CLASSES] for s in SCENARIOS])
    im = ax[0].imshow(heat, vmin=0, vmax=1, cmap="Blues")
    for i in range(len(SCENARIOS)):
        for j in range(len(CLASSES)):
            ax[0].text(j, i, f"{heat[i,j]:.2f}", ha="center", va="center",
                       color="white" if heat[i,j] > .5 else "#222222", fontsize=8)
    ax[0].set_xticks(range(3)); ax[0].set_xticklabels(["global", "group", "task"])
    ax[0].set_yticks(range(3)); ax[0].set_yticklabels(["global", "group", "task"])
    ax[0].set(xlabel="Audit class", ylabel="True generator", title="(a) Undercoverage at $m=64$")
    cbar = fig.colorbar(im, ax=ax[0], fraction=.045, pad=.03)
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label("rate", fontsize=7)
    for scenario in SCENARIOS:
        rr = [x for x in summary if x["scenario"] == scenario and x["audit_class"] == scenario]
        rr.sort(key=lambda x: x["budget"])
        ax[1].plot([x["budget"] for x in rr], [x["mean_radius"] for x in rr],
                   marker="o", linewidth=1.4, markersize=3.5, color=colors[scenario], label=labels[scenario])
    ax[1].set(xlabel="Audit anchors", ylabel="Bootstrap radius", title="(b) Matched radius")
    ax[1].legend(frameon=False, fontsize=6.2, loc="upper right")
    for scenario in SCENARIOS:
        rr = [x for x in summary if x["scenario"] == scenario and x["audit_class"] == scenario]
        rr.sort(key=lambda x: x["budget"])
        ax[2].plot([x["budget"] for x in rr], [x["mean_point"] for x in rr], marker="o", color=colors[scenario], label=f"{labels[scenario]} point")
        ax[2].plot([x["budget"] for x in rr], [x["mean_truth"] for x in rr], linestyle="--", color=colors[scenario], label=f"{labels[scenario]} truth")
    ax[2].set(xlabel="Audit anchors", ylabel="Robust loss difference", title="(c) Matched target")
    ax[2].axhline(0, color="#555555", linestyle=":", linewidth=.8)
    ax[2].legend(frameon=False, fontsize=5.8, ncol=2, loc="upper center", bbox_to_anchor=(.5, -.30))
    for a in ax[1:]:
        a.set_xscale("log", basex=2); a.set_xticks(BUDGETS); a.set_xticklabels([str(x) for x in BUDGETS])
        a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
        a.grid(axis="y", color="#dddddd", linewidth=.55); a.set_axisbelow(True)
        a.tick_params(labelsize=7)
    ax[0].spines["top"].set_visible(False); ax[0].spines["right"].set_visible(False)
    ax[0].grid(False); ax[0].tick_params(labelsize=7)
    path = OUT / "fig_controlled_granularity"
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=.03)
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=.03)
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight", pad_inches=.03)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = run()
    summary = summarize(rows)
    write_csv(OUT / "controlled_raw.csv", rows)
    write_csv(OUT / "controlled_summary.csv", summary)
    manifest = {
        "population": {"groups": N_GROUPS, "anchors_per_group": ANCHORS_PER_GROUP, "styles": N_STYLES},
        "scenarios": SCENARIOS, "audit_classes": CLASSES, "rho": RHO,
        "budgets": BUDGETS, "repetitions": REPS, "bootstrap_repetitions": BOOT,
        "observation_noise_sd": 0.035,
        "claim_scope": "known finite-state separation and empirical coverage audit; not an LLM theorem test",
    }
    (OUT / "controlled_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    plot(summary)
    print(json.dumps(manifest, indent=2))
    print("raw rows", len(rows), "summary rows", len(summary))


if __name__ == "__main__":
    main()
