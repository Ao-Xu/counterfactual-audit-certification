"""Camera-ready real-LLM evidence figure.

All values are read from frozen raw result files.  The script does not select
seeds or conditions after inspecting the plotted outcomes.  It is a visual
recomposition of four already audited analyses:

  (a) matched-error LoRA direction effect;
  (b) same-budget decision utility;
  (c) crossed consistency intervention and its three primary gates;
  (d) finite-pool transfer slack decomposition.

Outputs live beside the source results so the PDF/SVG/PNG can be inspected or
edited without changing the underlying observations.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "real_llm_evidence"
OUT.mkdir(parents=True, exist_ok=True)

BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
VERMILION = "#D55E00"
PURPLE = "#7B61A8"
GRAY = "#5B6573"
LIGHT = "#D9DEE5"
DARK = "#1F2933"


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.0,
            "axes.labelsize": 7.8,
            "axes.titlesize": 8.8,
            "axes.titleweight": "bold",
            "legend.fontsize": 6.8,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.035,
        }
    )


def style_axis(ax: plt.Axes, grid: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=2.4, width=0.6)
    if grid:
        ax.grid(axis=grid, color=LIGHT, linewidth=0.5, alpha=0.78)
        ax.set_axisbelow(True)


def mean_ci(values: List[float]) -> Tuple[float, float, float]:
    a = np.asarray(values, dtype=float)
    mean = float(np.mean(a))
    if len(a) < 2:
        return mean, mean, mean
    half = 2.365 * float(np.std(a, ddof=1)) / np.sqrt(len(a))  # t_.975,7
    return mean, mean - half, mean + half


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def direction_data() -> Dict[str, object]:
    data = load_json(ROOT / "confirmatory_v1" / "lora_metrics.json")
    rows = data["rows"]
    by_seed = {}
    for row in rows:
        by_seed.setdefault(row["seed"], {})[row["condition"]] = row["action_risk"]
    conditions = ["clean", "random", "oppose", "reinforce"]
    values = {}
    for condition in conditions:
        values[condition] = [
            float(by_seed[seed][condition] - by_seed[seed]["factual"])
            for seed in sorted(by_seed)
        ]
    return {"values": values, "seeds": sorted(by_seed)}


def decision_data() -> List[Dict[str, object]]:
    rows = []
    with (ROOT / "decision_results" / "decision_summary.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            row["budget"] = int(row["budget"])
            for key in ["mae", "sign_accuracy"]:
                row[key] = float(row[key])
            rows.append(row)
    return rows


def intervention_data() -> Dict[str, object]:
    return load_json(ROOT / "crossed_consistency_v4" / "results.json")


def slack_data() -> Dict[str, object]:
    return load_json(ROOT / "transfer_tightness" / "metrics.json")


def draw_direction(ax: plt.Axes, data: Dict[str, object]) -> None:
    names = {"clean": "clean CF", "random": "random 20%", "oppose": "oppose 20%", "reinforce": "reinforce 20%"}
    colors = {"clean": BLUE, "random": GRAY, "oppose": GREEN, "reinforce": VERMILION}
    x = np.arange(4)
    rng = np.random.default_rng(719)
    for idx, condition in enumerate(["clean", "random", "oppose", "reinforce"]):
        vals = np.asarray(data["values"][condition], dtype=float)
        jitter = rng.uniform(-0.085, 0.085, size=len(vals))
        ax.scatter(np.full(len(vals), idx) + jitter, vals, s=18, color=colors[condition],
                   alpha=0.72, edgecolor="white", linewidth=0.35, zorder=3)
        mean, low, high = mean_ci(vals.tolist())
        ax.errorbar(idx, mean, yerr=[[mean - low], [high - mean]], fmt="o", color=DARK,
                    markerfacecolor="white", markeredgewidth=0.9, ms=4.4, capsize=2.2,
                    zorder=4)
    ax.axhline(0.0, color=DARK, ls="--", lw=0.75)
    ax.set_xticks(x)
    ax.set_xticklabels([names[c] for c in ["clean", "random", "oppose", "reinforce"]], rotation=17, ha="right")
    ax.set_ylabel("Action-risk change from factual")
    ax.set_title("(a) Matched error rate, different risk", loc="left")
    ax.text(0.98, 0.96, "7/8: reinforce > oppose", transform=ax.transAxes, ha="right", va="top",
            fontsize=6.9, color=VERMILION,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.2})
    style_axis(ax)


def draw_decision(ax: plt.Axes, rows: List[Dict[str, object]]) -> None:
    colors = {"direct": DARK, "full": BLUE, "quadratic": ORANGE, "rate_only": GRAY}
    labels = {"direct": "direct target-law", "full": "directional", "quadratic": "quadratic-only", "rate_only": "rate-only"}
    for method in ["direct", "full", "quadratic", "rate_only"]:
        rr = sorted([r for r in rows if r["method"] == method], key=lambda r: r["budget"])
        ax.plot([r["budget"] for r in rr], [r["mae"] for r in rr], "o-", color=colors[method],
                markerfacecolor="white", markeredgewidth=0.75, ms=3.5, label=labels[method])
    ax.set_xscale("log")
    ax.set_xticks([50, 100, 200, 400, 800])
    ax.set_xticklabels(["50", "100", "200", "400", "800"])
    ax.set_xlabel("Clean calibration labels")
    ax.set_ylabel("Risk-difference MAE (lower is better)")
    ax.set_title("(b) Direction helps before test scoring", loc="left")
    ax.legend(frameon=False, loc="upper right", fontsize=6.0, handlelength=1.3)
    r = next(r for r in rows if r["method"] == "full" and r["budget"] == 800)
    ax.annotate("directional: %.3f" % r["mae"], xy=(800, r["mae"]), xytext=(370, 0.24),
                arrowprops={"arrowstyle": "-", "lw": 0.6, "color": BLUE}, color=BLUE, fontsize=6.6)
    r = next(r for r in rows if r["method"] == "rate_only" and r["budget"] == 800)
    ax.annotate("rate-only: %.3f" % r["mae"], xy=(800, r["mae"]), xytext=(315, 0.57),
                arrowprops={"arrowstyle": "-", "lw": 0.6, "color": GRAY}, color=GRAY, fontsize=6.6)
    style_axis(ax)


def draw_intervention(ax: plt.Axes, data: Dict[str, object]) -> None:
    # Primary endpoint contrasts, all taken directly from the frozen audit.
    specs = [
        ("KL", "W$_2$--W$_0$", data["contrasts"]["calibration"]["Wstar-W0"]["KL"], 100.0, BLUE),
        ("G", "W$_2$--W$_0$", data["contrasts"]["test"]["Wstar-W0"]["G"], 1000.0, GREEN),
        ("R", "W$_2$--U$_0$", data["contrasts"]["test"]["Wstar-U0"]["Rnu"], 100.0, VERMILION),
    ]
    y = np.arange(len(specs))
    for idx, (key, label, record, scale, color) in enumerate(specs):
        diff = float(record["difference"]) * scale
        # These primary records have a one-sided upper95, not a lower95.
        # Fall back to the unscaled point, then scale exactly once. The point
        # is not a lower confidence bound; percentile95 is a separate summary.
        lower = float(record.get("lower95", record.get("low", record["difference"]))) * scale
        upper = float(record.get("upper95", record.get("high", record["difference"]))) * scale
        ax.errorbar(diff, idx, xerr=[[diff - lower], [upper - diff]], fmt="o", color=color,
                    markerfacecolor="white", markeredgewidth=0.9, ms=4.6, capsize=2.2, zorder=4)
        ax.text(upper + 0.02 * max(1.0, abs(upper)), idx + 0.12,
                {"KL": "sensitivity met", "G": "not confirmed", "R": "NI not met"}[key],
                fontsize=6.6, color=color, va="center")
    ax.axvline(0, color=DARK, ls="--", lw=0.75)
    # The reference-risk criterion has a +1 percentage-point non-inferiority
    # margin, while the other two criteria require a negative contrast.
    r = specs[-1][2]
    ni = 0.01 * 100.0
    ax.axvline(ni, color=VERMILION, ls=":", lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(["KL (×100)", r"$G_{\rm style}$ (×1000)", r"$R_\nu$ (×100)"])
    ax.set_xlabel("Contrast (point to one-sided 95% upper bound)")
    ax.set_title("(c) Sensitivity reduction is not all-round robustness", loc="left")
    ax.set_xlim(-8.8, 2.2)
    ax.grid(axis="x", color=LIGHT, linewidth=0.5, alpha=0.78)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=2.4, width=0.6)


def draw_slack(ax: plt.Axes, data: Dict[str, object]) -> None:
    rows = data["qwen"]
    alphas = [float(r["alpha"]) for r in rows]
    x = np.arange(len(alphas))
    width = 0.24
    for j, (key, label, color) in enumerate(
        [("cancellation", "sign coherence", BLUE), ("heterogeneity", "magnitude homogeneity", ORANGE), ("utilization", "overall utilization", GREEN)]
    ):
        ax.bar(x + (j - 1) * width, [float(r[key]) for r in rows], width=width, color=color, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(["0", "0.5", "1", "2"])
    ax.set_xlabel(r"Head perturbation $\alpha$")
    ax.set_ylabel("Ratio of transfer bound used")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("(d) Realized gaps are slack because of cancellation", loc="left")
    ax.legend(frameon=False, fontsize=6.1, loc="upper left", ncol=1)
    ax.text(0.98, 0.96, "oracle stress test reaches equality", transform=ax.transAxes, ha="right", va="top",
            fontsize=6.5, color=DARK,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.2})
    style_axis(ax)


def main() -> None:
    set_style()
    direction = direction_data()
    decisions = decision_data()
    intervention = intervention_data()
    slack = slack_data()
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.15), gridspec_kw={"wspace": 0.34, "hspace": 0.43})
    draw_direction(axes[0, 0], direction)
    draw_decision(axes[0, 1], decisions)
    draw_intervention(axes[1, 0], intervention)
    draw_slack(axes[1, 1], slack)
    fig.subplots_adjust(left=0.085, right=0.975, bottom=0.10, top=0.94, wspace=0.34, hspace=0.43)
    for ext in ["pdf", "svg", "png"]:
        fig.savefig(OUT / ("fig_real_llm_evidence." + ext), dpi=320)
    plt.close(fig)

    # A compact main-text strip keeps the three strongest, directly actionable
    # findings visible without spending a full page on a four-panel appendix
    # dashboard.  It is generated from exactly the same frozen inputs.
    strip, sax = plt.subplots(1, 3, figsize=(7.25, 2.55),
                              gridspec_kw={"wspace": 0.38})
    draw_direction(sax[0], direction)
    draw_decision(sax[1], decisions)
    draw_intervention(sax[2], intervention)
    for a in sax:
        a.tick_params(labelsize=6.2)
        a.xaxis.label.set_size(6.8)
        a.yaxis.label.set_size(6.8)
        a.title.set_fontsize(7.7)
    # The endpoint annotations are useful in the dashboard but too dense in a
    # strip; retain the plotted intervals and gate lines, remove only prose
    # callouts that would collide at column scale.
    for axis in sax:
        for annotation in list(axis.texts):
            annotation.remove()
    strip.subplots_adjust(left=0.065, right=0.995, bottom=0.23, top=0.83, wspace=0.38)
    for ext in ["pdf", "svg", "png"]:
        strip.savefig(OUT / ("fig_real_llm_main_strip." + ext), dpi=360)
    plt.close(strip)

    sources = [
        ROOT / "confirmatory_v1" / "lora_metrics.json",
        ROOT / "decision_results" / "decision_summary.csv",
        ROOT / "crossed_consistency_v4" / "results.json",
        ROOT / "transfer_tightness" / "metrics.json",
    ]
    manifest = {
        "source_files": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "scope": "recomposition of frozen real-LLM results; no seed/condition selection",
        "direction_pools": len(direction["seeds"]),
        "decision_rows": len(decisions),
        "consistency_pools": len(intervention["metrics"]["calibration"]["W0"]["KL"]["per_pool"]),
        "slack_heads": len(slack["qwen"]),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
