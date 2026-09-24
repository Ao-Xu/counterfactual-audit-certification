"""Construct and audit finite-sample benefit--harm certificates.

The script has two deliberately separate parts.  The discrete response-
invariant witness supplies a known population target, so the bounded-moment
radius can be checked for empirical coverage.  The fixed-Qwen part uses an
independent split of the already locked calibration resamples to report an
operational bootstrap interval and its uncertain rate.  The latter is not
presented as a population coverage theorem.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
CONF = ROOT / "confirmatory_v1"
DECISION = ROOT / "decision_results" / "decision_raw.json"
OUT = ROOT / "certificate_results"
BUDGETS = [32, 64, 128, 256, 512]
REPETITIONS = 240
BOOTSTRAPS = 120
DELTA = 0.05


def moments(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return x.T @ x / len(y), x.T @ y / len(y), float(np.mean(y * y))


def risk_from_moments(sigma, b, c, w):
    return float(c - 2 * w @ b + w @ sigma @ w)


def plugin_delta(blocks):
    factual, intervention, test, corrupted = blocks
    sf, bf, cf = moments(*factual)
    sl, bl, cl = moments(*intervention)
    se, be, ce = moments(*test)
    sc, bc, cc = moments(*corrupted)
    wf = np.linalg.solve(sf, bf)
    wcf = np.linalg.solve(sl, bc)
    return risk_from_moments(se, be, ce, wcf) - risk_from_moments(se, be, ce, wf)


def radius_from_bounds(m, d=2, rx=1.0, ry=2.0, kappa=1.0, W=2.0, Se=2.0, Be=1.0):
    """The explicit conservative radius from Corollary explicit-calibration."""
    r_sigma = d * rx * rx * np.sqrt(2 * np.log(8 * d * d / DELTA) / m)
    r_b = np.sqrt(d) * rx * ry * np.sqrt(2 * np.log(8 * d / DELTA) / m)
    r_d = 2 * np.sqrt(d) * rx * ry * np.sqrt(2 * np.log(8 * d / DELTA) / m)
    r_c = ry * ry * np.sqrt(np.log(8 / DELTA) / (2 * m))
    if r_sigma >= kappa / 2:
        return float("inf")
    ef = 2 * (r_b + r_sigma * W) / kappa
    ecf = 2 * (r_b + r_d + r_sigma * W) / kappa
    ew = max(ef, ecf)
    er = r_c + 2 * (W + ew) * r_b + (W + ew) ** 2 * r_sigma + (2 * Se * W + 2 * Be) * ew + Se * ew * ew
    return float(2 * er)


def sample_witness(m, rng, sign):
    c = rng.choice([-1.0, 1.0], size=m)
    # Factual: C independent of U,S.
    u_f, s_f = rng.choice([-1.0, 1.0], size=m), rng.choice([-1.0, 1.0], size=m)
    # Intervention: U=C, S independent.
    u_l, s_l = c.copy(), rng.choice([-1.0, 1.0], size=m)
    # Test: S=C, U independent.
    u_e, s_e = rng.choice([-1.0, 1.0], size=m), c.copy()
    y = c / 2.0
    d = np.asarray([-0.5, 0.5]) if sign == "+" else np.asarray([0.5, -0.5])
    return (
        (np.column_stack([u_f, s_f]), y.copy()),
        (np.column_stack([u_l, s_l]), y.copy()),
        (np.column_stack([u_e, s_e]), y.copy()),
        (np.column_stack([u_l, s_l]), y + np.column_stack([u_l, s_l]) @ d),
    )


def exact_witness_delta(sign):
    # The response-invariant construction in Appendix A has exact risks.
    return -0.25 if sign == "+" else 1.75


def bootstrap_radius(blocks, rng):
    n = len(blocks[0][1])
    values = []
    for _ in range(BOOTSTRAPS):
        draws = []
        for x, y in blocks:
            ix = rng.integers(0, len(y), size=len(y))
            draws.append((x[ix], y[ix]))
        values.append(plugin_delta(draws))
    point = plugin_delta(blocks)
    return float(np.quantile(np.abs(np.asarray(values) - point), 0.95))


def run_synthetic():
    rows = []
    for m in BUDGETS:
        for rep in range(REPETITIONS):
            for sign in ("+", "-"):
                rng = np.random.default_rng(510000 + 1000 * m + rep * 2 + (sign == "-"))
                blocks = sample_witness(m, rng, sign)
                point = plugin_delta(blocks)
                analytic = radius_from_bounds(m)
                boot = bootstrap_radius(blocks, np.random.default_rng(520000 + 1000 * m + rep * 2 + (sign == "-")))
                truth = exact_witness_delta(sign)
                for method, radius in (("analytic", analytic), ("bootstrap", boot)):
                    lo, hi = point - radius, point + radius
                    decision = "benefit" if hi < 0 else "harm" if lo > 0 else "uncertain"
                    rows.append({
                        "source": "discrete_witness",
                        "budget": m,
                        "method": method,
                        "truth": truth,
                        "estimate": point,
                        "radius": radius,
                        "covered": bool(lo <= truth <= hi),
                        "decision": decision,
                        "correct_decision": bool((truth < 0 and decision == "benefit") or (truth > 0 and decision == "harm")),
                    })
    return rows


def run_qwen_operational():
    raw = json.loads(DECISION.read_text(encoding="utf-8"))
    # Raw records have one row per condition/resample and expose the full
    # directional estimate as the `full` field.
    groups = defaultdict(list)
    for r in raw:
        key = (int(r["budget"]), int(r["row_index"]))
        groups[key].append(r)
    rows = []
    for (budget, row_index), values in sorted(groups.items()):
        values.sort(key=lambda r: r["repetition"])
        train = np.asarray([r["full"] for r in values[: len(values) // 2]], dtype=float)
        radius = float(np.quantile(np.abs(train - np.median(train)), 0.95))
        for r in values[len(values) // 2 :]:
            point = float(r["full"])
            truth = float(r["true_gap"])
            lo, hi = point - radius, point + radius
            decision = "benefit" if hi < 0 else "harm" if lo > 0 else "uncertain"
            rows.append({
                "source": "qwen_fixed_feature",
                "budget": budget,
                "method": "split_bootstrap",
                "truth": truth,
                "estimate": point,
                "radius": radius,
                "covered": bool(lo <= truth <= hi),
                "decision": decision,
                "correct_decision": bool((truth < 0 and decision == "benefit") or (truth > 0 and decision == "harm")),
            })
    return rows


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["source"], row["budget"], row["method"])].append(row)
    out = []
    for key, values in sorted(grouped.items()):
        decisions = [r["decision"] for r in values]
        nonuncertain = [r for r in values if r["decision"] != "uncertain"]
        correct = [r for r in nonuncertain if r["correct_decision"]]
        out.append({
            "source": key[0],
            "budget": key[1],
            "method": key[2],
            "coverage": float(np.mean([r["covered"] for r in values])),
            "uncertain_rate": float(np.mean([d == "uncertain" for d in decisions])),
            "nonuncertain_reliability": float(len(correct) / len(nonuncertain)) if nonuncertain else float("nan"),
            "mean_radius": float(np.mean([r["radius"] for r in values])),
            "n": len(values),
        })
    return out


def write_outputs(rows, summary):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "certificate_raw.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with (OUT / "certificate_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    lines = [
        "% Generated by certificate_protocol.py. Values are empirical audits, not fabricated guarantees.",
        r"\begin{tabular}{llrrrrr}", r"\toprule",
        r"Source & Radius & $m$ & Coverage & Uncertain & Reliability & Width \\", r"\midrule",
    ]
    labels = {"analytic": "Analytic Hoeffding", "bootstrap": "Bootstrap", "split_bootstrap": "Split bootstrap"}
    def tex_value(value):
        if not np.isfinite(value):
            return r"$\infty$" if np.isinf(value) else "--"
        return f"{value:.3f}"
    for r in summary:
        lines.append(
            f"{r['source'].replace('_',' ')} & {labels[r['method']]} & {r['budget']} & "
            f"{tex_value(r['coverage'])} & {tex_value(r['uncertain_rate'])} & "
            f"{tex_value(r['nonuncertain_reliability'])} & {tex_value(r['mean_radius'])} " + r"\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "certificate_table_generated.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    plot(summary)
    manifest = {
        "nominal_delta": DELTA,
        "synthetic_repetitions": REPETITIONS,
        "bootstrap_repetitions": BOOTSTRAPS,
        "qwen_scope": "split operational bootstrap over locked decision resamples",
        "population_coverage_claim": False,
        "analytic_radius": "bounded primitive-moment corollary",
    }
    (OUT / "certificate_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def plot(summary):
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.6), constrained_layout=True)
    styles = {
        ("discrete_witness", "analytic"): ("#666666", "--", "witness analytic"),
        ("discrete_witness", "bootstrap"): ("#0072B2", "-", "witness bootstrap"),
        ("qwen_fixed_feature", "split_bootstrap"): ("#D55E00", "-", "Qwen split bootstrap"),
    }
    for (source, method), (color, ls, label) in styles.items():
        rr = sorted([r for r in summary if r["source"] == source and r["method"] == method], key=lambda r: r["budget"])
        if not rr:
            continue
        x = [r["budget"] for r in rr]
        axes[0].plot(x, [r["coverage"] for r in rr], marker="o", color=color, linestyle=ls, label=label, linewidth=1.5, markersize=3)
        axes[1].plot(x, [r["uncertain_rate"] for r in rr], marker="o", color=color, linestyle=ls, linewidth=1.5, markersize=3)
        axes[2].plot(x, [r["mean_radius"] if np.isfinite(r["mean_radius"]) else np.nan for r in rr], marker="o", color=color, linestyle=ls, linewidth=1.5, markersize=3)
    axes[0].axhline(0.95, color="#222222", linewidth=.8, linestyle=":", label="nominal 95%")
    axes[0].set_ylabel("Empirical coverage")
    axes[1].set_ylabel("Uncertain decision rate")
    axes[2].set_ylabel("Mean interval radius")
    axes[2].set_yscale("log")
    for ax in axes:
        ax.set_xlabel("Calibration size $m$")
        ax.set_xscale("log", base=2)
        ax.set_xticks(BUDGETS)
        ax.set_xticklabels([str(x) for x in BUDGETS])
        ax.grid(axis="y", color="#dddddd", linewidth=.55)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=7)
    axes[0].legend(frameon=False, fontsize=5.4, loc="lower right")
    fig.suptitle("Coverage and uncertainty of benefit--harm certificates", fontsize=9.5, weight="bold")
    fig.savefig(OUT / "fig_certificate_audit.pdf", bbox_inches="tight", pad_inches=.03)
    fig.savefig(OUT / "fig_certificate_audit.png", dpi=220, bbox_inches="tight", pad_inches=.03)
    plt.close(fig)


if __name__ == "__main__":
    all_rows = run_synthetic() + run_qwen_operational()
    write_outputs(all_rows, summarize(all_rows))
