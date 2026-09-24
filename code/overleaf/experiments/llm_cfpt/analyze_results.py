"""Aggregate real-LLM runs and create Appendix C figures/tables."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
from collections import defaultdict
from typing import Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "blue": "#0072B2",
    "orange": "#D55E00",
    "green": "#009E73",
    "purple": "#CC79A7",
    "yellow": "#E69F00",
    "gray": "#666666",
}


def read_results(result_dir: pathlib.Path) -> List[Dict]:
    rows = []
    for path in sorted(result_dir.glob("*.json")):
        if path.name.startswith("manifest"):
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("status") == "completed":
            rows.append(value)
    if not rows:
        raise RuntimeError(f"No completed result files in {result_dir}")
    return rows


def mean_std(values: Iterable[float]) -> tuple[float, float, int]:
    values = np.asarray(list(values), dtype=float)
    if len(values) == 0:
        return float("nan"), float("nan"), 0
    return float(values.mean()), float(values.std(ddof=1) if len(values) > 1 else 0.0), len(values)


def aggregate(rows: List[Dict], keys: List[str]) -> List[Dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key) for key in keys)].append(row)
    output = []
    for key, members in sorted(groups.items(), key=lambda item: item[0]):
        item = {name: value for name, value in zip(keys, key)}
        for metric, getter in {
            "matched_accuracy": lambda r: r["matched"]["accuracy"],
            "mismatched_accuracy": lambda r: r["mismatched"]["accuracy"],
            "shift_gap": lambda r: r["shift_gap_accuracy"],
            "predictive_kl": lambda r: r["predictive_pairwise_kl"],
        }.items():
            values = [getter(r) for r in members]
            mu, sd, count = mean_std(values)
            item[metric] = mu
            item[metric + "_std"] = sd
            item[metric + "_count"] = count
        mixtures = {}
        for delta in ["0.0", "0.25", "0.5", "0.75", "1.0"]:
            values = [r["mixture_accuracy"].get(delta, float("nan")) for r in members]
            mu, sd, count = mean_std(values)
            mixtures[delta] = {"mean": mu, "std": sd, "count": count}
        item["mixture_accuracy"] = mixtures
        output.append(item)
    return output


def ci(std: float, count: int) -> float:
    return 1.96 * std / np.sqrt(count) if count > 1 else 0.0


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#dddddd", linewidth=0.6, alpha=0.65)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8)


def save_figure(fig, pdf_path: pathlib.Path, png_path: pathlib.Path) -> None:
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(png_path, dpi=220, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def plot_fixed_budget(rows: List[Dict], figure_dir: pathlib.Path) -> None:
    cf_rows = [r for r in rows if r["kind"] == "cf" and r["eta"] == 0.0]
    groups = aggregate(cf_rows, ["n", "k", "eta", "kind"])
    groups = sorted(groups, key=lambda r: r["k"])
    factual = aggregate([r for r in rows if r["kind"] == "factual"], ["n", "k", "eta", "kind"])[0]

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.8), constrained_layout=True)
    ax = axes[0]
    x = np.arange(len(groups))
    values = [g["mismatched_accuracy"] for g in groups]
    errors = [ci(g["mismatched_accuracy_std"], g["mismatched_accuracy_count"]) for g in groups]
    ax.errorbar(x, values, yerr=errors, color=COLORS["blue"], marker="o", linewidth=2,
                markersize=5, capsize=3, label="CFPT")
    ax.axhline(factual["mismatched_accuracy"], color=COLORS["gray"], linestyle="--", linewidth=1.2,
               label="factual baseline")
    ax.set_xticks(x)
    ax.set_xticklabels([f"$K={g['k']}$\n$n={g['n']}$" for g in groups])
    ax.set_xlabel("Sibling edits per anchor (fixed $nK=1000$)", fontsize=8)
    ax.set_ylabel("OOD accuracy", fontsize=8)
    ax.set_ylim(0.45, 0.90)
    ax.set_title("Fixed-budget allocation", fontsize=9, loc="left", weight="bold")
    style_axes(ax)
    ax.legend(frameon=False, fontsize=7, loc="lower right")

    ax = axes[1]
    markers = {1: "o", 2: "s", 4: "^", 8: "D"}
    for g in groups:
        ax.scatter(g["predictive_kl"], g["shift_gap"], s=45, marker=markers[g["k"]],
                   color=COLORS["orange"], edgecolor="white", linewidth=0.6)
        ax.annotate(f"K={g['k']}", (g["predictive_kl"], g["shift_gap"]),
                    textcoords="offset points", xytext=(4, 3), fontsize=7)
    xs = np.asarray([g["predictive_kl"] for g in groups])
    ys = np.asarray([g["shift_gap"] for g in groups])
    if len(xs) >= 2 and np.ptp(xs) > 0:
        slope, intercept = np.polyfit(xs, ys, 1)
        grid = np.linspace(xs.min(), xs.max(), 100)
        ax.plot(grid, slope * grid + intercept, color=COLORS["orange"], alpha=0.55, linewidth=1.3)
    ax.set_xlabel("Predictive pairwise KL (proxy for $L_{\\rm CF}$)", fontsize=8)
    ax.set_ylabel("Matched--mismatched accuracy gap", fontsize=8)
    ax.set_title("Residual sensitivity and shift-gap diagnostic", fontsize=9, loc="left", weight="bold")
    style_axes(ax)
    fig.suptitle("Real-LLM pilot: fixed-budget and transfer diagnostics", fontsize=10, weight="bold")
    save_figure(fig, figure_dir / "fig_c1_fixed_budget.pdf", figure_dir / "fig_c1_fixed_budget.png")


def plot_corruption(rows: List[Dict], figure_dir: pathlib.Path) -> None:
    corr_rows = [r for r in rows if r["kind"] == "cf" and r["n"] == 250 and r["k"] == 4]
    groups = aggregate(corr_rows, ["n", "k", "eta", "kind"])
    groups = sorted(groups, key=lambda r: r["eta"])
    factual = aggregate([r for r in rows if r["kind"] == "factual"], ["n", "k", "eta", "kind"])[0]

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.8), constrained_layout=True)
    ax = axes[0]
    x = np.asarray([g["eta"] for g in groups])
    y = np.asarray([g["mismatched_accuracy"] for g in groups])
    e = np.asarray([ci(g["mismatched_accuracy_std"], g["mismatched_accuracy_count"]) for g in groups])
    ax.errorbar(x * 100, y, yerr=e, color=COLORS["purple"], marker="o", linewidth=2,
                markersize=5, capsize=3, label="CFPT, $n=250,K=4$")
    ax.axhline(factual["mismatched_accuracy"], color=COLORS["gray"], linestyle="--", linewidth=1.2,
               label="factual baseline")
    ax.set_xlabel("Injected label-flip rate (%)", fontsize=8)
    ax.set_ylabel("OOD accuracy", fontsize=8)
    ax.set_xticks(x * 100)
    ax.set_xticklabels([f"{int(round(value * 100))}" for value in x])
    ax.set_ylim(0.45, 0.90)
    ax.set_title("Controlled corruption sensitivity", fontsize=9, loc="left", weight="bold")
    style_axes(ax)
    ax.legend(frameon=False, fontsize=7, loc="lower left")

    ax = axes[1]
    for eta, color, label in [(0.0, COLORS["green"], "0% flips"), (0.20, COLORS["purple"], "20% flips")]:
        selected = [g for g in groups if abs(g["eta"] - eta) < 1e-8]
        if not selected:
            continue
        g = selected[0]
        deltas = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0])
        vals = np.asarray([g["mixture_accuracy"][str(delta)]["mean"] for delta in deltas])
        errs = np.asarray([ci(g["mixture_accuracy"][str(delta)]["std"], g["mixture_accuracy"][str(delta)]["count"]) for delta in deltas])
        ax.errorbar(deltas, vals, yerr=errs, color=color, marker="o", linewidth=1.8,
                    markersize=4, capsize=2.5, label=label)
    ax.set_xlabel("Shift mixture weight $\\delta$", fontsize=8)
    ax.set_ylabel("Mixture accuracy", fontsize=8)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
    ax.set_ylim(0.45, 0.90)
    ax.set_title("Covered-to-shifted evaluation", fontsize=9, loc="left", weight="bold")
    style_axes(ax)
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    fig.suptitle("Real-LLM pilot: semantic supervision and shift sensitivity", fontsize=10, weight="bold")
    save_figure(fig, figure_dir / "fig_c2_corruption.pdf", figure_dir / "fig_c2_corruption.png")


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def fmt_pm(mean: float, std: float, digits: int = 3) -> str:
    return f"{fmt(mean, digits)}$\\pm${fmt(std, digits)}"


def write_csv(path: pathlib.Path, rows: List[Dict]) -> None:
    fieldnames = [
        "kind", "n", "k", "eta", "matched_accuracy", "matched_accuracy_std",
        "mismatched_accuracy", "mismatched_accuracy_std", "shift_gap", "shift_gap_std",
        "predictive_kl", "predictive_kl_std", "count",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "kind": row["kind"], "n": row["n"], "k": row["k"], "eta": row["eta"],
                "matched_accuracy": row["matched_accuracy"], "matched_accuracy_std": row["matched_accuracy_std"],
                "mismatched_accuracy": row["mismatched_accuracy"], "mismatched_accuracy_std": row["mismatched_accuracy_std"],
                "shift_gap": row["shift_gap"], "shift_gap_std": row["shift_gap_std"],
                "predictive_kl": row["predictive_kl"], "predictive_kl_std": row["predictive_kl_std"],
                "count": row["mismatched_accuracy_count"],
            })


def write_latex(path: pathlib.Path, rows: List[Dict], raw_dir: pathlib.Path) -> None:
    main = sorted([r for r in rows if r["kind"] == "cf" and r["eta"] == 0.0], key=lambda r: r["k"])
    factual = [r for r in rows if r["kind"] == "factual"][0]
    corr = sorted([r for r in rows if r["kind"] == "cf" and r["n"] == 250 and r["k"] == 4], key=lambda r: r["eta"])
    data_summary = json.loads((raw_dir / "data_summary.json").read_text(encoding="utf-8"))
    diagnostic_records = [
        json.loads(line)
        for line in (raw_dir / "diagnostic_cf_k4.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    diagnostic_valid = sum(bool(item.get("generation_valid", False)) for item in diagnostic_records)
    valid = sum(item["valid_generations"] for item in data_summary["counterfactual"].values()) + diagnostic_valid
    total = sum(item["rows"] for item in data_summary["counterfactual"].values()) + len(diagnostic_records)
    best = max(main, key=lambda r: r["mismatched_accuracy"])
    main_lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Training data & $n$ & $K$ & matched acc. & OOD acc. & pred. KL \\",
        r"\midrule",
        (
            f"Factual & 1000 & 1 & {fmt_pm(factual['matched_accuracy'], factual['matched_accuracy_std'])} "
            f"& {fmt_pm(factual['mismatched_accuracy'], factual['mismatched_accuracy_std'])} & -- "
            + r"\\"
        ),
    ]
    for row in main:
        main_lines.append(
            f"CFPT & {row['n']} & {row['k']} & {fmt_pm(row['matched_accuracy'], row['matched_accuracy_std'])} "
            f"& {fmt_pm(row['mismatched_accuracy'], row['mismatched_accuracy_std'])} "
            f"& {fmt_pm(row['predictive_kl'], row['predictive_kl_std'])} " + r"\\"
        )
    main_lines += [r"\bottomrule", r"\end{tabular}"]
    corr_lines = [
        r"\begin{tabular}{rrrrr}",
        r"\toprule",
        r"$\eta$ & matched acc. & OOD acc. & shift gap & pred. KL \\",
        r"\midrule",
    ]
    for row in corr:
        corr_lines.append(
            f"{row['eta']:.2f} & {fmt_pm(row['matched_accuracy'], row['matched_accuracy_std'])} "
            f"& {fmt_pm(row['mismatched_accuracy'], row['mismatched_accuracy_std'])} "
            f"& {fmt_pm(row['shift_gap'], row['shift_gap_std'])} "
            f"& {fmt_pm(row['predictive_kl'], row['predictive_kl_std'])} " + r"\\"
        )
    corr_lines += [r"\bottomrule", r"\end{tabular}"]
    lines = [
        "% Auto-generated from experiments/llm_cfpt/results/*.json. Do not edit by hand.",
        f"\\newcommand{{\\CGenValid}}{{{valid}\\,/\\,{total}}}",
        f"\\newcommand{{\\CBestK}}{{{best['k']}}}",
        f"\\newcommand{{\\CBestOOD}}{{{fmt(best['mismatched_accuracy'])}}}",
        f"\\newcommand{{\\CFactualOOD}}{{{fmt(factual['mismatched_accuracy'])}}}",
        f"\\newcommand{{\\CMainSeeds}}{{{main[0]['mismatched_accuracy_count']}}}",
        "\\newcommand{\\CMainTable}{%",
    ] + main_lines + ["}", "\\newcommand{\\CCorruptionTable}{%"] + corr_lines + ["}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=pathlib.Path, required=True)
    parser.add_argument("--raw-dir", type=pathlib.Path, required=True)
    parser.add_argument("--figure-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    rows = read_results(args.result_dir)
    summary_rows = aggregate(rows, ["kind", "n", "k", "eta"])
    write_csv(args.result_dir / "summary.csv", summary_rows)
    write_latex(args.result_dir / "appendix_c_generated.tex", summary_rows, args.raw_dir)
    plot_fixed_budget(rows, args.figure_dir)
    plot_corruption(rows, args.figure_dir)
    print(json.dumps({"completed": len(rows), "summary_rows": len(summary_rows)}, indent=2))


if __name__ == "__main__":
    main()
