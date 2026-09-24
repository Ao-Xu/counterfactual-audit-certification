"""Finite-class paired-audit simulation for the comparative utility interface.

This is a controlled response-invariant witness.  It is not a Qwen result and
does not fabricate an end-to-end training outcome.  Each candidate has a
known paired loss difference g_h(S)=delta_h+alpha_h S under a uniform
reference style law.  The script compares the candidate-specific paired
certificate with the fully bounded fallback-to-baseline rule.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "safe_selection_results"
N = 1024
K = 4
J = 3
RHO = 0.04
DELTA = 0.05
B = 0.5
REPS = 500
NOISE = 0.02

# delta_nu and nuisance slope alpha.  The reference law is S in {-1,+1}
# uniformly; the worst covered shift has delta_pi = sign(alpha)*sqrt(rho).
CANDIDATES = {
    "safe_cf": {"delta_nu": -0.14, "alpha": 0.01},
    "borderline_cf": {"delta_nu": -0.12, "alpha": 0.08},
    "harmful_cf": {"delta_nu": 0.03, "alpha": 0.02},
}


def radius(delta: float = DELTA / J) -> float:
    t = np.log(8.0 / delta)
    return (
        2.0 * B * t / (3.0 * N)
        + np.sqrt(2.0 * B**2 * t / N + 4.0 * B**2 * t**2 / (9.0 * N**2))
        + 2.0 * B * t / (3.0 * N * K)
        + np.sqrt(
            2.0 * B**2 * t / (N * K)
            + 4.0 * B**2 * t**2 / (9.0 * N**2 * K**2)
        )
    )


def run_one(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    rows = []
    for name, pars in CANDIDATES.items():
        styles = rng.choice([-1.0, 1.0], size=(N, K))
        g = pars["delta_nu"] + pars["alpha"] * styles
        z = np.clip(g + NOISE * rng.choice([-1.0, 1.0], size=(N, K)), -B, B)
        estimate = float(z.mean())
        paired_cert = estimate + radius() + abs(pars["alpha"]) * np.sqrt(RHO)
        bounded_cert = estimate + radius() + B * np.sqrt(RHO)
        worst = pars["delta_nu"] + abs(pars["alpha"]) * np.sqrt(RHO)
        rows.append(
            {
                "candidate": name,
                "estimate": estimate,
                "paired_certificate": paired_cert,
                "bounded_certificate": bounded_cert,
                "true_reference_gap": pars["delta_nu"],
                "true_worst_gap": worst,
                "paired_safe": bool(paired_cert < 0),
                "bounded_safe": bool(bounded_cert < 0),
            }
        )
    selected_pair = min(rows, key=lambda r: r["paired_certificate"])
    selected_bounded = min(rows, key=lambda r: r["bounded_certificate"])

    def choose(row: dict, key: str) -> tuple[str, float]:
        if row[key] >= 0:
            return "baseline", 0.0
        return row["candidate"], row["true_worst_gap"]

    selected_pair_name, selected_pair_worst = choose(
        selected_pair, "paired_certificate"
    )
    selected_bounded_name, selected_bounded_worst = choose(
        selected_bounded, "bounded_certificate"
    )
    return {
        "seed": seed,
        "selected_pair": selected_pair_name,
        "selected_pair_worst_gap": selected_pair_worst,
        "paired_safe_selection": bool(selected_pair_worst <= 0),
        "selected_bounded": selected_bounded_name,
        "selected_bounded_worst_gap": selected_bounded_worst,
        "bounded_safe_selection": bool(selected_bounded_worst <= 0),
        "rows": rows,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    runs = [run_one(910000 + i) for i in range(REPS)]
    summary = {
        "n": N,
        "k": K,
        "candidate_count": J,
        "rho": RHO,
        "delta": DELTA,
        "bounded_loss": B,
        "repetitions": REPS,
        "radius": radius(),
        "paired_selected_rate": {
            name: float(np.mean([r["selected_pair"] == name for r in runs]))
            for name in ["baseline", *CANDIDATES]
        },
        "bounded_selected_rate": {
            name: float(np.mean([r["selected_bounded"] == name for r in runs]))
            for name in ["baseline", *CANDIDATES]
        },
        "paired_safe_selection_rate": float(
            np.mean([r["paired_safe_selection"] for r in runs])
        ),
        "bounded_safe_selection_rate": float(
            np.mean([r["bounded_safe_selection"] for r in runs])
        ),
        "paired_false_acceptance_rate": float(
            np.mean(
                [
                    (r["selected_pair"] not in ("baseline",))
                    and not r["paired_safe_selection"]
                    for r in runs
                ]
            )
        ),
        "bounded_false_acceptance_rate": float(
            np.mean(
                [
                    (r["selected_bounded"] not in ("baseline",))
                    and not r["bounded_safe_selection"]
                    for r in runs
                ]
            )
        ),
        "mean_paired_selected_worst_gap": float(
            np.mean([r["selected_pair_worst_gap"] for r in runs])
        ),
        "mean_bounded_selected_worst_gap": float(
            np.mean([r["selected_bounded_worst_gap"] for r in runs])
        ),
    }
    (OUT / "safe_selection_raw.json").write_text(
        json.dumps(runs, indent=2), encoding="utf-8"
    )
    (OUT / "safe_selection_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    with (OUT / "safe_selection_candidates.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "candidate",
                "delta_nu",
                "alpha",
                "true_worst_gap",
                "mean_estimate",
                "mean_paired_certificate",
                "mean_bounded_certificate",
            ],
        )
        writer.writeheader()
        for name, pars in CANDIDATES.items():
            vals = [r for run in runs for r in run["rows"] if r["candidate"] == name]
            writer.writerow(
                {
                    "candidate": name,
                    "delta_nu": pars["delta_nu"],
                    "alpha": pars["alpha"],
                    "true_worst_gap": pars["delta_nu"]
                    + abs(pars["alpha"]) * np.sqrt(RHO),
                    "mean_estimate": np.mean([r["estimate"] for r in vals]),
                    "mean_paired_certificate": np.mean(
                        [r["paired_certificate"] for r in vals]
                    ),
                    "mean_bounded_certificate": np.mean(
                        [r["bounded_certificate"] for r in vals]
                    ),
                }
            )
    write_table(summary)
    plot(runs, summary)


def write_table(summary: dict) -> None:
    lines = [
        "% Generated by safe_selection_simulation.py.",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Policy & $\Delta_\nu$ & worst gap & paired sel. & bounded sel. \\",
        r"\midrule",
    ]
    for name, pars in CANDIDATES.items():
        paired_selected = summary["paired_selected_rate"][name]
        bounded_selected = summary["bounded_selected_rate"][name]
        lines.append(
            f"{name.replace('_', ' ')} & {pars['delta_nu']:.2f} & "
            f"{pars['delta_nu'] + abs(pars['alpha']) * np.sqrt(RHO):.3f} & "
            f"{paired_selected:.3f} & {bounded_selected:.3f} \\\\"
        )
    lines.append(
        f"baseline & 0.00 & 0.000 & "
        f"{summary['paired_selected_rate']['baseline']:.3f} & "
        f"{summary['bounded_selected_rate']['baseline']:.3f} \\\\"
    )
    lines += [
        r"\midrule",
        f"paired safe rate & \\multicolumn{{4}}{{c}}{{{summary['paired_safe_selection_rate']:.3f}}} \\\\",
        f"bounded safe rate & \\multicolumn{{4}}{{c}}{{{summary['bounded_safe_selection_rate']:.3f}}} \\\\",
        f"paired false accept & \\multicolumn{{4}}{{c}}{{{summary['paired_false_acceptance_rate']:.3f}}} \\\\",
        f"bounded false accept & \\multicolumn{{4}}{{c}}{{{summary['bounded_false_acceptance_rate']:.3f}}} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    (OUT / "safe_selection_table.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def plot(runs: list[dict], summary: dict) -> None:
    names = list(CANDIDATES)
    certs = {
        name: np.asarray(
            [
                row["paired_certificate"]
                for run in runs
                for row in run["rows"]
                if row["candidate"] == name
            ]
        )
        for name in names
    }
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.55), constrained_layout=True)
    colors = ["#0072B2", "#E69F00", "#D55E00"]
    for name, color in zip(names, colors):
        axes[0].hist(
            certs[name],
            bins=28,
            alpha=0.48,
            color=color,
            label=name.replace("_", " "),
        )
    axes[0].axvline(0, color="#222222", linestyle="--", linewidth=0.9)
    axes[0].set_xlabel("paired upper certificate $U_h^{\\rm pair}$")
    axes[0].set_ylabel("replications")
    axes[0].set_title("(a) Candidate certificates", loc="left", weight="bold")
    axes[0].legend(frameon=False, fontsize=6)
    labels = ["baseline", "safe", "borderline", "harmful"]
    x = np.arange(len(labels))
    paired_rates = [
        summary["paired_selected_rate"]["baseline"],
        *[summary["paired_selected_rate"][name] for name in names],
    ]
    bounded_rates = [
        summary["bounded_selected_rate"]["baseline"],
        *[summary["bounded_selected_rate"][name] for name in names],
    ]
    axes[1].bar(x - 0.18, paired_rates, width=0.36, label="paired", color="#0072B2")
    axes[1].bar(x + 0.18, bounded_rates, width=0.36, label="bounded", color="#999999")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=18, ha="right")
    axes[1].set_xlim(-0.6, len(labels) - 0.4)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("selection frequency")
    axes[1].set_title("(b) Fallback-aware selection", loc="left", weight="bold")
    axes[1].text(
        0.02,
        0.96,
        f"paired safe={summary['paired_safe_selection_rate']:.3f}\\n"
        f"bounded safe={summary['bounded_safe_selection_rate']:.3f}",
        transform=axes[1].transAxes,
        va="top",
        fontsize=7,
        bbox=dict(facecolor="white", edgecolor="#cccccc", pad=3),
    )
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="#dddddd", linewidth=0.55)
        ax.tick_params(labelsize=7)
    axes[1].legend(frameon=False, fontsize=6, loc="upper right")
    fig.savefig(OUT / "fig_safe_selection.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(OUT / "fig_safe_selection.png", dpi=240, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


if __name__ == "__main__":
    main()
