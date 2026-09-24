"""Generate the theorem-aligned Appendix C analysis from completed artifacts.

The script does not train models.  It consumes the completed nested runs and
coverage runs, adds a cluster bootstrap over the independent diagnostic pool,
computes the predictive-KL transfer certificate, and aggregates the phase-
mixture grid.  All manuscript numbers and figures are derived here.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t


COLORS = {
    "blue": "#0072B2",
    "orange": "#D55E00",
    "green": "#009E73",
    "purple": "#CC79A7",
    "yellow": "#E69F00",
    "gray": "#666666",
    "teal": "#56B4E9",
    "black": "#222222",
}


def completed(directory: pathlib.Path) -> List[Dict]:
    rows = []
    for path in sorted(directory.glob("*.json")):
        if path.name.startswith("manifest"):
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("status") == "completed":
            rows.append(value)
    return rows


def mean_std(values: Iterable[float]) -> Tuple[float, float, int]:
    values = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if len(values) == 0:
        return float("nan"), float("nan"), 0
    return float(values.mean()), float(values.std(ddof=1) if len(values) > 1 else 0.0), len(values)


def ci(std: float, count: int) -> float:
    return float(student_t.ppf(0.975, count - 1)) * std / np.sqrt(count) if count > 1 else 0.0


def fmt(value: float, digits: int = 3) -> str:
    if not np.isfinite(value):
        return "--"
    return f"{value:.{digits}f}"


def fmt_pm(mean: float, std: float, digits: int = 3) -> str:
    return f"{fmt(mean, digits)}$\\pm${fmt(std, digits)}"


def _row_values(result: Dict, metric: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    probabilities = np.asarray(result["probabilities"], dtype=float)
    targets = np.asarray(result["targets"], dtype=int)
    anchors = np.asarray(result["anchor_ids"])
    genres = np.asarray(result["genres"])
    target_probabilities = probabilities[np.arange(len(targets)), targets]
    if metric == "randomized_error":
        values = 1.0 - target_probabilities
    elif metric == "nll":
        values = -np.log(np.clip(target_probabilities, 1e-12, 1.0))
    elif metric == "accuracy":
        values = (np.asarray(result["predictions"], dtype=int) == targets).astype(float)
    else:
        raise ValueError(f"unsupported row metric: {metric}")
    return values, anchors, genres, probabilities


def conditional_weights(
    available: Sequence[str], global_weights: Dict[str, float]
) -> Optional[Dict[str, float]]:
    masses = {genre: float(global_weights.get(genre, 0.0)) for genre in available}
    total = sum(masses.values())
    if total <= 0:
        return None
    return {genre: mass / total for genre, mass in masses.items() if mass > 0}


def conditional_mixture_metric(
    result: Dict, global_weights: Dict[str, float], metric: str
) -> float:
    """Evaluate a genre mixture conditional on each observed anchor.

    The diagnostic pool has four of the five genres for each anchor.  We first
    restrict the reference and test genre scores to the genres observed for
    that anchor and renormalize.  This preserves the anchor marginal and makes
    the finite calculation match the conditional nuisance laws in the theory.
    """
    values, anchors, genres, _ = _row_values(result, metric)
    totals = []
    for anchor in sorted(set(anchors.tolist())):
        indices = np.flatnonzero(anchors == anchor)
        local = conditional_weights(genres[indices].tolist(), global_weights)
        if local is None:
            return float("nan")
        totals.append(
            sum(
                local[genre] * float(np.mean(values[indices[genres[indices] == genre]]))
                for genre in local
            )
        )
    return float(np.mean(totals)) if totals else float("nan")


def conditional_chi_square(
    result: Dict, reference: Dict[str, float], test: Dict[str, float]
) -> float:
    _, anchors, genres, _ = _row_values(result, "accuracy")
    values = []
    for anchor in sorted(set(anchors.tolist())):
        indices = np.flatnonzero(anchors == anchor)
        available = genres[indices].tolist()
        ref = conditional_weights(available, reference)
        target = conditional_weights(available, test)
        if ref is None or target is None:
            return float("nan")
        if any(mass > 0 and ref.get(genre, 0.0) == 0 for genre, mass in target.items()):
            return float("inf")
        # Sum over the union of supports.  Iterating only over target's
        # positive-mass categories misses a reference-only category and can
        # understate chi^2 (e.g. ref=(1/2,1/2), target=(1,0)).
        support = set(ref) | set(target)
        values.append(
            sum(
                (target.get(genre, 0.0) - ref.get(genre, 0.0)) ** 2
                / ref[genre]
                for genre in support
                if ref.get(genre, 0.0) > 0
            )
        )
    return float(np.mean(values)) if values else float("nan")


def conditional_pairwise_kl(
    result: Dict, reference: Dict[str, float]
) -> float:
    _, anchors, genres, probabilities = _row_values(result, "accuracy")
    anchor_values = []
    for anchor in sorted(set(anchors.tolist())):
        indices = np.flatnonzero(anchors == anchor)
        available = genres[indices].tolist()
        local = conditional_weights(available, reference)
        if local is None:
            return float("inf")
        distributions = {
            genre: probabilities[index]
            for genre, index in zip(genres[indices].tolist(), indices.tolist())
        }
        total = 0.0
        for left, left_weight in local.items():
            for right, right_weight in local.items():
                p = distributions[left]
                q = distributions[right]
                total += (
                    left_weight
                    * right_weight
                    * float(np.sum(p * (np.log(p + 1e-12) - np.log(q + 1e-12))))
                )
        anchor_values.append(total)
    return float(np.mean(anchor_values)) if anchor_values else float("nan")


def conditional_risk_variance(
    result: Dict, global_weights: Dict[str, float]
) -> float:
    """Estimate V_nu for the bounded expected-zero-one action loss.

    For each anchor, the diagnostic pool supplies one conditional score per
    observed genre.  We compute the variance of those genre-conditioned scores
    under the same conditional mixture used by the risk calculation, then
    average over anchors.  This is the finite-pool counterpart of V_nu in
    Theorem 2, not a variance of argmax accuracy.
    """
    values, anchors, genres, _ = _row_values(result, "randomized_error")
    anchor_variances = []
    for anchor in sorted(set(anchors.tolist())):
        indices = np.flatnonzero(anchors == anchor)
        local = conditional_weights(genres[indices].tolist(), global_weights)
        if local is None:
            return float("nan")
        genre_values = {
            genre: float(np.mean(values[indices[genres[indices] == genre]]))
            for genre in local
        }
        mean = sum(local[genre] * genre_values[genre] for genre in local)
        anchor_variances.append(
            sum(local[genre] * (genre_values[genre] - mean) ** 2 for genre in local)
        )
    return float(np.mean(anchor_variances)) if anchor_variances else float("nan")


def style_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#dddddd", linewidth=0.6, alpha=0.65)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8)


def save_figure(fig, figure_dir: pathlib.Path, stem: str) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(figure_dir / f"{stem}.png", dpi=240, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Theorem 1: two-level fresh-evaluation variance


def diagnostic_loss_groups(result: Dict) -> Dict[str, np.ndarray]:
    diagnostic = result["diagnostic"]
    probabilities = np.asarray(diagnostic["probabilities"], dtype=float)
    targets = np.asarray(diagnostic["targets"], dtype=int)
    anchors = np.asarray(diagnostic["anchor_ids"])
    losses = 1.0 - probabilities[np.arange(len(targets)), targets]
    groups = {}
    for anchor in sorted(set(anchors.tolist())):
        groups[str(anchor)] = losses[anchors == anchor]
    return groups


def bootstrap_fresh_variance(
    groups: Dict[str, np.ndarray], n_eval: int, k_eval: int, reps: int, seed: int
) -> float:
    keys = list(groups)
    rng = np.random.default_rng(seed)
    risks = np.empty(reps, dtype=float)
    for index in range(reps):
        anchor_indices = rng.integers(0, len(keys), size=n_eval)
        values = []
        for anchor_index in anchor_indices:
            members = groups[keys[int(anchor_index)]]
            sibling_indices = rng.integers(0, len(members), size=k_eval)
            values.extend(members[sibling_indices].tolist())
        risks[index] = float(np.mean(values))
    return float(np.var(risks, ddof=1))


def cluster_scaling(aligned_rows: Sequence[Dict]) -> Tuple[List[Dict], Dict[str, float]]:
    n_values = [16, 32, 64]
    k_values = [1, 2, 4]
    cells = []
    for n_eval in n_values:
        for k_eval in k_values:
            values = []
            for index, result in enumerate(aligned_rows):
                groups = diagnostic_loss_groups(result)
                values.append(bootstrap_fresh_variance(groups, n_eval, k_eval, 1200, 202700 + index * 100 + n_eval * 10 + k_eval))
            mu, sd, count = mean_std(values)
            cells.append({"n_eval": n_eval, "k_eval": k_eval, "variance": mu, "std": sd, "count": count})

    x = np.asarray([[1.0 / cell["n_eval"], 1.0 / (cell["n_eval"] * cell["k_eval"])] for cell in cells])
    y = np.asarray([cell["variance"] for cell in cells])
    coefficients, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    fitted = x @ coefficients
    residual = y - fitted
    total = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - float(np.sum(residual ** 2) / total) if total > 0 else float("nan")

    naive_x = (1.0 / np.asarray([cell["n_eval"] * cell["k_eval"] for cell in cells]))[:, None]
    naive_coef = float(np.linalg.lstsq(naive_x, y, rcond=None)[0][0])
    naive_fit = naive_x[:, 0] * naive_coef
    naive_residual = y - naive_fit
    naive_r2 = 1.0 - float(np.sum(naive_residual ** 2) / total) if total > 0 else float("nan")
    for cell, prediction, naive_prediction in zip(cells, fitted, naive_fit):
        cell["two_level_fit"] = float(prediction)
        cell["naive_fit"] = float(naive_prediction)

    stats = {
        "a": float(coefficients[0]),
        "b": float(coefficients[1]),
        "r2": r2,
        "naive_r2": naive_r2,
        "naive_coef": naive_coef,
        "n_models": len(aligned_rows),
        "n_eval_max": max(n_values),
    }
    return cells, stats


def fixed_budget_slice(aligned_rows: Sequence[Dict], budget: int = 1000) -> List[Dict]:
    """Summarize the preregistered fixed synthetic-budget slice.

    The aligned grid contains four conditions with n*K=1000.  We report both
    matched and independent diagnostic accuracy; the latter is the relevant
    OOD-style metric for the sibling-generation question.
    """
    grouped: Dict[Tuple[int, int], List[Dict]] = defaultdict(list)
    for result in aligned_rows:
        n = int(result["n"])
        k = int(result["k"])
        if n * k == budget:
            grouped[(n, k)].append(result)
    rows = []
    for (n, k), results in sorted(grouped.items()):
        diagnostic_mean, diagnostic_std, count = mean_std(
            [float(result["diagnostic"]["accuracy"]) for result in results]
        )
        matched_mean, matched_std, _ = mean_std(
            [float(result["matched"]["accuracy"]) for result in results]
        )
        rows.append(
            {
                "n": n,
                "k": k,
                "budget": n * k,
                "diagnostic_accuracy": diagnostic_mean,
                "diagnostic_std": diagnostic_std,
                "matched_accuracy": matched_mean,
                "matched_std": matched_std,
                "count": count,
            }
        )
    return rows


def plot_cluster_scaling(cells: Sequence[Dict], stats: Dict[str, float], figure_dir: pathlib.Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.15), constrained_layout=True)

    ax = axes[0, 0]
    for color, n_eval, marker in [(COLORS["blue"], 16, "o"), (COLORS["green"], 32, "s"), (COLORS["orange"], 64, "^")]:
        selected = [cell for cell in cells if cell["n_eval"] == n_eval]
        selected.sort(key=lambda cell: cell["k_eval"])
        x = np.asarray([cell["k_eval"] for cell in selected])
        y = np.asarray([cell["variance"] for cell in selected])
        e = np.asarray([cell["std"] for cell in selected])
        ax.errorbar(x, y, yerr=e, color=color, marker=marker, linewidth=1.8, capsize=2.5, label=f"n={n_eval}")
    ax.set_xlabel("Fresh siblings per anchor $K$", fontsize=8)
    ax.set_ylabel("Fresh-risk variance", fontsize=8)
    ax.set_title("(a) Increasing $K$ removes only part of the variance", fontsize=9, loc="left", weight="bold")
    ax.set_xticks([1, 2, 4])
    style_axes(ax)
    ax.legend(frameon=False, fontsize=7)

    ax = axes[0, 1]
    selected = [cell for cell in cells if cell["k_eval"] == 4]
    selected.sort(key=lambda cell: cell["n_eval"])
    x = np.asarray([cell["n_eval"] for cell in selected])
    between = stats["a"] / x
    within = stats["b"] / (x * 4.0)
    ax.bar(x, between, width=6, color=COLORS["blue"], label="$a/n$")
    ax.bar(x, within, width=6, bottom=between, color=COLORS["orange"], label="$b/(nK)$")
    ax.set_xlabel("Fresh anchors $n$", fontsize=8)
    ax.set_ylabel("Fitted variance components", fontsize=8)
    ax.set_title("(b) Anchor and sibling components", fontsize=9, loc="left", weight="bold")
    ax.set_xticks([16, 32, 64])
    style_axes(ax)
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1, 0]
    x = np.asarray([cell["two_level_fit"] for cell in cells])
    y = np.asarray([cell["variance"] for cell in cells])
    ax.scatter(x, y, s=42, color=COLORS["blue"], edgecolor="white", linewidth=0.6)
    lo, hi = float(min(x.min(), y.min())), float(max(x.max(), y.max()))
    ax.plot([lo, hi], [lo, hi], color=COLORS["gray"], linestyle="--", linewidth=1)
    ax.set_xlabel("$a/n+b/(nK)$ fitted", fontsize=8)
    ax.set_ylabel("Bootstrap variance", fontsize=8)
    ax.set_title(f"(c) Two-level fit ($R^2={stats['r2']:.2f}$)", fontsize=9, loc="left", weight="bold")
    style_axes(ax)

    ax = axes[1, 1]
    x = np.asarray([cell["naive_fit"] for cell in cells])
    y = np.asarray([cell["variance"] for cell in cells])
    ax.scatter(x, y, s=42, color=COLORS["purple"], edgecolor="white", linewidth=0.6)
    lo, hi = float(min(x.min(), y.min())), float(max(x.max(), y.max()))
    ax.plot([lo, hi], [lo, hi], color=COLORS["gray"], linestyle="--", linewidth=1)
    ax.set_xlabel("$c/(nK)$ fitted", fontsize=8)
    ax.set_ylabel("Bootstrap variance", fontsize=8)
    ax.set_title(f"(d) Naive fit ($R^2={stats['naive_r2']:.2f}$)", fontsize=9, loc="left", weight="bold")
    style_axes(ax)

    fig.suptitle("Fresh-evaluation variance decomposition on independent diagnostic clusters", fontsize=10, weight="bold")
    save_figure(fig, figure_dir, "fig_c5_cluster_scaling")


# ---------------------------------------------------------------------------
# Theorem 2: direct transfer certificate


def flatten_transfer(coverage_rows: Sequence[Dict]) -> List[Dict]:
    output = []
    for row in coverage_rows:
        for model_name in ["full", "partial"]:
            diagnostic = row[model_name]["diagnostic"]
            for path in row[model_name]["paths"]:
                reference = path["reference"]
                test = path["test"]
                l_cf = conditional_pairwise_kl(diagnostic, reference)
                chi = conditional_chi_square(diagnostic, reference, test)
                path_name = path["path"]
                if np.isnan(chi):
                    path_name = "unavailable_" + model_name
                elif np.isinf(chi) and path_name.startswith("covered_shift"):
                    path_name = (
                        "support_escape_full"
                        if model_name == "full"
                        else "support_escape_partial"
                    )
                bound = float(np.sqrt(chi * l_cf / 2.0)) if not np.isnan(chi) else float("nan")
                v_nu = conditional_risk_variance(diagnostic, reference)
                centered_bound = (
                    float(np.sqrt(chi * v_nu))
                    if np.isfinite(chi) and np.isfinite(v_nu)
                    else float("nan")
                )
                reference_randomized_error = conditional_mixture_metric(
                    diagnostic, reference, "randomized_error"
                )
                test_randomized_error = conditional_mixture_metric(
                    diagnostic, test, "randomized_error"
                )
                output.append({
                    "seed": row["seed"],
                    "model": model_name,
                    "path": path_name,
                    "delta": float(path["delta"]),
                    "chi_square": chi,
                    "l_cf": l_cf,
                    "bound": bound,
                    "v_nu": v_nu,
                    "centered_bound": centered_bound,
                    "accuracy_gap": abs(float(path["test_accuracy"]) - float(path["reference_accuracy"])),
                    "nll_gap": abs(float(path["test_nll"]) - float(path["reference_nll"])),
                    "reference_randomized_error": reference_randomized_error,
                    "test_randomized_error": test_randomized_error,
                    "randomized_error_gap": abs(test_randomized_error - reference_randomized_error),
                    "supported": bool(path["supported"]) and np.isfinite(chi),
                })
    return output


def transfer_summary(records: Sequence[Dict]) -> List[Dict]:
    groups = defaultdict(list)
    for record in records:
        groups[(record["model"], record["path"], record["delta"])].append(record)
    output = []
    for key, members in sorted(groups.items()):
        model, path, delta = key
        item = {"model": model, "path": path, "delta": delta}
        for metric in [
            "chi_square", "l_cf", "bound", "v_nu", "centered_bound",
            "accuracy_gap", "nll_gap", "randomized_error_gap",
        ]:
            values = [member[metric] for member in members]
            if any(np.isinf(value) for value in values):
                item[metric] = float("inf")
                item[metric + "_std"] = 0.0
                item[metric + "_count"] = len(values)
            else:
                finite = [value for value in values if np.isfinite(value)]
                mu, sd, count = mean_std(finite)
                item[metric] = mu
                item[metric + "_std"] = sd
                item[metric + "_count"] = count
        item["supported"] = all(member["supported"] for member in members)
        item["finite_certificate"] = all(np.isfinite(member["bound"]) for member in members)
        item["coverage_fraction"] = float(np.mean([
            (member["randomized_error_gap"] <= member["bound"] + 1e-12)
            for member in members if np.isfinite(member["bound"])
        ])) if any(np.isfinite(member["bound"]) for member in members) else float("nan")
        output.append(item)
    return output


def plot_transfer(summary: Sequence[Dict], figure_dir: pathlib.Path) -> None:
    covered = [item for item in summary if item["finite_certificate"] and item["delta"] > 0]
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.95), constrained_layout=True)

    ax = axes[0]
    markers = {"full": "o", "partial": "s"}
    colors = {"full": COLORS["blue"], "partial": COLORS["green"]}
    for model in ["full", "partial"]:
        selected = [item for item in covered if item["model"] == model]
        x = np.asarray([item["bound"] for item in selected])
        y = np.asarray([item["randomized_error_gap"] for item in selected])
        e = np.asarray([ci(item["randomized_error_gap_std"], item["randomized_error_gap_count"]) for item in selected])
        ax.errorbar(x, y, yerr=e, fmt=markers[model], color=colors[model], capsize=2.5, linewidth=1.4, label=model)
    finite_x = np.asarray([item["bound"] for item in covered])
    finite_y = np.asarray([item["randomized_error_gap"] for item in covered])
    lo, hi = 0.0, float(max(finite_x.max(), finite_y.max(), 1e-3)) * 1.08
    ax.plot([lo, hi], [lo, hi], color=COLORS["gray"], linestyle="--", linewidth=1, label="$y=x$")
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_xlabel(r"Certificate $\sqrt{\chi^2 L_{\rm CF}^{\rm pred}/2}$", fontsize=8)
    ax.set_ylabel("Expected zero-one risk gap", fontsize=8)
    ax.set_title("(a) Covered points versus certificate", fontsize=9, loc="left", weight="bold")
    style_axes(ax)
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1]
    for model in ["full", "partial"]:
        selected = [
            item
            for item in summary
            if item["model"] == model
            and item["path"] not in {
                "support_escape",
                "support_escape_full",
                "support_escape_partial",
                "unavailable_full",
                "unavailable_partial",
            }
        ]
        selected.sort(key=lambda item: item["chi_square"])
        x = np.asarray([item["chi_square"] for item in selected])
        y = np.asarray([item["randomized_error_gap"] for item in selected])
        e = np.asarray([ci(item["randomized_error_gap_std"], item["randomized_error_gap_count"]) for item in selected])
        b = np.asarray([item["bound"] for item in selected])
        ax.errorbar(x, y, yerr=e, color=colors[model], marker=markers[model], linewidth=1.5, capsize=2.5, label=f"risk gap, {model}")
        ax.plot(x, b, color=colors[model], linestyle="--", linewidth=1.1, alpha=0.8, label=f"bound, {model}")
    ax.set_xlabel(r"Exact covered $χ^2(P_e\Vert P_{\rm ref})$", fontsize=8)
    ax.set_ylabel("Expected zero-one risk gap / bound", fontsize=8)
    ax.set_title("(b) Shift path and finite certificate", fontsize=9, loc="left", weight="bold")
    style_axes(ax)
    ax.legend(frameon=False, fontsize=6.5, ncol=2)
    fig.suptitle("Predictive-KL transfer certificate on the real LLM diagnostic pool", fontsize=10, weight="bold")
    save_figure(fig, figure_dir, "fig_c6_transfer_certificate")


# ---------------------------------------------------------------------------
# Theorem 3: intervention-mixture phase boundary


def nll(result: Dict) -> float:
    probabilities = np.asarray(result["probabilities"], dtype=float)
    targets = np.asarray(result["targets"], dtype=int)
    return float(-np.mean(np.log(np.clip(probabilities[np.arange(len(targets)), targets], 1e-12, 1.0))))


def phase_summary(phase_rows: Sequence[Dict], factual_rows: Sequence[Dict]) -> List[Dict]:
    factual_by_seed = {row["seed"]: row for row in factual_rows}
    groups = defaultdict(list)
    for row in phase_rows:
        factual = factual_by_seed.get(row["seed"])
        if factual is None:
            continue
        cf_nll = nll(row["mismatched"])
        factual_nll = nll(factual["mismatched"])
        groups[(row["lambda_cf"], row["eta"])].append({
            "delta_nll": cf_nll - factual_nll,
            "delta_accuracy": float(row["mismatched"]["accuracy"] - factual["mismatched"]["accuracy"]),
            "cf_accuracy": float(row["mismatched"]["accuracy"]),
            "cf_nll": cf_nll,
        })
    output = []
    for (lambda_cf, eta), members in sorted(groups.items()):
        item = {"lambda_cf": lambda_cf, "eta": eta}
        for metric in ["delta_nll", "delta_accuracy", "cf_accuracy", "cf_nll"]:
            mu, sd, count = mean_std([member[metric] for member in members])
            item[metric] = mu
            item[metric + "_std"] = sd
            item[metric + "_count"] = count
        output.append(item)
    return output


def first_crossover(summary: Sequence[Dict], lambda_cf: float) -> float:
    clean = [item for item in summary if abs(item["lambda_cf"] - lambda_cf) < 1e-8]
    clean.sort(key=lambda item: item["eta"])
    for item in clean:
        if item["delta_nll"] >= 0:
            return float(item["eta"])
    return float("nan")


def plot_phase(summary: Sequence[Dict], figure_dir: pathlib.Path) -> None:
    lambdas = sorted(set(item["lambda_cf"] for item in summary))
    etas = sorted(set(item["eta"] for item in summary))
    matrix = np.asarray([[next(item["delta_nll"] for item in summary if abs(item["lambda_cf"] - lam) < 1e-8 and abs(item["eta"] - eta) < 1e-8) for lam in lambdas] for eta in etas])
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.0), constrained_layout=True)

    ax = axes[0]
    limit = float(max(abs(matrix.min()), abs(matrix.max()), 1e-3))
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto", origin="lower")
    ax.set_xticks(np.arange(len(lambdas)))
    ax.set_xticklabels([f"{lam:.2f}" for lam in lambdas])
    ax.set_yticks(np.arange(len(etas)))
    ax.set_yticklabels([f"{eta:.2f}" for eta in etas])
    ax.set_xlabel(r"Counterfactual fraction $\lambda_{\rm cf}$", fontsize=8)
    ax.set_ylabel(r"Injected corruption $\eta$", fontsize=8)
    ax.set_title("(a) OOD NLL difference to factual", fontsize=9, loc="left", weight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label("CFPT NLL $-$ factual NLL", fontsize=7)

    ax = axes[1]
    for color, lam, marker in [(COLORS["blue"], lambdas[0], "o"), (COLORS["orange"], lambdas[-1], "s")]:
        selected = [item for item in summary if abs(item["lambda_cf"] - lam) < 1e-8]
        selected.sort(key=lambda item: item["eta"])
        x = np.asarray([item["eta"] for item in selected])
        y = np.asarray([item["delta_nll"] for item in selected])
        e = np.asarray([ci(item["delta_nll_std"], item["delta_nll_count"]) for item in selected])
        ax.errorbar(x * 100, y, yerr=e, color=color, marker=marker, linewidth=1.8, capsize=2.5, label=fr"$\lambda_{{\rm cf}}={lam:.2f}$")
    ax.axhline(0, color=COLORS["gray"], linestyle="--", linewidth=1)
    ax.set_xlabel("Injected corruption rate (%)", fontsize=8)
    ax.set_ylabel("OOD NLL difference", fontsize=8)
    ax.set_title("(b) Empirical benefit--harm diagnostic", fontsize=9, loc="left", weight="bold")
    style_axes(ax)
    ax.legend(frameon=False, fontsize=7)
    fig.suptitle("Operational phase-boundary check on the real LLM", fontsize=10, weight="bold")
    save_figure(fig, figure_dir, "fig_c7_phase_boundary")


def csv_write(path: pathlib.Path, rows: Sequence[Dict], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in fields} for row in rows])


def latex_table(rows: Sequence[str], columns: str) -> str:
    return "\\begin{tabular}{" + columns + "}\n\\toprule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}"


"""
def write_latex(
    path: pathlib.Path,
    cluster_cells: Sequence[Dict],
    cluster_stats: Dict[str, float],
    budget_rows_data: Sequence[Dict],
    transfer: Sequence[Dict],
    phase: Sequence[Dict],
) -> None:
    cluster_rows = [r"$n$ & $K$ & bootstrap var. & two-level fit & naive fit \", r"\midrule"]
    for cell in cluster_cells:
        cluster_rows.append(f"{cell['n_eval']} & {cell['k_eval']} & {fmt_pm(cell['variance'], cell['std'])} & {fmt(cell['two_level_fit'])} & {fmt(cell['naive_fit'])} " + r"\")

    selected_transfer = []
    for model, path_name, deltas in [
        ("full", "covered_shift", [0.0, 0.50, 1.0]),
        ("full", "unavailable_full", [1.0]),
        ("partial", "covered_shift_partial", [0.0, 0.50, 1.0]),
        ("partial", "support_escape", [0.25, 1.0]),
    ]:
        selected_transfer.extend([item for item in transfer if item["model"] == model and item["path"] == path_name and any(abs(item["delta"] - delta) < 1e-8 for delta in deltas)])
    transfer_rows = [r"path & $δ$ & $χ^2$ & $L_{m CF}^{m pred}$ & certificate & acc. gap \", r"\midrule"]
    for item in sorted(selected_transfer, key=lambda x: (x["model"], x["path"], x["delta"])):
        path_label = item["path"].replace("_", " ") + " (" + item["model"] + ")"
        chi = r"$\infty$" if not np.isfinite(item["chi_square"]) else fmt(item["chi_square"], 2)
        bound = r"$\infty$" if not np.isfinite(item["bound"]) else fmt(item["bound"], 3)
        transfer_rows.append(f"{path_label} & {item['delta']:.2f} & {chi} & {fmt(item['l_cf'])} & {bound} & {fmt_pm(item['accuracy_gap'], item['accuracy_gap_std'])} " + r"\")

    phase_rows = [r"$λ_{m cf}$ & $η$ & OOD NLL gap & OOD acc. gap \", r"\midrule"]
    selected_phase = [item for item in phase if abs(item["eta"] - 0.0) < 1e-8 or abs(item["eta"] - 0.08) < 1e-8 or abs(item["eta"] - 0.16) < 1e-8]
    for item in sorted(selected_phase, key=lambda x: (x["lambda_cf"], x["eta"])):
        phase_rows.append(f"{item['lambda_cf']:.2f} & {item['eta']:.2f} & {fmt_pm(item['delta_nll'], item['delta_nll_std'])} & {fmt_pm(item['delta_accuracy'], item['delta_accuracy_std'])} " + r"\")

    half_cross = first_crossover(phase, 0.5)
    full_cross = first_crossover(phase, 1.0)
    half_text = "not observed" if not np.isfinite(half_cross) else f"{100 * half_cross:.0f}\\%"
    full_text = "not observed" if not np.isfinite(full_cross) else f"{100 * full_cross:.0f}\\%"
    lines = [
        "% Auto-generated from strict theorem-aligned Appendix C artifacts.",
        f"\\newcommand{{\\CClusterA}}{{{fmt(cluster_stats['a'])}}}",
        f"\\newcommand{{\\CClusterB}}{{{fmt(cluster_stats['b'])}}}",
        f"\\newcommand{{\\CClusterRtwo}}{{{fmt(cluster_stats['r2'], 2)}}}",
        f"\\newcommand{{\\CClusterRnaive}}{{{fmt(cluster_stats['naive_r2'], 2)}}}",
        f"\\newcommand{{\\CClusterModels}}{{{cluster_stats['n_models']}}}",
        f"\\newcommand{{\\CPhaseCrossHalf}}{{{half_text}}}",
        f"\\newcommand{{\\CPhaseCrossFull}}{{{full_text}}}",
        "\\newcommand{\\CClusterTable}{%\n" + latex_table(cluster_rows, "rrrrr") + "\n}",
        "\\newcommand{\\CTransferTable}{%\n" + latex_table(transfer_rows, "lrrrrr") + "\n}",
        "\\newcommand{\\CPhaseTable}{%\n" + latex_table(phase_rows, "rrrr") + "\n}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


"""

def write_latex(
    path: pathlib.Path,
    cluster_cells: Sequence[Dict],
    cluster_stats: Dict[str, float],
    budget_rows_data: Sequence[Dict],
    transfer: Sequence[Dict],
    phase: Sequence[Dict],
) -> None:
    bs = chr(92)
    row_end = bs * 2
    cluster_rows = [
        "$n$ & $K$ & bootstrap var. & anchor-only $a/n$ & two-level fit & naive fit " + row_end,
        bs + "midrule",
    ]
    for cell in cluster_cells:
        anchor_fit = cluster_stats["a"] / cell["n_eval"]
        cluster_rows.append(
            f"{cell['n_eval']} & {cell['k_eval']} & "
            f"{fmt_pm(cell['variance'], cell['std'])} & "
            f"{fmt(anchor_fit, 4)} & {fmt(cell['two_level_fit'], 4)} & {fmt(cell['naive_fit'], 4)} "
            + row_end
        )

    budget_rows = [
        "$n$ & $K$ & total & diagnostic acc. & matched acc. " + row_end,
        bs + "midrule",
    ]
    for item in budget_rows_data:
        budget_rows.append(
            f"{item['n']} & {item['k']} & {item['budget']} & "
            f"{fmt_pm(item['diagnostic_accuracy'], item['diagnostic_std'])} & "
            f"{fmt_pm(item['matched_accuracy'], item['matched_std'])} "
            + row_end
        )

    selected_transfer = []
    for model, path_name, deltas in [
        ("full", "covered_shift", [0.0, 0.50, 1.0]),
        ("full", "support_escape_full", [1.0]),
        ("partial", "covered_shift_partial", [0.0, 0.50, 1.0]),
        ("partial", "unavailable_partial", [1.0]),
        ("partial", "support_escape", [0.25, 1.0]),
    ]:
        selected_transfer.extend(
            [
                item
                for item in transfer
                if item["model"] == model
                and item["path"] == path_name
                and any(abs(item["delta"] - delta) < 1e-8 for delta in deltas)
            ]
        )
    transfer_rows = [
        "path & $" + bs + "delta$ & $" + bs + "chi^2$ & $"
        + "L_{"
        + bs
        + "rm CF}^{"
        + bs
        + "rm pred}$ & KL cert. & centered cert. & expected-01 gap "
        + row_end,
        bs + "midrule",
    ]
    for item in sorted(selected_transfer, key=lambda x: (x["model"], x["path"], x["delta"])):
        path_label = item["path"].replace("_", " ") + " (" + item["model"] + ")"
        unavailable = item["path"].startswith("unavailable")
        chi = (
            "--"
            if unavailable
            else ("$" + bs + "infty$" if not np.isfinite(item["chi_square"]) else fmt(item["chi_square"], 2))
        )
        bound = (
            "--"
            if unavailable
            else ("$" + bs + "infty$" if not np.isfinite(item["bound"]) else fmt(item["bound"], 3))
        )
        centered = (
            "--"
            if unavailable
            else ("$" + bs + "infty$" if not np.isfinite(item["centered_bound"]) else fmt(item["centered_bound"], 3))
        )
        gap = (
            fmt_pm(item["randomized_error_gap"], item["randomized_error_gap_std"])
            if np.isfinite(item["randomized_error_gap"])
            else "--"
        )
        transfer_rows.append(
            f"{path_label} & {item['delta']:.2f} & {chi} & {fmt(item['l_cf'])} "
            f"& {bound} & {centered} & {gap} "
            + row_end
        )

    phase_rows = [
        "$" + bs + "lambda_{" + bs + "rm cf}$ & $" + bs + "eta$ & "
        "OOD NLL gap & OOD acc. gap "
        + row_end,
        bs + "midrule",
    ]
    selected_phase = [
        item
        for item in phase
        if abs(item["eta"] - 0.0) < 1e-8
        or abs(item["eta"] - 0.08) < 1e-8
        or abs(item["eta"] - 0.16) < 1e-8
    ]
    for item in sorted(selected_phase, key=lambda x: (x["lambda_cf"], x["eta"])):
        phase_rows.append(
            f"{item['lambda_cf']:.2f} & {item['eta']:.2f} & "
            f"{fmt_pm(item['delta_nll'], item['delta_nll_std'])} & "
            f"{fmt_pm(item['delta_accuracy'], item['delta_accuracy_std'])} "
            + row_end
        )

    half_cross = first_crossover(phase, 0.5)
    full_cross = first_crossover(phase, 1.0)
    half_text = "not observed" if not np.isfinite(half_cross) else f"{100 * half_cross:.0f}{bs}%"
    full_text = "not observed" if not np.isfinite(full_cross) else f"{100 * full_cross:.0f}{bs}%"
    finite_transfer = [item for item in transfer if np.isfinite(item["bound"])]
    worst_transfer = min(
        [item["coverage_fraction"] for item in finite_transfer if np.isfinite(item["coverage_fraction"])],
        default=float("nan"),
    )

    def transfer_value(model: str, path_name: str, delta: float, field: str) -> float:
        for item in transfer:
            if (
                item["model"] == model
                and item["path"] == path_name
                and abs(item["delta"] - delta) < 1e-8
            ):
                return float(item[field])
        return float("nan")

    def macro(name: str, value: object) -> str:
        return bs + "newcommand{" + bs + name + "}{" + str(value) + "}"

    lines = [
        "% Auto-generated from strict theorem-aligned Appendix C artifacts.",
        macro("CClusterA", fmt(cluster_stats["a"])),
        macro("CClusterB", fmt(cluster_stats["b"])),
        macro("CClusterRtwo", fmt(cluster_stats["r2"], 2)),
        macro("CClusterRnaive", fmt(cluster_stats["naive_r2"], 2)),
        macro("CClusterModels", cluster_stats["n_models"]),
        macro("CPhaseCrossHalf", half_text),
        macro("CPhaseCrossFull", full_text),
        macro("CTransferWorstCoverage", fmt(worst_transfer, 2)),
        macro("CTransferFiniteCells", len(finite_transfer)),
        macro("CTransferFullHalfPredCert", fmt(transfer_value("full", "covered_shift", 0.50, "bound"))),
        macro("CTransferFullHalfCenteredCert", fmt(transfer_value("full", "covered_shift", 0.50, "centered_bound"))),
        macro("CTransferFullHalfGap", fmt(transfer_value("full", "covered_shift", 0.50, "randomized_error_gap"))),
        macro("CTransferPartialHalfPredCert", fmt(transfer_value("partial", "covered_shift_partial", 0.50, "bound"))),
        macro("CTransferPartialHalfCenteredCert", fmt(transfer_value("partial", "covered_shift_partial", 0.50, "centered_bound"))),
        macro("CTransferPartialHalfGap", fmt(transfer_value("partial", "covered_shift_partial", 0.50, "randomized_error_gap"))),
        bs + "newcommand{" + bs + "CClusterTable}{%\n"
         + latex_table(cluster_rows, "rrrrrr")
        + "\n}",
        bs + "newcommand{" + bs + "CClusterBudgetTable}{%\n"
        + latex_table(budget_rows, "rrrrr")
        + "\n}",
        bs + "newcommand{" + bs + "CTransferTable}{%\n"
        + latex_table(transfer_rows, "lrrrrrr")
        + "\n}",
        bs + "newcommand{" + bs + "CPhaseTable}{%\n"
        + latex_table(phase_rows, "rrrr")
        + "\n}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aligned-dir", type=pathlib.Path, required=True)
    parser.add_argument("--coverage-dir", type=pathlib.Path, required=True)
    parser.add_argument("--phase-dir", type=pathlib.Path, required=True)
    parser.add_argument("--result-dir", type=pathlib.Path, required=True)
    parser.add_argument("--figure-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()

    aligned_rows = completed(args.aligned_dir)
    coverage_rows = completed(args.coverage_dir)
    phase_rows = completed(args.phase_dir)
    factual_rows = [row for row in completed(args.result_dir) if row.get("kind") == "factual"]
    if not aligned_rows or not coverage_rows or not phase_rows or not factual_rows:
        raise RuntimeError("strict analysis requires completed aligned, coverage, phase, and factual artifacts")

    cluster_cells, cluster_stats = cluster_scaling(aligned_rows)
    budget_rows = fixed_budget_slice(aligned_rows)
    transfer_records = flatten_transfer(coverage_rows)
    transfer = transfer_summary(transfer_records)
    phase = phase_summary(phase_rows, factual_rows)

    args.result_dir.mkdir(parents=True, exist_ok=True)
    csv_write(args.result_dir / "strict_cluster_scaling.csv", cluster_cells, ["n_eval", "k_eval", "variance", "std", "two_level_fit", "naive_fit", "count"])
    csv_write(args.result_dir / "strict_transfer_certificate.csv", transfer, ["model", "path", "delta", "chi_square", "l_cf", "bound", "v_nu", "centered_bound", "accuracy_gap", "nll_gap", "randomized_error_gap", "supported", "coverage_fraction"])
    csv_write(args.phase_dir / "phase_summary.csv", phase, ["lambda_cf", "eta", "delta_nll", "delta_nll_std", "delta_accuracy", "delta_accuracy_std", "cf_accuracy", "cf_nll"])
    csv_write(args.result_dir / "strict_fixed_budget.csv", budget_rows, ["n", "k", "budget", "diagnostic_accuracy", "diagnostic_std", "matched_accuracy", "matched_std", "count"])
    write_latex(args.result_dir / "appendix_c_strict_generated.tex", cluster_cells, cluster_stats, budget_rows, transfer, phase)
    plot_cluster_scaling(cluster_cells, cluster_stats, args.figure_dir)
    plot_transfer(transfer, args.figure_dir)
    plot_phase(phase, args.figure_dir)
    print(json.dumps({
        "aligned_models": len(aligned_rows),
        "coverage_runs": len(coverage_rows),
        "phase_runs": len(phase_rows),
        "cluster_a": cluster_stats["a"],
        "cluster_b": cluster_stats["b"],
        "cluster_r2": cluster_stats["r2"],
        "cluster_naive_r2": cluster_stats["naive_r2"],
        "transfer_rows": len(transfer),
        "phase_rows": len(phase),
    }, indent=2))


if __name__ == "__main__":
    main()
