"""P0 audit for shift-class granularity on locked Qwen predictions.

This script is intentionally diagnostic rather than confirmatory.  It consumes
only the locked verified-text prediction files and makes the granularity of
the nuisance adversary explicit:

* global: one categorical nuisance law shared by every anchor;
* group: one nuisance law per task-label group;
* task: one nuisance law per anchor.

The three classes are nested.  For a fixed candidate-vs-factual loss
difference, each class is summarized by the same chi-square Cauchy--Schwarz
certificate, while the realized shift path is evaluated exactly on the
observed categorical nuisance responses.  This is not presented as a proof
of the population theorem or as a new training result.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1] / "llm_cfpt" / "verified_results"
OUT = Path(__file__).resolve().parent / "p0_outputs"
TAGS = ["amber archive", "cedar archive", "violet archive"]
CONDITIONS = ["cf_0", "cf_20", "cf_40"]
SIZES = ["0p5b", "1p5b"]
TASKS = ["attributes", "relations"]
SEEDS = [2027, 2028, 2029]
RHO_VALUES = [0.05, 0.10, 0.25, 0.50, 1.00]
DELTA_VALUES = [0.0, 0.25, 0.50, 0.75, 1.0]
Q = np.full(3, 1.0 / 3.0)


def read(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_rows(result: Dict) -> Dict[str, Dict]:
    ev = result["evaluation"]
    split = np.asarray(ev["splits"])
    mask = split == "test"
    anchors = np.asarray(ev["anchor_ids"])[mask]
    tags = np.asarray(ev["genres"])[mask]
    targets = np.asarray(ev["targets"], dtype=int)[mask]
    probs = np.asarray(ev["probabilities"], dtype=float)[mask]
    out = {}
    for anchor in sorted(set(anchors.tolist())):
        ix = np.flatnonzero(anchors == anchor)
        order = [int(ix[np.flatnonzero(tags[ix] == tag)[0]]) for tag in TAGS]
        out[str(anchor)] = {
            "anchor_id": str(anchor),
            "tags": TAGS,
            "target": int(targets[order[0]]),
            "probs": probs[order],
        }
    if len(out) != 256 or any(x["probs"].shape != (3, 3) for x in out.values()):
        raise ValueError(f"expected 256 anchors x 3 tags, got {len(out)}")
    return out


def loss_difference(candidate: Dict[str, Dict], factual: Dict[str, Dict]) -> Dict[str, Dict]:
    out = {}
    if set(candidate) != set(factual):
        raise ValueError("candidate and factual test anchors do not align")
    for anchor in candidate:
        c, f = candidate[anchor], factual[anchor]
        if c["target"] != f["target"]:
            raise ValueError("candidate/factual targets do not align")
        y = int(c["target"])
        dc = 1.0 - np.asarray(c["probs"])[:, y]
        df = 1.0 - np.asarray(f["probs"])[:, y]
        out[anchor] = {
            "target": y,
            "d": dc - df,
            "candidate_loss": dc,
            "factual_loss": df,
        }
    return out


def summarize_values(values: np.ndarray, rho: float) -> Tuple[float, float, float]:
    """Return mean, variance under Q, and CS chi-square upper certificate."""
    values = np.asarray(values, dtype=float)
    mu = float(np.dot(Q, values))
    var = float(np.dot(Q, (values - mu) ** 2))
    cert = float(mu + math.sqrt(max(0.0, rho * var)))
    return mu, var, cert


def exact_path(values: np.ndarray, delta: float, style: int = 0) -> float:
    p = (1.0 - delta) * Q.copy()
    p[style] += delta
    return float(np.dot(p, values))


def records_for(diff: Dict[str, Dict], size: str, task: str, seed: int, condition: str) -> List[Dict]:
    rows = []
    for rho in RHO_VALUES:
        global_values = np.asarray([x["d"] for x in diff.values()])
        mu, var, cert = summarize_values(global_values.mean(axis=0), rho)
        rows.append({
            "size": size, "task": task, "seed": seed, "condition": condition,
            "class": "global", "rho": rho, "estimate": mu, "variance": var,
            "certificate": cert, "n_groups": len(diff),
        })
        # A group is the clean task label.  The law can differ by label group,
        # but is shared by all anchors in that group.
        for class_name, key_fn in [("group", lambda a, x: str(x["target"])),
                                   ("task", lambda a, x: a)]:
            groups = defaultdict(list)
            for anchor, x in diff.items():
                groups[key_fn(anchor, x)].append(np.asarray(x["d"], dtype=float))
            stats = [summarize_values(np.mean(np.stack(v), axis=0), rho) for v in groups.values()]
            rows.append({
                "size": size, "task": task, "seed": seed, "condition": condition,
                "class": class_name, "rho": rho,
                "estimate": float(np.mean([s[0] for s in stats])),
                "variance": float(np.mean([s[1] for s in stats])),
                "certificate": float(np.mean([s[2] for s in stats])),
                "n_groups": len(stats),
            })
    for delta in DELTA_VALUES:
        for style in range(3):
            actual_global = float(np.mean([exact_path(x["d"], delta, style) for x in diff.values()]))
            groups = defaultdict(list)
            for anchor, x in diff.items():
                groups[int(x["target"])].append(np.asarray(x["d"], dtype=float))
            actual_group = float(np.mean([
                exact_path(np.mean(np.stack(v), axis=0), delta, style) for v in groups.values()
            ]))
            actual_task = float(np.mean([exact_path(x["d"], delta, style) for x in diff.values()]))
            # global and task have the same mean on a common shift path; they
            # are retained separately because their certificates differ.
            for cls, value in [("global", actual_global), ("group", actual_group), ("task", actual_task)]:
                rows.append({
                    "size": size, "task": task, "seed": seed, "condition": condition,
                    "class": cls, "delta": delta, "style": TAGS[style],
                    "actual_shift": value,
                })
    return rows


def aggregate(records: List[Dict]) -> List[Dict]:
    groups = defaultdict(list)
    for r in records:
        if "rho" in r:
            groups[(r["size"], r["task"], r["condition"], r["class"], r["rho"])].append(r)
    out = []
    for key, vals in sorted(groups.items()):
        size, task, condition, cls, rho = key
        out.append({
            "size": size, "task": task, "condition": condition, "class": cls, "rho": rho,
            "estimate": float(np.mean([x["estimate"] for x in vals])),
            "certificate": float(np.mean([x["certificate"] for x in vals])),
            "certificate_sd": float(np.std([x["certificate"] for x in vals], ddof=1)),
            "n": len(vals),
        })
    return out


def write_csv(path: Path, rows: Iterable[Dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def style(ax):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#dddddd", linewidth=.55); ax.set_axisbelow(True)
    ax.tick_params(labelsize=7)


def plot(agg: List[Dict], path: Path) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42,
                         "ps.fonttype": 42, "axes.titlesize": 9, "axes.labelsize": 8})
    fig, ax = plt.subplots(1, 3, figsize=(7.15, 2.55), constrained_layout=True)
    colors = {"global": "#0072B2", "group": "#D55E00", "task": "#009E73"}
    labels = {"global": "Global class", "group": "Label-group class", "task": "Per-anchor class"}
    # Pool all settings to make the diagnostic about the class, not a selected
    # task.  Factual/candidate effects are shown as faint points behind means.
    for cls in ["global", "group", "task"]:
        rr = [x for x in agg if x["class"] == cls and abs(float(x["rho"]) - .25) < 1e-8]
        by = defaultdict(list)
        for x in rr: by[(x["size"], x["task"], x["condition"])].append(x)
        means = [float(np.mean([x["certificate"] for x in v])) for v in by.values()]
        ests = [float(np.mean([x["estimate"] for x in v])) for v in by.values()]
        ax[0].scatter(ests, means, s=22, alpha=.75, color=colors[cls], label=labels[cls])
        ax[1].bar(cls, float(np.mean(means)), color=colors[cls], alpha=.85)
    lims = ax[0].get_xlim()
    ax[0].plot(lims, lims, color="#555555", linestyle="--", linewidth=.8)
    ax[0].set(xlabel="Observed reference effect", ylabel=r"$χ^2$ certificate at $\rho=0.25$",
              title="(a) Certificate slack")
    ax[0].legend(frameon=False, fontsize=6.5, loc="upper left")
    ax[1].set(xlabel="Shift class", ylabel="Mean certificate", title="(b) Class complexity")
    ax[1].set_xticklabels(["global", "group", "task"], rotation=15)
    # Realized shift path for one fixed, pre-specified direction; all settings
    # are pooled, and the individual task/size/candidate trajectories remain in
    # the CSV for audit.
    # Plot one pooled trajectory per candidate condition.  The earlier
    # development version drew one line per size/task combination, which
    # obscured rather than clarified the finite-pool diagnostic.
    for condition in CONDITIONS:
        by_delta = defaultdict(list)
        for x in [r for r in all_records if r.get("condition") == condition
                  and r.get("class") == "global" and r.get("style") == TAGS[0]]:
            by_delta[float(x["delta"])].append(x["actual_shift"])
        xs = sorted(by_delta)
        ax[2].plot(xs, [np.mean(by_delta[x]) for x in xs], marker="o", linewidth=1.3,
                   label=condition.replace("cf_", "eta="))
    ax[2].axhline(0, color="#555555", linestyle=":", linewidth=.8)
    ax[2].set(xlabel="Prespecified mixture shift $\delta$", ylabel="Observed effect", title="(c) Held-out shift path")
    ax[2].legend(frameon=False, fontsize=6.5, ncol=1)
    for a in ax: style(a)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=.03)
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=.03)
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight", pad_inches=.03)
    plt.close(fig)


def main() -> None:
    global all_records, records_cache
    all_records = []
    for size in SIZES:
        for task in TASKS:
            for seed in SEEDS:
                prefix = f"{size}_{task}_{seed}_"
                factual = test_rows(read(ROOT / f"{prefix}factual.json"))
                for condition in CONDITIONS:
                    cand = test_rows(read(ROOT / f"{prefix}{condition}.json"))
                    diff = loss_difference(cand, factual)
                    all_records.extend(records_for(diff, size, task, seed, condition))
    records_cache = all_records
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "p0_all_records.csv", all_records)
    agg = aggregate(all_records)
    write_csv(OUT / "p0_aggregate.csv", agg)
    plot(agg, OUT / "fig_p0_shift_granularity")
    manifest = {
        "source": str(ROOT), "n_conditions": len(SIZES) * len(TASKS) * len(SEEDS) * len(CONDITIONS),
        "test_anchors_per_run": 256, "nuisance_tags": TAGS,
        "classes": {
            "global": "one nuisance law shared by all anchors",
            "group": "one nuisance law per observed target-label group",
            "task": "one nuisance law per anchor",
        },
        "rho_values": RHO_VALUES, "delta_values": DELTA_VALUES,
        "loss": "candidate expected-zero-one minus factual expected-zero-one",
        "claim_scope": "locked finite-pool diagnostic; not a population theorem test",
    }
    (OUT / "p0_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print("records", len(all_records), "aggregate", len(agg))


if __name__ == "__main__":
    main()
