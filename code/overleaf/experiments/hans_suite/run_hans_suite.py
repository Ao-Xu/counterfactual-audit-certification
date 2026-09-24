"""HANS-aligned empirical suite for the comparative-utility paper.

This script is deliberately self-contained and auditable.  It uses the locked
MultiNLI anchor pool already present in the project, the public HANS train and
evaluation files, a frozen ANLI R1/R2/R3 development pool, and a local RoBERTa
checkpoint.  It does not select HANS or ANLI subsets after reading model
outcomes: all sampling rules are fixed below.

Conditions
----------
factual: the existing factual anchors only;
natural_cf: factual anchors plus the locked k=1 generated siblings;
hans_balanced: factual anchors plus a balanced HANS intervention pool;
aligned_corrupt: 20% of HANS non-entailment challenge rows are flipped to
                  entailment, reinforcing the overlap/subsequence heuristic;
reversed_corrupt: 20% of HANS entailment rows are flipped to non-entailment,
                  pushing in the opposite direction.

The last two conditions have the same corruption rate and matched pool size.
Outputs are CSV/JSON plus publication-oriented figures; all figures are built
only from the generated result tables.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "experiments" / "llm_cfpt" / "raw"
HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
HANS_TRAIN = HERE / "heuristics_train_set.txt"
HANS_EVAL = HERE / "heuristics_evaluation_set.txt"
ANLI_EVAL = HERE / "data" / "anli_dev_r123.jsonl"
MODEL = os.environ.get("CFPT_ROBERTA", "roberta-base")

LABELS = {"non-entailment": 0, "entailment": 1, "contradiction": 0, "neutral": 0}
GROUPS = ["lexical_overlap", "subsequence", "constituent"]
CONDITIONS = ["factual", "natural_cf", "hans_balanced", "aligned_corrupt", "reversed_corrupt"]


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_hans(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df["y"] = df["gold_label"].map(LABELS).astype(int)
    df["shortcut_label"] = (df["subcase"].str.startswith("e_")).astype(int)
    return df


def load_anli(path: Path) -> pd.DataFrame:
    """Load a frozen ANLI R1/R2/R3 dev pool as binary entailment test data."""
    rows = read_jsonl(path)
    df = pd.DataFrame(rows)
    df = df.rename(columns={"premise": "sentence1", "hypothesis": "sentence2"})
    # ANLI label 0 is entailment; neutral and contradiction are non-entailment.
    df["y"] = (df["label_id"].astype(int) == 0).astype(int)
    df["y_train"] = df["y"]
    return df


def balanced_hans_pool(df: pd.DataFrame, n_per_group: int, seed: int) -> pd.DataFrame:
    parts = []
    for group in GROUPS:
        g = df[df.heuristic == group].copy()
        # Match both gold labels and make the pool independent of model scores.
        take = max(1, n_per_group // 2)
        for y in [0, 1]:
            h = g[g.y == y].sample(n=take, random_state=seed + y + len(group))
            parts.append(h)
    out = pd.concat(parts, ignore_index=True)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def corrupt_hans(pool: pd.DataFrame, mode: str, rate: float, seed: int) -> pd.DataFrame:
    out = pool.copy()
    n = int(round(rate * len(out)))
    if mode == "aligned":
        # Flip challenge non-entailments to entailment: this agrees with the
        # shallow entailment default induced by the three HANS heuristics.
        eligible = out.index[(out.y == 0)].to_numpy()
    elif mode == "reversed":
        # Flip heuristic-compatible entailments to non-entailment.
        eligible = out.index[(out.y == 1)].to_numpy()
    else:
        raise ValueError(mode)
    if len(eligible) < n:
        raise RuntimeError(f"not enough eligible rows for {mode}: {len(eligible)} < {n}")
    rng = np.random.default_rng(seed)
    chosen = rng.choice(eligible, size=n, replace=False)
    out["y_train"] = out.y
    out.loc[chosen, "y_train"] = 1 - out.loc[chosen, "y"].astype(int)
    out["corrupted"] = False
    out.loc[chosen, "corrupted"] = True
    return out


def make_conditions(seed: int, hans_train: pd.DataFrame):
    factual = pd.DataFrame(read_jsonl(RAW / "train_factual.jsonl"))
    factual = factual.rename(columns={"premise": "sentence1", "hypothesis": "sentence2"})
    factual["y"] = factual.label.map(LABELS).astype(int)
    factual["y_train"] = factual.y
    factual["heuristic"] = "factual"
    factual["corrupted"] = False
    factual["source"] = "mnli_anchor"

    natural = pd.DataFrame(read_jsonl(RAW / "train_cf_n1000_k1.jsonl"))
    natural = natural.rename(columns={"premise": "sentence1", "hypothesis": "sentence2"})
    natural = natural[natural.generation_valid == True].copy()
    natural["y"] = natural.label.map(LABELS).astype(int)
    natural["y_train"] = natural.y
    natural["heuristic"] = "natural_cf"
    natural["corrupted"] = False
    natural["source"] = "locked_generated_sibling"

    pool = balanced_hans_pool(hans_train, n_per_group=600, seed=seed)
    pool["y_train"] = pool.y
    pool["corrupted"] = False
    pool["source"] = "hans_intervention"
    aligned = corrupt_hans(pool, "aligned", 0.20, seed + 11)
    aligned["source"] = "hans_aligned_corruption"
    reversed_ = corrupt_hans(pool, "reversed", 0.20, seed + 17)
    reversed_["source"] = "hans_reversed_corruption"

    def convert(base, extra=None):
        cols = ["sentence1", "sentence2", "y_train", "heuristic", "corrupted", "source"]
        a = base[[c for c in cols if c in base.columns]].copy()
        if extra is not None:
            b = extra[[c for c in cols if c in extra.columns]].copy()
            a = pd.concat([a, b], ignore_index=True)
        return a.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return {
        "factual": convert(factual),
        "natural_cf": convert(factual, natural),
        "hans_balanced": convert(factual, pool),
        "aligned_corrupt": convert(factual, aligned),
        "reversed_corrupt": convert(factual, reversed_),
    }


def encode(df: pd.DataFrame, tokenizer, max_len: int):
    enc = tokenizer(
        df.sentence1.tolist(), df.sentence2.tolist(),
        padding="max_length", truncation=True, max_length=max_len,
        return_tensors="pt",
    )
    y = torch.tensor(df.y_train.to_numpy(), dtype=torch.long)
    return TensorDataset(enc["input_ids"], enc["attention_mask"], y)


@torch.no_grad()
def predict(model, tokenizer, df: pd.DataFrame, device, max_len: int, batch_size: int = 32):
    if len(df) == 0:
        return np.empty((0, 2)), np.empty((0,), dtype=int)
    enc = tokenizer(
        df.sentence1.tolist(), df.sentence2.tolist(),
        padding="max_length", truncation=True, max_length=max_len,
        return_tensors="pt",
    )
    ds = TensorDataset(enc["input_ids"], enc["attention_mask"])
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    probs = []
    model.eval()
    for ids, mask in dl:
        out = model(input_ids=ids.to(device), attention_mask=mask.to(device))
        probs.append(torch.softmax(out.logits.float(), dim=-1).cpu().numpy())
    return np.concatenate(probs), df.y.to_numpy(dtype=int)


def train_one(train_df, hans_eval, mnli_eval, anli_eval, seed, condition, epochs, max_len, batch_size):
    seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=2, local_files_only=True, ignore_mismatched_sizes=True,
    ).to(device)
    ds = encode(train_df, tokenizer, max_len)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    model.train()
    for ep in range(epochs):
        for ids, mask, y in dl:
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda", dtype=torch.float16):
                logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
                loss = torch.nn.functional.cross_entropy(logits.float(), y.to(device))
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

    def eval_frame(frame, split):
        probs, y = predict(model, tokenizer, frame, device, max_len)
        pred = probs[:, 1] >= 0.5
        row = {
            "condition": condition, "seed": seed, "split": split,
            "n": int(len(y)), "accuracy": float((pred == y).mean()),
            "nll": float(log_loss(y, probs, labels=[0, 1])),
            "brier": float(np.mean((probs[:, 1] - y) ** 2)),
        }
        return row, probs, y

    # Pre-specified HANS evaluation subset: 500 examples per heuristic,
    # balanced by y, sampled by the seed rule.
    hes = []
    for i, g in enumerate(GROUPS):
        gdf = hans_eval[hans_eval.heuristic == g]
        for y in [0, 1]:
            hes.append(gdf[gdf.y == y].sample(n=250, random_state=7000 + seed + i + y))
    heval = pd.concat(hes, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    rows = []
    base, hp, hy = eval_frame(heval, "hans_all")
    rows.append(base)
    for g in GROUPS:
        sub = heval[heval.heuristic == g].copy()
        rr, _, _ = eval_frame(sub, f"hans_{g}")
        rows.append(rr)
    mr, _, _ = eval_frame(mnli_eval, "mnli_matched")
    rows.append(mr)
    ar, _, _ = eval_frame(anli_eval, "anli_dev_r123")
    rows.append(ar)
    group_acc = {r["split"]: r["accuracy"] for r in rows if r["split"].startswith("hans_") and r["split"] != "hans_all"}
    group_brier = {r["split"]: r["brier"] for r in rows if r["split"].startswith("hans_") and r["split"] != "hans_all"}
    ref = float(np.mean(list(group_acc.values())))
    ref_brier = float(np.mean(list(group_brier.values())))
    for r in rows:
        if r["split"].startswith("hans_") and r["split"] != "hans_all":
            r["reference_accuracy"] = ref
            r["group_gap"] = float(r["accuracy"] - ref)
            r["risk"] = float(r["brier"])
            r["reference_risk"] = float(ref_brier)
            r["risk_gap"] = float(r["risk"] - r["reference_risk"])
            r["reference_brier"] = ref_brier
    return rows


def write_figures(df: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.0)
    pal = {"factual": "#666666", "natural_cf": "#56B4E9", "hans_balanced": "#009E73", "aligned_corrupt": "#D55E00", "reversed_corrupt": "#0072B2"}
    order = CONDITIONS
    # Main high-information figure: group-risk slopes, bounded risk, and
    # matched-direction outcome.  Brier risk is bounded and therefore aligns
    # with the finite-risk certificate in the theory.
    sub = df[(df.split.str.startswith("hans_")) & (df.split != "hans_all")].copy()
    sub["group"] = sub.split.str.replace("hans_", "", regex=False)
    fig, ax = plt.subplots(1, 3, figsize=(12.4, 3.6), gridspec_kw={"width_ratios": [1.25, 1, .95]})
    for cond in order:
        s = sub[sub.condition == cond]
        x = s.groupby("group", sort=False).risk.mean().reindex(GROUPS)
        e = s.groupby("group", sort=False).risk.std().reindex(GROUPS).fillna(0).to_numpy() / max(1, math.sqrt(s.seed.nunique()))
        if x.isna().all():
            continue
        ax[0].errorbar(range(3), x.values, yerr=1.96*e, marker="o", lw=2.0, capsize=2.5, label=cond.replace("_", " ").title(), color=pal[cond])
    ax[0].set_xticks(range(3), ["Lexical", "Subsequence", "Constituent"])
    ax[0].set_ylabel("Brier risk (bounded)")
    ax[0].set_title("Heuristic-group risk")
    ax[0].legend(frameon=True, fontsize=6.5, loc="best")
    # Mean bounded risk with seed-level 95% uncertainty.
    g = sub.groupby("condition").risk.agg(["mean", "std", "count"]).reindex(order)
    ax[1].bar(range(len(order)), g["mean"].values, yerr=1.96*g["std"].fillna(0).values/np.sqrt(g["count"].values), capsize=3, color=[pal[c] for c in order], alpha=.9)
    ax[1].set_xticks(range(len(order)), ["Factual", "Natural\nCF", "HANS\nbalanced", "Aligned\nerror", "Reversed\nerror"], rotation=25, ha="right")
    ax[1].set_ylabel("Mean HANS Brier risk")
    ax[1].set_title("Matched corruption directions")
    # NLL carries the direction-sensitive signal that accuracy can hide.
    n = sub.groupby("condition").nll.agg(["mean", "std", "count"]).reindex(order)
    ax[2].bar(range(len(order)), n["mean"].values, yerr=1.96*n["std"].fillna(0).values/np.sqrt(n["count"].values), capsize=3, color=[pal[c] for c in order], alpha=.9)
    ax[2].set_xticks(range(len(order)), ["Factual", "Natural\nCF", "HANS\nbalanced", "Aligned\nerror", "Reversed\nerror"], rotation=25, ha="right")
    ax[2].set_ylabel("Mean HANS NLL")
    ax[2].set_title("Calibration-sensitive view")
    fig.suptitle("HANS diagnostic: group transfer and corruption direction", y=1.02, fontsize=11, weight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig_hans_main.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_hans_main.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "fig_hans_main.svg", bbox_inches="tight")
    plt.close(fig)

    # External-shift companion: the same trained models are evaluated on the
    # frozen ANLI dev pool.  This panel is intentionally separate from the
    # HANS family plot so that covered-family gains and external degradation
    # cannot be visually conflated.
    h = df[df.split == "hans_all"].copy()
    a = df[df.split == "anli_dev_r123"].copy()
    if not a.empty:
        fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.15), gridspec_kw={"width_ratios": [1.15, 1]})
        x = np.arange(len(order))
        width = 0.34
        for j, (frame, label, color, offset) in enumerate([(h, "HANS", "#009E73", -width / 2), (a, "ANLI dev", "#CC79A7", width / 2)]):
            g = frame.groupby("condition").brier.agg(["mean", "std", "count"]).reindex(order)
            err = 1.96 * g["std"].fillna(0).to_numpy() / np.sqrt(g["count"].to_numpy())
            ax[0].bar(x + offset, g["mean"].to_numpy(), width=width, yerr=err,
                      capsize=2.5, color=color, alpha=.9, label=label)
        ax[0].set_xticks(x, ["Factual", "Natural\nCF", "HANS\nbalanced", "Aligned\nerror", "Reversed\nerror"], rotation=25, ha="right")
        ax[0].set_ylabel("Brier risk (bounded)")
        ax[0].set_title("Covered vs. external shift")
        ax[0].legend(frameon=True, fontsize=7)

        piv = a.pivot(index="seed", columns="condition", values="brier")
        delta_order = ["natural_cf", "hans_balanced", "aligned_corrupt", "reversed_corrupt"]
        vals, errs = [], []
        for cond in delta_order:
            z = piv[cond] - piv["factual"]
            vals.append(float(z.mean()))
            errs.append(float(1.96 * z.std(ddof=1) / np.sqrt(z.size)))
        colors = [pal[c] for c in delta_order]
        ax[1].axhline(0, color="#333333", lw=1)
        ax[1].bar(np.arange(len(delta_order)), vals, yerr=errs, capsize=3, color=colors, alpha=.9)
        ax[1].set_xticks(np.arange(len(delta_order)), ["Natural\nCF", "HANS\nbalanced", "Aligned\nerror", "Reversed\nerror"], rotation=25, ha="right")
        ax[1].set_ylabel("ANLI Brier $-$ factual")
        ax[1].set_title("External shift gap")
        fig.suptitle("External-shift check: a covered gain need not transfer", y=1.03, fontsize=10.5, weight="bold")
        fig.tight_layout()
        fig.savefig(OUT / "fig_hans_external.pdf", bbox_inches="tight")
        fig.savefig(OUT / "fig_hans_external.png", dpi=300, bbox_inches="tight")
        fig.savefig(OUT / "fig_hans_external.svg", bbox_inches="tight")
        plt.close(fig)


def write_summary_table(df: pd.DataFrame):
    """Write the appendix table from the same frozen result CSV."""
    order = CONDITIONS
    labels = {
        "factual": "Factual", "natural_cf": "Natural CF",
        "hans_balanced": "HANS-balanced", "aligned_corrupt": "Aligned corruption",
        "reversed_corrupt": "Reversed corruption",
    }
    rows = []
    slash = chr(92)
    for condition in order:
        h = df[(df.condition == condition) & (df.split == "hans_all")]
        a = df[(df.condition == condition) & (df.split == "anli_dev_r123")]
        cells = []
        for metric, frame in [("brier", h), ("brier", a), ("nll", a), ("accuracy", a)]:
            x = frame[metric].to_numpy(dtype=float)
            err = 1.96 * x.std(ddof=1) / np.sqrt(x.size)
            cells.append(f"${x.mean():.3f} {slash}pm {err:.3f}$")
        rows.append(" & ".join([labels[condition]] + cells) + " " + slash + slash)
    (OUT / "hans_summary_table.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="2027,2028,2029,2030,2031")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    hans_train = load_hans(HANS_TRAIN)
    hans_eval = load_hans(HANS_EVAL)
    anli_eval = load_anli(ANLI_EVAL)
    mnli = pd.DataFrame(read_jsonl(RAW / "eval_matched.jsonl"))
    mnli = mnli.rename(columns={"premise": "sentence1", "hypothesis": "sentence2"})
    mnli["y"] = mnli.label.map(LABELS).astype(int)
    all_rows = []
    manifest = {
        "model": str(MODEL), "device": str(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
        "seeds": seeds, "epochs": args.epochs, "max_len": args.max_len, "batch_size": args.batch_size,
        "conditions": CONDITIONS, "hans_groups": GROUPS, "corruption_rate": 0.20,
        "hans_train_rows": int(len(hans_train)), "hans_eval_rows": int(len(hans_eval)),
        "anli_eval_rows": int(len(anli_eval)),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    for seed in seeds:
        conditions = make_conditions(seed, hans_train)
        for condition in CONDITIONS:
            print(f"[RUN] seed={seed} condition={condition}", flush=True)
            rows = train_one(conditions[condition], hans_eval, mnli, anli_eval, seed, condition, args.epochs, args.max_len, args.batch_size)
            all_rows.extend(rows)
            pd.DataFrame(all_rows).to_csv(OUT / "results_partial.csv", index=False)
    df = pd.DataFrame(all_rows)
    df.to_csv(OUT / "results.csv", index=False)
    manifest["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    manifest["rows"] = int(len(df))
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_figures(df)
    write_summary_table(df)
    print(df.to_string(index=False), flush=True)
    print(f"[DONE] outputs at {OUT}", flush=True)


if __name__ == "__main__":
    main()
