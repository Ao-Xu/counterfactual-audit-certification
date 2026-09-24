"""Controlled coverage and nuisance-shift experiment for Appendix C.

The experiment uses the fixed diagnostic counterfactual pool to separate two
cases in Theorem 2.  A full adapter is trained on all five target genres.  A
partial adapter is trained on only two target genres, with deterministic
cycling used only to keep the number of optimizer examples fixed.  Evaluation
then reweights the same held-out rows along (i) a covered shift path and (ii)
an explicit support-escape path.  The exact categorical chi-square between
the reference and test genre mixtures is known, so the plot does not use a
learned shift proxy as its x-axis.
"""

from __future__ import annotations

import argparse
import gc
import json
import pathlib
import time
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
from peft import PeftModel
from transformers import Trainer, TrainingArguments

from train_and_eval import (
    CompletionCollator,
    PromptDataset,
    build_model,
    evaluate_rows,
    load_base_model,
    read_jsonl,
    write_json,
)


ALL_GENRES = ["letters", "facetoface", "nineeleven", "oup", "verbatim"]
SUPPORTED_GENRES = ["letters", "facetoface"]
UNSUPPORTED_GENRES = [genre for genre in ALL_GENRES if genre not in SUPPORTED_GENRES]
SEEDS = [2027, 2028, 2029]


def add_diagnostic_fields(result: Dict, rows: Sequence[Dict]) -> Dict:
    result["anchor_ids"] = [row["anchor_id"] for row in rows]
    result["genres"] = [row.get("genre", "unknown") for row in rows]
    probabilities = np.asarray(result["probabilities"], dtype=float)
    targets = np.asarray(result["targets"], dtype=int)
    correct = np.asarray(result["predictions"], dtype=int) == targets
    losses = -np.log(np.clip(probabilities[np.arange(len(targets)), targets], 1e-12, 1.0))
    randomized_error = 1.0 - probabilities[np.arange(len(targets)), targets]
    result["nll"] = float(np.mean(losses))
    result["accuracy"] = float(np.mean(correct))
    result["randomized_error"] = float(np.mean(randomized_error))
    result["group_stats"] = group_stats(result)
    result["predictive_pairwise_kl"] = pairwise_kl(result)
    return result


def group_stats(result: Dict) -> Dict[str, Dict[str, float]]:
    genres = np.asarray(result["genres"])
    probabilities = np.asarray(result["probabilities"], dtype=float)
    targets = np.asarray(result["targets"], dtype=int)
    predictions = np.asarray(result["predictions"], dtype=int)
    output = {}
    for genre in sorted(set(genres.tolist())):
        indices = np.flatnonzero(genres == genre)
        losses = -np.log(
            np.clip(probabilities[indices, targets[indices]], 1e-12, 1.0)
        )
        randomized_error = 1.0 - probabilities[indices, targets[indices]]
        output[genre] = {
            "count": int(len(indices)),
            "accuracy": float(np.mean(predictions[indices] == targets[indices])),
            "nll": float(np.mean(losses)),
            "randomized_error": float(np.mean(randomized_error)),
        }
    return output


def pairwise_kl(result: Dict) -> float:
    probabilities = np.asarray(result["probabilities"], dtype=float)
    groups = defaultdict(list)
    for anchor_id, probability in zip(result["anchor_ids"], probabilities):
        groups[anchor_id].append(probability)
    values = []
    for members in groups.values():
        if len(members) < 2:
            continue
        for left in members:
            for right in members:
                if left is right:
                    continue
                values.append(
                    float(
                        np.sum(
                            left
                            * (
                                np.log(left + 1e-12)
                                - np.log(right + 1e-12)
                            )
                        )
                    )
                )
    return float(np.mean(values)) if values else float("nan")


def normalize(weights: Dict[str, float]) -> Dict[str, float]:
    total = float(sum(weights.values()))
    if total <= 0:
        raise ValueError("distribution weights must have positive mass")
    return {genre: float(value) / total for genre, value in weights.items()}


def chi_square(test: Dict[str, float], reference: Dict[str, float]) -> float:
    for genre, mass in test.items():
        if mass > 0 and reference.get(genre, 0.0) == 0:
            return float("inf")
    value = 0.0
    for genre in ALL_GENRES:
        p_e = float(test.get(genre, 0.0))
        p_ref = float(reference.get(genre, 0.0))
        if p_ref > 0:
            value += (p_e - p_ref) ** 2 / p_ref
    return float(value)


def weighted_metric(
    result: Dict, weights: Dict[str, float], metric: str
) -> float:
    stats = result["group_stats"]
    missing = [genre for genre, mass in weights.items() if mass > 0 and genre not in stats]
    if missing:
        raise ValueError(f"diagnostic pool has no rows for genres: {missing}")
    return float(sum(weights.get(genre, 0.0) * stats[genre][metric] for genre in weights))


def make_partial_rows(rows: Sequence[Dict], total: int = 1000) -> List[Dict]:
    support_rows = [row for row in rows if row.get("genre") in SUPPORTED_GENRES]
    if not support_rows:
        raise RuntimeError("no supported-genre rows available for partial adapter")
    output = []
    for index in range(total):
        row = dict(support_rows[index % len(support_rows)])
        row["partial_repeat_index"] = index // len(support_rows)
        output.append(row)
    return output


def train_partial(
    model_path: pathlib.Path,
    rows: Sequence[Dict],
    checkpoint_path: pathlib.Path,
    seed: int,
) -> tuple[object, object, float]:
    model, tokenizer = build_model(model_path, seed)
    dataset = PromptDataset(rows, tokenizer)
    bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    args = TrainingArguments(
        output_dir=str(checkpoint_path),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=1.0,
        learning_rate=2e-4,
        weight_decay=0.0,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=20,
        save_strategy="no",
        report_to=[],
        bf16=bf16,
        fp16=False,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        optim="adamw_torch",
        seed=seed,
        data_seed=seed,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=CompletionCollator(tokenizer),
    )
    output = trainer.train()
    model.save_pretrained(checkpoint_path, safe_serialization=True)
    loss = float(output.training_loss)
    del trainer
    return model, tokenizer, loss


def load_adapter(
    model_path: pathlib.Path, checkpoint_path: pathlib.Path, seed: int
) -> tuple[object, object]:
    base_model, tokenizer = load_base_model(model_path, seed)
    model = PeftModel.from_pretrained(base_model, checkpoint_path, is_trainable=False)
    return model, tokenizer


def mixture_record(
    result: Dict,
    model_name: str,
    path_name: str,
    reference: Dict[str, float],
    test: Dict[str, float],
    delta: float,
) -> Dict:
    reference = normalize(reference)
    test = normalize(test)
    ref_accuracy = weighted_metric(result, reference, "accuracy")
    test_accuracy = weighted_metric(result, test, "accuracy")
    return {
        "model": model_name,
        "path": path_name,
        "delta": float(delta),
        "reference": reference,
        "test": test,
        "chi_square": chi_square(test, reference),
        "reference_accuracy": ref_accuracy,
        "test_accuracy": test_accuracy,
        "accuracy_drop": float(ref_accuracy - test_accuracy),
        "reference_nll": weighted_metric(result, reference, "nll"),
        "test_nll": weighted_metric(result, test, "nll"),
        "reference_randomized_error": weighted_metric(result, reference, "randomized_error"),
        "test_randomized_error": weighted_metric(result, test, "randomized_error"),
        "randomized_error_gap": abs(
            weighted_metric(result, test, "randomized_error")
            - weighted_metric(result, reference, "randomized_error")
        ),
        "supported": all(
            mass <= 0 or reference.get(genre, 0.0) > 0
            for genre, mass in test.items()
        ),
    }


def build_paths(model_name: str, result: Dict) -> List[Dict]:
    uniform_all = normalize({genre: 1.0 for genre in ALL_GENRES})
    uniform_support = normalize({genre: 1.0 for genre in SUPPORTED_GENRES})
    uniform_escape = normalize({genre: 1.0 for genre in UNSUPPORTED_GENRES})
    paths = []
    if model_name == "full":
        for delta in [0.0, 0.10, 0.25, 0.50, 0.75, 1.0]:
            target = normalize(
                {
                    genre: (1.0 - delta) * uniform_all[genre]
                    + delta * (1.0 if genre == "letters" else 0.0)
                    for genre in ALL_GENRES
                }
            )
            paths.append(
                mixture_record(result, model_name, "covered_shift", uniform_all, target, delta)
            )
    else:
        for delta in [0.0, 0.25, 0.50, 0.75, 1.0]:
            target = normalize(
                {
                    genre: (1.0 - delta) * uniform_support
                    .get(genre, 0.0)
                    + delta * (1.0 if genre == "letters" else 0.0)
                    for genre in ALL_GENRES
                }
            )
            paths.append(
                mixture_record(
                    result, model_name, "covered_shift_partial", uniform_support, target, delta
                )
            )
        for delta in [0.0, 0.10, 0.25, 0.50, 0.75, 1.0]:
            target = normalize(
                {
                    genre: (1.0 - delta) * uniform_support.get(genre, 0.0)
                    + delta * uniform_escape.get(genre, 0.0)
                    for genre in ALL_GENRES
                }
            )
            paths.append(
                mixture_record(
                    result, model_name, "support_escape", uniform_support, target, delta
                )
            )
    return paths


def run_one(
    model_path: pathlib.Path,
    raw_dir: pathlib.Path,
    result_dir: pathlib.Path,
    checkpoint_dir: pathlib.Path,
    seed: int,
) -> Dict:
    run_id = f"coverage_seed{seed}"
    result_path = result_dir / f"{run_id}.json"
    if result_path.exists():
        cached = json.loads(result_path.read_text(encoding="utf-8"))
        if cached.get("status") == "completed" and cached.get("evaluation_rule") == "left_pad_last_v2":
            return cached
    start = time.time()
    model = tokenizer = None
    try:
        train_rows = read_jsonl(raw_dir / "train_cf_n250_k4.jsonl")
        diagnostic_rows = read_jsonl(raw_dir / "diagnostic_cf_k4.jsonl")
        partial_rows = make_partial_rows(train_rows, total=len(train_rows))

        full_checkpoint = checkpoint_dir.parent / "checkpoints" / f"cf_n250_k4_eta0p00_seed{seed}"
        if not full_checkpoint.exists():
            raise FileNotFoundError(f"missing full adapter checkpoint: {full_checkpoint}")
        model, tokenizer = load_adapter(model_path, full_checkpoint, seed)
        full_eval = add_diagnostic_fields(
            evaluate_rows(model, tokenizer, diagnostic_rows, 16, "coverage_diagnostic")
            , diagnostic_rows
        )
        full_loss = float("nan")
        del model, tokenizer
        model = tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        partial_checkpoint = checkpoint_dir / run_id
        if partial_checkpoint.exists():
            model, tokenizer = load_adapter(model_path, partial_checkpoint, seed)
            partial_loss = float("nan")
        else:
            model, tokenizer, partial_loss = train_partial(
                model_path, partial_rows, partial_checkpoint, seed
            )
        partial_eval = add_diagnostic_fields(
            evaluate_rows(model, tokenizer, diagnostic_rows, 16, "coverage_diagnostic")
            , diagnostic_rows
        )

        metrics = {
            "run_id": run_id,
            "kind": "coverage",
            "seed": seed,
            "train_count_full": len(train_rows),
            "train_count_partial": len(partial_rows),
            "partial_support": SUPPORTED_GENRES,
            "unsupported_genres": UNSUPPORTED_GENRES,
            "full": {
                "train_loss": full_loss,
                "diagnostic": full_eval,
                "paths": build_paths("full", full_eval),
            },
            "partial": {
                "train_loss": partial_loss,
                "diagnostic": partial_eval,
                "paths": build_paths("partial", partial_eval),
            },
            "model": "Qwen2.5-1.5B-Instruct",
            "evaluation_rule": "left_pad_last_v2",
            "status": "completed",
            "elapsed_seconds": time.time() - start,
        }
        write_json(result_path, metrics)
        return metrics
    except Exception as error:
        failure = {
            "run_id": run_id,
            "kind": "coverage",
            "seed": seed,
            "status": "failed",
            "error": repr(error),
            "elapsed_seconds": time.time() - start,
        }
        write_json(result_path, failure)
        print(json.dumps(failure, ensure_ascii=False), flush=True)
        return failure
    finally:
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=pathlib.Path, required=True)
    parser.add_argument("--raw-dir", type=pathlib.Path, required=True)
    parser.add_argument("--result-dir", type=pathlib.Path, required=True)
    parser.add_argument("--checkpoint-dir", type=pathlib.Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = parser.parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for index, seed in enumerate(args.seeds, start=1):
        print(f"run={index}/{len(args.seeds)} coverage seed={seed}", flush=True)
        results.append(
            run_one(
                args.model_path,
                args.raw_dir,
                args.result_dir,
                args.checkpoint_dir,
                seed,
            )
        )
    manifest = {
        "mode": "coverage",
        "seeds": args.seeds,
        "planned_runs": len(args.seeds),
        "completed_runs": sum(item.get("status") == "completed" for item in results),
        "failed_runs": sum(item.get("status") == "failed" for item in results),
        "results": results,
    }
    write_json(args.result_dir / "manifest_coverage.json", manifest)
    print(json.dumps({key: manifest[key] for key in manifest if key != "results"}, indent=2))


if __name__ == "__main__":
    main()
