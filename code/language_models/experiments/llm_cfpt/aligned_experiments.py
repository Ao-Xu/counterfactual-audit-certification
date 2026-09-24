"""Theorem-aligned real-LLM experiments for Appendix C.

This script reuses the frozen MultiNLI counterfactual pools and constructs
nested anchor/sibling training splits.  It reports matched, mismatched, and
same-anchor fresh-sibling evaluation using the same left-padding evaluator as
the main pilot.
"""

from __future__ import annotations

import argparse
import gc
import json
import pathlib
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from peft import PeftModel
from transformers import Trainer, TrainingArguments

from train_and_eval import (
    CompletionCollator,
    LABELS,
    PromptDataset,
    build_model,
    evaluate_rows,
    load_base_model,
    predictive_kl,
    read_jsonl,
    write_json,
)


ALIGNED_CONDITIONS: List[Tuple[int, int, str, int]] = [
    (125, 1, "train_cf_n125_k8.jsonl", 8),
    (125, 4, "train_cf_n125_k8.jsonl", 8),
    (125, 8, "train_cf_n125_k8.jsonl", 8),
    (250, 1, "train_cf_n250_k4.jsonl", 4),
    (250, 4, "train_cf_n250_k4.jsonl", 4),
    (500, 1, "train_cf_n500_k2.jsonl", 2),
    (500, 2, "train_cf_n500_k2.jsonl", 2),
    (1000, 1, "train_cf_n1000_k1.jsonl", 1),
]


def anchor_order(raw_dir: pathlib.Path) -> List[str]:
    return [row["anchor_id"] for row in read_jsonl(raw_dir / "anchors.jsonl")]


def build_nested_rows(
    raw_dir: pathlib.Path, n: int, k: int, source_name: str, source_k: int
) -> Tuple[List[Dict], List[Dict]]:
    source = read_jsonl(raw_dir / source_name)
    selected_ids = set(anchor_order(raw_dir)[:n])
    source = [row for row in source if row["anchor_id"] in selected_ids]
    train_rows = [
        row for row in source if int(row.get("sibling_index", 0)) < k
    ]
    fresh_rows = [
        row for row in source if int(row.get("sibling_index", 0)) >= k
    ]
    expected_train = n * k
    if len(train_rows) != expected_train:
        raise RuntimeError(
            f"nested training split has {len(train_rows)} rows; "
            f"expected {expected_train} for n={n}, k={k}"
        )
    if k >= source_k:
        fresh_rows = []
    return train_rows, fresh_rows


def add_loss_metrics(result: Dict) -> Dict:
    probabilities = np.asarray(result["probabilities"], dtype=float)
    targets = np.asarray(result["targets"], dtype=int)
    if len(targets):
        clipped = np.clip(probabilities[np.arange(len(targets)), targets], 1e-12, 1.0)
        result["nll"] = float(-np.mean(np.log(clipped)))
        one_hot = np.zeros_like(probabilities)
        one_hot[np.arange(len(targets)), targets] = 1.0
        result["brier"] = float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))
    else:
        result["nll"] = float("nan")
        result["brier"] = float("nan")
    return result


def add_anchor_ids(result: Dict, rows: Sequence[Dict]) -> Dict:
    result["anchor_ids"] = [row["anchor_id"] for row in rows]
    result["genres"] = [
        row.get("genre", row.get("source_genre", "unknown")) for row in rows
    ]
    return result


def train_adapter(
    model_path: pathlib.Path,
    train_rows: Sequence[Dict],
    checkpoint_path: pathlib.Path,
    seed: int,
) -> Tuple[object, object, float]:
    model, tokenizer = build_model(model_path, seed)
    dataset = PromptDataset(train_rows, tokenizer)
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
    del trainer
    return model, tokenizer, float(output.training_loss)


def load_adapter(
    model_path: pathlib.Path, checkpoint_path: pathlib.Path, seed: int
) -> Tuple[object, object]:
    base_model, tokenizer = load_base_model(model_path, seed)
    model = PeftModel.from_pretrained(base_model, checkpoint_path, is_trainable=False)
    return model, tokenizer


def run_one(
    model_path: pathlib.Path,
    raw_dir: pathlib.Path,
    result_dir: pathlib.Path,
    checkpoint_dir: pathlib.Path,
    n: int,
    k: int,
    source_name: str,
    source_k: int,
    seed: int,
    eval_batch_size: int,
) -> Dict:
    run_id = f"aligned_n{n}_k{k}_seed{seed}"
    result_path = result_dir / f"{run_id}.json"
    if result_path.exists():
        cached = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            cached.get("status") == "completed"
            and cached.get("evaluation_rule") == "left_pad_last_v2"
        ):
            return cached

    start = time.time()
    checkpoint_path = checkpoint_dir / run_id
    train_rows, fresh_rows = build_nested_rows(
        raw_dir, n, k, source_name, source_k
    )
    matched_rows = read_jsonl(raw_dir / "eval_matched.jsonl")
    mismatched_rows = read_jsonl(raw_dir / "eval_mismatched.jsonl")
    diagnostic_rows = read_jsonl(raw_dir / "diagnostic_cf_k4.jsonl")

    existing_checkpoint = (
        checkpoint_dir.parent / "checkpoints"
        / f"cf_n{n}_k{k}_eta0p00_seed{seed}"
    )
    try:
        if existing_checkpoint.exists() and k == source_k:
            model, tokenizer = load_adapter(model_path, existing_checkpoint, seed)
            train_loss = float("nan")
            checkpoint_used = str(existing_checkpoint)
        else:
            model, tokenizer, train_loss = train_adapter(
                model_path, train_rows, checkpoint_path, seed
            )
            checkpoint_used = str(checkpoint_path)

        matched = add_anchor_ids(
            add_loss_metrics(
                evaluate_rows(model, tokenizer, matched_rows, eval_batch_size, "matched")
            ),
            matched_rows,
        )
        mismatched = add_anchor_ids(
            add_loss_metrics(
                evaluate_rows(
                    model, tokenizer, mismatched_rows, eval_batch_size, "mismatched"
                )
            ),
            mismatched_rows,
        )
        diagnostic = add_anchor_ids(
            add_loss_metrics(
                evaluate_rows(
                    model, tokenizer, diagnostic_rows, eval_batch_size, "diagnostic"
                )
            ),
            diagnostic_rows,
        )
        if fresh_rows:
            fresh = add_anchor_ids(
                add_loss_metrics(
                    evaluate_rows(
                        model, tokenizer, fresh_rows, eval_batch_size, "same_anchor_fresh"
                    )
                ),
                fresh_rows,
            )
            fresh["predictive_pairwise_kl"] = predictive_kl(
                model, tokenizer, fresh_rows, eval_batch_size
            )
        else:
            fresh = {
                "name": "same_anchor_fresh",
                "count": 0,
                "accuracy": float("nan"),
                "error": float("nan"),
                "nll": float("nan"),
                "brier": float("nan"),
                "predictive_pairwise_kl": float("nan"),
                "probabilities": [],
                "predictions": [],
                "targets": [],
                "anchor_ids": [],
                "genres": [],
            }

        metrics = {
            "run_id": run_id,
            "kind": "aligned_cfpt",
            "n": n,
            "k": k,
            "seed": seed,
            "train_count": len(train_rows),
            "fresh_count": len(fresh_rows),
            "train_loss": train_loss,
            "matched": matched,
            "mismatched": mismatched,
            "diagnostic": diagnostic,
            "same_anchor_fresh": fresh,
            "shift_gap_accuracy": float(
                abs(matched["accuracy"] - mismatched["accuracy"])
            ),
            "model": "Qwen2.5-1.5B-Instruct",
            "checkpoint_used": checkpoint_used,
            "evaluation_rule": "left_pad_last_v2",
            "status": "completed",
            "elapsed_seconds": time.time() - start,
        }
        write_json(result_path, metrics)
        return metrics
    except Exception as error:
        failure = {
            "run_id": run_id,
            "kind": "aligned_cfpt",
            "n": n,
            "k": k,
            "seed": seed,
            "status": "failed",
            "error": repr(error),
            "elapsed_seconds": time.time() - start,
        }
        write_json(result_path, failure)
        print(json.dumps(failure, ensure_ascii=False), flush=True)
        return failure
    finally:
        try:
            del model, tokenizer
        except UnboundLocalError:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=pathlib.Path, required=True)
    parser.add_argument("--raw-dir", type=pathlib.Path, required=True)
    parser.add_argument("--result-dir", type=pathlib.Path, required=True)
    parser.add_argument("--checkpoint-dir", type=pathlib.Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2027, 2028, 2029])
    parser.add_argument("--eval-batch-size", type=int, default=16)
    args = parser.parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    all_results = []
    total = len(ALIGNED_CONDITIONS) * len(args.seeds)
    index = 0
    for seed in args.seeds:
        for n, k, source_name, source_k in ALIGNED_CONDITIONS:
            index += 1
            print(
                f"run={index}/{total} n={n} k={k} seed={seed}",
                flush=True,
            )
            all_results.append(
                run_one(
                    args.model_path,
                    args.raw_dir,
                    args.result_dir,
                    args.checkpoint_dir,
                    n,
                    k,
                    source_name,
                    source_k,
                    seed,
                    args.eval_batch_size,
                )
            )
    manifest = {
        "mode": "aligned",
        "seeds": args.seeds,
        "planned_runs": total,
        "completed_runs": sum(r.get("status") == "completed" for r in all_results),
        "failed_runs": sum(r.get("status") == "failed" for r in all_results),
    }
    write_json(args.result_dir / "manifest_aligned.json", {**manifest, "results": all_results})
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
