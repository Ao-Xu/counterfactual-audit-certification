"""Real-LLM phase-boundary experiment for Appendix C.

The population theorem uses an intervention strength that interpolates the
original nuisance and an intervention draw.  A text pipeline cannot linearly
interpolate arbitrary natural-language strings, so this experiment uses the
corresponding empirical intervention-mixture analogue: for each anchor, a
fraction ``lambda_cf`` of the 1,000 training rows is replaced by its frozen
counterfactual rewrite and the remaining anchors keep their factual row.
The semantic corruption rate is applied only to the counterfactual rows.

The experiment keeps the anchor set, optimizer budget, tokenizer, LoRA
configuration, evaluation rule, and generated text fixed across the grid.
It tests whether the qualitative benefit--harm boundary moves with the
amount of counterfactual intervention in a real language model.
"""

from __future__ import annotations

import argparse
import gc
import json
import pathlib
import random
import time
from typing import Dict, List, Sequence

import numpy as np
import torch
from transformers import Trainer, TrainingArguments

from train_and_eval import (
    CompletionCollator,
    PromptDataset,
    build_model,
    evaluate_rows,
    read_jsonl,
    write_json,
)


DEFAULT_LAMBDAS = [0.50]
DEFAULT_ETAS = [0.00, 0.04, 0.08, 0.12, 0.16, 0.20]


def add_loss_metrics(result: Dict) -> Dict:
    probabilities = np.asarray(result["probabilities"], dtype=float)
    targets = np.asarray(result["targets"], dtype=int)
    clipped = np.clip(probabilities[np.arange(len(targets)), targets], 1e-12, 1.0)
    result["nll"] = float(-np.mean(np.log(clipped)))
    return result


def predictive_kl_from_evaluation(rows: Sequence[Dict], result: Dict) -> float:
    """Compute sibling predictive KL from the already cached diagnostic pass.

    Re-running the model on the same 320 diagnostic rows is unnecessary and
    makes the phase grid disproportionately slow on the 8GB GPU.  The
    probabilities in ``result`` were produced by the deterministic evaluator
    immediately above, so reusing them is numerically identical.
    """
    probabilities = np.asarray(result["probabilities"], dtype=float)
    grouped: Dict[str, List[np.ndarray]] = {}
    for row, probability in zip(rows, probabilities):
        grouped.setdefault(str(row["anchor_id"]), []).append(probability)
    values = []
    for members in grouped.values():
        for i, left in enumerate(members):
            for j, right in enumerate(members):
                if i == j:
                    continue
                values.append(float(np.sum(left * (np.log(left + 1e-12) - np.log(right + 1e-12)))))
    return float(np.mean(values)) if values else float("nan")


def make_mixture_rows(
    factual_rows: Sequence[Dict],
    counterfactual_rows: Sequence[Dict],
    lambda_cf: float,
    eta: float,
    seed: int,
) -> List[Dict]:
    factual = {row["anchor_id"]: dict(row) for row in factual_rows}
    counterfactual = {row["anchor_id"]: dict(row) for row in counterfactual_rows}
    common = sorted(set(factual) & set(counterfactual))
    if len(common) != len(factual_rows) or len(common) != len(counterfactual_rows):
        raise RuntimeError("factual and counterfactual pools must share the full anchor set")

    rng = random.Random(seed)
    count_cf = int(round(len(common) * lambda_cf))
    selected_cf = set(rng.sample(common, count_cf))
    rows = []
    for anchor_id in common:
        if anchor_id in selected_cf:
            row = dict(counterfactual[anchor_id])
            row["phase_source"] = "counterfactual"
        else:
            row = dict(factual[anchor_id])
            row["phase_source"] = "factual"
        rows.append(row)

    cf_indices = [i for i, row in enumerate(rows) if row["phase_source"] == "counterfactual"]
    flip_count = int(round(len(cf_indices) * eta))
    rng.shuffle(cf_indices)
    for position in cf_indices[:flip_count]:
        row = rows[position]
        labels = ["entailment", "neutral", "contradiction"]
        old = labels.index(row["label"])
        alternatives = [value for value in range(len(labels)) if value != old]
        row["label"] = labels[alternatives[(seed + position) % len(alternatives)]]
        row["semantic_flip_injected"] = True
    return rows


def train_one(
    model_path: pathlib.Path,
    raw_dir: pathlib.Path,
    result_dir: pathlib.Path,
    checkpoint_dir: pathlib.Path,
    lambda_cf: float,
    eta: float,
    seed: int,
    eval_batch_size: int,
) -> Dict:
    run_id = f"phase_mix{lambda_cf:.2f}_eta{eta:.2f}_seed{seed}".replace(".", "p")
    result_path = result_dir / f"{run_id}.json"
    if result_path.exists():
        cached = json.loads(result_path.read_text(encoding="utf-8"))
        if cached.get("status") == "completed" and cached.get("evaluation_rule") == "left_pad_last_v2":
            return cached

    start = time.time()
    model = trainer = None
    try:
        factual_rows = read_jsonl(raw_dir / "train_factual.jsonl")
        counterfactual_rows = read_jsonl(raw_dir / "train_cf_n1000_k1.jsonl")
        train_rows = make_mixture_rows(factual_rows, counterfactual_rows, lambda_cf, eta, seed)
        matched_rows = read_jsonl(raw_dir / "eval_matched.jsonl")
        mismatched_rows = read_jsonl(raw_dir / "eval_mismatched.jsonl")
        diagnostic_rows = read_jsonl(raw_dir / "diagnostic_cf_k4.jsonl")

        model, tokenizer = build_model(model_path, seed)
        dataset = PromptDataset(train_rows, tokenizer)
        bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
        args = TrainingArguments(
            output_dir=str(checkpoint_dir / run_id),
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
        train_output = trainer.train()
        matched = add_loss_metrics(evaluate_rows(model, tokenizer, matched_rows, eval_batch_size, "matched"))
        mismatched = add_loss_metrics(evaluate_rows(model, tokenizer, mismatched_rows, eval_batch_size, "mismatched"))
        diagnostic = add_loss_metrics(evaluate_rows(model, tokenizer, diagnostic_rows, eval_batch_size, "diagnostic"))
        checkpoint_path = checkpoint_dir / run_id
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(checkpoint_path, safe_serialization=True)
        cf_count = sum(row["phase_source"] == "counterfactual" for row in train_rows)
        metrics = {
            "run_id": run_id,
            "kind": "phase_mixture",
            "lambda_cf": float(lambda_cf),
            "eta": float(eta),
            "seed": int(seed),
            "train_count": len(train_rows),
            "counterfactual_count": int(cf_count),
            "factual_count": int(len(train_rows) - cf_count),
            "corrupted_counterfactual_count": int(sum(row.get("semantic_flip_injected", False) for row in train_rows)),
            "train_loss": float(train_output.training_loss),
            "train_runtime_seconds": float(train_output.metrics.get("train_runtime", float("nan"))),
            "matched": matched,
            "mismatched": mismatched,
            "diagnostic": diagnostic,
            "shift_gap_accuracy": float(abs(matched["accuracy"] - mismatched["accuracy"])),
            "predictive_pairwise_kl": predictive_kl_from_evaluation(diagnostic_rows, diagnostic),
            "model": "Qwen2.5-1.5B-Instruct",
            "training": {
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "target_modules": ["q_proj", "v_proj"],
                "learning_rate": 2e-4,
                "epochs": 1,
                "micro_batch_size": 2,
                "gradient_accumulation_steps": 4,
                "max_length": 384,
                "dtype": "bf16" if bf16 else "fp32",
            },
            "evaluation_rule": "left_pad_last_v2",
            "status": "completed",
            "elapsed_seconds": time.time() - start,
        }
        write_json(result_path, metrics)
        return metrics
    except Exception as error:
        failure = {
            "run_id": run_id,
            "kind": "phase_mixture",
            "lambda_cf": float(lambda_cf),
            "eta": float(eta),
            "seed": int(seed),
            "status": "failed",
            "error": repr(error),
            "elapsed_seconds": time.time() - start,
        }
        write_json(result_path, failure)
        print(json.dumps(failure, ensure_ascii=False), flush=True)
        return failure
    finally:
        if trainer is not None:
            del trainer
        if model is not None:
            del model
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
    parser.add_argument("--lambda-values", type=float, nargs="+", default=DEFAULT_LAMBDAS)
    parser.add_argument("--eta-values", type=float, nargs="+", default=DEFAULT_ETAS)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    args = parser.parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    planned = [(lam, eta, seed) for lam in args.lambda_values for eta in args.eta_values for seed in args.seeds]
    results = []
    for index, (lam, eta, seed) in enumerate(planned, start=1):
        print(f"run={index}/{len(planned)} lambda_cf={lam} eta={eta} seed={seed}", flush=True)
        results.append(train_one(args.model_path, args.raw_dir, args.result_dir, args.checkpoint_dir, lam, eta, seed, args.eval_batch_size))
    manifest = {
        "mode": "phase_mixture",
        "seeds": args.seeds,
        "lambda_values": args.lambda_values,
        "eta_values": args.eta_values,
        "planned_runs": len(planned),
        "completed_runs": sum(item.get("status") == "completed" for item in results),
        "failed_runs": sum(item.get("status") == "failed" for item in results),
        "results": results,
    }
    write_json(args.result_dir / "manifest_phase_mixture.json", manifest)
    print(json.dumps({key: manifest[key] for key in manifest if key != "results"}, indent=2))


if __name__ == "__main__":
    main()
