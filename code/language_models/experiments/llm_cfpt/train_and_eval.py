"""LoRA post-training and grouped evaluation for the real-LLM pilot."""

from __future__ import annotations

import argparse
import gc
import json
import math
import pathlib
import random
import time
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)


LABELS = ["entailment", "neutral", "contradiction"]
MAIN_RUNS = [
    ("factual", 1000, 1, 0.0, None),
    ("cf", 1000, 1, 0.0, "experiments/llm_cfpt/raw/train_cf_n1000_k1.jsonl"),
    ("cf", 500, 2, 0.0, "experiments/llm_cfpt/raw/train_cf_n500_k2.jsonl"),
    ("cf", 250, 4, 0.0, "experiments/llm_cfpt/raw/train_cf_n250_k4.jsonl"),
    ("cf", 125, 8, 0.0, "experiments/llm_cfpt/raw/train_cf_n125_k8.jsonl"),
]
# The coarse pilot rates are retained in the cache; the intermediate points
# resolve the local benefit--harm transition for Appendix C.
CORRUPTION_RATES = [0.0, 0.02, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.16, 0.20]


def read_jsonl(path: pathlib.Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: pathlib.Path, value: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def prompt_for_row(row: Dict) -> str:
    genre = row.get("genre", row.get("source_genre", "unknown"))
    return (
        "You are an NLI classifier. Decide whether the hypothesis is entailed "
        "by, neutral to, or contradicts the premise. Return exactly one label: "
        "entailment, neutral, or contradiction.\n"
        f"Genre: {genre}\n"
        f"Premise: {row['premise']}\n"
        f"Hypothesis: {row['hypothesis']}\n"
        "Label:"
    )


class PromptDataset(torch.utils.data.Dataset):
    def __init__(self, rows: Sequence[Dict], tokenizer, max_length: int = 384):
        self.items = []
        for row in rows:
            prompt = prompt_for_row(row)
            target = " " + str(row["label"])
            prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
            full_ids = tokenizer(prompt + target, add_special_tokens=True)["input_ids"]
            if len(full_ids) > max_length:
                full_ids = full_ids[:max_length]
            labels = [-100] * min(len(prompt_ids), len(full_ids))
            labels.extend(full_ids[len(labels) :])
            labels = labels[: len(full_ids)]
            if not any(value != -100 for value in labels):
                continue
            self.items.append(
                {
                    "input_ids": full_ids,
                    "attention_mask": [1] * len(full_ids),
                    "labels": labels,
                }
            )
        if not self.items:
            raise RuntimeError("No trainable rows remained after tokenization")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class CompletionCollator:
    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id

    def __call__(self, features: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        width = max(len(item["input_ids"]) for item in features)
        batch_size = len(features)
        input_ids = torch.full((batch_size, width), self.pad_id, dtype=torch.long)
        attention = torch.zeros((batch_size, width), dtype=torch.long)
        labels = torch.full((batch_size, width), -100, dtype=torch.long)
        for index, item in enumerate(features):
            length = len(item["input_ids"])
            input_ids[index, :length] = torch.tensor(item["input_ids"], dtype=torch.long)
            attention[index, :length] = 1
            labels[index, :length] = torch.tensor(item["labels"], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention, "labels": labels}


def load_base_model(model_path: pathlib.Path, seed: int):
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer


def build_model(model_path: pathlib.Path, seed: int):
    model, tokenizer = load_base_model(model_path, seed)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    lora = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora)
    return model, tokenizer


def candidate_score_batch(model, tokenizer, prompts: Sequence[str], batch_size: int = 8):
    """Return a three-label distribution from the next-token label scores.

    The three labels have distinct first tokens in the Qwen tokenizer.  Scoring
    those first tokens at the ``Label:`` position gives the same deterministic
    decision rule for every run while requiring one forward pass per prompt
    batch instead of one pass per prompt-label pair.
    """
    first_ids = [tokenizer.encode(" " + label, add_special_tokens=False)[0] for label in LABELS]
    model.eval()
    all_scores = []
    old_padding_side = tokenizer.padding_side
    # The fast ``logits_to_keep=1`` path returns the final sequence position.
    # Left padding guarantees that this is the final non-padding token even
    # when prompts in a batch have different lengths.
    tokenizer.padding_side = "left"
    try:
        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start : start + batch_size]
            encoded = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=384,
            )
            encoded = {key: value.to(model.device) for key, value in encoded.items()}
            with torch.inference_mode():
                # Qwen2 supports ``logits_to_keep``.  We only need the next-token
                # logits at the final non-padding position; materializing the full
                # vocabulary projection at every prefix position is needlessly
                # expensive on the 8GB pilot GPU.
                logits = model(**encoded, logits_to_keep=1).logits.float()
                last_logits = logits[:, -1, :]
                scores = last_logits[:, first_ids]
                scores = scores - scores.max(dim=1, keepdim=True).values
                probabilities = torch.softmax(scores, dim=1).cpu().numpy()
            all_scores.append(probabilities)
    finally:
        tokenizer.padding_side = old_padding_side
    probabilities = np.concatenate(all_scores, axis=0)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities


def evaluate_rows(model, tokenizer, rows: Sequence[Dict], batch_size: int, name: str) -> Dict:
    prompts = [prompt_for_row(row) for row in rows]
    probabilities = candidate_score_batch(model, tokenizer, prompts, batch_size=batch_size)
    predictions = np.argmax(probabilities, axis=1)
    targets = np.array([LABELS.index(row["label"]) for row in rows])
    result = {
        "name": name,
        "count": len(rows),
        "accuracy": float(np.mean(predictions == targets)),
        "error": float(np.mean(predictions != targets)),
        "mean_confidence": float(np.mean(np.max(probabilities, axis=1))),
        "probabilities": probabilities.tolist(),
        "predictions": predictions.tolist(),
        "targets": targets.tolist(),
    }
    by_genre = {}
    for genre in sorted({row.get("genre", row.get("source_genre", "unknown")) for row in rows}):
        indices = [i for i, row in enumerate(rows) if row.get("genre", row.get("source_genre", "unknown")) == genre]
        by_genre[genre] = float(np.mean(predictions[indices] == targets[indices]))
    result["accuracy_by_genre"] = by_genre
    return result


def mixture_accuracy(matched: Dict, mismatched: Dict, seed: int) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    matched_pred = np.array(matched["predictions"])
    matched_target = np.array(matched["targets"])
    mismatched_pred = np.array(mismatched["predictions"])
    mismatched_target = np.array(mismatched["targets"])
    output = {}
    for delta in [0.0, 0.25, 0.5, 0.75, 1.0]:
        choose_shift = rng.random(len(matched_pred)) < delta
        pred = np.where(choose_shift, mismatched_pred, matched_pred)
        target = np.where(choose_shift, mismatched_target, matched_target)
        output[str(delta)] = float(np.mean(pred == target))
    return output


def predictive_kl(model, tokenizer, rows: Sequence[Dict], batch_size: int) -> float:
    probabilities = candidate_score_batch(
        model, tokenizer, [prompt_for_row(row) for row in rows], batch_size=batch_size
    )
    grouped = defaultdict(list)
    for row, probability in zip(rows, probabilities):
        grouped[row["anchor_id"]].append(probability)
    values = []
    for members in grouped.values():
        if len(members) < 2:
            continue
        for i, left in enumerate(members):
            for j, right in enumerate(members):
                if i == j:
                    continue
                values.append(float(np.sum(left * (np.log(left + 1e-12) - np.log(right + 1e-12)))))
    return float(np.mean(values)) if values else float("nan")


def make_training_rows(base_rows: Sequence[Dict], eta: float, seed: int) -> List[Dict]:
    rows = [dict(row) for row in base_rows]
    if eta <= 0:
        return rows
    count = int(round(len(rows) * eta))
    rng = random.Random(seed)
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    for index in indices[:count]:
        old = LABELS.index(rows[index]["label"])
        alternatives = [value for value in range(len(LABELS)) if value != old]
        rows[index]["label"] = LABELS[alternatives[(seed + index) % len(alternatives)]]
        rows[index]["semantic_flip_injected"] = True
    return rows


def resolve_config(kind: str, n: int, k: int, eta: float, raw_dir: pathlib.Path) -> List[Dict]:
    if kind == "factual":
        return read_jsonl(raw_dir / "train_factual.jsonl")
    return read_jsonl(raw_dir / f"train_cf_n{n}_k{k}.jsonl")


def train_one(
    model_path: pathlib.Path,
    raw_dir: pathlib.Path,
    result_dir: pathlib.Path,
    checkpoint_dir: pathlib.Path,
    kind: str,
    n: int,
    k: int,
    eta: float,
    seed: int,
    eval_batch_size: int,
    max_train_examples: int | None = None,
) -> Dict:
    run_id = f"{kind}_n{n}_k{k}_eta{eta:.2f}_seed{seed}".replace(".", "p")
    result_path = result_dir / f"{run_id}.json"
    if result_path.exists() and not max_train_examples:
        cached = json.loads(result_path.read_text(encoding="utf-8"))
        # A failed run is deliberately resumable: only completed artifacts are
        # valid cache entries.  This prevents a transient OOM or data error
        # from being silently reported as an experimental result.
        if cached.get("status") == "completed" and cached.get("evaluation_rule") == "left_pad_last_v2":
            return cached
    start_time = time.time()
    try:
        base_rows = resolve_config(kind, n, k, eta, raw_dir)
        if max_train_examples:
            base_rows = base_rows[:max_train_examples]
        train_rows = make_training_rows(base_rows, eta, seed)
        matched_rows = read_jsonl(raw_dir / "eval_matched.jsonl")
        mismatched_rows = read_jsonl(raw_dir / "eval_mismatched.jsonl")
        diagnostic_rows = read_jsonl(raw_dir / "diagnostic_cf_k4.jsonl")
        model, tokenizer = build_model(model_path, seed)
        train_dataset = PromptDataset(train_rows, tokenizer)
        if torch.cuda.is_available():
            bf16 = torch.cuda.is_bf16_supported()
        else:
            bf16 = False
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
            train_dataset=train_dataset,
            data_collator=CompletionCollator(tokenizer),
        )
        train_output = trainer.train()
        matched_result = evaluate_rows(model, tokenizer, matched_rows, eval_batch_size, "matched")
        mismatched_result = evaluate_rows(model, tokenizer, mismatched_rows, eval_batch_size, "mismatched")
        mixture = mixture_accuracy(matched_result, mismatched_result, seed)
        kl = predictive_kl(model, tokenizer, diagnostic_rows, eval_batch_size)
        checkpoint_path = checkpoint_dir / run_id
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(checkpoint_path, safe_serialization=True)
        metrics = {
            "run_id": run_id,
            "kind": kind,
            "n": n,
            "k": k,
            "eta": eta,
            "seed": seed,
            "train_count": len(train_rows),
            "valid_generation_count": int(sum(bool(row.get("generation_valid", True)) for row in train_rows)),
            "train_loss": float(train_output.training_loss),
            "train_runtime_seconds": float(train_output.metrics.get("train_runtime", float("nan"))),
            "matched": matched_result,
            "mismatched": mismatched_result,
            "shift_gap_accuracy": float(abs(matched_result["accuracy"] - mismatched_result["accuracy"])),
            "mixture_accuracy": mixture,
            "predictive_pairwise_kl": kl,
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
            "elapsed_seconds": time.time() - start_time,
            "evaluation_rule": "left_pad_last_v2",
            "status": "completed",
        }
        write_json(result_path, metrics)
        del trainer, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return metrics
    except Exception as error:
        failure = {
            "run_id": run_id,
            "kind": kind,
            "n": n,
            "k": k,
            "eta": eta,
            "seed": seed,
            "status": "failed",
            "error": repr(error),
            "elapsed_seconds": time.time() - start_time,
        }
        write_json(result_path, failure)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(json.dumps(failure, ensure_ascii=False), flush=True)
        return failure


def reevaluate_one(
    model_path: pathlib.Path,
    raw_dir: pathlib.Path,
    result_dir: pathlib.Path,
    checkpoint_dir: pathlib.Path,
    kind: str,
    n: int,
    k: int,
    eta: float,
    seed: int,
    eval_batch_size: int,
) -> Dict:
    """Recompute metrics for an already trained adapter without retraining.

    This is used only to repair the first pilot batch whose training artifacts
    were valid but whose right-padding evaluator selected a padding position.
    The original training metadata is retained; only evaluation fields are
    replaced by the left-padding implementation.
    """
    run_id = f"{kind}_n{n}_k{k}_eta{eta:.2f}_seed{seed}".replace(".", "p")
    result_path = result_dir / f"{run_id}.json"
    checkpoint_path = checkpoint_dir / run_id
    start_time = time.time()
    try:
        if not result_path.exists():
            raise FileNotFoundError(f"missing result metadata: {result_path}")
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"missing adapter checkpoint: {checkpoint_path}")
        metrics = json.loads(result_path.read_text(encoding="utf-8"))
        if metrics.get("status") != "completed":
            raise RuntimeError(f"cannot reevaluate non-completed run: {run_id}")
        if metrics.get("evaluation_rule") == "left_pad_last_v2":
            return metrics
        matched_rows = read_jsonl(raw_dir / "eval_matched.jsonl")
        mismatched_rows = read_jsonl(raw_dir / "eval_mismatched.jsonl")
        diagnostic_rows = read_jsonl(raw_dir / "diagnostic_cf_k4.jsonl")
        base_model, tokenizer = load_base_model(model_path, seed)
        model = PeftModel.from_pretrained(base_model, checkpoint_path, is_trainable=False)
        matched_result = evaluate_rows(model, tokenizer, matched_rows, eval_batch_size, "matched")
        mismatched_result = evaluate_rows(model, tokenizer, mismatched_rows, eval_batch_size, "mismatched")
        metrics["matched"] = matched_result
        metrics["mismatched"] = mismatched_result
        metrics["shift_gap_accuracy"] = float(
            abs(matched_result["accuracy"] - mismatched_result["accuracy"])
        )
        metrics["mixture_accuracy"] = mixture_accuracy(matched_result, mismatched_result, seed)
        metrics["predictive_pairwise_kl"] = predictive_kl(
            model, tokenizer, diagnostic_rows, eval_batch_size
        )
        metrics["evaluation_rule"] = "left_pad_last_v2"
        metrics["reevaluation_seconds"] = time.time() - start_time
        write_json(result_path, metrics)
        del model, base_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return metrics
    except Exception as error:
        failure = {
            "run_id": run_id,
            "kind": kind,
            "n": n,
            "k": k,
            "eta": eta,
            "seed": seed,
            "status": "failed",
            "error": repr(error),
            "elapsed_seconds": time.time() - start_time,
        }
        print(json.dumps(failure, ensure_ascii=False), flush=True)
        return failure


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=pathlib.Path, required=True)
    parser.add_argument("--raw-dir", type=pathlib.Path, required=True)
    parser.add_argument("--result-dir", type=pathlib.Path, required=True)
    parser.add_argument("--checkpoint-dir", type=pathlib.Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2027, 2028, 2029])
    parser.add_argument(
        "--mode", choices=["main", "corruption", "reevaluate_main", "all"], default="all"
    )
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--max-train-examples", type=int, default=None)
    args = parser.parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    if args.mode in {"main", "all"}:
        for seed in args.seeds:
            for kind, n, k, eta, _ in MAIN_RUNS:
                runs.append((kind, n, k, eta, seed))
    if args.mode == "reevaluate_main":
        for seed in args.seeds:
            for kind, n, k, eta, _ in MAIN_RUNS:
                runs.append((kind, n, k, eta, seed))
    if args.mode in {"corruption", "all"}:
        for seed in args.seeds:
            for eta in CORRUPTION_RATES:
                if eta == 0.0:
                    continue  # the eta=0 result is already in the main grid
                runs.append(("cf", 250, 4, eta, seed))
    all_results = []
    for index, (kind, n, k, eta, seed) in enumerate(runs, start=1):
        print(f"run={index}/{len(runs)} kind={kind} n={n} k={k} eta={eta} seed={seed}", flush=True)
        if args.mode == "reevaluate_main":
            result = reevaluate_one(
                args.model_path,
                args.raw_dir,
                args.result_dir,
                args.checkpoint_dir,
                kind,
                n,
                k,
                eta,
                seed,
                args.eval_batch_size,
            )
        else:
            result = train_one(
                args.model_path,
                args.raw_dir,
                args.result_dir,
                args.checkpoint_dir,
                kind,
                n,
                k,
                eta,
                seed,
                args.eval_batch_size,
                args.max_train_examples,
            )
        all_results.append(result)
    manifest = {
        "mode": args.mode,
        "seeds": args.seeds,
        "planned_runs": len(runs),
        "completed_runs": sum(item.get("status") == "completed" for item in all_results),
        "failed_runs": sum(item.get("status") == "failed" for item in all_results),
        "results": all_results,
    }
    write_json(args.result_dir / f"manifest_{args.mode}.json", manifest)
    print(json.dumps({key: manifest[key] for key in manifest if key != "results"}, indent=2))


if __name__ == "__main__":
    main()
