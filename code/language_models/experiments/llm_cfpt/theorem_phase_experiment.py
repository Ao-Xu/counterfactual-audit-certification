"""Theorem-matched real-LLM experiment for the semantic phase boundary.

This experiment keeps Qwen2.5-1.5B as the learner but renders the Gaussian
model of Theorem 3 as text.  Unlike the natural-text mixture pilot, the
intervention parameter is applied to every training example as
S_lambda=(1-lambda)S+lambda*S_tilde, and the response corruption is exactly
Y_eta=xi_eta*Y.  A scalar regression head is trained with squared loss so the
primary metric is the theorem's clean test risk.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import pathlib
import random
import time
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments, set_seed


DEFAULT_LAMBDAS = [0.50, 0.75, 1.00]
DEFAULT_ETA_VALUES = {
    0.50: [0.000, 0.005, 0.010, 0.020, 0.040],
    0.75: [0.000, 0.040, 0.060, 0.080, 0.100],
    1.00: [0.000, 0.060, 0.080, 0.100, 0.120],
}
DEFAULT_SEEDS = [2027, 2028, 2029]
ALPHA = 0.5
ALPHA_E = -1.5
SIGMA_U2 = 0.25
SIGMA_S2 = 4.0
SIGMA_Y2 = 0.10
TEST_SEED = 99173
TEMPLATES = [
    "Estimate the response Y from the observed coordinates.\n"
    "Task coordinate U = {u:.5f}.\nNuisance coordinate S = {s:.5f}.\n"
    "Response:",
    "Predict the real-valued response.\nObserved task signal: U={u:.5f}\n"
    "Observed nuisance signal: S={s:.5f}\nPrediction:",
    "Use the two measurements below to estimate Y.\nU (task) = {u:.5f}\n"
    "S (nuisance) = {s:.5f}\nY =",
    "Regression example\nfeature_u: {u:.5f}\nfeature_s: {s:.5f}\n"
    "target value:",
]


def write_json(path: pathlib.Path, value: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def write_jsonl(path: pathlib.Path, rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False) + "\n")


def read_jsonl(path: pathlib.Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sample_base(seed: int, count: int, alpha: float) -> List[Dict]:
    rng = np.random.default_rng(seed)
    c = rng.normal(0.0, 1.0, count)
    eps_u = rng.normal(0.0, math.sqrt(SIGMA_U2), count)
    eps_s = rng.normal(0.0, math.sqrt(SIGMA_S2), count)
    eps_y = rng.normal(0.0, math.sqrt(SIGMA_Y2), count)
    s_tilde = rng.normal(0.0, math.sqrt(ALPHA**2 + SIGMA_S2), count)
    flip_order = rng.permutation(count)
    rows = []
    for index in range(count):
        rows.append(
            {
                "row_id": index,
                "c": float(c[index]),
                "u": float(c[index] + eps_u[index]),
                "s": float(alpha * c[index] + eps_s[index]),
                "s_tilde": float(s_tilde[index]),
                "y": float(c[index] + eps_y[index]),
                "flip_rank": int(np.where(flip_order == index)[0][0]),
            }
        )
    return rows


def prepare_raw(raw_dir: pathlib.Path, seeds: Sequence[int], n_train: int, n_test: int) -> None:
    manifest_path = raw_dir / "manifest_theorem_phase.json"
    expected = {
        "seeds": list(seeds),
        "n_train": int(n_train),
        "n_test": int(n_test),
        "alpha": ALPHA,
        "alpha_e": ALPHA_E,
        "test_seed": TEST_SEED,
    }
    if manifest_path.exists():
        current = json.loads(manifest_path.read_text(encoding="utf-8"))
        if all(current.get(key) == value for key, value in expected.items()):
            return
    raw_dir.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        write_jsonl(raw_dir / f"base_seed{seed}.jsonl", sample_base(seed, n_train, ALPHA))
    test_rows = sample_base(TEST_SEED, n_test, ALPHA_E)
    write_jsonl(raw_dir / "test_clean.jsonl", test_rows)
    write_json(manifest_path, expected)


def render_prompt(u: float, s: float, row_id: int) -> str:
    return TEMPLATES[row_id % len(TEMPLATES)].format(u=u, s=s)


def make_train_rows(base_rows: Sequence[Dict], lambda_int: float, eta: float) -> List[Dict]:
    n = len(base_rows)
    flip_count = int(math.floor(eta * n + 1e-9))
    rows = []
    for item in base_rows:
        s_lambda = (1.0 - lambda_int) * item["s"] + lambda_int * item["s_tilde"]
        corrupted = item["flip_rank"] < flip_count
        y_eta = -item["y"] if corrupted else item["y"]
        rows.append(
            {
                "row_id": int(item["row_id"]),
                "text": render_prompt(item["u"], s_lambda, int(item["row_id"])),
                "target": float(y_eta),
                "u": float(item["u"]),
                "s_lambda": float(s_lambda),
                "y_clean": float(item["y"]),
                "corrupted": bool(corrupted),
            }
        )
    return rows


def make_test_rows(test_base: Sequence[Dict]) -> List[Dict]:
    return [
        {
            "row_id": int(item["row_id"]),
            "text": render_prompt(item["u"], item["s"], int(item["row_id"])),
            "target": float(item["y"]),
            "u": float(item["u"]),
            "s_e": float(item["s"]),
        }
        for item in test_base
    ]


class RegressionDataset(Dataset):
    def __init__(self, rows: Sequence[Dict], tokenizer, max_length: int) -> None:
        self.items = []
        for row in rows:
            encoded = tokenizer(
                row["text"],
                add_special_tokens=True,
                truncation=True,
                max_length=max_length,
            )
            self.items.append(
                {
                    "input_ids": encoded["input_ids"],
                    "attention_mask": encoded["attention_mask"],
                    "labels": float(row["target"]),
                }
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Dict:
        return self.items[index]


class RegressionCollator:
    def __init__(self, tokenizer) -> None:
        self.pad_id = tokenizer.pad_token_id

    def __call__(self, features: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        width = max(len(item["input_ids"]) for item in features)
        input_ids = torch.full((len(features), width), self.pad_id, dtype=torch.long)
        attention = torch.zeros((len(features), width), dtype=torch.long)
        labels = torch.zeros(len(features), dtype=torch.float32)
        for index, item in enumerate(features):
            length = len(item["input_ids"])
            input_ids[index, :length] = torch.tensor(item["input_ids"], dtype=torch.long)
            attention[index, :length] = 1
            labels[index] = float(item["labels"])
        return {"input_ids": input_ids, "attention_mask": attention, "labels": labels}


class ScalarRegressionModel(nn.Module):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.regression_head = nn.Linear(int(backbone.config.hidden_size), 1)
        self.config = backbone.config

    def gradient_checkpointing_enable(self, **kwargs):
        if hasattr(self.backbone, "gradient_checkpointing_enable"):
            return self.backbone.gradient_checkpointing_enable(**kwargs)
        return None

    def gradient_checkpointing_disable(self):
        if hasattr(self.backbone, "gradient_checkpointing_disable"):
            return self.backbone.gradient_checkpointing_disable()
        return None

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state
        last_index = attention_mask.long().sum(dim=1) - 1
        batch_index = torch.arange(hidden.shape[0], device=hidden.device)
        pooled = hidden[batch_index, last_index]
        prediction = self.regression_head(pooled).squeeze(-1)
        loss = None
        if labels is not None:
            loss = F.mse_loss(prediction.float(), labels.float())
        return {"loss": loss, "logits": prediction.unsqueeze(-1)}


def build_model(model_path: pathlib.Path, seed: int):
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    backbone = AutoModel.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    backbone.config.use_cache = False
    if hasattr(backbone, "enable_input_require_grads"):
        backbone.enable_input_require_grads()
    if hasattr(backbone, "gradient_checkpointing_enable"):
        backbone.gradient_checkpointing_enable()
    lora = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
        target_modules=["q_proj", "v_proj"],
    )
    backbone = get_peft_model(backbone, lora)
    return ScalarRegressionModel(backbone), tokenizer, dtype


def batched(items: Sequence[Dict], batch_size: int) -> Iterable[Sequence[Dict]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def evaluate(model: ScalarRegressionModel, tokenizer, rows: Sequence[Dict], batch_size: int, max_length: int) -> Dict:
    model.eval()
    device = next(model.parameters()).device
    predictions = []
    targets = []
    with torch.no_grad():
        for batch in batched(rows, batch_size):
            encoded = tokenizer(
                [item["text"] for item in batch],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            output = model(**encoded)
            predictions.extend(output["logits"].squeeze(-1).float().cpu().numpy().tolist())
            targets.extend([float(item["target"]) for item in batch])
    pred = np.asarray(predictions, dtype=float)
    target = np.asarray(targets, dtype=float)
    residual = pred - target
    return {
        "count": int(len(target)),
        "mse": float(np.mean(residual**2)),
        "mae": float(np.mean(np.abs(residual))),
        "prediction_mean": float(np.mean(pred)),
        "target_mean": float(np.mean(target)),
    }


def train_one(
    model_path: pathlib.Path,
    raw_dir: pathlib.Path,
    result_dir: pathlib.Path,
    checkpoint_dir: pathlib.Path,
    lambda_int: float,
    eta: float,
    seed: int,
    train_batch_size: int,
    eval_batch_size: int,
    max_length: int,
) -> Dict:
    run_id = f"theorem_phase_lam{lambda_int:.2f}_eta{eta:.3f}_seed{seed}".replace(".", "p")
    result_path = result_dir / f"{run_id}.json"
    if result_path.exists():
        cached = json.loads(result_path.read_text(encoding="utf-8"))
        if cached.get("status") == "completed" and cached.get("evaluation_rule") == "scalar_regression_last_hidden_v1":
            return cached

    start = time.time()
    model = trainer = None
    try:
        base_rows = read_jsonl(raw_dir / f"base_seed{seed}.jsonl")
        test_base = read_jsonl(raw_dir / "test_clean.jsonl")
        train_rows = make_train_rows(base_rows, lambda_int, eta)
        test_rows = make_test_rows(test_base)
        model, tokenizer, dtype = build_model(model_path, seed)
        train_dataset = RegressionDataset(train_rows, tokenizer, max_length)
        bf16 = dtype == torch.bfloat16
        args = TrainingArguments(
            output_dir=str(checkpoint_dir / run_id),
            per_device_train_batch_size=train_batch_size,
            gradient_accumulation_steps=4,
            num_train_epochs=1.0,
            learning_rate=2e-4,
            weight_decay=0.0,
            warmup_ratio=0.05,
            lr_scheduler_type="cosine",
            logging_steps=25,
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
            data_collator=RegressionCollator(tokenizer),
        )
        train_output = trainer.train()
        test_metrics = evaluate(model, tokenizer, test_rows, eval_batch_size, max_length)
        empirical_alpha = float(np.corrcoef([item["u"] for item in train_rows], [item["s_lambda"] for item in train_rows])[0, 1])
        metrics = {
            "run_id": run_id,
            "kind": "theorem_phase_regression",
            "lambda_int": float(lambda_int),
            "eta": float(eta),
            "seed": int(seed),
            "train_count": len(train_rows),
            "corrupted_count": int(sum(item["corrupted"] for item in train_rows)),
            "train_loss": float(train_output.training_loss),
            "train_runtime_seconds": float(train_output.metrics.get("train_runtime", float("nan"))),
            "test": test_metrics,
            "empirical_train_u_s_correlation": empirical_alpha,
            "model": "Qwen2.5-1.5B-Instruct",
            "data_model": {
                "alpha": ALPHA,
                "alpha_e": ALPHA_E,
                "sigma_U_squared": SIGMA_U2,
                "sigma_S_squared": SIGMA_S2,
                "sigma_Y_squared": SIGMA_Y2,
            },
            "training": {
                "head": "scalar regression head on final non-padding hidden state",
                "loss": "squared error",
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "target_modules": ["q_proj", "v_proj"],
                "learning_rate": 2e-4,
                "epochs": 1,
                "micro_batch_size": train_batch_size,
                "gradient_accumulation_steps": 4,
                "max_length": max_length,
                "dtype": "bf16" if bf16 else "fp32",
            },
            "evaluation_rule": "scalar_regression_last_hidden_v1",
            "status": "completed",
            "elapsed_seconds": time.time() - start,
        }
        write_json(result_path, metrics)
        return metrics
    except Exception as error:
        failure = {
            "run_id": run_id,
            "kind": "theorem_phase_regression",
            "lambda_int": float(lambda_int),
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
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--lambda-values", type=float, nargs="+", default=DEFAULT_LAMBDAS)
    parser.add_argument("--train-count", type=int, default=1000)
    parser.add_argument("--test-count", type=int, default=4000)
    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    prepare_raw(args.raw_dir, args.seeds, args.train_count, args.test_count)
    if args.prepare_only:
        return
    args.result_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    planned = []
    for lambda_int in args.lambda_values:
        eta_values = DEFAULT_ETA_VALUES.get(round(lambda_int, 2), [0.0, 0.04, 0.08, 0.10, 0.12])
        for eta in eta_values:
            for seed in args.seeds:
                planned.append((lambda_int, eta, seed))
    baseline = [(0.0, 0.0, seed) for seed in args.seeds]
    planned = baseline + planned
    results = []
    for index, (lambda_int, eta, seed) in enumerate(planned, start=1):
        print(f"run={index}/{len(planned)} lambda_int={lambda_int} eta={eta} seed={seed}", flush=True)
        results.append(
            train_one(
                args.model_path,
                args.raw_dir,
                args.result_dir,
                args.checkpoint_dir,
                lambda_int,
                eta,
                seed,
                args.train_batch_size,
                args.eval_batch_size,
                args.max_length,
            )
        )
    manifest = {
        "mode": "theorem_phase_regression",
        "seeds": args.seeds,
        "lambda_values": args.lambda_values,
        "eta_values_by_lambda": {str(k): v for k, v in DEFAULT_ETA_VALUES.items()},
        "planned_runs": len(planned),
        "completed_runs": sum(item.get("status") == "completed" for item in results),
        "failed_runs": sum(item.get("status") == "failed" for item in results),
        "results": results,
    }
    write_json(args.result_dir / "manifest_theorem_phase.json", manifest)


if __name__ == "__main__":
    main()
