"""Prepare MultiNLI anchors and generate LLM counterfactual siblings.

The generated files are the frozen input to the post-training runs.  This
separation prevents later training choices from changing the intervention
kernel used by the fixed-budget comparison.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


LABELS = ["entailment", "neutral", "contradiction"]
SOURCE_GENRES = ["telephone", "government", "travel", "fiction", "slate"]
TARGET_GENRES = ["letters", "facetoface", "nineeleven", "oup", "verbatim"]
TARGET_DESCRIPTIONS = {
    "letters": "a personal letter with a polite, written tone",
    "facetoface": "an informal face-to-face conversation",
    "nineeleven": "a concise news report about a public event",
    "oup": "a formal expository or textbook passage",
    "verbatim": "a spoken transcript with natural conversational wording",
}


def write_jsonl(path: pathlib.Path, rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: pathlib.Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def stratified_rows(dataset, count: int, seed: int, used_ids=None) -> List[Dict]:
    used_ids = set(used_ids or [])
    groups = defaultdict(list)
    for row in dataset:
        pair_id = str(row["pairID"])
        if pair_id in used_ids:
            continue
        label = int(row["label"])
        genre = str(row["genre"])
        groups[(genre, label)].append(
            {
                "anchor_id": pair_id,
                "premise": str(row["premise"]).strip(),
                "hypothesis": str(row["hypothesis"]).strip(),
                "label": LABELS[label],
                "source_genre": genre,
            }
        )
    rng = random.Random(seed)
    for values in groups.values():
        rng.shuffle(values)

    keys = sorted(groups)
    selected = []
    while len(selected) < count:
        progressed = False
        for key in keys:
            if groups[key] and len(selected) < count:
                selected.append(groups[key].pop())
                progressed = True
        if not progressed:
            raise RuntimeError(f"Not enough rows for stratified sample of {count}")
    return selected


def generator_prompt(row: Dict, target_genre: str) -> str:
    return (
        "You generate one counterfactual NLI hypothesis. Keep the premise "
        "unchanged and rewrite only the hypothesis. Preserve the exact logical "
        "relation given below. The rewritten hypothesis should sound like "
        f"{TARGET_DESCRIPTIONS[target_genre]}. Output only the rewritten "
        "hypothesis, with no label, explanation, quotation marks, or extra text. "
        "Your entire response must be one rewritten sentence.\n\n"
        f"Premise: {row['premise']}\n"
        f"Original hypothesis: {row['hypothesis']}\n"
        f"Relation to preserve: {row['label']}\n"
        f"Target genre: {target_genre}\n"
        "Rewritten hypothesis:"
    )


def clean_generation(text: str, fallback: str) -> tuple[str, bool]:
    text = text.replace("\r", "").strip()
    for marker in ["Rewritten hypothesis:", "Hypothesis:", "Output:"]:
        if marker in text:
            text = text.split(marker, 1)[1].strip()
    for marker in [
        "To preserve",
        "To ensure",
        "Note:",
        "Explanation:",
        "This rewritten",
        "I have ",
        "Here is",
    ]:
        if marker in text:
            text = text.split(marker, 1)[0].strip()
    if "Label:" in text:
        text = text.split("Label:", 1)[0].strip()
    text = text.split("\n\n", 1)[0].strip(" \t\"'`*_:")
    text = " ".join(text.split())
    valid = (
        3 <= len(text) <= 300
        and "Relation to preserve" not in text
        and not text.lower().startswith(("the rewritten hypothesis", "sure,", "certainly,"))
    )
    return (text if valid else fallback, valid)


def generate_counterfactuals(
    model,
    tokenizer,
    rows: Sequence[Dict],
    target_genres: Sequence[str],
    seed: int,
    batch_size: int,
) -> List[Dict]:
    rng = random.Random(seed)
    prompts = []
    metadata = []
    for row in rows:
        for sibling_index in range(row["k"]):
            target_genre = target_genres[(row["ordinal"] + sibling_index) % len(target_genres)]
            prompts.append(generator_prompt(row, target_genre))
            metadata.append((row, target_genre, sibling_index))

    output_rows = []
    model.eval()
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        batch_meta = metadata[start : start + batch_size]
        encoded = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=40,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        input_width = encoded["input_ids"].shape[1]
        for index, ((row, target_genre, sibling_index), sequence) in enumerate(
            zip(batch_meta, generated)
        ):
            continuation = sequence[input_width:]
            raw = tokenizer.decode(continuation, skip_special_tokens=True)
            hypothesis, valid = clean_generation(raw, row["hypothesis"])
            output_rows.append(
                {
                    "anchor_id": row["anchor_id"],
                    "premise": row["premise"],
                    "hypothesis": hypothesis,
                    "label": row["label"],
                    "source_genre": row["source_genre"],
                    "genre": target_genre,
                    "kind": "counterfactual",
                    "sibling_index": sibling_index,
                    "generation_valid": valid,
                    "raw_generation": raw[:500],
                }
            )
        print(f"generated={min(start + batch_size, len(prompts))}/{len(prompts)}", flush=True)
    rng.shuffle(output_rows)
    return output_rows


def load_local_model(model_path: pathlib.Path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--generation-batch-size", type=int, default=4)
    args = parser.parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("multi_nli")
    anchors = stratified_rows(dataset["train"], 1000, args.seed)
    diagnostic = stratified_rows(
        dataset["train"], 80, args.seed + 11, used_ids={r["anchor_id"] for r in anchors}
    )
    matched = stratified_rows(dataset["validation_matched"], 1000, args.seed + 21)
    mismatched = stratified_rows(dataset["validation_mismatched"], 1000, args.seed + 31)
    write_jsonl(args.output_dir / "anchors.jsonl", anchors)
    write_jsonl(args.output_dir / "eval_matched.jsonl", matched)
    write_jsonl(args.output_dir / "eval_mismatched.jsonl", mismatched)

    factual = [
        {
            **row,
            "genre": row["source_genre"],
            "kind": "factual",
            "sibling_index": 0,
            "generation_valid": True,
        }
        for row in anchors
    ]
    write_jsonl(args.output_dir / "train_factual.jsonl", factual)

    model, tokenizer = load_local_model(args.model_path)
    counterfactual_summary = {}
    for n, k in [(1000, 1), (500, 2), (250, 4), (125, 8)]:
        selected = []
        for ordinal, row in enumerate(anchors[:n]):
            selected.append({**row, "ordinal": ordinal, "k": k})
        rows = generate_counterfactuals(
            model,
            tokenizer,
            selected,
            TARGET_GENRES,
            args.seed + n + k,
            args.generation_batch_size,
        )
        path = args.output_dir / f"train_cf_n{n}_k{k}.jsonl"
        write_jsonl(path, rows)
        counterfactual_summary[f"n{n}_k{k}"] = {
            "rows": len(rows),
            "valid_generations": sum(bool(r["generation_valid"]) for r in rows),
            "path": str(path),
        }

    diagnostic_selected = [
        {**row, "ordinal": ordinal, "k": 4} for ordinal, row in enumerate(diagnostic)
    ]
    diagnostic_rows = generate_counterfactuals(
        model,
        tokenizer,
        diagnostic_selected,
        TARGET_GENRES,
        args.seed + 77,
        args.generation_batch_size,
    )
    write_jsonl(args.output_dir / "diagnostic_cf_k4.jsonl", diagnostic_rows)
    summary = {
        "dataset": "MultiNLI",
        "seed": args.seed,
        "anchor_count": len(anchors),
        "diagnostic_anchor_count": len(diagnostic),
        "matched_eval_count": len(matched),
        "mismatched_eval_count": len(mismatched),
        "target_genres": TARGET_GENRES,
        "counterfactual": counterfactual_summary,
        "diagnostic_rows": len(diagnostic_rows),
        "model_path": str(args.model_path),
    }
    (args.output_dir / "data_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
