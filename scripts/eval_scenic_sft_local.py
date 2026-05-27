#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from chatlm_encoder.scenic_sft import load_scenic_checkpoint, prompt_from_row, read_json_list  # noqa: E402


# Edit these paths if you want to run the evaluator without command-line flags.
LOCAL_JSON_PATH = "data/scenic/iot_instruction_benchmark_200.json"
CHECKPOINT_DIR = "runs/scenic-sft-training-dataset/latest"
OUTPUT_PATH = "eval_results/scenic_sft/benchmark_200_predictions.jsonl"
SUMMARY_OUTPUT_PATH = "eval_results/scenic_sft/benchmark_200_summary.json"
MAX_LENGTH = 128
BATCH_SIZE = 128
EVAL_DTYPE = "auto"
GROUP_FIELDS = ("difficulty", "task_type", "source")


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_eval_dtype(dtype_name: str, device: torch.device) -> torch.dtype:
    dtype_name = str(dtype_name).lower()
    if dtype_name == "auto":
        return torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    if dtype_name in {"fp32", "float32"}:
        return torch.float32
    if dtype_name in {"bf16", "bfloat16"}:
        if device.type != "cuda":
            raise ValueError("--dtype bf16 is only supported for CUDA evaluation in this script.")
        return torch.bfloat16
    raise ValueError(f"Unknown dtype: {dtype_name}. Use auto, fp32, or bf16.")


def load_eval_rows(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    for index, row in enumerate(read_json_list(path)):
        prompt = prompt_from_row(row)
        response = "" if row.get("response") is None else str(row.get("response")).strip()
        if not prompt:
            raise ValueError(f"{path}:{index} does not contain a prompt or anchor field.")
        rows.append({"index": index, "prompt": prompt, "expected_response": response, "raw": row})
    return rows


def batched(items: list[dict[str, Any]], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def new_metric_bucket() -> dict[str, int]:
    return {
        "rows": 0,
        "scored_rows": 0,
        "expected_in_label_space": 0,
        "exact_match_correct": 0,
        "top5_correct": 0,
    }


def update_metric_bucket(bucket: dict[str, int], expected: str, predicted: str, top5: set[str], label_set: set[str]) -> None:
    bucket["rows"] += 1
    if not expected:
        return
    bucket["scored_rows"] += 1
    bucket["expected_in_label_space"] += int(expected in label_set)
    bucket["exact_match_correct"] += int(predicted == expected)
    bucket["top5_correct"] += int(expected in top5)


def summarize_bucket(bucket: dict[str, int]) -> dict[str, int | float | None]:
    scored = int(bucket["scored_rows"])
    return {
        **bucket,
        "label_space_coverage": bucket["expected_in_label_space"] / scored if scored else None,
        "exact_match_accuracy": bucket["exact_match_correct"] / scored if scored else None,
        "top5_accuracy": bucket["top5_correct"] / scored if scored else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a SCENIC encoder SFT checkpoint on a local JSON file.")
    parser.add_argument("--json", default=LOCAL_JSON_PATH, help="Local JSON list with prompt/response or anchor/response rows.")
    parser.add_argument("--checkpoint", default=CHECKPOINT_DIR, help="SCENIC SFT checkpoint directory.")
    parser.add_argument("--output", default=OUTPUT_PATH, help="Prediction JSONL output path.")
    parser.add_argument("--summary-output", default=SUMMARY_OUTPUT_PATH, help="Summary JSON output path.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--dtype", default=EVAL_DTYPE, choices=("auto", "fp32", "bf16"), help="Model dtype for evaluation.")
    args = parser.parse_args()

    device = select_device()
    model, tokenizer, label2response = load_scenic_checkpoint(args.checkpoint, device=device)
    eval_dtype = resolve_eval_dtype(args.dtype, device)
    model.to(device=device, dtype=eval_dtype)
    model.eval()
    label_set = set(label2response)
    rows = load_eval_rows(args.json)
    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    scored = 0
    correct = 0
    top5_correct = 0
    expected_in_label_space = 0
    grouped_metrics: dict[str, defaultdict[str, dict[str, int]]] = {
        field: defaultdict(new_metric_bucket) for field in GROUP_FIELDS
    }

    batch_count = math.ceil(len(rows) / int(args.batch_size)) if rows else 0
    with output_path.open("w", encoding="utf-8") as handle, torch.no_grad():
        for batch in tqdm(batched(rows, int(args.batch_size)), total=batch_count, desc="eval scenic", unit="batch"):
            prompts = [item["prompt"] for item in batch]
            tokens = tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=int(args.max_length),
                return_tensors="pt",
            )
            tokens = {key: value.to(device) for key, value in tokens.items()}
            logits = model(tokens)["logits"]
            probabilities_all = torch.softmax(logits, dim=-1)
            top_probs, top_indices = torch.topk(probabilities_all, k=min(5, logits.shape[-1]), dim=-1)
            probabilities = top_probs.detach().cpu().tolist()
            top_indices_list = top_indices.detach().cpu().tolist()

            for item, top_ids, top_probs in zip(batch, top_indices_list, probabilities):
                predicted = label2response[int(top_ids[0])]
                expected = item["expected_response"]
                top5_responses = {label2response[int(label_id)] for label_id in top_ids}
                expected_covered = bool(expected and expected in label_set)
                result = {
                    "index": item["index"],
                    "prompt": item["prompt"],
                    "expected_response": expected,
                    "expected_in_label_space": expected_covered,
                    "predicted_response": predicted,
                    "correct": bool(expected and predicted == expected),
                    "top5": [
                        {"response": label2response[int(label_id)], "score": float(score)}
                        for label_id, score in zip(top_ids, top_probs)
                    ],
                }
                for field in ("id", "difficulty", "task_type", "source", "response_action_count", "device_term_count"):
                    if field in item["raw"]:
                        result[field] = item["raw"][field]
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                total += 1
                if expected:
                    scored += 1
                    expected_in_label_space += int(expected_covered)
                    correct += int(predicted == expected)
                    top5_correct += int(expected in top5_responses)
                for field, buckets in grouped_metrics.items():
                    value = item["raw"].get(field)
                    if value is not None and str(value).strip():
                        update_metric_bucket(buckets[str(value)], expected, predicted, top5_responses, label_set)

    summary = {
        "checkpoint": str(args.checkpoint),
        "json": str(args.json),
        "device": str(device),
        "dtype": str(eval_dtype).replace("torch.", ""),
        "predictions_output": str(output_path),
        "rows": total,
        "scored_rows": scored,
        "expected_in_label_space": expected_in_label_space,
        "label_space_coverage": expected_in_label_space / scored if scored else None,
        "exact_match_correct": correct,
        "exact_match_accuracy": correct / scored if scored else None,
        "top5_correct": top5_correct,
        "top5_accuracy": top5_correct / scored if scored else None,
        "groups": {
            field: {
                value: summarize_bucket(bucket)
                for value, bucket in sorted(buckets.items(), key=lambda item: item[0])
            }
            for field, buckets in grouped_metrics.items()
            if buckets
        },
    }
    summary_path = Path(args.summary_output).expanduser()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"checkpoint: {args.checkpoint}")
    print(f"json: {args.json}")
    print(f"device: {device}")
    print(f"dtype: {str(eval_dtype).replace('torch.', '')}")
    print(f"output: {output_path}")
    print(f"summary_output: {summary_path}")
    print(f"rows: {total:,}")
    if scored:
        print(f"label_space_coverage: {expected_in_label_space / scored:.6f} ({expected_in_label_space:,}/{scored:,})")
        print(f"exact_accuracy: {correct / scored:.6f} ({correct:,}/{scored:,})")
        print(f"top5_accuracy: {top5_correct / scored:.6f} ({top5_correct:,}/{scored:,})")
    else:
        print("No expected response fields found; wrote predictions only.")


if __name__ == "__main__":
    main()
