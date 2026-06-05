#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from chatlm_encoder.scenic_sft import prompt_from_row, read_json_list  # noqa: E402


LOCAL_JSON_PATH = "data/scenic/iot_instruction_benchmark_200.json"
CHECKPOINT_DIR = "runs/scenic-sft-training-dataset/latest"
ONNX_MODEL = "runs/scenic-onnx-nvidia/onnx/fp16_dense/model.onnx"
OUTPUT_PATH = "eval_results/scenic_sft/onnx_nvidia/fp16_dense/benchmark_predictions.jsonl"
SUMMARY_OUTPUT_PATH = "eval_results/scenic_sft/onnx_nvidia/fp16_dense/benchmark_summary.json"
MAX_LENGTH = 128
BATCH_SIZE = 128
GROUP_FIELDS = ("difficulty", "task_type", "source")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a SCENIC encoder SFT ONNX response selector.")
    parser.add_argument("--json", default=LOCAL_JSON_PATH, help="Local JSON list with prompt/response or anchor/response rows.")
    parser.add_argument("--checkpoint", default=CHECKPOINT_DIR, help="SCENIC checkpoint directory for tokenizer and labels.")
    parser.add_argument("--onnx", default=ONNX_MODEL, help="ONNX model path.")
    parser.add_argument("--output", default=OUTPUT_PATH, help="Prediction JSONL output path.")
    parser.add_argument("--summary-output", default=SUMMARY_OUTPUT_PATH, help="Summary JSON output path.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument(
        "--providers",
        default="auto",
        choices=("auto", "cpu", "cuda", "tensorrt"),
        help="ONNX Runtime execution providers.",
    )
    parser.add_argument(
        "--trt-engine-cache-dir",
        default=None,
        help="Optional TensorRT EP engine cache directory when --providers tensorrt is used.",
    )
    return parser.parse_args()


def require_onnxruntime():
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise SystemExit("Missing optional dependency 'onnxruntime'. Install it with: pip install onnxruntime") from exc
    return ort


def providers_for(ort: Any, requested: str, trt_engine_cache_dir: str | None = None) -> list[Any]:
    available = set(ort.get_available_providers())
    if requested == "cpu":
        return ["CPUExecutionProvider"]
    if requested == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise ValueError("CUDAExecutionProvider is not available in this onnxruntime install.")
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if requested == "tensorrt":
        if "TensorrtExecutionProvider" not in available:
            raise ValueError("TensorrtExecutionProvider is not available in this onnxruntime install.")
        options = {"trt_fp16_enable": "1"}
        if trt_engine_cache_dir:
            cache_dir = Path(trt_engine_cache_dir).expanduser()
            cache_dir.mkdir(parents=True, exist_ok=True)
            options.update(
                {
                    "trt_engine_cache_enable": "1",
                    "trt_engine_cache_path": str(cache_dir),
                }
            )
        providers: list[Any] = [("TensorrtExecutionProvider", options)]
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        return providers
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def load_eval_rows(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    for index, row in enumerate(read_json_list(path)):
        prompt = prompt_from_row(row)
        response = "" if row.get("response") is None else str(row.get("response")).strip()
        if not prompt:
            raise ValueError(f"{path}:{index} does not contain a prompt or anchor field.")
        rows.append({"index": index, "prompt": prompt, "expected_response": response, "raw": row})
    return rows


def load_labels(checkpoint: str | Path) -> list[str]:
    path = Path(checkpoint).expanduser() / "label2response.json"
    labels = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(labels, list):
        raise ValueError(f"{path} must contain a JSON list.")
    return [str(item) for item in labels]


def batched(items: list[dict[str, Any]], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def ensure_token_type_ids(encoded: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if "token_type_ids" not in encoded:
        encoded["token_type_ids"] = np.zeros_like(encoded["input_ids"], dtype=np.int64)
    return encoded


def softmax_topk(logits: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    k = min(k, logits.shape[-1])
    partition = np.argpartition(-logits, kth=k - 1, axis=-1)[:, :k]
    partition_scores = np.take_along_axis(logits, partition, axis=-1)
    order = np.argsort(-partition_scores, axis=-1)
    top_indices = np.take_along_axis(partition, order, axis=-1)
    top_logits = np.take_along_axis(logits, top_indices, axis=-1)

    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    denom = exp.sum(axis=-1, keepdims=True)
    probabilities = np.take_along_axis(exp / denom, top_indices, axis=-1)
    return top_indices, probabilities


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
    args = parse_args()
    ort = require_onnxruntime()
    onnx_path = Path(args.onnx).expanduser()
    checkpoint_path = Path(args.checkpoint).expanduser()
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path), use_fast=True)
    label2response = load_labels(checkpoint_path)
    label_set = set(label2response)
    rows = load_eval_rows(args.json)

    session = ort.InferenceSession(
        str(onnx_path),
        providers=providers_for(ort, args.providers, args.trt_engine_cache_dir),
    )
    input_names = {item.name for item in session.get_inputs()}
    providers = session.get_providers()
    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    scored = 0
    correct = 0
    top5_correct = 0
    expected_in_label_space = 0
    predicted_counts: Counter[str] = Counter()
    expected_counts: Counter[str] = Counter()
    grouped_metrics: dict[str, defaultdict[str, dict[str, int]]] = {
        field: defaultdict(new_metric_bucket) for field in GROUP_FIELDS
    }

    batch_count = math.ceil(len(rows) / int(args.batch_size)) if rows else 0
    with output_path.open("w", encoding="utf-8") as handle:
        for batch in tqdm(batched(rows, int(args.batch_size)), total=batch_count, desc="eval scenic onnx", unit="batch"):
            prompts = [item["prompt"] for item in batch]
            encoded = tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=int(args.max_length),
                return_tensors="np",
            )
            encoded = ensure_token_type_ids(dict(encoded))
            feed = {
                name: encoded[name].astype(np.int64, copy=False)
                for name in ("input_ids", "attention_mask", "token_type_ids")
                if name in input_names
            }
            logits = session.run(["logits"], feed)[0]
            top_indices, probabilities = softmax_topk(logits, k=5)

            for item, top_ids, top_probs in zip(batch, top_indices.tolist(), probabilities.tolist()):
                predicted = label2response[int(top_ids[0])]
                expected = item["expected_response"]
                top5_responses = {label2response[int(label_id)] for label_id in top_ids}
                expected_covered = bool(expected and expected in label_set)
                predicted_counts[predicted] += 1
                if expected:
                    expected_counts[expected] += 1
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

    top_predictions = [
        {"response": response, "count": count, "share": count / total if total else None}
        for response, count in predicted_counts.most_common(20)
    ]
    top_expected_responses = [
        {"response": response, "count": count, "share": count / total if total else None}
        for response, count in expected_counts.most_common(20)
    ]
    top_prediction = top_predictions[0] if top_predictions else {}

    summary = {
        "onnx": str(onnx_path),
        "checkpoint": str(checkpoint_path),
        "json": str(args.json),
        "providers": providers,
        "predictions_output": str(output_path),
        "rows": total,
        "scored_rows": scored,
        "prediction_unique_count": len(predicted_counts),
        "prediction_unique_ratio": len(predicted_counts) / total if total else None,
        "top_prediction": top_prediction.get("response"),
        "top_prediction_count": top_prediction.get("count"),
        "top_prediction_share": top_prediction.get("share"),
        "top_predictions": top_predictions,
        "expected_unique_count": len(expected_counts),
        "top_expected_responses": top_expected_responses,
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

    print(f"onnx: {onnx_path}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"json: {args.json}")
    print(f"providers: {providers}")
    print(f"output: {output_path}")
    print(f"summary_output: {summary_path}")
    print(f"rows: {total:,}")
    if scored:
        print(f"label_space_coverage: {expected_in_label_space / scored:.6f} ({expected_in_label_space:,}/{scored:,})")
        print(f"exact_accuracy: {correct / scored:.6f} ({correct:,}/{scored:,})")
        print(f"top5_accuracy: {top5_correct / scored:.6f} ({top5_correct:,}/{scored:,})")
        print(f"prediction_unique_count: {len(predicted_counts):,}/{total:,}")
        if top_predictions:
            print(
                "top_prediction: "
                f"{top_predictions[0]['response']} "
                f"({top_predictions[0]['count']:,}/{total:,}, {top_predictions[0]['share']:.6f})"
            )
    else:
        print("No expected response fields found; wrote predictions only.")


if __name__ == "__main__":
    main()
