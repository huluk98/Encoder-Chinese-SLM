#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer


BASE_CHECKPOINT = "runs/h20-8gpu-bert-0p2b-mlm-deepspeed/latest"
TRAINING_JSON = "data/scenic/SCENIC_full_training_dataset.json"
BENCHMARK_JSON = "data/scenic/iot_instruction_benchmark_200.json"
OUTPUT_DIR = "eval_results/scenic_base_model/em1_em5_check"

USER_TOKEN = "<|user|>"
ASSISTANT_TOKEN = "<|assistant|>"
EOS_TOKEN = "<|eos|>"
BATCH_SIZE = 32
MAX_LENGTH = 128
EVAL_DTYPE = "auto"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check the MLM-pretrained base encoder on SCENIC response selection "
            "with EM@1 and EM@5 cloze scoring."
        )
    )
    parser.add_argument("--checkpoint", default=BASE_CHECKPOINT, help="Base AutoModelForMaskedLM checkpoint.")
    parser.add_argument("--tokenizer", default=None, help="Tokenizer path. Defaults to --checkpoint.")
    parser.add_argument("--training-json", default=TRAINING_JSON)
    parser.add_argument("--benchmark-json", default=BENCHMARK_JSON)
    parser.add_argument(
        "--candidate-json",
        default=TRAINING_JSON,
        help="JSON whose unique response values define the response candidate set.",
    )
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--dtype", choices=("auto", "fp32", "bf16", "fp16"), default=EVAL_DTYPE)
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0, mps, or cpu.")
    parser.add_argument("--normalize-by-length", action="store_true", help="Average candidate log score by token length.")
    parser.add_argument("--no-eos-score", action="store_true", help="Do not include <|eos|> in candidate scoring.")
    parser.add_argument("--training-limit", type=int, default=None, help="Optional row limit for quick checks.")
    parser.add_argument("--benchmark-limit", type=int, default=None, help="Optional row limit for quick checks.")
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def dtype_for(name: str, device: torch.device) -> torch.dtype:
    if name == "auto":
        return torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    if name == "bf16":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    if name == "fp16":
        return torch.float16 if device.type == "cuda" else torch.float32
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unknown dtype: {name}")


def read_json_list(path: str | Path) -> list[dict[str, Any]]:
    data = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list.")
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"{path}:{index} must contain a JSON object.")
        rows.append(item)
    return rows


def clean_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def prompt_from_row(row: dict[str, Any]) -> str:
    for field in ("prompt", "anchor", "instruction", "input", "text", "query"):
        text = clean_text(row.get(field))
        if text:
            return text
    return ""


def load_eval_rows(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(read_json_list(path)):
        prompt = prompt_from_row(row)
        response = clean_text(row.get("response"))
        if not prompt:
            raise ValueError(f"{path}:{index} does not contain a prompt-like field.")
        rows.append({"index": index, "prompt": prompt, "expected_response": response, "raw": row})
        if limit is not None and len(rows) >= int(limit):
            break
    return rows


def load_candidate_responses(path: str | Path) -> list[str]:
    responses = sorted({clean_text(row.get("response")) for row in read_json_list(path) if clean_text(row.get("response"))})
    if not responses:
        raise ValueError(f"No non-empty response values found in {path}.")
    return responses


def format_prefix(prompt: str) -> str:
    return f"{USER_TOKEN}\n{prompt}\n{ASSISTANT_TOKEN}\n"


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "case"


def batched(items: list[dict[str, Any]], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def encode_response(tokenizer: Any, response: str, include_eos: bool) -> list[int]:
    text = f"{response}{EOS_TOKEN}" if include_eos else response
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if not ids:
        raise ValueError(f"Response encoded to no tokens: {response!r}")
    return [int(token_id) for token_id in ids]


def build_candidate_groups(
    tokenizer: Any,
    responses: list[str],
    include_eos: bool,
    max_length: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    groups_by_length: dict[int, dict[str, Any]] = {}
    kept_responses: list[str] = []
    for response in responses:
        ids = encode_response(tokenizer, response, include_eos)
        if len(ids) >= max_length:
            continue
        candidate_index = len(kept_responses)
        kept_responses.append(response)
        group = groups_by_length.setdefault(len(ids), {"length": len(ids), "indices": [], "ids": []})
        group["indices"].append(candidate_index)
        group["ids"].append(ids)
    if not kept_responses:
        raise ValueError("No candidate responses fit under --max-length.")

    groups: list[dict[str, Any]] = []
    for length in sorted(groups_by_length):
        group = groups_by_length[length]
        groups.append(
            {
                "length": int(group["length"]),
                "indices": torch.tensor(group["indices"], dtype=torch.long),
                "ids": torch.tensor(group["ids"], dtype=torch.long),
            }
        )
    return groups, kept_responses


def trim_prefix(prefix_ids: list[int], candidate_len: int, max_length: int) -> list[int]:
    max_prefix_len = max(1, int(max_length) - int(candidate_len))
    if len(prefix_ids) <= max_prefix_len:
        return prefix_ids
    return prefix_ids[-max_prefix_len:]


def score_rows(
    model: AutoModelForMaskedLM,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    candidate_groups: list[dict[str, Any]],
    candidate_responses: list[str],
    device: torch.device,
    max_length: int,
    batch_size: int,
    normalize_by_length: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if tokenizer.mask_token_id is None:
        raise ValueError("Tokenizer must define a mask token for base MLM cloze scoring.")
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    label_set = set(candidate_responses)
    predictions: list[dict[str, Any]] = []
    predicted_counts: Counter[str] = Counter()
    expected_counts: Counter[str] = Counter()
    total = 0
    scored = 0
    covered = 0
    em1_correct = 0
    em5_correct = 0

    batch_count = math.ceil(len(rows) / int(batch_size)) if rows else 0
    row_offsets = torch.arange(0, int(batch_size), dtype=torch.long, device=device)
    with torch.inference_mode():
        for batch in tqdm(batched(rows, int(batch_size)), total=batch_count, desc="base mlm em", unit="batch"):
            prefixes = [
                tokenizer(format_prefix(item["prompt"]), add_special_tokens=False)["input_ids"]
                for item in batch
            ]
            scores = torch.full((len(batch), len(candidate_responses)), -torch.inf, dtype=torch.float32)

            for group in candidate_groups:
                length = int(group["length"])
                trimmed_prefixes = [trim_prefix([int(token_id) for token_id in prefix], length, int(max_length)) for prefix in prefixes]
                sequence_lengths = [len(prefix) + length for prefix in trimmed_prefixes]
                batch_max_len = max(sequence_lengths)
                input_ids = torch.full((len(batch), batch_max_len), int(pad_id), dtype=torch.long)
                attention_mask = torch.zeros((len(batch), batch_max_len), dtype=torch.long)
                mask_positions = torch.zeros((len(batch), length), dtype=torch.long)

                for row_index, prefix_ids in enumerate(trimmed_prefixes):
                    row_ids = prefix_ids + [int(tokenizer.mask_token_id)] * length
                    input_ids[row_index, : len(row_ids)] = torch.tensor(row_ids, dtype=torch.long)
                    attention_mask[row_index, : len(row_ids)] = 1
                    mask_positions[row_index] = torch.arange(len(prefix_ids), len(prefix_ids) + length, dtype=torch.long)

                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                mask_positions = mask_positions.to(device)
                candidate_ids = group["ids"].to(device)
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                log_probs = torch.log_softmax(logits.float(), dim=-1)

                group_scores = torch.zeros((len(batch), candidate_ids.shape[0]), dtype=torch.float32, device=device)
                active_offsets = row_offsets[: len(batch)]
                for position in range(length):
                    position_log_probs = log_probs[active_offsets, mask_positions[:, position]]
                    group_scores += position_log_probs[:, candidate_ids[:, position]]
                if normalize_by_length:
                    group_scores = group_scores / float(length)

                scores[:, group["indices"]] = group_scores.cpu()

            top_scores, top_indices = torch.topk(scores, k=min(5, scores.shape[1]), dim=-1)
            for item, row_top_indices, row_top_scores in zip(batch, top_indices.tolist(), top_scores.tolist()):
                expected = item["expected_response"]
                top5 = [
                    {"response": candidate_responses[int(index)], "score": float(score)}
                    for index, score in zip(row_top_indices, row_top_scores, strict=True)
                ]
                predicted = top5[0]["response"] if top5 else ""
                top5_responses = {entry["response"] for entry in top5}
                expected_covered = bool(expected and expected in label_set)
                predicted_counts[predicted] += 1
                if expected:
                    expected_counts[expected] += 1
                    scored += 1
                    covered += int(expected_covered)
                    em1_correct += int(predicted == expected)
                    em5_correct += int(expected in top5_responses)
                total += 1
                predictions.append(
                    {
                        "index": item["index"],
                        "prompt": item["prompt"],
                        "expected_response": expected,
                        "expected_in_candidate_space": expected_covered,
                        "predicted_response": predicted,
                        "correct": bool(expected and predicted == expected),
                        "top5": top5,
                    }
                )

    top_predictions = [
        {"response": response, "count": count, "share": count / total if total else None}
        for response, count in predicted_counts.most_common(20)
    ]
    top_expected = [
        {"response": response, "count": count, "share": count / total if total else None}
        for response, count in expected_counts.most_common(20)
    ]
    top_prediction = top_predictions[0] if top_predictions else {}
    summary = {
        "rows": total,
        "scored_rows": scored,
        "candidate_count": len(candidate_responses),
        "candidate_length_groups": len(candidate_groups),
        "expected_in_candidate_space": covered,
        "label_space_coverage": covered / scored if scored else None,
        "em1_correct": em1_correct,
        "em5_correct": em5_correct,
        "em1": em1_correct / scored if scored else None,
        "em5": em5_correct / scored if scored else None,
        "prediction_unique_count": len(predicted_counts),
        "prediction_unique_ratio": len(predicted_counts) / total if total else None,
        "top_prediction": top_prediction.get("response"),
        "top_prediction_count": top_prediction.get("count"),
        "top_prediction_share": top_prediction.get("share"),
        "top_predictions": top_predictions,
        "expected_unique_count": len(expected_counts),
        "top_expected_responses": top_expected,
    }
    return predictions, summary


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "model",
        "dataset",
        "em1",
        "em5",
        "em1_correct",
        "em5_correct",
        "scored_rows",
        "rows",
        "candidate_count",
        "label_space_coverage",
        "prediction_unique_count",
        "prediction_unique_ratio",
        "top_prediction",
        "top_prediction_count",
        "top_prediction_share",
        "checkpoint",
        "json",
        "summary_output",
        "predictions_output",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def pct(value: Any) -> str:
    return "n/a" if value is None else f"{float(value) * 100:.2f}%"


def print_table(rows: list[dict[str, Any]]) -> None:
    print("\n[base-em] EM@1 / EM@5")
    print(f"{'dataset':18s} {'EM@1':>9s} {'EM@5':>9s} {'scored':>12s} {'cands':>8s} {'unique':>10s} {'top share':>10s}")
    for row in rows:
        scored = f"{row.get('scored_rows', 0)}/{row.get('rows', 0)}"
        print(
            f"{row['dataset']:18s} {pct(row.get('em1')):>9s} {pct(row.get('em5')):>9s} "
            f"{scored:>12s} {str(row.get('candidate_count')):>8s} "
            f"{str(row.get('prediction_unique_count')):>10s} {pct(row.get('top_prediction_share')):>10s}"
        )


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    dtype = dtype_for(args.dtype, device)
    tokenizer_source = args.tokenizer or args.checkpoint
    tokenizer = AutoTokenizer.from_pretrained(str(Path(tokenizer_source).expanduser()), use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(str(Path(args.checkpoint).expanduser()))
    model.to(device=device, dtype=dtype)
    model.eval()

    candidate_responses = load_candidate_responses(args.candidate_json)
    candidate_groups, candidate_responses = build_candidate_groups(
        tokenizer=tokenizer,
        responses=candidate_responses,
        include_eos=not bool(args.no_eos_score),
        max_length=int(args.max_length),
    )

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = [
        ("training_data", args.training_json, args.training_limit),
        ("benchmark_200", args.benchmark_json, args.benchmark_limit),
    ]
    summaries: list[dict[str, Any]] = []
    for dataset_name, json_path, limit in cases:
        rows = load_eval_rows(json_path, limit=limit)
        predictions, summary = score_rows(
            model=model,
            tokenizer=tokenizer,
            rows=rows,
            candidate_groups=candidate_groups,
            candidate_responses=candidate_responses,
            device=device,
            max_length=int(args.max_length),
            batch_size=max(1, int(args.batch_size)),
            normalize_by_length=bool(args.normalize_by_length),
        )
        predictions_output = output_dir / f"{safe_name(dataset_name)}_predictions.jsonl"
        summary_output = output_dir / f"{safe_name(dataset_name)}_summary.json"
        write_jsonl(predictions_output, predictions)
        row = {
            "model": "base_mlm",
            "dataset": dataset_name,
            "checkpoint": args.checkpoint,
            "json": json_path,
            "summary_output": str(summary_output),
            "predictions_output": str(predictions_output),
            "device": str(device),
            "dtype": str(dtype).replace("torch.", ""),
            "candidate_json": args.candidate_json,
            "include_eos_score": not bool(args.no_eos_score),
            "normalize_by_length": bool(args.normalize_by_length),
            **summary,
        }
        summary_output.write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        summaries.append(row)

    summary_json = output_dir / "base_model_em1_em5_summary.json"
    summary_csv = output_dir / "base_model_em1_em5_summary.csv"
    summary_json.write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(summary_csv, summaries)
    print_table(summaries)
    print(f"\n[base-em] wrote {summary_json}")
    print(f"[base-em] wrote {summary_csv}")


if __name__ == "__main__":
    main()
