#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


BENCHMARK_JSON = "data/scenic/iot_instruction_benchmark_200.json"
TRAINING_DATASET_JSON = "data/scenic/SCENIC_full_training_dataset.json"
CONTRASTIVE_ANCHOR_JSON = "data/scenic/SCENIC_full_anchor_positive_negative.json"

TRAINING_DATASET_CHECKPOINT = "runs/scenic-sft-training-dataset/latest"
CONTRASTIVE_DATASET_CHECKPOINT = "runs/scenic-sft-contrastive-dataset/latest"

OUTPUT_DIR = "eval_results/scenic_sft/comparison"
BATCH_SIZE = 128
MAX_LENGTH = 128
EVAL_DTYPE = "auto"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate both SCENIC SFT models on the 200-row benchmark and each model's own "
            "training source for retention."
        )
    )
    parser.add_argument("--benchmark-json", default=BENCHMARK_JSON)
    parser.add_argument("--training-json", default=TRAINING_DATASET_JSON)
    parser.add_argument("--contrastive-json", default=CONTRASTIVE_ANCHOR_JSON)
    parser.add_argument("--training-checkpoint", default=TRAINING_DATASET_CHECKPOINT)
    parser.add_argument("--contrastive-checkpoint", default=CONTRASTIVE_DATASET_CHECKPOINT)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--dtype", default=EVAL_DTYPE, choices=("auto", "fp32", "bf16"))
    parser.add_argument("--skip-missing", action="store_true", help="Skip missing checkpoints instead of failing.")
    return parser.parse_args()


def sanitize(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value).strip("_")


def cases(args: argparse.Namespace) -> list[dict[str, str]]:
    return [
        {
            "model": "training_dataset_model",
            "dataset": "benchmark_200",
            "checkpoint": args.training_checkpoint,
            "json": args.benchmark_json,
        },
        {
            "model": "training_dataset_model",
            "dataset": "training_dataset_retention",
            "checkpoint": args.training_checkpoint,
            "json": args.training_json,
        },
        {
            "model": "contrastive_anchor_model",
            "dataset": "benchmark_200",
            "checkpoint": args.contrastive_checkpoint,
            "json": args.benchmark_json,
        },
        {
            "model": "contrastive_anchor_model",
            "dataset": "contrastive_anchor_retention",
            "checkpoint": args.contrastive_checkpoint,
            "json": args.contrastive_json,
        },
    ]


def run_eval(case: dict[str, str], args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    case_name = f"{sanitize(case['model'])}__{sanitize(case['dataset'])}"
    predictions_path = output_dir / f"{case_name}_predictions.jsonl"
    summary_path = output_dir / f"{case_name}_summary.json"
    command = [
        sys.executable,
        "scripts/eval_scenic_sft_local.py",
        "--json",
        case["json"],
        "--checkpoint",
        case["checkpoint"],
        "--output",
        str(predictions_path),
        "--summary-output",
        str(summary_path),
        "--batch-size",
        str(args.batch_size),
        "--max-length",
        str(args.max_length),
        "--dtype",
        args.dtype,
    ]
    print(f"[scenic-eval-suite] {case['model']} on {case['dataset']}")
    subprocess.run(command, check=True)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        "model": case["model"],
        "dataset": case["dataset"],
        "checkpoint": case["checkpoint"],
        "json": case["json"],
        "summary_output": str(summary_path),
        **summary,
    }


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "model",
        "dataset",
        "rows",
        "scored_rows",
        "label_space_coverage",
        "exact_match_accuracy",
        "top5_accuracy",
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


def write_group_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "model",
        "dataset",
        "group_field",
        "group_value",
        "rows",
        "scored_rows",
        "label_space_coverage",
        "exact_match_accuracy",
        "top5_accuracy",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            for group_field, values in row.get("groups", {}).items():
                for group_value, metrics in values.items():
                    writer.writerow(
                        {
                            "model": row["model"],
                            "dataset": row["dataset"],
                            "group_field": group_field,
                            "group_value": group_value,
                            "rows": metrics.get("rows"),
                            "scored_rows": metrics.get("scored_rows"),
                            "label_space_coverage": metrics.get("label_space_coverage"),
                            "exact_match_accuracy": metrics.get("exact_match_accuracy"),
                            "top5_accuracy": metrics.get("top5_accuracy"),
                        }
                    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    for case in cases(args):
        checkpoint_path = Path(case["checkpoint"]).expanduser()
        json_path = Path(case["json"]).expanduser()
        if args.skip_missing and (not checkpoint_path.exists() or not json_path.exists()):
            print(f"[scenic-eval-suite] skipping missing case: {case}")
            continue
        summaries.append(run_eval(case, args, output_dir))

    summary_json = output_dir / "comparison_summary.json"
    summary_csv = output_dir / "comparison_summary.csv"
    group_csv = output_dir / "comparison_groups.csv"
    summary_json.write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_summary_csv(summary_csv, summaries)
    write_group_csv(group_csv, summaries)

    print(f"[scenic-eval-suite] wrote {summary_json}")
    print(f"[scenic-eval-suite] wrote {summary_csv}")
    print(f"[scenic-eval-suite] wrote {group_csv}")


if __name__ == "__main__":
    main()
