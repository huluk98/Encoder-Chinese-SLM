#!/usr/bin/env python
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


# =============================================================================
# EDIT THESE PATHS AND SETTINGS
# =============================================================================

# SFT checkpoints to evaluate. Leave either one as "" to skip that model.
TRAINING_CHECKPOINT = "runs/scenic-sft-training-dataset/latest"
CONTRASTIVE_CHECKPOINT = "runs/scenic-sft-contrastive-dataset/latest"

# Evaluation JSON files.
BENCHMARK_JSON = "data/scenic/iot_instruction_benchmark_200.json"
TRAINING_DATASET_JSON = "data/scenic/SCENIC_full_training_dataset.json"
CONTRASTIVE_DATASET_JSON = "data/scenic/SCENIC_full_anchor_positive_negative.json"

# Output location.
OUTPUT_DIR = "eval_results/scenic_sft/comparison"

# Runtime settings.
BATCH_SIZE = 128
MAX_LENGTH = 128
EVAL_DTYPE = "auto"  # "auto", "fp32", or "bf16"
SKIP_MISSING = True

# =============================================================================
# END EDIT BLOCK
# =============================================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sanitize(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value).strip("_")


def build_cases() -> list[dict[str, str]]:
    cases: list[dict[str, str]] = []
    if TRAINING_CHECKPOINT:
        cases.extend(
            [
                {
                    "model": "training_dataset_model",
                    "dataset": "benchmark_200",
                    "checkpoint": TRAINING_CHECKPOINT,
                    "json": BENCHMARK_JSON,
                },
                {
                    "model": "training_dataset_model",
                    "dataset": "training_dataset_retention",
                    "checkpoint": TRAINING_CHECKPOINT,
                    "json": TRAINING_DATASET_JSON,
                },
            ]
        )
    if CONTRASTIVE_CHECKPOINT:
        cases.extend(
            [
                {
                    "model": "contrastive_anchor_model",
                    "dataset": "benchmark_200",
                    "checkpoint": CONTRASTIVE_CHECKPOINT,
                    "json": BENCHMARK_JSON,
                },
                {
                    "model": "contrastive_anchor_model",
                    "dataset": "contrastive_anchor_retention",
                    "checkpoint": CONTRASTIVE_CHECKPOINT,
                    "json": CONTRASTIVE_DATASET_JSON,
                },
            ]
        )
    return cases


def run_eval(case: dict[str, str], output_dir: Path) -> dict[str, Any]:
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
        str(BATCH_SIZE),
        "--max-length",
        str(MAX_LENGTH),
        "--dtype",
        EVAL_DTYPE,
    ]
    print(f"\n[scenic-sft-eval] {case['model']} on {case['dataset']}", flush=True)
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
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


def print_table(rows: list[dict[str, Any]]) -> None:
    print("\n[scenic-sft-eval] accuracy summary")
    print(f"{'model':28s} {'dataset':30s} {'exact':>10s} {'top5':>10s} {'scored':>12s}")
    for row in rows:
        exact = row.get("exact_match_accuracy")
        top5 = row.get("top5_accuracy")
        exact_text = f"{float(exact) * 100:.2f}%" if exact is not None else "n/a"
        top5_text = f"{float(top5) * 100:.2f}%" if top5 is not None else "n/a"
        scored = f"{row.get('scored_rows', 0)}/{row.get('rows', 0)}"
        print(f"{row['model']:28s} {row['dataset']:30s} {exact_text:>10s} {top5_text:>10s} {scored:>12s}")


def main() -> None:
    output_dir = PROJECT_ROOT / OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    for case in build_cases():
        checkpoint_path = PROJECT_ROOT / case["checkpoint"]
        json_path = PROJECT_ROOT / case["json"]
        if SKIP_MISSING and (not checkpoint_path.exists() or not json_path.exists()):
            print(f"[scenic-sft-eval] skipping missing case: {case}", flush=True)
            continue
        summaries.append(run_eval(case, output_dir))

    if not summaries:
        raise RuntimeError("No evaluation cases ran. Check the checkpoint and JSON paths at the top of this file.")

    summary_json = output_dir / "comparison_summary.json"
    summary_csv = output_dir / "comparison_summary.csv"
    group_csv = output_dir / "comparison_groups.csv"
    summary_json.write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_summary_csv(summary_csv, summaries)
    write_group_csv(group_csv, summaries)
    print_table(summaries)
    print(f"\n[scenic-sft-eval] wrote {summary_json.relative_to(PROJECT_ROOT)}")
    print(f"[scenic-sft-eval] wrote {summary_csv.relative_to(PROJECT_ROOT)}")
    print(f"[scenic-sft-eval] wrote {group_csv.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
