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
TRAINING_JSON = "data/scenic/SCENIC_full_training_dataset.json"

TRAINING_CHECKPOINT = "runs/scenic-sft-training-dataset/latest"

OUTPUT_DIR = "eval_results/scenic_sft/em1_em5_check"
BATCH_SIZE = 128
MAX_LENGTH = 128
EVAL_DTYPE = "auto"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check the base SCENIC SFT checkpoint's EM@1 and EM@5 on its "
            "training data and the 200-row benchmark."
        )
    )
    parser.add_argument("--training-checkpoint", default=TRAINING_CHECKPOINT)
    parser.add_argument("--training-json", default=TRAINING_JSON)
    parser.add_argument("--benchmark-json", default=BENCHMARK_JSON)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--dtype", choices=("auto", "fp32", "bf16"), default=EVAL_DTYPE)
    parser.add_argument("--skip-missing", action="store_true", help="Skip missing checkpoints or JSON files.")
    return parser.parse_args()


def sanitize(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value).strip("_")


def cases(args: argparse.Namespace) -> list[dict[str, str]]:
    return [
        {
            "model": "base_sft",
            "dataset": "training_data",
            "checkpoint": args.training_checkpoint,
            "json": args.training_json,
        },
        {
            "model": "base_sft",
            "dataset": "benchmark_200",
            "checkpoint": args.training_checkpoint,
            "json": args.benchmark_json,
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
    print(f"[em-check] {case['model']} on {case['dataset']}", flush=True)
    subprocess.run(command, check=True)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        "model": case["model"],
        "dataset": case["dataset"],
        "checkpoint": case["checkpoint"],
        "json": case["json"],
        "summary_output": str(summary_path),
        "predictions_output": str(predictions_path),
        "rows": summary.get("rows"),
        "scored_rows": summary.get("scored_rows"),
        "label_space_coverage": summary.get("label_space_coverage"),
        "em1": summary.get("exact_match_accuracy"),
        "em5": summary.get("top5_accuracy"),
        "em1_correct": summary.get("exact_match_correct"),
        "em5_correct": summary.get("top5_correct"),
        "prediction_unique_count": summary.get("prediction_unique_count"),
        "prediction_unique_ratio": summary.get("prediction_unique_ratio"),
        "top_prediction": summary.get("top_prediction"),
        "top_prediction_count": summary.get("top_prediction_count"),
        "top_prediction_share": summary.get("top_prediction_share"),
    }


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
    if value is None:
        return "n/a"
    return f"{float(value) * 100:.2f}%"


def print_table(rows: list[dict[str, Any]]) -> None:
    print("\n[em-check] EM@1 / EM@5")
    print(f"{'model':16s} {'dataset':26s} {'EM@1':>9s} {'EM@5':>9s} {'scored':>12s} {'unique':>10s} {'top share':>10s}")
    for row in rows:
        scored = f"{row.get('scored_rows', 0)}/{row.get('rows', 0)}"
        unique = str(row.get("prediction_unique_count", "n/a"))
        top_share = pct(row.get("top_prediction_share"))
        print(
            f"{row['model']:16s} {row['dataset']:26s} "
            f"{pct(row.get('em1')):>9s} {pct(row.get('em5')):>9s} "
            f"{scored:>12s} {unique:>10s} {top_share:>10s}"
        )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for case in cases(args):
        checkpoint_path = Path(case["checkpoint"]).expanduser()
        json_path = Path(case["json"]).expanduser()
        if args.skip_missing and (not checkpoint_path.exists() or not json_path.exists()):
            print(f"[em-check] skipping missing case: {case}", flush=True)
            continue
        rows.append(run_eval(case, args, output_dir))

    if not rows:
        raise RuntimeError("No EM checks ran. Check checkpoint paths or remove --skip-missing.")

    summary_json = output_dir / "em1_em5_summary.json"
    summary_csv = output_dir / "em1_em5_summary.csv"
    summary_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(summary_csv, rows)
    print_table(rows)
    print(f"\n[em-check] wrote {summary_json}")
    print(f"[em-check] wrote {summary_csv}")


if __name__ == "__main__":
    main()
