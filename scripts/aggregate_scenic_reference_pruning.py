#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_EVAL_ROOT = "eval_results/scenic_sft/pruned50_reference_methods"
DEFAULT_RUN_ROOT = "runs/scenic-pruned50-reference-methods"
DEFAULT_METHODS = "magnitude,nvidia_2_4,wanda,gradient"

EXPECTED_CASES = {
    ("training_dataset_model", "benchmark_200"),
    ("training_dataset_model", "training_dataset_retention"),
    ("contrastive_anchor_model", "benchmark_200"),
    ("contrastive_anchor_model", "contrastive_anchor_retention"),
}

CSV_FIELDNAMES = [
    "prune_method",
    "model",
    "dataset",
    "rows",
    "scored_rows",
    "prediction_unique_count",
    "prediction_unique_ratio",
    "top_prediction",
    "top_prediction_count",
    "top_prediction_share",
    "label_space_coverage",
    "exact_match_accuracy",
    "top5_accuracy",
    "checkpoint",
    "json",
    "summary_output",
    "predictions_output",
    "pruned_checkpoint",
    "prune_summary_output",
    "prune_method_detail",
    "prune_scope",
    "prune_include_classifier",
    "prune_requested_sparsity",
    "prune_targeted_sparsity_after",
    "prune_model_sparsity_after",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate per-method SCENIC reference pruning evaluations into one "
            "validated JSON/CSV summary."
        )
    )
    parser.add_argument("--eval-root", default=DEFAULT_EVAL_ROOT)
    parser.add_argument("--run-root", default=DEFAULT_RUN_ROOT)
    parser.add_argument("--methods", default=DEFAULT_METHODS, help="Comma-separated method labels or aliases.")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--allow-missing", action="store_true", help="Skip missing method outputs instead of failing.")
    return parser.parse_args()


def method_label(method: str) -> str:
    normalized = method.strip().lower()
    if normalized == "magnitude":
        return "magnitude"
    if normalized in {"nvidia", "nvidia-2:4", "nvidia_2_4"}:
        return "nvidia_2_4"
    if normalized == "wanda":
        return "wanda"
    if normalized in {"gradient", "taylor"}:
        return "gradient"
    raise ValueError(f"Unknown method {method!r}. Use magnitude, nvidia, wanda, or gradient.")


def parse_methods(value: str) -> list[str]:
    labels: list[str] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        label = method_label(item)
        if label not in labels:
            labels.append(label)
    if not labels:
        raise ValueError("--methods did not contain any valid method names.")
    return labels


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def checkpoint_prefix(model_name: str) -> str:
    if model_name == "training_dataset_model":
        return "training_dataset"
    if model_name == "contrastive_anchor_model":
        return "contrastive_anchor"
    raise ValueError(f"Unknown model in comparison summary: {model_name!r}")


def require_or_skip(path: Path, allow_missing: bool, what: str) -> bool:
    if path.exists():
        return True
    if allow_missing:
        print(f"[reference-prune] skipping missing {what}: {path}")
        return False
    raise FileNotFoundError(f"Missing {what}: {path}")


def add_pruning_fields(row: dict[str, Any], run_root: Path, method: str, allow_missing: bool) -> dict[str, Any] | None:
    prefix = checkpoint_prefix(str(row.get("model", "")))
    pruned_checkpoint = run_root / f"{prefix}_{method}"
    prune_summary_output = pruned_checkpoint / "prune_summary.json"
    if not require_or_skip(prune_summary_output, allow_missing, "prune summary"):
        return None

    checkpoint = Path(str(row.get("checkpoint", "")))
    if not allow_missing and checkpoint.as_posix() != pruned_checkpoint.as_posix():
        raise ValueError(
            "Comparison row checkpoint does not match expected pruned checkpoint: "
            f"row={checkpoint} expected={pruned_checkpoint}"
        )

    prune_summary = load_json(prune_summary_output)
    return {
        "prune_method": method,
        **row,
        "pruned_checkpoint": str(pruned_checkpoint),
        "prune_summary_output": str(prune_summary_output),
        "prune_method_detail": prune_summary.get("method_detail", prune_summary.get("method")),
        "prune_scope": prune_summary.get("scope"),
        "prune_include_classifier": prune_summary.get("include_classifier"),
        "prune_requested_sparsity": prune_summary.get("requested_sparsity"),
        "prune_targeted_sparsity_after": prune_summary.get("targeted_sparsity_after"),
        "prune_model_sparsity_after": prune_summary.get("model_sparsity_after"),
    }


def validate_cases(method: str, rows: list[dict[str, Any]]) -> None:
    found = {(str(row.get("model")), str(row.get("dataset"))) for row in rows}
    missing = sorted(EXPECTED_CASES - found)
    extras = sorted(found - EXPECTED_CASES)
    if missing:
        raise ValueError(f"{method} is missing expected evaluation rows: {missing}")
    if extras:
        raise ValueError(f"{method} has unexpected evaluation rows: {extras}")


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in CSV_FIELDNAMES})


def main() -> None:
    args = parse_args()
    eval_root = Path(args.eval_root).expanduser()
    run_root = Path(args.run_root).expanduser()
    methods = parse_methods(args.methods)
    output_json = Path(args.output_json).expanduser() if args.output_json else eval_root / "reference_methods_summary.json"
    output_csv = Path(args.output_csv).expanduser() if args.output_csv else eval_root / "reference_methods_summary.csv"

    rows: list[dict[str, Any]] = []
    for method in methods:
        comparison_path = eval_root / method / "comparison_summary.json"
        if not require_or_skip(comparison_path, bool(args.allow_missing), "comparison summary"):
            continue
        comparison_rows = load_json(comparison_path)
        if not isinstance(comparison_rows, list):
            raise ValueError(f"{comparison_path} must contain a JSON list.")
        if not args.allow_missing:
            validate_cases(method, comparison_rows)

        for row in comparison_rows:
            if not isinstance(row, dict):
                raise ValueError(f"{comparison_path} contains a non-object row: {row!r}")
            enriched = add_pruning_fields(row, run_root, method, bool(args.allow_missing))
            if enriched is not None:
                rows.append(enriched)

    if not rows and not args.allow_missing:
        raise RuntimeError("No reference pruning outcomes were aggregated.")

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_summary_csv(output_csv, rows)

    print(f"[reference-prune] methods: {', '.join(methods)}")
    print(f"[reference-prune] outcomes: {len(rows)}")
    print(f"[reference-prune] wrote {output_json}")
    print(f"[reference-prune] wrote {output_csv}")


if __name__ == "__main__":
    main()
