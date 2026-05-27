#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1

METHODS="${METHODS:-encoder-linear,all-linear-classifier,all-matrix-classifier}"
SPARSITY="${SPARSITY:-0.5}"
RUN_ROOT="${RUN_ROOT:-runs/scenic-pruned50-all-methods}"
ALL_OUTPUT_DIR="${ALL_OUTPUT_DIR:-eval_results/scenic_sft/pruned50_all_methods}"

TRAINING_CHECKPOINT="${TRAINING_CHECKPOINT:-runs/scenic-sft-training-dataset/latest}"
CONTRASTIVE_CHECKPOINT="${CONTRASTIVE_CHECKPOINT:-runs/scenic-sft-contrastive-dataset/latest}"

BENCHMARK_JSON="${BENCHMARK_JSON:-data/scenic/iot_instruction_benchmark_200.json}"
TRAINING_JSON="${TRAINING_JSON:-data/scenic/SCENIC_full_training_dataset.json}"
CONTRASTIVE_JSON="${CONTRASTIVE_JSON:-data/scenic/SCENIC_full_anchor_positive_negative.json}"

EVAL_DTYPE="${EVAL_DTYPE:-auto}"
BATCH_SIZE="${BATCH_SIZE:-128}"
MAX_LENGTH="${MAX_LENGTH:-128}"
OVERWRITE="${OVERWRITE:-1}"

mkdir -p "$RUN_ROOT" "$ALL_OUTPUT_DIR"

method_names=()
IFS=',' read -r -a method_names <<< "$METHODS"
completed_methods=()

for raw_method in "${method_names[@]}"; do
  method="${raw_method//[[:space:]]/}"
  if [[ -z "$method" ]]; then
    continue
  fi

  case "$method" in
    encoder-linear)
      label="encoder_linear"
      scope="encoder-linear"
      include_classifier="0"
      ;;
    all-linear-classifier)
      label="all_linear_classifier"
      scope="all-linear"
      include_classifier="1"
      ;;
    all-matrix-classifier)
      label="all_matrix_classifier"
      scope="all-matrix"
      include_classifier="1"
      ;;
    *)
      echo "[prune-all-methods] unknown method: $method" >&2
      echo "Known methods: encoder-linear, all-linear-classifier, all-matrix-classifier" >&2
      exit 2
      ;;
  esac

  echo "[prune-all-methods] running $method"
  TRAINING_CHECKPOINT="$TRAINING_CHECKPOINT" \
  CONTRASTIVE_CHECKPOINT="$CONTRASTIVE_CHECKPOINT" \
  TRAINING_OUTPUT="$RUN_ROOT/training_dataset_$label" \
  CONTRASTIVE_OUTPUT="$RUN_ROOT/contrastive_anchor_$label" \
  BENCHMARK_JSON="$BENCHMARK_JSON" \
  TRAINING_JSON="$TRAINING_JSON" \
  CONTRASTIVE_JSON="$CONTRASTIVE_JSON" \
  EVAL_OUTPUT_DIR="$ALL_OUTPUT_DIR/$label" \
  SPARSITY="$SPARSITY" \
  PRUNE_SCOPE="$scope" \
  INCLUDE_CLASSIFIER="$include_classifier" \
  EVAL_DTYPE="$EVAL_DTYPE" \
  BATCH_SIZE="$BATCH_SIZE" \
  MAX_LENGTH="$MAX_LENGTH" \
  OVERWRITE="$OVERWRITE" \
  ./scripts/prune_scenic_sft_50_eval.sh

  completed_methods+=("$label")
done

python - "$ALL_OUTPUT_DIR" "${completed_methods[@]}" <<'PY'
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
methods = sys.argv[2:]
rows = []

for method in methods:
    summary_path = root / method / "comparison_summary.csv"
    if not summary_path.exists():
        continue
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append({"prune_method": method, **row})

json_path = root / "all_methods_summary.json"
csv_path = root / "all_methods_summary.csv"
json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

fieldnames = [
    "prune_method",
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
with csv_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field) for field in fieldnames})

print(f"[prune-all-methods] wrote {json_path}")
print(f"[prune-all-methods] wrote {csv_path}")
PY

echo "[prune-all-methods] done"
echo "  $ALL_OUTPUT_DIR/all_methods_summary.csv"
echo "  $ALL_OUTPUT_DIR/all_methods_summary.json"
