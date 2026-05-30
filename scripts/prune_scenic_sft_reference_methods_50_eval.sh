#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1

METHODS="${METHODS:-magnitude,nvidia,wanda}"
SPARSITY="${SPARSITY:-0.5}"
PRUNE_SCOPE="${PRUNE_SCOPE:-all-linear}"
RUN_ROOT="${RUN_ROOT:-runs/scenic-pruned50-reference-methods}"
ALL_OUTPUT_DIR="${ALL_OUTPUT_DIR:-eval_results/scenic_sft/pruned50_reference_methods}"

TRAINING_CHECKPOINT="${TRAINING_CHECKPOINT:-runs/scenic-sft-training-dataset/latest}"
CONTRASTIVE_CHECKPOINT="${CONTRASTIVE_CHECKPOINT:-runs/scenic-sft-contrastive-dataset/latest}"

BENCHMARK_JSON="${BENCHMARK_JSON:-data/scenic/iot_instruction_benchmark_200.json}"
TRAINING_JSON="${TRAINING_JSON:-data/scenic/SCENIC_full_training_dataset.json}"
CONTRASTIVE_JSON="${CONTRASTIVE_JSON:-data/scenic/SCENIC_full_anchor_positive_negative.json}"
TRAINING_CALIBRATION_JSON="${TRAINING_CALIBRATION_JSON:-$TRAINING_JSON}"
CONTRASTIVE_CALIBRATION_JSON="${CONTRASTIVE_CALIBRATION_JSON:-$CONTRASTIVE_JSON}"

PRUNE_DEVICE="${PRUNE_DEVICE:-auto}"
PRUNE_DTYPE="${PRUNE_DTYPE:-fp32}"
CALIBRATION_BATCH_SIZE="${CALIBRATION_BATCH_SIZE:-4}"
CALIBRATION_BATCHES="${CALIBRATION_BATCHES:-64}"
EVAL_DTYPE="${EVAL_DTYPE:-auto}"
BATCH_SIZE="${BATCH_SIZE:-128}"
MAX_LENGTH="${MAX_LENGTH:-128}"
OVERWRITE="${OVERWRITE:-1}"
INCLUDE_CLASSIFIER="${INCLUDE_CLASSIFIER:-1}"

mkdir -p "$RUN_ROOT" "$ALL_OUTPUT_DIR"

overwrite_args=()
if [[ "$OVERWRITE" == "1" ]]; then
  overwrite_args+=(--overwrite)
fi

classifier_args=()
if [[ "$INCLUDE_CLASSIFIER" != "1" ]]; then
  classifier_args+=(--exclude-classifier)
fi

method_names=()
IFS=',' read -r -a method_names <<< "$METHODS"
completed_methods=()

for raw_method in "${method_names[@]}"; do
  method="${raw_method//[[:space:]]/}"
  if [[ -z "$method" ]]; then
    continue
  fi

  case "$method" in
    magnitude)
      label="magnitude"
      ;;
    nvidia|nvidia-2:4|nvidia_2_4)
      method="nvidia"
      label="nvidia_2_4"
      ;;
    wanda)
      label="wanda"
      ;;
    *)
      echo "[reference-prune] unknown method: $method" >&2
      echo "Known methods: magnitude, nvidia, wanda" >&2
      exit 2
      ;;
  esac

  training_output="$RUN_ROOT/training_dataset_$label"
  contrastive_output="$RUN_ROOT/contrastive_anchor_$label"
  eval_output="$ALL_OUTPUT_DIR/$label"

  echo "[reference-prune] $method | training-dataset model -> $training_output"
  python scripts/prune_scenic_sft_reference_methods.py \
    --method "$method" \
    --checkpoint "$TRAINING_CHECKPOINT" \
    --output "$training_output" \
    --sparsity "$SPARSITY" \
    --scope "$PRUNE_SCOPE" \
    --calibration-json "$TRAINING_CALIBRATION_JSON" \
    --calibration-batch-size "$CALIBRATION_BATCH_SIZE" \
    --calibration-batches "$CALIBRATION_BATCHES" \
    --max-length "$MAX_LENGTH" \
    --device "$PRUNE_DEVICE" \
    --dtype "$PRUNE_DTYPE" \
    "${overwrite_args[@]}" \
    "${classifier_args[@]}"

  echo "[reference-prune] $method | contrastive-anchor model -> $contrastive_output"
  python scripts/prune_scenic_sft_reference_methods.py \
    --method "$method" \
    --checkpoint "$CONTRASTIVE_CHECKPOINT" \
    --output "$contrastive_output" \
    --sparsity "$SPARSITY" \
    --scope "$PRUNE_SCOPE" \
    --calibration-json "$CONTRASTIVE_CALIBRATION_JSON" \
    --calibration-batch-size "$CALIBRATION_BATCH_SIZE" \
    --calibration-batches "$CALIBRATION_BATCHES" \
    --max-length "$MAX_LENGTH" \
    --device "$PRUNE_DEVICE" \
    --dtype "$PRUNE_DTYPE" \
    "${overwrite_args[@]}" \
    "${classifier_args[@]}"

  echo "[reference-prune] evaluating $method outputs"
  python scripts/eval_scenic_sft_comparison.py \
    --benchmark-json "$BENCHMARK_JSON" \
    --training-json "$TRAINING_JSON" \
    --contrastive-json "$CONTRASTIVE_JSON" \
    --training-checkpoint "$training_output" \
    --contrastive-checkpoint "$contrastive_output" \
    --output-dir "$eval_output" \
    --batch-size "$BATCH_SIZE" \
    --max-length "$MAX_LENGTH" \
    --dtype "$EVAL_DTYPE"

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

json_path = root / "reference_methods_summary.json"
csv_path = root / "reference_methods_summary.csv"
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

print(f"[reference-prune] wrote {json_path}")
print(f"[reference-prune] wrote {csv_path}")
PY

echo "[reference-prune] done"
echo "  $ALL_OUTPUT_DIR/reference_methods_summary.csv"
echo "  $ALL_OUTPUT_DIR/reference_methods_summary.json"
