#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /path/to/base-encoder-checkpoint" >&2
  exit 2
fi

BASE_MODEL="$1"
PYTHON="${PYTHON:-python}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-scenic_linear_sparsity_0_30_50}"
RUN_ROOT="${RUN_ROOT:-runs/$EXPERIMENT_NAME}"
RESULT_ROOT="${RESULT_ROOT:-results/$EXPERIMENT_NAME}"
SFT_RUN_DIR="${SFT_RUN_DIR:-$RUN_ROOT/scenic-sft}"
DENSE_CHECKPOINT="${DENSE_CHECKPOINT:-$SFT_RUN_DIR/latest}"
SFT_CONFIG_TEMPLATE="${SFT_CONFIG_TEMPLATE:-configs/scenic_sft_training_dataset_8gpu.yaml}"
SFT_CONFIG="${SFT_CONFIG:-$RUN_ROOT/scenic_sft_from_base.yaml}"

TRAIN_JSON="${TRAIN_JSON:-data/scenic/SCENIC_full_training_dataset.json}"
BENCHMARK_JSON="${BENCHMARK_JSON:-data/scenic/iot_instruction_benchmark_200.json}"
BENCHMARK_DIFFICULTY_PATH="${BENCHMARK_DIFFICULTY_PATH:-}"
TOKENIZER_PATH="${TOKENIZER_PATH:-}"

RETRAIN="${RETRAIN:-1}"
OVERWRITE="${OVERWRITE:-1}"
TRAIN_WITH_TORCHRUN="${TRAIN_WITH_TORCHRUN:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

ORIGINAL_METHODS="${ORIGINAL_METHODS:-magnitude,nvidia,wanda,gradient}"
ORIGINAL_SPARSITY_LEVELS="${ORIGINAL_SPARSITY_LEVELS:-${ORIGINAL_SPARSITY:-0.3 0.5}}"
ORIGINAL_RUN_ROOT="${ORIGINAL_RUN_ROOT:-$RUN_ROOT/original_one_shot_reference_methods}"
ORIGINAL_RESULT_ROOT="${ORIGINAL_RESULT_ROOT:-$RESULT_ROOT/original_one_shot_reference_methods}"
PRUNE_SCOPE="${PRUNE_SCOPE:-encoder-linear}"
PRUNE_DEVICE="${PRUNE_DEVICE:-auto}"
PRUNE_DTYPE="${PRUNE_DTYPE:-fp32}"
CALIBRATION_BATCH_SIZE="${CALIBRATION_BATCH_SIZE:-4}"
CALIBRATION_BATCHES="${CALIBRATION_BATCHES:-64}"
REINIT_CLASSIFIER="${REINIT_CLASSIFIER:-1}"
CLASSIFIER_INIT_BATCH_SIZE="${CLASSIFIER_INIT_BATCH_SIZE:-128}"
CLASSIFIER_INIT_MAX_LENGTH="${CLASSIFIER_INIT_MAX_LENGTH:-128}"

BATCH_SIZE="${BATCH_SIZE:-128}"
MAX_LENGTH="${MAX_LENGTH:-128}"
EVAL_DTYPE="${EVAL_DTYPE:-auto}"
SEED="${SEED:-42}"
SPARSITY_LEVELS="${SPARSITY_LEVELS:-0 0.3 0.5}"
RETUNE_EPOCHS="${RETUNE_EPOCHS:-1}"
RUN_PLOTS="${RUN_PLOTS:-1}"

mkdir -p "$RUN_ROOT" "$RESULT_ROOT" "$ORIGINAL_RUN_ROOT" "$ORIGINAL_RESULT_ROOT"

echo "[sparsity-revision] base_model=$BASE_MODEL"
echo "[sparsity-revision] dense_checkpoint=$DENSE_CHECKPOINT"

if [[ "$RETRAIN" == "1" || ! -e "$DENSE_CHECKPOINT" ]]; then
  echo "[sparsity-revision] writing SFT config -> $SFT_CONFIG"
  "$PYTHON" - "$SFT_CONFIG_TEMPLATE" "$SFT_CONFIG" "$BASE_MODEL" "$SFT_RUN_DIR" "$TRAIN_JSON" "$TOKENIZER_PATH" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

import yaml

template_path, output_path, base_model, run_dir, train_json, tokenizer_path = sys.argv[1:]
with Path(template_path).open("r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle) or {}

config.setdefault("run", {})["output_dir"] = run_dir
config.setdefault("model", {})["base_model"] = base_model
if tokenizer_path:
    config["model"]["tokenizer_path"] = tokenizer_path
config.setdefault("data", {})["train_json"] = train_json

output = Path(output_path)
output.parent.mkdir(parents=True, exist_ok=True)
with output.open("w", encoding="utf-8") as handle:
    yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
PY

  echo "[sparsity-revision] training dense SFT checkpoint"
  if [[ "$TRAIN_WITH_TORCHRUN" == "1" ]]; then
    torchrun --nproc_per_node "$NPROC_PER_NODE" scripts/train_scenic_sft.py --config "$SFT_CONFIG"
  else
    "$PYTHON" scripts/train_scenic_sft.py --config "$SFT_CONFIG"
  fi
else
  echo "[sparsity-revision] reusing existing dense checkpoint"
fi

overwrite_args=()
if [[ "$OVERWRITE" == "1" ]]; then
  overwrite_args+=(--overwrite)
fi

reinit_classifier_args=()
if [[ "$REINIT_CLASSIFIER" == "1" ]]; then
  reinit_classifier_args+=(
    --reinitialize-classifier-from-responses
    --classifier-init-batch-size "$CLASSIFIER_INIT_BATCH_SIZE"
    --classifier-init-max-length "$CLASSIFIER_INIT_MAX_LENGTH"
  )
fi

method_names=()
IFS=',' read -r -a method_names <<< "$ORIGINAL_METHODS"
original_sparsity_levels_text="${ORIGINAL_SPARSITY_LEVELS//,/ }"
original_sparsity_levels=()
read -r -a original_sparsity_levels <<< "$original_sparsity_levels_text"
completed_original=()
skipped_original=()

echo "[sparsity-revision] original methods stay one-shot; classifier_rebuild=$REINIT_CLASSIFIER"
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
    gradient|taylor)
      method="gradient"
      label="gradient"
      ;;
    *)
      echo "[sparsity-revision] unknown original method: $method" >&2
      echo "Known methods: magnitude, nvidia, wanda, gradient" >&2
      exit 2
      ;;
  esac

  for sparsity in "${original_sparsity_levels[@]}"; do
    if [[ -z "$sparsity" ]]; then
      continue
    fi
    if [[ "$method" == "nvidia" && "$sparsity" != "0.5" && "$sparsity" != ".5" && "$sparsity" != "0.50" ]]; then
      echo "[sparsity-revision] skip one-shot nvidia at sparsity=$sparsity; NVIDIA 2:4 is exactly 50%"
      skipped_original+=("$label|$sparsity|NVIDIA 2:4 is exactly 50% sparse")
      continue
    fi

    sparsity_tag="${sparsity//./p}"
    run_label="${label}_${sparsity_tag}"
    pruned_checkpoint="$ORIGINAL_RUN_ROOT/$run_label"
    eval_dir="$ORIGINAL_RESULT_ROOT/$run_label"
    mkdir -p "$eval_dir"

    echo "[sparsity-revision] one-shot $method sparsity=$sparsity -> $pruned_checkpoint"
    "$PYTHON" scripts/prune_scenic_sft_reference_methods.py \
      --method "$method" \
      --checkpoint "$DENSE_CHECKPOINT" \
      --output "$pruned_checkpoint" \
      --sparsity "$sparsity" \
      --scope "$PRUNE_SCOPE" \
      --exclude-classifier \
      --calibration-json "$TRAIN_JSON" \
      --calibration-batch-size "$CALIBRATION_BATCH_SIZE" \
      --calibration-batches "$CALIBRATION_BATCHES" \
      --max-length "$MAX_LENGTH" \
      --device "$PRUNE_DEVICE" \
      --dtype "$PRUNE_DTYPE" \
      "${overwrite_args[@]}" \
      "${reinit_classifier_args[@]}"

    echo "[sparsity-revision] eval one-shot $method sparsity=$sparsity"
    "$PYTHON" scripts/eval_scenic_sft_local.py \
      --json "$BENCHMARK_JSON" \
      --checkpoint "$pruned_checkpoint" \
      --output "$eval_dir/benchmark_predictions.jsonl" \
      --summary-output "$eval_dir/benchmark_summary.json" \
      --batch-size "$BATCH_SIZE" \
      --max-length "$MAX_LENGTH" \
      --dtype "$EVAL_DTYPE"

    completed_original+=("$run_label|$label|$method|$sparsity")
  done
done

"$PYTHON" - "$ORIGINAL_RESULT_ROOT" "${completed_original[@]}" <<'PY'
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
specs = sys.argv[2:]
rows = []
for spec in specs:
    run_label, label, method, sparsity = spec.split("|", 3)
    summary_path = root / run_label / "benchmark_summary.json"
    if not summary_path.exists():
        continue
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    checkpoint = Path(str(summary.get("checkpoint") or ""))
    pruning = {}
    metadata_path = checkpoint / "scenic_sft_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        pruning = metadata.get("pruning") if isinstance(metadata.get("pruning"), dict) else {}
    rows.append({
        "result_block": "original_one_shot_reference_methods",
        "run_label": run_label,
        "pruning_mode": "oneshot",
        "method": label,
        "pruning_method": method,
        "target_sparsity": float(sparsity),
        "targeted_linear_sparsity_actual": pruning.get("targeted_sparsity_after"),
        "whole_model_sparsity_actual": pruning.get("model_sparsity_after"),
        "checkpoint": summary.get("checkpoint"),
        "rows": summary.get("rows"),
        "scored_rows": summary.get("scored_rows"),
        "em1": summary.get("exact_match_accuracy"),
        "em5": summary.get("top5_accuracy"),
        "summary_output": str(summary_path),
        "predictions_output": summary.get("predictions_output"),
    })

json_path = root / "original_one_shot_summary.json"
csv_path = root / "original_one_shot_summary.csv"
json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
with csv_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=[
            "result_block",
            "run_label",
            "pruning_mode",
            "method",
            "pruning_method",
            "target_sparsity",
            "targeted_linear_sparsity_actual",
            "whole_model_sparsity_actual",
            "checkpoint",
            "rows",
            "scored_rows",
            "em1",
            "em5",
            "summary_output",
            "predictions_output",
        ],
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
print(f"[sparsity-revision] wrote {csv_path}")
PY

difficulty_args=()
if [[ -n "$BENCHMARK_DIFFICULTY_PATH" ]]; then
  difficulty_args+=(--benchmark_difficulty_path "$BENCHMARK_DIFFICULTY_PATH")
fi

block_head_args=()
if [[ "$REINIT_CLASSIFIER" == "1" ]]; then
  block_head_args+=(
    --reinitialize_classifier_from_responses
    --classifier_init_batch_size "$CLASSIFIER_INIT_BATCH_SIZE"
    --classifier_init_max_length "$CLASSIFIER_INIT_MAX_LENGTH"
  )
fi

echo "[sparsity-revision] added linear-sparsity block: progressive masks + ${RETUNE_EPOCHS} epoch retune"
"$PYTHON" scripts/run_sparsity_experiments.py \
  --experiment_name "$EXPERIMENT_NAME" \
  --model_family encoder_only \
  --model_checkpoint "$DENSE_CHECKPOINT" \
  --benchmark_path "$BENCHMARK_JSON" \
  "${difficulty_args[@]}" \
  --sparsity_levels $SPARSITY_LEVELS \
  --pruning_modes dense oneshot progressive \
  --prune_scope linear_weights \
  --prune_method magnitude \
  --progressive_schedule staged \
  --recovery_epochs_per_stage 0 \
  --final_recovery_epochs "$RETUNE_EPOCHS" \
  --recovery_train_path "$TRAIN_JSON" \
  --batch_size "$BATCH_SIZE" \
  --max_length "$MAX_LENGTH" \
  --seed "$SEED" \
  --output_dir "$RESULT_ROOT/linear_sparsity_retune" \
  "${block_head_args[@]}"

if [[ "$RUN_PLOTS" == "1" ]]; then
  if ! "$PYTHON" scripts/plot_sparsity_results.py \
    --experiment_name "$EXPERIMENT_NAME" \
    --results_dir "$RESULT_ROOT/linear_sparsity_retune"; then
    echo "[sparsity-revision] plot generation failed; metrics are still available. Install matplotlib and rerun scripts/plot_sparsity_results.py." >&2
  fi
fi

"$PYTHON" - "$RESULT_ROOT" "$ORIGINAL_RESULT_ROOT/original_one_shot_summary.json" "$RESULT_ROOT/linear_sparsity_retune/summary_metrics.csv" "${skipped_original[@]}" <<'PY'
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

result_root = Path(sys.argv[1])
original_json = Path(sys.argv[2])
retune_csv = Path(sys.argv[3])
skipped_specs = sys.argv[4:]


def as_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    return float(value)


rows: list[dict[str, Any]] = []
if original_json.exists():
    for row in json.loads(original_json.read_text(encoding="utf-8")):
        rows.append(
            {
                "result_block": row.get("result_block") or "original_one_shot_reference_methods",
                "run_label": row.get("run_label"),
                "model_family": "encoder_only",
                "pruning_mode": row.get("pruning_mode") or "oneshot",
                "pruning_method": row.get("pruning_method") or row.get("method"),
                "method_label": row.get("method"),
                "target_sparsity": row.get("target_sparsity"),
                "targeted_linear_sparsity_actual": row.get("targeted_linear_sparsity_actual"),
                "whole_model_sparsity_actual": row.get("whole_model_sparsity_actual"),
                "seed": None,
                "em1_overall": row.get("em1"),
                "em5_overall": row.get("em5"),
                "count_total": row.get("rows"),
                "scored_rows": row.get("scored_rows"),
                "checkpoint_path": row.get("checkpoint"),
                "mask_path": None,
                "predictions_path": row.get("predictions_output"),
                "summary_output": row.get("summary_output"),
                "notes": "reference one-shot pruning; classifier rebuild follows REINIT_CLASSIFIER",
            }
        )

if retune_csv.exists():
    with retune_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "result_block": "linear_sparsity_retune",
                    "run_label": f"{row.get('pruning_mode')}_{row.get('target_sparsity')}",
                    "model_family": row.get("model_family"),
                    "pruning_mode": row.get("pruning_mode"),
                    "pruning_method": row.get("pruning_method"),
                    "method_label": row.get("pruning_method"),
                    "target_sparsity": as_float(row.get("target_sparsity")),
                    "targeted_linear_sparsity_actual": as_float(row.get("targeted_linear_sparsity_actual")),
                    "whole_model_sparsity_actual": as_float(row.get("whole_model_sparsity_actual")),
                    "seed": int(row["seed"]) if row.get("seed") not in {None, ""} else None,
                    "em1_overall": as_float(row.get("em1_overall")),
                    "em5_overall": as_float(row.get("em5_overall")),
                    "em1_retention_overall": as_float(row.get("em1_retention_overall")),
                    "em5_retention_overall": as_float(row.get("em5_retention_overall")),
                    "count_total": int(row["count_total"]) if row.get("count_total") not in {None, ""} else None,
                    "checkpoint_path": row.get("checkpoint_path"),
                    "mask_path": row.get("mask_path"),
                    "predictions_path": row.get("predictions_path"),
                    "summary_output": str(retune_csv),
                    "training_config": json.loads(row["training_config_json"]) if row.get("training_config_json") else None,
                    "pruning_config": json.loads(row["pruning_config_json"]) if row.get("pruning_config_json") else None,
                    "notes": "dense/one-shot/progressive magnitude block from run_sparsity_experiments.py",
                }
            )

skipped = []
for spec in skipped_specs:
    method, sparsity, reason = spec.split("|", 2)
    skipped.append({"method": method, "target_sparsity": float(sparsity), "reason": reason})

payload = {
    "report_type": "scenic_sparsity_revision_combined_results",
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "source_files": {
        "original_one_shot_summary": str(original_json),
        "linear_sparsity_retune_summary": str(retune_csv),
    },
    "rows": rows,
    "skipped": skipped,
}
output = result_root / "all_sparsity_results.json"
output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"[sparsity-revision] wrote {output}")
PY

echo "[sparsity-revision] done"
echo "  original one-shot: $ORIGINAL_RESULT_ROOT/original_one_shot_summary.csv"
echo "  added retune block: $RESULT_ROOT/linear_sparsity_retune/summary_metrics.csv"
echo "  all results json: $RESULT_ROOT/all_sparsity_results.json"
