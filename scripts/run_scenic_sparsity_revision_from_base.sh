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

BASELINE_RUN_DIR="${BASELINE_RUN_DIR:-$RUN_ROOT/base_encoder_dense}"
BASELINE_DENSE_CHECKPOINT="${BASELINE_DENSE_CHECKPOINT:-$BASELINE_RUN_DIR/latest}"
BASELINE_RESULT_ROOT="${BASELINE_RESULT_ROOT:-$RESULT_ROOT/base_encoder_dense}"
INCLUDE_BASE_ENCODER_BASELINE="${INCLUDE_BASE_ENCODER_BASELINE:-1}"

REGULAR_SFT_RUN_DIR="${REGULAR_SFT_RUN_DIR:-${SFT_RUN_DIR:-$RUN_ROOT/regular_sft}}"
REGULAR_DENSE_CHECKPOINT="${REGULAR_DENSE_CHECKPOINT:-${DENSE_CHECKPOINT:-$REGULAR_SFT_RUN_DIR/latest}}"
REGULAR_SFT_CONFIG_TEMPLATE="${REGULAR_SFT_CONFIG_TEMPLATE:-${SFT_CONFIG_TEMPLATE:-configs/scenic_sft_training_dataset_8gpu.yaml}}"
REGULAR_SFT_CONFIG="${REGULAR_SFT_CONFIG:-${SFT_CONFIG:-$RUN_ROOT/regular_sft_from_base.yaml}}"

CONTRASTIVE_SFT_RUN_DIR="${CONTRASTIVE_SFT_RUN_DIR:-$RUN_ROOT/contrastive_sft}"
CONTRASTIVE_DENSE_CHECKPOINT="${CONTRASTIVE_DENSE_CHECKPOINT:-$CONTRASTIVE_SFT_RUN_DIR/latest}"
CONTRASTIVE_SFT_CONFIG_TEMPLATE="${CONTRASTIVE_SFT_CONFIG_TEMPLATE:-configs/scenic_sft_contrastive_dataset_8gpu.yaml}"
CONTRASTIVE_SFT_CONFIG="${CONTRASTIVE_SFT_CONFIG:-$RUN_ROOT/contrastive_sft_from_base.yaml}"

TRAIN_JSON="${TRAIN_JSON:-data/scenic/SCENIC_full_training_dataset.json}"
CONTRASTIVE_TRAIN_JSON="${CONTRASTIVE_TRAIN_JSON:-data/scenic/SCENIC_full_anchor_positive_negative.json}"
CONTRASTIVE_JSON="${CONTRASTIVE_JSON:-$CONTRASTIVE_TRAIN_JSON}"
BENCHMARK_JSON="${BENCHMARK_JSON:-data/scenic/iot_instruction_benchmark_200.json}"
BENCHMARK_DIFFICULTY_PATH="${BENCHMARK_DIFFICULTY_PATH:-}"
TOKENIZER_PATH="${TOKENIZER_PATH:-}"

RETRAIN="${RETRAIN:-1}"
OVERWRITE="${OVERWRITE:-1}"
TRAIN_WITH_TORCHRUN="${TRAIN_WITH_TORCHRUN:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
SPARSITY_GPU_IDS="${SPARSITY_GPU_IDS:-$CUDA_VISIBLE_DEVICES}"
SPARSITY_MAX_PARALLEL_GPU_JOBS="${SPARSITY_MAX_PARALLEL_GPU_JOBS:-${MAX_PARALLEL_GPU_JOBS:-}}"
LEGACY_WITH_TORCHRUN="${LEGACY_WITH_TORCHRUN:-1}"
SFT_EPOCHS="${SFT_EPOCHS:-5}"

ORIGINAL_METHODS="${ORIGINAL_METHODS:-magnitude,wanda,gradient,nvidia24}"
ORIGINAL_SPARSITY_LEVELS="${ORIGINAL_SPARSITY_LEVELS:-${ORIGINAL_SPARSITY:-0.3 0.5}}"
ORIGINAL_RUN_ROOT="${ORIGINAL_RUN_ROOT:-$RUN_ROOT/original_one_shot_reference_methods}"
ORIGINAL_RESULT_ROOT="${ORIGINAL_RESULT_ROOT:-$RESULT_ROOT/original_one_shot_reference_methods}"
PRUNE_SCOPE="${PRUNE_SCOPE:-encoder-linear}"
PRUNE_DEVICE="${PRUNE_DEVICE:-auto}"
PRUNE_DTYPE="${PRUNE_DTYPE:-fp32}"
CALIBRATION_BATCH_SIZE="${CALIBRATION_BATCH_SIZE:-4}"
CALIBRATION_BATCHES="${CALIBRATION_BATCHES:-64}"
GRADIENT_CALIBRATION_BATCH_SIZE="${GRADIENT_CALIBRATION_BATCH_SIZE:-$CALIBRATION_BATCH_SIZE}"
GRADIENT_CALIBRATION_BATCHES="${GRADIENT_CALIBRATION_BATCHES:-$CALIBRATION_BATCHES}"
REINIT_CLASSIFIER="${REINIT_CLASSIFIER:-1}"
CLASSIFIER_INIT_BATCH_SIZE="${CLASSIFIER_INIT_BATCH_SIZE:-128}"
CLASSIFIER_INIT_MAX_LENGTH="${CLASSIFIER_INIT_MAX_LENGTH:-128}"

BATCH_SIZE="${BATCH_SIZE:-128}"
MAX_LENGTH="${MAX_LENGTH:-128}"
EVAL_DTYPE="${EVAL_DTYPE:-auto}"
SEED="${SEED:-42}"
SPARSITY_LEVELS="${SPARSITY_LEVELS:-0.3 0.5}"
RETUNE_EPOCHS="${RETUNE_EPOCHS:-1}"
RECOVERY_EPOCHS_PER_STAGE="${RECOVERY_EPOCHS_PER_STAGE:-$RETUNE_EPOCHS}"
FINAL_RECOVERY_EPOCHS="${FINAL_RECOVERY_EPOCHS:-1}"
LINEAR_PRUNING_MODES="${LINEAR_PRUNING_MODES:-dense progressive}"
GRADUAL_PRUNE_METHOD="${GRADUAL_PRUNE_METHOD:-magnitude}"
SPARSITY_DEVICE="${SPARSITY_DEVICE:-auto}"
RUN_PLOTS="${RUN_PLOTS:-1}"

mkdir -p "$RUN_ROOT" "$RESULT_ROOT" "$ORIGINAL_RUN_ROOT" "$ORIGINAL_RESULT_ROOT"

echo "[sparsity-revision] base_model=$BASE_MODEL"
echo "[sparsity-revision] base_encoder_baseline=$INCLUDE_BASE_ENCODER_BASELINE checkpoint=$BASELINE_DENSE_CHECKPOINT"
echo "[sparsity-revision] regular_dense_checkpoint=$REGULAR_DENSE_CHECKPOINT"
echo "[sparsity-revision] contrastive_dense_checkpoint=$CONTRASTIVE_DENSE_CHECKPOINT"
echo "[sparsity-revision] sft_epochs=$SFT_EPOCHS train_with_torchrun=$TRAIN_WITH_TORCHRUN nproc=$NPROC_PER_NODE"
echo "[sparsity-revision] cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
echo "[sparsity-revision] legacy_with_torchrun=$LEGACY_WITH_TORCHRUN"
echo "[sparsity-revision] sparsity_gpu_ids=$SPARSITY_GPU_IDS"

write_sft_config() {
  local template_path="$1"
  local output_path="$2"
  local run_dir="$3"
  local train_json="$4"
  local contrastive_json="$5"

  "$PYTHON" - "$template_path" "$output_path" "$BASE_MODEL" "$run_dir" "$train_json" "$contrastive_json" "$TOKENIZER_PATH" "$SFT_EPOCHS" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

import yaml

(
    template_path,
    output_path,
    base_model,
    run_dir,
    train_json,
    contrastive_json,
    tokenizer_path,
    epochs,
) = sys.argv[1:]
with Path(template_path).open("r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle) or {}

config.setdefault("run", {})["output_dir"] = run_dir
config.setdefault("model", {})["base_model"] = base_model
if tokenizer_path:
    config["model"]["tokenizer_path"] = tokenizer_path

data_config = config.setdefault("data", {})
data_config["train_json"] = train_json
data_config["contrastive_json"] = None if contrastive_json == "__NONE__" else contrastive_json
config.setdefault("train", {})["epochs"] = int(epochs)

output = Path(output_path)
output.parent.mkdir(parents=True, exist_ok=True)
with output.open("w", encoding="utf-8") as handle:
    yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
PY
}

train_sft_variant() {
  local variant="$1"
  local template_path="$2"
  local config_path="$3"
  local run_dir="$4"
  local checkpoint="$5"
  local train_json="$6"
  local contrastive_json="$7"

  if [[ "$RETRAIN" == "1" || ! -e "$checkpoint" ]]; then
    echo "[sparsity-revision] writing $variant SFT config -> $config_path"
    write_sft_config "$template_path" "$config_path" "$run_dir" "$train_json" "$contrastive_json"

    echo "[sparsity-revision] training $variant dense SFT checkpoint for $SFT_EPOCHS epochs"
    if [[ "$TRAIN_WITH_TORCHRUN" == "1" ]]; then
      CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" NPROC="$NPROC_PER_NODE" CONFIG="$config_path" ./scripts/launch_scenic_sft_8gpu.sh
    else
      "$PYTHON" scripts/train_scenic_sft.py --config "$config_path"
    fi
  else
    echo "[sparsity-revision] reusing existing $variant dense checkpoint"
  fi
}

train_sft_variant \
  "regular_sft" \
  "$REGULAR_SFT_CONFIG_TEMPLATE" \
  "$REGULAR_SFT_CONFIG" \
  "$REGULAR_SFT_RUN_DIR" \
  "$REGULAR_DENSE_CHECKPOINT" \
  "$TRAIN_JSON" \
  "__NONE__"

train_sft_variant \
  "contrastive_sft" \
  "$CONTRASTIVE_SFT_CONFIG_TEMPLATE" \
  "$CONTRASTIVE_SFT_CONFIG" \
  "$CONTRASTIVE_SFT_RUN_DIR" \
  "$CONTRASTIVE_DENSE_CHECKPOINT" \
  "$CONTRASTIVE_TRAIN_JSON" \
  "$CONTRASTIVE_JSON"

overwrite_args=()
if [[ "$OVERWRITE" == "1" ]]; then
  overwrite_args+=(--overwrite)
fi

parse_list() {
  local raw="$1"
  raw="${raw//,/ }"
  read -r -a PARSED_LIST <<< "$raw"
}

parse_gpu_ids() {
  local raw="$1"
  raw="${raw//,/ }"
  read -r -a GPU_ID_LIST <<< "$raw"
  if [[ "${#GPU_ID_LIST[@]}" -eq 0 ]]; then
    echo "[sparsity-revision] GPU_IDS resolved to no devices: $raw" >&2
    exit 2
  fi
}

gpu_cursor=0
next_gpu_id() {
  local gpu="${GPU_ID_LIST[$gpu_cursor]}"
  gpu_cursor=$(( (gpu_cursor + 1) % ${#GPU_ID_LIST[@]} ))
  echo "$gpu"
}

gpu_job_pids=()
gpu_job_names=()
run_gpu_job() {
  local job_name="$1"
  shift
  local gpu
  gpu="$(next_gpu_id)"
  echo "[sparsity-revision] launch gpu=$gpu job=$job_name"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    "$@"
  ) &
  gpu_job_pids+=("$!")
  gpu_job_names+=("$job_name")
  if [[ "${#gpu_job_pids[@]}" -ge "$MAX_PARALLEL_GPU_JOBS" ]]; then
    wait_gpu_jobs
  fi
}

wait_gpu_jobs() {
  local failed=0
  local index=0
  local pid
  for pid in "${gpu_job_pids[@]}"; do
    if ! wait "$pid"; then
      echo "[sparsity-revision] gpu job failed: ${gpu_job_names[$index]} (pid=$pid)" >&2
      failed=1
    fi
    index=$((index + 1))
  done
  gpu_job_pids=()
  gpu_job_names=()
  if [[ "$failed" != "0" ]]; then
    exit 1
  fi
}

device_for_isolated_gpu() {
  local requested="$1"
  if [[ "$requested" == "auto" ]]; then
    echo "cuda"
  else
    echo "$requested"
  fi
}

eval_checkpoint_json() {
  local checkpoint="$1"
  local json_path="$2"
  local predictions_path="$3"
  local summary_path="$4"

  "$PYTHON" scripts/eval_scenic_sft_local.py \
    --json "$json_path" \
    --checkpoint "$checkpoint" \
    --output "$predictions_path" \
    --summary-output "$summary_path" \
    --batch-size "$BATCH_SIZE" \
    --max-length "$MAX_LENGTH" \
    --dtype "$EVAL_DTYPE"
}

build_base_encoder_baseline() {
  local output_checkpoint="$1"

  "$PYTHON" - "$BASE_MODEL" "$TOKENIZER_PATH" "$TRAIN_JSON" "$REGULAR_SFT_CONFIG" "$REGULAR_SFT_CONFIG_TEMPLATE" "$output_checkpoint" "$CLASSIFIER_INIT_BATCH_SIZE" "$CLASSIFIER_INIT_MAX_LENGTH" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from chatlm_encoder.scenic_sft import (  # noqa: E402
    build_label_maps,
    initialize_classifier_from_responses,
    load_base_scenic_model,
    load_prompt_response_rows,
    save_scenic_checkpoint,
)

(
    base_model,
    tokenizer_path,
    train_json,
    regular_config_path,
    regular_template_path,
    output_checkpoint,
    classifier_init_batch_size,
    classifier_init_max_length,
) = sys.argv[1:]

config_path = Path(regular_config_path)
if not config_path.exists():
    config_path = Path(regular_template_path)
with config_path.open("r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle) or {}

model_config = dict(config.get("model") or {})
data_config = dict(config.get("data") or {})
model_config["base_model"] = base_model
if tokenizer_path:
    model_config["tokenizer_path"] = tokenizer_path
data_config["train_json"] = train_json

rows = load_prompt_response_rows(train_json)
_response_to_label, label2response = build_label_maps(rows)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, tokenizer = load_base_scenic_model(
    base_model=base_model,
    tokenizer_path=model_config.get("tokenizer_path"),
    num_labels=len(label2response),
    dropout=float(model_config.get("dropout", 0.1)),
    pooling=str(model_config.get("pooling", "cls")),
    normalize_logits=bool(model_config.get("normalize_logits", False)),
    logit_scale=float(model_config.get("logit_scale", 20.0)),
)
model.to(device)
initialize_classifier_from_responses(
    model=model,
    tokenizer=tokenizer,
    label2response=label2response,
    device=device,
    max_length=int(classifier_init_max_length or data_config.get("max_length", 128)),
    batch_size=int(classifier_init_batch_size or model_config.get("classifier_init_batch_size", 128)),
)
metadata = {
    "step": 0,
    "finished": True,
    "baseline": "base_encoder_dense",
    "train_rows": len(rows),
    "classifier_initialized_from_responses": True,
    "config": {
        **config,
        "model": model_config,
        "data": data_config,
    },
}
save_scenic_checkpoint(model, tokenizer, output_checkpoint, label2response, metadata)
print(f"[sparsity-revision] saved base encoder baseline checkpoint: {output_checkpoint}")
PY
}

if [[ "$INCLUDE_BASE_ENCODER_BASELINE" == "1" ]]; then
  mkdir -p "$BASELINE_RESULT_ROOT"
  if [[ "$OVERWRITE" == "1" || ! -e "$BASELINE_DENSE_CHECKPOINT" ]]; then
    echo "[sparsity-revision] building base encoder dense SCENIC baseline"
    build_base_encoder_baseline "$BASELINE_DENSE_CHECKPOINT"
  else
    echo "[sparsity-revision] reusing base encoder dense SCENIC baseline"
  fi

  echo "[sparsity-revision] eval base encoder baseline on training data"
  eval_checkpoint_json \
    "$BASELINE_DENSE_CHECKPOINT" \
    "$TRAIN_JSON" \
    "$BASELINE_RESULT_ROOT/training_predictions.jsonl" \
    "$BASELINE_RESULT_ROOT/training_summary.json"

  echo "[sparsity-revision] eval base encoder baseline on benchmark data"
  eval_checkpoint_json \
    "$BASELINE_DENSE_CHECKPOINT" \
    "$BENCHMARK_JSON" \
    "$BASELINE_RESULT_ROOT/benchmark_predictions.jsonl" \
    "$BASELINE_RESULT_ROOT/benchmark_summary.json"
fi

parse_list "$ORIGINAL_METHODS"
method_names=("${PARSED_LIST[@]}")
parse_list "$ORIGINAL_SPARSITY_LEVELS"
original_sparsity_levels=("${PARSED_LIST[@]}")

completed_original=()
skipped_original=()

echo "[sparsity-revision] original methods stay one-shot; classifier_rebuild=$REINIT_CLASSIFIER"
for variant_spec in \
  "regular_sft|$REGULAR_DENSE_CHECKPOINT|$TRAIN_JSON" \
  "contrastive_sft|$CONTRASTIVE_DENSE_CHECKPOINT|$CONTRASTIVE_TRAIN_JSON"; do
  IFS='|' read -r variant checkpoint train_json <<< "$variant_spec"

  for raw_method in "${method_names[@]}"; do
    method="${raw_method//[[:space:]]/}"
    if [[ -z "$method" ]]; then
      continue
    fi

    case "$method" in
      magnitude)
        label="magnitude"
        ;;
      nvidia|nvidia-2:4|nvidia_2_4|nvidia24|nvidia2:4|nvidia2_4)
        method="nvidia"
        label="nvidia24"
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
        echo "[sparsity-revision] skip one-shot $variant nvidia at sparsity=$sparsity; NVIDIA 2:4 is exactly 50%"
        skipped_original+=("$variant|$label|$sparsity|NVIDIA 2:4 is exactly 50% sparse")
        continue
      fi

      sparsity_tag="${sparsity//./p}"
      run_label="${label}_${sparsity_tag}"
      completed_original+=("$variant|$checkpoint|$train_json|$run_label|$label|$method|$sparsity")
    done
  done
done

ORIGINAL_JOBS_JSON="$ORIGINAL_RUN_ROOT/one_shot_jobs.json"
"$PYTHON" - "$ORIGINAL_JOBS_JSON" "$ORIGINAL_RUN_ROOT" "$ORIGINAL_RESULT_ROOT" "$BENCHMARK_JSON" "$PRUNE_SCOPE" "$PRUNE_DEVICE" "$PRUNE_DTYPE" "$CALIBRATION_BATCH_SIZE" "$CALIBRATION_BATCHES" "$MAX_LENGTH" "$BATCH_SIZE" "$EVAL_DTYPE" "$OVERWRITE" "$REINIT_CLASSIFIER" "$CLASSIFIER_INIT_BATCH_SIZE" "$CLASSIFIER_INIT_MAX_LENGTH" "${completed_original[@]}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

(
    output_path,
    original_run_root,
    original_result_root,
    benchmark_json,
    prune_scope,
    prune_device,
    prune_dtype,
    calibration_batch_size,
    calibration_batches,
    max_length,
    batch_size,
    eval_dtype,
    overwrite,
    reinit_classifier,
    classifier_init_batch_size,
    classifier_init_max_length,
) = sys.argv[1:17]
specs = sys.argv[17:]

jobs = []
for spec in specs:
    variant, checkpoint, train_json, run_label, label, method, sparsity = spec.split("|", 6)
    jobs.append(
        {
            "variant": variant,
            "checkpoint": checkpoint,
            "train_json": train_json,
            "run_label": run_label,
            "label": label,
            "method": method,
            "sparsity": float(sparsity),
            "pruned_checkpoint": str(Path(original_run_root) / variant / run_label),
            "eval_dir": str(Path(original_result_root) / variant / run_label),
        }
    )

payload = {
    "settings": {
        "benchmark_json": benchmark_json,
        "prune_scope": prune_scope,
        "prune_device": prune_device,
        "prune_dtype": prune_dtype,
        "calibration_batch_size": int(calibration_batch_size),
        "calibration_batches": int(calibration_batches),
        "max_length": int(max_length),
        "batch_size": int(batch_size),
        "eval_dtype": eval_dtype,
        "overwrite": overwrite == "1",
        "reinitialize_classifier": reinit_classifier == "1",
        "classifier_init_batch_size": int(classifier_init_batch_size),
        "classifier_init_max_length": int(classifier_init_max_length),
    },
    "jobs": jobs,
}
output = Path(output_path)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"[sparsity-revision] wrote one-shot torchrun jobs: {output}")
PY

if [[ "$LEGACY_WITH_TORCHRUN" == "1" ]]; then
  echo "[sparsity-revision] running legacy one-shot pruning with torchrun nproc=$NPROC_PER_NODE"
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" torchrun --standalone --nproc_per_node "$NPROC_PER_NODE" \
    scripts/run_scenic_one_shot_pruning_worker.py \
    --jobs-json "$ORIGINAL_JOBS_JSON"
else
  echo "[sparsity-revision] running legacy one-shot pruning single-process"
  "$PYTHON" scripts/run_scenic_one_shot_pruning_worker.py --jobs-json "$ORIGINAL_JOBS_JSON"
fi

"$PYTHON" - "$ORIGINAL_RESULT_ROOT" "${completed_original[@]}" <<'PY'
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

root = Path(sys.argv[1])
specs = sys.argv[2:]


def metric_block(summary: dict[str, Any], summary_path: Path) -> dict[str, Any]:
    difficulty = {}
    for label in ("easy", "medium", "hard"):
        bucket = summary.get("groups", {}).get("difficulty", {}).get(label, {})
        difficulty[label] = {
            "rows": bucket.get("rows"),
            "scored_rows": bucket.get("scored_rows"),
            "em1": bucket.get("exact_match_accuracy"),
            "em5": bucket.get("top5_accuracy"),
        }
    return {
        "json": summary.get("json"),
        "rows": summary.get("rows"),
        "scored_rows": summary.get("scored_rows"),
        "em1": summary.get("exact_match_accuracy"),
        "em5": summary.get("top5_accuracy"),
        "difficulty": difficulty,
        "summary_output": str(summary_path),
        "predictions_output": summary.get("predictions_output"),
    }


rows = []
for spec in specs:
    variant, _checkpoint, _train_json, run_label, label, method, sparsity = spec.split("|", 6)
    eval_dir = root / variant / run_label
    benchmark_summary_path = eval_dir / "benchmark_summary.json"
    training_summary_path = eval_dir / "training_summary.json"
    if not benchmark_summary_path.exists():
        continue
    benchmark_summary = json.loads(benchmark_summary_path.read_text(encoding="utf-8"))
    training_summary = (
        json.loads(training_summary_path.read_text(encoding="utf-8"))
        if training_summary_path.exists()
        else {}
    )
    checkpoint = Path(str(benchmark_summary.get("checkpoint") or ""))
    pruning = {}
    metadata_path = checkpoint / "scenic_sft_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        pruning = metadata.get("pruning") if isinstance(metadata.get("pruning"), dict) else {}
    rows.append({
        "result_block": "original_one_shot_reference_methods",
        "model_training": variant,
        "run_label": run_label,
        "pruning_mode": "oneshot",
        "method": label,
        "pruning_method": method,
        "target_sparsity": float(sparsity),
        "targeted_linear_sparsity_actual": pruning.get("targeted_sparsity_after"),
        "whole_model_sparsity_actual": pruning.get("model_sparsity_after"),
        "checkpoint": benchmark_summary.get("checkpoint"),
        "training_metrics": metric_block(training_summary, training_summary_path) if training_summary else None,
        "benchmark_metrics": metric_block(benchmark_summary, benchmark_summary_path),
    })

json_path = root / "original_one_shot_summary.json"
csv_path = root / "original_one_shot_summary.csv"
json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
with csv_path.open("w", encoding="utf-8", newline="") as handle:
    fieldnames = [
        "result_block",
        "model_training",
        "run_label",
        "pruning_mode",
        "method",
        "pruning_method",
        "target_sparsity",
        "targeted_linear_sparsity_actual",
        "whole_model_sparsity_actual",
        "checkpoint",
        "training_em1",
        "training_em5",
        "benchmark_em1",
        "benchmark_em5",
        "benchmark_easy_em1",
        "benchmark_easy_em5",
        "benchmark_medium_em1",
        "benchmark_medium_em5",
        "benchmark_hard_em1",
        "benchmark_hard_em5",
    ]
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        training = row.get("training_metrics") or {}
        benchmark = row.get("benchmark_metrics") or {}
        difficulty = benchmark.get("difficulty") or {}
        writer.writerow({
            "result_block": row.get("result_block"),
            "model_training": row.get("model_training"),
            "run_label": row.get("run_label"),
            "pruning_mode": row.get("pruning_mode"),
            "method": row.get("method"),
            "pruning_method": row.get("pruning_method"),
            "target_sparsity": row.get("target_sparsity"),
            "targeted_linear_sparsity_actual": row.get("targeted_linear_sparsity_actual"),
            "whole_model_sparsity_actual": row.get("whole_model_sparsity_actual"),
            "checkpoint": row.get("checkpoint"),
            "training_em1": training.get("em1"),
            "training_em5": training.get("em5"),
            "benchmark_em1": benchmark.get("em1"),
            "benchmark_em5": benchmark.get("em5"),
            "benchmark_easy_em1": difficulty.get("easy", {}).get("em1"),
            "benchmark_easy_em5": difficulty.get("easy", {}).get("em5"),
            "benchmark_medium_em1": difficulty.get("medium", {}).get("em1"),
            "benchmark_medium_em5": difficulty.get("medium", {}).get("em5"),
            "benchmark_hard_em1": difficulty.get("hard", {}).get("em1"),
            "benchmark_hard_em5": difficulty.get("hard", {}).get("em5"),
        })
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

parse_list "$SPARSITY_LEVELS"
linear_sparsity_levels=("${PARSED_LIST[@]}")
parse_list "$LINEAR_PRUNING_MODES"
linear_pruning_modes=("${PARSED_LIST[@]}")
parse_gpu_ids "$SPARSITY_GPU_IDS"
if [[ -z "$SPARSITY_MAX_PARALLEL_GPU_JOBS" ]]; then
  MAX_PARALLEL_GPU_JOBS="${#GPU_ID_LIST[@]}"
else
  MAX_PARALLEL_GPU_JOBS="$SPARSITY_MAX_PARALLEL_GPU_JOBS"
fi
if [[ "$MAX_PARALLEL_GPU_JOBS" -lt 1 ]]; then
  echo "[sparsity-revision] SPARSITY_MAX_PARALLEL_GPU_JOBS must be >= 1" >&2
  exit 2
fi
if [[ "$MAX_PARALLEL_GPU_JOBS" -gt "${#GPU_ID_LIST[@]}" ]]; then
  MAX_PARALLEL_GPU_JOBS="${#GPU_ID_LIST[@]}"
fi
echo "[sparsity-revision] progressive jobs use SPARSITY_GPU_IDS=$SPARSITY_GPU_IDS max_parallel=$MAX_PARALLEL_GPU_JOBS"

run_linear_retune_job() {
  local variant="$1"
  local checkpoint="$2"
  local train_json="$3"
  local output_dir="$RESULT_ROOT/linear_sparsity_retune/$variant"
  local sparsity_device
  sparsity_device="$(device_for_isolated_gpu "$SPARSITY_DEVICE")"

  echo "[sparsity-revision] added linear-sparsity block $variant: method=$GRADUAL_PRUNE_METHOD modes=${linear_pruning_modes[*]} per_stage_recovery_epochs=$RECOVERY_EPOCHS_PER_STAGE final_recovery_epochs=$FINAL_RECOVERY_EPOCHS"
  "$PYTHON" scripts/run_sparsity_experiments.py \
    --experiment_name "$EXPERIMENT_NAME" \
    --model_family encoder_only \
    --model_checkpoint "$checkpoint" \
    --benchmark_path "$BENCHMARK_JSON" \
    "${difficulty_args[@]}" \
    --sparsity_levels "${linear_sparsity_levels[@]}" \
    --pruning_modes "${linear_pruning_modes[@]}" \
    --prune_scope linear_weights \
    --prune_method "$GRADUAL_PRUNE_METHOD" \
    --gradient_calibration_batch_size "$GRADIENT_CALIBRATION_BATCH_SIZE" \
    --gradient_calibration_batches "$GRADIENT_CALIBRATION_BATCHES" \
    --progressive_schedule staged \
    --recovery_epochs_per_stage "$RECOVERY_EPOCHS_PER_STAGE" \
    --final_recovery_epochs "$FINAL_RECOVERY_EPOCHS" \
    --recovery_train_path "$train_json" \
    --batch_size "$BATCH_SIZE" \
    --max_length "$MAX_LENGTH" \
    --seed "$SEED" \
    --device "$sparsity_device" \
    --output_dir "$output_dir" \
    "${block_head_args[@]}"
}

run_gpu_job "regular_sft:linear_retune" run_linear_retune_job "regular_sft" "$REGULAR_DENSE_CHECKPOINT" "$TRAIN_JSON"
run_gpu_job "contrastive_sft:linear_retune" run_linear_retune_job "contrastive_sft" "$CONTRASTIVE_DENSE_CHECKPOINT" "$CONTRASTIVE_TRAIN_JSON"
wait_gpu_jobs

run_retune_training_eval_job() {
  local checkpoint="$1"
  local train_json="$2"
  local eval_dir="$3"
  mkdir -p "$eval_dir"
  echo "[sparsity-revision] eval retune checkpoint on training data -> $eval_dir"
  eval_checkpoint_json \
    "$checkpoint" \
    "$train_json" \
    "$eval_dir/training_predictions.jsonl" \
    "$eval_dir/training_summary.json"
}

queue_retune_training_evals() {
  local variant="$1"
  local train_json="$2"
  local summary_csv="$RESULT_ROOT/linear_sparsity_retune/$variant/summary_metrics.csv"
  local eval_root="$RESULT_ROOT/linear_sparsity_retune/$variant/training_eval"

  if [[ ! -e "$summary_csv" ]]; then
    return
  fi

  while IFS= read -r spec; do
    if [[ -z "$spec" ]]; then
      continue
    fi
    IFS='|' read -r job_name checkpoint eval_dir <<< "$spec"
    run_gpu_job "$job_name" run_retune_training_eval_job "$checkpoint" "$train_json" "$eval_dir"
  done < <("$PYTHON" - "$summary_csv" "$variant" "$eval_root" <<'PY'
from __future__ import annotations

import csv
import sys
from pathlib import Path

summary_csv = Path(sys.argv[1])
variant = sys.argv[2]
eval_root = Path(sys.argv[3])


def slug(value: str) -> str:
    return str(value).strip().replace(".", "p")


with summary_csv.open("r", encoding="utf-8", newline="") as handle:
    for row in csv.DictReader(handle):
        checkpoint = row.get("checkpoint_path")
        if not checkpoint:
            continue
        mode = row.get("pruning_mode") or "unknown"
        target = row.get("target_sparsity") or "unknown"
        seed = row.get("seed") or "noseed"
        eval_dir = eval_root / f"{mode}_{slug(target)}_{seed}"
        print(f"{variant}:{mode}_{target}|{checkpoint}|{eval_dir}")
PY
  )
}

queue_retune_training_evals "regular_sft" "$TRAIN_JSON"
queue_retune_training_evals "contrastive_sft" "$CONTRASTIVE_TRAIN_JSON"
wait_gpu_jobs

if [[ "$RUN_PLOTS" == "1" ]]; then
  for variant in regular_sft contrastive_sft; do
    if ! "$PYTHON" scripts/plot_sparsity_results.py \
      --experiment_name "$EXPERIMENT_NAME" \
      --results_dir "$RESULT_ROOT/linear_sparsity_retune/$variant"; then
      echo "[sparsity-revision] plot generation failed for $variant; metrics are still available. Install matplotlib and rerun scripts/plot_sparsity_results.py." >&2
    fi
  done
fi

"$PYTHON" - "$RESULT_ROOT" "$ORIGINAL_RESULT_ROOT/original_one_shot_summary.json" "$BASE_MODEL" "$SFT_EPOCHS" "$GRADUAL_PRUNE_METHOD" "$RECOVERY_EPOCHS_PER_STAGE" "$FINAL_RECOVERY_EPOCHS" "$GRADIENT_CALIBRATION_BATCH_SIZE" "$GRADIENT_CALIBRATION_BATCHES" "${skipped_original[@]}" <<'PY'
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

result_root = Path(sys.argv[1])
original_json = Path(sys.argv[2])
base_model = sys.argv[3]
sft_epochs = int(sys.argv[4])
gradual_prune_method = sys.argv[5]
recovery_epochs_per_stage = int(sys.argv[6])
final_recovery_epochs = int(sys.argv[7])
gradient_calibration_batch_size = int(sys.argv[8])
gradient_calibration_batches = int(sys.argv[9])
skipped_specs = sys.argv[10:]


def as_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    return float(value)


def as_int(value: Any) -> int | None:
    if value in {None, ""}:
        return None
    return int(float(value))


def flat_metric_fields(prefix: str, metrics: dict[str, Any] | None) -> dict[str, Any]:
    metrics = metrics or {}
    difficulty = metrics.get("difficulty") or {}
    values = {
        f"{prefix}_em1_overall": metrics.get("em1"),
        f"{prefix}_em5_overall": metrics.get("em5"),
        f"{prefix}_count_total": metrics.get("rows"),
        f"{prefix}_scored_rows": metrics.get("scored_rows"),
    }
    for label in ("easy", "medium", "hard"):
        group = difficulty.get(label, {})
        values[f"{prefix}_em1_{label}"] = group.get("em1")
        values[f"{prefix}_em5_{label}"] = group.get("em5")
        values[f"{prefix}_count_{label}"] = group.get("rows")
    return values


def metric_block_from_eval_summary(summary_path: Path) -> dict[str, Any] | None:
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    difficulty = {}
    for label in ("easy", "medium", "hard"):
        bucket = summary.get("groups", {}).get("difficulty", {}).get(label, {})
        difficulty[label] = {
            "rows": bucket.get("rows"),
            "scored_rows": bucket.get("scored_rows"),
            "em1": bucket.get("exact_match_accuracy"),
            "em5": bucket.get("top5_accuracy"),
        }
    return {
        "json": summary.get("json"),
        "rows": summary.get("rows"),
        "scored_rows": summary.get("scored_rows"),
        "em1": summary.get("exact_match_accuracy"),
        "em5": summary.get("top5_accuracy"),
        "difficulty": difficulty,
        "summary_output": str(summary_path),
        "predictions_output": summary.get("predictions_output"),
    }


def slug(value: Any) -> str:
    return str(value).strip().replace(".", "p")


rows: list[dict[str, Any]] = []
base_training_summary_path = result_root / "base_encoder_dense" / "training_summary.json"
base_benchmark_summary_path = result_root / "base_encoder_dense" / "benchmark_summary.json"
base_training_metrics = metric_block_from_eval_summary(base_training_summary_path)
base_benchmark_metrics = metric_block_from_eval_summary(base_benchmark_summary_path)
base_encoder_baseline_rows_expected = 1 if (base_training_metrics or base_benchmark_metrics) else 0
if base_training_metrics or base_benchmark_metrics:
    checkpoint_path = None
    for summary_path in (base_benchmark_summary_path, base_training_summary_path):
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            checkpoint_path = summary.get("checkpoint")
            if checkpoint_path:
                break
    rows.append(
        {
            "result_block": "base_encoder_baseline",
            "model_training": "base_encoder",
            "run_label": "base_encoder_dense_0.0",
            "model_family": "encoder_only",
            "pruning_mode": "dense",
            "pruning_method": "dense",
            "method_label": "dense",
            "target_sparsity": 0.0,
            "targeted_linear_sparsity_actual": 0.0,
            "whole_model_sparsity_actual": 0.0,
            "seed": None,
            "checkpoint_path": checkpoint_path,
            "mask_path": None,
            "training_metrics": base_training_metrics,
            "benchmark_metrics": base_benchmark_metrics,
            **flat_metric_fields("training", base_training_metrics),
            **flat_metric_fields("benchmark", base_benchmark_metrics),
            "notes": "base encoder-only dense baseline; response classifier initialized from training responses; no SFT recovery or pruning",
        }
    )

if original_json.exists():
    for row in json.loads(original_json.read_text(encoding="utf-8")):
        training_metrics = row.get("training_metrics")
        benchmark_metrics = row.get("benchmark_metrics")
        rows.append(
            {
                "result_block": row.get("result_block") or "original_one_shot_reference_methods",
                "model_training": row.get("model_training"),
                "run_label": row.get("run_label"),
                "model_family": "encoder_only",
                "pruning_mode": row.get("pruning_mode") or "oneshot",
                "pruning_method": row.get("pruning_method") or row.get("method"),
                "method_label": row.get("method"),
                "target_sparsity": row.get("target_sparsity"),
                "targeted_linear_sparsity_actual": row.get("targeted_linear_sparsity_actual"),
                "whole_model_sparsity_actual": row.get("whole_model_sparsity_actual"),
                "seed": None,
                "checkpoint_path": row.get("checkpoint"),
                "mask_path": None,
                "training_metrics": training_metrics,
                "benchmark_metrics": benchmark_metrics,
                **flat_metric_fields("training", training_metrics),
                **flat_metric_fields("benchmark", benchmark_metrics),
                "notes": "reference one-shot pruning; classifier rebuild follows REINIT_CLASSIFIER",
            }
        )

retune_csvs = sorted((result_root / "linear_sparsity_retune").glob("*/summary_metrics.csv"))
for retune_csv in retune_csvs:
    variant = retune_csv.parent.name
    with retune_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            target_sparsity = row.get("target_sparsity")
            seed = row.get("seed") or "noseed"
            training_summary_path = (
                retune_csv.parent
                / "training_eval"
                / f"{row.get('pruning_mode')}_{slug(target_sparsity)}_{seed}"
                / "training_summary.json"
            )
            training_metrics = metric_block_from_eval_summary(training_summary_path)
            benchmark_metrics = {
                "rows": as_int(row.get("count_total")),
                "scored_rows": as_int(row.get("count_total")),
                "em1": as_float(row.get("em1_overall")),
                "em5": as_float(row.get("em5_overall")),
                "difficulty": {
                    label: {
                        "rows": as_int(row.get(f"count_{label}")),
                        "scored_rows": as_int(row.get(f"count_{label}")),
                        "em1": as_float(row.get(f"em1_{label}")),
                        "em5": as_float(row.get(f"em5_{label}")),
                    }
                    for label in ("easy", "medium", "hard")
                },
                "summary_output": str(retune_csv),
                "predictions_output": row.get("predictions_path"),
            }
            rows.append(
                {
                    "result_block": "linear_sparsity_retune",
                    "model_training": variant,
                    "run_label": f"{variant}_{row.get('pruning_mode')}_{row.get('target_sparsity')}",
                    "model_family": row.get("model_family"),
                    "pruning_mode": row.get("pruning_mode"),
                    "pruning_method": row.get("pruning_method"),
                    "method_label": row.get("pruning_method"),
                    "target_sparsity": as_float(row.get("target_sparsity")),
                    "targeted_linear_sparsity_actual": as_float(row.get("targeted_linear_sparsity_actual")),
                    "whole_model_sparsity_actual": as_float(row.get("whole_model_sparsity_actual")),
                    "seed": int(row["seed"]) if row.get("seed") not in {None, ""} else None,
                    "checkpoint_path": row.get("checkpoint_path"),
                    "mask_path": row.get("mask_path"),
                    "training_metrics": training_metrics,
                    "benchmark_metrics": benchmark_metrics,
                    **flat_metric_fields("training", training_metrics),
                    **flat_metric_fields("benchmark", benchmark_metrics),
                    "em1_retention_overall": as_float(row.get("em1_retention_overall")),
                    "em5_retention_overall": as_float(row.get("em5_retention_overall")),
                    "training_config": json.loads(row["training_config_json"]) if row.get("training_config_json") else None,
                    "pruning_config": json.loads(row["pruning_config_json"]) if row.get("pruning_config_json") else None,
                    "notes": "linear-sparsity retune block from run_sparsity_experiments.py",
                }
            )

skipped = []
for spec in skipped_specs:
    variant, method, sparsity, reason = spec.split("|", 3)
    skipped.append({"model_training": variant, "method": method, "target_sparsity": float(sparsity), "reason": reason})

payload = {
    "report_type": "scenic_sparsity_revision_combined_results",
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "experiment": {
        "base_model": base_model,
        "sft_epochs": sft_epochs,
        "model_trainings": ["regular_sft", "contrastive_sft"],
        "base_encoder_baseline_expected_rows_total": base_encoder_baseline_rows_expected,
        "original_one_shot_expected_rows_per_training": 7,
        "original_one_shot_expected_rows_total": 14,
        "sft_dense_baseline_expected_rows_total": 2,
        "dense_baseline_expected_rows_total": 2 + base_encoder_baseline_rows_expected,
        "progressive_expected_rows_per_training": 2,
        "progressive_expected_rows_total": 4,
        "expected_rows_total": 20 + base_encoder_baseline_rows_expected,
        "gradual_prune_method": gradual_prune_method,
        "recovery_epochs_per_stage": recovery_epochs_per_stage,
        "final_recovery_epochs": final_recovery_epochs,
        "gradient_calibration_batch_size": (
            gradient_calibration_batch_size if gradual_prune_method == "gradient" else None
        ),
        "gradient_calibration_batches": (
            gradient_calibration_batches if gradual_prune_method == "gradient" else None
        ),
    },
    "source_files": {
        "base_encoder_baseline_training_summary": str(base_training_summary_path),
        "base_encoder_baseline_benchmark_summary": str(base_benchmark_summary_path),
        "original_one_shot_summary": str(original_json),
        "linear_sparsity_retune_summaries": [str(path) for path in retune_csvs],
    },
    "actual_rows_total": len(rows),
    "rows": rows,
    "skipped": skipped,
}
output = result_root / "all_sparsity_results.json"
output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"[sparsity-revision] wrote {output}")
PY

echo "[sparsity-revision] done"
echo "  original one-shot: $ORIGINAL_RESULT_ROOT/original_one_shot_summary.csv"
echo "  regular retune block: $RESULT_ROOT/linear_sparsity_retune/regular_sft/summary_metrics.csv"
echo "  contrastive retune block: $RESULT_ROOT/linear_sparsity_retune/contrastive_sft/summary_metrics.csv"
echo "  all results json: $RESULT_ROOT/all_sparsity_results.json"
