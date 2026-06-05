#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1

DENSE_CHECKPOINT="${DENSE_CHECKPOINT:-runs/scenic-sft-training-dataset/latest}"
RUN_ROOT="${RUN_ROOT:-runs/scenic-onnx-nvidia}"
PRUNED_CHECKPOINT="${PRUNED_CHECKPOINT:-$RUN_ROOT/training_dataset_nvidia_2_4}"
ONNX_ROOT="${ONNX_ROOT:-$RUN_ROOT/onnx}"
EVAL_ROOT="${EVAL_ROOT:-eval_results/scenic_sft/onnx_nvidia}"

BENCHMARK_JSON="${BENCHMARK_JSON:-data/scenic/iot_instruction_benchmark_200.json}"
TRAINING_JSON="${TRAINING_JSON:-data/scenic/SCENIC_full_training_dataset.json}"

SPARSITY="${SPARSITY:-0.5}"
PRUNE_SCOPE="${PRUNE_SCOPE:-encoder-linear}"
PRUNE_DEVICE="${PRUNE_DEVICE:-auto}"
PRUNE_DTYPE="${PRUNE_DTYPE:-fp32}"
REBUILD_PRUNED="${REBUILD_PRUNED:-0}"
REINIT_CLASSIFIER="${REINIT_CLASSIFIER:-1}"
CLASSIFIER_INIT_BATCH_SIZE="${CLASSIFIER_INIT_BATCH_SIZE:-128}"
CLASSIFIER_INIT_MAX_LENGTH="${CLASSIFIER_INIT_MAX_LENGTH:-128}"

BATCH_SIZE="${BATCH_SIZE:-128}"
MAX_LENGTH="${MAX_LENGTH:-128}"
OPSET="${OPSET:-17}"
PROVIDERS="${PROVIDERS:-auto}"
OVERWRITE="${OVERWRITE:-1}"
RUN_INT8="${RUN_INT8:-1}"
RUN_EDGE_BENCHMARK="${RUN_EDGE_BENCHMARK:-0}"
RUN_TENSORRT="${RUN_TENSORRT:-auto}"

EDGE_ROOT="${EDGE_ROOT:-$EVAL_ROOT/edge_runtime}"
EDGE_BATCH_SIZE="${EDGE_BATCH_SIZE:-1}"
EDGE_INPUT_LENGTHS="${EDGE_INPUT_LENGTHS:-64,128}"
EDGE_WARMUP_QUERIES="${EDGE_WARMUP_QUERIES:-20}"
EDGE_MEASURE_QUERIES="${EDGE_MEASURE_QUERIES:-200}"
EDGE_ONNX_PROVIDERS="${EDGE_ONNX_PROVIDERS:-auto}"
PYTORCH_DEVICE="${PYTORCH_DEVICE:-auto}"
TRT_CACHE_ROOT="${TRT_CACHE_ROOT:-$RUN_ROOT/tensorrt_cache}"

mkdir -p "$RUN_ROOT" "$ONNX_ROOT" "$EVAL_ROOT" "$EDGE_ROOT"

require_path() {
  local path="$1"
  local label="$2"
  if [[ ! -e "$path" ]]; then
    echo "[onnx-nvidia] missing $label: $path" >&2
    exit 1
  fi
}

check_onnx_dependencies() {
  python - <<'PY'
import importlib
import sys

required = [
    ("onnx", "onnx"),
    ("onnxruntime", "onnxruntime"),
    ("onnxconverter_common", "onnxconverter-common"),
]
missing = []
for module_name, package_name in required:
    try:
        importlib.import_module(module_name)
    except ImportError:
        missing.append(package_name)

if missing:
    print(
        "[onnx-nvidia] missing optional ONNX dependencies: "
        + ", ".join(missing),
        file=sys.stderr,
    )
    print(
        "[onnx-nvidia] install with: pip install " + " ".join(missing),
        file=sys.stderr,
    )
    print(
        "[onnx-nvidia] for CUDA/NVIDIA ORT execution, install onnxruntime-gpu instead of onnxruntime.",
        file=sys.stderr,
    )
    sys.exit(1)
PY
}

has_tensorrt_provider() {
  python - <<'PY'
import onnxruntime as ort
raise SystemExit(0 if "TensorrtExecutionProvider" in ort.get_available_providers() else 1)
PY
}

export_variant() {
  local variant="$1"
  local precision="$2"
  local checkpoint="$3"
  local onnx_path="$4"

  local overwrite_args=()
  if [[ "$OVERWRITE" == "1" ]]; then
    overwrite_args+=(--overwrite)
  fi

  mkdir -p "$(dirname "$onnx_path")"
  echo "[onnx-nvidia] export $variant ($precision) -> $onnx_path"
  python scripts/export_scenic_sft_onnx.py \
    --checkpoint "$checkpoint" \
    --output "$onnx_path" \
    --precision "$precision" \
    --max-length "$MAX_LENGTH" \
    --opset "$OPSET" \
    "${overwrite_args[@]}"
}

eval_variant() {
  local variant="$1"
  local checkpoint="$2"
  local onnx_path="$3"
  local dataset_name="$4"
  local dataset_path="$5"

  local output_dir="$EVAL_ROOT/$variant"
  mkdir -p "$output_dir"
  echo "[onnx-nvidia] eval $variant on $dataset_name"
  python scripts/eval_scenic_sft_onnx_local.py \
    --json "$dataset_path" \
    --checkpoint "$checkpoint" \
    --onnx "$onnx_path" \
    --output "$output_dir/${dataset_name}_predictions.jsonl" \
    --summary-output "$output_dir/${dataset_name}_summary.json" \
    --batch-size "$BATCH_SIZE" \
    --max-length "$MAX_LENGTH" \
    --providers "$PROVIDERS"
}

eval_pytorch_fp16_variant() {
  local variant="$1"
  local checkpoint="$2"

  local output_dir="$EVAL_ROOT/$variant"
  mkdir -p "$output_dir"
  echo "[onnx-nvidia] eval $variant on benchmark_200"
  python scripts/eval_scenic_sft_local.py \
    --json "$BENCHMARK_JSON" \
    --checkpoint "$checkpoint" \
    --output "$output_dir/benchmark_200_predictions.jsonl" \
    --summary-output "$output_dir/benchmark_200_summary.json" \
    --batch-size "$BATCH_SIZE" \
    --max-length "$MAX_LENGTH" \
    --dtype fp16
}

eval_tensorrt_fp16_variant() {
  local variant="$1"
  local checkpoint="$2"
  local onnx_path="$3"

  local output_dir="$EVAL_ROOT/$variant"
  local cache_dir="$TRT_CACHE_ROOT/${variant}_accuracy"
  mkdir -p "$output_dir" "$cache_dir"
  echo "[onnx-nvidia] eval $variant on benchmark_200"
  python scripts/eval_scenic_sft_onnx_local.py \
    --json "$BENCHMARK_JSON" \
    --checkpoint "$checkpoint" \
    --onnx "$onnx_path" \
    --output "$output_dir/benchmark_200_predictions.jsonl" \
    --summary-output "$output_dir/benchmark_200_summary.json" \
    --batch-size "$BATCH_SIZE" \
    --max-length "$MAX_LENGTH" \
    --providers tensorrt \
    --trt-engine-cache-dir "$cache_dir"
}

benchmark_edge_runtime() {
  local runtime_label="$1"
  local runtime="$2"
  local checkpoint="$3"
  local onnx_path="$4"
  local input_length="$5"

  local output="$EDGE_ROOT/${runtime_label}_seq${input_length}.json"
  local trt_cache="$TRT_CACHE_ROOT/${runtime_label}_seq${input_length}"
  local provider_args=()
  if [[ "$runtime" == "onnx" ]]; then
    provider_args+=(--providers "$EDGE_ONNX_PROVIDERS")
  elif [[ "$runtime" == "tensorrt" ]]; then
    provider_args+=(--providers tensorrt --trt-engine-cache-dir "$trt_cache")
  fi

  echo "[onnx-nvidia] benchmark $runtime_label seq=$input_length batch=$EDGE_BATCH_SIZE"
  python scripts/benchmark_scenic_sft_edge_runtime.py \
    --runtime "$runtime" \
    --runtime-label "$runtime_label" \
    --checkpoint "$checkpoint" \
    --onnx "$onnx_path" \
    --json "$BENCHMARK_JSON" \
    --output "$output" \
    --precision fp16 \
    --max-length "$input_length" \
    --batch-size "$EDGE_BATCH_SIZE" \
    --warmup-queries "$EDGE_WARMUP_QUERIES" \
    --measure-queries "$EDGE_MEASURE_QUERIES" \
    --device "$PYTORCH_DEVICE" \
    "${provider_args[@]}"
}

aggregate_results() {
  python - "$EVAL_ROOT" <<'PY'
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

eval_root = Path(sys.argv[1])
variants = [
    ("fp16_dense", "fp16", "dense"),
    ("fp16_nvidia_2_4", "fp16", "nvidia_2_4"),
    ("int8_dense", "int8_dynamic_weight", "dense"),
    ("int8_nvidia_2_4", "int8_dynamic_weight", "nvidia_2_4"),
]
datasets = [
    ("benchmark_200", "benchmark"),
    ("training_retention", "training"),
]

rows = []
for variant, precision, sparsity in variants:
    for dataset_name, dataset_kind in datasets:
        summary_path = eval_root / variant / f"{dataset_name}_summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        checkpoint = summary.get("checkpoint")
        prune_summary_path = Path(checkpoint) / "prune_summary.json" if checkpoint else None
        prune_summary = {}
        if prune_summary_path and prune_summary_path.exists():
            prune_summary = json.loads(prune_summary_path.read_text(encoding="utf-8"))
        em1 = summary.get("exact_match_accuracy")
        em5 = summary.get("top5_accuracy")
        rows.append(
            {
                "variant": variant,
                "precision": precision,
                "sparsity": sparsity,
                "dataset": dataset_name,
                "dataset_kind": dataset_kind,
                "checkpoint": checkpoint,
                "onnx": summary.get("onnx"),
                "providers": "|".join(summary.get("providers", [])),
                "prune_summary_output": str(prune_summary_path) if prune_summary else None,
                "prune_scope": prune_summary.get("scope"),
                "classifier_reinitialized_after_pruning": prune_summary.get("classifier_reinitialized_after_pruning"),
                "prune_targeted_sparsity_after": prune_summary.get("targeted_sparsity_after"),
                "prune_model_sparsity_after": prune_summary.get("model_sparsity_after"),
                "rows": summary.get("rows"),
                "scored_rows": summary.get("scored_rows"),
                "em1": em1,
                "em5": em5,
                "em1_percent": None if em1 is None else em1 * 100.0,
                "em5_percent": None if em5 is None else em5 * 100.0,
                "exact_match_correct": summary.get("exact_match_correct"),
                "top5_correct": summary.get("top5_correct"),
                "label_space_coverage": summary.get("label_space_coverage"),
                "prediction_unique_count": summary.get("prediction_unique_count"),
                "top_prediction": summary.get("top_prediction"),
                "top_prediction_share": summary.get("top_prediction_share"),
                "summary_output": str(summary_path),
                "predictions_output": summary.get("predictions_output"),
            }
        )

payload = {
    "report_type": "scenic_onnx_nvidia_eval",
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "notes": [
        "INT8 variants use ONNX Runtime dynamic weight quantization.",
        "2:4 variants are exported from the NVIDIA-pruned checkpoint; dense variants are exported from the original checkpoint.",
    ],
    "rows": rows,
}

json_path = eval_root / "onnx_nvidia_summary.json"
csv_path = eval_root / "onnx_nvidia_summary.csv"
json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

fieldnames = list(rows[0].keys()) if rows else []
with csv_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"[onnx-nvidia] wrote {json_path}")
print(f"[onnx-nvidia] wrote {csv_path}")
PY
}

aggregate_edge_report() {
  python - "$EVAL_ROOT" "$EDGE_ROOT" <<'PY'
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

eval_root = Path(sys.argv[1])
edge_root = Path(sys.argv[2])

accuracy_sources = {
    "pytorch_fp16_dense": eval_root / "pytorch_fp16_dense" / "benchmark_200_summary.json",
    "pytorch_fp16_nvidia_2_4": eval_root / "pytorch_fp16_nvidia_2_4" / "benchmark_200_summary.json",
    "onnx_fp16_dense": eval_root / "fp16_dense" / "benchmark_200_summary.json",
    "onnx_fp16_nvidia_2_4": eval_root / "fp16_nvidia_2_4" / "benchmark_200_summary.json",
    "tensorrt_fp16_dense": eval_root / "tensorrt_fp16_dense" / "benchmark_200_summary.json",
    "tensorrt_fp16_nvidia_2_4": eval_root / "tensorrt_fp16_nvidia_2_4" / "benchmark_200_summary.json",
}

accuracy = {}
for label, path in accuracy_sources.items():
    if not path.exists():
        continue
    summary = json.loads(path.read_text(encoding="utf-8"))
    accuracy[label] = {
        "benchmark_em1": summary.get("exact_match_accuracy"),
        "benchmark_em5": summary.get("top5_accuracy"),
        "benchmark_em1_percent": (
            None
            if summary.get("exact_match_accuracy") is None
            else summary.get("exact_match_accuracy") * 100.0
        ),
        "benchmark_em5_percent": (
            None
            if summary.get("top5_accuracy") is None
            else summary.get("top5_accuracy") * 100.0
        ),
        "benchmark_summary_output": str(path),
    }

rows = []
for path in sorted(edge_root.glob("*.json")):
    summary = json.loads(path.read_text(encoding="utf-8"))
    label = summary.get("runtime_label")
    if not label:
        continue
    row = {
        "runtime_label": label,
        "runtime": summary.get("runtime"),
        "runtime_display": summary.get("runtime_display"),
        "precision": summary.get("precision"),
        "sparsity": "nvidia_2_4" if "nvidia_2_4" in label else "dense",
        "input_length": summary.get("input_length"),
        "batch_size": summary.get("batch_size"),
        "latency_scope": summary.get("latency_scope"),
        "mean_latency_ms": summary.get("mean_latency_ms"),
        "p95_latency_ms": summary.get("p95_latency_ms"),
        "throughput_qps": summary.get("throughput_qps"),
        "peak_gpu_memory_mb_process": summary.get("peak_gpu_memory_mb_process"),
        "peak_torch_cuda_allocated_mb": summary.get("peak_torch_cuda_allocated_mb"),
        "peak_torch_cuda_reserved_mb": summary.get("peak_torch_cuda_reserved_mb"),
        "peak_cpu_rss_mb": summary.get("peak_cpu_rss_mb"),
        "source_model_size_mb": summary.get("source_model_size_mb"),
        "engine_model_size_mb": summary.get("engine_model_size_mb"),
        "providers": "|".join(summary.get("providers") or []),
        "runtime_summary_output": str(path),
    }
    row.update(
        accuracy.get(
            label,
            {
                "benchmark_em1": None,
                "benchmark_em5": None,
                "benchmark_em1_percent": None,
                "benchmark_em5_percent": None,
                "benchmark_summary_output": None,
            },
        )
    )
    rows.append(row)

payload = {
    "report_type": "scenic_fp16_edge_inference_report",
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "notes": [
        "Latency is model-forward latency on pre-tokenized fixed-length inputs.",
        "Batch size defaults to 1 for interactive smart-home commands.",
        "TensorRT rows use ONNX Runtime TensorrtExecutionProvider when available.",
        "INT8 rows are not part of this FP16 edge report; set RUN_INT8=0 to skip the separate ONNX INT8 accuracy sweep.",
    ],
    "rows": rows,
}

json_path = eval_root / "edge_fp16_report.json"
csv_path = eval_root / "edge_fp16_report.csv"
json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

fieldnames = list(rows[0].keys()) if rows else []
with csv_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"[onnx-nvidia] wrote {json_path}")
print(f"[onnx-nvidia] wrote {csv_path}")
PY
}

require_path "$DENSE_CHECKPOINT" "dense checkpoint"
require_path "$BENCHMARK_JSON" "benchmark json"
require_path "$TRAINING_JSON" "training json"
check_onnx_dependencies

prune_overwrite_args=()
if [[ "$OVERWRITE" == "1" ]]; then
  prune_overwrite_args+=(--overwrite)
fi

reinit_classifier_args=()
if [[ "$REINIT_CLASSIFIER" == "1" ]]; then
  reinit_classifier_args+=(
    --reinitialize-classifier-from-responses
    --classifier-init-batch-size "$CLASSIFIER_INIT_BATCH_SIZE"
    --classifier-init-max-length "$CLASSIFIER_INIT_MAX_LENGTH"
  )
fi

if [[ "$REBUILD_PRUNED" == "1" || ! -d "$PRUNED_CHECKPOINT" ]]; then
  echo "[onnx-nvidia] build NVIDIA 2:4 checkpoint -> $PRUNED_CHECKPOINT"
  python scripts/prune_scenic_sft_reference_methods.py \
    --method nvidia \
    --checkpoint "$DENSE_CHECKPOINT" \
    --output "$PRUNED_CHECKPOINT" \
    --sparsity "$SPARSITY" \
    --scope "$PRUNE_SCOPE" \
    --exclude-classifier \
    --calibration-json "$TRAINING_JSON" \
    --device "$PRUNE_DEVICE" \
    --dtype "$PRUNE_DTYPE" \
    "${prune_overwrite_args[@]}" \
    "${reinit_classifier_args[@]}"
else
  echo "[onnx-nvidia] reuse existing NVIDIA 2:4 checkpoint: $PRUNED_CHECKPOINT"
fi

variant_specs=(
  "fp16_dense|fp16|dense|$DENSE_CHECKPOINT|$ONNX_ROOT/fp16_dense/model.onnx"
  "fp16_nvidia_2_4|fp16|nvidia_2_4|$PRUNED_CHECKPOINT|$ONNX_ROOT/fp16_nvidia_2_4/model.onnx"
)

if [[ "$RUN_INT8" == "1" ]]; then
  variant_specs+=(
    "int8_dense|int8|dense|$DENSE_CHECKPOINT|$ONNX_ROOT/int8_dense/model.onnx"
    "int8_nvidia_2_4|int8|nvidia_2_4|$PRUNED_CHECKPOINT|$ONNX_ROOT/int8_nvidia_2_4/model.onnx"
  )
fi

for spec in "${variant_specs[@]}"; do
  IFS='|' read -r variant precision _sparsity checkpoint onnx_path <<< "$spec"
  export_variant "$variant" "$precision" "$checkpoint" "$onnx_path"
done

for spec in "${variant_specs[@]}"; do
  IFS='|' read -r variant _precision _sparsity checkpoint onnx_path <<< "$spec"
  eval_variant "$variant" "$checkpoint" "$onnx_path" "benchmark_200" "$BENCHMARK_JSON"
  eval_variant "$variant" "$checkpoint" "$onnx_path" "training_retention" "$TRAINING_JSON"
done

aggregate_results

if [[ "$RUN_EDGE_BENCHMARK" == "1" ]]; then
  eval_pytorch_fp16_variant "pytorch_fp16_dense" "$DENSE_CHECKPOINT"
  eval_pytorch_fp16_variant "pytorch_fp16_nvidia_2_4" "$PRUNED_CHECKPOINT"

  run_tensorrt=0
  if [[ "$RUN_TENSORRT" == "1" ]]; then
    run_tensorrt=1
  elif [[ "$RUN_TENSORRT" == "auto" ]] && has_tensorrt_provider; then
    run_tensorrt=1
  fi
  if [[ "$RUN_TENSORRT" == "1" && "$run_tensorrt" != "1" ]]; then
    echo "[onnx-nvidia] RUN_TENSORRT=1 but TensorrtExecutionProvider is unavailable" >&2
    exit 1
  fi

  if [[ "$run_tensorrt" == "1" ]]; then
    eval_tensorrt_fp16_variant "tensorrt_fp16_dense" "$DENSE_CHECKPOINT" "$ONNX_ROOT/fp16_dense/model.onnx"
    eval_tensorrt_fp16_variant "tensorrt_fp16_nvidia_2_4" "$PRUNED_CHECKPOINT" "$ONNX_ROOT/fp16_nvidia_2_4/model.onnx"
  else
    echo "[onnx-nvidia] skipping TensorRT accuracy/latency because TensorrtExecutionProvider is unavailable"
  fi

  input_lengths=()
  IFS=',' read -r -a input_lengths <<< "$EDGE_INPUT_LENGTHS"
  for raw_length in "${input_lengths[@]}"; do
    input_length="${raw_length//[[:space:]]/}"
    if [[ -z "$input_length" ]]; then
      continue
    fi
    benchmark_edge_runtime "pytorch_fp16_dense" "pytorch" "$DENSE_CHECKPOINT" "$ONNX_ROOT/fp16_dense/model.onnx" "$input_length"
    benchmark_edge_runtime "pytorch_fp16_nvidia_2_4" "pytorch" "$PRUNED_CHECKPOINT" "$ONNX_ROOT/fp16_nvidia_2_4/model.onnx" "$input_length"
    benchmark_edge_runtime "onnx_fp16_dense" "onnx" "$DENSE_CHECKPOINT" "$ONNX_ROOT/fp16_dense/model.onnx" "$input_length"
    benchmark_edge_runtime "onnx_fp16_nvidia_2_4" "onnx" "$PRUNED_CHECKPOINT" "$ONNX_ROOT/fp16_nvidia_2_4/model.onnx" "$input_length"
    if [[ "$run_tensorrt" == "1" ]]; then
      benchmark_edge_runtime "tensorrt_fp16_dense" "tensorrt" "$DENSE_CHECKPOINT" "$ONNX_ROOT/fp16_dense/model.onnx" "$input_length"
      benchmark_edge_runtime "tensorrt_fp16_nvidia_2_4" "tensorrt" "$PRUNED_CHECKPOINT" "$ONNX_ROOT/fp16_nvidia_2_4/model.onnx" "$input_length"
    fi
  done

  aggregate_edge_report
  python scripts/render_fp16_deployment_table.py \
    --encoder-report "$EVAL_ROOT/edge_fp16_report.json" \
    --seq-len 64 \
    --output "$EVAL_ROOT/fp16_deployment_table.tex"
fi

echo "[onnx-nvidia] done"
echo "  $EVAL_ROOT/onnx_nvidia_summary.json"
echo "  $EVAL_ROOT/onnx_nvidia_summary.csv"
if [[ "$RUN_EDGE_BENCHMARK" == "1" ]]; then
  echo "  $EVAL_ROOT/edge_fp16_report.json"
  echo "  $EVAL_ROOT/edge_fp16_report.csv"
  echo "  $EVAL_ROOT/fp16_deployment_table.tex"
fi
