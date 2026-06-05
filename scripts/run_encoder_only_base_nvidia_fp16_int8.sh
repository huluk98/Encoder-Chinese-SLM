#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1

BASE_ENCODER="${1:-${BASE_ENCODER:-}}"
if [[ -z "$BASE_ENCODER" ]]; then
  echo "usage: $0 /path/to/encoder-base-checkpoint" >&2
  echo "or: BASE_ENCODER=/path/to/encoder-base-checkpoint $0" >&2
  exit 2
fi

TEMPLATE_CONFIG="${TEMPLATE_CONFIG:-configs/scenic_sft_training_dataset_8gpu.yaml}"
RUN_ROOT="${RUN_ROOT:-runs/encoder-only-nvidia-fp16-int8}"
SFT_OUTPUT="${SFT_OUTPUT:-$RUN_ROOT/scenic-sft-5epoch}"
SFT_CONFIG="${SFT_CONFIG:-$RUN_ROOT/scenic_sft_5epoch.yaml}"
DENSE_CHECKPOINT="${DENSE_CHECKPOINT:-$SFT_OUTPUT/latest}"
PRUNED_CHECKPOINT="${PRUNED_CHECKPOINT:-$RUN_ROOT/scenic-sft-5epoch-nvidia-2-4}"
ONNX_ROOT="${ONNX_ROOT:-$RUN_ROOT/onnx}"
EVAL_ROOT="${EVAL_ROOT:-eval_results/scenic_sft/encoder_only_nvidia_fp16_int8}"

TRAIN_JSON="${TRAIN_JSON:-data/scenic/SCENIC_full_training_dataset.json}"
BENCHMARK_JSON="${BENCHMARK_JSON:-data/scenic/iot_instruction_benchmark_200.json}"

EPOCHS="${EPOCHS:-5}"
MAX_LENGTH="${MAX_LENGTH:-128}"
BATCH_SIZE="${BATCH_SIZE:-128}"
OPSET="${OPSET:-17}"
PROVIDERS="${PROVIDERS:-cuda}"
FP16_EXPORT_DEVICE="${FP16_EXPORT_DEVICE:-cuda}"
RUN_EDGE_BENCHMARK="${RUN_EDGE_BENCHMARK:-1}"
EDGE_ROOT="${EDGE_ROOT:-$EVAL_ROOT/edge_runtime}"
EDGE_INPUT_LENGTH="${EDGE_INPUT_LENGTH:-64}"
EDGE_BATCH_SIZE="${EDGE_BATCH_SIZE:-1}"
EDGE_WARMUP_QUERIES="${EDGE_WARMUP_QUERIES:-20}"
EDGE_MEASURE_QUERIES="${EDGE_MEASURE_QUERIES:-200}"
EDGE_PROVIDERS="${EDGE_PROVIDERS:-$PROVIDERS}"
PARALLEL_GPU_EVAL="${PARALLEL_GPU_EVAL:-1}"
PARALLEL_GPU_BENCHMARK="${PARALLEL_GPU_BENCHMARK:-1}"

TRAIN_WITH_TORCHRUN="${TRAIN_WITH_TORCHRUN:-1}"
NPROC="${NPROC:-8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
GPU_IDS="${GPU_IDS:-$CUDA_VISIBLE_DEVICES}"
RETRAIN="${RETRAIN:-1}"
OVERWRITE="${OVERWRITE:-1}"

PRUNE_SCOPE="${PRUNE_SCOPE:-encoder-linear}"
PRUNE_DEVICE="${PRUNE_DEVICE:-auto}"
PRUNE_DTYPE="${PRUNE_DTYPE:-fp32}"
REINIT_CLASSIFIER="${REINIT_CLASSIFIER:-1}"
CLASSIFIER_INIT_BATCH_SIZE="${CLASSIFIER_INIT_BATCH_SIZE:-128}"
CLASSIFIER_INIT_MAX_LENGTH="${CLASSIFIER_INIT_MAX_LENGTH:-128}"

mkdir -p "$RUN_ROOT" "$ONNX_ROOT" "$EVAL_ROOT" "$EDGE_ROOT"

GPU_ID_LIST=()

parse_gpu_ids() {
  local raw="$1"
  local part
  local IFS=','
  read -ra GPU_ID_LIST <<< "$raw"
  for index in "${!GPU_ID_LIST[@]}"; do
    part="${GPU_ID_LIST[$index]}"
    GPU_ID_LIST[$index]="${part//[[:space:]]/}"
  done
  local compact=()
  for part in "${GPU_ID_LIST[@]}"; do
    if [[ -n "$part" ]]; then
      compact+=("$part")
    fi
  done
  GPU_ID_LIST=("${compact[@]}")
  if [[ "${#GPU_ID_LIST[@]}" -eq 0 ]]; then
    echo "[encoder-only-fp16-int8] GPU_IDS resolved to no GPUs: $raw" >&2
    exit 2
  fi
}

gpu_for_job() {
  local job_index="$1"
  echo "${GPU_ID_LIST[$((job_index % ${#GPU_ID_LIST[@]}))]}"
}

wait_for_jobs() {
  local failed=0
  local pid
  for pid in "$@"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if [[ "$failed" != "0" ]]; then
    echo "[encoder-only-fp16-int8] one or more GPU jobs failed" >&2
    exit 1
  fi
}

require_path() {
  local path="$1"
  local label="$2"
  if [[ ! -e "$path" ]]; then
    echo "[encoder-only-fp16-int8] missing $label: $path" >&2
    exit 1
  fi
}

check_dependencies() {
  python - "$PROVIDERS" "$EDGE_PROVIDERS" "$FP16_EXPORT_DEVICE" "$GPU_IDS" <<'PY'
import importlib
import sys

providers, edge_providers, fp16_export_device, gpu_ids = sys.argv[1:]
required = [("onnx", "onnx"), ("onnxruntime", "onnxruntime")]
missing = []
for module_name, package_name in required:
    try:
        importlib.import_module(module_name)
    except ImportError:
        missing.append(package_name)

if missing:
    print("[encoder-only-fp16-int8] missing: " + ", ".join(missing), file=sys.stderr)
    print("[encoder-only-fp16-int8] install with: pip install " + " ".join(missing), file=sys.stderr)
    sys.exit(1)

import onnxruntime as ort
import torch

available = set(ort.get_available_providers())
for label, requested in (("PROVIDERS", providers), ("EDGE_PROVIDERS", edge_providers)):
    if requested == "cuda" and "CUDAExecutionProvider" not in available:
        print(
            f"[encoder-only-fp16-int8] {label}=cuda but ONNX Runtime has providers: "
            + ", ".join(sorted(available)),
            file=sys.stderr,
        )
        print("[encoder-only-fp16-int8] install/use onnxruntime-gpu for the real GPU run.", file=sys.stderr)
        sys.exit(1)
    if requested == "tensorrt" and "TensorrtExecutionProvider" not in available:
        print(
            f"[encoder-only-fp16-int8] {label}=tensorrt but TensorRT EP is unavailable.",
            file=sys.stderr,
        )
        sys.exit(1)

if fp16_export_device == "cuda" and not torch.cuda.is_available():
    print("[encoder-only-fp16-int8] FP16_EXPORT_DEVICE=cuda but torch.cuda is unavailable.", file=sys.stderr)
    sys.exit(1)

if providers == "cuda" or edge_providers == "cuda" or fp16_export_device == "cuda":
    requested_gpu_count = len([part.strip() for part in gpu_ids.split(",") if part.strip()])
    visible_gpu_count = torch.cuda.device_count()
    if visible_gpu_count <= 0:
        print("[encoder-only-fp16-int8] no CUDA GPUs are visible to PyTorch.", file=sys.stderr)
        sys.exit(1)
    if requested_gpu_count > visible_gpu_count:
        print(
            f"[encoder-only-fp16-int8] GPU_IDS asks for {requested_gpu_count} GPUs, "
            f"but PyTorch sees {visible_gpu_count}.",
            file=sys.stderr,
        )
        sys.exit(1)

print("[encoder-only-fp16-int8] onnxruntime providers: " + ", ".join(ort.get_available_providers()))
if torch.cuda.is_available():
    print(f"[encoder-only-fp16-int8] torch cuda devices visible: {torch.cuda.device_count()}")
PY
}

write_sft_config() {
  python - "$TEMPLATE_CONFIG" "$BASE_ENCODER" "$SFT_OUTPUT" "$SFT_CONFIG" "$EPOCHS" "$TRAIN_JSON" <<'PY'
import os
import sys
from pathlib import Path

import yaml

template_path, base_encoder, output_dir, output_config, epochs, train_json = sys.argv[1:]
with Path(template_path).open("r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle) or {}

config.setdefault("run", {})["output_dir"] = output_dir
config.setdefault("model", {})["base_model"] = base_encoder
config.setdefault("data", {})["train_json"] = train_json
config["data"]["contrastive_json"] = None
config["data"]["max_length"] = int(os.environ.get("MAX_LENGTH", config["data"].get("max_length", 128)))
config.setdefault("train", {})["epochs"] = int(epochs)
config["train"]["max_steps"] = None

tokenizer_path = os.environ.get("TOKENIZER_PATH")
if tokenizer_path:
    config["model"]["tokenizer_path"] = tokenizer_path
else:
    current_tokenizer = str(config["model"].get("tokenizer_path") or "")
    if current_tokenizer and not Path(current_tokenizer).expanduser().exists():
        config["model"]["tokenizer_path"] = base_encoder

Path(output_config).parent.mkdir(parents=True, exist_ok=True)
with Path(output_config).open("w", encoding="utf-8") as handle:
    yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
print(f"[encoder-only-fp16-int8] wrote SFT config: {output_config}")
PY
}

train_sft() {
  if [[ "$RETRAIN" != "1" && -d "$DENSE_CHECKPOINT" ]]; then
    echo "[encoder-only-fp16-int8] reuse SFT checkpoint: $DENSE_CHECKPOINT"
    return
  fi

  echo "[encoder-only-fp16-int8] train 5 epoch SFT from base: $BASE_ENCODER"
  if [[ "$TRAIN_WITH_TORCHRUN" == "1" ]]; then
    CONFIG="$SFT_CONFIG" NPROC="$NPROC" CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" ./scripts/launch_scenic_sft_8gpu.sh
  else
    python scripts/train_scenic_sft.py --config "$SFT_CONFIG"
  fi
}

prune_nvidia() {
  local overwrite_args=()
  if [[ "$OVERWRITE" == "1" ]]; then
    overwrite_args+=(--overwrite)
  fi

  local reinit_args=()
  if [[ "$REINIT_CLASSIFIER" == "1" ]]; then
    reinit_args+=(
      --reinitialize-classifier-from-responses
      --classifier-init-batch-size "$CLASSIFIER_INIT_BATCH_SIZE"
      --classifier-init-max-length "$CLASSIFIER_INIT_MAX_LENGTH"
    )
  fi

  echo "[encoder-only-fp16-int8] NVIDIA 2:4 prune -> $PRUNED_CHECKPOINT"
  python scripts/prune_scenic_sft_reference_methods.py \
    --method nvidia \
    --checkpoint "$DENSE_CHECKPOINT" \
    --output "$PRUNED_CHECKPOINT" \
    --sparsity 0.5 \
    --scope "$PRUNE_SCOPE" \
    --exclude-classifier \
    --calibration-json "$TRAIN_JSON" \
    --device "$PRUNE_DEVICE" \
    --dtype "$PRUNE_DTYPE" \
    "${overwrite_args[@]}" \
    "${reinit_args[@]}"
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
  echo "[encoder-only-fp16-int8] export $variant -> $onnx_path"
  python scripts/export_scenic_sft_onnx.py \
    --checkpoint "$checkpoint" \
    --output "$onnx_path" \
    --precision "$precision" \
    --max-length "$MAX_LENGTH" \
    --opset "$OPSET" \
    --fp16-export-device "$FP16_EXPORT_DEVICE" \
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
  echo "[encoder-only-fp16-int8] eval $variant on $dataset_name"
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

eval_variant_on_gpu() {
  local gpu_id="$1"
  shift
  echo "[encoder-only-fp16-int8] assign GPU $gpu_id for eval $1 on $4"
  (
    export CUDA_VISIBLE_DEVICES="$gpu_id"
    eval_variant "$@"
  )
}

benchmark_variant() {
  local variant="$1"
  local precision="$2"
  local checkpoint="$3"
  local onnx_path="$4"
  local output="$EDGE_ROOT/${variant}_seq${EDGE_INPUT_LENGTH}.json"

  echo "[encoder-only-fp16-int8] benchmark $variant seq=$EDGE_INPUT_LENGTH batch=$EDGE_BATCH_SIZE"
  python scripts/benchmark_scenic_sft_edge_runtime.py \
    --runtime onnx \
    --runtime-label "$variant" \
    --checkpoint "$checkpoint" \
    --onnx "$onnx_path" \
    --json "$BENCHMARK_JSON" \
    --output "$output" \
    --precision "$precision" \
    --max-length "$EDGE_INPUT_LENGTH" \
    --batch-size "$EDGE_BATCH_SIZE" \
    --warmup-queries "$EDGE_WARMUP_QUERIES" \
    --measure-queries "$EDGE_MEASURE_QUERIES" \
    --providers "$EDGE_PROVIDERS"
}

benchmark_variant_on_gpu() {
  local gpu_id="$1"
  shift
  echo "[encoder-only-fp16-int8] assign GPU $gpu_id for benchmark $1"
  (
    export CUDA_VISIBLE_DEVICES="$gpu_id"
    benchmark_variant "$@"
  )
}

run_accuracy_evals() {
  if [[ "$PARALLEL_GPU_EVAL" == "1" ]]; then
    local pids=()
    local job_index=0
    local spec
    for spec in "${variant_specs[@]}"; do
      IFS='|' read -r variant _precision checkpoint onnx_path <<< "$spec"
      gpu_id="$(gpu_for_job "$job_index")"
      eval_variant_on_gpu "$gpu_id" "$variant" "$checkpoint" "$onnx_path" "benchmark_200" "$BENCHMARK_JSON" &
      pids+=("$!")
      job_index=$((job_index + 1))

      gpu_id="$(gpu_for_job "$job_index")"
      eval_variant_on_gpu "$gpu_id" "$variant" "$checkpoint" "$onnx_path" "training_retention" "$TRAIN_JSON" &
      pids+=("$!")
      job_index=$((job_index + 1))
    done
    wait_for_jobs "${pids[@]}"
    return
  fi

  local spec
  for spec in "${variant_specs[@]}"; do
    IFS='|' read -r variant _precision checkpoint onnx_path <<< "$spec"
    eval_variant "$variant" "$checkpoint" "$onnx_path" "benchmark_200" "$BENCHMARK_JSON"
    eval_variant "$variant" "$checkpoint" "$onnx_path" "training_retention" "$TRAIN_JSON"
  done
}

run_edge_benchmarks() {
  if [[ "$PARALLEL_GPU_BENCHMARK" == "1" ]]; then
    local pids=()
    local job_index=0
    local spec
    for spec in "${variant_specs[@]}"; do
      IFS='|' read -r variant precision checkpoint onnx_path <<< "$spec"
      gpu_id="$(gpu_for_job "$job_index")"
      benchmark_variant_on_gpu "$gpu_id" "$variant" "$precision" "$checkpoint" "$onnx_path" &
      pids+=("$!")
      job_index=$((job_index + 1))
    done
    wait_for_jobs "${pids[@]}"
    return
  fi

  local spec
  for spec in "${variant_specs[@]}"; do
    IFS='|' read -r variant precision checkpoint onnx_path <<< "$spec"
    benchmark_variant "$variant" "$precision" "$checkpoint" "$onnx_path"
  done
}

aggregate_table() {
  python - "$EVAL_ROOT" "$EDGE_ROOT" "$EDGE_INPUT_LENGTH" "$EDGE_BATCH_SIZE" <<'PY'
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

eval_root = Path(sys.argv[1])
edge_root = Path(sys.argv[2])
edge_input_length = int(sys.argv[3])
edge_batch_size = int(sys.argv[4])
variant_specs = [
    ("fp16_dense", "FP16", "Dense"),
    ("fp16_nvidia_2_4", "FP16", "NVIDIA 2:4"),
    ("int8_dense", "INT8", "Dense"),
    ("int8_nvidia_2_4", "INT8", "NVIDIA 2:4"),
]

def load_summary(variant: str, dataset: str) -> dict:
    path = eval_root / variant / f"{dataset}_summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))

def load_runtime(variant: str) -> dict:
    path = edge_root / f"{variant}_seq{edge_input_length}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    matches = sorted(edge_root.glob(f"{variant}_seq*.json"))
    if not matches:
        return {}
    return json.loads(matches[-1].read_text(encoding="utf-8"))

def pct(value):
    return None if value is None else float(value) * 100.0

def fmt(value):
    return "--" if value is None else f"{float(value):.2f}"

def mb_from_path(path_value):
    if not path_value:
        return None
    path = Path(str(path_value)).expanduser()
    if not path.exists():
        return None
    if path.is_file():
        size = path.stat().st_size
    else:
        size = sum(child.stat().st_size for child in path.rglob("*") if child.is_file())
    return size / (1024.0 * 1024.0)

def memory_value(runtime):
    for key in ("peak_gpu_memory_mb_process", "peak_torch_cuda_allocated_mb", "peak_cpu_rss_mb"):
        value = runtime.get(key)
        if value is not None:
            return float(value)
    return None

def providers_text(value):
    if isinstance(value, list):
        return "|".join(str(item) for item in value)
    return value

rows = []
for variant, precision, sparsity in variant_specs:
    benchmark = load_summary(variant, "benchmark_200")
    training = load_summary(variant, "training_retention")
    runtime = load_runtime(variant)
    onnx_path = benchmark.get("onnx") or training.get("onnx") or runtime.get("onnx")
    providers = runtime.get("providers") or benchmark.get("providers") or training.get("providers")
    model_size_mb = runtime.get("source_model_size_mb")
    if model_size_mb is None:
        model_size_mb = mb_from_path(onnx_path)
    rows.append(
        {
            "variant": variant,
            "runtime": runtime.get("runtime") or "onnx",
            "runtime_display": runtime.get("runtime_display") or f"ONNX Runtime {precision}",
            "precision": precision,
            "sparsity": sparsity,
            "seq_len": runtime.get("input_length") or edge_input_length,
            "batch_size": runtime.get("batch_size") or edge_batch_size,
            "mean_latency_ms": runtime.get("mean_latency_ms"),
            "p95_latency_ms": runtime.get("p95_latency_ms"),
            "throughput_qps": runtime.get("throughput_qps"),
            "peak_memory_mb": memory_value(runtime),
            "onnx_model_size_mb": model_size_mb,
            "benchmark_em1_percent": pct(benchmark.get("exact_match_accuracy")),
            "benchmark_em5_percent": pct(benchmark.get("top5_accuracy")),
            "training_em1_percent": pct(training.get("exact_match_accuracy")),
            "training_em5_percent": pct(training.get("top5_accuracy")),
            "benchmark_summary": str(eval_root / variant / "benchmark_200_summary.json"),
            "training_summary": str(eval_root / variant / "training_retention_summary.json"),
            "runtime_summary": str(edge_root / f"{variant}_seq{edge_input_length}.json"),
            "onnx": onnx_path,
            "checkpoint": benchmark.get("checkpoint") or training.get("checkpoint"),
            "providers": providers_text(providers),
        }
    )

payload = {
    "report_type": "encoder_only_nvidia_fp16_int8",
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "rows": rows,
}
json_path = eval_root / "encoder_only_fp16_int8_table.json"
csv_path = eval_root / "encoder_only_fp16_int8_table.csv"
tex_path = eval_root / "encoder_only_fp16_int8_table.tex"
json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

with csv_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

lines = [
    r"\begin{table}[t]",
    r"\centering",
    r"\caption{Encoder-only ONNX FP16/INT8 Results After NVIDIA 2:4 Pruning}",
    r"\label{tab:encoder_only_fp16_int8}",
    r"\begin{tabular}{lcccccccc}",
    r"\hline",
    r"Variant & Precision & Seq. Len. & Latency & P95 Lat. & Memory & Model Size & Benchmark EM@1/EM@5 & Training EM@1/EM@5 \\",
    r"\hline",
]
for row in rows:
    label = "Dense" if row["sparsity"] == "Dense" else "NVIDIA 2:4"
    latency = f"{fmt(row['mean_latency_ms'])} ms" if row["mean_latency_ms"] is not None else "--"
    p95 = f"{fmt(row['p95_latency_ms'])} ms" if row["p95_latency_ms"] is not None else "--"
    memory = f"{fmt(row['peak_memory_mb'])} MB" if row["peak_memory_mb"] is not None else "--"
    model_size = f"{fmt(row['onnx_model_size_mb'])} MB" if row["onnx_model_size_mb"] is not None else "--"
    benchmark = f"{fmt(row['benchmark_em1_percent'])}/{fmt(row['benchmark_em5_percent'])}"
    training = f"{fmt(row['training_em1_percent'])}/{fmt(row['training_em5_percent'])}"
    lines.append(
        f"{label} & {row['precision']} & {row['seq_len']} & {latency} & {p95} & "
        f"{memory} & {model_size} & {benchmark} & {training} \\\\"
    )
lines.extend([r"\hline", r"\end{tabular}", r"\end{table}", ""])
tex_path.write_text("\n".join(lines), encoding="utf-8")

print(f"[encoder-only-fp16-int8] wrote {json_path}")
print(f"[encoder-only-fp16-int8] wrote {csv_path}")
print(f"[encoder-only-fp16-int8] wrote {tex_path}")
PY
}

require_path "$BASE_ENCODER" "base encoder checkpoint"
require_path "$TEMPLATE_CONFIG" "template SFT config"
require_path "$TRAIN_JSON" "training json"
require_path "$BENCHMARK_JSON" "benchmark json"
parse_gpu_ids "$GPU_IDS"
check_dependencies
write_sft_config
train_sft
require_path "$DENSE_CHECKPOINT" "5 epoch SFT checkpoint"
prune_nvidia

variant_specs=(
  "fp16_dense|fp16|$DENSE_CHECKPOINT|$ONNX_ROOT/fp16_dense/model.onnx"
  "fp16_nvidia_2_4|fp16|$PRUNED_CHECKPOINT|$ONNX_ROOT/fp16_nvidia_2_4/model.onnx"
  "int8_dense|int8|$DENSE_CHECKPOINT|$ONNX_ROOT/int8_dense/model.onnx"
  "int8_nvidia_2_4|int8|$PRUNED_CHECKPOINT|$ONNX_ROOT/int8_nvidia_2_4/model.onnx"
)

for spec in "${variant_specs[@]}"; do
  IFS='|' read -r variant precision checkpoint onnx_path <<< "$spec"
  export_variant "$variant" "$precision" "$checkpoint" "$onnx_path"
done

run_accuracy_evals

if [[ "$RUN_EDGE_BENCHMARK" == "1" ]]; then
  run_edge_benchmarks
fi

aggregate_table

echo "[encoder-only-fp16-int8] done"
echo "  $EVAL_ROOT/encoder_only_fp16_int8_table.json"
echo "  $EVAL_ROOT/encoder_only_fp16_int8_table.csv"
echo "  $EVAL_ROOT/encoder_only_fp16_int8_table.tex"
