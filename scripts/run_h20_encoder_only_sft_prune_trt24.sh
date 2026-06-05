#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

PYTHON="${PYTHON:-python}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

BASE_MODEL="${BASE_MODEL:-}"
TRAIN_JSONL="${TRAIN_JSONL:-data/scenic/SCENIC_full_training_dataset.json}"
IOT200_JSONL="${IOT200_JSONL:-data/scenic/iot_instruction_benchmark_200.json}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/h20_encoder_only_trt24}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
EPOCHS="${EPOCHS:-5}"
SEQ_LEN="${SEQ_LEN:-64}"
BATCH_SIZE="${BATCH_SIZE:-64}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
MEASURE_ITERS="${MEASURE_ITERS:-1000}"
WARMUP_ITERS="${WARMUP_ITERS:-100}"

TEMPLATE_CONFIG="${TEMPLATE_CONFIG:-configs/scenic_sft_8gpu.yaml}"
TRAIN_PRECISION="${TRAIN_PRECISION:-bf16}"
OPSET="${OPSET:-17}"
RETRAIN="${RETRAIN:-1}"
OVERWRITE="${OVERWRITE:-1}"
REQUIRE_TRT="${REQUIRE_TRT:-1}"
PRUNE_SCOPE="${PRUNE_SCOPE:-encoder-linear}"
PRUNE_EXCLUDE_CLASSIFIER="${PRUNE_EXCLUDE_CLASSIFIER:-1}"
REINIT_CLASSIFIER="${REINIT_CLASSIFIER:-1}"
CLASSIFIER_INIT_BATCH_SIZE="${CLASSIFIER_INIT_BATCH_SIZE:-128}"
CLASSIFIER_INIT_MAX_LENGTH="${CLASSIFIER_INIT_MAX_LENGTH:-$SEQ_LEN}"
ALLOW_DENSE_CLASSIFIER="${ALLOW_DENSE_CLASSIFIER:-1}"
PRUNE_DEVICE="${PRUNE_DEVICE:-cuda}"
PRUNE_DTYPE="${PRUNE_DTYPE:-fp16}"

DENSE_INT8_QDQ_ONNX="${DENSE_INT8_QDQ_ONNX:-}"
SPARSE_INT8_QDQ_ONNX="${SPARSE_INT8_QDQ_ONNX:-}"
DENSE_INT8_CALIBRATION_CACHE="${DENSE_INT8_CALIBRATION_CACHE:-}"
SPARSE_INT8_CALIBRATION_CACHE="${SPARSE_INT8_CALIBRATION_CACHE:-}"

usage() {
  cat <<'EOF'
usage:
  bash scripts/run_h20_encoder_only_sft_prune_trt24.sh --base_model /PATH/TO/BASE_MODEL

Only --base_model is required. Defaults baked into the script:
  --train_jsonl       data/scenic/SCENIC_full_training_dataset.json
  --iot200_jsonl      data/scenic/iot_instruction_benchmark_200.json
  --output_dir        runs/h20_encoder_only_trt24
  --gpus              0,1,2,3,4,5,6,7
  --epochs            5
  --seq_len           64
  --batch_size        64
  --eval_batch_size   128
  --warmup_iters      100
  --measure_iters     1000

2:4 pruning defaults:
  PRUNE_SCOPE=encoder-linear
  REINIT_CLASSIFIER=1
  ALLOW_DENSE_CLASSIFIER=1
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base_model)
      BASE_MODEL="$2"
      shift 2
      ;;
    --base_model=*)
      BASE_MODEL="${1#*=}"
      shift
      ;;
    --train_jsonl)
      TRAIN_JSONL="$2"
      shift 2
      ;;
    --train_jsonl=*)
      TRAIN_JSONL="${1#*=}"
      shift
      ;;
    --iot200_jsonl)
      IOT200_JSONL="$2"
      shift 2
      ;;
    --iot200_jsonl=*)
      IOT200_JSONL="${1#*=}"
      shift
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --output_dir=*)
      OUTPUT_DIR="${1#*=}"
      shift
      ;;
    --gpus)
      GPUS="$2"
      shift 2
      ;;
    --gpus=*)
      GPUS="${1#*=}"
      shift
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --epochs=*)
      EPOCHS="${1#*=}"
      shift
      ;;
    --seq_len)
      SEQ_LEN="$2"
      shift 2
      ;;
    --seq_len=*)
      SEQ_LEN="${1#*=}"
      shift
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --batch_size=*)
      BATCH_SIZE="${1#*=}"
      shift
      ;;
    --eval_batch_size)
      EVAL_BATCH_SIZE="$2"
      shift 2
      ;;
    --eval_batch_size=*)
      EVAL_BATCH_SIZE="${1#*=}"
      shift
      ;;
    --measure_iters)
      MEASURE_ITERS="$2"
      shift 2
      ;;
    --measure_iters=*)
      MEASURE_ITERS="${1#*=}"
      shift
      ;;
    --warmup_iters)
      WARMUP_ITERS="$2"
      shift 2
      ;;
    --warmup_iters=*)
      WARMUP_ITERS="${1#*=}"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      if [[ -z "$BASE_MODEL" && "$1" != --* ]]; then
        BASE_MODEL="$1"
        shift
      else
        echo "[h20-trt24] unknown argument: $1" >&2
        usage >&2
        exit 2
      fi
      ;;
  esac
done

if [[ -z "$BASE_MODEL" ]]; then
  echo "[h20-trt24] missing required --base_model" >&2
  usage >&2
  exit 2
fi

ENV_DIR="$OUTPUT_DIR/env"
WORK_DIR="$OUTPUT_DIR/work"
CHECKPOINT_DIR="$OUTPUT_DIR/checkpoints"
REPORT_DIR="$OUTPUT_DIR/reports"
LOG_DIR="$OUTPUT_DIR/logs"
EVAL_DIR="$OUTPUT_DIR/eval"
ONNX_DIR="$OUTPUT_DIR/onnx"
ENGINE_DIR="$OUTPUT_DIR/engines"
RUNTIME_DIR="$OUTPUT_DIR/runtime"
RESULT_DIR="$OUTPUT_DIR/results"

SFT_CONFIG="$WORK_DIR/scenic_sft_h20_seq${SEQ_LEN}.yaml"
SFT_RUN_DIR="$WORK_DIR/dense_sft_training_run"
DENSE_CHECKPOINT="$CHECKPOINT_DIR/dense_sft_fp16"
DENSE_TRAIN_LATEST="$SFT_RUN_DIR/latest"
PRUNED_WORK_CHECKPOINT="$WORK_DIR/nvidia_2_4_sft_pruned_fp32"
PRUNED_CHECKPOINT="$CHECKPOINT_DIR/nvidia_2_4_sft_fp16"

TRAIN_JSON="$WORK_DIR/data/train.json"
IOT200_JSON="$WORK_DIR/data/iot200.json"
DENSE_ONNX="$ONNX_DIR/dense_sft_fp16/model.onnx"
SPARSE_ONNX="$ONNX_DIR/nvidia_2_4_sft_fp16/model.onnx"
DENSE_ENGINE="$ENGINE_DIR/dense_sft_fp16_seq${SEQ_LEN}.plan"
SPARSE_ENGINE="$ENGINE_DIR/nvidia_2_4_sft_fp16_seq${SEQ_LEN}.plan"

IFS=',' read -r -a GPU_ID_LIST <<< "$GPUS"
COMPACT_GPU_IDS=()
for raw_gpu_id in "${GPU_ID_LIST[@]}"; do
  gpu_id="${raw_gpu_id//[[:space:]]/}"
  if [[ -n "$gpu_id" ]]; then
    COMPACT_GPU_IDS+=("$gpu_id")
  fi
done
GPU_ID_LIST=("${COMPACT_GPU_IDS[@]}")
if [[ "${#GPU_ID_LIST[@]}" -eq 0 ]]; then
  echo "[h20-trt24] --gpus resolved to no devices: $GPUS" >&2
  exit 2
fi
NPROC="${NPROC:-${#GPU_ID_LIST[@]}}"
PRIMARY_GPU="${GPU_ID_LIST[0]}"

log() {
  echo "[h20-trt24] $*"
}

require_path() {
  local path="$1"
  local label="$2"
  if [[ ! -e "$path" ]]; then
    echo "[h20-trt24] missing $label: $path" >&2
    exit 1
  fi
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
    echo "[h20-trt24] one or more background jobs failed" >&2
    exit 1
  fi
}

mkdir -p \
  "$ENV_DIR" \
  "$WORK_DIR/data" \
  "$CHECKPOINT_DIR" \
  "$REPORT_DIR" \
  "$LOG_DIR" \
  "$EVAL_DIR" \
  "$ONNX_DIR" \
  "$ENGINE_DIR" \
  "$RUNTIME_DIR" \
  "$RESULT_DIR"

require_path "$BASE_MODEL" "base model"
require_path "$TEMPLATE_CONFIG" "template SFT config"
require_path "$TRAIN_JSONL" "training data"
require_path "$IOT200_JSONL" "IoT200 benchmark data"

normalize_json_dataset() {
  local input_path="$1"
  local output_path="$2"
  local label="$3"
  "$PYTHON" - "$input_path" "$output_path" "$label" <<'PY'
import json
import sys
from pathlib import Path

input_path = Path(sys.argv[1]).expanduser()
output_path = Path(sys.argv[2]).expanduser()
label = sys.argv[3]
text = input_path.read_text(encoding="utf-8")
stripped = text.lstrip()
if not stripped:
    raise SystemExit(f"{label} is empty: {input_path}")
if stripped[0] == "[":
    rows = json.loads(text)
else:
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
    raise SystemExit(f"{label} must be a JSON list or JSONL of objects: {input_path}")
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"[h20-trt24] normalized {label}: {input_path} -> {output_path} ({len(rows):,} rows)")
PY
}

write_env_report() {
  log "writing environment report"
  CUDA_VISIBLE_DEVICES="$GPUS" "$PYTHON" - "$ENV_DIR/env_report.txt" "$ENV_DIR/env_report.json" <<'PY'
import importlib
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

txt_path = Path(sys.argv[1])
json_path = Path(sys.argv[2])

def run_command(command):
    if shutil.which(command[0]) is None:
        return {"command": command, "found": False, "returncode": None, "stdout": "", "stderr": ""}
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    return {
        "command": command,
        "found": True,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }

def maybe_version(module_name):
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return {"available": False, "error": str(exc), "version": None}
    return {"available": True, "error": None, "version": getattr(module, "__version__", None)}

report = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "platform": platform.platform(),
    "python_version": sys.version,
    "commands": {
        "nvidia_smi": run_command(["nvidia-smi"]),
        "nvcc": run_command(["nvcc", "--version"]),
        "trtexec": run_command(["trtexec", "--version"]),
    },
    "torch": {},
    "onnx": maybe_version("onnx"),
    "onnxruntime": maybe_version("onnxruntime"),
    "tensorrt": maybe_version("tensorrt"),
}

try:
    import torch

    cuda_available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count()) if cuda_available else 0
    report["torch"] = {
        "available": True,
        "version": torch.__version__,
        "torch_version_cuda": torch.version.cuda,
        "cuda_is_available": cuda_available,
        "cuda_device_count": device_count,
        "cuda_devices": [
            {"index": index, "name": torch.cuda.get_device_name(index)}
            for index in range(device_count)
        ],
    }
except Exception as exc:
    report["torch"] = {"available": False, "error": str(exc)}

if report["onnxruntime"]["available"]:
    import onnxruntime as ort

    report["onnxruntime"]["available_providers"] = ort.get_available_providers()
else:
    report["onnxruntime"]["available_providers"] = []

lines = []
lines.append("# H20 Encoder-Only Environment Report")
lines.append(f"generated_at: {report['generated_at']}")
lines.append(f"platform: {report['platform']}")
lines.append("")
for key in ("nvidia_smi", "nvcc", "trtexec"):
    command_report = report["commands"][key]
    lines.append(f"## {' '.join(command_report['command'])}")
    if not command_report["found"]:
        lines.append("not found")
    else:
        lines.append(f"returncode: {command_report['returncode']}")
        lines.append(command_report["stdout"].rstrip() or "(no stdout)")
        if command_report["stderr"]:
            lines.append("stderr:")
            lines.append(command_report["stderr"].rstrip())
    lines.append("")

lines.append("## Python / ML")
lines.append(f"python_version: {report['python_version'].splitlines()[0]}")
lines.append(f"torch_version: {report['torch'].get('version')}")
lines.append(f"torch.version.cuda: {report['torch'].get('torch_version_cuda')}")
lines.append(f"torch.cuda.is_available(): {report['torch'].get('cuda_is_available')}")
lines.append(f"torch.cuda.device_count(): {report['torch'].get('cuda_device_count')}")
for device in report["torch"].get("cuda_devices", []):
    lines.append(f"torch.cuda.get_device_name({device['index']}): {device['name']}")
lines.append(f"TensorRT version: {report['tensorrt'].get('version')}")
lines.append(f"onnx version: {report['onnx'].get('version')}")
lines.append(f"onnxruntime version: {report['onnxruntime'].get('version')}")
lines.append(f"onnxruntime.get_available_providers(): {report['onnxruntime'].get('available_providers')}")
lines.append("")

txt_path.write_text("\n".join(lines), encoding="utf-8")
json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("\n".join(lines))

if not report["torch"].get("cuda_is_available"):
    raise SystemExit("No CUDA GPU is available. This benchmark must run on a CUDA H20 host.")
if int(report["torch"].get("cuda_device_count") or 0) <= 0:
    raise SystemExit("CUDA is available but no CUDA devices are visible.")
PY
}

check_python_dependencies() {
  "$PYTHON" - <<'PY'
import importlib
import sys

required = [
    ("torch", "torch"),
    ("transformers", "transformers"),
    ("yaml", "PyYAML"),
    ("tqdm", "tqdm"),
    ("numpy", "numpy"),
    ("onnx", "onnx"),
    ("onnxruntime", "onnxruntime-gpu"),
]
missing = []
for module_name, package_name in required:
    try:
        importlib.import_module(module_name)
    except Exception:
        missing.append(package_name)
if missing:
    print("[h20-trt24] missing Python packages: " + ", ".join(missing), file=sys.stderr)
    print("[h20-trt24] install with: pip install " + " ".join(missing), file=sys.stderr)
    sys.exit(1)
PY
}

write_sft_config() {
  log "writing SFT config: $SFT_CONFIG"
  "$PYTHON" - "$TEMPLATE_CONFIG" "$BASE_MODEL" "$SFT_RUN_DIR" "$SFT_CONFIG" "$TRAIN_JSON" "$EPOCHS" "$SEQ_LEN" "$BATCH_SIZE" "$TRAIN_PRECISION" <<'PY'
import os
import sys
from pathlib import Path

import yaml

template_path, base_model, output_dir, output_config, train_json, epochs, seq_len, batch_size, precision = sys.argv[1:]
with Path(template_path).open("r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle) or {}

config.setdefault("run", {})["output_dir"] = output_dir
model = config.setdefault("model", {})
model["base_model"] = base_model
tokenizer_path = os.environ.get("TOKENIZER_PATH")
if tokenizer_path:
    model["tokenizer_path"] = tokenizer_path
elif model.get("tokenizer_path") and not Path(str(model["tokenizer_path"])).expanduser().exists():
    model["tokenizer_path"] = base_model

data = config.setdefault("data", {})
data["train_json"] = train_json
data["contrastive_json"] = None
data["max_length"] = int(seq_len)

train = config.setdefault("train", {})
train["epochs"] = int(epochs)
train["max_steps"] = None
train["batch_size"] = int(batch_size)
train["contrastive_batch_size"] = int(batch_size)
train["precision"] = precision
train.setdefault("grad_accum_steps", 1)
train.setdefault("num_workers", 4)
train.setdefault("pin_memory", True)
train.setdefault("persistent_workers", True)

Path(output_config).parent.mkdir(parents=True, exist_ok=True)
with Path(output_config).open("w", encoding="utf-8") as handle:
    yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
PY
}

materialize_fp16_checkpoint() {
  local input_checkpoint="$1"
  local output_checkpoint="$2"
  local label="$3"
  log "materializing $label FP16 checkpoint: $output_checkpoint"
  "$PYTHON" - "$input_checkpoint" "$output_checkpoint" "$label" <<'PY'
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from chatlm_encoder.scenic_sft import load_scenic_checkpoint, save_scenic_checkpoint

input_checkpoint = Path(sys.argv[1]).expanduser()
output_checkpoint = Path(sys.argv[2]).expanduser()
label = sys.argv[3]
metadata_path = input_checkpoint / "scenic_sft_metadata.json"
metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
model, tokenizer, label2response = load_scenic_checkpoint(input_checkpoint, device="cpu")
model.to(device="cpu", dtype=torch.float16)
metadata["materialized_checkpoint"] = {
    "label": label,
    "source_checkpoint": str(input_checkpoint),
    "precision": "fp16",
}
save_scenic_checkpoint(model, tokenizer, output_checkpoint, label2response, metadata)
print(f"[h20-trt24] wrote {output_checkpoint}")
PY
}

train_dense_sft() {
  if [[ "$RETRAIN" != "1" && -d "$DENSE_CHECKPOINT" ]]; then
    log "reusing existing dense checkpoint: $DENSE_CHECKPOINT"
    return
  fi
  if [[ "$OVERWRITE" == "1" ]]; then
    rm -rf "$SFT_RUN_DIR" "$DENSE_CHECKPOINT"
  fi
  write_sft_config
  log "training dense SFT for $EPOCHS epochs on $NPROC GPUs"
  CUDA_VISIBLE_DEVICES="$GPUS" \
    NPROC="$NPROC" \
    CONFIG="$SFT_CONFIG" \
    bash scripts/launch_scenic_sft_8gpu.sh
  require_path "$DENSE_TRAIN_LATEST" "dense training latest checkpoint"
  materialize_fp16_checkpoint "$DENSE_TRAIN_LATEST" "$DENSE_CHECKPOINT" "dense_sft_fp16"
}

split_eval_shards() {
  local dataset_path="$1"
  local shard_dir="$2"
  local shard_count="$3"
  rm -rf "$shard_dir"
  mkdir -p "$shard_dir"
  "$PYTHON" - "$dataset_path" "$shard_dir" "$shard_count" <<'PY'
import json
import math
import sys
from pathlib import Path

dataset_path = Path(sys.argv[1]).expanduser()
shard_dir = Path(sys.argv[2]).expanduser()
shard_count = int(sys.argv[3])
rows = json.loads(dataset_path.read_text(encoding="utf-8"))
if not isinstance(rows, list):
    raise SystemExit(f"{dataset_path} must contain a JSON list after normalization")
chunk_size = max(1, math.ceil(len(rows) / max(1, shard_count)))
manifest = {"dataset": str(dataset_path), "rows": len(rows), "shards": []}
for rank in range(shard_count):
    start = min(len(rows), rank * chunk_size)
    end = min(len(rows), start + chunk_size)
    shard_rows = rows[start:end]
    shard_path = shard_dir / f"shard_{rank:02d}.json"
    shard_path.write_text(json.dumps(shard_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest["shards"].append({"rank": rank, "start": start, "end": end, "rows": len(shard_rows), "path": str(shard_path)})
(shard_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"[h20-trt24] split {dataset_path} into {shard_count} shards under {shard_dir}")
PY
}

merge_eval_shards() {
  local shard_dir="$1"
  local checkpoint="$2"
  local dataset_path="$3"
  local predictions_out="$4"
  local metrics_out="$5"
  "$PYTHON" - "$shard_dir" "$checkpoint" "$dataset_path" "$predictions_out" "$metrics_out" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

shard_dir = Path(sys.argv[1]).expanduser()
checkpoint = sys.argv[2]
dataset_path = sys.argv[3]
predictions_out = Path(sys.argv[4]).expanduser()
metrics_out = Path(sys.argv[5]).expanduser()
manifest = json.loads((shard_dir / "manifest.json").read_text(encoding="utf-8"))
merged = []
for shard in manifest["shards"]:
    prediction_path = shard_dir / f"shard_{int(shard['rank']):02d}_predictions.jsonl"
    if not prediction_path.exists():
        continue
    for line in prediction_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        item["index"] = int(item.get("index", 0)) + int(shard["start"])
        merged.append(item)
merged.sort(key=lambda item: int(item.get("index", -1)))

predictions_out.parent.mkdir(parents=True, exist_ok=True)
with predictions_out.open("w", encoding="utf-8") as handle:
    for item in merged:
        handle.write(json.dumps(item, ensure_ascii=False) + "\n")

total = len(merged)
scored = 0
exact = 0
top5 = 0
expected_in_label_space = 0
predicted_counts = Counter()
expected_counts = Counter()
for item in merged:
    expected = str(item.get("expected_response") or "")
    predicted = str(item.get("predicted_response") or "")
    top5_responses = {str(entry.get("response") or "") for entry in item.get("top5", [])}
    predicted_counts[predicted] += 1
    if expected:
        expected_counts[expected] += 1
        scored += 1
        exact += int(predicted == expected)
        top5 += int(expected in top5_responses)
        expected_in_label_space += int(bool(item.get("expected_in_label_space")))

metrics = {
    "checkpoint": checkpoint,
    "json": dataset_path,
    "predictions_output": str(predictions_out),
    "rows": total,
    "scored_rows": scored,
    "expected_in_label_space": expected_in_label_space,
    "label_space_coverage": expected_in_label_space / scored if scored else None,
    "exact_match_correct": exact,
    "top5_correct": top5,
    "exact_match_accuracy": exact / scored if scored else None,
    "top5_accuracy": top5 / scored if scored else None,
    "prediction_unique_count": len(predicted_counts),
    "prediction_unique_ratio": len(predicted_counts) / total if total else None,
    "top_predictions": [
        {"response": response, "count": count, "share": count / total if total else None}
        for response, count in predicted_counts.most_common(20)
    ],
    "top_expected_responses": [
        {"response": response, "count": count, "share": count / total if total else None}
        for response, count in expected_counts.most_common(20)
    ],
    "sharded": True,
    "shard_count": len(manifest["shards"]),
}
metrics_out.parent.mkdir(parents=True, exist_ok=True)
metrics_out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"[h20-trt24] merged predictions: {predictions_out}")
print(f"[h20-trt24] merged metrics: {metrics_out}")
PY
}

eval_pytorch_sharded() {
  local variant="$1"
  local checkpoint="$2"
  local dataset_name="$3"
  local dataset_path="$4"
  local output_dir="$EVAL_DIR/$variant"
  local shard_dir="$WORK_DIR/eval_shards/$variant/$dataset_name"
  mkdir -p "$output_dir"
  split_eval_shards "$dataset_path" "$shard_dir" "$NPROC"
  log "evaluating $variant on $dataset_name with $NPROC GPU shards"
  local pids=()
  local rank
  for rank in $(seq 0 $((NPROC - 1))); do
    local gpu_index=$((rank % ${#GPU_ID_LIST[@]}))
    local gpu_id="${GPU_ID_LIST[$gpu_index]}"
    (
      export CUDA_VISIBLE_DEVICES="$gpu_id"
      "$PYTHON" scripts/eval_scenic_sft_local.py \
        --json "$shard_dir/shard_$(printf "%02d" "$rank").json" \
        --checkpoint "$checkpoint" \
        --output "$shard_dir/shard_$(printf "%02d" "$rank")_predictions.jsonl" \
        --summary-output "$shard_dir/shard_$(printf "%02d" "$rank")_summary.json" \
        --batch-size "$EVAL_BATCH_SIZE" \
        --max-length "$SEQ_LEN" \
        --dtype fp16
    ) &
    pids+=("$!")
  done
  wait_for_jobs "${pids[@]}"
  merge_eval_shards \
    "$shard_dir" \
    "$checkpoint" \
    "$dataset_path" \
    "$output_dir/${dataset_name}_predictions.jsonl" \
    "$output_dir/${dataset_name}_metrics.json"
}

prune_nvidia_2_4() {
  if [[ "$OVERWRITE" == "1" ]]; then
    rm -rf "$PRUNED_WORK_CHECKPOINT" "$PRUNED_CHECKPOINT"
  fi
  log "applying NVIDIA 2:4 pruning"
  local overwrite_args=()
  if [[ "$OVERWRITE" == "1" ]]; then
    overwrite_args+=(--overwrite)
  fi
  local classifier_args=()
  if [[ "$PRUNE_EXCLUDE_CLASSIFIER" == "1" ]]; then
    classifier_args+=(--exclude-classifier)
  fi
  if [[ "$REINIT_CLASSIFIER" == "1" ]]; then
    classifier_args+=(
      --reinitialize-classifier-from-responses
      --classifier-init-batch-size "$CLASSIFIER_INIT_BATCH_SIZE"
      --classifier-init-max-length "$CLASSIFIER_INIT_MAX_LENGTH"
    )
  fi
  CUDA_VISIBLE_DEVICES="$PRIMARY_GPU" "$PYTHON" scripts/prune_scenic_sft_reference_methods.py \
    --method nvidia \
    --checkpoint "$DENSE_CHECKPOINT" \
    --output "$PRUNED_WORK_CHECKPOINT" \
    --sparsity 0.5 \
    --scope "$PRUNE_SCOPE" \
    --calibration-json "$TRAIN_JSON" \
    --device "$PRUNE_DEVICE" \
    --dtype "$PRUNE_DTYPE" \
    "${classifier_args[@]}" \
    "${overwrite_args[@]}"
  materialize_fp16_checkpoint "$PRUNED_WORK_CHECKPOINT" "$PRUNED_CHECKPOINT" "nvidia_2_4_sft_fp16"
}

verify_2_4_sparsity() {
  log "verifying NVIDIA 2:4 sparsity"
  "$PYTHON" - "$PRUNED_CHECKPOINT" "$REPORT_DIR/sparsity_2_4_report.json" "$ALLOW_DENSE_CLASSIFIER" <<'PY'
import json
import sys
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from chatlm_encoder.scenic_sft import load_scenic_checkpoint

checkpoint = Path(sys.argv[1]).expanduser()
output = Path(sys.argv[2]).expanduser()
allow_dense_classifier = sys.argv[3] == "1"
model, _tokenizer, _labels = load_scenic_checkpoint(checkpoint, device="cpu")
model.eval()

total_checked_weights = 0
total_blocks = 0
exact_2_zero_blocks = 0
at_least_2_zero_blocks = 0
total_zero_count = 0
total_weight_count = 0
per_layer = []
non_compliant_layers = []
ignored_dense_layers = []


def is_classifier_layer(layer_name: str) -> bool:
    normalized = layer_name.lower()
    return normalized == "classifier" or normalized.startswith("classifier.")

for name, module in model.named_modules():
    if not isinstance(module, nn.Linear):
        continue
    weight = module.weight.detach().cpu()
    shape = list(weight.shape)
    ignored_for_2_4_target = bool(allow_dense_classifier and is_classifier_layer(name))
    total_weight_count += int(weight.numel())
    zeros = int((weight == 0).sum().item())
    total_zero_count += zeros
    layer = {
        "name": name,
        "shape": shape,
        "numel": int(weight.numel()),
        "zero_count": zeros,
        "sparsity_pct": zeros / int(weight.numel()) * 100.0 if int(weight.numel()) else None,
        "in_features_divisible_by_4": bool(weight.shape[1] % 4 == 0),
        "included_in_2_4_totals": not ignored_for_2_4_target,
        "ignored_for_2_4_target": ignored_for_2_4_target,
    }
    if weight.shape[1] % 4 != 0:
        layer.update({
            "checked_weights": 0,
            "total_blocks": 0,
            "exact_2_zero_blocks": 0,
            "at_least_2_zero_blocks": 0,
            "exact_2_zero_block_pct": None,
            "tensorrt_eligible_block_pct": None,
            "compliant": False,
            "non_compliance_reason": "in_features is not divisible by 4",
        })
        if ignored_for_2_4_target:
            ignored_dense_layers.append(layer["name"])
        else:
            non_compliant_layers.append(layer["name"])
        per_layer.append(layer)
        continue
    grouped = weight.reshape(weight.shape[0], weight.shape[1] // 4, 4)
    zero_counts = (grouped == 0).sum(dim=2)
    blocks = int(zero_counts.numel())
    exact_blocks = int((zero_counts == 2).sum().item())
    eligible_blocks = int((zero_counts >= 2).sum().item())
    checked_weights = int(grouped.numel())
    compliant = exact_blocks == blocks
    layer.update({
        "checked_weights": checked_weights,
        "total_blocks": blocks,
        "exact_2_zero_blocks": exact_blocks,
        "at_least_2_zero_blocks": eligible_blocks,
        "exact_2_zero_block_pct": exact_blocks / blocks * 100.0 if blocks else None,
        "tensorrt_eligible_block_pct": eligible_blocks / blocks * 100.0 if blocks else None,
        "compliant": compliant,
    })
    if ignored_for_2_4_target:
        ignored_dense_layers.append(layer["name"])
    elif not compliant:
        non_compliant_layers.append(layer["name"])
    if not ignored_for_2_4_target:
        total_checked_weights += checked_weights
        total_blocks += blocks
        exact_2_zero_blocks += exact_blocks
        at_least_2_zero_blocks += eligible_blocks
    per_layer.append(layer)

report = {
    "checkpoint": str(checkpoint),
    "total_checked_weights": total_checked_weights,
    "total_blocks": total_blocks,
    "exact_2_zero_blocks": exact_2_zero_blocks,
    "at_least_2_zero_blocks": at_least_2_zero_blocks,
    "exact_2_zero_block_pct": exact_2_zero_blocks / total_blocks * 100.0 if total_blocks else None,
    "tensorrt_eligible_block_pct": at_least_2_zero_blocks / total_blocks * 100.0 if total_blocks else None,
    "total_zero_count": total_zero_count,
    "total_sparsity_pct": total_zero_count / total_weight_count * 100.0 if total_weight_count else None,
    "per_layer": per_layer,
    "non_compliant_layers": non_compliant_layers,
    "ignored_dense_layers": ignored_dense_layers,
    "allow_dense_classifier": allow_dense_classifier,
}
output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"[h20-trt24] wrote {output}")
if non_compliant_layers:
    raise SystemExit(f"2:4 sparsity verification found non-compliant Linear layers: {non_compliant_layers[:10]}")
PY
}

export_onnx_fp16() {
  local checkpoint="$1"
  local output_path="$2"
  log "exporting ONNX: $output_path"
  mkdir -p "$(dirname "$output_path")"
  CUDA_VISIBLE_DEVICES="$PRIMARY_GPU" "$PYTHON" scripts/export_scenic_sft_onnx.py \
    --checkpoint "$checkpoint" \
    --output "$output_path" \
    --precision fp16 \
    --max-length "$SEQ_LEN" \
    --opset "$OPSET" \
    --fp16-export-device cuda \
    --overwrite
}

inspect_onnx() {
  local onnx_path="$1"
  local output_path="$2"
  log "inspecting ONNX: $onnx_path"
  "$PYTHON" - "$onnx_path" "$output_path" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

import onnx
from onnx import TensorProto

onnx_path = Path(sys.argv[1]).expanduser()
output_path = Path(sys.argv[2]).expanduser()
model = onnx.load(str(onnx_path))
onnx.checker.check_model(model)
initializer_names = {initializer.name for initializer in model.graph.initializer}

def shape_for(value_info):
    dims = []
    tensor_type = value_info.type.tensor_type
    for dim in tensor_type.shape.dim:
        if dim.dim_param:
            dims.append(dim.dim_param)
        elif dim.dim_value:
            dims.append(dim.dim_value)
        else:
            dims.append(None)
    return dims

def dtype_for(value_info):
    elem_type = value_info.type.tensor_type.elem_type
    return TensorProto.DataType.Name(elem_type)

inputs = [
    {"name": item.name, "shape": shape_for(item), "dtype": dtype_for(item)}
    for item in model.graph.input
    if item.name not in initializer_names
]
outputs = [
    {"name": item.name, "shape": shape_for(item), "dtype": dtype_for(item)}
    for item in model.graph.output
]
initializer_dtypes = Counter(TensorProto.DataType.Name(item.data_type) for item in model.graph.initializer)
weight_initializers = [
    item for item in model.graph.initializer
    if item.data_type in {TensorProto.FLOAT, TensorProto.FLOAT16, TensorProto.BFLOAT16, TensorProto.DOUBLE}
]
fp16_weight_count = sum(1 for item in weight_initializers if item.data_type == TensorProto.FLOAT16)
dynamic_axes = {
    item["name"]: {str(index): dim for index, dim in enumerate(item["shape"]) if isinstance(dim, str)}
    for item in inputs + outputs
    if any(isinstance(dim, str) for dim in item["shape"])
}
report = {
    "onnx_path": str(onnx_path),
    "exists": onnx_path.exists(),
    "input_names": [item["name"] for item in inputs],
    "output_names": [item["name"] for item in outputs],
    "inputs": inputs,
    "outputs": outputs,
    "input_shapes": {item["name"]: item["shape"] for item in inputs},
    "output_shapes": {item["name"]: item["shape"] for item in outputs},
    "dynamic_axes": dynamic_axes,
    "initializer_dtypes": dict(initializer_dtypes),
    "weight_initializers": len(weight_initializers),
    "fp16_weight_initializers": fp16_weight_count,
    "weights_are_fp16": bool(weight_initializers) and fp16_weight_count == len(weight_initializers),
    "model_size_mb": onnx_path.stat().st_size / (1024.0 * 1024.0),
    "onnx_checker_passed": True,
}
output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"[h20-trt24] wrote {output_path}")
PY
}

shape_spec_for_onnx() {
  local onnx_path="$1"
  "$PYTHON" - "$onnx_path" "$SEQ_LEN" <<'PY'
import sys
from pathlib import Path

import onnx

onnx_path = Path(sys.argv[1]).expanduser()
seq_len = int(sys.argv[2])
model = onnx.load(str(onnx_path))
initializer_names = {initializer.name for initializer in model.graph.initializer}
specs = []
for item in model.graph.input:
    if item.name in initializer_names:
        continue
    dims = item.type.tensor_type.shape.dim
    if len(dims) == 2:
        specs.append(f"{item.name}:1x{seq_len}")
    else:
        concrete = []
        for dim in dims:
            concrete.append(str(dim.dim_value if dim.dim_value else 1))
        specs.append(f"{item.name}:{'x'.join(concrete)}")
print(",".join(specs))
PY
}

require_trtexec() {
  if command -v trtexec >/dev/null 2>&1; then
    return
  fi
  "$PYTHON" - "$REPORT_DIR/tensorrt_status.json" <<'PY'
import json
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({
    "trtexec_available": False,
    "status": "missing",
    "message": "trtexec was not found in PATH. Install TensorRT or use an NVIDIA TensorRT container.",
}, indent=2) + "\n", encoding="utf-8")
PY
  if [[ "$REQUIRE_TRT" == "1" ]]; then
    echo "[h20-trt24] trtexec was not found in PATH. TensorRT engine builds require native TensorRT." >&2
    exit 1
  fi
}

build_trt_engine() {
  local label="$1"
  local onnx_path="$2"
  local engine_path="$3"
  local sparsity_mode="$4"
  local log_path="$5"
  local shapes
  shapes="$(shape_spec_for_onnx "$onnx_path")"
  log "building TensorRT engine: $label"
  mkdir -p "$(dirname "$engine_path")" "$(dirname "$log_path")"
  local args=(
    --onnx="$onnx_path"
    --saveEngine="$engine_path"
    --fp16
    --sparsity="$sparsity_mode"
    --minShapes="$shapes"
    --optShapes="$shapes"
    --maxShapes="$shapes"
    --buildOnly
  )
  if [[ "$sparsity_mode" == "enable" ]]; then
    args+=(--verbose --profilingVerbosity=detailed)
  fi
  CUDA_VISIBLE_DEVICES="$PRIMARY_GPU" trtexec "${args[@]}" 2>&1 | tee "$log_path"
}

parse_sparse_tactics_log() {
  "$PYTHON" - "$LOG_DIR/build_nvidia_2_4_sparse_fp16.log" "$REPORT_DIR/trt_sparse_tactics_report.json" <<'PY'
import json
import re
import sys
from pathlib import Path

log_path = Path(sys.argv[1]).expanduser()
output_path = Path(sys.argv[2]).expanduser()
text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
lines = [line for line in text.splitlines() if re.search(r"spars|tactic|Sparsity|Chose", line, re.IGNORECASE)]
eligible = bool(re.search(r"eligible to use sparse tactics|sparse.*eligible", text, re.IGNORECASE))
selected = bool(re.search(r"Chose.*sparse|sparse tactics selected|selected.*sparse tactic|using sparse tactic", text, re.IGNORECASE))
report = {
    "log_path": str(log_path),
    "sparse_tactics_eligible": eligible,
    "sparse_tactics_selected": selected,
    "matched_lines": lines[:200],
}
output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"[h20-trt24] wrote {output_path}")
PY
}

parse_trtexec_benchmark() {
  local label="$1"
  local engine_path="$2"
  local log_path="$3"
  local output_path="$4"
  local precision="$5"
  local sparsity="$6"
  "$PYTHON" - "$label" "$engine_path" "$log_path" "$output_path" "$precision" "$sparsity" "$SEQ_LEN" <<'PY'
import json
import re
import sys
from pathlib import Path

label, engine_path, log_path, output_path, precision, sparsity, seq_len = sys.argv[1:]
engine_path = Path(engine_path).expanduser()
log_path = Path(log_path).expanduser()
output_path = Path(output_path).expanduser()
text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""

def find_float(patterns):
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None

mean = find_float([
    r"GPU Compute Time:.*?mean\s*=\s*([0-9.]+)\s*ms",
    r"mean\s*=\s*([0-9.]+)\s*ms",
])
median = find_float([
    r"GPU Compute Time:.*?median\s*=\s*([0-9.]+)\s*ms",
    r"median\s*=\s*([0-9.]+)\s*ms",
])
p95 = find_float([
    r"percentile\(95%\)\s*=\s*([0-9.]+)\s*ms",
    r"p95\s*[:=]\s*([0-9.]+)",
])
p99 = find_float([
    r"percentile\(99%\)\s*=\s*([0-9.]+)\s*ms",
    r"p99\s*[:=]\s*([0-9.]+)",
])
throughput = find_float([r"Throughput:\s*([0-9.]+)\s*qps", r"throughput\s*[:=]\s*([0-9.]+)"])
peak_memory = find_float([r"Total Host Persistent Memory:\s*([0-9.]+)", r"Device Persistent Memory:\s*([0-9.]+)"])
payload = {
    "runtime_label": label,
    "runtime": "native_tensorrt",
    "runtime_display": "TensorRT FP16",
    "precision": precision,
    "sparsity": sparsity,
    "input_length": int(seq_len),
    "batch_size": 1,
    "mean_latency_ms": mean,
    "median_latency_ms": median,
    "p95_latency_ms": p95,
    "p99_latency_ms": p99,
    "throughput_qps": throughput,
    "peak_gpu_memory_mb_process": peak_memory,
    "engine_model_size_mb": engine_path.stat().st_size / (1024.0 * 1024.0) if engine_path.exists() else None,
    "engine": str(engine_path),
    "log": str(log_path),
}
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"[h20-trt24] wrote {output_path}")
PY
}

benchmark_trt_engine() {
  local label="$1"
  local engine_path="$2"
  local onnx_path="$3"
  local precision="$4"
  local sparsity="$5"
  local log_path="$LOG_DIR/benchmark_${label}.log"
  local output_path="$RUNTIME_DIR/${label}.json"
  local shapes
  shapes="$(shape_spec_for_onnx "$onnx_path")"
  log "benchmarking TensorRT engine: $label"
  CUDA_VISIBLE_DEVICES="$PRIMARY_GPU" trtexec \
    --loadEngine="$engine_path" \
    --shapes="$shapes" \
    --warmUp="$WARMUP_ITERS" \
    --iterations="$MEASURE_ITERS" \
    --useCudaGraph \
    2>&1 | tee "$log_path"
  parse_trtexec_benchmark "$label" "$engine_path" "$log_path" "$output_path" "$precision" "$sparsity"
}

benchmark_pytorch() {
  local label="$1"
  local checkpoint="$2"
  local output_path="$3"
  log "benchmarking PyTorch FP16: $label"
  CUDA_VISIBLE_DEVICES="$PRIMARY_GPU" "$PYTHON" scripts/benchmark_scenic_sft_edge_runtime.py \
    --runtime pytorch \
    --runtime-label "$label" \
    --checkpoint "$checkpoint" \
    --json "$IOT200_JSON" \
    --output "$output_path" \
    --precision fp16 \
    --max-length "$SEQ_LEN" \
    --batch-size 1 \
    --warmup-queries "$WARMUP_ITERS" \
    --measure-queries "$MEASURE_ITERS" \
    --device cuda
}

ort_provider_available() {
  local provider="$1"
  "$PYTHON" - "$provider" <<'PY'
import sys
import onnxruntime as ort

provider = sys.argv[1]
raise SystemExit(0 if provider in ort.get_available_providers() else 1)
PY
}

write_skip_report() {
  local output_path="$1"
  local reason="$2"
  "$PYTHON" - "$output_path" "$reason" <<'PY'
import json
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({"status": "skipped", "reason": sys.argv[2]}, indent=2) + "\n", encoding="utf-8")
PY
}

benchmark_onnx_cuda_if_available() {
  if ort_provider_available "CUDAExecutionProvider"; then
    log "benchmarking ONNX Runtime CUDA: dense_sft_fp16"
    CUDA_VISIBLE_DEVICES="$PRIMARY_GPU" "$PYTHON" scripts/benchmark_scenic_sft_edge_runtime.py \
      --runtime onnx \
      --runtime-label dense_sft_fp16_onnx_cuda \
      --checkpoint "$DENSE_CHECKPOINT" \
      --onnx "$DENSE_ONNX" \
      --json "$IOT200_JSON" \
      --output "$RUNTIME_DIR/dense_sft_fp16_onnx_cuda.json" \
      --precision fp16 \
      --max-length "$SEQ_LEN" \
      --batch-size 1 \
      --warmup-queries "$WARMUP_ITERS" \
      --measure-queries "$MEASURE_ITERS" \
      --providers cuda
  else
    log "skipping ONNX Runtime CUDA benchmark because CUDAExecutionProvider is unavailable"
    write_skip_report "$RUNTIME_DIR/dense_sft_fp16_onnx_cuda.json" "CUDAExecutionProvider is unavailable; CPU ONNX is not used for GPU speedup claims."
  fi
}

benchmark_ort_tensorrt_if_available() {
  if ort_provider_available "TensorrtExecutionProvider"; then
    log "benchmarking ONNX Runtime TensorRT EP: dense_sft_fp16"
    CUDA_VISIBLE_DEVICES="$PRIMARY_GPU" "$PYTHON" scripts/benchmark_scenic_sft_edge_runtime.py \
      --runtime tensorrt \
      --runtime-label dense_sft_fp16_ort_tensorrt \
      --checkpoint "$DENSE_CHECKPOINT" \
      --onnx "$DENSE_ONNX" \
      --json "$IOT200_JSON" \
      --output "$RUNTIME_DIR/dense_sft_fp16_ort_tensorrt.json" \
      --precision fp16 \
      --max-length "$SEQ_LEN" \
      --batch-size 1 \
      --warmup-queries "$WARMUP_ITERS" \
      --measure-queries "$MEASURE_ITERS" \
      --providers tensorrt \
      --trt-engine-cache-dir "$WORK_DIR/ort_trt_cache/dense_sft_fp16"
  else
    log "skipping ONNX Runtime TensorRT EP benchmark because TensorrtExecutionProvider is unavailable"
    write_skip_report "$RUNTIME_DIR/dense_sft_fp16_ort_tensorrt.json" "TensorrtExecutionProvider is unavailable; native trtexec is used for TensorRT."
  fi
}

eval_ort_tensorrt_accuracy_if_available() {
  if ! ort_provider_available "TensorrtExecutionProvider"; then
    write_skip_report "$REPORT_DIR/tensorrt_accuracy_status.json" "TensorrtExecutionProvider is unavailable; native TensorRT .plan accuracy evaluation is not implemented in this repo."
    return
  fi
  log "evaluating ONNX Runtime TensorRT EP accuracy on IoT200"
  mkdir -p "$EVAL_DIR/dense_sft_fp16_trt" "$EVAL_DIR/nvidia_2_4_sft_fp16_trt"
  CUDA_VISIBLE_DEVICES="$PRIMARY_GPU" "$PYTHON" scripts/eval_scenic_sft_onnx_local.py \
    --json "$IOT200_JSON" \
    --checkpoint "$DENSE_CHECKPOINT" \
    --onnx "$DENSE_ONNX" \
    --output "$EVAL_DIR/dense_sft_fp16_trt/iot200_predictions.jsonl" \
    --summary-output "$EVAL_DIR/dense_sft_fp16_trt/iot200_metrics.json" \
    --batch-size "$EVAL_BATCH_SIZE" \
    --max-length "$SEQ_LEN" \
    --providers tensorrt \
    --trt-engine-cache-dir "$WORK_DIR/ort_trt_accuracy/dense_sft_fp16"
  CUDA_VISIBLE_DEVICES="$PRIMARY_GPU" "$PYTHON" scripts/eval_scenic_sft_onnx_local.py \
    --json "$IOT200_JSON" \
    --checkpoint "$PRUNED_CHECKPOINT" \
    --onnx "$SPARSE_ONNX" \
    --output "$EVAL_DIR/nvidia_2_4_sft_fp16_trt/iot200_predictions.jsonl" \
    --summary-output "$EVAL_DIR/nvidia_2_4_sft_fp16_trt/iot200_metrics.json" \
    --batch-size "$EVAL_BATCH_SIZE" \
    --max-length "$SEQ_LEN" \
    --providers tensorrt \
    --trt-engine-cache-dir "$WORK_DIR/ort_trt_accuracy/nvidia_2_4_sft_fp16"
}

handle_int8_optional() {
  local status_path="$REPORT_DIR/int8_status.json"
  if [[ -n "$DENSE_INT8_QDQ_ONNX" && -n "$SPARSE_INT8_QDQ_ONNX" ]]; then
    require_path "$DENSE_INT8_QDQ_ONNX" "dense INT8 Q/DQ ONNX"
    require_path "$SPARSE_INT8_QDQ_ONNX" "sparse INT8 Q/DQ ONNX"
    log "INT8 Q/DQ ONNX inputs were provided; building INT8 engines"
    build_trt_int8_engine "dense_sft_int8" "$DENSE_INT8_QDQ_ONNX" "$ENGINE_DIR/dense_sft_int8_seq${SEQ_LEN}.plan" "$LOG_DIR/build_dense_int8.log" "" "disable"
    build_trt_int8_engine "nvidia_2_4_sft_int8" "$SPARSE_INT8_QDQ_ONNX" "$ENGINE_DIR/nvidia_2_4_sft_int8_seq${SEQ_LEN}.plan" "$LOG_DIR/build_nvidia_2_4_int8.log" "" "enable"
    benchmark_trt_engine "dense_sft_int8_trt" "$ENGINE_DIR/dense_sft_int8_seq${SEQ_LEN}.plan" "$DENSE_INT8_QDQ_ONNX" "INT8" "Dense"
    benchmark_trt_engine "nvidia_2_4_sft_int8_trt" "$ENGINE_DIR/nvidia_2_4_sft_int8_seq${SEQ_LEN}.plan" "$SPARSE_INT8_QDQ_ONNX" "INT8" "NVIDIA 2:4"
    "$PYTHON" - "$status_path" "$DENSE_INT8_QDQ_ONNX" "$SPARSE_INT8_QDQ_ONNX" <<'PY'
import json
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({
    "status": "built_from_qdq_onnx",
    "dense_int8_qdq_onnx": sys.argv[2],
    "sparse_int8_qdq_onnx": sys.argv[3],
}, indent=2) + "\n", encoding="utf-8")
PY
  elif [[ -n "$DENSE_INT8_CALIBRATION_CACHE" && -n "$SPARSE_INT8_CALIBRATION_CACHE" ]]; then
    require_path "$DENSE_INT8_CALIBRATION_CACHE" "dense INT8 calibration cache"
    require_path "$SPARSE_INT8_CALIBRATION_CACHE" "sparse INT8 calibration cache"
    log "INT8 calibration caches were provided; building INT8 engines"
    build_trt_int8_engine "dense_sft_int8" "$DENSE_ONNX" "$ENGINE_DIR/dense_sft_int8_seq${SEQ_LEN}.plan" "$LOG_DIR/build_dense_int8.log" "$DENSE_INT8_CALIBRATION_CACHE" "disable"
    build_trt_int8_engine "nvidia_2_4_sft_int8" "$SPARSE_ONNX" "$ENGINE_DIR/nvidia_2_4_sft_int8_seq${SEQ_LEN}.plan" "$LOG_DIR/build_nvidia_2_4_int8.log" "$SPARSE_INT8_CALIBRATION_CACHE" "enable"
    benchmark_trt_engine "dense_sft_int8_trt" "$ENGINE_DIR/dense_sft_int8_seq${SEQ_LEN}.plan" "$DENSE_ONNX" "INT8" "Dense"
    benchmark_trt_engine "nvidia_2_4_sft_int8_trt" "$ENGINE_DIR/nvidia_2_4_sft_int8_seq${SEQ_LEN}.plan" "$SPARSE_ONNX" "INT8" "NVIDIA 2:4"
    "$PYTHON" - "$status_path" "$DENSE_INT8_CALIBRATION_CACHE" "$SPARSE_INT8_CALIBRATION_CACHE" <<'PY'
import json
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({
    "status": "built_from_calibration_cache",
    "dense_int8_calibration_cache": sys.argv[2],
    "sparse_int8_calibration_cache": sys.argv[3],
}, indent=2) + "\n", encoding="utf-8")
PY
  else
    log "skipping INT8 because no Q/DQ ONNX or real calibration cache was provided"
    "$PYTHON" - "$status_path" <<'PY'
import json
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({
    "status": "skipped",
    "reason": "No Q/DQ ONNX export or real TensorRT calibration cache was provided. Dynamic/random INT8 is not used for EM claims.",
}, indent=2) + "\n", encoding="utf-8")
PY
  fi
}

build_trt_int8_engine() {
  local label="$1"
  local onnx_path="$2"
  local engine_path="$3"
  local log_path="$4"
  local calibration_cache="$5"
  local sparsity_mode="${6:-disable}"
  local shapes
  shapes="$(shape_spec_for_onnx "$onnx_path")"
  local args=(
    --onnx="$onnx_path"
    --saveEngine="$engine_path"
    --int8
    --sparsity="$sparsity_mode"
    --minShapes="$shapes"
    --optShapes="$shapes"
    --maxShapes="$shapes"
    --buildOnly
  )
  if [[ -n "$calibration_cache" ]]; then
    args+=(--calib="$calibration_cache")
  fi
  CUDA_VISIBLE_DEVICES="$PRIMARY_GPU" trtexec "${args[@]}" 2>&1 | tee "$log_path"
  log "built INT8 engine: $label -> $engine_path"
}

aggregate_final_outputs() {
  log "aggregating final metrics and summary"
  "$PYTHON" - \
    "$OUTPUT_DIR" \
    "$SEQ_LEN" \
    "$ENV_DIR/env_report.json" \
    "$REPORT_DIR/trt_sparse_tactics_report.json" \
    "$REPORT_DIR/int8_status.json" \
    "$REPORT_DIR/sparsity_2_4_report.json" \
    "$REPORT_DIR/onnx_inspection_dense_sft_fp16.json" \
    "$REPORT_DIR/onnx_inspection_nvidia_2_4_sft_fp16.json" <<'PY'
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

output_dir = Path(sys.argv[1]).expanduser()
seq_len = int(sys.argv[2])
env_path = Path(sys.argv[3])
sparse_tactics_path = Path(sys.argv[4])
int8_status_path = Path(sys.argv[5])
sparsity_report_path = Path(sys.argv[6])
dense_onnx_report_path = Path(sys.argv[7])
sparse_onnx_report_path = Path(sys.argv[8])

def load_json(path):
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))

def metric(variant, dataset):
    return load_json(output_dir / "eval" / variant / f"{dataset}_metrics.json")

def runtime(name):
    return load_json(output_dir / "runtime" / f"{name}.json")

def get_latency(row):
    return row.get("mean_latency_ms")

env = load_json(env_path)
sparse_tactics = load_json(sparse_tactics_path)
int8_status = load_json(int8_status_path)
sparsity_report = load_json(sparsity_report_path)
dense_onnx_report = load_json(dense_onnx_report_path)
sparse_onnx_report = load_json(sparse_onnx_report_path)

gpu_name = None
devices = env.get("torch", {}).get("cuda_devices") or []
if devices:
    gpu_name = devices[0].get("name")
ort_providers = env.get("onnxruntime", {}).get("available_providers") or []
cpu_onnx_only = bool(ort_providers) and set(ort_providers) == {"CPUExecutionProvider"}

dense_iot = metric("dense_sft_fp16", "iot200")
dense_train = metric("dense_sft_fp16", "train")
sparse_iot = metric("nvidia_2_4_sft_fp16", "iot200")
sparse_train = metric("nvidia_2_4_sft_fp16", "train")
dense_trt_iot = load_json(output_dir / "eval" / "dense_sft_fp16_trt" / "iot200_metrics.json")
sparse_trt_iot = load_json(output_dir / "eval" / "nvidia_2_4_sft_fp16_trt" / "iot200_metrics.json")

dense_trt = runtime("dense_sft_fp16_trt")
sparse_trt = runtime("nvidia_2_4_sft_fp16_trt")
dense_trt_latency = get_latency(dense_trt)
sparse_trt_latency = get_latency(sparse_trt)
main_speedup = dense_trt_latency / sparse_trt_latency if dense_trt_latency and sparse_trt_latency else None
dense_trt_qps = dense_trt.get("throughput_qps")
sparse_trt_qps = sparse_trt.get("throughput_qps")
throughput_gain = sparse_trt_qps / dense_trt_qps if dense_trt_qps and sparse_trt_qps else None

def em(metrics, key):
    value = metrics.get(key)
    return None if value is None else float(value)

def mb(report):
    return report.get("model_size_mb")

def memory(summary):
    for key in ("peak_gpu_memory_mb_process", "peak_torch_cuda_allocated_mb", "peak_cpu_rss_mb"):
        value = summary.get(key)
        if value is not None:
            return value
    return None

def has_measured_runtime(summary):
    return bool(summary) and summary.get("status") != "skipped" and (
        summary.get("mean_latency_ms") is not None
        or summary.get("throughput_qps") is not None
        or summary.get("engine_model_size_mb") is not None
    )

rows = []

def add_row(
    *,
    model,
    runtime_label,
    precision,
    sparsity,
    provider,
    runtime_summary,
    iot_metrics,
    train_metrics,
    onnx_mb=None,
    engine_mb=None,
    sparse_selected=None,
    speedup=None,
):
    rows.append({
        "Model": model,
        "Architecture": "Encoder-only",
        "Runtime": runtime_label,
        "Precision": precision,
        "Sparsity": sparsity,
        "Seq. Len.": seq_len,
        "Batch Size": runtime_summary.get("batch_size", 1) if runtime_summary else 1,
        "Latency": runtime_summary.get("mean_latency_ms") if runtime_summary else None,
        "Median Lat.": runtime_summary.get("median_latency_ms") if runtime_summary else None,
        "P95 Lat.": runtime_summary.get("p95_latency_ms") if runtime_summary else None,
        "P99 Lat.": runtime_summary.get("p99_latency_ms") if runtime_summary else None,
        "Throughput QPS": runtime_summary.get("throughput_qps") if runtime_summary else None,
        "Memory": memory(runtime_summary) if runtime_summary else None,
        "ONNX MB": onnx_mb,
        "Engine MB": engine_mb if engine_mb is not None else (runtime_summary.get("engine_model_size_mb") if runtime_summary else None),
        "EM@1 IoT200": em(iot_metrics, "exact_match_accuracy"),
        "EM@5 IoT200": em(iot_metrics, "top5_accuracy"),
        "EM@1 Train": em(train_metrics, "exact_match_accuracy"),
        "EM@5 Train": em(train_metrics, "top5_accuracy"),
        "GPU": gpu_name,
        "Provider": provider,
        "Sparse Tactics Selected": sparse_selected,
        "Speedup vs Dense TRT FP16": speedup,
    })

add_row(
    model="Dense SFT",
    runtime_label="PyTorch",
    precision="FP16",
    sparsity="Dense",
    provider="torch.cuda",
    runtime_summary=runtime("dense_sft_fp16_pytorch"),
    iot_metrics=dense_iot,
    train_metrics=dense_train,
    onnx_mb=mb(dense_onnx_report),
)
add_row(
    model="Dense SFT",
    runtime_label="ONNX Runtime CUDA",
    precision="FP16",
    sparsity="Dense",
    provider="CUDAExecutionProvider" if "CUDAExecutionProvider" in ort_providers else "unavailable",
    runtime_summary=runtime("dense_sft_fp16_onnx_cuda"),
    iot_metrics=dense_iot,
    train_metrics=dense_train,
    onnx_mb=mb(dense_onnx_report),
)
add_row(
    model="Dense SFT",
    runtime_label="ONNX Runtime TensorRT EP",
    precision="FP16",
    sparsity="Dense",
    provider="TensorrtExecutionProvider" if "TensorrtExecutionProvider" in ort_providers else "unavailable",
    runtime_summary=runtime("dense_sft_fp16_ort_tensorrt"),
    iot_metrics=dense_trt_iot or dense_iot,
    train_metrics={},
    onnx_mb=mb(dense_onnx_report),
)
add_row(
    model="Dense SFT",
    runtime_label="Native TensorRT",
    precision="FP16",
    sparsity="Dense",
    provider="trtexec",
    runtime_summary=dense_trt,
    iot_metrics=dense_trt_iot or {},
    train_metrics={},
    onnx_mb=mb(dense_onnx_report),
    sparse_selected=False,
    speedup=1.0 if dense_trt_latency else None,
)
add_row(
    model="NVIDIA 2:4 SFT",
    runtime_label="PyTorch",
    precision="FP16",
    sparsity="NVIDIA 2:4",
    provider="torch.cuda",
    runtime_summary=runtime("nvidia_2_4_sft_fp16_pytorch"),
    iot_metrics=sparse_iot,
    train_metrics=sparse_train,
    onnx_mb=mb(sparse_onnx_report),
    sparse_selected=None,
)
add_row(
    model="NVIDIA 2:4 SFT",
    runtime_label="Native TensorRT",
    precision="FP16",
    sparsity="NVIDIA 2:4",
    provider="trtexec",
    runtime_summary=sparse_trt,
    iot_metrics=sparse_trt_iot or {},
    train_metrics={},
    onnx_mb=mb(sparse_onnx_report),
    sparse_selected=sparse_tactics.get("sparse_tactics_selected"),
    speedup=main_speedup,
)

dense_int8_trt = runtime("dense_sft_int8_trt")
sparse_int8_trt = runtime("nvidia_2_4_sft_int8_trt")
if has_measured_runtime(dense_int8_trt):
    add_row(
        model="Dense SFT",
        runtime_label="Native TensorRT",
        precision="INT8",
        sparsity="Dense",
        provider="trtexec",
        runtime_summary=dense_int8_trt,
        iot_metrics={},
        train_metrics={},
        onnx_mb=mb(dense_onnx_report),
        sparse_selected=False,
    )
if has_measured_runtime(sparse_int8_trt):
    add_row(
        model="NVIDIA 2:4 SFT",
        runtime_label="Native TensorRT",
        precision="INT8",
        sparsity="NVIDIA 2:4",
        provider="trtexec",
        runtime_summary=sparse_int8_trt,
        iot_metrics={},
        train_metrics={},
        onnx_mb=mb(sparse_onnx_report),
        sparse_selected=sparse_tactics.get("sparse_tactics_selected"),
    )

fieldnames = [
    "Model",
    "Architecture",
    "Runtime",
    "Precision",
    "Sparsity",
    "Seq. Len.",
    "Batch Size",
    "Latency",
    "Median Lat.",
    "P95 Lat.",
    "P99 Lat.",
    "Throughput QPS",
    "Memory",
    "ONNX MB",
    "Engine MB",
    "EM@1 IoT200",
    "EM@5 IoT200",
    "EM@1 Train",
    "EM@5 Train",
    "GPU",
    "Provider",
    "Sparse Tactics Selected",
    "Speedup vs Dense TRT FP16",
]
results_dir = output_dir / "results"
results_dir.mkdir(parents=True, exist_ok=True)
json_path = results_dir / "final_metrics.json"
csv_path = results_dir / "final_metrics.csv"
payload = {
    "report_type": "h20_encoder_only_sft_nvidia_2_4_trt24",
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "main_speedup_dense_trt_fp16_over_nvidia_2_4_trt_fp16": main_speedup,
    "throughput_gain_nvidia_2_4_trt_fp16_over_dense_trt_fp16": throughput_gain,
    "cpu_onnx_only": cpu_onnx_only,
    "onnxruntime_providers": ort_providers,
    "sparse_tactics": sparse_tactics,
    "sparsity_report": {
        "exact_2_zero_block_pct": sparsity_report.get("exact_2_zero_block_pct"),
        "tensorrt_eligible_block_pct": sparsity_report.get("tensorrt_eligible_block_pct"),
        "total_sparsity_pct": sparsity_report.get("total_sparsity_pct"),
        "non_compliant_layers": sparsity_report.get("non_compliant_layers"),
    },
    "int8_status": int8_status,
    "rows": rows,
}
json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
with csv_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

def pct(value):
    return "n/a" if value is None else f"{float(value) * 100.0:.2f}%"

def num(value, suffix=""):
    return "n/a" if value is None else f"{float(value):.4f}{suffix}"

dense_acc = f"IoT200 EM@1/EM@5 {pct(dense_iot.get('exact_match_accuracy'))}/{pct(dense_iot.get('top5_accuracy'))}; train {pct(dense_train.get('exact_match_accuracy'))}/{pct(dense_train.get('top5_accuracy'))}"
sparse_acc = f"IoT200 EM@1/EM@5 {pct(sparse_iot.get('exact_match_accuracy'))}/{pct(sparse_iot.get('top5_accuracy'))}; train {pct(sparse_train.get('exact_match_accuracy'))}/{pct(sparse_train.get('top5_accuracy'))}"
summary_lines = [
    "# H20 Encoder-Only SFT + NVIDIA 2:4 + TensorRT 24 Summary",
    "",
    f"- GPU benchmark environment: CUDA available = {env.get('torch', {}).get('cuda_is_available')}, GPU = {gpu_name}, visible GPU count = {env.get('torch', {}).get('cuda_device_count')}.",
    f"- ONNX Runtime providers: {ort_providers}. CPU ONNX fallback used for speedup claims: no. CPU-only ONNX detected: {cpu_onnx_only}.",
    f"- TensorRT sparse tactics selected: {sparse_tactics.get('sparse_tactics_selected')}; eligible evidence found: {sparse_tactics.get('sparse_tactics_eligible')}.",
    f"- NVIDIA 2:4 target sparsity: exact 2-zero block pct = {num(sparsity_report.get('exact_2_zero_block_pct'), '%')}; non-compliant target layers = {sparsity_report.get('non_compliant_layers') or []}; intentionally dense rebuilt layers = {sparsity_report.get('ignored_dense_layers') or []}.",
    f"- Dense SFT PyTorch accuracy: {dense_acc}.",
    f"- NVIDIA 2:4 PyTorch accuracy: {sparse_acc}.",
    f"- Main speedup, computed only as dense native TensorRT FP16 latency / NVIDIA 2:4 native TensorRT FP16 latency: {num(main_speedup)}.",
    f"- Throughput gain, computed only as NVIDIA 2:4 native TensorRT QPS / dense native TensorRT QPS: {num(throughput_gain)}.",
    f"- Real sparse hardware speedup observed: {bool(main_speedup and main_speedup > 1.0 and sparse_tactics.get('sparse_tactics_selected'))}.",
    f"- INT8 status: {int8_status.get('status')}; {int8_status.get('reason') or ''}",
    "",
    "Limitations:",
    "- Native TensorRT .plan accuracy evaluation is not implemented in the repo. When ONNX Runtime TensorrtExecutionProvider is available, IoT200 TensorRT EP EM is reported as a TensorRT consistency proxy.",
    "- trtexec latency uses generated input tensors at batch size 1 and sequence length 64; PyTorch and ONNX Runtime latency use the repo benchmark script on pre-tokenized IoT prompts.",
    "- CPU ONNX is saved only as portability evidence if it appears; it is excluded from speedup claims.",
    "",
    f"Machine-readable metrics: `{json_path}` and `{csv_path}`.",
]
(output_dir / "SUMMARY.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
print(f"[h20-trt24] wrote {json_path}")
print(f"[h20-trt24] wrote {csv_path}")
print(f"[h20-trt24] wrote {output_dir / 'SUMMARY.md'}")
PY
}

normalize_json_dataset "$TRAIN_JSONL" "$TRAIN_JSON" "training data"
normalize_json_dataset "$IOT200_JSONL" "$IOT200_JSON" "IoT200 data"
write_env_report
check_python_dependencies
train_dense_sft

eval_pytorch_sharded "dense_sft_fp16" "$DENSE_CHECKPOINT" "iot200" "$IOT200_JSON"
eval_pytorch_sharded "dense_sft_fp16" "$DENSE_CHECKPOINT" "train" "$TRAIN_JSON"

prune_nvidia_2_4
verify_2_4_sparsity

eval_pytorch_sharded "nvidia_2_4_sft_fp16" "$PRUNED_CHECKPOINT" "iot200" "$IOT200_JSON"
eval_pytorch_sharded "nvidia_2_4_sft_fp16" "$PRUNED_CHECKPOINT" "train" "$TRAIN_JSON"

export_onnx_fp16 "$DENSE_CHECKPOINT" "$DENSE_ONNX"
export_onnx_fp16 "$PRUNED_CHECKPOINT" "$SPARSE_ONNX"
inspect_onnx "$DENSE_ONNX" "$REPORT_DIR/onnx_inspection_dense_sft_fp16.json"
inspect_onnx "$SPARSE_ONNX" "$REPORT_DIR/onnx_inspection_nvidia_2_4_sft_fp16.json"

require_trtexec
if command -v trtexec >/dev/null 2>&1; then
  build_trt_engine "dense_sft_fp16" "$DENSE_ONNX" "$DENSE_ENGINE" "disable" "$LOG_DIR/build_dense_fp16.log"
  build_trt_engine "nvidia_2_4_sft_fp16" "$SPARSE_ONNX" "$SPARSE_ENGINE" "enable" "$LOG_DIR/build_nvidia_2_4_sparse_fp16.log"
  parse_sparse_tactics_log
  benchmark_trt_engine "dense_sft_fp16_trt" "$DENSE_ENGINE" "$DENSE_ONNX" "FP16" "Dense"
  benchmark_trt_engine "nvidia_2_4_sft_fp16_trt" "$SPARSE_ENGINE" "$SPARSE_ONNX" "FP16" "NVIDIA 2:4"
  handle_int8_optional
else
  write_skip_report "$REPORT_DIR/trt_sparse_tactics_report.json" "trtexec is unavailable."
  write_skip_report "$REPORT_DIR/int8_status.json" "trtexec is unavailable."
fi

benchmark_pytorch "dense_sft_fp16_pytorch" "$DENSE_CHECKPOINT" "$RUNTIME_DIR/dense_sft_fp16_pytorch.json"
benchmark_pytorch "nvidia_2_4_sft_fp16_pytorch" "$PRUNED_CHECKPOINT" "$RUNTIME_DIR/nvidia_2_4_sft_fp16_pytorch.json"
benchmark_onnx_cuda_if_available
benchmark_ort_tensorrt_if_available
eval_ort_tensorrt_accuracy_if_available

aggregate_final_outputs

log "done"
echo "shell script: scripts/run_h20_encoder_only_sft_prune_trt24.sh"
echo "example:"
echo "  bash scripts/run_h20_encoder_only_sft_prune_trt24.sh \\"
echo "    --base_model /PATH/TO/BASE_MODEL"
echo "outputs:"
echo "  $OUTPUT_DIR/SUMMARY.md"
echo "  $OUTPUT_DIR/results/final_metrics.json"
echo "  $OUTPUT_DIR/results/final_metrics.csv"
