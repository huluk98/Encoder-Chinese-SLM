#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/scenic_sft_8gpu.yaml}"
NPROC="${NPROC:-8}"
PYTHON="${PYTHON:-python}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

count_visible_gpus() {
  local devices="${CUDA_VISIBLE_DEVICES}"
  local count=0
  local IFS=','
  read -ra ids <<< "${devices}"
  for id in "${ids[@]}"; do
    id="${id//[[:space:]]/}"
    if [[ -n "${id}" ]]; then
      count=$((count + 1))
    fi
  done
  echo "${count}"
}

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi was not found. Run this launcher on the 8-GPU training host." >&2
  exit 2
fi

if ! command -v torchrun >/dev/null 2>&1; then
  echo "torchrun was not found in PATH." >&2
  exit 2
fi

VISIBLE_GPU_COUNT="$(count_visible_gpus)"
if [[ "${VISIBLE_GPU_COUNT}" -ne "${NPROC}" ]]; then
  echo "Expected ${NPROC} visible GPUs, got ${VISIBLE_GPU_COUNT}: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2
  exit 2
fi

echo "[scenic-sft] 8-GPU launch | config=${CONFIG} | CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv

exec torchrun --standalone --nproc_per_node="${NPROC}" scripts/train_scenic_sft.py --config "${CONFIG}" "$@"
