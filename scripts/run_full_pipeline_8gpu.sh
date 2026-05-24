#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/h20_8gpu_bert_0p2b_deepspeed.yaml}"
PYTHON="${PYTHON:-python}"
FORCE_DOWNLOAD="${FORCE_DOWNLOAD:-0}"
FORCE_PREPARE="${FORCE_PREPARE:-0}"
FORCE_PACK="${FORCE_PACK:-0}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
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
  echo "nvidia-smi was not found. Run this script on the 8-GPU training host." >&2
  exit 2
fi

VISIBLE_GPU_COUNT="$(count_visible_gpus)"
if [[ "${VISIBLE_GPU_COUNT}" -ne 8 ]]; then
  echo "Expected 8 visible GPUs, got ${VISIBLE_GPU_COUNT}: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2
  exit 2
fi

echo "[pipeline] GPU inventory"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv

echo "[pipeline] Python/DeepSpeed inventory"
"${PYTHON}" -c "import torch, transformers, datasets, yaml, deepspeed; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'devices', torch.cuda.device_count()); print('transformers', transformers.__version__); print('datasets', datasets.__version__); print('deepspeed', deepspeed.__version__)"

download_args=(--config "${CONFIG}")
prepare_args=(--config "${CONFIG}")
pack_args=(--config "${CONFIG}")

if [[ "${FORCE_DOWNLOAD}" == "1" || "${FORCE_DOWNLOAD}" == "true" ]]; then
  download_args+=(--force-download)
  prepare_args+=(--force-download)
fi
if [[ "${FORCE_PREPARE}" == "1" || "${FORCE_PREPARE}" == "true" ]]; then
  prepare_args+=(--force)
fi
if [[ "${FORCE_PACK}" == "1" || "${FORCE_PACK}" == "true" ]]; then
  pack_args+=(--force)
fi

echo "[pipeline] Downloading configured corpus snapshots"
"${PYTHON}" scripts/download_data.py "${download_args[@]}"

echo "[pipeline] Normalizing corpus"
"${PYTHON}" scripts/prepare_data.py "${prepare_args[@]}"

echo "[pipeline] Training encoder tokenizer"
"${PYTHON}" scripts/train_tokenizer.py --config "${CONFIG}"

echo "[pipeline] Packing token ids"
"${PYTHON}" scripts/pack_tokens.py "${pack_args[@]}"

echo "[pipeline] Launching 8-GPU masked LM training"
PACK_TOKENS=0 CONFIG="${CONFIG}" PYTHON="${PYTHON}" ./scripts/launch_h20_8gpu.sh "$@"
