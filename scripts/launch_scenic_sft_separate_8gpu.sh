#!/usr/bin/env bash
set -euo pipefail

TRAINING_CONFIG="${TRAINING_CONFIG:-configs/scenic_sft_training_dataset_8gpu.yaml}"
CONTRASTIVE_CONFIG="${CONTRASTIVE_CONFIG:-configs/scenic_sft_contrastive_dataset_8gpu.yaml}"

echo "[scenic-sft:separate] training prompt-response dataset model | config=${TRAINING_CONFIG}"
CONFIG="${TRAINING_CONFIG}" ./scripts/launch_scenic_sft_8gpu.sh "$@"

echo "[scenic-sft:separate] training anchor-positive-negative dataset model | config=${CONTRASTIVE_CONFIG}"
CONFIG="${CONTRASTIVE_CONFIG}" ./scripts/launch_scenic_sft_8gpu.sh "$@"

echo "[scenic-sft:separate] done"
