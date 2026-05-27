#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1

TRAINING_CHECKPOINT="${TRAINING_CHECKPOINT:-runs/scenic-sft-training-dataset/latest}"
CONTRASTIVE_CHECKPOINT="${CONTRASTIVE_CHECKPOINT:-runs/scenic-sft-contrastive-dataset/latest}"

TRAINING_OUTPUT="${TRAINING_OUTPUT:-runs/scenic-sft-training-dataset-pruned50}"
CONTRASTIVE_OUTPUT="${CONTRASTIVE_OUTPUT:-runs/scenic-sft-contrastive-dataset-pruned50}"

BENCHMARK_JSON="${BENCHMARK_JSON:-data/scenic/iot_instruction_benchmark_200.json}"
TRAINING_JSON="${TRAINING_JSON:-data/scenic/SCENIC_full_training_dataset.json}"
CONTRASTIVE_JSON="${CONTRASTIVE_JSON:-data/scenic/SCENIC_full_anchor_positive_negative.json}"

EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-eval_results/scenic_sft/pruned50_comparison}"
SPARSITY="${SPARSITY:-0.5}"
PRUNE_SCOPE="${PRUNE_SCOPE:-encoder-linear}"
EVAL_DTYPE="${EVAL_DTYPE:-auto}"
BATCH_SIZE="${BATCH_SIZE:-128}"
MAX_LENGTH="${MAX_LENGTH:-128}"
OVERWRITE="${OVERWRITE:-1}"
INCLUDE_CLASSIFIER="${INCLUDE_CLASSIFIER:-0}"

overwrite_args=()
if [[ "$OVERWRITE" == "1" ]]; then
  overwrite_args+=(--overwrite)
fi

classifier_args=()
if [[ "$INCLUDE_CLASSIFIER" == "1" ]]; then
  classifier_args+=(--include-classifier)
fi

echo "[prune-scenic] pruning training-dataset model -> $TRAINING_OUTPUT"
python scripts/prune_scenic_sft.py \
  --checkpoint "$TRAINING_CHECKPOINT" \
  --output "$TRAINING_OUTPUT" \
  --sparsity "$SPARSITY" \
  --scope "$PRUNE_SCOPE" \
  "${overwrite_args[@]}" \
  "${classifier_args[@]}"

echo "[prune-scenic] pruning contrastive-anchor model -> $CONTRASTIVE_OUTPUT"
python scripts/prune_scenic_sft.py \
  --checkpoint "$CONTRASTIVE_CHECKPOINT" \
  --output "$CONTRASTIVE_OUTPUT" \
  --sparsity "$SPARSITY" \
  --scope "$PRUNE_SCOPE" \
  "${overwrite_args[@]}" \
  "${classifier_args[@]}"

echo "[prune-scenic] evaluating pruned models on benchmark and original training sources"
python scripts/eval_scenic_sft_comparison.py \
  --benchmark-json "$BENCHMARK_JSON" \
  --training-json "$TRAINING_JSON" \
  --contrastive-json "$CONTRASTIVE_JSON" \
  --training-checkpoint "$TRAINING_OUTPUT" \
  --contrastive-checkpoint "$CONTRASTIVE_OUTPUT" \
  --output-dir "$EVAL_OUTPUT_DIR" \
  --batch-size "$BATCH_SIZE" \
  --max-length "$MAX_LENGTH" \
  --dtype "$EVAL_DTYPE"

echo "[prune-scenic] pruning summaries:"
echo "  $TRAINING_OUTPUT/prune_summary.json"
echo "  $CONTRASTIVE_OUTPUT/prune_summary.json"
echo "[prune-scenic] evaluation outcomes:"
echo "  $EVAL_OUTPUT_DIR/comparison_summary.csv"
echo "  $EVAL_OUTPUT_DIR/comparison_groups.csv"
