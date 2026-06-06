# Linear Sparsity and Progressive Recovery Experiments

This experiment block measures whether SCENIC IoT command-normalization accuracy is stable when linear weights are pruned at 0%, 30%, and 50% sparsity. It keeps the original four pruning methods as one-shot controls with classifier rebuild, and adds a separate progressive magnitude-pruning condition with one recovery retune epoch, plus easy/medium/hard benchmark reporting for EM@1 and EM@5.

## Experiment Conditions

The one-line wrapper creates these conditions from a base encoder checkpoint:

- Original one-shot controls at 50%: `magnitude`, `nvidia_2_4`, `wanda`, and `gradient`. These are still one-shot only and use `--reinitialize-classifier-from-responses`.
- Added linear-sparsity retune block at 0%, 30%, and 50%: dense baseline plus progressive staged magnitude masks, followed by exactly one recovery retune epoch by default.

The lower-level Python runner can create these conditions from an existing SCENIC SFT checkpoint:

- `dense_0`: no pruning, target sparsity `0.00`.
- `oneshot_30`: magnitude-prune selected Linear weights once to `0.30`.
- `oneshot_50`: magnitude-prune selected Linear weights once to `0.50`.
- `progressive_30`: staged pruning through `0.10`, `0.20`, `0.30`, then one recovery retune epoch by default.
- `progressive_50`: staged pruning through `0.10`, `0.20`, `0.30`, `0.40`, `0.50`, then one recovery retune epoch by default.

The 30% and 50% levels bracket a moderate compression setting and the original 50% setting, making it possible to report whether the paper conclusion is stable across sparsity severity.

## Pruning Scope

The new scripts prune only `torch.nn.Linear.weight` tensors by default. They exclude:

- bias terms
- embeddings
- LayerNorm and RMSNorm parameters
- `classifier`
- `lm_head`
- final response or output projection heads
- other output-head-like modules selected by name

Set `--prune_output_heads` only for an explicit ablation that includes output heads. Set `--global_pruning` to compute a single threshold across selected Linear weights; otherwise each selected Linear layer reaches the requested sparsity independently.

The result tables report both `targeted_linear_sparsity_actual` and `whole_model_sparsity_actual`, because pruning Linear weights does not make the full parameter set equally sparse.

## EM Metrics

EM@1 is true when the normalized top prediction exactly matches the normalized target response.

EM@5 is true when the normalized target response appears anywhere in the normalized top five predictions.

The encoder-only SCENIC checkpoint uses the highest-scoring canonical response for EM@1 and the top five canonical responses for EM@5. Decoder-only and encoder-decoder checkpoints use Hugging Face generation with `num_return_sequences >= 5`.

Text normalization strips whitespace, applies Unicode NFKC normalization, removes duplicated spaces, standardizes common punctuation spacing, and preserves Chinese characters.

## Difficulty Labels

The benchmark can carry an inline `difficulty`, `complexity`, or `level` column. Labels are normalized to lowercase and must be exactly `easy`, `medium`, or `hard`.

If the benchmark has no difficulty column, pass an external CSV or JSONL with one of:

- `id,difficulty`
- `sample_id,difficulty`
- `input,difficulty`

The evaluator joins by sample id first, then exact input command string. It raises a clear error if labels cannot be joined.

To create a blank labeling file:

```bash
python scripts/create_benchmark_difficulty_template.py \
  --benchmark_path data/scenic/iot_instruction_benchmark_200.json \
  --output_dir results/scenic_linear_sparsity_0_30_50
```

## Reproduction

Preferred one-line run from a base encoder checkpoint:

```bash
./scripts/run_scenic_sparsity_revision_from_base.sh /path/to/encoder-base-checkpoint
```

This trains the SCENIC SFT checkpoint from the supplied base model unless `RETRAIN=0` and `DENSE_CHECKPOINT` already exists. It then runs the original four methods as one-shot pruning with classifier rebuild, followed by the added 0/30/50 linear-sparsity retune block.

Useful overrides:

```bash
RETRAIN=0 DENSE_CHECKPOINT=runs/scenic-sft-training-dataset/latest ./scripts/run_scenic_sparsity_revision_from_base.sh /path/to/encoder-base-checkpoint
RETUNE_EPOCHS=1 RUN_PLOTS=0 ./scripts/run_scenic_sparsity_revision_from_base.sh /path/to/encoder-base-checkpoint
TRAIN_WITH_TORCHRUN=1 NPROC_PER_NODE=8 ./scripts/run_scenic_sparsity_revision_from_base.sh /path/to/encoder-base-checkpoint
```

Lower-level encoder-decoder-style command:

```bash
python scripts/run_sparsity_experiments.py \
  --experiment_name scenic_linear_sparsity_0_30_50 \
  --model_family encoder_decoder \
  --model_checkpoint PATH_TO_CHECKPOINT \
  --benchmark_path PATH_TO_BENCHMARK \
  --benchmark_difficulty_path PATH_TO_DIFFICULTY_LABELS \
  --sparsity_levels 0 0.3 0.5 \
  --pruning_modes dense oneshot progressive \
  --prune_scope linear_weights \
  --prune_method magnitude \
  --recovery_epochs_per_stage 0 \
  --final_recovery_epochs 1 \
  --num_beams 5 \
  --num_return_sequences 5 \
  --seed 42 \
  --output_dir results/scenic_linear_sparsity_0_30_50
```

For the repository's native SCENIC encoder-only SFT checkpoint:

```bash
python scripts/run_sparsity_experiments.py \
  --experiment_name scenic_linear_sparsity_0_30_50 \
  --model_family encoder_only \
  --model_checkpoint runs/scenic-sft-training-dataset/latest \
  --benchmark_path data/scenic/iot_instruction_benchmark_200.json \
  --sparsity_levels 0 0.3 0.5 \
  --pruning_modes dense oneshot progressive \
  --seed 42 \
  --output_dir results/scenic_linear_sparsity_0_30_50
```

Progressive recovery uses `--recovery_train_path` when supplied. For native SCENIC encoder checkpoints, the runner can also infer `config.data.train_json` from `scenic_sft_metadata.json`.

After the run, create figures with:

```bash
python scripts/plot_sparsity_results.py \
  --experiment_name scenic_linear_sparsity_0_30_50 \
  --results_dir results/scenic_linear_sparsity_0_30_50
```

## Outputs

Each run writes:

- `predictions_{model_family}_{pruning_mode}_{sparsity}_{seed}.csv`
- `summary_metrics.csv`
- `paper_table_sparsity_difficulty.csv`
- `progressive_logs_{model_family}_{target_sparsity}_{seed}.csv`
- `masks/*.pt`
- `checkpoints/*`
- `figures/*.png`

`summary_metrics.csv` includes overall and difficulty-specific EM@1/EM@5, counts, 95% bootstrap confidence intervals, actual targeted Linear sparsity, whole-model sparsity, decoding config JSON, training config JSON, pruning config JSON, checkpoint paths, and mask paths.

## Paper Use

Use `paper_table_sparsity_difficulty.csv` for the main revised-paper table. Cite `summary_metrics.csv` for retention values and confidence intervals. For methods text, state that pruning is unstructured magnitude pruning over selected Linear weights, with per-layer sparsity by default and masks enforced after every optimizer step during progressive recovery.
