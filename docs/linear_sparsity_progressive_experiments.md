# Linear Sparsity and Progressive Recovery Experiments

This experiment block measures whether SCENIC IoT command-normalization accuracy is stable when linear weights are pruned at 30% and 50% sparsity. The base wrapper evaluates the encoder-only base checkpoint as a dense SCENIC baseline, trains both regular SFT and contrastive SFT for 5 epochs, keeps the original pruning methods as one-shot controls with classifier rebuild, and adds dense baselines plus progressive magnitude-pruning conditions with one recovery retune epoch after each pruning stage and one final recovery epoch, plus easy/medium/hard benchmark reporting for EM@1 and EM@5.

## Experiment Conditions

The one-line wrapper creates these conditions from a base encoder checkpoint:

- Original one-shot controls for each SFT checkpoint: `magnitude`, `wanda`, and `gradient` at 30%, plus `magnitude`, `wanda`, `gradient`, and `nvidia24` at 50%. NVIDIA 2:4 is skipped at 30% because it is exactly 50% sparse. These are still one-shot only and use `--reinitialize-classifier-from-responses`.
- Base encoder-only dense baseline, evaluated before SFT on both the SCENIC training data and benchmark. The response classifier is initialized from the training response texts so the base encoder can be scored with the same EM@1/EM@5 evaluator.
- Dense baselines for regular SFT and contrastive SFT.
- Added gradual retune block for each SFT checkpoint: progressive staged magnitude masks at 30% and 50%, with one recovery retune epoch after every pruning stage and one final recovery epoch after all stages.

The lower-level Python runner can create these conditions from an existing SCENIC SFT checkpoint:

- `dense_0`: no pruning, target sparsity `0.00`.
- `progressive_30`: staged magnitude pruning through `0.10`, `0.20`, `0.30`, with one recovery retune epoch after each stage and one final recovery epoch by default.
- `progressive_50`: staged magnitude pruning through `0.10`, `0.20`, `0.30`, `0.40`, `0.50`, with one recovery retune epoch after each stage and one final recovery epoch by default.

The expected final wrapper count is 21 result rows: 1 base encoder-only dense baseline row, 7 original one-shot rows for regular SFT, 7 original one-shot rows for contrastive SFT, 2 SFT dense baseline rows, 2 progressive magnitude rows for regular SFT, and 2 progressive magnitude rows for contrastive SFT.

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

This trains regular SFT and contrastive SFT checkpoints from the supplied base model unless `RETRAIN=0` and the dense checkpoint paths already exist. Training uses `./scripts/launch_scenic_sft_8gpu.sh` by default (`TRAIN_WITH_TORCHRUN=1`, `NPROC_PER_NODE=8`, `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`). Legacy one-shot pruning runs through `torchrun --nproc_per_node=$NPROC_PER_NODE`, with ranks sharding the one-shot job manifest. Added progressive/linear pruning jobs are split across `SPARSITY_GPU_IDS`.

Useful overrides:

```bash
RETRAIN=0 REGULAR_DENSE_CHECKPOINT=runs/scenic-sft-training-dataset/latest CONTRASTIVE_DENSE_CHECKPOINT=runs/scenic-sft-contrastive-dataset/latest ./scripts/run_scenic_sparsity_revision_from_base.sh /path/to/encoder-base-checkpoint
RECOVERY_EPOCHS_PER_STAGE=1 FINAL_RECOVERY_EPOCHS=1 RUN_PLOTS=0 ./scripts/run_scenic_sparsity_revision_from_base.sh /path/to/encoder-base-checkpoint
SFT_EPOCHS=5 TRAIN_WITH_TORCHRUN=1 NPROC_PER_NODE=8 SPARSITY_GPU_IDS=0,1,2,3,4,5,6,7 ./scripts/run_scenic_sparsity_revision_from_base.sh /path/to/encoder-base-checkpoint
```

Lower-level encoder-decoder-style command:

```bash
python scripts/run_sparsity_experiments.py \
  --experiment_name scenic_linear_sparsity_0_30_50 \
  --model_family encoder_decoder \
  --model_checkpoint PATH_TO_CHECKPOINT \
  --benchmark_path PATH_TO_BENCHMARK \
  --benchmark_difficulty_path PATH_TO_DIFFICULTY_LABELS \
  --sparsity_levels 0.3 0.5 \
  --pruning_modes dense progressive \
  --prune_scope linear_weights \
  --prune_method magnitude \
  --recovery_epochs_per_stage 1 \
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
  --sparsity_levels 0.3 0.5 \
  --pruning_modes dense progressive \
  --prune_method magnitude \
  --recovery_epochs_per_stage 1 \
  --final_recovery_epochs 1 \
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

- `all_sparsity_results.json` with regular and contrastive SFT rows in one payload
- `original_one_shot_reference_methods/original_one_shot_summary.csv`
- `linear_sparsity_retune/{regular_sft,contrastive_sft}/summary_metrics.csv`
- `predictions_{model_family}_{pruning_mode}_{sparsity}_{seed}.csv`
- `summary_metrics.csv`
- `paper_table_sparsity_difficulty.csv`
- `progressive_logs_{model_family}_{target_sparsity}_{seed}.csv`
- `masks/*.pt`
- `checkpoints/*`
- `figures/*.png`

`summary_metrics.csv` includes overall and difficulty-specific EM@1/EM@5, counts, 95% bootstrap confidence intervals, actual targeted Linear sparsity, whole-model sparsity, decoding config JSON, training config JSON, pruning config JSON, checkpoint paths, and mask paths.

## Paper Use

Use `paper_table_sparsity_difficulty.csv` for the main revised-paper table. Cite `summary_metrics.csv` for confidence intervals and progressive logs for the per-stage recovery schedule. For methods text, state that the gradual block uses unstructured magnitude pruning over selected Linear weights, with per-layer sparsity by default, one recovery retune epoch after each stage, one final recovery epoch after all stages, and masks enforced after every optimizer step during recovery.
