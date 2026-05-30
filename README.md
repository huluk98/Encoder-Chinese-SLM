# Encoder-Only Chinese Mini MLM

This project mirrors the corpus and systems workflow from `/Users/luke/Documents/Decoder Only`, but trains a BERT-style encoder-only masked language model instead of a decoder-only causal LM.

The full H20 recipe keeps the same public Chinese data blend and tokenizer-first preparation:

- webText2019zh and wiki-like Firefly data
- Baike QA
- Chinese medical dialogue
- Zhihu-KOL
- BELLE 1M, 2M, and 3.5M Chinese instruction/chat data

See [REFERENCES.md](REFERENCES.md) for the model, systems, and dataset sources cited for this setup.
See [ENVIRONMENT.md](ENVIRONMENT.md) for the dedicated H20 environment recipe and preflight checks.
See [ARCHITECTURE.md](ARCHITECTURE.md) for the exact architecture classification and citations.
See [PRETRAINING_AUDIT.md](PRETRAINING_AUDIT.md) for the BERT/RoBERTa tokenization and MLM training audit.
See [BENCHMARKS.md](BENCHMARKS.md) for C-Eval output notes after training.

## Setup

```bash
conda env create -f environment.yml
conda activate chatlm-encoder
pip install -e ".[deepspeed]"
```

## Full 8-GPU Workflow

One command on the 8-GPU host:

```bash
./scripts/run_full_pipeline_8gpu.sh
```

The launcher defaults to the DeepSpeed CLI and uses all visible GPUs:

```bash
LAUNCHER=deepspeed CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./scripts/launch_h20_8gpu.sh
```

Use `LAUNCHER=torchrun` only for debugging the same DeepSpeed-initialized training script under torchrun.

Equivalent manual steps:

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 python scripts/download_data.py --config configs/h20_8gpu_bert_0p2b_deepspeed.yaml
python scripts/prepare_data.py --config configs/h20_8gpu_bert_0p2b_deepspeed.yaml
python scripts/train_tokenizer.py --config configs/h20_8gpu_bert_0p2b_deepspeed.yaml
python scripts/pack_tokens.py --config configs/h20_8gpu_bert_0p2b_deepspeed.yaml
./scripts/launch_h20_8gpu.sh
```

Resume after stopping or crashing:

```bash
./scripts/launch_h20_8gpu.sh --resume runs/h20-8gpu-bert-0p2b-mlm-deepspeed/latest
```

Monitor training:

```bash
tail -f runs/h20-8gpu-bert-0p2b-mlm-deepspeed/metrics/training_metrics.csv
python scripts/summarize_training_run.py runs/h20-8gpu-bert-0p2b-mlm-deepspeed
```

Plot loss:

```bash
python scripts/plot_loss.py --metrics runs/h20-8gpu-bert-0p2b-mlm-deepspeed/metrics/training_metrics.csv
```

The full pipeline also runs encoder-style C-Eval after training and writes results to `eval_results/ceval/latest`:

```bash
python scripts/eval_ceval.py --checkpoint runs/h20-8gpu-bert-0p2b-mlm-deepspeed/latest --split val --n-shot 5
```

Recorded C-Eval result for `step-100000` (`val`, 5-shot, MLM cloze scoring):

| Scope | Accuracy | Correct / Total |
| --- | ---: | ---: |
| Overall | 26.52% | 357 / 1346 |
| Humanities | 21.40% | 55 / 257 |
| Other | 27.60% | 106 / 384 |
| STEM | 27.21% | 117 / 430 |
| Social Science | 28.73% | 79 / 275 |

Set `RUN_CEVAL_AFTER_TRAIN=0` if you want to train first and run C-Eval manually later.

The pipeline skips an already-created tokenizer and packed token file by default. Use `FORCE_TOKENIZER=1` or `FORCE_PACK=1` only when you intentionally want to rebuild those artifacts.

## Fast 0.194B Check

Run this before launching a fresh H20 training job to confirm the config creates the intended 0.194B encoder:

```bash
git pull
PYTHONPATH=src python - <<'PY'
from chatlm_encoder.config import load_config
from chatlm_encoder.model import count_parameters, create_model
from chatlm_encoder.tokenizer import load_tokenizer

cfg = load_config("configs/h20_8gpu_bert_0p2b_deepspeed.yaml")
tok = load_tokenizer(cfg["tokenizer"]["path"])
model = create_model(cfg["model"], tok)

print("tokenizer_size:", len(tok))
print("layers:", model.config.num_hidden_layers)
print("hidden_size:", model.config.hidden_size)
print("heads:", model.config.num_attention_heads)
print("intermediate:", model.config.intermediate_size)
print("vocab_size:", model.config.vocab_size)
print("parameters:", f"{count_parameters(model):,}")
PY
```

Expected:

```text
tokenizer_size: 29298
layers: 24
hidden_size: 768
heads: 12
intermediate: 3072
vocab_size: 29298
parameters: 193,700,466
```

If an older-size run already exists, move it aside and start a fresh run without `--resume`:

```bash
mv runs/h20-8gpu-bert-0p2b-mlm-deepspeed "runs/old-wrong-size-run-$(date +%Y%m%d-%H%M%S)"
PACK_TOKENS=0 ./scripts/launch_h20_8gpu.sh
```

The startup log should print `Parameters: 193,700,466`. After the first checkpoint, confirm the saved model shape:

```bash
python - <<'PY'
import json
p = "runs/h20-8gpu-bert-0p2b-mlm-deepspeed/latest/config.json"
cfg = json.load(open(p, encoding="utf-8"))
print(cfg["num_hidden_layers"], cfg["hidden_size"], cfg["num_attention_heads"], cfg["intermediate_size"], cfg["vocab_size"])
PY
```

Expected:

```text
24 768 12 3072 29298
```

## SCENIC Encoder SFT

This is encoder-only supervised fine-tuning, so it trains prompt-to-response **classifiers** rather than generative decoders. The two SCENIC datasets are trained into separate checkpoints for comparison tables:

- `data/scenic/SCENIC_full_training_dataset.json` -> `runs/scenic-sft-training-dataset/latest`
- `data/scenic/SCENIC_full_anchor_positive_negative.json` -> `runs/scenic-sft-contrastive-dataset/latest`

The contrastive-dataset model uses `anchor -> response` as the supervised task and also uses the positive/negative fields as an auxiliary contrastive loss.

These SCENIC JSON files are tracked in this repo, so `git pull` brings them down with the SFT scripts. Train both models sequentially on the 8-GPU H20 box:

```bash
./scripts/launch_scenic_sft_separate_8gpu.sh
```

Or train one model at a time:

```bash
CONFIG=configs/scenic_sft_training_dataset_8gpu.yaml ./scripts/launch_scenic_sft_8gpu.sh
CONFIG=configs/scenic_sft_contrastive_dataset_8gpu.yaml ./scripts/launch_scenic_sft_8gpu.sh
```

Both configs expect the MLM-pretrained encoder at:

```text
runs/h20-8gpu-bert-0p2b-mlm-deepspeed/latest
```

Set `model.base_model` in each SCENIC SFT config if your checkpoint path is different. Change each model output path with `run.output_dir`.

Evaluate a local SCENIC-style JSON file:

```bash
python scripts/eval_scenic_sft_local.py
```

By default this evaluates on the 200-row SCENIC IoT benchmark:

```text
data/scenic/iot_instruction_benchmark_200.json
```

Run the full comparison suite:

```bash
python scripts/eval_scenic_sft_comparison.py
```

This runs four evaluations:

- training-dataset model on the 200-row benchmark
- training-dataset model on `SCENIC_full_training_dataset.json` for retention
- contrastive-dataset model on the 200-row benchmark
- contrastive-dataset model on `SCENIC_full_anchor_positive_negative.json` for retention, using only `anchor` as the input prompt

The suite writes:

```text
eval_results/scenic_sft/comparison/comparison_summary.json
eval_results/scenic_sft/comparison/comparison_summary.csv
eval_results/scenic_sft/comparison/comparison_groups.csv
```

### 50% SCENIC SFT pruning and outcomes

After both SCENIC SFT checkpoints exist, run the pruning-and-evaluation launcher:

```bash
./scripts/prune_scenic_sft_50_eval.sh
```

This creates two separate 50% sparse pruned checkpoints:

```text
runs/scenic-sft-training-dataset-pruned50
runs/scenic-sft-contrastive-dataset-pruned50
```

By default, pruning is unstructured magnitude pruning over encoder transformer linear weights. It leaves token embeddings, LayerNorm/bias tensors, and the response classifier unchanged. The checkpoint files are not expected to become 50% smaller unless you use sparse storage or sparse inference kernels later; the script reports the real zero-rate in each `prune_summary.json`.

The launcher evaluates the pruned models on:

- the 200-row SCENIC benchmark: `data/scenic/iot_instruction_benchmark_200.json`
- the original training source for the training-dataset model: `data/scenic/SCENIC_full_training_dataset.json`
- the original anchor source for the contrastive model: `data/scenic/SCENIC_full_anchor_positive_negative.json`

Outcome files:

```text
runs/scenic-sft-training-dataset-pruned50/prune_summary.json
runs/scenic-sft-contrastive-dataset-pruned50/prune_summary.json
eval_results/scenic_sft/pruned50_comparison/comparison_summary.csv
eval_results/scenic_sft/pruned50_comparison/comparison_groups.csv
```

Use `comparison_summary.csv` for the main table. The `benchmark_200` rows are the SCENIC benchmark outcomes. The `training_dataset_retention` and `contrastive_anchor_retention` rows are the original-data retention outcomes.

Useful overrides:

```bash
SPARSITY=0.5 PRUNE_SCOPE=encoder-linear ./scripts/prune_scenic_sft_50_eval.sh
EVAL_DTYPE=fp32 BATCH_SIZE=64 ./scripts/prune_scenic_sft_50_eval.sh
BENCHMARK_JSON=/path/to/benchmark.json ./scripts/prune_scenic_sft_50_eval.sh
```

If you want 50% sparsity over every matrix weight, including embeddings, use:

```bash
PRUNE_SCOPE=all-matrix ./scripts/prune_scenic_sft_50_eval.sh
```

Run all pruning variants and aggregate the benchmark/original-data outcomes:

```bash
./scripts/prune_scenic_sft_all_methods_eval.sh
```

This runs:

- `encoder-linear`: encoder transformer linear weights only, classifier kept dense
- `all-linear-classifier`: all non-embedding 2D weights, including the response classifier
- `all-matrix-classifier`: all matrix weights, including embeddings and the response classifier

The combined table is written to:

```text
eval_results/scenic_sft/pruned50_all_methods/all_methods_summary.csv
eval_results/scenic_sft/pruned50_all_methods/all_methods_summary.json
```

To run only selected methods:

```bash
METHODS=encoder-linear,all-matrix-classifier ./scripts/prune_scenic_sft_all_methods_eval.sh
```

Evaluate both checkpoints on the same local JSON file and write exact-match summaries:

```bash
python scripts/eval_scenic_sft_local.py \
  --json data/scenic/iot_instruction_benchmark_200.json \
  --checkpoint runs/scenic-sft-training-dataset/latest \
  --output eval_results/scenic_sft/training_dataset_predictions.jsonl \
  --summary-output eval_results/scenic_sft/training_dataset_summary.json \
  --dtype auto

python scripts/eval_scenic_sft_local.py \
  --json data/scenic/iot_instruction_benchmark_200.json \
  --checkpoint runs/scenic-sft-contrastive-dataset/latest \
  --output eval_results/scenic_sft/contrastive_dataset_predictions.jsonl \
  --summary-output eval_results/scenic_sft/contrastive_dataset_summary.json \
  --dtype auto
```

Evaluate a different local JSON file:

```bash
python scripts/eval_scenic_sft_local.py \
  --json /path/to/eval.json \
  --checkpoint runs/scenic-sft-training-dataset/latest \
  --output eval_results/scenic_sft/eval_predictions.jsonl \
  --summary-output eval_results/scenic_sft/eval_summary.json
```

For no-flag usage, edit these constants at the top of `scripts/eval_scenic_sft_local.py`:

```python
LOCAL_JSON_PATH = "data/scenic/iot_instruction_benchmark_200.json"
CHECKPOINT_DIR = "runs/scenic-sft-training-dataset/latest"
OUTPUT_PATH = "eval_results/scenic_sft/benchmark_200_predictions.jsonl"
SUMMARY_OUTPUT_PATH = "eval_results/scenic_sft/benchmark_200_summary.json"
EVAL_DTYPE = "auto"
```

The evaluator accepts JSON lists with either `prompt` + `response` rows or `anchor` + `response` rows. It prints exact-match accuracy, top-5 accuracy, label-space coverage, writes per-row predictions, and saves a summary JSON containing `exact_match_accuracy` plus grouped accuracy by `difficulty`, `task_type`, and `source` when those fields exist.

## Smoke Run

```bash
python scripts/train.py --config configs/smoke.yaml
```

The H20 config uses 8 GPUs, BF16, DeepSpeed ZeRO-1 with FusedAdam, SDPA attention, a 29,298-token Chinese BPE tokenizer with `<|mask|>`, 512-token bidirectional blocks with a CLS-style prefix, and 15% dynamic masked language modeling. The model is approximately 0.194B parameters, close to the intended 0.2B class. The per-GPU microbatch is set to 128 with gradient accumulation 2, matching the decoder recipe's 1,048,576 input tokens per optimizer step while giving the encoder shorter, high-throughput sequences.
