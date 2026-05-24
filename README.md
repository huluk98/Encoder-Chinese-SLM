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

Set `RUN_CEVAL_AFTER_TRAIN=0` if you want to train first and run C-Eval manually later.

## Smoke Run

```bash
python scripts/train.py --config configs/smoke.yaml
```

The H20 config uses 8 GPUs, BF16, DeepSpeed ZeRO-1 with FusedAdam, SDPA attention, a 29,298-token Chinese BPE tokenizer with `<|mask|>`, 512-token bidirectional blocks, and 15% dynamic masked language modeling. The per-GPU microbatch is set to 128 with gradient accumulation 2, matching the decoder recipe's 1,048,576 input tokens per optimizer step while giving the encoder shorter, high-throughput sequences.
