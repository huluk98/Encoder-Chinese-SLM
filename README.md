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

## Smoke Run

```bash
python scripts/train.py --config configs/smoke.yaml
```

The H20 config uses 8 GPUs, BF16, DeepSpeed ZeRO-1 with FusedAdam, SDPA attention, a 29,298-token Chinese BPE tokenizer with `<|mask|>`, 512-token bidirectional blocks with a CLS-style prefix, and 15% dynamic masked language modeling. The model is approximately 0.194B parameters, close to the intended 0.2B class. The per-GPU microbatch is set to 128 with gradient accumulation 2, matching the decoder recipe's 1,048,576 input tokens per optimizer step while giving the encoder shorter, high-throughput sequences.
