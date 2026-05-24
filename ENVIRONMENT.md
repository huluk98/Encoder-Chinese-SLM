# Environment

This repo is intended to run on the 8x NVIDIA H20 training host, not on the local macOS laptop shell.

## Hardware And OS

- Linux x86_64 training host.
- 8x NVIDIA H20 GPUs visible as `0,1,2,3,4,5,6,7`.
- NVIDIA driver new enough for CUDA 12.4 runtime.
- At least 100 GB free disk for downloaded snapshots, normalized JSONL, packed token IDs, and checkpoints. More is better; 300 GB gives comfortable headroom for retries and multiple checkpoints.
- Fast local SSD/NVMe storage is strongly preferred for `data/raw`, `data/processed`, and `runs`.

## Core Software

- Conda or Mamba.
- Python 3.11.
- PyTorch with CUDA 12.4 (`pytorch-cuda=12.4`).
- DeepSpeed with CUDA extension build support.
- Git.
- A working C/C++ compiler toolchain. DeepSpeed may compile fused CUDA/optimizer extensions on first use.

## Recommended Conda Setup

```bash
git clone https://github.com/huluk98/Encoder-Chinese-SLM.git
cd Encoder-Chinese-SLM

conda env create -f environment.yml
conda activate chatlm-encoder
pip install -e ".[deepspeed]"
```

The repo's `environment.yml` pins the important GPU runtime condition:

```yaml
python=3.11
pytorch>=2.4
pytorch-cuda=12.4
deepspeed>=0.14.5
transformers>=4.48
datasets>=2.19
huggingface_hub>=0.24
hf-transfer>=0.1.8
hf-xet>=1.1.0
```

## Preflight Check

Run this before the full training pipeline:

```bash
nvidia-smi
python -c "import torch, transformers, datasets, yaml, deepspeed; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'devices', torch.cuda.device_count()); print('transformers', transformers.__version__); print('datasets', datasets.__version__); print('deepspeed', deepspeed.__version__)"
deepspeed --version
```

Expected:

- `torch.cuda.is_available()` is `True`.
- `torch.cuda.device_count()` is `8`.
- `deepspeed --version` runs without import errors.
- `nvidia-smi` shows the eight H20 GPUs.

## Runtime Environment Variables

The launch scripts set these by default:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
HF_HUB_ENABLE_HF_TRANSFER=1
TOKENIZERS_PARALLELISM=false
NCCL_DEBUG=WARN
TORCH_NCCL_ASYNC_ERROR_HANDLING=1
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OMP_NUM_THREADS=8
```

## Full Run

After the environment passes preflight:

```bash
./scripts/run_full_pipeline_8gpu.sh
```

This performs corpus download, normalization, tokenizer training, packed-token generation, and 8-GPU DeepSpeed training.
