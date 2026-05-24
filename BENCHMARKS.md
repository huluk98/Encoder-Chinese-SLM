# Benchmark Remarks

This file records benchmark results after training runs complete.

## C-Eval

The full 8-GPU pipeline runs C-Eval automatically after a successful training run:

```bash
./scripts/run_full_pipeline_8gpu.sh
```

To skip the automatic post-training benchmark:

```bash
RUN_CEVAL_AFTER_TRAIN=0 ./scripts/run_full_pipeline_8gpu.sh
```

The post-training C-Eval command is equivalent to:

```bash
python scripts/eval_ceval.py \
  --checkpoint runs/h20-8gpu-bert-0p2b-mlm-deepspeed/latest \
  --split val \
  --n-shot 5 \
  --subjects all \
  --output-dir eval_results/ceval/latest \
  --dtype bf16 \
  --device auto
```

This encoder-only evaluator uses **MLM cloze scoring**: it formats each C-Eval item as a Chinese multiple-choice question ending in `答案：`, masks the candidate answer token(s), and selects the A/B/C/D option with the highest masked-token log probability. This is appropriate for an encoder-only masked language model, but it is not the same scoring method as an autoregressive decoder-only C-Eval evaluation.

### Latest Result

Pending. After you run the pipeline, send the contents of:

- `eval_results/ceval/latest/ceval_summary.json`
- or the final console line printed by `scripts/eval_ceval.py`

I will update this section with the exact overall accuracy, category breakdown, checkpoint path, and notes.
