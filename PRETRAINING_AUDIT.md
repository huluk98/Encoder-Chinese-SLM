# Pretraining Audit

## Verdict

The repo now follows a **RoBERTa-style BERT-family encoder pretraining recipe**, not a strict reproduction of the original 2018 BERT data pipeline.

This is the right description because the model is a bidirectional `BertForMaskedLM` trained with dynamic MLM, 15% masking, and the 80/10/10 replacement rule, while deliberately using byte-level BPE, packed text blocks, and no next-sentence-prediction objective.

## BERT-Critical Checks

| Area | Current Status | Notes |
| --- | --- | --- |
| Encoder architecture | OK | `BertForMaskedLM` with bidirectional self-attention and an MLM head. |
| CLS at block start | OK | `data.add_cls: true` prepends `tokenizer.cls_token_id` to every training block, including packed `.bin` training. |
| SEP/EOS boundaries | Mostly OK | Each formatted record ends with `<|eos|>`, and that token is mapped as `sep_token`. Packed blocks can still start or end mid-record, which is RoBERTa-style packing rather than strict sentence-pair BERT. |
| Masking rate | OK | `mask_probability: 0.15`. |
| Mask replacement | OK | 80% mask, 10% random token, 10% unchanged via `random_token_probability: 0.1` and `keep_token_probability: 0.1`. |
| Special tokens | OK | Pad, CLS/BOS, SEP/EOS, MASK, and role tokens are excluded from MLM labels. |
| Tokenizer family | Modern BERT-family, not strict original BERT | Uses byte-level BPE. Original BERT used WordPiece; RoBERTa uses byte-level BPE. |
| NSP | Modern BERT-family, not strict original BERT | NSP is not used. RoBERTa showed strong BERT-family training without NSP. |
| Token type IDs | OK for no-NSP | `type_vocab_size: 1`; no sentence-A/sentence-B segment training. |
| Whole-word masking | Not enabled | This is token-level MLM. Whole-word masking is an optional BERT-family variant, not required for the original recipe. |

## Tokenization And Packing

The tokenizer is trained with normalized Chinese text, byte-level BPE, and explicit special tokens:

- `<|bos|>` is mapped to the Hugging Face `cls_token`.
- `<|eos|>` is mapped to the Hugging Face `sep_token`.
- `<|mask|>` is mapped to the Hugging Face `mask_token`.

Packed token files remain raw corpus token streams. The training dataloader now uses `block_size - 1` raw tokens as payload and prepends CLS dynamically, so existing packed `.bin` files do not need to be rebuilt just to pick up the CLS fix. Re-running `scripts/pack_tokens.py` is only needed if you want the manifest counts to reflect `payload_tokens_per_block: 511`.

## Training Process

The H20 recipe trains with:

- dynamic MLM at batch collation time;
- BF16 precision;
- DeepSpeed ZeRO-1;
- DeepSpeed `FusedAdam` when available;
- cosine learning-rate decay with warmup;
- packed-token dataloading from the pre-tokenized `.bin` corpus.

The main caveat is terminology: call this **BERT/RoBERTa-style MLM pretraining**, not "strict original BERT pretraining." Strict original BERT would require WordPiece, `[CLS] sentence A [SEP] sentence B [SEP]`, token type IDs for sentence pairs, and NSP examples.

## Sources

- BERT paper: bidirectional encoder pretraining with MLM and NSP, plus the original BERT framing.  
  https://arxiv.org/abs/1810.04805
- Google Research BERT code: original pretraining-data generation with masked LM and next-sentence prediction, including CLS/SEP formatting.  
  https://github.com/google-research/bert
- Hugging Face data collator docs: `mlm_probability=0.15`, 80% mask replacement, 10% random replacement, remainder unchanged.  
  https://huggingface.co/docs/transformers/main_classes/data_collator
- RoBERTa docs: dynamic masking, sentence packing, larger batches, byte-level BPE, and no token type IDs requirement.  
  https://huggingface.co/docs/transformers/main/model_doc/roberta
- RoBERTa paper: optimized BERT-family pretraining without NSP.  
  https://arxiv.org/abs/1907.11692
