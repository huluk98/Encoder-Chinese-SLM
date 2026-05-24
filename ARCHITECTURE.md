# Architecture

This repo trains a **custom BERT-family encoder-only masked language model**.

More specifically, it is a **BERT/RoBERTa-style bidirectional Transformer encoder** trained with dynamic masked language modeling (MLM). It is not a decoder-only autoregressive LM, not an encoder-decoder model, and not a ModernBERT or DeBERTaV3 implementation.

## What It Uses

- **Model family:** encoder-only Transformer.
- **Implementation class:** Hugging Face `BertForMaskedLM`.
- **Pretraining objective:** masked language modeling.
- **Attention direction:** bidirectional full self-attention.
- **Tokenizer:** Chinese-friendly byte-level BPE with explicit `<|mask|>` token.
- **NSP objective:** not used.
- **Position encoding:** classic BERT absolute position embeddings.
- **Training style:** RoBERTa-like in the sense that masks are generated dynamically, blocks are packed, and no next-sentence-prediction loss is used.
- **Block format:** every 512-token training block receives a CLS-style prefix token; records are separated with `<|eos|>`, mapped as the tokenizer SEP token.

## H20 Model Shape

The main config is `configs/h20_8gpu_bert_0p2b_deepspeed.yaml`:

| Field | Value |
| --- | --- |
| Architecture | `bert` |
| Objective | MLM |
| Vocabulary | 29,298 tokens |
| Context length | 512 tokens |
| Layers | 24 |
| Hidden size | 768 |
| Attention heads | 12 |
| FFN/intermediate size | 3072 |
| Activation | GELU |
| Dropout | 0.0 |
| Embeddings | tied MLM decoder embeddings |
| Attention implementation | SDPA |
| Precision | BF16 |
| Distributed runtime | DeepSpeed ZeRO-1 |

This is a custom compact Chinese BERT-style encoder. It borrows the BERT encoder/MLM form, but its dimensions are not exactly BERT-base or BERT-large. With the H20 config above, the model is approximately 0.194B parameters, close to the intended 0.2B class.

## Pretraining Recipe Classification

The training pipeline is **correct for modern BERT-family MLM pretraining**, but it is **not a strict original BERT reproduction**. The important distinction:

- It keeps the BERT-family essentials: bidirectional encoder, CLS token at block start, SEP/EOS record separators, MLM head, 15% dynamic masking, and the 80/10/10 mask replacement rule.
- It follows RoBERTa-style updates: byte-level BPE, packed token blocks, dynamic masking, and no NSP.
- It does not implement original BERT sentence-pair NSP data generation, WordPiece tokenization, or token-type segment prediction.

See [PRETRAINING_AUDIT.md](PRETRAINING_AUDIT.md) for the tokenization/training audit.

## What It Is Not

- **Not decoder-only:** it does not train with causal next-token prediction.
- **Not encoder-decoder:** it has no separate decoder stack.
- **Not ModernBERT:** it does not yet use RoPE, alternating local/global attention, long 8192-token context, or ModernBERT-specific block choices.
- **Not DeBERTaV3:** it does not use disentangled attention or ELECTRA-style replaced token detection.

## Citations

- BERT introduced the bidirectional Transformer encoder pretraining setup and masked language modeling objective:  
  Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"  
  https://arxiv.org/abs/1810.04805

- This repo uses Hugging Face's BERT masked-LM implementation surface:  
  Hugging Face Transformers BERT documentation, `BertForMaskedLM`  
  https://huggingface.co/docs/transformers/en/model_doc/bert

- The original BERT codebase describes masked-LM and next-sentence-prediction pretraining; this repo keeps MLM but does not use NSP:  
  Google Research BERT repository  
  https://github.com/google-research/bert

- RoBERTa motivates dynamic masking and removing NSP from the BERT pretraining recipe:  
  Liu et al., "RoBERTa: A Robustly Optimized BERT Pretraining Approach"  
  https://arxiv.org/abs/1907.11692

- The underlying encoder block is from the Transformer architecture:  
  Vaswani et al., "Attention Is All You Need"  
  https://papers.neurips.cc/paper/7181-attention-is-all-you-need

- ModernBERT is a newer encoder-only architecture and is useful future work, but it is not the architecture currently implemented here:  
  Warner et al., "Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference"  
  https://arxiv.org/abs/2412.13663

- DeBERTaV3 is another newer encoder-family alternative using disentangled attention and replaced token detection, but it is not implemented here:  
  He et al., "DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing"  
  https://arxiv.org/abs/2111.09543
