# References

This project is an encoder-only masked language model training counterpart to the local decoder-only Chinese SLM recipe. The implementation and launch scripts are informed by the following model, systems, and data sources.

## Model And Training Objective

- Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"  
  https://arxiv.org/abs/1810.04805
- Google Research BERT repository, including the original BERT masked-language-modeling description  
  https://github.com/google-research/bert
- Hugging Face Transformers BERT documentation, including `BertForMaskedLM`  
  https://huggingface.co/docs/transformers/en/model_doc/bert
- Vaswani et al., "Attention Is All You Need"  
  https://papers.neurips.cc/paper/7181-attention-is-all-you-need
- PyTorch scaled dot product attention documentation  
  https://docs.pytorch.org/docs/main/generated/torch.nn.functional.scaled_dot_product_attention.html

## Distributed Training And Data Pipeline

- DeepSpeed ZeRO documentation  
  https://deepspeed.readthedocs.io/en/latest/zero3.html
- DeepSpeed optimizer documentation, including GPU `FusedAdam`  
  https://deepspeed.readthedocs.io/en/latest/optimizers.html
- Hugging Face Hub file download guide, including `snapshot_download`  
  https://huggingface.co/docs/huggingface_hub/en/guides/download
- Hugging Face Datasets loading documentation  
  https://huggingface.co/docs/datasets/loading
- Hugging Face dataset streaming documentation  
  https://huggingface.co/docs/datasets/v3.0.2/stream

## Configured Public Corpus Sources

- YeungNLP/firefly-pretrain-dataset  
  https://huggingface.co/datasets/YeungNLP/firefly-pretrain-dataset
- ZhouLV/Chinese-Train-Datasets  
  https://huggingface.co/datasets/ZhouLV/Chinese-Train-Datasets
- ticoAg/Chinese-medical-dialogue  
  https://huggingface.co/datasets/ticoAg/Chinese-medical-dialogue
- wangrui6/Zhihu-KOL  
  https://huggingface.co/datasets/wangrui6/Zhihu-KOL
- BelleGroup/train_1M_CN  
  https://huggingface.co/datasets/BelleGroup/train_1M_CN
- BelleGroup/train_2M_CN  
  https://huggingface.co/datasets/BelleGroup/train_2M_CN
- BelleGroup/train_3.5M_CN  
  https://huggingface.co/datasets/BelleGroup/train_3.5M_CN

## Local Reference Project

- Decoder-only source project used as the workflow template  
  https://github.com/huluk98/Decoder-Chinese-SLM
