from __future__ import annotations

from typing import Any

from transformers import BertConfig, BertForMaskedLM


def _token_id(tokenizer: Any | None, name: str) -> int | None:
    return getattr(tokenizer, name, None) if tokenizer is not None else None


def create_model(model_config: dict[str, Any], tokenizer: Any | None = None) -> BertForMaskedLM:
    architecture = str(model_config.get("architecture", "bert")).lower()
    if architecture not in {"bert", "roberta", "encoder", "encoder_only", "encoder-only"}:
        raise ValueError(f"Unknown encoder-only architecture: {architecture}")

    vocab_size = int(model_config.get("vocab_size") or (len(tokenizer) if tokenizer is not None else 29298))
    block_size = int(model_config.get("block_size", 512))
    config = BertConfig(
        vocab_size=vocab_size,
        max_position_embeddings=block_size,
        hidden_size=int(model_config.get("hidden_size", model_config.get("n_embd", 768))),
        num_hidden_layers=int(model_config.get("num_hidden_layers", model_config.get("n_layer", 24))),
        num_attention_heads=int(model_config.get("num_attention_heads", model_config.get("n_head", 12))),
        intermediate_size=int(model_config.get("intermediate_size", model_config.get("n_inner", 2048))),
        hidden_act=str(model_config.get("hidden_act", "gelu")),
        hidden_dropout_prob=float(model_config.get("hidden_dropout_prob", model_config.get("dropout", 0.0))),
        attention_probs_dropout_prob=float(
            model_config.get("attention_probs_dropout_prob", model_config.get("attention_dropout", 0.0))
        ),
        layer_norm_eps=float(model_config.get("layer_norm_eps", 1e-12)),
        type_vocab_size=int(model_config.get("type_vocab_size", 1)),
        initializer_range=float(model_config.get("initializer_range", 0.02)),
        pad_token_id=_token_id(tokenizer, "pad_token_id"),
        bos_token_id=_token_id(tokenizer, "bos_token_id"),
        eos_token_id=_token_id(tokenizer, "eos_token_id"),
    )
    config.tie_word_embeddings = bool(model_config.get("tie_word_embeddings", True))
    attn_implementation = model_config.get("attn_implementation")
    if attn_implementation:
        config._attn_implementation = str(attn_implementation)

    model = BertForMaskedLM(config)
    if tokenizer is not None and len(tokenizer) != model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))

    if bool(model_config.get("gradient_checkpointing", False)):
        model.gradient_checkpointing_enable()

    return model


def count_parameters(model: Any) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
