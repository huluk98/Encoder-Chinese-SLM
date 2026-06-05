from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer


PROMPT_FIELDS = ("prompt", "anchor", "instruction", "input", "text", "query")
POOLING_MODES = {"cls", "mean"}


def read_json_list(path: str | Path) -> list[dict[str, Any]]:
    data_path = Path(path).expanduser()
    value = json.loads(data_path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"{data_path} must contain a JSON list.")
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"{data_path}:{index} must contain a JSON object.")
        rows.append(item)
    return rows


def clean_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def prompt_from_row(row: dict[str, Any]) -> str:
    for field in PROMPT_FIELDS:
        text = clean_text(row.get(field))
        if text:
            return text
    return ""


def load_prompt_response_rows(path: str | Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for index, row in enumerate(read_json_list(path)):
        prompt = prompt_from_row(row)
        response = clean_text(row.get("response"))
        if not prompt or not response:
            raise ValueError(f"{path}:{index} must contain a prompt/anchor and response.")
        rows.append({"prompt": prompt, "response": response})
    return rows


def load_contrastive_rows(path: str | Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in read_json_list(path):
        anchor = clean_text(row.get("anchor") or row.get("prompt"))
        positive = clean_text(row.get("positive"))
        negative = clean_text(row.get("negative"))
        invalid_negative = clean_text(row.get("invalid_negative"))
        response = clean_text(row.get("response"))
        negative_response = clean_text(row.get("negative_response"))
        if anchor and positive and negative:
            item = {"anchor": anchor, "positive": positive, "negative": negative}
            if invalid_negative:
                item["invalid_negative"] = invalid_negative
            if response:
                item["response"] = response
            if negative_response:
                item["negative_response"] = negative_response
            rows.append(item)
    return rows


def build_label_maps(rows: list[dict[str, str]]) -> tuple[dict[str, int], list[str]]:
    responses = sorted({row["response"] for row in rows})
    response_to_label = {response: index for index, response in enumerate(responses)}
    return response_to_label, responses


class ScenicEncoderForResponseSelection(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        num_labels: int,
        dropout: float = 0.1,
        pooling: str = "cls",
        normalize_logits: bool = False,
        logit_scale: float = 20.0,
    ) -> None:
        super().__init__()
        pooling = str(pooling).lower()
        if pooling not in POOLING_MODES:
            raise ValueError(f"Unsupported pooling={pooling!r}. Use one of {sorted(POOLING_MODES)}.")
        self.encoder = encoder
        self.pooling = pooling
        self.normalize_logits = bool(normalize_logits)
        self.logit_scale = float(logit_scale)
        hidden_size = int(getattr(encoder.config, "hidden_size"))
        self.dropout = nn.Dropout(float(dropout))
        self.classifier = nn.Linear(hidden_size, int(num_labels))

    def pooled_output(self, outputs: Any, attention_mask: torch.Tensor | None) -> torch.Tensor:
        hidden = outputs.last_hidden_state
        if self.pooling == "cls":
            return hidden[:, 0]
        if attention_mask is None:
            return hidden.mean(dim=1)
        mask = attention_mask.unsqueeze(-1).to(dtype=hidden.dtype, device=hidden.device)
        denominator = mask.sum(dim=1).clamp_min(1.0)
        return (hidden * mask).sum(dim=1) / denominator

    def encode(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = self.encoder(**batch)
        return self.pooled_output(outputs, batch.get("attention_mask"))

    def classify(self, embeddings: torch.Tensor) -> torch.Tensor:
        embeddings = self.dropout(embeddings)
        if embeddings.is_floating_point() and embeddings.dtype != self.classifier.weight.dtype:
            embeddings = embeddings.to(dtype=self.classifier.weight.dtype)
        if not self.normalize_logits:
            return self.classifier(embeddings)
        normalized_embeddings = nn.functional.normalize(embeddings, dim=-1)
        normalized_weights = nn.functional.normalize(self.classifier.weight, dim=-1)
        logits = nn.functional.linear(normalized_embeddings, normalized_weights) * self.logit_scale
        if self.classifier.bias is not None:
            logits = logits + self.classifier.bias
        return logits

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        embeddings = self.encode(batch)
        logits = self.classify(embeddings)
        output = {"logits": logits, "embeddings": embeddings}
        if labels is not None:
            output["loss"] = nn.functional.cross_entropy(logits, labels)
        return output


def load_base_scenic_model(
    base_model: str | Path,
    tokenizer_path: str | Path | None,
    num_labels: int,
    dropout: float = 0.1,
    pooling: str = "cls",
    normalize_logits: bool = False,
    logit_scale: float = 20.0,
) -> tuple[ScenicEncoderForResponseSelection, Any]:
    tokenizer_source = str(Path(tokenizer_path).expanduser()) if tokenizer_path else str(Path(base_model).expanduser())
    model_source = str(Path(base_model).expanduser())
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
    encoder = load_encoder_without_pooler(model_source)
    if len(tokenizer) != int(encoder.config.vocab_size):
        encoder.resize_token_embeddings(len(tokenizer))
    return (
        ScenicEncoderForResponseSelection(
            encoder,
            num_labels=num_labels,
            dropout=dropout,
            pooling=pooling,
            normalize_logits=normalize_logits,
            logit_scale=logit_scale,
        ),
        tokenizer,
    )


def ensure_token_type_ids(encoded: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if "token_type_ids" not in encoded:
        encoded["token_type_ids"] = torch.zeros_like(encoded["input_ids"])
    return encoded


@torch.no_grad()
def initialize_classifier_from_responses(
    model: ScenicEncoderForResponseSelection,
    tokenizer: Any,
    label2response: list[str],
    device: torch.device,
    max_length: int = 128,
    batch_size: int = 128,
) -> None:
    was_training = model.training
    model.eval()
    embeddings: list[torch.Tensor] = []
    for start in range(0, len(label2response), max(1, int(batch_size))):
        texts = label2response[start : start + max(1, int(batch_size))]
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=int(max_length),
            return_tensors="pt",
        )
        encoded = ensure_token_type_ids(dict(encoded))
        encoded = {key: value.to(device) for key, value in encoded.items()}
        batch_embeddings = model.encode(encoded).float()
        embeddings.append(nn.functional.normalize(batch_embeddings, dim=-1).cpu())

    weight = torch.cat(embeddings, dim=0).to(
        device=model.classifier.weight.device,
        dtype=model.classifier.weight.dtype,
    )
    if weight.shape != model.classifier.weight.shape:
        raise ValueError(
            f"Response classifier init shape mismatch: {list(weight.shape)} "
            f"vs {list(model.classifier.weight.shape)}"
        )
    model.classifier.weight.copy_(weight)
    if model.classifier.bias is not None:
        model.classifier.bias.zero_()
    model.train(was_training)


def load_encoder_without_pooler(model_source: str) -> nn.Module:
    try:
        return AutoModel.from_pretrained(model_source, add_pooling_layer=False)
    except TypeError:
        return AutoModel.from_pretrained(model_source)


def save_scenic_checkpoint(
    model: ScenicEncoderForResponseSelection,
    tokenizer: Any,
    output_dir: str | Path,
    label2response: list[str],
    metadata: dict[str, Any],
) -> None:
    checkpoint_dir = Path(output_dir).expanduser()
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model.encoder.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    torch.save(
        {
            "classifier": model.classifier.state_dict(),
            "num_labels": len(label2response),
            "dropout": float(model.dropout.p),
            "pooling": model.pooling,
            "normalize_logits": model.normalize_logits,
            "logit_scale": model.logit_scale,
        },
        checkpoint_dir / "classifier.pt",
    )
    (checkpoint_dir / "label2response.json").write_text(
        json.dumps(label2response, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (checkpoint_dir / "scenic_sft_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_scenic_checkpoint(
    checkpoint_dir: str | Path,
    device: torch.device | str = "cpu",
) -> tuple[ScenicEncoderForResponseSelection, Any, list[str]]:
    checkpoint_path = Path(checkpoint_dir).expanduser()
    label2response = json.loads((checkpoint_path / "label2response.json").read_text(encoding="utf-8"))
    classifier_state = torch.load(checkpoint_path / "classifier.pt", map_location="cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path), use_fast=True)
    encoder = load_encoder_without_pooler(str(checkpoint_path))
    model = ScenicEncoderForResponseSelection(
        encoder,
        num_labels=int(classifier_state.get("num_labels", len(label2response))),
        dropout=float(classifier_state.get("dropout", 0.1)),
        pooling=str(classifier_state.get("pooling", "cls")),
        normalize_logits=bool(classifier_state.get("normalize_logits", False)),
        logit_scale=float(classifier_state.get("logit_scale", 20.0)),
    )
    model.classifier.load_state_dict(classifier_state["classifier"])
    model.to(device)
    model.eval()
    return model, tokenizer, list(label2response)
