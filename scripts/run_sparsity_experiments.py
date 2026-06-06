#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from chatlm_encoder.linear_sparsity import (  # noqa: E402
    LinearSparsityConfig,
    apply_magnitude_pruning,
    apply_masks,
    current_linear_sparsity_summary,
    register_mask_gradient_hooks,
    remove_hooks,
    save_masks,
)
from chatlm_encoder.scenic_sft import (  # noqa: E402
    ScenicEncoderForResponseSelection,
    ensure_token_type_ids,
    initialize_classifier_from_responses,
    load_scenic_checkpoint,
    save_scenic_checkpoint,
)
from chatlm_encoder.sparsity_eval import (  # noqa: E402
    BenchmarkSample,
    compute_metric_breakdown,
    first_text,
    load_benchmark_samples,
    normalize_text,
    prediction_record,
    read_table,
    retention,
    write_prediction_csv,
    write_rows_csv,
)


@dataclass
class ModelBundle:
    model_family: str
    model: nn.Module
    tokenizer: Any
    label2response: list[str] | None = None
    metadata: dict[str, Any] | None = None


PREDICTION_FIELDNAMES = [
    "sample_id",
    "input",
    "target",
    "difficulty",
    "top1_prediction",
    "top5_predictions",
    "em1",
    "em5",
    "model_family",
    "pruning_mode",
    "pruning_method",
    "target_sparsity",
    "targeted_linear_sparsity_actual",
    "whole_model_sparsity_actual",
    "seed",
]

SUMMARY_FIELDNAMES = [
    "experiment_name",
    "model_family",
    "pruning_mode",
    "pruning_method",
    "target_sparsity",
    "targeted_linear_sparsity_actual",
    "whole_model_sparsity_actual",
    "seed",
    "em1_overall",
    "em1_overall_ci_low",
    "em1_overall_ci_high",
    "em5_overall",
    "em5_overall_ci_low",
    "em5_overall_ci_high",
    "em1_easy",
    "em1_easy_ci_low",
    "em1_easy_ci_high",
    "em1_easy_ci_status",
    "em5_easy",
    "em5_easy_ci_low",
    "em5_easy_ci_high",
    "em5_easy_ci_status",
    "count_easy",
    "em1_medium",
    "em1_medium_ci_low",
    "em1_medium_ci_high",
    "em1_medium_ci_status",
    "em5_medium",
    "em5_medium_ci_low",
    "em5_medium_ci_high",
    "em5_medium_ci_status",
    "count_medium",
    "em1_hard",
    "em1_hard_ci_low",
    "em1_hard_ci_high",
    "em1_hard_ci_status",
    "em5_hard",
    "em5_hard_ci_low",
    "em5_hard_ci_high",
    "em5_hard_ci_status",
    "count_hard",
    "count_total",
    "em1_retention_overall",
    "em5_retention_overall",
    "em1_retention_easy",
    "em5_retention_easy",
    "em1_retention_medium",
    "em5_retention_medium",
    "em1_retention_hard",
    "em5_retention_hard",
    "decoding_config_json",
    "training_config_json",
    "pruning_config_json",
    "checkpoint_path",
    "mask_path",
    "predictions_path",
]

PAPER_FIELDNAMES = [
    "model_family",
    "pruning_mode",
    "target_sparsity",
    "overall EM@1",
    "overall EM@5",
    "easy EM@1",
    "easy EM@5",
    "medium EM@1",
    "medium EM@5",
    "hard EM@1",
    "hard EM@5",
    "targeted linear sparsity",
    "whole-model sparsity",
]

PROGRESSIVE_FIELDNAMES = [
    "stage",
    "stage_target_sparsity",
    "targeted_linear_sparsity_actual",
    "whole_model_sparsity_actual",
    "recovery_epoch",
    "train_loss",
    "val_em1",
    "val_em5",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run controlled SCENIC linear-weight sparsity experiments at 0%, 30%, and 50%."
    )
    parser.add_argument("--experiment_name", required=True)
    parser.add_argument("--model_family", required=True)
    parser.add_argument("--model_checkpoint", required=True)
    parser.add_argument("--sparsity_levels", type=float, nargs="+", default=[0.0, 0.3, 0.5])
    parser.add_argument(
        "--pruning_modes",
        "--pruning_mode",
        dest="pruning_modes",
        nargs="+",
        default=["dense", "oneshot", "progressive"],
        choices=("dense", "oneshot", "progressive"),
    )
    parser.add_argument("--prune_scope", default="linear_weights", choices=("linear_weights",))
    parser.add_argument("--prune_method", default="magnitude", choices=("magnitude",))
    parser.add_argument("--progressive_schedule", default="staged", choices=("staged",))
    parser.add_argument("--recovery_epochs_per_stage", type=int, default=0)
    parser.add_argument("--final_recovery_epochs", type=int, default=1)
    parser.add_argument("--prune_output_heads", action="store_true")
    parser.add_argument("--global_pruning", action="store_true")
    parser.add_argument("--regrowth", action="store_true")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--benchmark_path", required=True)
    parser.add_argument("--benchmark_difficulty_path", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--num_return_sequences", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--length_penalty", type=float, default=1.0)
    parser.add_argument("--early_stopping", action="store_true")
    parser.add_argument("--normalization_mode", default="scenic")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--max_grad_norm", type=float, default=None)
    parser.add_argument("--recovery_train_path", default=None)
    parser.add_argument("--validation_path", default=None)
    parser.add_argument("--bootstrap_resamples", type=int, default=1000)
    parser.add_argument(
        "--reinitialize_classifier_from_responses",
        action="store_true",
        help=(
            "For encoder-only SCENIC runs, rebuild the dense response classifier "
            "from response embeddings after one-shot/progressive pruning, matching "
            "the reference one-shot pruning control."
        ),
    )
    parser.add_argument("--classifier_init_batch_size", type=int, default=128)
    parser.add_argument("--classifier_init_max_length", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=("auto", "fp32", "bf16", "fp16"))
    return parser.parse_args()


def normalize_model_family(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    aliases = {
        "encoder": "encoder_only",
        "encoderonly": "encoder_only",
        "decoder": "decoder_only",
        "decoderonly": "decoder_only",
        "seq2seq": "encoder_decoder",
        "encoderdecoder": "encoder_decoder",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"encoder_only", "decoder_only", "encoder_decoder"}:
        raise ValueError("model_family must be encoder-only, decoder-only, or encoder-decoder.")
    return normalized


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def dtype_for(name: str, device: torch.device) -> torch.dtype:
    if name == "auto":
        return torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    if name == "bf16":
        if device.type != "cuda":
            raise ValueError("--dtype bf16 requires CUDA.")
        return torch.bfloat16
    if name == "fp16":
        if device.type != "cuda":
            raise ValueError("--dtype fp16 requires CUDA.")
        return torch.float16
    return torch.float32


def autocast_for(device: torch.device, dtype: torch.dtype):
    if device.type == "cuda" and dtype in {torch.bfloat16, torch.float16}:
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def sparsity_label(value: float) -> str:
    if math.isclose(float(value), round(float(value))):
        return str(int(round(float(value))))
    return f"{float(value):.2f}".rstrip("0").rstrip(".").replace(".", "p")


def run_plan(sparsity_levels: list[float], pruning_modes: list[str]) -> list[tuple[str, float]]:
    levels = sorted({round(float(level), 8) for level in sparsity_levels})
    plan: list[tuple[str, float]] = []
    if "dense" in pruning_modes:
        plan.append(("dense", 0.0))
    if "oneshot" in pruning_modes:
        plan.extend(("oneshot", level) for level in levels if level > 0.0)
    if "progressive" in pruning_modes:
        plan.extend(("progressive", level) for level in levels if level > 0.0)
    return plan


def stage_schedule(target_sparsity: float) -> list[float]:
    target = round(float(target_sparsity), 8)
    if math.isclose(target, 0.3, abs_tol=1e-8):
        return [0.1, 0.2, 0.3]
    if math.isclose(target, 0.5, abs_tol=1e-8):
        return [0.1, 0.2, 0.3, 0.4, 0.5]
    stages: list[float] = []
    current = 0.1
    while current < target:
        stages.append(round(current, 2))
        current += 0.1
    if not stages or not math.isclose(stages[-1], target):
        stages.append(target)
    return stages


def read_scenic_metadata(checkpoint: str | Path) -> dict[str, Any]:
    metadata_path = Path(checkpoint).expanduser() / "scenic_sft_metadata.json"
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def load_bundle(
    model_family: str,
    checkpoint: str | Path,
    device: torch.device,
    dtype: torch.dtype,
) -> ModelBundle:
    if model_family == "encoder_only":
        model, tokenizer, label2response = load_scenic_checkpoint(checkpoint, device="cpu")
        model.to(device=device, dtype=dtype)
        model.eval()
        return ModelBundle(
            model_family=model_family,
            model=model,
            tokenizer=tokenizer,
            label2response=label2response,
            metadata=read_scenic_metadata(checkpoint),
        )

    from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(Path(checkpoint).expanduser()), use_fast=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if model_family == "decoder_only":
        model = AutoModelForCausalLM.from_pretrained(str(Path(checkpoint).expanduser()))
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(str(Path(checkpoint).expanduser()))
    model.to(device=device, dtype=dtype)
    model.eval()
    return ModelBundle(model_family=model_family, model=model, tokenizer=tokenizer, metadata={})


def save_bundle_checkpoint(bundle: ModelBundle, output_dir: Path, metadata: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if bundle.model_family == "encoder_only":
        assert isinstance(bundle.label2response, list)
        save_scenic_checkpoint(
            model=bundle.model,  # type: ignore[arg-type]
            tokenizer=bundle.tokenizer,
            output_dir=output_dir,
            label2response=bundle.label2response,
            metadata=metadata,
        )
        return
    bundle.model.save_pretrained(output_dir)
    bundle.tokenizer.save_pretrained(output_dir)
    (output_dir / "sparsity_experiment_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def batched(items: list[Any], batch_size: int):
    for start in range(0, len(items), max(1, int(batch_size))):
        yield items[start : start + max(1, int(batch_size))]


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


@torch.inference_mode()
def evaluate_encoder(
    bundle: ModelBundle,
    samples: list[BenchmarkSample],
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, Any]]:
    assert bundle.label2response is not None
    model = bundle.model
    model.eval()
    records: list[dict[str, Any]] = []
    k = min(max(5, int(args.num_return_sequences)), len(bundle.label2response))
    batch_size = int(args.batch_size or 128)
    for batch in tqdm(list(batched(samples, batch_size)), desc="eval encoder", unit="batch"):
        tokens = bundle.tokenizer(
            [sample.input for sample in batch],
            padding=True,
            truncation=True,
            max_length=int(args.max_length),
            return_tensors="pt",
        )
        tokens = ensure_token_type_ids(dict(tokens))
        logits = model(move_batch(tokens, device))["logits"]
        top_indices = torch.topk(logits, k=k, dim=-1).indices.detach().cpu().tolist()
        for sample, indices in zip(batch, top_indices):
            candidates = [bundle.label2response[int(index)] for index in indices]
            records.append(prediction_record(sample, candidates, args.normalization_mode))
    return records


@torch.inference_mode()
def evaluate_generation(
    bundle: ModelBundle,
    samples: list[BenchmarkSample],
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, Any]]:
    model = bundle.model
    tokenizer = bundle.tokenizer
    model.eval()
    records: list[dict[str, Any]] = []
    nret = max(5, int(args.num_return_sequences))
    beams = max(int(args.num_beams), nret)
    batch_size = int(args.batch_size or 8)
    for batch in tqdm(list(batched(samples, batch_size)), desc="eval generation", unit="batch"):
        tokens = tokenizer(
            [sample.input for sample in batch],
            padding=True,
            truncation=True,
            max_length=int(args.max_length),
            return_tensors="pt",
        )
        tokens = move_batch(dict(tokens), device)
        generated = model.generate(
            **tokens,
            num_beams=beams,
            num_return_sequences=nret,
            max_new_tokens=int(args.max_new_tokens),
            length_penalty=float(args.length_penalty),
            early_stopping=bool(args.early_stopping),
        )
        if bundle.model_family == "decoder_only":
            input_lengths = tokens["attention_mask"].sum(dim=1).detach().cpu().tolist()
            decoded: list[str] = []
            for row_index in range(len(batch)):
                for seq_index in range(nret):
                    output_ids = generated[row_index * nret + seq_index]
                    continuation_ids = output_ids[int(input_lengths[row_index]) :]
                    decoded.append(tokenizer.decode(continuation_ids, skip_special_tokens=True).strip())
        else:
            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)

        for row_index, sample in enumerate(batch):
            candidates = [decoded[row_index * nret + seq_index].strip() for seq_index in range(nret)]
            records.append(prediction_record(sample, candidates, args.normalization_mode))
    return records


def evaluate_bundle(
    bundle: ModelBundle,
    samples: list[BenchmarkSample],
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, Any]]:
    if bundle.model_family == "encoder_only":
        return evaluate_encoder(bundle, samples, args, device)
    return evaluate_generation(bundle, samples, args, device)


def load_prompt_target_rows(path: str | Path) -> list[dict[str, str]]:
    rows = []
    for index, row in enumerate(read_table(path)):
        prompt = first_text(row, ("prompt", "anchor", "instruction", "input", "text", "query"))
        target = first_text(row, ("response", "target", "expected_response", "output", "answer", "positive"))
        if not prompt or not target:
            raise ValueError(f"{path}:{index} must contain prompt/input and response/target fields.")
        rows.append({"input": prompt, "target": target})
    return rows


def infer_recovery_train_path(args: argparse.Namespace, bundle: ModelBundle) -> str | None:
    if args.recovery_train_path:
        return str(args.recovery_train_path)
    metadata = bundle.metadata or {}
    config = metadata.get("config") if isinstance(metadata.get("config"), dict) else {}
    data_config = config.get("data") if isinstance(config.get("data"), dict) else {}
    train_json = data_config.get("train_json")
    if train_json:
        train_path = Path(str(train_json)).expanduser()
        if not train_path.is_absolute():
            train_path = PROJECT_ROOT / train_path
        if train_path.exists():
            return str(train_path)
    return None


def training_defaults(args: argparse.Namespace, bundle: ModelBundle) -> dict[str, Any]:
    metadata = bundle.metadata or {}
    config = metadata.get("config") if isinstance(metadata.get("config"), dict) else {}
    train_config = config.get("train") if isinstance(config.get("train"), dict) else {}
    return {
        "learning_rate": float(args.learning_rate or train_config.get("learning_rate", 2e-5)),
        "batch_size": int(args.batch_size or train_config.get("batch_size", 64)),
        "max_grad_norm": float(args.max_grad_norm or train_config.get("max_grad_norm", 1.0)),
        "optimizer": "AdamW",
    }


def train_encoder_epoch(
    bundle: ModelBundle,
    rows: list[dict[str, str]],
    optimizer: torch.optim.Optimizer,
    masks: dict[str, torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    train_config: dict[str, Any],
    seed: int,
) -> float:
    assert bundle.label2response is not None
    response_to_label = {response: index for index, response in enumerate(bundle.label2response)}
    examples = []
    for row in rows:
        if row["target"] not in response_to_label:
            raise ValueError(f"Recovery target not in checkpoint label space: {row['target']!r}")
        examples.append((row["input"], response_to_label[row["target"]]))
    random.Random(seed).shuffle(examples)
    bundle.model.train()
    losses: list[float] = []
    for batch in batched(examples, int(train_config["batch_size"])):
        prompts = [item[0] for item in batch]
        labels = torch.tensor([item[1] for item in batch], dtype=torch.long, device=device)
        tokens = bundle.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=int(args.max_length),
            return_tensors="pt",
        )
        tokens = ensure_token_type_ids(dict(tokens))
        optimizer.zero_grad(set_to_none=True)
        with autocast_for(device, dtype):
            output = bundle.model(move_batch(tokens, device), labels=labels)
            loss = output["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bundle.model.parameters(), float(train_config["max_grad_norm"]))
        optimizer.step()
        apply_masks(bundle.model, masks)
        losses.append(float(loss.detach().cpu()))
    return float(sum(losses) / len(losses)) if losses else 0.0


def _target_tokenize(tokenizer: Any, targets: list[str], max_length: int) -> dict[str, torch.Tensor]:
    try:
        return dict(
            tokenizer(
                text_target=targets,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
        )
    except TypeError:
        with tokenizer.as_target_tokenizer():
            return dict(
                tokenizer(
                    targets,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
            )


def train_encoder_decoder_epoch(
    bundle: ModelBundle,
    rows: list[dict[str, str]],
    optimizer: torch.optim.Optimizer,
    masks: dict[str, torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    train_config: dict[str, Any],
    seed: int,
) -> float:
    random.Random(seed).shuffle(rows)
    bundle.model.train()
    losses: list[float] = []
    for batch in batched(rows, int(train_config["batch_size"])):
        inputs = bundle.tokenizer(
            [row["input"] for row in batch],
            padding=True,
            truncation=True,
            max_length=int(args.max_length),
            return_tensors="pt",
        )
        labels = _target_tokenize(bundle.tokenizer, [row["target"] for row in batch], int(args.max_new_tokens))
        label_ids = labels["input_ids"]
        pad_id = bundle.tokenizer.pad_token_id
        if pad_id is not None:
            label_ids = label_ids.masked_fill(label_ids == int(pad_id), -100)
        optimizer.zero_grad(set_to_none=True)
        with autocast_for(device, dtype):
            output = bundle.model(**move_batch(dict(inputs), device), labels=label_ids.to(device))
            loss = output.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bundle.model.parameters(), float(train_config["max_grad_norm"]))
        optimizer.step()
        apply_masks(bundle.model, masks)
        losses.append(float(loss.detach().cpu()))
    return float(sum(losses) / len(losses)) if losses else 0.0


def train_decoder_epoch(
    bundle: ModelBundle,
    rows: list[dict[str, str]],
    optimizer: torch.optim.Optimizer,
    masks: dict[str, torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    train_config: dict[str, Any],
    seed: int,
) -> float:
    random.Random(seed).shuffle(rows)
    tokenizer = bundle.tokenizer
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("Decoder-only recovery requires tokenizer.pad_token_id or eos_token_id.")
    bundle.model.train()
    losses: list[float] = []
    for batch in batched(rows, int(train_config["batch_size"])):
        encoded_rows = []
        max_len = 0
        for row in batch:
            prompt_ids = tokenizer(row["input"], add_special_tokens=False)["input_ids"]
            target_text = row["target"] + (tokenizer.eos_token or "")
            target_ids = tokenizer(target_text, add_special_tokens=False)["input_ids"]
            input_ids = (prompt_ids + target_ids)[-int(args.max_length) :]
            labels = ([-100] * len(prompt_ids) + target_ids)[-int(args.max_length) :]
            max_len = max(max_len, len(input_ids))
            encoded_rows.append((input_ids, labels))

        input_tensor = torch.full((len(encoded_rows), max_len), int(pad_id), dtype=torch.long)
        label_tensor = torch.full((len(encoded_rows), max_len), -100, dtype=torch.long)
        attention_mask = torch.zeros((len(encoded_rows), max_len), dtype=torch.long)
        for row_index, (input_ids, labels) in enumerate(encoded_rows):
            input_tensor[row_index, : len(input_ids)] = torch.tensor(input_ids, dtype=torch.long)
            label_tensor[row_index, : len(labels)] = torch.tensor(labels, dtype=torch.long)
            attention_mask[row_index, : len(input_ids)] = 1

        optimizer.zero_grad(set_to_none=True)
        with autocast_for(device, dtype):
            output = bundle.model(
                input_ids=input_tensor.to(device),
                attention_mask=attention_mask.to(device),
                labels=label_tensor.to(device),
            )
            loss = output.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bundle.model.parameters(), float(train_config["max_grad_norm"]))
        optimizer.step()
        apply_masks(bundle.model, masks)
        losses.append(float(loss.detach().cpu()))
    return float(sum(losses) / len(losses)) if losses else 0.0


def train_recovery_epoch(
    bundle: ModelBundle,
    rows: list[dict[str, str]],
    optimizer: torch.optim.Optimizer,
    masks: dict[str, torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    train_config: dict[str, Any],
    seed: int,
) -> float:
    if bundle.model_family == "encoder_only":
        return train_encoder_epoch(bundle, rows, optimizer, masks, args, device, dtype, train_config, seed)
    if bundle.model_family == "encoder_decoder":
        return train_encoder_decoder_epoch(bundle, rows, optimizer, masks, args, device, dtype, train_config, seed)
    return train_decoder_epoch(bundle, rows, optimizer, masks, args, device, dtype, train_config, seed)


def validation_scores(
    bundle: ModelBundle,
    validation_samples: list[BenchmarkSample] | None,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[float | None, float | None]:
    if not validation_samples:
        return None, None
    records = evaluate_bundle(bundle, validation_samples, args, device)
    metrics = compute_metric_breakdown(
        records,
        bootstrap_resamples=max(10, min(100, int(args.bootstrap_resamples))),
        seed=int(args.seed),
    )
    return metrics["overall"]["em1"], metrics["overall"]["em5"]


def maybe_reinitialize_encoder_classifier(
    bundle: ModelBundle,
    args: argparse.Namespace,
    device: torch.device,
) -> bool:
    if not bool(args.reinitialize_classifier_from_responses):
        return False
    if bundle.model_family != "encoder_only":
        raise ValueError("--reinitialize_classifier_from_responses is only supported for encoder_only.")
    if bool(args.prune_output_heads):
        raise ValueError(
            "--reinitialize_classifier_from_responses cannot be combined with --prune_output_heads, "
            "because it would overwrite a pruned output/classifier head."
        )
    assert bundle.label2response is not None
    initialize_classifier_from_responses(
        model=bundle.model,  # type: ignore[arg-type]
        tokenizer=bundle.tokenizer,
        label2response=bundle.label2response,
        device=device,
        max_length=int(args.classifier_init_max_length),
        batch_size=int(args.classifier_init_batch_size),
    )
    bundle.model.eval()
    return True


def augment_prediction_rows(
    records: list[dict[str, Any]],
    args: argparse.Namespace,
    pruning_mode: str,
    target_sparsity: float,
    stats: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        row = dict(record)
        row.update(
            {
                "model_family": normalize_model_family(args.model_family),
                "pruning_mode": pruning_mode,
                "pruning_method": args.prune_method,
                "target_sparsity": float(target_sparsity),
                "targeted_linear_sparsity_actual": stats["targeted_linear_sparsity_actual"],
                "whole_model_sparsity_actual": stats["whole_model_sparsity_actual"],
                "seed": int(args.seed),
            }
        )
        rows.append(row)
    return rows


def summary_row(
    args: argparse.Namespace,
    pruning_mode: str,
    target_sparsity: float,
    stats: dict[str, Any],
    records: list[dict[str, Any]],
    checkpoint_path: str,
    mask_path: str,
    predictions_path: str,
    training_config: dict[str, Any],
    pruning_config: dict[str, Any],
) -> dict[str, Any]:
    metrics = compute_metric_breakdown(
        records,
        bootstrap_resamples=int(args.bootstrap_resamples),
        seed=int(args.seed),
    )
    row: dict[str, Any] = {
        "experiment_name": args.experiment_name,
        "model_family": normalize_model_family(args.model_family),
        "pruning_mode": pruning_mode,
        "pruning_method": args.prune_method,
        "target_sparsity": float(target_sparsity),
        "targeted_linear_sparsity_actual": stats["targeted_linear_sparsity_actual"],
        "whole_model_sparsity_actual": stats["whole_model_sparsity_actual"],
        "seed": int(args.seed),
        "em1_overall": metrics["overall"]["em1"],
        "em1_overall_ci_low": metrics["overall"]["em1_ci_low"],
        "em1_overall_ci_high": metrics["overall"]["em1_ci_high"],
        "em5_overall": metrics["overall"]["em5"],
        "em5_overall_ci_low": metrics["overall"]["em5_ci_low"],
        "em5_overall_ci_high": metrics["overall"]["em5_ci_high"],
        "count_total": metrics["overall"]["count"],
        "decoding_config_json": json_dumps(
            {
                "num_beams": max(int(args.num_beams), max(5, int(args.num_return_sequences)))
                if normalize_model_family(args.model_family) != "encoder_only"
                else None,
                "num_return_sequences": max(5, int(args.num_return_sequences)),
                "max_new_tokens": int(args.max_new_tokens),
                "length_penalty": float(args.length_penalty),
                "early_stopping": bool(args.early_stopping),
                "normalization_mode": args.normalization_mode,
            }
        ),
        "training_config_json": json_dumps(training_config),
        "pruning_config_json": json_dumps(pruning_config),
        "checkpoint_path": checkpoint_path,
        "mask_path": mask_path,
        "predictions_path": predictions_path,
    }
    for difficulty in ("easy", "medium", "hard"):
        group = metrics["difficulty"][difficulty]
        row[f"em1_{difficulty}"] = group["em1"]
        row[f"em1_{difficulty}_ci_low"] = group["em1_ci_low"]
        row[f"em1_{difficulty}_ci_high"] = group["em1_ci_high"]
        row[f"em1_{difficulty}_ci_status"] = group["em1_ci_status"]
        row[f"em5_{difficulty}"] = group["em5"]
        row[f"em5_{difficulty}_ci_low"] = group["em5_ci_low"]
        row[f"em5_{difficulty}_ci_high"] = group["em5_ci_high"]
        row[f"em5_{difficulty}_ci_status"] = group["em5_ci_status"]
        row[f"count_{difficulty}"] = group["count"]
    return row


def fill_retention(rows: list[dict[str, Any]]) -> None:
    dense_by_key = {
        (row["model_family"], row["seed"]): row
        for row in rows
        if row.get("pruning_mode") == "dense" and math.isclose(float(row.get("target_sparsity", 0)), 0.0)
    }
    for row in rows:
        dense = dense_by_key.get((row["model_family"], row["seed"]))
        if not dense:
            continue
        row["em1_retention_overall"] = retention(row.get("em1_overall"), dense.get("em1_overall"))
        row["em5_retention_overall"] = retention(row.get("em5_overall"), dense.get("em5_overall"))
        for difficulty in ("easy", "medium", "hard"):
            row[f"em1_retention_{difficulty}"] = retention(
                row.get(f"em1_{difficulty}"), dense.get(f"em1_{difficulty}")
            )
            row[f"em5_retention_{difficulty}"] = retention(
                row.get(f"em5_{difficulty}"), dense.get(f"em5_{difficulty}")
            )


def write_paper_table(path: Path, rows: list[dict[str, Any]]) -> None:
    paper_rows = [
        {
            "model_family": row.get("model_family"),
            "pruning_mode": row.get("pruning_mode"),
            "target_sparsity": row.get("target_sparsity"),
            "overall EM@1": row.get("em1_overall"),
            "overall EM@5": row.get("em5_overall"),
            "easy EM@1": row.get("em1_easy"),
            "easy EM@5": row.get("em5_easy"),
            "medium EM@1": row.get("em1_medium"),
            "medium EM@5": row.get("em5_medium"),
            "hard EM@1": row.get("em1_hard"),
            "hard EM@5": row.get("em5_hard"),
            "targeted linear sparsity": row.get("targeted_linear_sparsity_actual"),
            "whole-model sparsity": row.get("whole_model_sparsity_actual"),
        }
        for row in rows
    ]
    write_rows_csv(path, paper_rows, PAPER_FIELDNAMES)


def run_condition(
    args: argparse.Namespace,
    model_family: str,
    pruning_mode: str,
    target_sparsity: float,
    samples: list[BenchmarkSample],
    validation_samples: list[BenchmarkSample] | None,
    output_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    print(
        f"\n[sparsity] model={model_family} mode={pruning_mode} target={target_sparsity:.2f}",
        flush=True,
    )
    bundle = load_bundle(model_family, args.model_checkpoint, device, dtype)
    sparsity_config = LinearSparsityConfig(
        prune_output_heads=bool(args.prune_output_heads),
        global_pruning=bool(args.global_pruning),
        regrowth=bool(args.regrowth),
    )
    training_config = training_defaults(args, bundle)
    pruning_config = {
        "prune_scope": args.prune_scope,
        "prune_method": args.prune_method,
        "prune_output_heads": bool(args.prune_output_heads),
        "global_pruning": bool(args.global_pruning),
        "regrowth": bool(args.regrowth),
        "progressive_schedule": args.progressive_schedule,
        "recovery_epochs_per_stage": int(args.recovery_epochs_per_stage),
        "final_recovery_epochs": int(args.final_recovery_epochs),
        "classifier_reinitialized_after_pruning": bool(args.reinitialize_classifier_from_responses),
        "classifier_init_batch_size": (
            int(args.classifier_init_batch_size)
            if bool(args.reinitialize_classifier_from_responses)
            else None
        ),
        "classifier_init_max_length": (
            int(args.classifier_init_max_length)
            if bool(args.reinitialize_classifier_from_responses)
            else None
        ),
    }

    label = sparsity_label(target_sparsity)
    checkpoint_path = str(Path(args.model_checkpoint).expanduser())
    mask_path = ""
    progressive_logs: list[dict[str, Any]] = []
    masks: dict[str, torch.Tensor] = {}

    if pruning_mode == "dense":
        stats = current_linear_sparsity_summary(bundle.model, sparsity_config, target_sparsity=0.0)
    elif pruning_mode == "oneshot":
        masks, stats = apply_magnitude_pruning(
            bundle.model,
            sparsity=target_sparsity,
            config=sparsity_config,
        )
        maybe_reinitialize_encoder_classifier(bundle, args, device)
        stats = current_linear_sparsity_summary(bundle.model, sparsity_config, target_sparsity)
        mask_file = output_dir / "masks" / f"masks_{model_family}_{pruning_mode}_{label}_{args.seed}.pt"
        save_masks(mask_file, masks, {**pruning_config, **stats})
        mask_path = str(mask_file)
        checkpoint_dir = output_dir / "checkpoints" / f"{model_family}_{pruning_mode}_{label}_{args.seed}"
        checkpoint_path = str(checkpoint_dir)
    elif pruning_mode == "progressive":
        train_path = infer_recovery_train_path(args, bundle)
        if not train_path:
            raise ValueError(
                "Progressive recovery fine-tuning needs --recovery_train_path, or an encoder checkpoint "
                "with scenic_sft_metadata.json config.data.train_json."
            )
        train_rows = load_prompt_target_rows(train_path)
        optimizer = torch.optim.AdamW(bundle.model.parameters(), lr=float(training_config["learning_rate"]))
        hooks: list[Any] = []
        try:
            for stage_index, stage_sparsity in enumerate(stage_schedule(target_sparsity), start=1):
                if hooks:
                    remove_hooks(hooks)
                masks, stats = apply_magnitude_pruning(
                    bundle.model,
                    sparsity=stage_sparsity,
                    config=sparsity_config,
                    masks=masks or None,
                )
                hooks = register_mask_gradient_hooks(bundle.model, masks)
                if int(args.recovery_epochs_per_stage) <= 0:
                    val_em1, val_em5 = validation_scores(bundle, validation_samples, args, device)
                    progressive_logs.append(
                        {
                            "stage": stage_index,
                            "stage_target_sparsity": stage_sparsity,
                            "targeted_linear_sparsity_actual": stats["targeted_linear_sparsity_actual"],
                            "whole_model_sparsity_actual": stats["whole_model_sparsity_actual"],
                            "recovery_epoch": 0,
                            "train_loss": None,
                            "val_em1": val_em1,
                            "val_em5": val_em5,
                        }
                    )
                for recovery_epoch in range(1, int(args.recovery_epochs_per_stage) + 1):
                    train_loss = train_recovery_epoch(
                        bundle,
                        train_rows,
                        optimizer,
                        masks,
                        args,
                        device,
                        dtype,
                        training_config,
                        seed=int(args.seed) + stage_index * 100 + recovery_epoch,
                    )
                    val_em1, val_em5 = validation_scores(bundle, validation_samples, args, device)
                    stats = current_linear_sparsity_summary(bundle.model, sparsity_config, stage_sparsity)
                    progressive_logs.append(
                        {
                            "stage": stage_index,
                            "stage_target_sparsity": stage_sparsity,
                            "targeted_linear_sparsity_actual": stats["targeted_linear_sparsity_actual"],
                            "whole_model_sparsity_actual": stats["whole_model_sparsity_actual"],
                            "recovery_epoch": recovery_epoch,
                            "train_loss": train_loss,
                            "val_em1": val_em1,
                            "val_em5": val_em5,
                        }
                    )

            for final_epoch in range(1, int(args.final_recovery_epochs) + 1):
                train_loss = train_recovery_epoch(
                    bundle,
                    train_rows,
                    optimizer,
                    masks,
                    args,
                    device,
                    dtype,
                    training_config,
                    seed=int(args.seed) + 10_000 + final_epoch,
                )
                val_em1, val_em5 = validation_scores(bundle, validation_samples, args, device)
                stats = current_linear_sparsity_summary(bundle.model, sparsity_config, target_sparsity)
                progressive_logs.append(
                    {
                        "stage": "final",
                        "stage_target_sparsity": target_sparsity,
                        "targeted_linear_sparsity_actual": stats["targeted_linear_sparsity_actual"],
                        "whole_model_sparsity_actual": stats["whole_model_sparsity_actual"],
                        "recovery_epoch": final_epoch,
                        "train_loss": train_loss,
                        "val_em1": val_em1,
                        "val_em5": val_em5,
                    }
                )
        finally:
            remove_hooks(hooks)

        maybe_reinitialize_encoder_classifier(bundle, args, device)
        stats = current_linear_sparsity_summary(bundle.model, sparsity_config, target_sparsity)
        mask_file = output_dir / "masks" / f"masks_{model_family}_{pruning_mode}_{label}_{args.seed}.pt"
        save_masks(mask_file, masks, {**pruning_config, **stats})
        mask_path = str(mask_file)
        checkpoint_dir = output_dir / "checkpoints" / f"{model_family}_{pruning_mode}_{label}_{args.seed}"
        checkpoint_path = str(checkpoint_dir)
        log_path = output_dir / f"progressive_logs_{model_family}_{label}_{args.seed}.csv"
        write_rows_csv(log_path, progressive_logs, PROGRESSIVE_FIELDNAMES)
    else:
        raise ValueError(f"Unsupported pruning_mode: {pruning_mode}")

    records = evaluate_bundle(bundle, samples, args, device)
    prediction_rows = augment_prediction_rows(records, args, pruning_mode, target_sparsity, stats)
    predictions_path = output_dir / f"predictions_{model_family}_{pruning_mode}_{label}_{args.seed}.csv"
    write_prediction_csv(predictions_path, prediction_rows, PREDICTION_FIELDNAMES)

    if pruning_mode in {"oneshot", "progressive"}:
        metadata = {
            "experiment_name": args.experiment_name,
            "input_checkpoint": str(Path(args.model_checkpoint).expanduser()),
            "model_family": model_family,
            "pruning_mode": pruning_mode,
            "target_sparsity": float(target_sparsity),
            "training_config": training_config,
            "pruning_config": pruning_config,
            "sparsity": stats,
        }
        save_bundle_checkpoint(bundle, Path(checkpoint_path), metadata)

    row = summary_row(
        args=args,
        pruning_mode=pruning_mode,
        target_sparsity=target_sparsity,
        stats=stats,
        records=records,
        checkpoint_path=checkpoint_path,
        mask_path=mask_path,
        predictions_path=str(predictions_path),
        training_config=training_config,
        pruning_config=pruning_config,
    )
    print(
        f"[sparsity] EM@1={row['em1_overall']:.4f} EM@5={row['em5_overall']:.4f} "
        f"targeted={row['targeted_linear_sparsity_actual']:.4f} whole={row['whole_model_sparsity_actual']:.4f}",
        flush=True,
    )
    return row


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    model_family = normalize_model_family(args.model_family)
    output_dir = Path(args.output_dir or (PROJECT_ROOT / "results" / args.experiment_name)).expanduser()
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = select_device(args.device)
    dtype = dtype_for(args.dtype, device)
    samples = load_benchmark_samples(args.benchmark_path, args.benchmark_difficulty_path)
    validation_samples = (
        load_benchmark_samples(args.validation_path, args.benchmark_difficulty_path)
        if args.validation_path
        else None
    )

    rows: list[dict[str, Any]] = []
    for pruning_mode, target_sparsity in run_plan(args.sparsity_levels, args.pruning_modes):
        rows.append(
            run_condition(
                args=args,
                model_family=model_family,
                pruning_mode=pruning_mode,
                target_sparsity=target_sparsity,
                samples=samples,
                validation_samples=validation_samples,
                output_dir=output_dir,
                device=device,
                dtype=dtype,
            )
        )

    fill_retention(rows)
    summary_path = output_dir / "summary_metrics.csv"
    write_rows_csv(summary_path, rows, SUMMARY_FIELDNAMES)
    paper_path = output_dir / "paper_table_sparsity_difficulty.csv"
    write_paper_table(paper_path, rows)

    print("\n[sparsity] summary")
    print(f"{'mode':12s} {'sparsity':>9s} {'EM@1':>9s} {'EM@5':>9s} {'targeted':>10s} {'whole':>10s}")
    for row in rows:
        print(
            f"{row['pruning_mode']:12s} {float(row['target_sparsity']):9.2f} "
            f"{float(row['em1_overall']):9.4f} {float(row['em5_overall']):9.4f} "
            f"{float(row['targeted_linear_sparsity_actual']):10.4f} "
            f"{float(row['whole_model_sparsity_actual']):10.4f}"
        )
    print(f"\n[sparsity] wrote {summary_path}")
    print(f"[sparsity] wrote {paper_path}")
    print("[sparsity] run plots with:")
    print(f"python scripts/plot_sparsity_results.py --experiment_name {args.experiment_name} --results_dir {output_dir}")


if __name__ == "__main__":
    main()
