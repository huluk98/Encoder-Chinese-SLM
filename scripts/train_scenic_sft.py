#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler, RandomSampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from chatlm_encoder.scenic_sft import (  # noqa: E402
    ScenicEncoderForResponseSelection,
    build_label_maps,
    load_base_scenic_model,
    load_contrastive_rows,
    load_prompt_response_rows,
    save_scenic_checkpoint,
)


class PromptResponseDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], response_to_label: dict[str, int]) -> None:
        self.items = [(row["prompt"], response_to_label[row["response"]]) for row in rows]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[str, int]:
        return self.items[index]


class ContrastiveDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, str]:
        return self.rows[index]


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    config["_config_path"] = str(config_path)
    return config


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_distributed() -> tuple[torch.device, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed SCENIC SFT requires CUDA.")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return torch.device("cuda", local_rank), rank, local_rank, world_size
    if torch.cuda.is_available():
        return torch.device("cuda"), rank, local_rank, world_size
    return torch.device("cpu"), rank, local_rank, world_size


def is_main(rank: int) -> bool:
    return rank == 0


def unwrap(model: torch.nn.Module) -> ScenicEncoderForResponseSelection:
    while hasattr(model, "module"):
        model = model.module
    return model  # type: ignore[return-value]


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def autocast_for(device: torch.device, precision: str):
    if device.type == "cuda" and precision in {"bf16", "fp16"}:
        dtype = torch.bfloat16 if precision == "bf16" else torch.float16
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def ensure_token_type_ids(encoded: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if "token_type_ids" not in encoded:
        encoded["token_type_ids"] = torch.zeros_like(encoded["input_ids"])
    return encoded


def lr_for_step(step: int, total_steps: int, config: dict[str, Any]) -> float:
    max_lr = float(config.get("learning_rate", 2e-5))
    min_lr = float(config.get("min_learning_rate", max_lr * 0.1))
    warmup_steps = int(config.get("warmup_steps") or round(total_steps * float(config.get("warmup_ratio", 0.06))))
    if warmup_steps > 0 and step < warmup_steps:
        return max_lr * float(step + 1) / float(warmup_steps)
    decay_steps = max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, float(step - warmup_steps) / float(decay_steps)))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + cosine * (max_lr - min_lr)


def collate_prompt_response(tokenizer: Any, max_length: int):
    def _collate(batch: list[tuple[str, int]]) -> dict[str, Any]:
        texts, labels = zip(*batch)
        encoded = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = ensure_token_type_ids(dict(encoded))
        return {"tokens": encoded, "labels": torch.tensor(labels, dtype=torch.long)}

    return _collate


def collate_contrastive(tokenizer: Any, max_length: int):
    def _collate(batch: list[dict[str, str]]) -> dict[str, Any]:
        anchors = [item["anchor"] for item in batch]
        positives = [item["positive"] for item in batch]
        negatives = [item["negative"] for item in batch]
        invalid_negatives = [item.get("invalid_negative", item["negative"]) for item in batch]
        texts = anchors + positives + negatives + invalid_negatives
        encoded = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        encoded = ensure_token_type_ids(dict(encoded))
        return {
            "tokens": encoded,
            "batch_size": len(batch),
        }

    return _collate


def endless(loader: DataLoader) -> Iterator[Any]:
    while True:
        yield from loader


def contrastive_loss(
    model: torch.nn.Module,
    batch: dict[str, Any],
    device: torch.device,
    margin: float,
) -> torch.Tensor:
    size = int(batch["batch_size"])
    embeddings = torch.nn.functional.normalize(model(move_batch(batch["tokens"], device))["embeddings"], dim=-1)
    anchor, positive, negative, invalid_negative = embeddings.split(size, dim=0)
    valid_loss = torch.nn.functional.triplet_margin_loss(anchor, positive, negative, margin=float(margin))
    invalid_loss = torch.nn.functional.triplet_margin_loss(anchor, positive, invalid_negative, margin=float(margin))
    return 0.5 * (valid_loss + invalid_loss)


def save_latest(
    model: torch.nn.Module,
    tokenizer: Any,
    output_dir: Path,
    label2response: list[str],
    metadata: dict[str, Any],
    step: int,
) -> None:
    checkpoint_dir = output_dir / f"step-{step:06d}"
    latest_dir = output_dir / "latest"
    save_scenic_checkpoint(unwrap(model), tokenizer, checkpoint_dir, label2response, metadata)
    if latest_dir.is_symlink() or latest_dir.exists():
        if latest_dir.is_dir() and not latest_dir.is_symlink():
            import shutil

            shutil.rmtree(latest_dir)
        else:
            latest_dir.unlink()
    try:
        latest_dir.symlink_to(os.path.relpath(checkpoint_dir, start=output_dir), target_is_directory=True)
    except OSError:
        import shutil

        shutil.copytree(checkpoint_dir, latest_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="8-GPU encoder-only SCENIC supervised fine-tuning.")
    parser.add_argument("--config", default="configs/scenic_sft_8gpu.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    run_config = config.get("run", {})
    model_config = config.get("model", {})
    data_config = config.get("data", {})
    train_config = config.get("train", {})

    set_seed(int(run_config.get("seed", 42)))
    device, rank, _local_rank, world_size = setup_distributed()
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    rows = load_prompt_response_rows(data_config["train_json"])
    response_to_label, label2response = build_label_maps(rows)
    contrastive_json = data_config.get("contrastive_json")
    contrast_rows = load_contrastive_rows(contrastive_json) if contrastive_json else []
    if is_main(rank):
        print(f"[scenic-sft] rows={len(rows):,} labels={len(label2response):,} contrastive_rows={len(contrast_rows):,}")

    base_model_path = Path(model_config["base_model"]).expanduser()
    if not base_model_path.exists():
        raise FileNotFoundError(
            f"Base encoder checkpoint not found: {base_model_path}. "
            "Finish MLM pretraining first or set model.base_model in configs/scenic_sft_8gpu.yaml."
        )

    model, tokenizer = load_base_scenic_model(
        base_model=base_model_path,
        tokenizer_path=model_config.get("tokenizer_path"),
        num_labels=len(label2response),
        dropout=float(model_config.get("dropout", 0.1)),
    )
    model.to(device)
    if world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[device.index],
            output_device=device.index,
            broadcast_buffers=False,
        )

    max_length = int(data_config.get("max_length", 128))
    train_dataset = PromptResponseDataset(rows, response_to_label)
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if world_size > 1 else RandomSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(train_config.get("batch_size", 64)),
        sampler=train_sampler,
        num_workers=int(train_config.get("num_workers", 0)),
        pin_memory=bool(train_config.get("pin_memory", False)),
        persistent_workers=bool(train_config.get("persistent_workers", False)) and int(train_config.get("num_workers", 0)) > 0,
        collate_fn=collate_prompt_response(tokenizer, max_length),
    )

    contrast_loader = None
    contrast_sampler = None
    if contrast_rows and float(train_config.get("contrastive_weight", 0.0)) > 0:
        contrast_dataset = ContrastiveDataset(contrast_rows)
        contrast_sampler = DistributedSampler(contrast_dataset, shuffle=True) if world_size > 1 else RandomSampler(contrast_dataset)
        contrast_loader = DataLoader(
            contrast_dataset,
            batch_size=int(train_config.get("contrastive_batch_size", train_config.get("batch_size", 64))),
            sampler=contrast_sampler,
            num_workers=int(train_config.get("num_workers", 0)),
            pin_memory=bool(train_config.get("pin_memory", False)),
            persistent_workers=bool(train_config.get("persistent_workers", False)) and int(train_config.get("num_workers", 0)) > 0,
            collate_fn=collate_contrastive(tokenizer, max_length),
        )

    epochs = int(train_config.get("epochs", 3))
    grad_accum_steps = int(train_config.get("grad_accum_steps", 1))
    epoch_steps = max(1, math.ceil(len(train_loader) / grad_accum_steps))
    configured_max_steps = train_config.get("max_steps")
    total_steps = int(configured_max_steps) if configured_max_steps else max(1, epochs * epoch_steps)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_config.get("learning_rate", 2e-5)),
        weight_decay=float(train_config.get("weight_decay", 0.01)),
        eps=float(train_config.get("adam_eps", 1e-8)),
    )
    precision = str(train_config.get("precision", "bf16")).lower()
    contrast_iter = endless(contrast_loader) if contrast_loader is not None else None
    contrastive_weight = float(train_config.get("contrastive_weight", 0.1))
    triplet_margin = float(train_config.get("triplet_margin", 0.25))
    log_every = int(train_config.get("log_every", 10))
    save_every = int(train_config.get("save_every", 100))
    output_dir = Path(run_config.get("output_dir", "runs/scenic-sft")).expanduser()
    if is_main(rank):
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[scenic-sft] output={output_dir}")
        print(f"[scenic-sft] world_size={world_size} total_steps={total_steps:,} batch_per_gpu={int(train_config.get('batch_size', 64))}")

    optimizer.zero_grad(set_to_none=True)
    model.train()
    optimizer_step = 0
    start_time = time.time()
    stop_training = False

    for epoch in range(epochs):
        if hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)
        if contrast_sampler is not None and hasattr(contrast_sampler, "set_epoch"):
            contrast_sampler.set_epoch(epoch)

        for micro_step, batch in enumerate(train_loader, start=1):
            lr = lr_for_step(optimizer_step, total_steps, train_config)
            for group in optimizer.param_groups:
                group["lr"] = lr

            tokens = move_batch(batch["tokens"], device)
            labels = batch["labels"].to(device, non_blocking=True)
            with autocast_for(device, precision):
                output = model(tokens, labels=labels)
                loss = output["loss"]
                clf_loss = loss.detach()
                cont_loss_value = torch.tensor(0.0, device=device)
                if contrast_iter is not None and contrastive_weight > 0:
                    cont_batch = next(contrast_iter)
                    cont_loss = contrastive_loss(model, cont_batch, device, triplet_margin)
                    cont_loss_value = cont_loss.detach()
                    loss = loss + contrastive_weight * cont_loss
                scaled_loss = loss / grad_accum_steps

            scaled_loss.backward()
            if micro_step % grad_accum_steps != 0:
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_config.get("max_grad_norm", 1.0)))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1

            if is_main(rank) and (optimizer_step == 1 or optimizer_step % log_every == 0):
                elapsed = max(1e-6, time.time() - start_time)
                with torch.no_grad():
                    predictions = output["logits"].argmax(dim=-1)
                    accuracy = (predictions == labels).float().mean().item()
                print(
                    f"[scenic-sft] step={optimizer_step:,}/{total_steps:,} "
                    f"epoch={epoch + 1}/{epochs} lr={lr:.3e} "
                    f"loss={float(loss.detach()):.4f} clf={float(clf_loss):.4f} "
                    f"contrastive={float(cont_loss_value):.4f} acc={accuracy:.4f} "
                    f"steps_per_sec={optimizer_step / elapsed:.3f}"
                )

            if is_main(rank) and save_every > 0 and optimizer_step % save_every == 0:
                save_latest(
                    model,
                    tokenizer,
                    output_dir,
                    label2response,
                    {
                        "step": optimizer_step,
                        "config": config,
                        "train_rows": len(rows),
                        "contrastive_rows": len(contrast_rows),
                    },
                    optimizer_step,
                )
                print(f"[scenic-sft] saved checkpoint at step {optimizer_step:,}")

            if optimizer_step >= total_steps:
                stop_training = True
                break

        if stop_training:
            break

    if is_main(rank):
        save_latest(
            model,
            tokenizer,
            output_dir,
            label2response,
            {
                "step": optimizer_step,
                "config": config,
                "train_rows": len(rows),
                "contrastive_rows": len(contrast_rows),
                "finished": True,
            },
            max(optimizer_step, 1),
        )
        print(f"[scenic-sft] done | latest={output_dir / 'latest'}")

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
