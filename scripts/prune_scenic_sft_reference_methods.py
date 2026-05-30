#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from chatlm_encoder.scenic_sft import (  # noqa: E402
    load_scenic_checkpoint,
    prompt_from_row,
    read_json_list,
    save_scenic_checkpoint,
)


CHECKPOINT_DIR = "runs/scenic-sft-training-dataset/latest"
OUTPUT_DIR = "runs/scenic-sft-training-dataset-pruned50-reference"
CALIBRATION_JSON = "data/scenic/SCENIC_full_training_dataset.json"
SPARSITY = 0.5
MAX_LENGTH = 128
CALIBRATION_BATCH_SIZE = 4
CALIBRATION_BATCHES = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prune a SCENIC encoder SFT checkpoint with reference T5-style methods: "
            "unstructured magnitude, NVIDIA 2:4, WANDA, or gradient/Taylor pruning."
        )
    )
    parser.add_argument("--checkpoint", default=CHECKPOINT_DIR, help="Input SCENIC SFT checkpoint directory.")
    parser.add_argument("--output", default=OUTPUT_DIR, help="Output directory for the pruned checkpoint.")
    parser.add_argument("--method", required=True, choices=("magnitude", "nvidia", "wanda", "gradient"))
    parser.add_argument("--sparsity", type=float, default=SPARSITY, help="Target sparsity. NVIDIA requires 0.5.")
    parser.add_argument(
        "--scope",
        default="all-linear",
        choices=("encoder-linear", "all-linear"),
        help=(
            "encoder-linear prunes encoder nn.Linear weights only; "
            "all-linear prunes every nn.Linear weight, including the response classifier unless excluded."
        ),
    )
    parser.add_argument(
        "--exclude-classifier",
        action="store_true",
        help="Leave the SCENIC response classifier dense. By default it is pruned to match the T5 scripts' all-linear behavior.",
    )
    parser.add_argument("--calibration-json", default=CALIBRATION_JSON, help="Prompt JSON used for WANDA activation calibration.")
    parser.add_argument("--calibration-batch-size", type=int, default=CALIBRATION_BATCH_SIZE)
    parser.add_argument("--calibration-batches", type=int, default=CALIBRATION_BATCHES)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0, mps, or cpu.")
    parser.add_argument("--dtype", default="fp32", choices=("fp32", "bf16", "fp16"))
    parser.add_argument("--overwrite", action="store_true", help="Replace the output directory if it exists.")
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def dtype_for(name: str, device: torch.device) -> torch.dtype:
    if name == "bf16" and device.type == "cuda":
        return torch.bfloat16
    if name == "fp16" and device.type == "cuda":
        return torch.float16
    return torch.float32


def read_metadata(checkpoint_dir: Path) -> dict[str, Any]:
    metadata_path = checkpoint_dir / "scenic_sft_metadata.json"
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def count_zeros(tensor: torch.Tensor) -> int:
    return int((tensor.detach() == 0).sum().item())


def parameter_totals(parameters: Iterable[tuple[str, torch.nn.Parameter]]) -> dict[str, int]:
    numel = 0
    zeros = 0
    for _, parameter in parameters:
        tensor = parameter.detach()
        numel += int(tensor.numel())
        zeros += count_zeros(tensor)
    return {"numel": numel, "zeros": zeros}


def selected_linear_modules(
    model: nn.Module,
    scope: str,
    include_classifier: bool,
) -> list[tuple[str, nn.Linear]]:
    modules: list[tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        normalized = name.lower()
        is_classifier = normalized == "classifier" or normalized.startswith("classifier.")
        if is_classifier and not include_classifier:
            continue
        if scope == "encoder-linear" and not normalized.startswith("encoder."):
            continue
        modules.append((name, module))
    return modules


def module_stats(name: str, module: nn.Linear, skipped_reason: str | None = None) -> dict[str, Any]:
    weight = module.weight.detach()
    numel = int(weight.numel())
    zeros = count_zeros(weight)
    stats: dict[str, Any] = {
        "name": name,
        "shape": list(weight.shape),
        "numel": numel,
        "zeros": zeros,
        "sparsity": zeros / numel if numel else 0.0,
    }
    if skipped_reason:
        stats["skipped_reason"] = skipped_reason
    return stats


def magnitude_prune_(module: nn.Linear, sparsity: float) -> str | None:
    weight = module.weight.data
    scores = weight.detach().abs().float().reshape(-1)
    keep_count = int(scores.numel() * (1.0 - sparsity))
    if keep_count <= 0:
        weight.zero_()
        return None
    if keep_count >= scores.numel():
        return None
    threshold = torch.topk(scores, keep_count).values.min().to(weight.device)
    mask = weight.detach().abs() >= threshold
    weight.mul_(mask)
    return None


def nvidia_2_4_prune_(module: nn.Linear) -> str | None:
    weight = module.weight.data
    if weight.shape[1] % 4 != 0:
        return f"in_features={weight.shape[1]} is not divisible by 4"
    grouped = weight.view(weight.shape[0], -1, 4)
    _, prune_indices = torch.topk(grouped.detach().abs().float(), 2, dim=2, largest=False)
    mask = torch.ones_like(grouped)
    mask.scatter_(2, prune_indices.to(grouped.device), 0)
    grouped.mul_(mask)
    return None


def calibration_texts(path: str | Path) -> list[str]:
    texts: list[str] = []
    for row in read_json_list(path):
        prompt = prompt_from_row(row)
        if prompt:
            texts.append(prompt)
    if not texts:
        raise ValueError(f"No prompt-like rows found in calibration JSON: {path}")
    return texts


def calibration_prompt_label_rows(path: str | Path, label2response: list[str]) -> list[tuple[str, int]]:
    response_to_label = {response: index for index, response in enumerate(label2response)}
    rows: list[tuple[str, int]] = []
    for row in read_json_list(path):
        prompt = prompt_from_row(row)
        response = "" if row.get("response") is None else str(row.get("response")).strip()
        if prompt and response in response_to_label:
            rows.append((prompt, response_to_label[response]))
    if not rows:
        raise ValueError(f"No prompt/response rows matching checkpoint labels found in calibration JSON: {path}")
    return rows


def ensure_token_type_ids(encoded: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if "token_type_ids" not in encoded:
        encoded["token_type_ids"] = torch.zeros_like(encoded["input_ids"])
    return encoded


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


@torch.inference_mode()
def collect_activation_norms(
    model: nn.Module,
    tokenizer: Any,
    modules: list[tuple[str, nn.Linear]],
    texts: list[str],
    device: torch.device,
    max_length: int,
    batch_size: int,
    calibration_batches: int,
) -> dict[str, torch.Tensor]:
    square_sums: dict[str, torch.Tensor | None] = {name: None for name, _ in modules}
    handles: list[Any] = []

    def make_hook(name: str):
        def hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...], _output: Any) -> None:
            if not inputs:
                return
            activations = inputs[0].detach()
            if activations.numel() == 0:
                return
            activations = activations.reshape(-1, activations.shape[-1]).float()
            values = activations.pow(2).sum(dim=0).cpu()
            previous = square_sums[name]
            square_sums[name] = values if previous is None else previous + values

        return hook

    for name, module in modules:
        handles.append(module.register_forward_hook(make_hook(name)))

    model.eval()
    used_batches = 0
    try:
        for start in tqdm(range(0, len(texts), batch_size), desc="wanda calibration", unit="batch"):
            if used_batches >= calibration_batches:
                break
            batch_texts = texts[start : start + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = ensure_token_type_ids(dict(encoded))
            model(move_batch(encoded, device))
            used_batches += 1
    finally:
        for handle in handles:
            handle.remove()

    norms: dict[str, torch.Tensor] = {}
    for name, module in modules:
        values = square_sums[name]
        if values is None:
            values = torch.zeros(module.weight.shape[1], dtype=torch.float32)
        norms[name] = values.sqrt()
    return norms


def collect_gradient_saliency(
    model: nn.Module,
    tokenizer: Any,
    modules: list[tuple[str, nn.Linear]],
    rows: list[tuple[str, int]],
    device: torch.device,
    max_length: int,
    batch_size: int,
    calibration_batches: int,
) -> dict[str, torch.Tensor]:
    saliency = {name: torch.zeros_like(module.weight.data, dtype=torch.float32) for name, module in modules}
    model.train()
    used_batches = 0
    for start in tqdm(range(0, len(rows), batch_size), desc="gradient calibration", unit="batch"):
        if used_batches >= calibration_batches:
            break
        batch_rows = rows[start : start + batch_size]
        prompts = [item[0] for item in batch_rows]
        labels = torch.tensor([item[1] for item in batch_rows], dtype=torch.long, device=device)
        encoded = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = ensure_token_type_ids(dict(encoded))
        output = model(move_batch(encoded, device), labels=labels)
        output["loss"].backward()

        for name, module in modules:
            if module.weight.grad is not None:
                saliency[name].add_((module.weight.detach().float() * module.weight.grad.detach().float()).abs())
        model.zero_grad(set_to_none=True)
        used_batches += 1

    model.eval()
    return saliency


def wanda_prune_(module: nn.Linear, activation_norm: torch.Tensor, sparsity: float) -> str | None:
    weight = module.weight.data
    if activation_norm.numel() != weight.shape[1]:
        return f"activation width {activation_norm.numel()} does not match in_features={weight.shape[1]}"
    keep_count = int(weight.shape[1] * (1.0 - sparsity))
    if keep_count <= 0:
        weight.zero_()
        return None
    if keep_count >= weight.shape[1]:
        return None
    score = weight.detach().abs().float() * activation_norm.to(weight.device).float().unsqueeze(0)
    keep_indices = torch.topk(score, keep_count, dim=1).indices
    mask = torch.zeros_like(weight, dtype=torch.bool)
    mask.scatter_(1, keep_indices, True)
    weight.mul_(mask)
    return None


def gradient_prune_(module: nn.Linear, saliency: torch.Tensor, sparsity: float) -> str | None:
    weight = module.weight.data
    if saliency.shape != weight.shape:
        return f"saliency shape {list(saliency.shape)} does not match weight shape {list(weight.shape)}"
    scores = saliency.reshape(-1)
    keep_count = int(scores.numel() * (1.0 - sparsity))
    if keep_count <= 0:
        weight.zero_()
        return None
    if keep_count >= scores.numel():
        return None
    threshold = torch.topk(scores, keep_count).values.min().to(weight.device)
    mask = saliency.to(weight.device) >= threshold
    weight.mul_(mask)
    return None


def main() -> None:
    args = parse_args()
    if not 0.0 <= float(args.sparsity) <= 1.0:
        raise ValueError("--sparsity must be between 0.0 and 1.0")
    if args.method == "nvidia" and not math.isclose(float(args.sparsity), 0.5):
        raise ValueError("NVIDIA 2:4 pruning is exactly 50% sparse, so --sparsity must be 0.5.")

    checkpoint_dir = Path(args.checkpoint).expanduser()
    output_dir = Path(args.output).expanduser()
    if output_dir.resolve() == checkpoint_dir.resolve():
        raise ValueError("--output must be different from --checkpoint.")
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"{output_dir} already exists. Pass --overwrite to replace it.")

    device = select_device(args.device)
    dtype = dtype_for(args.dtype, device)
    model, tokenizer, label2response = load_scenic_checkpoint(checkpoint_dir, device="cpu")
    model.to(device=device, dtype=dtype)
    model.eval()

    include_classifier = not bool(args.exclude_classifier)
    modules = selected_linear_modules(model, args.scope, include_classifier=include_classifier)
    if not modules:
        raise RuntimeError(f"No nn.Linear modules matched scope={args.scope!r}.")

    activation_norms: dict[str, torch.Tensor] = {}
    gradient_saliency: dict[str, torch.Tensor] = {}
    calibration_count = 0
    if args.method == "wanda":
        texts = calibration_texts(args.calibration_json)
        calibration_count = min(len(texts), int(args.calibration_batch_size) * int(args.calibration_batches))
        activation_norms = collect_activation_norms(
            model=model,
            tokenizer=tokenizer,
            modules=modules,
            texts=texts,
            device=device,
            max_length=int(args.max_length),
            batch_size=max(1, int(args.calibration_batch_size)),
            calibration_batches=max(1, int(args.calibration_batches)),
        )
    elif args.method == "gradient":
        prompt_label_rows = calibration_prompt_label_rows(args.calibration_json, label2response)
        calibration_count = min(
            len(prompt_label_rows),
            int(args.calibration_batch_size) * int(args.calibration_batches),
        )
        gradient_saliency = collect_gradient_saliency(
            model=model,
            tokenizer=tokenizer,
            modules=modules,
            rows=prompt_label_rows,
            device=device,
            max_length=int(args.max_length),
            batch_size=max(1, int(args.calibration_batch_size)),
            calibration_batches=max(1, int(args.calibration_batches)),
        )

    all_parameters = list(model.named_parameters())
    target_parameters = [(f"{name}.weight", module.weight) for name, module in modules]
    model_before = parameter_totals(all_parameters)
    target_before = parameter_totals(target_parameters)
    tensor_summaries: list[dict[str, Any]] = []
    pruned_tensors = 0

    with torch.no_grad():
        for name, module in tqdm(modules, desc=f"prune {args.method}", unit="tensor"):
            before = module_stats(name, module)
            skipped_reason: str | None = None
            if args.method == "magnitude":
                skipped_reason = magnitude_prune_(module, float(args.sparsity))
            elif args.method == "nvidia":
                skipped_reason = nvidia_2_4_prune_(module)
            elif args.method == "wanda":
                skipped_reason = wanda_prune_(module, activation_norms[name], float(args.sparsity))
            elif args.method == "gradient":
                skipped_reason = gradient_prune_(module, gradient_saliency[name], float(args.sparsity))
            else:
                raise ValueError(f"Unsupported method: {args.method}")

            after = module_stats(name, module, skipped_reason=skipped_reason)
            if skipped_reason is None:
                pruned_tensors += 1
            tensor_summaries.append(
                {
                    "name": name,
                    "shape": after["shape"],
                    "numel": after["numel"],
                    "zeros_before": before["zeros"],
                    "zeros_after": after["zeros"],
                    "new_zeros": max(0, int(after["zeros"]) - int(before["zeros"])),
                    "sparsity_before": before["sparsity"],
                    "sparsity_after": after["sparsity"],
                    **({"skipped_reason": skipped_reason} if skipped_reason else {}),
                }
            )

    model_after = parameter_totals(all_parameters)
    target_after = parameter_totals(target_parameters)
    metadata = read_metadata(checkpoint_dir)
    prune_summary = {
        "input_checkpoint": str(checkpoint_dir),
        "output_checkpoint": str(output_dir),
        "method": args.method,
        "method_detail": {
            "magnitude": "per_linear_weight_abs_keep_topk",
            "nvidia": "2_to_4_structured_smallest_abs_per_group",
            "wanda": "per_row_abs_weight_times_input_activation_norm_keep_topk",
            "gradient": "per_linear_taylor_abs_weight_times_gradient_keep_topk",
        }[args.method],
        "scope": args.scope,
        "include_classifier": include_classifier,
        "requested_sparsity": float(args.sparsity),
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "calibration_json": str(args.calibration_json) if args.method in {"wanda", "gradient"} else None,
        "calibration_examples": calibration_count if args.method in {"wanda", "gradient"} else None,
        "calibration_batch_size": int(args.calibration_batch_size) if args.method in {"wanda", "gradient"} else None,
        "calibration_batches": int(args.calibration_batches) if args.method in {"wanda", "gradient"} else None,
        "targeted_tensors": len(modules),
        "pruned_tensors": pruned_tensors,
        "targeted_parameters": target_after["numel"],
        "targeted_zeros_before": target_before["zeros"],
        "targeted_zeros_after": target_after["zeros"],
        "targeted_sparsity_before": target_before["zeros"] / target_before["numel"],
        "targeted_sparsity_after": target_after["zeros"] / target_after["numel"],
        "model_parameters": model_after["numel"],
        "model_zeros_before": model_before["zeros"],
        "model_zeros_after": model_after["zeros"],
        "model_sparsity_before": model_before["zeros"] / model_before["numel"],
        "model_sparsity_after": model_after["zeros"] / model_after["numel"],
        "tensors": tensor_summaries,
    }
    metadata["pruning"] = prune_summary

    model.to(device="cpu", dtype=torch.float32)
    save_scenic_checkpoint(
        model=model,
        tokenizer=tokenizer,
        output_dir=output_dir,
        label2response=label2response,
        metadata=metadata,
    )
    (output_dir / "prune_summary.json").write_text(
        json.dumps(prune_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"input_checkpoint: {checkpoint_dir}")
    print(f"output_checkpoint: {output_dir}")
    print(f"method: {args.method}")
    print(f"scope: {args.scope}")
    print(f"include_classifier: {include_classifier}")
    print(f"requested_sparsity: {float(args.sparsity):.4f}")
    print(f"targeted_tensors: {len(modules):,}")
    print(f"pruned_tensors: {pruned_tensors:,}")
    print(f"targeted_sparsity_after: {prune_summary['targeted_sparsity_after']:.6f}")
    print(f"model_sparsity_after: {prune_summary['model_sparsity_after']:.6f}")
    print(f"summary: {output_dir / 'prune_summary.json'}")


if __name__ == "__main__":
    main()
