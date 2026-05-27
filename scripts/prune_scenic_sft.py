#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from chatlm_encoder.scenic_sft import load_scenic_checkpoint, save_scenic_checkpoint  # noqa: E402


CHECKPOINT_DIR = "runs/scenic-sft-training-dataset/latest"
OUTPUT_DIR = "runs/scenic-sft-training-dataset-pruned50"
SPARSITY = 0.5
SCOPE = "encoder-linear"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create an unstructured magnitude-pruned SCENIC encoder SFT checkpoint. "
            "The default prunes 50% of encoder transformer linear weights and leaves "
            "embeddings, LayerNorm/bias, and the response classifier unchanged."
        )
    )
    parser.add_argument("--checkpoint", default=CHECKPOINT_DIR, help="Input SCENIC SFT checkpoint directory.")
    parser.add_argument("--output", default=OUTPUT_DIR, help="Output directory for the pruned checkpoint.")
    parser.add_argument("--sparsity", type=float, default=SPARSITY, help="Target sparsity for each selected tensor.")
    parser.add_argument(
        "--scope",
        default=SCOPE,
        choices=("encoder-linear", "all-linear", "all-matrix"),
        help=(
            "encoder-linear: encoder non-embedding 2D weights only; "
            "all-linear: all non-embedding 2D weights; "
            "all-matrix: all matrix weights, including embeddings."
        ),
    )
    parser.add_argument(
        "--include-classifier",
        action="store_true",
        help="Also allow pruning the SCENIC response classifier when the chosen scope matches it.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace the output directory if it exists.")
    return parser.parse_args()


def read_metadata(checkpoint_dir: Path) -> dict[str, Any]:
    metadata_path = checkpoint_dir / "scenic_sft_metadata.json"
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def is_layer_norm_or_bias(name: str, tensor: torch.Tensor) -> bool:
    normalized = name.lower()
    return tensor.ndim < 2 or "layernorm" in normalized or "layer_norm" in normalized


def should_prune(name: str, tensor: torch.Tensor, scope: str, include_classifier: bool) -> bool:
    normalized = name.lower()
    if is_layer_norm_or_bias(name, tensor):
        return False
    if normalized.startswith("classifier.") and not include_classifier:
        return False
    if not normalized.endswith(".weight"):
        return False
    if scope == "encoder-linear":
        return normalized.startswith("encoder.") and "embeddings" not in normalized and tensor.ndim == 2
    if scope == "all-linear":
        return "embeddings" not in normalized and tensor.ndim == 2
    if scope == "all-matrix":
        return tensor.ndim >= 2
    raise ValueError(f"Unsupported pruning scope: {scope}")


def count_zeros(tensor: torch.Tensor) -> int:
    return int((tensor.detach() == 0).sum().item())


def parameter_totals(parameters: list[tuple[str, torch.nn.Parameter]]) -> dict[str, int]:
    numel = 0
    zeros = 0
    for _, parameter in parameters:
        tensor = parameter.detach()
        numel += int(tensor.numel())
        zeros += count_zeros(tensor)
    return {"numel": numel, "zeros": zeros}


def prune_tensor_(parameter: torch.nn.Parameter, sparsity: float) -> dict[str, int | float]:
    tensor = parameter.data
    flat = tensor.view(-1)
    numel = int(flat.numel())
    prune_count = int(round(numel * sparsity))
    prune_count = max(0, min(prune_count, numel))
    zeros_before = count_zeros(flat)

    if prune_count:
        scores = flat.detach().abs().float()
        indices = torch.topk(scores, k=prune_count, largest=False, sorted=False).indices
        flat[indices] = 0

    zeros_after = count_zeros(flat)
    return {
        "numel": numel,
        "requested_pruned_weights": prune_count,
        "zeros_before": zeros_before,
        "zeros_after": zeros_after,
        "new_zeros": max(0, zeros_after - zeros_before),
        "sparsity_before": zeros_before / numel if numel else 0.0,
        "sparsity_after": zeros_after / numel if numel else 0.0,
    }


def write_summary(output_dir: Path, summary: dict[str, Any]) -> None:
    (output_dir / "prune_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if not 0.0 <= float(args.sparsity) <= 1.0:
        raise ValueError("--sparsity must be between 0.0 and 1.0")

    checkpoint_dir = Path(args.checkpoint).expanduser()
    output_dir = Path(args.output).expanduser()
    if output_dir.resolve() == checkpoint_dir.resolve():
        raise ValueError("--output must be different from --checkpoint so the source checkpoint is not overwritten.")
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"{output_dir} already exists. Pass --overwrite to replace it.")

    model, tokenizer, label2response = load_scenic_checkpoint(checkpoint_dir, device="cpu")
    model.eval()

    all_parameters = list(model.named_parameters())
    target_parameters = [
        (name, parameter)
        for name, parameter in all_parameters
        if should_prune(name, parameter, args.scope, bool(args.include_classifier))
    ]
    if not target_parameters:
        raise RuntimeError(f"No parameters matched pruning scope {args.scope!r}.")

    model_before = parameter_totals(all_parameters)
    target_before = parameter_totals(target_parameters)
    tensor_summaries: list[dict[str, Any]] = []

    with torch.no_grad():
        for name, parameter in tqdm(target_parameters, desc="prune scenic", unit="tensor"):
            stats = prune_tensor_(parameter, float(args.sparsity))
            tensor_summaries.append({"name": name, **stats})

    model_after = parameter_totals(all_parameters)
    target_after = parameter_totals(target_parameters)
    metadata = read_metadata(checkpoint_dir)
    prune_summary = {
        "input_checkpoint": str(checkpoint_dir),
        "output_checkpoint": str(output_dir),
        "method": "unstructured_magnitude_per_tensor",
        "scope": args.scope,
        "include_classifier": bool(args.include_classifier),
        "requested_sparsity": float(args.sparsity),
        "targeted_tensors": len(target_parameters),
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

    save_scenic_checkpoint(
        model=model,
        tokenizer=tokenizer,
        output_dir=output_dir,
        label2response=label2response,
        metadata=metadata,
    )
    write_summary(output_dir, prune_summary)

    print(f"input_checkpoint: {checkpoint_dir}")
    print(f"output_checkpoint: {output_dir}")
    print(f"scope: {args.scope}")
    print(f"requested_sparsity: {float(args.sparsity):.4f}")
    print(f"targeted_tensors: {len(target_parameters):,}")
    print(f"targeted_parameters: {target_after['numel']:,}")
    print(f"targeted_sparsity_after: {prune_summary['targeted_sparsity_after']:.6f}")
    print(f"model_sparsity_after: {prune_summary['model_sparsity_after']:.6f}")
    print(f"summary: {output_dir / 'prune_summary.json'}")


if __name__ == "__main__":
    main()
