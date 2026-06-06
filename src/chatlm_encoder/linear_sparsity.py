from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


HEAD_PREFIXES = ("classifier", "lm_head", "qa_outputs", "score", "cls")
HEAD_SUBSTRINGS = (
    "response_classifier",
    "response_projection",
    "final_response_projection",
    "final_projection",
    "output_head",
    "prediction_head",
)


@dataclass(frozen=True)
class LinearSparsityConfig:
    prune_output_heads: bool = False
    global_pruning: bool = False
    regrowth: bool = False


def canonical_module_name(name: str) -> str:
    return name.removeprefix("module.")


def is_output_head_module(name: str) -> bool:
    normalized = canonical_module_name(name).lower()
    parts = tuple(part for part in normalized.split(".") if part)
    if parts and parts[0] in HEAD_PREFIXES:
        return True
    if parts and parts[-1] in HEAD_PREFIXES:
        return True
    return any(fragment in normalized for fragment in HEAD_SUBSTRINGS)


def collect_prunable_linear_modules(
    model: nn.Module,
    prune_output_heads: bool = False,
) -> list[tuple[str, nn.Linear]]:
    modules: list[tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not prune_output_heads and is_output_head_module(name):
            continue
        modules.append((canonical_module_name(name), module))
    return modules


def count_zeros(tensor: torch.Tensor) -> int:
    return int((tensor.detach() == 0).sum().item())


def parameter_sparsity(parameters: list[tuple[str, torch.Tensor | nn.Parameter]]) -> dict[str, int | float]:
    numel = 0
    zeros = 0
    for _name, parameter in parameters:
        tensor = parameter.detach()
        numel += int(tensor.numel())
        zeros += count_zeros(tensor)
    return {
        "numel": numel,
        "zeros": zeros,
        "sparsity": zeros / numel if numel else 0.0,
    }


def targeted_linear_sparsity(modules: list[tuple[str, nn.Linear]]) -> dict[str, int | float]:
    return parameter_sparsity([(f"{name}.weight", module.weight) for name, module in modules])


def whole_model_sparsity(model: nn.Module) -> dict[str, int | float]:
    return parameter_sparsity(list(model.named_parameters()))


def _keep_count(numel: int, sparsity: float) -> int:
    if not 0.0 <= float(sparsity) <= 1.0:
        raise ValueError("sparsity must be between 0.0 and 1.0")
    return max(0, min(numel, int(round(numel * (1.0 - float(sparsity))))))


def _mask_for_weight(
    weight: torch.Tensor,
    sparsity: float,
    existing_mask: torch.Tensor | None = None,
    regrowth: bool = False,
) -> torch.Tensor:
    flat_scores = weight.detach().abs().float().reshape(-1)
    numel = int(flat_scores.numel())
    keep_count = _keep_count(numel, sparsity)
    if keep_count <= 0:
        return torch.zeros_like(weight, dtype=torch.bool)

    eligible = torch.ones(numel, dtype=torch.bool, device=weight.device)
    if existing_mask is not None and not regrowth:
        eligible = existing_mask.to(device=weight.device, dtype=torch.bool).reshape(-1)
        keep_count = min(keep_count, int(eligible.sum().item()))
        if keep_count <= 0:
            return torch.zeros_like(weight, dtype=torch.bool)

    if keep_count >= numel and bool(eligible.all().item()):
        return torch.ones_like(weight, dtype=torch.bool)

    scores = flat_scores.clone()
    scores[~eligible] = -torch.inf
    keep_indices = torch.topk(scores, k=keep_count, largest=True, sorted=False).indices
    flat_mask = torch.zeros(numel, dtype=torch.bool, device=weight.device)
    flat_mask[keep_indices] = True
    return flat_mask.reshape_as(weight)


def _global_masks(
    modules: list[tuple[str, nn.Linear]],
    sparsity: float,
    existing_masks: dict[str, torch.Tensor] | None = None,
    regrowth: bool = False,
) -> dict[str, torch.Tensor]:
    names: list[str] = []
    scores: list[torch.Tensor] = []
    eligible_chunks: list[torch.Tensor] = []
    shapes: dict[str, torch.Size] = {}
    for name, module in modules:
        weight = module.weight.detach()
        flat = weight.abs().float().reshape(-1)
        names.append(name)
        scores.append(flat)
        shapes[name] = weight.shape
        existing = (existing_masks or {}).get(name)
        if existing is not None and not regrowth:
            eligible_chunks.append(existing.to(device=weight.device, dtype=torch.bool).reshape(-1))
        else:
            eligible_chunks.append(torch.ones_like(flat, dtype=torch.bool))

    if not scores:
        return {}

    all_scores = torch.cat(scores)
    eligible = torch.cat(eligible_chunks)
    keep_count = _keep_count(int(all_scores.numel()), sparsity)
    if not regrowth:
        keep_count = min(keep_count, int(eligible.sum().item()))
    masked_scores = all_scores.clone()
    masked_scores[~eligible] = -torch.inf

    flat_mask = torch.zeros_like(eligible, dtype=torch.bool)
    if keep_count > 0:
        keep_indices = torch.topk(masked_scores, k=keep_count, largest=True, sorted=False).indices
        flat_mask[keep_indices] = True

    masks: dict[str, torch.Tensor] = {}
    offset = 0
    for name, score in zip(names, scores):
        end = offset + int(score.numel())
        masks[name] = flat_mask[offset:end].reshape(shapes[name])
        offset = end
    return masks


def apply_masks(model: nn.Module, masks: dict[str, torch.Tensor]) -> None:
    modules = dict(model.named_modules())
    with torch.no_grad():
        for name, mask in masks.items():
            module = modules.get(name) or modules.get(f"module.{name}")
            if not isinstance(module, nn.Linear):
                raise KeyError(f"Mask target {name!r} is not an nn.Linear module in the model.")
            module.weight.mul_(mask.to(device=module.weight.device, dtype=module.weight.dtype))


def register_mask_gradient_hooks(model: nn.Module, masks: dict[str, torch.Tensor]) -> list[Any]:
    handles: list[Any] = []
    modules = dict(model.named_modules())
    for name, mask in masks.items():
        module = modules.get(name) or modules.get(f"module.{name}")
        if not isinstance(module, nn.Linear):
            raise KeyError(f"Mask target {name!r} is not an nn.Linear module in the model.")
        mask_for_hook = mask.to(device=module.weight.device, dtype=module.weight.dtype)
        handles.append(module.weight.register_hook(lambda grad, mask=mask_for_hook: grad * mask))
    return handles


def remove_hooks(handles: list[Any]) -> None:
    for handle in handles:
        handle.remove()


def apply_magnitude_pruning(
    model: nn.Module,
    sparsity: float,
    config: LinearSparsityConfig | None = None,
    masks: dict[str, torch.Tensor] | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    config = config or LinearSparsityConfig()
    modules = collect_prunable_linear_modules(model, prune_output_heads=config.prune_output_heads)
    if not modules:
        raise RuntimeError("No prunable nn.Linear modules matched the configured pruning scope.")

    if config.global_pruning:
        new_masks = _global_masks(
            modules,
            sparsity=sparsity,
            existing_masks=masks,
            regrowth=config.regrowth,
        )
    else:
        new_masks = {
            name: _mask_for_weight(
                module.weight,
                sparsity=sparsity,
                existing_mask=(masks or {}).get(name),
                regrowth=config.regrowth,
            )
            for name, module in modules
        }

    apply_masks(model, new_masks)
    target_stats = targeted_linear_sparsity(modules)
    whole_stats = whole_model_sparsity(model)
    summary = {
        "prune_scope": "linear_weights",
        "prune_method": "magnitude",
        "target_sparsity": float(sparsity),
        "prune_output_heads": bool(config.prune_output_heads),
        "global_pruning": bool(config.global_pruning),
        "regrowth": bool(config.regrowth),
        "targeted_linear_parameters": int(target_stats["numel"]),
        "targeted_linear_zeros": int(target_stats["zeros"]),
        "targeted_linear_sparsity_actual": float(target_stats["sparsity"]),
        "whole_model_parameters": int(whole_stats["numel"]),
        "whole_model_zeros": int(whole_stats["zeros"]),
        "whole_model_sparsity_actual": float(whole_stats["sparsity"]),
        "selected_linear_tensors": [
            {
                "name": name,
                "shape": list(module.weight.shape),
                "numel": int(module.weight.numel()),
                "zeros": count_zeros(module.weight),
                "sparsity": count_zeros(module.weight) / int(module.weight.numel())
                if int(module.weight.numel())
                else 0.0,
            }
            for name, module in modules
        ],
    }
    return {name: mask.detach().cpu() for name, mask in new_masks.items()}, summary


def current_linear_sparsity_summary(
    model: nn.Module,
    config: LinearSparsityConfig | None = None,
    target_sparsity: float = 0.0,
) -> dict[str, Any]:
    config = config or LinearSparsityConfig()
    modules = collect_prunable_linear_modules(model, prune_output_heads=config.prune_output_heads)
    target_stats = targeted_linear_sparsity(modules)
    whole_stats = whole_model_sparsity(model)
    return {
        "prune_scope": "linear_weights",
        "prune_method": "magnitude",
        "target_sparsity": float(target_sparsity),
        "prune_output_heads": bool(config.prune_output_heads),
        "global_pruning": bool(config.global_pruning),
        "regrowth": bool(config.regrowth),
        "targeted_linear_parameters": int(target_stats["numel"]),
        "targeted_linear_zeros": int(target_stats["zeros"]),
        "targeted_linear_sparsity_actual": float(target_stats["sparsity"]),
        "whole_model_parameters": int(whole_stats["numel"]),
        "whole_model_zeros": int(whole_stats["zeros"]),
        "whole_model_sparsity_actual": float(whole_stats["sparsity"]),
        "selected_linear_tensors": [
            {
                "name": name,
                "shape": list(module.weight.shape),
                "numel": int(module.weight.numel()),
                "zeros": count_zeros(module.weight),
                "sparsity": count_zeros(module.weight) / int(module.weight.numel())
                if int(module.weight.numel())
                else 0.0,
            }
            for name, module in modules
        ],
    }


def save_masks(
    path: str | Path,
    masks: dict[str, torch.Tensor],
    metadata: dict[str, Any],
) -> None:
    mask_path = Path(path).expanduser()
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "masks": {name: mask.detach().cpu().to(dtype=torch.uint8) for name, mask in masks.items()},
            "metadata": metadata,
        },
        mask_path,
    )


def load_masks(path: str | Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    payload = torch.load(Path(path).expanduser(), map_location="cpu")
    masks = {name: tensor.to(dtype=torch.bool) for name, tensor in payload.get("masks", {}).items()}
    metadata = dict(payload.get("metadata", {}))
    return masks, metadata
