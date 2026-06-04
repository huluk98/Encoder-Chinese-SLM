#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from chatlm_encoder.scenic_sft import load_scenic_checkpoint  # noqa: E402


SCOPE_CHOICES = ("encoder-linear", "all-linear", "all-matrix")
SELECTOR_STYLE_CHOICES = ("auto", "module-linear", "legacy-parameter")
DEFAULT_DISCOVER_ROOTS = ("runs",)
DEFAULT_OUTPUT_JSON = "eval_results/scenic_sft/active_parameters_audit.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit active/nonzero parameters in one or more generated SCENIC SFT "
            "checkpoints by reloading the saved tensors and recounting zeros."
        )
    )
    parser.add_argument("checkpoints", nargs="*", help="SCENIC checkpoint directories to audit.")
    parser.add_argument(
        "--discover-root",
        action="append",
        default=None,
        help=(
            "Root directory to search when no checkpoints are provided. "
            "Can be passed more than once. Defaults to runs/."
        ),
    )
    parser.add_argument(
        "--include-unpruned",
        action="store_true",
        help="When auto-discovering, also include SCENIC checkpoints without pruning metadata.",
    )
    parser.add_argument(
        "--scope",
        default="auto",
        choices=("auto", *SCOPE_CHOICES),
        help="Target scope to audit. auto reads prune_summary.json when present.",
    )
    parser.add_argument(
        "--include-classifier",
        default="auto",
        choices=("auto", "true", "false"),
        help="Whether the selected target scope includes classifier parameters.",
    )
    parser.add_argument(
        "--selector-style",
        default="auto",
        choices=SELECTOR_STYLE_CHOICES,
        help=(
            "module-linear matches scripts/prune_scenic_sft_reference_methods.py. "
            "legacy-parameter matches scripts/prune_scenic_sft.py."
        ),
    )
    parser.add_argument(
        "--output-json",
        default=DEFAULT_OUTPUT_JSON,
        help=f"JSON output path. Defaults to {DEFAULT_OUTPUT_JSON}.",
    )
    parser.add_argument("--output-csv", default=None, help="Optional per-tensor CSV output path.")
    parser.add_argument(
        "--no-tensor-details",
        action="store_true",
        help="Omit per-tensor details from the JSON output.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run deterministic toy-model tests for the audit counters and selectors.",
    )
    return parser.parse_args()


def is_scenic_checkpoint_dir(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "label2response.json").is_file()
        and (path / "classifier.pt").is_file()
        and (path / "config.json").is_file()
    )


def has_pruning_metadata(path: Path) -> bool:
    if (path / "prune_summary.json").is_file():
        return True
    metadata_path = path / "scenic_sft_metadata.json"
    if not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return isinstance(metadata.get("pruning"), dict)


def discover_scenic_checkpoints(
    roots: list[str | Path],
    *,
    include_unpruned: bool = False,
) -> list[Path]:
    discovered: dict[str, Path] = {}
    for root_value in roots:
        root = Path(root_value).expanduser()
        if is_scenic_checkpoint_dir(root):
            if include_unpruned or has_pruning_metadata(root):
                discovered[root.resolve().as_posix()] = root
            continue
        if not root.exists():
            continue
        for label_path in root.rglob("label2response.json"):
            checkpoint_dir = label_path.parent
            if not is_scenic_checkpoint_dir(checkpoint_dir):
                continue
            if not include_unpruned and not has_pruning_metadata(checkpoint_dir):
                continue
            discovered[checkpoint_dir.resolve().as_posix()] = checkpoint_dir
    return [discovered[key] for key in sorted(discovered)]


def read_prune_summary(checkpoint_dir: Path) -> dict[str, Any]:
    summary_path = checkpoint_dir / "prune_summary.json"
    if not summary_path.exists():
        metadata_path = checkpoint_dir / "scenic_sft_metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            pruning = metadata.get("pruning")
            if isinstance(pruning, dict):
                return pruning
        return {}
    return json.loads(summary_path.read_text(encoding="utf-8"))


def resolve_scope(requested: str, prune_summary: dict[str, Any]) -> str:
    if requested != "auto":
        return requested
    scope = prune_summary.get("scope")
    if scope in SCOPE_CHOICES:
        return str(scope)
    return "encoder-linear"


def resolve_include_classifier(requested: str, prune_summary: dict[str, Any]) -> bool:
    if requested == "true":
        return True
    if requested == "false":
        return False
    value = prune_summary.get("include_classifier")
    if isinstance(value, bool):
        return value
    return False


def resolve_selector_style(requested: str, prune_summary: dict[str, Any]) -> str:
    if requested != "auto":
        return requested
    method = str(prune_summary.get("method", "")).strip().lower()
    if method == "unstructured_magnitude_per_tensor":
        return "legacy-parameter"
    return "module-linear"


def linear_weight_names(model: nn.Module) -> set[str]:
    names: set[str] = set()
    for module_name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            names.add(f"{module_name}.weight" if module_name else "weight")
    return names


def is_classifier_parameter(name: str) -> bool:
    normalized = name.lower()
    return normalized == "classifier" or normalized.startswith("classifier.")


def is_encoder_parameter(name: str) -> bool:
    return name.lower().startswith("encoder.")


def legacy_parameter_scope_selected(
    name: str,
    tensor: torch.Tensor,
    scope: str,
    include_classifier: bool,
) -> bool:
    normalized = name.lower()
    if tensor.ndim < 2 or "layernorm" in normalized or "layer_norm" in normalized:
        return False
    if is_classifier_parameter(name) and not include_classifier:
        return False
    if not normalized.endswith(".weight"):
        return False
    if scope == "encoder-linear":
        return normalized.startswith("encoder.") and "embeddings" not in normalized and tensor.ndim == 2
    if scope == "all-linear":
        return "embeddings" not in normalized and tensor.ndim == 2
    if scope == "all-matrix":
        return tensor.ndim >= 2
    raise ValueError(f"Unsupported scope: {scope}")


def module_linear_scope_selected(
    name: str,
    tensor: torch.Tensor,
    scope: str,
    include_classifier: bool,
    linear_names: set[str],
) -> bool:
    if is_classifier_parameter(name) and not include_classifier:
        return False
    if scope == "encoder-linear":
        return name in linear_names and is_encoder_parameter(name)
    if scope == "all-linear":
        return name in linear_names
    if scope == "all-matrix":
        return name.lower().endswith(".weight") and tensor.ndim >= 2
    raise ValueError(f"Unsupported scope: {scope}")


def selected_by_scope(
    name: str,
    tensor: torch.Tensor,
    scope: str,
    include_classifier: bool,
    selector_style: str,
    linear_names: set[str],
) -> bool:
    if selector_style == "legacy-parameter":
        return legacy_parameter_scope_selected(name, tensor, scope, include_classifier)
    if selector_style == "module-linear":
        return module_linear_scope_selected(name, tensor, scope, include_classifier, linear_names)
    raise ValueError(f"Unsupported selector style: {selector_style}")


def tensor_activity_counts(tensor: torch.Tensor) -> dict[str, int | float]:
    detached = tensor.detach()
    numel = int(detached.numel())
    active = int(torch.count_nonzero(detached).item()) if numel else 0
    zeros = numel - active
    if detached.is_floating_point() or detached.is_complex():
        nonfinite = int((~torch.isfinite(detached)).sum().item()) if numel else 0
    else:
        nonfinite = 0
    return {
        "numel": numel,
        "active": active,
        "zeros": zeros,
        "sparsity": zeros / numel if numel else 0.0,
        "active_ratio": active / numel if numel else 0.0,
        "nonfinite": nonfinite,
    }


def nvidia_2_4_stats(tensor: torch.Tensor) -> dict[str, int | float | bool | None]:
    detached = tensor.detach()
    if detached.ndim != 2:
        return {
            "nvidia_2_4_eligible": False,
            "nvidia_2_4_groups": None,
            "nvidia_2_4_valid_groups": None,
            "nvidia_2_4_valid_ratio": None,
        }
    if int(detached.shape[1]) % 4 != 0:
        return {
            "nvidia_2_4_eligible": False,
            "nvidia_2_4_groups": None,
            "nvidia_2_4_valid_groups": None,
            "nvidia_2_4_valid_ratio": None,
        }
    grouped = detached.reshape(detached.shape[0], -1, 4)
    zero_counts = (grouped == 0).sum(dim=2)
    groups = int(zero_counts.numel())
    valid_groups = int((zero_counts == 2).sum().item())
    return {
        "nvidia_2_4_eligible": True,
        "nvidia_2_4_groups": groups,
        "nvidia_2_4_valid_groups": valid_groups,
        "nvidia_2_4_valid_ratio": valid_groups / groups if groups else 0.0,
    }


def empty_total() -> dict[str, int | float]:
    return {
        "tensors": 0,
        "numel": 0,
        "active": 0,
        "zeros": 0,
        "sparsity": 0.0,
        "active_ratio": 0.0,
        "nonfinite": 0,
    }


def add_counts(total: dict[str, int | float], counts: dict[str, int | float]) -> None:
    total["tensors"] = int(total["tensors"]) + 1
    total["numel"] = int(total["numel"]) + int(counts["numel"])
    total["active"] = int(total["active"]) + int(counts["active"])
    total["zeros"] = int(total["zeros"]) + int(counts["zeros"])
    total["nonfinite"] = int(total["nonfinite"]) + int(counts["nonfinite"])


def finalize_total(total: dict[str, int | float]) -> dict[str, int | float]:
    numel = int(total["numel"])
    zeros = int(total["zeros"])
    active = int(total["active"])
    return {
        **total,
        "sparsity": zeros / numel if numel else 0.0,
        "active_ratio": active / numel if numel else 0.0,
    }


def tensor_group(name: str) -> str:
    if is_classifier_parameter(name):
        return "classifier"
    if is_encoder_parameter(name):
        return "encoder"
    return "other"


def audit_model_parameters(
    model: nn.Module,
    *,
    scope: str,
    include_classifier: bool,
    selector_style: str,
    include_tensor_details: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    linear_names = linear_weight_names(model)
    totals = {
        "model_all_parameters": empty_total(),
        "selected_scope_parameters": empty_total(),
        "encoder_all_parameters": empty_total(),
        "classifier_parameters": empty_total(),
        "all_linear_module_weights": empty_total(),
        "encoder_linear_module_weights": empty_total(),
        "all_matrix_weight_parameters": empty_total(),
    }
    details: list[dict[str, Any]] = []
    selected_2_4_groups = 0
    selected_2_4_valid_groups = 0
    selected_2_4_eligible_tensors = 0
    selected_2_4_invalid_tensors: list[str] = []

    for name, parameter in model.named_parameters():
        tensor = parameter.detach()
        counts = tensor_activity_counts(tensor)
        selected = selected_by_scope(
            name=name,
            tensor=tensor,
            scope=scope,
            include_classifier=include_classifier,
            selector_style=selector_style,
            linear_names=linear_names,
        )
        stats_2_4 = nvidia_2_4_stats(tensor)
        row = {
            "name": name,
            "group": tensor_group(name),
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).replace("torch.", ""),
            "selected": selected,
            "is_linear_weight": name in linear_names,
            "is_matrix_weight": name.lower().endswith(".weight") and tensor.ndim >= 2,
            **counts,
            **stats_2_4,
        }
        add_counts(totals["model_all_parameters"], counts)
        if selected:
            add_counts(totals["selected_scope_parameters"], counts)
            if bool(stats_2_4["nvidia_2_4_eligible"]):
                selected_2_4_eligible_tensors += 1
                selected_2_4_groups += int(stats_2_4["nvidia_2_4_groups"] or 0)
                selected_2_4_valid_groups += int(stats_2_4["nvidia_2_4_valid_groups"] or 0)
                if not math.isclose(float(stats_2_4["nvidia_2_4_valid_ratio"] or 0.0), 1.0):
                    selected_2_4_invalid_tensors.append(name)
        if is_encoder_parameter(name):
            add_counts(totals["encoder_all_parameters"], counts)
        if is_classifier_parameter(name):
            add_counts(totals["classifier_parameters"], counts)
        if name in linear_names:
            add_counts(totals["all_linear_module_weights"], counts)
            if is_encoder_parameter(name):
                add_counts(totals["encoder_linear_module_weights"], counts)
        if name.lower().endswith(".weight") and tensor.ndim >= 2:
            add_counts(totals["all_matrix_weight_parameters"], counts)
        if include_tensor_details:
            details.append(row)

    summaries = {key: finalize_total(value) for key, value in totals.items()}
    summaries["selected_nvidia_2_4"] = {
        "eligible_tensors": selected_2_4_eligible_tensors,
        "groups": selected_2_4_groups,
        "valid_groups": selected_2_4_valid_groups,
        "valid_ratio": (
            selected_2_4_valid_groups / selected_2_4_groups
            if selected_2_4_groups
            else None
        ),
        "invalid_tensors": selected_2_4_invalid_tensors[:100],
    }
    return summaries, details


def compare_summary_field(
    checks: list[dict[str, Any]],
    prune_summary: dict[str, Any],
    field: str,
    computed: int | float,
    tolerance: float = 0.0,
) -> None:
    if field not in prune_summary:
        return
    reported = prune_summary[field]
    if isinstance(computed, float):
        ok = math.isclose(float(reported), computed, rel_tol=0.0, abs_tol=tolerance)
    else:
        ok = int(reported) == int(computed)
    checks.append(
        {
            "field": field,
            "reported": reported,
            "computed": computed,
            "ok": ok,
        }
    )


def summary_consistency_checks(
    prune_summary: dict[str, Any],
    summaries: dict[str, Any],
) -> dict[str, Any]:
    if not prune_summary:
        return {"available": False, "ok": None, "checks": []}
    checks: list[dict[str, Any]] = []
    selected = summaries["selected_scope_parameters"]
    model = summaries["model_all_parameters"]
    compare_summary_field(checks, prune_summary, "targeted_parameters", int(selected["numel"]))
    compare_summary_field(checks, prune_summary, "targeted_zeros_after", int(selected["zeros"]))
    compare_summary_field(checks, prune_summary, "targeted_sparsity_after", float(selected["sparsity"]), 1e-12)
    compare_summary_field(checks, prune_summary, "model_parameters", int(model["numel"]))
    compare_summary_field(checks, prune_summary, "model_zeros_after", int(model["zeros"]))
    compare_summary_field(checks, prune_summary, "model_sparsity_after", float(model["sparsity"]), 1e-12)
    return {
        "available": True,
        "ok": all(bool(item["ok"]) for item in checks),
        "checks": checks,
    }


def audit_scenic_checkpoint(
    checkpoint_dir: str | Path,
    *,
    scope: str = "auto",
    include_classifier: str = "auto",
    selector_style: str = "auto",
    include_tensor_details: bool = True,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_dir).expanduser()
    prune_summary = read_prune_summary(checkpoint_path)
    resolved_scope = resolve_scope(scope, prune_summary)
    resolved_include_classifier = resolve_include_classifier(include_classifier, prune_summary)
    resolved_selector_style = resolve_selector_style(selector_style, prune_summary)

    model, _tokenizer, label2response = load_scenic_checkpoint(checkpoint_path, device="cpu")
    model.eval()
    summaries, tensor_details = audit_model_parameters(
        model,
        scope=resolved_scope,
        include_classifier=resolved_include_classifier,
        selector_style=resolved_selector_style,
        include_tensor_details=include_tensor_details,
    )
    result: dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_resolved": str(checkpoint_path.resolve()),
        "label_count": len(label2response),
        "scope": resolved_scope,
        "include_classifier": resolved_include_classifier,
        "selector_style": resolved_selector_style,
        "prune_summary_present": bool(prune_summary),
        "prune_summary_method": prune_summary.get("method") if prune_summary else None,
        "prune_summary_method_detail": prune_summary.get("method_detail") if prune_summary else None,
        "summaries": summaries,
        "summary_consistency": summary_consistency_checks(prune_summary, summaries),
    }
    if include_tensor_details:
        result["tensors"] = tensor_details
    return result


def build_report(
    audits: list[dict[str, Any]],
    *,
    discovery: dict[str, Any],
) -> dict[str, Any]:
    return {
        "report_type": "scenic_active_parameters_audit",
        "status": "ready" if audits else "no_checkpoints_found",
        "checkpoint_count": len(audits),
        "discovery": discovery,
        "audits": audits,
    }


def write_json(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, audits: list[dict[str, Any]]) -> None:
    fieldnames = [
        "checkpoint",
        "name",
        "group",
        "shape",
        "dtype",
        "selected",
        "is_linear_weight",
        "is_matrix_weight",
        "numel",
        "active",
        "zeros",
        "sparsity",
        "active_ratio",
        "nonfinite",
        "nvidia_2_4_eligible",
        "nvidia_2_4_groups",
        "nvidia_2_4_valid_groups",
        "nvidia_2_4_valid_ratio",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for audit in audits:
            for row in audit.get("tensors", []):
                writer.writerow(
                    {
                        **{key: row.get(key) for key in fieldnames},
                        "checkpoint": audit.get("checkpoint"),
                        "shape": json.dumps(row.get("shape")),
                    }
                )


def print_compact(audit: dict[str, Any]) -> None:
    selected = audit["summaries"]["selected_scope_parameters"]
    model = audit["summaries"]["model_all_parameters"]
    encoder_linear = audit["summaries"]["encoder_linear_module_weights"]
    consistency = audit["summary_consistency"]
    print(f"checkpoint: {audit['checkpoint']}")
    print(f"scope: {audit['scope']}")
    print(f"include_classifier: {audit['include_classifier']}")
    print(f"selector_style: {audit['selector_style']}")
    print(
        "selected_active_parameters: "
        f"{int(selected['active']):,} / {int(selected['numel']):,} "
        f"(active_ratio={float(selected['active_ratio']):.6f}, "
        f"sparsity={float(selected['sparsity']):.6f})"
    )
    print(
        "model_active_parameters: "
        f"{int(model['active']):,} / {int(model['numel']):,} "
        f"(active_ratio={float(model['active_ratio']):.6f}, "
        f"sparsity={float(model['sparsity']):.6f})"
    )
    print(
        "encoder_linear_active_parameters: "
        f"{int(encoder_linear['active']):,} / {int(encoder_linear['numel']):,} "
        f"(active_ratio={float(encoder_linear['active_ratio']):.6f}, "
        f"sparsity={float(encoder_linear['sparsity']):.6f})"
    )
    if consistency["available"]:
        print(f"prune_summary_consistent: {consistency['ok']}")
    selected_2_4 = audit["summaries"]["selected_nvidia_2_4"]
    if selected_2_4["groups"]:
        print(
            "selected_nvidia_2_4_valid_ratio: "
            f"{float(selected_2_4['valid_ratio']):.6f} "
            f"({int(selected_2_4['valid_groups']):,} / {int(selected_2_4['groups']):,} groups)"
        )


def run_self_test() -> None:
    class TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Module()
            self.encoder.linear = nn.Linear(4, 2, bias=False)
            self.encoder.embeddings = nn.Embedding(2, 4)
            self.classifier = nn.Linear(4, 2)

    model = TinyModel()
    with torch.no_grad():
        model.encoder.linear.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0, 2.0, 0.0],
                    [0.0, 3.0, 0.0, 4.0],
                ]
            )
        )
        model.encoder.embeddings.weight.fill_(1.0)
        model.classifier.weight.copy_(
            torch.tensor(
                [
                    [5.0, 0.0, 6.0, 0.0],
                    [0.0, 7.0, 0.0, 8.0],
                ]
            )
        )
        model.classifier.bias.zero_()

    summaries, details = audit_model_parameters(
        model,
        scope="encoder-linear",
        include_classifier=False,
        selector_style="module-linear",
        include_tensor_details=True,
    )
    selected = summaries["selected_scope_parameters"]
    assert selected["numel"] == 8, selected
    assert selected["active"] == 4, selected
    assert selected["zeros"] == 4, selected
    assert math.isclose(float(selected["sparsity"]), 0.5), selected
    selected_2_4 = summaries["selected_nvidia_2_4"]
    assert selected_2_4["groups"] == 2, selected_2_4
    assert selected_2_4["valid_groups"] == 2, selected_2_4
    assert math.isclose(float(selected_2_4["valid_ratio"]), 1.0), selected_2_4
    assert any(row["name"] == "encoder.linear.weight" and row["selected"] for row in details)

    summaries, _details = audit_model_parameters(
        model,
        scope="all-linear",
        include_classifier=True,
        selector_style="module-linear",
        include_tensor_details=False,
    )
    selected = summaries["selected_scope_parameters"]
    assert selected["numel"] == 16, selected
    assert selected["active"] == 8, selected
    assert selected["zeros"] == 8, selected

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        pruned = root / "runs" / "pruned"
        unpruned = root / "runs" / "unpruned"
        for path in (pruned, unpruned):
            path.mkdir(parents=True)
            (path / "label2response.json").write_text("[]\n", encoding="utf-8")
            (path / "classifier.pt").write_text("placeholder\n", encoding="utf-8")
            (path / "config.json").write_text("{}\n", encoding="utf-8")
        (pruned / "prune_summary.json").write_text('{"scope": "encoder-linear"}\n', encoding="utf-8")
        discovered = discover_scenic_checkpoints([root / "runs"], include_unpruned=False)
        assert discovered == [pruned], discovered
        discovered = discover_scenic_checkpoints([root / "runs"], include_unpruned=True)
        assert discovered == [pruned, unpruned], discovered
    print("self-test passed")


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    checkpoints = [Path(checkpoint).expanduser() for checkpoint in args.checkpoints]
    discover_roots: list[Path] = []
    auto_discovered = False
    if not checkpoints:
        auto_discovered = True
        discover_roots = [Path(root).expanduser() for root in (args.discover_root or DEFAULT_DISCOVER_ROOTS)]
        checkpoints = discover_scenic_checkpoints(
            discover_roots,
            include_unpruned=bool(args.include_unpruned),
        )
        if checkpoints:
            roots = ", ".join(str(root) for root in discover_roots)
            print(f"auto_discovered_checkpoints: {len(checkpoints)} from {roots}")
        else:
            roots = ", ".join(str(root) for root in discover_roots)
            print(f"no_generated_pruned_checkpoints_discovered: searched {roots}")

    audits = [
        audit_scenic_checkpoint(
            checkpoint,
            scope=str(args.scope),
            include_classifier=str(args.include_classifier),
            selector_style=str(args.selector_style),
            include_tensor_details=not bool(args.no_tensor_details),
        )
        for checkpoint in checkpoints
    ]

    for index, audit in enumerate(audits):
        if index:
            print()
        print_compact(audit)

    discovery = {
        "auto_discovered": auto_discovered,
        "roots": [str(root) for root in discover_roots],
        "include_unpruned": bool(args.include_unpruned),
        "discovered_checkpoints": [str(checkpoint) for checkpoint in checkpoints] if auto_discovered else [],
        "explicit_checkpoints": [str(checkpoint) for checkpoint in args.checkpoints],
    }
    output_json = Path(args.output_json).expanduser()
    write_json(output_json, build_report(audits, discovery=discovery))
    print(f"json_output: {output_json}")
    if args.output_csv:
        if args.no_tensor_details:
            raise ValueError("--output-csv requires tensor details; remove --no-tensor-details.")
        write_csv(Path(args.output_csv).expanduser(), audits)


if __name__ == "__main__":
    main()
