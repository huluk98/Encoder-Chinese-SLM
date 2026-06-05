#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from chatlm_encoder.scenic_sft import ensure_token_type_ids, load_scenic_checkpoint  # noqa: E402


CHECKPOINT_DIR = "runs/scenic-sft-training-dataset/latest"
OUTPUT_ONNX = "runs/scenic-onnx-nvidia/onnx/fp16_dense/model.onnx"
MAX_LENGTH = 128
OPSET = 17


class ScenicOnnxWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        output = self.model(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            }
        )
        return output["logits"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a SCENIC encoder SFT response selector to ONNX.")
    parser.add_argument("--checkpoint", default=CHECKPOINT_DIR, help="SCENIC SFT checkpoint directory.")
    parser.add_argument("--output", default=OUTPUT_ONNX, help="Output ONNX file.")
    parser.add_argument("--precision", choices=("fp32", "fp16", "int8"), required=True)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--opset", type=int, default=OPSET)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--keep-fp32-intermediate",
        action="store_true",
        help="For fp16/int8 exports, keep the temporary fp32 ONNX model next to the output.",
    )
    return parser.parse_args()


def require_module(name: str, package_hint: str | None = None) -> Any:
    try:
        return __import__(name)
    except ImportError as exc:
        package = package_hint or name
        raise SystemExit(
            f"Missing optional dependency {name!r}. Install it with: pip install {package}"
        ) from exc


def output_path_for(path: str | Path, overwrite: bool) -> Path:
    output = Path(path).expanduser()
    if output.exists() and not overwrite:
        raise FileExistsError(f"{output} already exists. Pass --overwrite to replace it.")
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def dummy_inputs(tokenizer: Any, max_length: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    encoded = tokenizer(
        ["打开客厅灯"],
        padding="max_length",
        truncation=True,
        max_length=max(8, int(max_length)),
        return_tensors="pt",
    )
    encoded = ensure_token_type_ids(dict(encoded))
    return (
        encoded["input_ids"].to(torch.long),
        encoded["attention_mask"].to(torch.long),
        encoded["token_type_ids"].to(torch.long),
    )


def export_fp32(
    checkpoint: str | Path,
    output: Path,
    *,
    max_length: int,
    opset: int,
) -> dict[str, Any]:
    onnx = require_module("onnx")
    model, tokenizer, label2response = load_scenic_checkpoint(checkpoint, device="cpu")
    model.to(device="cpu", dtype=torch.float32)
    if hasattr(model.encoder, "config") and hasattr(model.encoder.config, "_attn_implementation"):
        model.encoder.config._attn_implementation = "eager"
    model.eval()
    wrapper = ScenicOnnxWrapper(model).eval()
    input_ids, attention_mask, token_type_ids = dummy_inputs(tokenizer, max_length)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (input_ids, attention_mask, token_type_ids),
            str(output),
            input_names=["input_ids", "attention_mask", "token_type_ids"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "token_type_ids": {0: "batch", 1: "sequence"},
                "logits": {0: "batch"},
            },
            opset_version=int(opset),
            do_constant_folding=True,
        )
    onnx_model = onnx.load(str(output))
    onnx.checker.check_model(onnx_model)
    return {
        "checkpoint": str(checkpoint),
        "label_count": len(label2response),
        "max_length": int(max_length),
        "opset": int(opset),
    }


def convert_fp16(fp32_path: Path, output: Path) -> None:
    require_module("onnx")
    try:
        from onnxconverter_common import float16
    except ImportError as exc:
        raise SystemExit(
            "Missing optional dependency 'onnxconverter_common'. "
            "Install it with: pip install onnxconverter-common"
        ) from exc
    import onnx

    model = onnx.load(str(fp32_path))
    try:
        model_fp16 = float16.convert_float_to_float16(
            model,
            keep_io_types=False,
            disable_shape_infer=True,
        )
    except TypeError:
        model_fp16 = float16.convert_float_to_float16(model, keep_io_types=False)
    model_fp16.graph.ClearField("value_info")
    onnx.save(model_fp16, str(output))
    onnx.checker.check_model(onnx.load(str(output)))


def convert_int8(fp32_path: Path, output: Path) -> None:
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError as exc:
        raise SystemExit(
            "Missing optional dependency 'onnxruntime'. Install it with: pip install onnxruntime"
        ) from exc

    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(output),
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=False,
    )


def write_metadata(output: Path, metadata: dict[str, Any]) -> None:
    metadata_path = output.with_suffix(output.suffix + ".metadata.json")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output = output_path_for(args.output, bool(args.overwrite))

    if args.precision == "fp32":
        metadata = export_fp32(args.checkpoint, output, max_length=int(args.max_length), opset=int(args.opset))
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            fp32_path = (
                output.with_name(output.stem + ".fp32_intermediate.onnx")
                if args.keep_fp32_intermediate
                else Path(tmpdir) / "model.fp32.onnx"
            )
            metadata = export_fp32(args.checkpoint, fp32_path, max_length=int(args.max_length), opset=int(args.opset))
            if args.precision == "fp16":
                convert_fp16(fp32_path, output)
            elif args.precision == "int8":
                convert_int8(fp32_path, output)
            else:
                raise ValueError(f"Unsupported precision: {args.precision}")

    metadata = {
        **metadata,
        "precision": args.precision,
        "quantization": "onnxruntime_dynamic_weight_int8" if args.precision == "int8" else None,
        "output": str(output),
    }
    write_metadata(output, metadata)
    print(f"onnx_output: {output}")
    print(f"metadata_output: {output.with_suffix(output.suffix + '.metadata.json')}")


if __name__ == "__main__":
    main()
