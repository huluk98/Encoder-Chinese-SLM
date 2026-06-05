#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_ENCODER_REPORT = "eval_results/scenic_sft/onnx_nvidia/edge_fp16_report.json"
DEFAULT_OUTPUT = "eval_results/scenic_sft/onnx_nvidia/fp16_deployment_table.tex"
ARCHITECTURES = ("Encoder-only", "Decoder-only", "Encoder--decoder")
RUNTIMES = ("PyTorch", "TensorRT")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the final FP16 compact-architecture deployment feasibility LaTeX table."
    )
    parser.add_argument("--encoder-report", default=DEFAULT_ENCODER_REPORT)
    parser.add_argument("--decoder-report", default=None)
    parser.add_argument("--encoder-decoder-report", default=None)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_rows(path: str | Path | None) -> list[dict[str, Any]]:
    if not path:
        return []
    report_path = Path(path).expanduser()
    if not report_path.exists():
        return []
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    if not isinstance(rows, list):
        raise ValueError(f"{report_path} must contain a JSON object with a rows list.")
    return [row for row in rows if isinstance(row, dict)]


def runtime_name(row: dict[str, Any]) -> str | None:
    label = str(row.get("runtime_display") or row.get("runtime_label") or row.get("runtime") or "").lower()
    if "tensorrt" in label:
        return "TensorRT"
    if "pytorch" in label:
        return "PyTorch"
    return None


def select_row(rows: list[dict[str, Any]], runtime: str, seq_len: int) -> dict[str, Any] | None:
    candidates = [
        row
        for row in rows
        if runtime_name(row) == runtime
        and int(row.get("input_length") or -1) == int(seq_len)
        and "nvidia_2_4" not in str(row.get("runtime_label") or "")
        and str(row.get("sparsity") or "dense") == "dense"
    ]
    if candidates:
        return candidates[0]
    fallback = [
        row
        for row in rows
        if runtime_name(row) == runtime
        and int(row.get("input_length") or -1) == int(seq_len)
    ]
    return fallback[0] if fallback else None


def format_number(value: Any, decimals: int = 2) -> str:
    if value is None:
        return "--"
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return "--"


def format_memory(row: dict[str, Any] | None) -> str:
    if not row:
        return "--"
    for key in ("peak_gpu_memory_mb_process", "peak_torch_cuda_allocated_mb", "peak_cpu_rss_mb"):
        value = row.get(key)
        if value is not None:
            return f"{format_number(value)} MB"
    return "--"


def format_em(row: dict[str, Any] | None) -> str:
    if not row:
        return "--"
    em1 = row.get("benchmark_em1_percent")
    em5 = row.get("benchmark_em5_percent")
    if em1 is None or em5 is None:
        return "--"
    return f"{format_number(em1)}/{format_number(em5)}"


def table_row(architecture: str, runtime: str, seq_len: int, row: dict[str, Any] | None) -> str:
    latency = f"{format_number(row.get('mean_latency_ms') if row else None)} ms" if row else "--"
    p95 = f"{format_number(row.get('p95_latency_ms') if row else None)} ms" if row else "--"
    return (
        f"{architecture} & {runtime} & {seq_len}  & {latency} & {p95} & "
        f"{format_memory(row)} & {format_em(row)} \\\\"
    )


def render_table(
    reports: dict[str, list[dict[str, Any]]],
    *,
    seq_len: int,
) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{FP16 Deployment Feasibility Across Compact Architectures}",
        r"\label{tab:fp16_deployment}",
        r"\begin{tabular}{lcccccc}",
        r"\hline",
        r"Architecture & Runtime & Seq. Len. & Latency & P95 Lat. & Memory & EM@1/EM@5 \\",
        r"\hline",
    ]
    for architecture in ARCHITECTURES:
        rows = reports.get(architecture, [])
        for runtime in RUNTIMES:
            lines.append(table_row(architecture, runtime, seq_len, select_row(rows, runtime, seq_len)))
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    reports = {
        "Encoder-only": load_rows(args.encoder_report),
        "Decoder-only": load_rows(args.decoder_report),
        "Encoder--decoder": load_rows(args.encoder_decoder_report),
    }
    table = render_table(reports, seq_len=int(args.seq_len))
    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(table, encoding="utf-8")
    print(f"latex_table_output: {output_path}")


if __name__ == "__main__":
    main()
