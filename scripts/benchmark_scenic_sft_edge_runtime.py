#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from chatlm_encoder.scenic_sft import ensure_token_type_ids, load_scenic_checkpoint, prompt_from_row, read_json_list  # noqa: E402


LOCAL_JSON_PATH = "data/scenic/iot_instruction_benchmark_200.json"
CHECKPOINT_DIR = "runs/scenic-sft-training-dataset/latest"
ONNX_MODEL = "runs/scenic-onnx-nvidia/onnx/fp16_dense/model.onnx"
OUTPUT_PATH = "eval_results/scenic_sft/onnx_nvidia/edge_runtime/pytorch_fp16_dense_seq128.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark SCENIC encoder SFT batch-1 edge inference latency on "
            "pre-tokenized fixed-length inputs."
        )
    )
    parser.add_argument("--runtime", required=True, choices=("pytorch", "onnx", "tensorrt"))
    parser.add_argument("--runtime-label", default=None)
    parser.add_argument("--checkpoint", default=CHECKPOINT_DIR, help="SCENIC checkpoint for PyTorch or tokenizer.")
    parser.add_argument("--onnx", default=ONNX_MODEL, help="ONNX model path for ONNX/TensorRT runtimes.")
    parser.add_argument("--json", default=LOCAL_JSON_PATH, help="Prompt JSON used to build benchmark inputs.")
    parser.add_argument("--output", default=OUTPUT_PATH, help="Runtime summary JSON output path.")
    parser.add_argument("--precision", default="fp16", choices=("fp16",), help="Runtime precision to benchmark.")
    parser.add_argument("--max-length", type=int, default=128, help="Fixed sequence length.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size. Use 1 for interactive IoT commands.")
    parser.add_argument("--warmup-queries", type=int, default=20)
    parser.add_argument("--measure-queries", type=int, default=200)
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0, or cpu for PyTorch.")
    parser.add_argument(
        "--providers",
        default="auto",
        choices=("auto", "cpu", "cuda", "tensorrt"),
        help="ONNX Runtime provider choice for ONNX/TensorRT runtimes.",
    )
    parser.add_argument("--trt-engine-cache-dir", default=None)
    parser.add_argument("--gpu-memory-poll-interval", type=float, default=0.10)
    return parser.parse_args()


def load_prompts(path: str | Path) -> list[str]:
    prompts: list[str] = []
    for row in read_json_list(path):
        prompt = prompt_from_row(row)
        if prompt:
            prompts.append(prompt)
    if not prompts:
        raise ValueError(f"No prompt-like rows found in {path}.")
    return prompts


def prompt_batches(prompts: list[str], query_count: int, batch_size: int) -> list[list[str]]:
    if query_count <= 0:
        return []
    batch_size = max(1, int(batch_size))
    batches: list[list[str]] = []
    index = 0
    while index < query_count:
        current_size = min(batch_size, query_count - index)
        batches.append([prompts[(index + offset) % len(prompts)] for offset in range(current_size)])
        index += current_size
    return batches


def tokenize_torch_batches(tokenizer: Any, batches: list[list[str]], max_length: int) -> list[dict[str, torch.Tensor]]:
    tokenized = []
    for batch in batches:
        encoded = tokenizer(
            batch,
            padding="max_length",
            truncation=True,
            max_length=max(8, int(max_length)),
            return_tensors="pt",
        )
        tokenized.append(ensure_token_type_ids(dict(encoded)))
    return tokenized


def ensure_numpy_token_type_ids(encoded: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if "token_type_ids" not in encoded:
        encoded["token_type_ids"] = np.zeros_like(encoded["input_ids"], dtype=np.int64)
    return encoded


def tokenize_numpy_batches(tokenizer: Any, batches: list[list[str]], max_length: int) -> list[dict[str, np.ndarray]]:
    tokenized = []
    for batch in batches:
        encoded = tokenizer(
            batch,
            padding="max_length",
            truncation=True,
            max_length=max(8, int(max_length)),
            return_tensors="np",
        )
        encoded = ensure_numpy_token_type_ids(dict(encoded))
        tokenized.append({key: value.astype(np.int64, copy=False) for key, value in encoded.items()})
    return tokenized


def select_torch_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def require_onnxruntime():
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise SystemExit("Missing optional dependency 'onnxruntime'. Install it with: pip install onnxruntime") from exc
    return ort


def ort_providers(
    ort: Any,
    runtime: str,
    requested: str,
    trt_engine_cache_dir: str | None,
) -> list[Any]:
    available = set(ort.get_available_providers())
    if runtime == "tensorrt" or requested == "tensorrt":
        if "TensorrtExecutionProvider" not in available:
            raise ValueError("TensorrtExecutionProvider is not available in this onnxruntime install.")
        options = {"trt_fp16_enable": "1"}
        if trt_engine_cache_dir:
            cache_dir = Path(trt_engine_cache_dir).expanduser()
            cache_dir.mkdir(parents=True, exist_ok=True)
            options.update(
                {
                    "trt_engine_cache_enable": "1",
                    "trt_engine_cache_path": str(cache_dir),
                }
            )
        providers: list[Any] = [("TensorrtExecutionProvider", options)]
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        return providers
    if requested == "cpu":
        return ["CPUExecutionProvider"]
    if requested == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise ValueError("CUDAExecutionProvider is not available in this onnxruntime install.")
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def path_size_bytes(path: str | Path | None) -> int | None:
    if path is None:
        return None
    resolved = Path(path).expanduser()
    if not resolved.exists():
        return None
    if resolved.is_file():
        return resolved.stat().st_size
    total = 0
    for child in resolved.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def bytes_to_mb(value: int | None) -> float | None:
    if value is None:
        return None
    return value / (1024.0 * 1024.0)


def current_rss_peak_mb() -> float | None:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if usage <= 0:
        return None
    if platform.system().lower() == "darwin":
        return usage / (1024.0 * 1024.0)
    return usage / 1024.0


def process_gpu_memory_mb() -> float | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=1.0,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    pid = str(os.getpid())
    total = 0.0
    found = False
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2 or parts[0] != pid:
            continue
        try:
            total += float(parts[1])
            found = True
        except ValueError:
            continue
    return total if found else None


class ResourcePoller:
    def __init__(self, interval_s: float) -> None:
        self.interval_s = max(0.02, float(interval_s))
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.peak_process_gpu_memory_mb: float | None = None

    def start(self) -> None:
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        while not self.stop_event.is_set():
            value = process_gpu_memory_mb()
            if value is not None:
                if self.peak_process_gpu_memory_mb is None:
                    self.peak_process_gpu_memory_mb = value
                else:
                    self.peak_process_gpu_memory_mb = max(self.peak_process_gpu_memory_mb, value)
            self.stop_event.wait(self.interval_s)

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=max(1.0, self.interval_s * 2.0))


def percentile(values: list[float], percent: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * (percent / 100.0)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize_latencies(latencies_ms_per_query: list[float], total_queries: int, total_time_s: float) -> dict[str, Any]:
    throughput = total_queries / total_time_s if total_time_s > 0 else None
    return {
        "latency_scope": "model_forward_pre_tokenized_inputs",
        "latency_count": len(latencies_ms_per_query),
        "mean_latency_ms": statistics.fmean(latencies_ms_per_query) if latencies_ms_per_query else None,
        "median_latency_ms": percentile(latencies_ms_per_query, 50.0),
        "p95_latency_ms": percentile(latencies_ms_per_query, 95.0),
        "min_latency_ms": min(latencies_ms_per_query) if latencies_ms_per_query else None,
        "max_latency_ms": max(latencies_ms_per_query) if latencies_ms_per_query else None,
        "throughput_qps": throughput,
        "total_measure_time_s": total_time_s,
    }


def benchmark_pytorch(args: argparse.Namespace, prompts: list[str]) -> dict[str, Any]:
    device = select_torch_device(args.device)
    if args.precision == "fp16" and device.type != "cuda":
        raise ValueError("PyTorch FP16 benchmarking requires a CUDA device.")

    model, tokenizer, _label2response = load_scenic_checkpoint(args.checkpoint, device="cpu")
    model.to(device=device, dtype=torch.float16)
    model.eval()

    warmup = tokenize_torch_batches(tokenizer, prompt_batches(prompts, args.warmup_queries, args.batch_size), args.max_length)
    measured = tokenize_torch_batches(tokenizer, prompt_batches(prompts, args.measure_queries, args.batch_size), args.max_length)

    @torch.no_grad()
    def run_batch(batch: dict[str, torch.Tensor]) -> None:
        moved = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        model(moved)["logits"]

    for batch in warmup:
        run_batch(batch)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    poller = ResourcePoller(args.gpu_memory_poll_interval)
    poller.start()
    latencies: list[float] = []
    start_total = time.perf_counter()
    try:
        for batch in measured:
            start = time.perf_counter()
            run_batch(batch)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start
            latencies.append((elapsed * 1000.0) / max(1, int(batch["input_ids"].shape[0])))
    finally:
        poller.stop()
    total_time = time.perf_counter() - start_total

    peak_allocated = None
    peak_reserved = None
    if device.type == "cuda":
        peak_allocated = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
        peak_reserved = torch.cuda.max_memory_reserved(device) / (1024.0 * 1024.0)

    return {
        "runtime": "pytorch",
        "runtime_display": "PyTorch FP16",
        "precision": args.precision,
        "checkpoint": str(Path(args.checkpoint).expanduser()),
        "onnx": None,
        "providers": None,
        "device": str(device),
        **summarize_latencies(latencies, args.measure_queries, total_time),
        "peak_gpu_memory_mb_process": poller.peak_process_gpu_memory_mb,
        "peak_torch_cuda_allocated_mb": peak_allocated,
        "peak_torch_cuda_reserved_mb": peak_reserved,
        "peak_cpu_rss_mb": current_rss_peak_mb(),
        "source_model_size_mb": bytes_to_mb(path_size_bytes(args.checkpoint)),
        "engine_model_size_mb": None,
    }


def benchmark_onnx_like(args: argparse.Namespace, prompts: list[str]) -> dict[str, Any]:
    ort = require_onnxruntime()
    runtime = "tensorrt" if args.runtime == "tensorrt" else "onnx"
    providers = ort_providers(ort, runtime, args.providers, args.trt_engine_cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(Path(args.checkpoint).expanduser()), use_fast=True)
    warmup = tokenize_numpy_batches(tokenizer, prompt_batches(prompts, args.warmup_queries, args.batch_size), args.max_length)
    measured = tokenize_numpy_batches(tokenizer, prompt_batches(prompts, args.measure_queries, args.batch_size), args.max_length)

    session_start = time.perf_counter()
    session = ort.InferenceSession(str(Path(args.onnx).expanduser()), providers=providers)
    session_build_time_s = time.perf_counter() - session_start
    input_names = {item.name for item in session.get_inputs()}

    def run_batch(batch: dict[str, np.ndarray]) -> None:
        feed = {
            name: batch[name]
            for name in ("input_ids", "attention_mask", "token_type_ids")
            if name in input_names
        }
        session.run(["logits"], feed)

    for batch in warmup:
        run_batch(batch)

    poller = ResourcePoller(args.gpu_memory_poll_interval)
    poller.start()
    latencies: list[float] = []
    start_total = time.perf_counter()
    try:
        for batch in measured:
            start = time.perf_counter()
            run_batch(batch)
            elapsed = time.perf_counter() - start
            latencies.append((elapsed * 1000.0) / max(1, int(batch["input_ids"].shape[0])))
    finally:
        poller.stop()
    total_time = time.perf_counter() - start_total

    engine_size = None
    if runtime == "tensorrt" and args.trt_engine_cache_dir:
        engine_size = bytes_to_mb(path_size_bytes(args.trt_engine_cache_dir))

    return {
        "runtime": runtime,
        "runtime_display": "TensorRT FP16" if runtime == "tensorrt" else "ONNX Runtime FP16",
        "precision": args.precision,
        "checkpoint": str(Path(args.checkpoint).expanduser()),
        "onnx": str(Path(args.onnx).expanduser()),
        "providers": session.get_providers(),
        "device": None,
        "session_build_time_s": session_build_time_s,
        **summarize_latencies(latencies, args.measure_queries, total_time),
        "peak_gpu_memory_mb_process": poller.peak_process_gpu_memory_mb,
        "peak_torch_cuda_allocated_mb": None,
        "peak_torch_cuda_reserved_mb": None,
        "peak_cpu_rss_mb": current_rss_peak_mb(),
        "source_model_size_mb": bytes_to_mb(path_size_bytes(args.onnx)),
        "engine_model_size_mb": engine_size,
    }


def main() -> None:
    args = parse_args()
    prompts = load_prompts(args.json)
    if int(args.batch_size) != 1:
        print(
            "[edge-bench] warning: the edge report is intended for batch size 1.",
            file=sys.stderr,
        )

    if args.runtime == "pytorch":
        summary = benchmark_pytorch(args, prompts)
    else:
        summary = benchmark_onnx_like(args, prompts)

    runtime_label = args.runtime_label or summary["runtime_display"].lower().replace(" ", "_")
    summary = {
        "report_type": "scenic_edge_runtime_benchmark",
        "runtime_label": runtime_label,
        "json": str(args.json),
        "input_length": int(args.max_length),
        "batch_size": int(args.batch_size),
        "warmup_queries": int(args.warmup_queries),
        "measure_queries": int(args.measure_queries),
        **summary,
    }

    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"runtime_label: {runtime_label}")
    print(f"runtime: {summary['runtime_display']}")
    print(f"input_length: {summary['input_length']}")
    print(f"batch_size: {summary['batch_size']}")
    print(f"mean_latency_ms: {summary['mean_latency_ms']}")
    print(f"p95_latency_ms: {summary['p95_latency_ms']}")
    print(f"throughput_qps: {summary['throughput_qps']}")
    print(f"output: {output_path}")


if __name__ == "__main__":
    main()
