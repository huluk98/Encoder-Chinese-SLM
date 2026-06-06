#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import platform
import resource
import statistics
import sys
import threading
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ImportError:  # pragma: no cover - exercised only on missing envs
    np = None  # type: ignore[assignment]

try:
    import psutil
except ImportError:  # pragma: no cover - optional dependency
    psutil = None  # type: ignore[assignment]


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from chatlm_encoder.scenic_sft import prompt_from_row, read_json_list
except Exception:  # pragma: no cover - fallback for non-SCENIC use
    prompt_from_row = None  # type: ignore[assignment]
    read_json_list = None  # type: ignore[assignment]


DEFAULT_SCENIC_JSON = PROJECT_ROOT / "data/scenic/iot_instruction_benchmark_200.json"
DEFAULT_SCENIC_CHECKPOINT = PROJECT_ROOT / "runs/scenic-sft-training-dataset/latest"
MB = 1024.0 * 1024.0
EDGE_HINT_PROVIDERS = {
    "QNNExecutionProvider",
    "NnapiExecutionProvider",
    "NNAPIExecutionProvider",
    "CoreMLExecutionProvider",
    "OpenVINOExecutionProvider",
}


@dataclass
class InputSpec:
    name: str
    shape: list[Any]
    onnx_type: str


@dataclass
class PrecisionModel:
    precision: str
    path: Path
    notes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkWindow:
    start_epoch_s: float
    end_epoch_s: float
    start_relative_s: float
    end_relative_s: float


@dataclass
class BenchmarkResult:
    row: dict[str, Any]
    failure: dict[str, Any] | None = None


class ListCalibrationDataReader:
    def __init__(self, batches: list[dict[str, np.ndarray]]) -> None:
        self.batches = list(batches)
        self.index = 0

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self.index >= len(self.batches):
            return None
        value = self.batches[self.index]
        self.index += 1
        return value

    def rewind(self) -> None:
        self.index = 0


class ResourcePoller:
    def __init__(self, interval_s: float = 0.02) -> None:
        self.interval_s = max(0.01, float(interval_s))
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.peak_rss_mb: float | None = None
        self.peak_device_memory_mb: float | None = None
        self._nvml: Any | None = None
        self._nvml_handles: list[Any] = []
        self._pid = os.getpid()

    def start(self, enable_device_memory: bool) -> None:
        if enable_device_memory:
            self._init_nvml()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _init_nvml(self) -> None:
        try:
            self._nvml = importlib.import_module("pynvml")
            self._nvml.nvmlInit()
            count = int(self._nvml.nvmlDeviceGetCount())
            self._nvml_handles = [self._nvml.nvmlDeviceGetHandleByIndex(i) for i in range(count)]
        except Exception:
            self._nvml = None
            self._nvml_handles = []

    def _sample_rss_mb(self) -> float | None:
        if psutil is not None:
            try:
                return psutil.Process(self._pid).memory_info().rss / MB
            except Exception:
                return None
        return current_peak_rss_mb_resource()

    def _sample_nvml_process_memory_mb(self) -> float | None:
        if self._nvml is None or not self._nvml_handles:
            return None
        total = 0.0
        found = False
        for handle in self._nvml_handles:
            processes: list[Any] = []
            for attr in ("nvmlDeviceGetComputeRunningProcesses_v3", "nvmlDeviceGetComputeRunningProcesses"):
                getter = getattr(self._nvml, attr, None)
                if getter is None:
                    continue
                try:
                    processes = list(getter(handle))
                    break
                except Exception:
                    continue
            for proc in processes:
                if int(getattr(proc, "pid", -1)) != self._pid:
                    continue
                used = getattr(proc, "usedGpuMemory", None)
                if used is not None and used > 0:
                    total += float(used) / MB
                    found = True
        return total if found else None

    def _run(self) -> None:
        while not self.stop_event.is_set():
            rss = self._sample_rss_mb()
            if rss is not None:
                self.peak_rss_mb = rss if self.peak_rss_mb is None else max(self.peak_rss_mb, rss)
            device_memory = self._sample_nvml_process_memory_mb()
            if device_memory is not None:
                self.peak_device_memory_mb = (
                    device_memory
                    if self.peak_device_memory_mb is None
                    else max(self.peak_device_memory_mb, device_memory)
                )
            self.stop_event.wait(self.interval_s)

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=max(1.0, self.interval_s * 2.0))
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass


class PowerLog:
    def __init__(self, path: Path, timestamp_column: str, power_column: str) -> None:
        self.path = path
        self.timestamp_column = timestamp_column
        self.power_column = power_column
        self.samples = self._read_samples()

    def _read_samples(self) -> list[tuple[float, float]]:
        samples: list[tuple[float, float]] = []
        with self.path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if self.timestamp_column not in (reader.fieldnames or []):
                raise ValueError(f"{self.path} does not contain timestamp column {self.timestamp_column!r}.")
            if self.power_column not in (reader.fieldnames or []):
                raise ValueError(f"{self.path} does not contain power column {self.power_column!r}.")
            for row in reader:
                try:
                    timestamp = float(row[self.timestamp_column])
                    power = float(row[self.power_column])
                except (TypeError, ValueError):
                    continue
                if math.isfinite(timestamp) and math.isfinite(power):
                    samples.append((timestamp, power))
        samples.sort(key=lambda item: item[0])
        if len(samples) < 2:
            raise ValueError(f"{self.path} needs at least two valid power samples.")
        return samples

    def estimate(
        self,
        window: BenchmarkWindow,
        inference_count: int,
    ) -> tuple[float | None, float | None, str | None]:
        if inference_count <= 0:
            return None, None, "power unavailable: no measured inferences"
        timestamps = [item[0] for item in self.samples]
        if max(timestamps) > 1_000_000_000:
            start = window.start_epoch_s
            end = window.end_epoch_s
            clock = "epoch"
        else:
            start = window.start_relative_s
            end = window.end_relative_s
            clock = "relative"
        if start < timestamps[0] or end > timestamps[-1]:
            return (
                None,
                None,
                f"power unavailable: {clock} power log does not cover benchmark window "
                f"[{start:.3f}, {end:.3f}]",
            )

        points = [(start, self._interp(start))]
        points.extend((t, p) for t, p in self.samples if start < t < end)
        points.append((end, self._interp(end)))
        energy_j = 0.0
        for (t0, p0), (t1, p1) in zip(points, points[1:]):
            energy_j += ((p0 + p1) / 2.0) * max(0.0, t1 - t0)
        duration_s = max(0.0, end - start)
        avg_power_w = energy_j / duration_s if duration_s > 0 else None
        energy_mj = (energy_j / inference_count) * 1000.0
        return avg_power_w, energy_mj, None

    def _interp(self, timestamp: float) -> float:
        samples = self.samples
        if timestamp <= samples[0][0]:
            return samples[0][1]
        if timestamp >= samples[-1][0]:
            return samples[-1][1]
        lo = 0
        hi = len(samples) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if samples[mid][0] <= timestamp:
                lo = mid
            else:
                hi = mid
        t0, p0 = samples[lo]
        t1, p1 = samples[hi]
        if t1 == t0:
            return p0
        ratio = (timestamp - t0) / (t1 - t0)
        return p0 * (1.0 - ratio) + p1 * ratio


class InputFactory:
    def __init__(
        self,
        specs: list[InputSpec],
        input_shapes: dict[str, list[int]],
        default_shape: list[int] | None,
        batch_size: int,
        max_length: int,
        seed: int,
    ) -> None:
        self.specs = specs
        self.input_shapes = input_shapes
        self.default_shape = default_shape
        self.batch_size = max(1, int(batch_size))
        self.max_length = max(1, int(max_length))
        self.seed = int(seed)
        self.source = "deterministic_dummy_inputs"
        self.notes = ["accuracy=N/A_dummy_inputs"]

    def make_batches(self, sample_count: int) -> list[dict[str, np.ndarray]]:
        batch_count = max(1, math.ceil(max(1, int(sample_count)) / self.batch_size))
        return [self._dummy_batch(i) for i in range(batch_count)]

    def metric_batches(self, sample_count: int) -> list[tuple[dict[str, np.ndarray], list[str]]]:
        del sample_count
        return []

    def _dummy_batch(self, batch_index: int) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(self.seed + batch_index)
        feed: dict[str, np.ndarray] = {}
        for spec in self.specs:
            shape = resolve_input_shape(
                spec,
                self.input_shapes.get(spec.name),
                self.default_shape,
                self.batch_size,
                self.max_length,
            )
            dtype = numpy_dtype_for_onnx_type(spec.onnx_type)
            name_lower = spec.name.lower()
            if np.issubdtype(dtype, np.integer):
                if "mask" in name_lower:
                    value = np.ones(shape, dtype=dtype)
                elif "token_type" in name_lower or "segment" in name_lower:
                    value = np.zeros(shape, dtype=dtype)
                elif "input" in name_lower or "ids" in name_lower:
                    value = rng.integers(1, 1024, size=shape, dtype=dtype)
                else:
                    value = rng.integers(0, 16, size=shape, dtype=dtype)
            elif dtype == np.bool_:
                value = np.ones(shape, dtype=dtype)
            else:
                value = rng.normal(0.0, 1.0, size=shape).astype(dtype)
            feed[spec.name] = value
        return feed


class ScenicInputFactory(InputFactory):
    def __init__(
        self,
        specs: list[InputSpec],
        input_shapes: dict[str, list[int]],
        default_shape: list[int] | None,
        batch_size: int,
        max_length: int,
        seed: int,
        checkpoint: Path,
        json_path: Path,
        notes: list[str],
    ) -> None:
        super().__init__(specs, input_shapes, default_shape, batch_size, max_length, seed)
        self.checkpoint = checkpoint
        self.json_path = json_path
        self.notes = list(notes)
        self.source = "scenic_json_tokenizer"
        self.rows = self._load_rows()
        self.tokenizer = self._load_tokenizer()
        self.label2response = self._load_labels()

    def _load_rows(self) -> list[dict[str, Any]]:
        if read_json_list is None or prompt_from_row is None:
            raise RuntimeError("chatlm_encoder.scenic_sft helpers are unavailable.")
        rows: list[dict[str, Any]] = []
        for index, row in enumerate(read_json_list(self.json_path)):
            prompt = prompt_from_row(row)
            if not prompt:
                continue
            rows.append(
                {
                    "index": index,
                    "prompt": prompt,
                    "expected_response": "" if row.get("response") is None else str(row.get("response")).strip(),
                    "raw": row,
                }
            )
        if not rows:
            raise ValueError(f"No prompt-like rows found in {self.json_path}.")
        return rows

    def _load_tokenizer(self) -> Any:
        try:
            transformers = importlib.import_module("transformers")
        except ImportError as exc:
            raise SystemExit(
                "Missing dependency 'transformers' needed for SCENIC dataset inputs. "
                "Install it with: pip install transformers"
            ) from exc
        return transformers.AutoTokenizer.from_pretrained(str(self.checkpoint), use_fast=True)

    def _load_labels(self) -> list[str] | None:
        path = self.checkpoint / "label2response.json"
        if not path.exists():
            self.notes.append(f"task accuracy unavailable: missing {path}")
            return None
        labels = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(labels, list):
            self.notes.append(f"task accuracy unavailable: {path} is not a JSON list")
            return None
        return [str(item) for item in labels]

    def make_batches(self, sample_count: int) -> list[dict[str, np.ndarray]]:
        rows = self._rows_for_sample_count(sample_count)
        return [self._tokenize_batch(rows[start : start + self.batch_size]) for start in range(0, len(rows), self.batch_size)]

    def metric_batches(self, sample_count: int) -> list[tuple[dict[str, np.ndarray], list[str]]]:
        if self.label2response is None:
            return []
        rows = self._rows_for_sample_count(sample_count)
        batches: list[tuple[dict[str, np.ndarray], list[str]]] = []
        for start in range(0, len(rows), self.batch_size):
            batch_rows = rows[start : start + self.batch_size]
            expected = [str(item.get("expected_response") or "") for item in batch_rows]
            batches.append((self._tokenize_batch(batch_rows), expected))
        return batches

    def _rows_for_sample_count(self, sample_count: int) -> list[dict[str, Any]]:
        count = max(1, int(sample_count))
        return [self.rows[index % len(self.rows)] for index in range(count)]

    def _tokenize_batch(self, rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
        prompts = [item["prompt"] for item in rows]
        encoded = self.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=max(8, int(self.max_length)),
            return_tensors="np",
        )
        values = dict(encoded)
        if "token_type_ids" not in values:
            values["token_type_ids"] = np.zeros_like(values["input_ids"], dtype=np.int64)
        feed: dict[str, np.ndarray] = {}
        for spec in self.specs:
            if spec.name in values:
                feed[spec.name] = values[spec.name].astype(numpy_dtype_for_onnx_type(spec.onnx_type), copy=False)
                continue
            fallback = self._dummy_batch(0)[spec.name]
            if fallback.shape[0] != len(rows):
                fallback = fallback[: len(rows)]
            feed[spec.name] = fallback
            if f"dummy_fallback:{spec.name}" not in self.notes:
                self.notes.append(f"dummy_fallback:{spec.name}")
        return feed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark FP32, FP16, and INT8 ONNX inference across ONNX Runtime "
            "Execution Providers and write paper-ready SCENIC precision tables."
        )
    )
    parser.add_argument("--fp32-onnx", required=True, help="Path to the FP32 baseline ONNX model.")
    parser.add_argument("--output-dir", required=True, help="Directory for tables, metadata, and generated models.")
    parser.add_argument("--providers", nargs="+", default=["CPUExecutionProvider"], help="ORT providers to try.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--calibration-samples", type=int, default=128)
    parser.add_argument("--table-formats", nargs="+", default=["csv", "markdown", "latex"], choices=("csv", "markdown", "latex"))
    parser.add_argument("--fp16-onnx", default=None, help="Existing FP16 ONNX model path.")
    parser.add_argument("--int8-onnx", default=None, help="Existing INT8 ONNX model path.")
    parser.add_argument("--skip-fp16-conversion", action="store_true")
    parser.add_argument("--skip-int8-quantization", action="store_true")
    parser.add_argument("--quantization-mode", default="static", choices=("static", "dynamic"))
    parser.add_argument("--quant-format", default="qdq", choices=("qdq", "qoperator"))
    parser.add_argument("--power-log", default=None, help="CSV containing real measured power readings.")
    parser.add_argument("--power-column", default="power_w")
    parser.add_argument("--timestamp-column", default="timestamp_s")
    parser.add_argument("--device-name", default=None, help="Human-readable measured device name.")
    parser.add_argument(
        "--input-shape",
        nargs="*",
        default=None,
        help=(
            "Fallback dummy input shape. Use either '1,128' for all dynamic inputs or "
            "name=1,128 entries for specific inputs."
        ),
    )
    parser.add_argument("--num-threads", type=int, default=None, help="Set ORT intra/inter op thread counts.")
    parser.add_argument("--disable-iobinding", action="store_true")
    parser.add_argument("--profile-ort", action="store_true")
    parser.add_argument("--checkpoint", default=None, help="SCENIC checkpoint for tokenizer and label mapping.")
    parser.add_argument("--json", default=None, help="SCENIC JSON list used for benchmark/calibration prompts.")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--metric-samples", type=int, default=200)
    parser.add_argument("--drift-samples", type=int, default=32)
    parser.add_argument("--input-seed", type=int, default=13)
    parser.add_argument("--rss-poll-interval", type=float, default=0.02)
    return parser.parse_args()


def require_module(name: str, package_hint: str | None = None) -> Any:
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        package = package_hint or name
        raise SystemExit(f"Missing required dependency {name!r}. Install it with: pip install {package}") from exc


def optional_module(name: str) -> Any | None:
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


def load_onnxruntime() -> Any:
    return require_module("onnxruntime", "onnxruntime")


def format_provider_name(name: str) -> str:
    aliases = {
        "cpu": "CPUExecutionProvider",
        "cuda": "CUDAExecutionProvider",
        "gpu": "CUDAExecutionProvider",
        "trt": "TensorrtExecutionProvider",
        "tensorrt": "TensorrtExecutionProvider",
        "nnapi": "NnapiExecutionProvider",
        "nnapiexecutionprovider": "NnapiExecutionProvider",
        "nnapiexecutionprovider".lower(): "NnapiExecutionProvider",
        "qnn": "QNNExecutionProvider",
        "coreml": "CoreMLExecutionProvider",
        "openvino": "OpenVINOExecutionProvider",
    }
    key = name.strip()
    return aliases.get(key.lower(), key)


def normalize_provider(requested: str, available: list[str]) -> str:
    provider = format_provider_name(requested)
    if provider in available:
        return provider
    if provider == "NnapiExecutionProvider" and "NNAPIExecutionProvider" in available:
        return "NNAPIExecutionProvider"
    if provider == "NNAPIExecutionProvider" and "NnapiExecutionProvider" in available:
        return "NnapiExecutionProvider"
    return provider


def parse_input_shapes(values: list[str] | None) -> tuple[dict[str, list[int]], list[int] | None]:
    named: dict[str, list[int]] = {}
    default: list[int] | None = None
    for value in values or []:
        if not value:
            continue
        if "=" in value:
            name, shape_text = value.split("=", 1)
            named[name.strip()] = parse_shape(shape_text)
        else:
            default = parse_shape(value)
    return named, default


def parse_shape(value: str) -> list[int]:
    shape: list[int] = []
    for part in value.replace("x", ",").split(","):
        part = part.strip()
        if not part:
            continue
        dim = int(part)
        if dim <= 0:
            raise ValueError(f"Input shape dimensions must be positive integers, got {value!r}.")
        shape.append(dim)
    if not shape:
        raise ValueError(f"Could not parse input shape {value!r}.")
    return shape


def get_input_specs(ort: Any, fp32_path: Path) -> list[InputSpec]:
    providers = ort.get_available_providers()
    provider_chain = ["CPUExecutionProvider"] if "CPUExecutionProvider" in providers else providers[:1]
    if not provider_chain:
        raise RuntimeError("No ONNX Runtime Execution Providers are available.")
    session = ort.InferenceSession(str(fp32_path), providers=provider_chain)
    return [InputSpec(item.name, list(item.shape), str(item.type)) for item in session.get_inputs()]


def numpy_dtype_for_onnx_type(onnx_type: str) -> Any:
    mapping = {
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
        "tensor(double)": np.float64,
        "tensor(int64)": np.int64,
        "tensor(int32)": np.int32,
        "tensor(int16)": np.int16,
        "tensor(int8)": np.int8,
        "tensor(uint64)": np.uint64,
        "tensor(uint32)": np.uint32,
        "tensor(uint16)": np.uint16,
        "tensor(uint8)": np.uint8,
        "tensor(bool)": np.bool_,
    }
    return mapping.get(onnx_type, np.float32)


def resolve_input_shape(
    spec: InputSpec,
    explicit_shape: list[int] | None,
    default_shape: list[int] | None,
    batch_size: int,
    max_length: int,
) -> tuple[int, ...]:
    if explicit_shape is not None:
        return tuple(explicit_shape)
    if default_shape is not None:
        return tuple(default_shape)
    resolved: list[int] = []
    for index, dim in enumerate(spec.shape):
        if isinstance(dim, int) and dim > 0:
            resolved.append(dim)
        elif index == 0:
            resolved.append(max(1, int(batch_size)))
        elif len(spec.shape) == 2:
            resolved.append(max(1, int(max_length)))
        else:
            resolved.append(1)
    if not resolved:
        resolved = [max(1, int(batch_size))]
    resolved[0] = max(1, int(batch_size))
    return tuple(resolved)


def discover_scenic_paths(args: argparse.Namespace, notes: list[str]) -> tuple[Path | None, Path | None]:
    checkpoint = Path(args.checkpoint).expanduser() if args.checkpoint else None
    json_path = Path(args.json).expanduser() if args.json else None
    if checkpoint is None and DEFAULT_SCENIC_CHECKPOINT.exists():
        checkpoint = DEFAULT_SCENIC_CHECKPOINT
        notes.append(f"auto_discovered_checkpoint={checkpoint}")
    if json_path is None and DEFAULT_SCENIC_JSON.exists():
        json_path = DEFAULT_SCENIC_JSON
        notes.append(f"auto_discovered_json={json_path}")
    if checkpoint is not None and not checkpoint.exists():
        notes.append(f"SCENIC checkpoint unavailable; using dummy inputs: {checkpoint}")
        checkpoint = None
    if json_path is not None and not json_path.exists():
        notes.append(f"SCENIC JSON unavailable; using dummy inputs: {json_path}")
        json_path = None
    return checkpoint, json_path


def make_input_factory(
    args: argparse.Namespace,
    specs: list[InputSpec],
    input_shapes: dict[str, list[int]],
    default_shape: list[int] | None,
    notes: list[str],
) -> InputFactory:
    checkpoint, json_path = discover_scenic_paths(args, notes)
    if checkpoint is not None and json_path is not None:
        try:
            return ScenicInputFactory(
                specs=specs,
                input_shapes=input_shapes,
                default_shape=default_shape,
                batch_size=int(args.batch_size),
                max_length=int(args.max_length),
                seed=int(args.input_seed),
                checkpoint=checkpoint,
                json_path=json_path,
                notes=notes,
            )
        except Exception as exc:
            notes.append(f"SCENIC input loading failed; using dummy inputs: {exc}")
    return InputFactory(
        specs=specs,
        input_shapes=input_shapes,
        default_shape=default_shape,
        batch_size=int(args.batch_size),
        max_length=int(args.max_length),
        seed=int(args.input_seed),
    )


def validate_onnx_model(path: Path) -> None:
    onnx = require_module("onnx", "onnx")
    onnx.checker.check_model(str(path))


def convert_fp16_model(fp32_path: Path, output_dir: Path) -> tuple[Path, list[str], dict[str, Any]]:
    onnx = require_module("onnx", "onnx")
    try:
        float16 = importlib.import_module("onnxconverter_common.float16")
    except ImportError as exc:
        raise SystemExit(
            "Missing required dependency 'onnxconverter_common'. "
            "Install it with: pip install onnxconverter-common"
        ) from exc
    output_path = output_dir / f"{fp32_path.stem}_fp16.onnx"
    notes: list[str] = []
    metadata = {"conversion": "onnxconverter_common.float16.convert_float_to_float16", "keep_io_types": True}
    model = onnx.load(str(fp32_path))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            converted = float16.convert_float_to_float16(model, keep_io_types=True)
        except Exception as exc:
            notes.append(f"FP16 keep_io_types=True failed; retried with keep_io_types=False: {exc}")
            metadata["keep_io_types"] = False
            converted = float16.convert_float_to_float16(model, keep_io_types=False)
        for warning in caught:
            notes.append(f"FP16 conversion warning: {warning.message}")
    onnx.save(converted, str(output_path))
    validate_onnx_model(output_path)
    metadata["output"] = str(output_path)
    return output_path, notes, metadata


def quantize_int8_model(
    args: argparse.Namespace,
    fp32_path: Path,
    output_dir: Path,
    calibration_batches: list[dict[str, np.ndarray]],
) -> tuple[Path, list[str], dict[str, Any]]:
    onnx = require_module("onnx", "onnx")
    quant = require_module("onnxruntime.quantization", "onnxruntime")
    output_path = output_dir / f"{fp32_path.stem}_int8.onnx"
    notes: list[str] = []
    metadata: dict[str, Any] = {
        "quantization_mode": args.quantization_mode,
        "quantization_format": args.quant_format,
        "calibration_batches": len(calibration_batches),
        "calibration_samples_requested": int(args.calibration_samples),
        "activation_type": "QInt8",
        "weight_type": "QInt8",
    }
    if args.quantization_mode == "dynamic":
        quant.quantize_dynamic(
            model_input=str(fp32_path),
            model_output=str(output_path),
            weight_type=quant.QuantType.QInt8,
            per_channel=True,
            reduce_range=False,
        )
    else:
        model_input = fp32_path
        preprocessed_path = output_dir / f"{fp32_path.stem}_quant_preprocess.onnx"
        quant_pre_process = getattr(quant, "quant_pre_process", None)
        if quant_pre_process is not None:
            try:
                quant_pre_process(str(fp32_path), str(preprocessed_path))
                model_input = preprocessed_path
                metadata["quant_pre_process"] = str(preprocessed_path)
            except Exception as exc:
                notes.append(f"quant_pre_process failed; static quantization used original graph: {exc}")
        else:
            notes.append("quant_pre_process unavailable in this onnxruntime build")
        format_value = quant.QuantFormat.QDQ if args.quant_format == "qdq" else quant.QuantFormat.QOperator
        reader = ListCalibrationDataReader(calibration_batches)
        quant.quantize_static(
            model_input=str(model_input),
            model_output=str(output_path),
            calibration_data_reader=reader,
            quant_format=format_value,
            activation_type=quant.QuantType.QInt8,
            weight_type=quant.QuantType.QInt8,
            per_channel=True,
        )
    onnx.checker.check_model(str(output_path))
    metadata["output"] = str(output_path)
    return output_path, notes, metadata


def build_precision_models(
    args: argparse.Namespace,
    fp32_path: Path,
    output_dir: Path,
    calibration_batches: list[dict[str, np.ndarray]],
    failures: list[dict[str, Any]],
) -> list[PrecisionModel]:
    models = [PrecisionModel("FP32", fp32_path, metadata={"source": "provided_fp32"})]
    if args.fp16_onnx:
        path = Path(args.fp16_onnx).expanduser()
        if path.exists():
            models.append(PrecisionModel("FP16", path, metadata={"source": "provided_fp16"}))
        else:
            failures.append({"precision": "FP16", "stage": "load", "reason": f"missing --fp16-onnx path: {path}"})
    elif not args.skip_fp16_conversion:
        try:
            path, notes, metadata = convert_fp16_model(fp32_path, output_dir)
            models.append(PrecisionModel("FP16", path, notes=notes, metadata=metadata))
        except Exception as exc:
            failures.append({"precision": "FP16", "stage": "conversion", "reason": str(exc)})

    if args.int8_onnx:
        path = Path(args.int8_onnx).expanduser()
        if path.exists():
            models.append(PrecisionModel("INT8", path, metadata={"source": "provided_int8"}))
        else:
            failures.append({"precision": "INT8", "stage": "load", "reason": f"missing --int8-onnx path: {path}"})
    elif not args.skip_int8_quantization:
        try:
            path, notes, metadata = quantize_int8_model(args, fp32_path, output_dir, calibration_batches)
            models.append(PrecisionModel("INT8", path, notes=notes, metadata=metadata))
        except Exception as exc:
            failures.append({"precision": "INT8", "stage": "quantization", "reason": str(exc)})
    return models


def make_session_options(args: argparse.Namespace, output_dir: Path, precision: str, provider: str) -> Any:
    ort = load_onnxruntime()
    options = ort.SessionOptions()
    if args.num_threads is not None:
        options.intra_op_num_threads = max(1, int(args.num_threads))
        options.inter_op_num_threads = max(1, int(args.num_threads))
    if args.profile_ort:
        safe_provider = provider.replace("ExecutionProvider", "").replace("/", "_")
        options.enable_profiling = True
        options.profile_file_prefix = str(output_dir / f"ort_profile_{precision.lower()}_{safe_provider.lower()}")
    return options


def provider_chain_for(provider: str, available: list[str]) -> list[str]:
    if provider == "CPUExecutionProvider":
        return [provider]
    chain = [provider]
    if "CPUExecutionProvider" in available:
        chain.append("CPUExecutionProvider")
    return chain


def run_regular(session: Any, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
    return session.run(None, feed)


def make_runner(session: Any, provider: str, args: argparse.Namespace, notes: list[str]) -> Any:
    if args.disable_iobinding or provider == "CPUExecutionProvider":
        if args.disable_iobinding:
            notes.append("iobinding disabled")
        return lambda feed: run_regular(session, feed)

    output_names = [item.name for item in session.get_outputs()]

    def run_iobinding(feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        binding = session.io_binding()
        for name, value in feed.items():
            binding.bind_cpu_input(name, value)
        for name in output_names:
            if provider in {"CUDAExecutionProvider", "TensorrtExecutionProvider"}:
                binding.bind_output(name, "cuda")
            else:
                binding.bind_output(name)
        session.run_with_iobinding(binding)
        return binding.copy_outputs_to_cpu()

    tested = False
    use_iobinding = True

    def guarded(feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        nonlocal tested, use_iobinding
        if not tested and use_iobinding:
            try:
                outputs = run_iobinding(feed)
                notes.append("iobinding enabled")
                tested = True
                return outputs
            except Exception as exc:
                notes.append(f"iobinding fallback to regular session.run: {exc}")
                tested = True
                use_iobinding = False
                return run_regular(session, feed)
        if use_iobinding:
            try:
                return run_iobinding(feed)
            except Exception as exc:
                notes.append(f"iobinding disabled after runtime failure: {exc}")
                use_iobinding = False
        return run_regular(session, feed)

    return guarded


def synchronize_provider(provider: str) -> None:
    if provider not in {"CUDAExecutionProvider", "TensorrtExecutionProvider"}:
        return
    try:
        torch = importlib.import_module("torch")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        return


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


def current_peak_rss_mb_resource() -> float | None:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if usage <= 0:
        return None
    if platform.system().lower() == "darwin":
        return usage / MB
    return usage / 1024.0


def flatten_numeric_outputs(outputs: list[Any]) -> np.ndarray:
    flattened: list[np.ndarray] = []
    for output in outputs:
        array = np.asarray(output)
        if not np.issubdtype(array.dtype, np.number):
            continue
        flattened.append(array.astype(np.float64, copy=False).reshape(-1))
    if not flattened:
        return np.asarray([], dtype=np.float64)
    return np.concatenate(flattened)


def compute_drift_metrics(
    reference_session: Any,
    test_session: Any,
    batches: list[dict[str, np.ndarray]],
) -> tuple[dict[str, float | None], list[str]]:
    notes: list[str] = []
    sum_abs = 0.0
    max_abs = 0.0
    dot = 0.0
    ref_norm = 0.0
    test_norm = 0.0
    count = 0
    for feed in batches:
        ref = flatten_numeric_outputs(run_regular(reference_session, feed))
        test = flatten_numeric_outputs(run_regular(test_session, feed))
        if ref.shape != test.shape:
            notes.append(f"drift shape mismatch: fp32={tuple(ref.shape)} test={tuple(test.shape)}")
            size = min(ref.size, test.size)
            ref = ref[:size]
            test = test[:size]
        if ref.size == 0:
            continue
        diff = np.abs(ref - test)
        sum_abs += float(diff.sum())
        max_abs = max(max_abs, float(diff.max()))
        dot += float(np.dot(ref, test))
        ref_norm += float(np.dot(ref, ref))
        test_norm += float(np.dot(test, test))
        count += int(ref.size)
    if count == 0:
        return {"mean_abs_error": None, "max_abs_error": None, "cosine_similarity": None}, notes
    denom = math.sqrt(ref_norm) * math.sqrt(test_norm)
    cosine = dot / denom if denom > 0 else None
    return {
        "mean_abs_error": sum_abs / count,
        "max_abs_error": max_abs,
        "cosine_similarity": cosine,
    }, notes


def compute_task_metrics(
    session: Any,
    metric_batches: list[tuple[dict[str, np.ndarray], list[str]]],
    label2response: list[str] | None,
) -> dict[str, Any]:
    if not metric_batches or not label2response:
        return {
            "task_metric_name": None,
            "task_metric_value": None,
            "top5_accuracy": None,
            "label_space_coverage": None,
        }
    label_set = set(label2response)
    total = 0
    scored = 0
    covered = 0
    correct = 0
    top5_correct = 0
    for feed, expected_values in metric_batches:
        logits = np.asarray(run_regular(session, feed)[0])
        if logits.ndim == 1:
            logits = logits.reshape(1, -1)
        top_k = min(5, logits.shape[-1])
        top_indices = np.argpartition(-logits, kth=top_k - 1, axis=-1)[:, :top_k]
        top_scores = np.take_along_axis(logits, top_indices, axis=-1)
        order = np.argsort(-top_scores, axis=-1)
        top_indices = np.take_along_axis(top_indices, order, axis=-1)
        for expected, ids in zip(expected_values, top_indices.tolist()):
            total += 1
            if not expected:
                continue
            scored += 1
            top_responses = [label2response[int(index)] for index in ids if int(index) < len(label2response)]
            predicted = top_responses[0] if top_responses else ""
            covered += int(expected in label_set)
            correct += int(predicted == expected)
            top5_correct += int(expected in set(top_responses))
    return {
        "task_metric_name": "exact_match_accuracy" if scored else None,
        "task_metric_value": correct / scored if scored else None,
        "top5_accuracy": top5_correct / scored if scored else None,
        "label_space_coverage": covered / scored if scored else None,
        "metric_rows": total,
        "metric_scored_rows": scored,
    }


def inspect_profile_for_fallback(profile_path: str | None, requested_provider: str) -> str | None:
    if not profile_path or not Path(profile_path).exists() or requested_provider == "CPUExecutionProvider":
        return None
    try:
        events = json.loads(Path(profile_path).read_text(encoding="utf-8"))
    except Exception:
        return None
    providers: set[str] = set()
    for event in events:
        args = event.get("args") if isinstance(event, dict) else None
        if isinstance(args, dict):
            provider = args.get("provider")
            if provider:
                providers.add(str(provider))
    if "CPUExecutionProvider" in providers and requested_provider != "CPUExecutionProvider":
        return "provider fallback detected in ORT profile: CPUExecutionProvider executed one or more nodes"
    return None


def benchmark_model_provider(
    args: argparse.Namespace,
    ort: Any,
    model: PrecisionModel,
    provider: str,
    requested_provider: str,
    available_providers: list[str],
    output_dir: Path,
    input_factory: InputFactory,
    benchmark_batches: list[dict[str, np.ndarray]],
    drift_batches: list[dict[str, np.ndarray]],
    metric_batches: list[tuple[dict[str, np.ndarray], list[str]]],
    fp32_path: Path,
    power_log: PowerLog | None,
    script_start_epoch_s: float,
    script_start_perf_s: float,
) -> BenchmarkResult:
    notes = list(model.notes) + list(input_factory.notes)
    provider_chain = provider_chain_for(provider, available_providers)
    if provider not in available_providers:
        return BenchmarkResult(
            row={},
            failure={
                "precision": model.precision,
                "provider_requested": requested_provider,
                "provider_normalized": provider,
                "stage": "provider_availability",
                "reason": f"{provider} is not available. Available providers: {available_providers}",
            },
        )
    try:
        session = ort.InferenceSession(
            str(model.path),
            sess_options=make_session_options(args, output_dir, model.precision, provider),
            providers=provider_chain,
        )
    except Exception as exc:
        return BenchmarkResult(
            row={},
            failure={
                "precision": model.precision,
                "provider_requested": requested_provider,
                "provider_normalized": provider,
                "stage": "session_create",
                "reason": str(exc),
            },
        )
    actual_providers = session.get_providers()
    if not actual_providers or actual_providers[0] != provider:
        notes.append(f"provider fallback possible: requested {provider}, session providers={actual_providers}")
    runner = make_runner(session, provider, args, notes)

    warmup_batches = benchmark_batches[: max(0, int(args.warmup))]
    measured_batches = benchmark_batches[max(0, int(args.warmup)) :]
    try:
        for feed in warmup_batches:
            runner(feed)
        synchronize_provider(provider)
    except Exception as exc:
        return BenchmarkResult(
            row={},
            failure={
                "precision": model.precision,
                "provider_requested": requested_provider,
                "provider_normalized": provider,
                "stage": "warmup",
                "reason": str(exc),
            },
        )

    poller = ResourcePoller(args.rss_poll_interval)
    enable_device_memory = provider in {"CUDAExecutionProvider", "TensorrtExecutionProvider"}
    poller.start(enable_device_memory=enable_device_memory)
    latencies_ms: list[float] = []
    start_epoch = time.time()
    start_perf = time.perf_counter()
    try:
        for feed in measured_batches:
            batch_size = int(next(iter(feed.values())).shape[0]) if feed else int(args.batch_size)
            start_ns = time.perf_counter_ns()
            runner(feed)
            synchronize_provider(provider)
            elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
            latencies_ms.append(elapsed_ms / max(1, batch_size))
    except Exception as exc:
        poller.stop()
        return BenchmarkResult(
            row={},
            failure={
                "precision": model.precision,
                "provider_requested": requested_provider,
                "provider_normalized": provider,
                "stage": "timed_runs",
                "reason": str(exc),
            },
        )
    finally:
        end_perf = time.perf_counter()
        end_epoch = time.time()
    poller.stop()

    inference_count = sum(
        int(next(iter(feed.values())).shape[0]) if feed else max(1, int(args.batch_size))
        for feed in measured_batches
    )
    elapsed_s = max(0.0, end_perf - start_perf)
    throughput = inference_count / elapsed_s if elapsed_s > 0 else None

    reference_session = None
    drift = {"mean_abs_error": None, "max_abs_error": None, "cosine_similarity": None}
    if model.precision == "FP32" and model.path == fp32_path:
        drift = {"mean_abs_error": 0.0, "max_abs_error": 0.0, "cosine_similarity": 1.0}
    else:
        try:
            reference_session = ort.InferenceSession(str(fp32_path), providers=provider_chain)
        except Exception:
            cpu_chain = ["CPUExecutionProvider"] if "CPUExecutionProvider" in available_providers else available_providers[:1]
            try:
                reference_session = ort.InferenceSession(str(fp32_path), providers=cpu_chain)
                notes.append("drift reference used CPU FP32 because provider-specific FP32 reference failed")
            except Exception as exc:
                notes.append(f"drift unavailable: FP32 reference session failed: {exc}")
        if reference_session is not None:
            try:
                drift, drift_notes = compute_drift_metrics(reference_session, session, drift_batches)
                notes.extend(drift_notes)
            except Exception as exc:
                notes.append(f"drift unavailable: {exc}")

    task_metrics: dict[str, Any]
    try:
        task_metrics = compute_task_metrics(
            session,
            metric_batches,
            getattr(input_factory, "label2response", None),
        )
    except Exception as exc:
        task_metrics = {
            "task_metric_name": None,
            "task_metric_value": None,
            "top5_accuracy": None,
            "label_space_coverage": None,
        }
        notes.append(f"task metric unavailable: {exc}")

    profile_path = None
    try:
        profile_path = session.end_profiling() if args.profile_ort else None
    except Exception as exc:
        notes.append(f"ORT profile close failed: {exc}")
    fallback_note = inspect_profile_for_fallback(profile_path, provider)
    if fallback_note:
        notes.append(fallback_note)

    average_power_w = None
    energy_mj = None
    window = BenchmarkWindow(
        start_epoch_s=start_epoch,
        end_epoch_s=end_epoch,
        start_relative_s=start_perf - script_start_perf_s,
        end_relative_s=end_perf - script_start_perf_s,
    )
    if power_log is None:
        notes.append("energy unavailable: no --power-log was provided")
    else:
        average_power_w, energy_mj, power_note = power_log.estimate(window, inference_count)
        if power_note:
            notes.append(power_note)

    row = {
        "precision": model.precision,
        "onnx_model_path": str(model.path),
        "onnx_file_size_mb": path_size_mb(model.path),
        "execution_provider_requested": requested_provider,
        "execution_provider_used": actual_providers[0] if actual_providers else None,
        "provider_chain": ",".join(actual_providers),
        "hardware_device": args.device_name or platform.node() or "unknown",
        "hardware_metadata_summary": hardware_summary(args.device_name, provider),
        "mean_latency_ms": statistics.fmean(latencies_ms) if latencies_ms else None,
        "median_latency_ms": percentile(latencies_ms, 50.0),
        "p90_latency_ms": percentile(latencies_ms, 90.0),
        "p95_latency_ms": percentile(latencies_ms, 95.0),
        "throughput_samples_s": throughput,
        "peak_host_memory_rss_mb": poller.peak_rss_mb or current_peak_rss_mb_resource(),
        "peak_device_memory_mb": poller.peak_device_memory_mb,
        "average_power_w": average_power_w,
        "energy_per_inference_mj": energy_mj,
        "task_metric_name": task_metrics.get("task_metric_name"),
        "task_metric_value": task_metrics.get("task_metric_value"),
        "top5_accuracy": task_metrics.get("top5_accuracy"),
        "label_space_coverage": task_metrics.get("label_space_coverage"),
        "mean_abs_error_vs_fp32": drift["mean_abs_error"],
        "max_abs_error_vs_fp32": drift["max_abs_error"],
        "cosine_similarity_vs_fp32": drift["cosine_similarity"],
        "speedup_vs_fp32": None,
        "size_reduction_vs_fp32_pct": None,
        "warmup_iterations": int(args.warmup),
        "timed_iterations": int(args.runs),
        "batch_size": int(args.batch_size),
        "benchmark_start_epoch_s": start_epoch,
        "benchmark_end_epoch_s": end_epoch,
        "ort_profile": profile_path,
        "notes": "; ".join(unique_preserve_order(notes)) if notes else "",
    }
    return BenchmarkResult(row=row)


def path_size_mb(path: Path) -> float | None:
    try:
        return path.stat().st_size / MB
    except OSError:
        return None


def hardware_summary(device_name: str | None, provider: str) -> str:
    provider_hint = "edge/accelerator provider" if provider in EDGE_HINT_PROVIDERS else provider
    if device_name:
        return f"user-declared device={device_name}; provider={provider_hint}"
    return f"platform={platform.system()} {platform.machine()}; provider={provider_hint}; edge hardware not user-declared"


def unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def add_relative_metrics(rows: list[dict[str, Any]], fp32_size_mb: float | None) -> None:
    baselines: dict[str, float] = {}
    for row in rows:
        if row.get("precision") == "FP32" and row.get("mean_latency_ms") is not None:
            baselines[str(row.get("execution_provider_requested"))] = float(row["mean_latency_ms"])
    global_fp32 = next((float(row["mean_latency_ms"]) for row in rows if row.get("precision") == "FP32" and row.get("mean_latency_ms") is not None), None)
    for row in rows:
        latency = row.get("mean_latency_ms")
        provider = str(row.get("execution_provider_requested"))
        baseline = baselines.get(provider, global_fp32)
        if latency is not None and baseline and float(latency) > 0:
            row["speedup_vs_fp32"] = baseline / float(latency)
        size = row.get("onnx_file_size_mb")
        if size is not None and fp32_size_mb and fp32_size_mb > 0:
            row["size_reduction_vs_fp32_pct"] = (1.0 - (float(size) / fp32_size_mb)) * 100.0


def collect_hardware_metadata(args: argparse.Namespace, ort: Any, notes: list[str]) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "device_name_user_provided": args.device_name,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "node": platform.node(),
            "python": sys.version,
        },
        "onnxruntime": {
            "version": getattr(ort, "__version__", None),
            "available_providers": ort.get_available_providers(),
        },
        "process": {
            "pid": os.getpid(),
            "cpu_count": os.cpu_count(),
        },
        "edge_measurement_interpretation": edge_interpretation(args.device_name, args.providers),
        "notes": notes,
    }
    if psutil is not None:
        try:
            memory = psutil.virtual_memory()
            metadata["host_memory"] = {
                "total_mb": memory.total / MB,
                "available_mb": memory.available / MB,
            }
        except Exception:
            pass
    else:
        metadata["host_memory"] = {"warning": "psutil unavailable; install with: pip install psutil"}
    pynvml = optional_module("pynvml")
    if pynvml is not None:
        try:
            pynvml.nvmlInit()
            gpus = []
            for index in range(int(pynvml.nvmlDeviceGetCount())):
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode("utf-8", errors="replace")
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpus.append({"index": index, "name": name, "total_memory_mb": mem.total / MB})
            metadata["cuda_nvml_devices"] = gpus
            pynvml.nvmlShutdown()
        except Exception as exc:
            metadata["cuda_nvml_devices"] = {"warning": str(exc)}
    else:
        metadata["cuda_nvml_devices"] = {"warning": "pynvml unavailable; GPU memory is optional. Install with: pip install pynvml"}
    return metadata


def edge_interpretation(device_name: str | None, providers: list[str]) -> str:
    normalized = {format_provider_name(provider) for provider in providers}
    if device_name and normalized.intersection(EDGE_HINT_PROVIDERS):
        return "edge/accelerator evidence: user-declared device name and edge-oriented provider requested"
    if device_name:
        return "device name was user-declared; edge status is not independently verified by this script"
    if normalized.intersection(EDGE_HINT_PROVIDERS):
        return "edge-oriented provider requested; hardware identity should be verified from device metadata"
    return "no actual edge hardware was declared; treat results as host platform measurements"


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


TABLE_COLUMNS = [
    "precision",
    "onnx_model_path",
    "onnx_file_size_mb",
    "execution_provider_requested",
    "execution_provider_used",
    "provider_chain",
    "hardware_device",
    "hardware_metadata_summary",
    "mean_latency_ms",
    "median_latency_ms",
    "p90_latency_ms",
    "p95_latency_ms",
    "throughput_samples_s",
    "peak_host_memory_rss_mb",
    "peak_device_memory_mb",
    "average_power_w",
    "energy_per_inference_mj",
    "task_metric_value",
    "mean_abs_error_vs_fp32",
    "max_abs_error_vs_fp32",
    "cosine_similarity_vs_fp32",
    "speedup_vs_fp32",
    "size_reduction_vs_fp32_pct",
    "notes",
]


def display_value(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "N/A"
        if abs(value) >= 100:
            return f"{value:.2f}"
        if abs(value) >= 1:
            return f"{value:.4f}"
        return f"{value:.6g}"
    return str(value)


def write_csv_table(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TABLE_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: display_value(row.get(key)) for key in TABLE_COLUMNS})


def escape_markdown(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def write_markdown_table(path: Path, rows: list[dict[str, Any]]) -> None:
    headers = TABLE_COLUMNS
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(escape_markdown(display_value(row.get(key))) for key in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def escape_latex(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in value)


def write_latex_table(path: Path, rows: list[dict[str, Any]]) -> None:
    compact_columns = [
        "precision",
        "execution_provider_used",
        "onnx_file_size_mb",
        "mean_latency_ms",
        "p95_latency_ms",
        "throughput_samples_s",
        "peak_host_memory_rss_mb",
        "average_power_w",
        "energy_per_inference_mj",
        "task_metric_value",
        "mean_abs_error_vs_fp32",
        "speedup_vs_fp32",
        "size_reduction_vs_fp32_pct",
        "notes",
    ]
    labels = {
        "precision": "Precision",
        "execution_provider_used": "Provider",
        "onnx_file_size_mb": "Size MB",
        "mean_latency_ms": "Mean ms",
        "p95_latency_ms": "p95 ms",
        "throughput_samples_s": "Samples/s",
        "peak_host_memory_rss_mb": "Host RSS MB",
        "average_power_w": "Power W",
        "energy_per_inference_mj": "Energy mJ",
        "task_metric_value": "Task metric",
        "mean_abs_error_vs_fp32": "MAE vs FP32",
        "speedup_vs_fp32": "Speedup",
        "size_reduction_vs_fp32_pct": "Size red. \\%",
        "notes": "Notes",
    }
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llllllllllllll}",
        r"\hline",
        " & ".join(labels[key] for key in compact_columns) + r" \\",
        r"\hline",
    ]
    for row in rows:
        values = [escape_latex(display_value(row.get(key))) for key in compact_columns]
        lines.append(" & ".join(values) + r" \\")
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}%",
            r"}",
            r"\caption{ONNX Runtime precision benchmark. Energy is reported only when measured from a supplied power log.}",
            r"\label{tab:onnx-precision-benchmark}",
            r"\end{table}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_tables(output_dir: Path, rows: list[dict[str, Any]], formats: list[str]) -> dict[str, str]:
    written: dict[str, str] = {}
    if "csv" in formats:
        path = output_dir / "onnx_precision_benchmark.csv"
        write_csv_table(path, rows)
        written["csv"] = str(path)
    if "markdown" in formats:
        path = output_dir / "onnx_precision_benchmark.md"
        write_markdown_table(path, rows)
        written["markdown"] = str(path)
    if "latex" in formats:
        path = output_dir / "onnx_precision_benchmark.tex"
        write_latex_table(path, rows)
        written["latex"] = str(path)
    return written


def write_summary(path: Path, rows: list[dict[str, Any]], failures: list[dict[str, Any]], hardware: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("ONNX precision benchmark summary")
    lines.append("")
    lines.append(f"Hardware used: {hardware.get('edge_measurement_interpretation')}")
    lines.append(f"Platform: {hardware.get('platform', {}).get('system')} {hardware.get('platform', {}).get('machine')}")
    device_name = hardware.get("device_name_user_provided") or "N/A"
    lines.append(f"Device name: {device_name}")
    providers = sorted({str(row.get("execution_provider_used")) for row in rows if row.get("execution_provider_used")})
    lines.append(f"Providers tested: {', '.join(providers) if providers else 'N/A'}")
    if rows:
        best_latency = min(
            (row for row in rows if row.get("mean_latency_ms") is not None),
            key=lambda row: float(row["mean_latency_ms"]),
            default=None,
        )
        if best_latency:
            lines.append(
                "Best latency configuration: "
                f"{best_latency.get('precision')} / {best_latency.get('execution_provider_used')} "
                f"at {display_value(best_latency.get('mean_latency_ms'))} ms"
            )
        smallest = min(
            (row for row in rows if row.get("onnx_file_size_mb") is not None),
            key=lambda row: float(row["onnx_file_size_mb"]),
            default=None,
        )
        if smallest:
            lines.append(
                "Smallest model configuration: "
                f"{smallest.get('precision')} at {display_value(smallest.get('onnx_file_size_mb'))} MB"
            )
        for precision in ("FP16", "INT8"):
            precision_rows = [row for row in rows if row.get("precision") == precision and row.get("speedup_vs_fp32") is not None]
            if precision_rows:
                improved = any(float(row["speedup_vs_fp32"]) > 1.0 for row in precision_rows)
                lines.append(f"{precision} improved latency versus FP32: {'yes' if improved else 'no'}")
            else:
                lines.append(f"{precision} improved latency versus FP32: N/A")
    energy_rows = [row for row in rows if row.get("energy_per_inference_mj") is not None]
    lines.append(f"Energy measured: {'yes' if energy_rows else 'no; N/A because no covering real power log was available'}")
    if failures:
        lines.append("")
        lines.append("Failures/warnings:")
        for failure in failures:
            lines.append(f"- {failure.get('precision', 'N/A')} {failure.get('provider_requested', '')}: {failure.get('reason')}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def dependency_warnings() -> list[str]:
    notes: list[str] = []
    if psutil is None:
        notes.append("psutil unavailable: install with `pip install psutil` for RSS polling")
    if optional_module("pynvml") is None:
        notes.append("pynvml unavailable: install with `pip install pynvml` for CUDA process memory")
    if optional_module("pandas") is None:
        notes.append("pandas unavailable: native CSV/Markdown/LaTeX writers will be used")
    if optional_module("tabulate") is None:
        notes.append("tabulate unavailable: native Markdown writer will be used")
    return notes


def main() -> None:
    args = parse_args()
    if np is None:
        raise SystemExit("Missing required dependency 'numpy'. Install it with: pip install numpy")
    script_start_epoch_s = time.time()
    script_start_perf_s = time.perf_counter()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    fp32_path = Path(args.fp32_onnx).expanduser()
    if not fp32_path.exists():
        raise SystemExit(f"--fp32-onnx does not exist: {fp32_path}")

    notes = dependency_warnings()
    ort = load_onnxruntime()
    available_providers = ort.get_available_providers()
    validate_onnx_model(fp32_path)
    input_specs = get_input_specs(ort, fp32_path)
    input_shapes, default_shape = parse_input_shapes(args.input_shape)
    input_factory = make_input_factory(args, input_specs, input_shapes, default_shape, notes)
    benchmark_batches = input_factory.make_batches((int(args.warmup) + int(args.runs)) * int(args.batch_size))
    calibration_batches = input_factory.make_batches(int(args.calibration_samples))
    drift_batches = input_factory.make_batches(int(args.drift_samples))
    metric_batches = input_factory.metric_batches(int(args.metric_samples))

    failures: list[dict[str, Any]] = []
    models = build_precision_models(args, fp32_path, output_dir, calibration_batches, failures)
    power_log = None
    if args.power_log:
        try:
            power_log = PowerLog(Path(args.power_log).expanduser(), args.timestamp_column, args.power_column)
        except Exception as exc:
            notes.append(f"power log unavailable: {exc}")

    rows: list[dict[str, Any]] = []
    for requested_provider in args.providers:
        provider = normalize_provider(requested_provider, available_providers)
        for model in models:
            result = benchmark_model_provider(
                args=args,
                ort=ort,
                model=model,
                provider=provider,
                requested_provider=requested_provider,
                available_providers=available_providers,
                output_dir=output_dir,
                input_factory=input_factory,
                benchmark_batches=benchmark_batches,
                drift_batches=drift_batches,
                metric_batches=metric_batches,
                fp32_path=fp32_path,
                power_log=power_log,
                script_start_epoch_s=script_start_epoch_s,
                script_start_perf_s=script_start_perf_s,
            )
            if result.failure:
                failures.append(result.failure)
            elif result.row:
                rows.append(result.row)

    add_relative_metrics(rows, path_size_mb(fp32_path))
    hardware = collect_hardware_metadata(args, ort, notes)
    tables = write_tables(output_dir, rows, list(args.table_formats))

    config = {
        "args": vars(args),
        "fp32_onnx": str(fp32_path),
        "input_specs": [spec.__dict__ for spec in input_specs],
        "input_source": input_factory.source,
        "input_notes": input_factory.notes,
        "available_providers": available_providers,
        "precision_models": [
            {"precision": model.precision, "path": str(model.path), "notes": model.notes, "metadata": model.metadata}
            for model in models
        ],
        "generated_tables": tables,
        "script_start_epoch_s": script_start_epoch_s,
        "script_end_epoch_s": time.time(),
        "notes": notes,
    }
    write_json(output_dir / "benchmark_config.json", config)
    write_json(output_dir / "hardware_metadata.json", hardware)
    write_json(output_dir / "onnx_precision_benchmark_failures.json", failures)
    write_summary(output_dir / "onnx_precision_benchmark_summary.txt", rows, failures, hardware)

    print(f"rows_written: {len(rows)}")
    print(f"failures: {len(failures)}")
    for name, path in tables.items():
        print(f"{name}: {path}")
    print(f"config: {output_dir / 'benchmark_config.json'}")
    print(f"hardware_metadata: {output_dir / 'hardware_metadata.json'}")
    print(f"summary: {output_dir / 'onnx_precision_benchmark_summary.txt'}")
    if not rows:
        raise SystemExit("No benchmark rows ran. Check provider availability and failure report.")


if __name__ == "__main__":
    main()
