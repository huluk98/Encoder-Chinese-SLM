from __future__ import annotations

import csv
import json
import re
import random
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


PROMPT_FIELDS = ("prompt", "anchor", "instruction", "input", "text", "query")
TARGET_FIELDS = ("response", "target", "expected_response", "output", "answer")
DIFFICULTY_FIELDS = ("difficulty", "complexity", "level")
ID_FIELDS = ("id", "sample_id")
VALID_DIFFICULTIES = ("easy", "medium", "hard")


@dataclass(frozen=True)
class BenchmarkSample:
    sample_id: str
    input: str
    target: str
    difficulty: str
    raw: dict[str, Any]


def normalize_text(value: Any, mode: str = "scenic") -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKC", text).strip()
    if mode in {"none", "raw"}:
        return text
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?，。；：！？、])", r"\1", text)
    text = re.sub(r"([,.;:!?，。；：！？、])\s+", r"\1", text)
    return text.strip()


def sample_identifier(row: dict[str, Any], fallback_index: int) -> str:
    for field in ID_FIELDS:
        value = row.get(field)
        if value is not None and str(value).strip():
            return str(value).strip()
    return str(fallback_index)


def first_text(row: dict[str, Any], fields: Iterable[str]) -> str:
    for field in fields:
        value = row.get(field)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def normalize_difficulty(value: Any) -> str:
    difficulty = "" if value is None else str(value).strip().lower()
    if difficulty not in VALID_DIFFICULTIES:
        raise ValueError(
            f"Invalid difficulty label {value!r}; expected one of {', '.join(VALID_DIFFICULTIES)}."
        )
    return difficulty


def read_table(path: str | Path) -> list[dict[str, Any]]:
    data_path = Path(path).expanduser()
    suffix = data_path.suffix.lower()
    if suffix == ".json":
        value = json.loads(data_path.read_text(encoding="utf-8"))
        if not isinstance(value, list):
            raise ValueError(f"{data_path} must contain a JSON list.")
        return [dict(item) for item in value]
    if suffix == ".jsonl":
        rows = []
        with data_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{data_path}:{line_number} must contain a JSON object.")
                rows.append(dict(value))
        return rows
    if suffix == ".csv":
        with data_path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    raise ValueError(f"Unsupported table format for {data_path}. Use JSON, JSONL, or CSV.")


def _difficulty_from_row(row: dict[str, Any]) -> str | None:
    for field in DIFFICULTY_FIELDS:
        value = row.get(field)
        if value is not None and str(value).strip():
            return normalize_difficulty(value)
    return None


def _external_difficulty_maps(path: str | Path) -> tuple[dict[str, str], dict[str, str]]:
    by_id: dict[str, str] = {}
    by_input: dict[str, str] = {}
    for index, row in enumerate(read_table(path)):
        difficulty = normalize_difficulty(row.get("difficulty"))
        identifier = first_text(row, ID_FIELDS)
        if identifier:
            by_id[identifier] = difficulty
        command = first_text(row, ("input", "prompt", "anchor", "instruction", "text", "query"))
        if command:
            by_input[command] = difficulty
        if not identifier and not command:
            raise ValueError(
                f"{path}:{index} must contain id, sample_id, or input/prompt for difficulty joining."
            )
    return by_id, by_input


def load_benchmark_samples(
    benchmark_path: str | Path,
    difficulty_path: str | Path | None = None,
) -> list[BenchmarkSample]:
    rows = read_table(benchmark_path)
    external_by_id: dict[str, str] = {}
    external_by_input: dict[str, str] = {}
    if difficulty_path:
        external_by_id, external_by_input = _external_difficulty_maps(difficulty_path)

    samples: list[BenchmarkSample] = []
    has_inline_difficulty_column = any(any(field in row for field in DIFFICULTY_FIELDS) for row in rows)
    if not has_inline_difficulty_column and not difficulty_path:
        raise ValueError(
            "Benchmark has no difficulty/complexity/level column. Pass --benchmark_difficulty_path."
        )

    for index, row in enumerate(rows):
        prompt = first_text(row, PROMPT_FIELDS)
        target = first_text(row, TARGET_FIELDS)
        if not prompt:
            raise ValueError(f"{benchmark_path}:{index} does not contain an input command field.")
        if not target:
            raise ValueError(f"{benchmark_path}:{index} does not contain a target/response field.")

        identifier = sample_identifier(row, index)
        difficulty = _difficulty_from_row(row)
        if difficulty is None:
            if identifier in external_by_id:
                difficulty = external_by_id[identifier]
            elif prompt in external_by_input:
                difficulty = external_by_input[prompt]
            else:
                raise ValueError(
                    f"Could not join difficulty for sample id={identifier!r}, input={prompt!r}. "
                    "Provide id/sample_id or exact input matches in --benchmark_difficulty_path."
                )

        samples.append(
            BenchmarkSample(
                sample_id=identifier,
                input=prompt,
                target=target,
                difficulty=difficulty,
                raw=dict(row),
            )
        )
    return samples


def prediction_record(
    sample: BenchmarkSample,
    candidates: list[str],
    normalization_mode: str = "scenic",
) -> dict[str, Any]:
    top5 = candidates[:5]
    normalized_target = normalize_text(sample.target, normalization_mode)
    normalized_candidates = [normalize_text(candidate, normalization_mode) for candidate in top5]
    top1 = top5[0] if top5 else ""
    em1 = bool(normalized_candidates and normalized_candidates[0] == normalized_target)
    em5 = bool(normalized_target and normalized_target in normalized_candidates)
    return {
        "sample_id": sample.sample_id,
        "input": sample.input,
        "target": sample.target,
        "difficulty": sample.difficulty,
        "top1_prediction": top1,
        "top5_predictions": top5,
        "em1": em1,
        "em5": em5,
    }


def bootstrap_ci(
    values: list[bool | int | float],
    resamples: int = 1000,
    seed: int = 42,
    min_n: int = 20,
) -> dict[str, Any]:
    if len(values) < min_n:
        return {"low": None, "high": None, "status": "insufficient_n"}
    if not values:
        return {"low": None, "high": None, "status": "empty"}
    array = [float(value) for value in values]
    rng = random.Random(int(seed))
    means = []
    for _index in range(int(resamples)):
        total = sum(rng.choice(array) for _sample in range(len(array)))
        means.append(total / len(array))
    low = percentile(means, 2.5)
    high = percentile(means, 97.5)
    return {"low": float(low), "high": float(high), "status": "ok"}


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    sorted_values = sorted(values)
    position = (len(sorted_values) - 1) * float(q) / 100.0
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def _score_group(
    records: list[dict[str, Any]],
    resamples: int,
    seed: int,
    ci_min_n: int,
) -> dict[str, Any]:
    em1_values = [bool(record["em1"]) for record in records]
    em5_values = [bool(record["em5"]) for record in records]
    em1_ci = bootstrap_ci(em1_values, resamples=resamples, seed=seed, min_n=ci_min_n)
    em5_ci = bootstrap_ci(em5_values, resamples=resamples, seed=seed + 17, min_n=ci_min_n)
    return {
        "count": len(records),
        "em1": sum(float(value) for value in em1_values) / len(em1_values) if em1_values else None,
        "em5": sum(float(value) for value in em5_values) / len(em5_values) if em5_values else None,
        "em1_ci_low": em1_ci["low"],
        "em1_ci_high": em1_ci["high"],
        "em1_ci_status": em1_ci["status"],
        "em5_ci_low": em5_ci["low"],
        "em5_ci_high": em5_ci["high"],
        "em5_ci_status": em5_ci["status"],
    }


def compute_metric_breakdown(
    records: list[dict[str, Any]],
    bootstrap_resamples: int = 1000,
    seed: int = 42,
    ci_min_n: int = 20,
) -> dict[str, Any]:
    by_difficulty = {
        difficulty: [record for record in records if record.get("difficulty") == difficulty]
        for difficulty in VALID_DIFFICULTIES
    }
    return {
        "overall": _score_group(records, bootstrap_resamples, seed, min(1, ci_min_n)),
        "difficulty": {
            difficulty: _score_group(group, bootstrap_resamples, seed + offset + 1, ci_min_n)
            for offset, (difficulty, group) in enumerate(by_difficulty.items())
        },
    }


def retention(numerator: Any, denominator: Any) -> float | None:
    if numerator is None or denominator in {None, 0}:
        return None
    return float(numerator) / float(denominator)


def write_prediction_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            serialized = dict(row)
            if isinstance(serialized.get("top5_predictions"), list):
                serialized["top5_predictions"] = json.dumps(
                    serialized["top5_predictions"], ensure_ascii=False
                )
            writer.writerow(serialized)


def write_rows_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})
