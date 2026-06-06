#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Torchrun worker that rank-shards SCENIC one-shot pruning/eval jobs."
    )
    parser.add_argument("--jobs-json", required=True)
    return parser.parse_args()


def visible_device_for_local_rank(local_rank: int) -> str | None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    devices = [part.strip() for part in visible.split(",") if part.strip()]
    if not devices:
        return None
    return devices[local_rank % len(devices)]


def run_command(command: list[str], env: dict[str, str]) -> None:
    print("[one-shot-worker] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)


def worker_env(local_rank: int) -> dict[str, str]:
    env = os.environ.copy()
    device = visible_device_for_local_rank(local_rank)
    if device is not None:
        env["CUDA_VISIBLE_DEVICES"] = device
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def device_arg(requested: str, env: dict[str, str]) -> str:
    if requested != "auto":
        return requested
    return "cuda" if env.get("CUDA_VISIBLE_DEVICES") else "auto"


def run_job(job: dict[str, Any], settings: dict[str, Any], env: dict[str, str]) -> None:
    eval_dir = Path(job["eval_dir"])
    eval_dir.mkdir(parents=True, exist_ok=True)
    pruned_checkpoint = str(job["pruned_checkpoint"])

    prune_command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "prune_scenic_sft_reference_methods.py"),
        "--method",
        str(job["method"]),
        "--checkpoint",
        str(job["checkpoint"]),
        "--output",
        pruned_checkpoint,
        "--sparsity",
        str(job["sparsity"]),
        "--scope",
        str(settings["prune_scope"]),
        "--exclude-classifier",
        "--calibration-json",
        str(job["train_json"]),
        "--calibration-batch-size",
        str(settings["calibration_batch_size"]),
        "--calibration-batches",
        str(settings["calibration_batches"]),
        "--max-length",
        str(settings["max_length"]),
        "--device",
        device_arg(str(settings["prune_device"]), env),
        "--dtype",
        str(settings["prune_dtype"]),
    ]
    if bool(settings.get("overwrite")):
        prune_command.append("--overwrite")
    if bool(settings.get("reinitialize_classifier")):
        prune_command.extend(
            [
                "--reinitialize-classifier-from-responses",
                "--classifier-init-batch-size",
                str(settings["classifier_init_batch_size"]),
                "--classifier-init-max-length",
                str(settings["classifier_init_max_length"]),
            ]
        )
    run_command(prune_command, env)

    for dataset_name, json_path in (
        ("training", str(job["train_json"])),
        ("benchmark", str(settings["benchmark_json"])),
    ):
        run_command(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "eval_scenic_sft_local.py"),
                "--json",
                json_path,
                "--checkpoint",
                pruned_checkpoint,
                "--output",
                str(eval_dir / f"{dataset_name}_predictions.jsonl"),
                "--summary-output",
                str(eval_dir / f"{dataset_name}_summary.json"),
                "--batch-size",
                str(settings["batch_size"]),
                "--max-length",
                str(settings["max_length"]),
                "--dtype",
                str(settings["eval_dtype"]),
            ],
            env,
        )


def main() -> None:
    args = parse_args()
    payload = json.loads(Path(args.jobs_json).read_text(encoding="utf-8"))
    settings = payload["settings"]
    jobs = list(payload["jobs"])

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    env = worker_env(local_rank)

    assigned = [job for index, job in enumerate(jobs) if index % world_size == rank]
    print(
        f"[one-shot-worker] rank={rank}/{world_size} local_rank={local_rank} "
        f"visible={env.get('CUDA_VISIBLE_DEVICES', '<unset>')} assigned={len(assigned)}",
        flush=True,
    )
    for job in assigned:
        print(
            f"[one-shot-worker] rank={rank} job={job['variant']}:{job['run_label']}",
            flush=True,
        )
        run_job(job, settings, env)


if __name__ == "__main__":
    main()
