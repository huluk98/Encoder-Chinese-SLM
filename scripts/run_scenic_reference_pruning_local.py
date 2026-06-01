#!/usr/bin/env python
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


# =============================================================================
# EDIT THESE PATHS AND SETTINGS
# =============================================================================

# Which pruning methods to run. Use any subset of:
# "magnitude", "nvidia", "wanda", "gradient".
METHODS = ("magnitude", "nvidia", "wanda", "gradient")

# SFT checkpoints created by:
#   ./scripts/launch_scenic_sft_separate_8gpu.sh
TRAINING_CHECKPOINT = "runs/scenic-sft-training-dataset/latest"
CONTRASTIVE_CHECKPOINT = "runs/scenic-sft-contrastive-dataset/latest"

# Evaluation JSON files.
BENCHMARK_JSON = "data/scenic/iot_instruction_benchmark_200.json"
TRAINING_JSON = "data/scenic/SCENIC_full_training_dataset.json"
CONTRASTIVE_JSON = "data/scenic/SCENIC_full_anchor_positive_negative.json"

# WANDA calibration data. For fair comparison, keep these matched to each model's
# own SFT data unless you intentionally want a different calibration set.
TRAINING_CALIBRATION_JSON = TRAINING_JSON
CONTRASTIVE_CALIBRATION_JSON = CONTRASTIVE_JSON

# Output locations.
RUN_ROOT = "runs/scenic-pruned50-reference-methods"
EVAL_OUTPUT_DIR = "eval_results/scenic_sft/pruned50_reference_methods"

# Pruning settings. Defaults protect the response classifier, which is the
# full-response label head for this encoder-only SFT setup.
SPARSITY = 0.5
PRUNE_SCOPE = "encoder-linear"  # "all-linear" or "encoder-linear"
INCLUDE_CLASSIFIER = False

# Runtime settings.
PRUNE_DEVICE = "auto"  # "auto", "cuda", "cuda:0", "mps", or "cpu"
PRUNE_DTYPE = "fp32"  # "fp32", "bf16", or "fp16"
CALIBRATION_BATCH_SIZE = 4
CALIBRATION_BATCHES = 64
EVAL_DTYPE = "auto"  # "auto", "fp32", or "bf16"
EVAL_BATCH_SIZE = 128
MAX_LENGTH = 128
OVERWRITE = True

# =============================================================================
# END EDIT BLOCK
# =============================================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def method_label(method: str) -> tuple[str, str]:
    normalized = method.strip().lower()
    if normalized == "magnitude":
        return "magnitude", "magnitude"
    if normalized in {"nvidia", "nvidia-2:4", "nvidia_2_4"}:
        return "nvidia", "nvidia_2_4"
    if normalized == "wanda":
        return "wanda", "wanda"
    if normalized in {"gradient", "taylor"}:
        return "gradient", "gradient"
    raise ValueError(f"Unknown method {method!r}. Use magnitude, nvidia, wanda, or gradient.")


def prune_args(
    *,
    method: str,
    checkpoint: str,
    output: str,
    calibration_json: str,
) -> list[str]:
    command = [
        sys.executable,
        "scripts/prune_scenic_sft_reference_methods.py",
        "--method",
        method,
        "--checkpoint",
        checkpoint,
        "--output",
        output,
        "--sparsity",
        str(SPARSITY),
        "--scope",
        PRUNE_SCOPE,
        "--calibration-json",
        calibration_json,
        "--calibration-batch-size",
        str(CALIBRATION_BATCH_SIZE),
        "--calibration-batches",
        str(CALIBRATION_BATCHES),
        "--max-length",
        str(MAX_LENGTH),
        "--device",
        PRUNE_DEVICE,
        "--dtype",
        PRUNE_DTYPE,
    ]
    if OVERWRITE:
        command.append("--overwrite")
    if not INCLUDE_CLASSIFIER:
        command.append("--exclude-classifier")
    return command


def eval_args(*, training_output: str, contrastive_output: str, eval_output: str) -> list[str]:
    return [
        sys.executable,
        "scripts/eval_scenic_sft_comparison.py",
        "--benchmark-json",
        BENCHMARK_JSON,
        "--training-json",
        TRAINING_JSON,
        "--contrastive-json",
        CONTRASTIVE_JSON,
        "--training-checkpoint",
        training_output,
        "--contrastive-checkpoint",
        contrastive_output,
        "--output-dir",
        eval_output,
        "--batch-size",
        str(EVAL_BATCH_SIZE),
        "--max-length",
        str(MAX_LENGTH),
        "--dtype",
        EVAL_DTYPE,
    ]


def aggregate(labels: list[str]) -> None:
    run(
        [
            sys.executable,
            "scripts/aggregate_scenic_reference_pruning.py",
            "--eval-root",
            EVAL_OUTPUT_DIR,
            "--run-root",
            RUN_ROOT,
            "--methods",
            ",".join(labels),
        ]
    )


def main() -> None:
    completed_labels: list[str] = []
    for raw_method in METHODS:
        method, label = method_label(raw_method)
        training_output = f"{RUN_ROOT}/training_dataset_{label}"
        contrastive_output = f"{RUN_ROOT}/contrastive_anchor_{label}"
        eval_output = f"{EVAL_OUTPUT_DIR}/{label}"

        print(f"\n[reference-prune] method={method} training model -> {training_output}", flush=True)
        run(
            prune_args(
                method=method,
                checkpoint=TRAINING_CHECKPOINT,
                output=training_output,
                calibration_json=TRAINING_CALIBRATION_JSON,
            )
        )

        print(f"\n[reference-prune] method={method} contrastive model -> {contrastive_output}", flush=True)
        run(
            prune_args(
                method=method,
                checkpoint=CONTRASTIVE_CHECKPOINT,
                output=contrastive_output,
                calibration_json=CONTRASTIVE_CALIBRATION_JSON,
            )
        )

        print(f"\n[reference-prune] evaluating method={method}", flush=True)
        run(eval_args(training_output=training_output, contrastive_output=contrastive_output, eval_output=eval_output))
        completed_labels.append(label)

    aggregate(completed_labels)
    print("\n[reference-prune] done", flush=True)


if __name__ == "__main__":
    main()
