#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from chatlm_encoder.sparsity_eval import first_text, read_table, sample_identifier  # noqa: E402


README_TEXT = """# Benchmark Difficulty Labeling Guide

Fill the `difficulty` column with exactly one of: `easy`, `medium`, `hard`.

easy:
Direct, single-intent, single-device command with explicit action and target.
Example: "Turn on the bedroom light."

medium:
Paraphrased, indirect, multi-device, or slightly contextual command, but still unambiguous.
Example: "It is too dark in the bedroom."

hard:
Indirect, compositional, conditional, rare-device, multi-step, negated, or potentially ambiguous command.
Example: "If the room gets too warm, lower the AC and turn off the heater."
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a blank easy/medium/hard difficulty template for a SCENIC benchmark."
    )
    parser.add_argument("--benchmark_path", required=True)
    parser.add_argument("--output_dir", default=".")
    parser.add_argument("--output_name", default="benchmark_difficulty_template.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_table(args.benchmark_path)
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / args.output_name
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "input", "target", "difficulty"])
        writer.writeheader()
        for index, row in enumerate(rows):
            writer.writerow(
                {
                    "id": sample_identifier(row, index),
                    "input": first_text(row, ("prompt", "anchor", "instruction", "input", "text", "query")),
                    "target": first_text(row, ("response", "target", "expected_response", "output", "answer")),
                    "difficulty": "",
                }
            )

    readme_path = output_dir / "benchmark_difficulty_template_README.md"
    readme_path.write_text(README_TEXT, encoding="utf-8")
    print(f"wrote {output_path}")
    print(f"wrote {readme_path}")


if __name__ == "__main__":
    main()
