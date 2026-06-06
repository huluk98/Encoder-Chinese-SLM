#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot SCENIC linear sparsity experiment summaries.")
    parser.add_argument("--experiment_name", required=True)
    parser.add_argument("--results_dir", default=None)
    parser.add_argument("--summary_csv", default=None)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def as_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    return float(value)


def mode_label(mode: str) -> str:
    return {"dense": "Dense", "oneshot": "One-shot", "progressive": "Progressive"}.get(mode, mode)


def line_plot(rows: list[dict[str, Any]], metric: str, ylabel: str, output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise SystemExit("matplotlib is required for plotting; install project requirements first.") from exc

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for mode in ("dense", "oneshot", "progressive"):
        mode_rows = sorted(
            [row for row in rows if row.get("pruning_mode") == mode],
            key=lambda row: float(row["target_sparsity"]),
        )
        if not mode_rows:
            continue
        x_values = [float(row["target_sparsity"]) * 100.0 for row in mode_rows]
        y_values = [as_float(row.get(metric)) for row in mode_rows]
        ax.plot(x_values, y_values, marker="o", linewidth=2, label=mode_label(mode))
    ax.set_xlabel("Target sparsity (%)")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def difficulty_bar(rows: list[dict[str, Any]], metric_prefix: str, ylabel: str, output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise SystemExit("matplotlib is required for plotting; install project requirements first.") from exc

    labels = []
    easy = []
    medium = []
    hard = []
    for row in sorted(rows, key=lambda item: (item.get("pruning_mode", ""), float(item["target_sparsity"]))):
        labels.append(f"{mode_label(row['pruning_mode'])}\n{float(row['target_sparsity']) * 100:.0f}%")
        easy.append(as_float(row.get(f"{metric_prefix}_easy")) or 0.0)
        medium.append(as_float(row.get(f"{metric_prefix}_medium")) or 0.0)
        hard.append(as_float(row.get(f"{metric_prefix}_hard")) or 0.0)

    x_values = list(range(len(labels)))
    width = 0.26
    fig, ax = plt.subplots(figsize=(max(8.0, len(labels) * 1.05), 4.8))
    ax.bar([x - width for x in x_values], easy, width=width, label="Easy", color="#2563eb")
    ax.bar(x_values, medium, width=width, label="Medium", color="#059669")
    ax.bar([x + width for x in x_values], hard, width=width, label="Hard", color="#dc2626")
    ax.set_xticks(x_values)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, ncols=3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.summary_csv:
        summary_path = Path(args.summary_csv).expanduser()
    else:
        results_dir = Path(args.results_dir or Path("results") / args.experiment_name).expanduser()
        summary_path = results_dir / "summary_metrics.csv"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)

    rows = read_rows(summary_path)
    figures_dir = summary_path.parent / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    line_plot(rows, "em1_overall", "EM@1", figures_dir / "em1_vs_sparsity.png")
    line_plot(rows, "em5_overall", "EM@5", figures_dir / "em5_vs_sparsity.png")
    difficulty_bar(rows, "em1", "EM@1", figures_dir / "difficulty_em1_breakdown.png")
    difficulty_bar(rows, "em5", "EM@5", figures_dir / "difficulty_em5_breakdown.png")
    print(f"wrote figures to {figures_dir}")


if __name__ == "__main__":
    main()
