from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from chatlm_encoder.linear_sparsity import (  # noqa: E402
    LinearSparsityConfig,
    apply_magnitude_pruning,
    apply_masks,
    collect_prunable_linear_modules,
    register_mask_gradient_hooks,
    remove_hooks,
)
from chatlm_encoder.sparsity_eval import (  # noqa: E402
    BenchmarkSample,
    compute_metric_breakdown,
    load_benchmark_samples,
    prediction_record,
)
from run_sparsity_experiments import SUMMARY_FIELDNAMES  # noqa: E402


class ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(8, 4)
        self.encoder = nn.Sequential(nn.Linear(4, 10), nn.LayerNorm(10), nn.Linear(10, 10))
        self.classifier = nn.Linear(10, 2)
        self.lm_head = nn.Linear(10, 8)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embedding(input_ids).mean(dim=1)
        return self.classifier(self.encoder(hidden))


def fill_nonzero(model: nn.Module) -> None:
    with torch.no_grad():
        for parameter in model.parameters():
            values = torch.arange(1, parameter.numel() + 1, dtype=parameter.dtype).reshape_as(parameter)
            parameter.copy_(values)


class LinearSparsityTests(unittest.TestCase):
    def test_linear_collection_excludes_heads_and_non_linear_modules(self) -> None:
        model = ToyModel()
        modules = collect_prunable_linear_modules(model)
        names = [name for name, _module in modules]
        self.assertEqual(names, ["encoder.0", "encoder.2"])

    def test_magnitude_pruning_reaches_30_percent_targeted_sparsity(self) -> None:
        model = ToyModel()
        fill_nonzero(model)
        _masks, stats = apply_magnitude_pruning(model, 0.30, LinearSparsityConfig())
        self.assertAlmostEqual(stats["targeted_linear_sparsity_actual"], 0.30, places=6)

    def test_magnitude_pruning_reaches_50_percent_targeted_sparsity(self) -> None:
        model = ToyModel()
        fill_nonzero(model)
        _masks, stats = apply_magnitude_pruning(model, 0.50, LinearSparsityConfig())
        self.assertAlmostEqual(stats["targeted_linear_sparsity_actual"], 0.50, places=6)

    def test_mask_enforcement_keeps_pruned_weights_zero_after_optimizer_step(self) -> None:
        model = nn.Sequential(nn.Linear(4, 2, bias=False))
        fill_nonzero(model)
        masks, _stats = apply_magnitude_pruning(model, 0.50, LinearSparsityConfig())
        handles = register_mask_gradient_hooks(model, masks)
        try:
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            output = model(torch.ones(1, 4)).sum()
            output.backward()
            optimizer.step()
            apply_masks(model, masks)
            weight = model[0].weight.detach()
            mask = masks["0"].to(dtype=torch.bool)
            self.assertTrue(torch.all(weight[~mask] == 0))
        finally:
            remove_hooks(handles)


class EvaluationTests(unittest.TestCase):
    def test_em1_and_em5_on_synthetic_candidates(self) -> None:
        sample = BenchmarkSample(
            sample_id="a",
            input="开灯",
            target="好的，已开灯。",
            difficulty="easy",
            raw={},
        )
        top1 = prediction_record(sample, ["好的,已开灯。"])
        self.assertTrue(top1["em1"])
        self.assertTrue(top1["em5"])

        top5 = prediction_record(sample, ["错", "也错", "好的，已开灯。"])
        self.assertFalse(top5["em1"])
        self.assertTrue(top5["em5"])

    def test_difficulty_join_by_id_and_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            benchmark_by_id = root / "benchmark_by_id.json"
            difficulty_by_id = root / "difficulty_by_id.csv"
            benchmark_by_id.write_text(
                json.dumps([{"id": "s1", "prompt": "开灯", "response": "好"}], ensure_ascii=False),
                encoding="utf-8",
            )
            with difficulty_by_id.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["id", "difficulty"])
                writer.writeheader()
                writer.writerow({"id": "s1", "difficulty": "medium"})
            samples = load_benchmark_samples(benchmark_by_id, difficulty_by_id)
            self.assertEqual(samples[0].difficulty, "medium")

            benchmark_by_input = root / "benchmark_by_input.json"
            difficulty_by_input = root / "difficulty_by_input.csv"
            benchmark_by_input.write_text(
                json.dumps([{"prompt": "关灯", "response": "好"}], ensure_ascii=False),
                encoding="utf-8",
            )
            with difficulty_by_input.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["input", "difficulty"])
                writer.writeheader()
                writer.writerow({"input": "关灯", "difficulty": "hard"})
            samples = load_benchmark_samples(benchmark_by_input, difficulty_by_input)
            self.assertEqual(samples[0].difficulty, "hard")

    def test_metric_breakdown_counts_difficulty_groups(self) -> None:
        records = [
            {"difficulty": "easy", "em1": True, "em5": True},
            {"difficulty": "medium", "em1": False, "em5": True},
            {"difficulty": "hard", "em1": False, "em5": False},
        ]
        metrics = compute_metric_breakdown(records, bootstrap_resamples=10, seed=1)
        self.assertEqual(metrics["difficulty"]["easy"]["count"], 1)
        self.assertEqual(metrics["difficulty"]["medium"]["count"], 1)
        self.assertEqual(metrics["difficulty"]["hard"]["count"], 1)

    def test_summary_fieldnames_include_difficulty_counts(self) -> None:
        self.assertIn("count_easy", SUMMARY_FIELDNAMES)
        self.assertIn("count_medium", SUMMARY_FIELDNAMES)
        self.assertIn("count_hard", SUMMARY_FIELDNAMES)


if __name__ == "__main__":
    unittest.main()
