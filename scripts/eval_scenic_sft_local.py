#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from chatlm_encoder.scenic_sft import load_scenic_checkpoint, prompt_from_row, read_json_list  # noqa: E402


# Edit these paths if you want to run the evaluator without command-line flags.
LOCAL_JSON_PATH = "data/scenic/SCENIC_full_training_dataset.json"
CHECKPOINT_DIR = "runs/scenic-sft/latest"
OUTPUT_PATH = "eval_results/scenic_sft/local_eval_predictions.jsonl"
MAX_LENGTH = 128
BATCH_SIZE = 128


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_eval_rows(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    for index, row in enumerate(read_json_list(path)):
        prompt = prompt_from_row(row)
        response = "" if row.get("response") is None else str(row.get("response")).strip()
        if not prompt:
            raise ValueError(f"{path}:{index} does not contain a prompt or anchor field.")
        rows.append({"index": index, "prompt": prompt, "expected_response": response, "raw": row})
    return rows


def batched(items: list[dict[str, Any]], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a SCENIC encoder SFT checkpoint on a local JSON file.")
    parser.add_argument("--json", default=LOCAL_JSON_PATH, help="Local JSON list with prompt/response or anchor/response rows.")
    parser.add_argument("--checkpoint", default=CHECKPOINT_DIR, help="SCENIC SFT checkpoint directory.")
    parser.add_argument("--output", default=OUTPUT_PATH, help="Prediction JSONL output path.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    args = parser.parse_args()

    device = select_device()
    model, tokenizer, label2response = load_scenic_checkpoint(args.checkpoint, device=device)
    rows = load_eval_rows(args.json)
    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    scored = 0
    correct = 0
    top5_correct = 0

    batch_count = math.ceil(len(rows) / int(args.batch_size)) if rows else 0
    with output_path.open("w", encoding="utf-8") as handle, torch.no_grad():
        for batch in tqdm(batched(rows, int(args.batch_size)), total=batch_count, desc="eval scenic", unit="batch"):
            prompts = [item["prompt"] for item in batch]
            tokens = tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=int(args.max_length),
                return_tensors="pt",
            )
            tokens = {key: value.to(device) for key, value in tokens.items()}
            logits = model(tokens)["logits"]
            probabilities_all = torch.softmax(logits, dim=-1)
            top_probs, top_indices = torch.topk(probabilities_all, k=min(5, logits.shape[-1]), dim=-1)
            probabilities = top_probs.detach().cpu().tolist()
            top_indices_list = top_indices.detach().cpu().tolist()

            for item, top_ids, top_probs in zip(batch, top_indices_list, probabilities):
                predicted = label2response[int(top_ids[0])]
                expected = item["expected_response"]
                result = {
                    "index": item["index"],
                    "prompt": item["prompt"],
                    "expected_response": expected,
                    "predicted_response": predicted,
                    "correct": bool(expected and predicted == expected),
                    "top5": [
                        {"response": label2response[int(label_id)], "score": float(score)}
                        for label_id, score in zip(top_ids, top_probs)
                    ],
                }
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                total += 1
                if expected:
                    scored += 1
                    correct += int(predicted == expected)
                    top5_correct += int(expected in {label2response[int(label_id)] for label_id in top_ids})

    print(f"checkpoint: {args.checkpoint}")
    print(f"json: {args.json}")
    print(f"output: {output_path}")
    print(f"rows: {total:,}")
    if scored:
        print(f"exact_accuracy: {correct / scored:.6f} ({correct:,}/{scored:,})")
        print(f"top5_accuracy: {top5_correct / scored:.6f} ({top5_correct:,}/{scored:,})")
    else:
        print("No expected response fields found; wrote predictions only.")


if __name__ == "__main__":
    main()
