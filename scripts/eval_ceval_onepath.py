#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

MISSING_DEPENDENCY_ERROR: ModuleNotFoundError | None = None
try:
    import torch
    from datasets import DownloadConfig, load_dataset
    from tqdm.auto import tqdm
    from transformers import AutoModelForMaskedLM, AutoTokenizer
except ModuleNotFoundError as exc:
    MISSING_DEPENDENCY_ERROR = exc
    torch = None  # type: ignore[assignment]
    DownloadConfig = None  # type: ignore[assignment]
    load_dataset = None  # type: ignore[assignment]
    tqdm = None  # type: ignore[assignment]
    AutoModelForMaskedLM = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]


CHOICES = ("A", "B", "C", "D")
CATEGORY_ORDER = ("Humanities", "Other", "STEM", "Social Science")
SUBJECT_CATEGORIES = {
    "computer_network": "STEM",
    "operating_system": "STEM",
    "computer_architecture": "STEM",
    "college_programming": "STEM",
    "college_physics": "STEM",
    "college_chemistry": "STEM",
    "advanced_mathematics": "STEM",
    "probability_and_statistics": "STEM",
    "discrete_mathematics": "STEM",
    "electrical_engineer": "STEM",
    "metrology_engineer": "STEM",
    "high_school_mathematics": "STEM",
    "high_school_physics": "STEM",
    "high_school_chemistry": "STEM",
    "high_school_biology": "STEM",
    "middle_school_mathematics": "STEM",
    "middle_school_biology": "STEM",
    "middle_school_physics": "STEM",
    "middle_school_chemistry": "STEM",
    "veterinary_medicine": "STEM",
    "college_economics": "Social Science",
    "business_administration": "Social Science",
    "marxism": "Social Science",
    "mao_zedong_thought": "Social Science",
    "education_science": "Social Science",
    "teacher_qualification": "Social Science",
    "high_school_politics": "Social Science",
    "high_school_geography": "Social Science",
    "middle_school_politics": "Social Science",
    "middle_school_geography": "Social Science",
    "modern_chinese_history": "Humanities",
    "ideological_and_moral_cultivation": "Humanities",
    "logic": "Humanities",
    "law": "Humanities",
    "chinese_language_and_literature": "Humanities",
    "art_studies": "Humanities",
    "professional_tour_guide": "Humanities",
    "legal_professional": "Humanities",
    "high_school_chinese": "Humanities",
    "high_school_history": "Humanities",
    "middle_school_history": "Humanities",
    "civil_servant": "Other",
    "sports_science": "Other",
    "plant_protection": "Other",
    "basic_medicine": "Other",
    "clinical_medicine": "Other",
    "urban_and_rural_planner": "Other",
    "accountant": "Other",
    "fire_engineer": "Other",
    "environmental_impact_assessment_engineer": "Other",
    "tax_accountant": "Other",
    "physician": "Other",
}
ALL_SUBJECTS = tuple(SUBJECT_CATEGORIES)


def log(message: str) -> None:
    print(message, flush=True)


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def dtype_for(name: str, device: torch.device) -> torch.dtype | str:
    if name == "auto":
        return "auto"
    if name == "bf16":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    if name == "fp16":
        return torch.float16 if device.type == "cuda" else torch.float32
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unknown dtype: {name}")


def safe_name(path: str) -> str:
    checkpoint = Path(path).expanduser()
    if checkpoint.name == "latest" and checkpoint.parent.name:
        name = checkpoint.parent.name
    else:
        name = checkpoint.name or "checkpoint"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "checkpoint"


def normalize_subject(subject: str) -> str:
    return subject.replace("_", " ")


def clean_answer(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().upper()
    match = re.search(r"[ABCD]", text)
    return match.group(0) if match else None


def format_question(row: dict[str, Any]) -> str:
    return (
        f"问题：{str(row['question']).strip()}\n"
        f"A. {str(row['A']).strip()}\n"
        f"B. {str(row['B']).strip()}\n"
        f"C. {str(row['C']).strip()}\n"
        f"D. {str(row['D']).strip()}\n"
        "答案："
    )


def format_example(row: dict[str, Any], include_answer: bool) -> str:
    prompt = format_question(row)
    if include_answer:
        answer = clean_answer(row.get("answer"))
        if answer:
            return f"{prompt}{answer}\n"
    return prompt


def build_prompt(row: dict[str, Any], subject: str, fewshot_rows: list[dict[str, Any]]) -> str:
    header = (
        f"以下是中国关于{normalize_subject(subject)}考试的单项选择题。"
        "请从 A、B、C、D 中选出唯一正确答案。\n\n"
    )
    shots = "".join(f"{format_example(example, include_answer=True)}\n" for example in fewshot_rows)
    return f"{header}{shots}{format_example(row, include_answer=False)}"


def trim_prefix(prefix_ids: list[int], candidate_len: int, max_length: int | None) -> list[int]:
    if not max_length or max_length <= 0:
        return prefix_ids
    max_prefix_len = max(1, max_length - candidate_len)
    if len(prefix_ids) <= max_prefix_len:
        return prefix_ids
    return prefix_ids[-max_prefix_len:]


def encode_candidate_rows(
    tokenizer: AutoTokenizer,
    prompts: list[str],
    max_length: int | None,
) -> list[dict[str, Any]]:
    if tokenizer.mask_token_id is None:
        raise ValueError("Tokenizer must define mask_token_id for C-Eval MLM scoring.")

    encoded_rows: list[dict[str, Any]] = []
    for question_idx, prompt in enumerate(prompts):
        prefix_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        for choice in CHOICES:
            candidate_ids = tokenizer(choice, add_special_tokens=False)["input_ids"]
            if not candidate_ids:
                candidate_ids = tokenizer(f" {choice}", add_special_tokens=False)["input_ids"]
            current_prefix_ids = trim_prefix(prefix_ids, len(candidate_ids), max_length)
            ids = current_prefix_ids + candidate_ids
            candidate_positions = list(range(len(current_prefix_ids), len(ids)))
            masked_ids = ids[:]
            for position in candidate_positions:
                masked_ids[position] = int(tokenizer.mask_token_id)
            encoded_rows.append(
                {
                    "question_idx": question_idx,
                    "choice": choice,
                    "input_ids": masked_ids,
                    "positions": candidate_positions,
                    "target_ids": candidate_ids,
                }
            )
    return encoded_rows


def score_prompt_batch(
    model: AutoModelForMaskedLM,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    device: torch.device,
    max_length: int | None,
    candidate_batch_size: int,
    normalize_by_length: bool,
) -> list[dict[str, float]]:
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    candidate_rows = encode_candidate_rows(tokenizer=tokenizer, prompts=prompts, max_length=max_length)
    scores_by_question: list[dict[str, float]] = [{} for _ in prompts]
    with torch.inference_mode():
        for start in range(0, len(candidate_rows), candidate_batch_size):
            batch = candidate_rows[start : start + candidate_batch_size]
            batch_max_len = max(len(item["input_ids"]) for item in batch)
            input_ids = torch.full((len(batch), batch_max_len), int(pad_id), dtype=torch.long)
            attention_mask = torch.zeros((len(batch), batch_max_len), dtype=torch.long)

            for row_idx, item in enumerate(batch):
                row_ids = item["input_ids"]
                input_ids[row_idx, : len(row_ids)] = torch.tensor(row_ids, dtype=torch.long)
                attention_mask[row_idx, : len(row_ids)] = 1

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            log_probs = torch.log_softmax(logits, dim=-1)

            for row_idx, item in enumerate(batch):
                score = 0.0
                for position, token_id in zip(item["positions"], item["target_ids"], strict=True):
                    score += float(log_probs[row_idx, int(position), int(token_id)].detach().cpu())
                if normalize_by_length and item["target_ids"]:
                    score /= len(item["target_ids"])
                scores_by_question[int(item["question_idx"])][str(item["choice"])] = score
    return scores_by_question


def batched(items: list[Any], batch_size: int) -> list[list[Any]]:
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def load_subject_rows(
    dataset_name: str,
    subject: str,
    split: str,
    dataset_local_files_only: bool,
) -> list[dict[str, Any]]:
    download_config = DownloadConfig(local_files_only=True) if dataset_local_files_only else None
    dataset = load_dataset(dataset_name, name=subject, split=split, download_config=download_config)
    return [dict(row) for row in dataset]


def evaluate_subject(
    model: AutoModelForMaskedLM,
    tokenizer: AutoTokenizer,
    dataset_name: str,
    subject: str,
    split: str,
    n_shot: int,
    limit: int | None,
    device: torch.device,
    max_length: int | None,
    batch_size: int,
    normalize_by_length: bool,
    dataset_local_files_only: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    eval_rows = load_subject_rows(
        dataset_name=dataset_name,
        subject=subject,
        split=split,
        dataset_local_files_only=dataset_local_files_only,
    )
    if limit is not None:
        eval_rows = eval_rows[: int(limit)]

    fewshot_rows: list[dict[str, Any]] = []
    if n_shot > 0:
        try:
            fewshot_rows = load_subject_rows(
                dataset_name=dataset_name,
                subject=subject,
                split="dev",
                dataset_local_files_only=dataset_local_files_only,
            )[: int(n_shot)]
        except Exception as exc:
            log(f"[warn] could not load dev shots for {subject}: {exc}")

    records: list[dict[str, Any]] = []
    correct = 0
    answered = 0
    question_batches = batched(eval_rows, max(1, int(batch_size)))
    for row_batch in tqdm(question_batches, desc=subject, leave=False, unit="batch"):
        prompts = [build_prompt(row, subject=subject, fewshot_rows=fewshot_rows) for row in row_batch]
        batch_scores = score_prompt_batch(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            device=device,
            max_length=max_length,
            candidate_batch_size=max(4, int(batch_size) * len(CHOICES)),
            normalize_by_length=normalize_by_length,
        )

        for row, scores in zip(row_batch, batch_scores, strict=True):
            answer = clean_answer(row.get("answer"))
            prediction = max(scores, key=scores.get)
            is_correct: bool | None = None
            if answer is not None:
                is_correct = prediction == answer
                correct += int(is_correct)
                answered += 1
            records.append(
                {
                    "subject": subject,
                    "id": row.get("id"),
                    "prediction": prediction,
                    "answer": answer or "",
                    "correct": "" if is_correct is None else is_correct,
                    **{f"score_{choice}": scores[choice] for choice in CHOICES},
                }
            )

    accuracy = correct / answered if answered else math.nan
    summary = {
        "subject": subject,
        "category": SUBJECT_CATEGORIES[subject],
        "split": split,
        "n_shot": n_shot,
        "prediction_count": len(records),
        "total": answered,
        "correct": correct,
        "accuracy": accuracy,
    }
    return summary, records


def build_category_summaries(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    categories = {
        category: {
            "category": category,
            "correct": 0,
            "question_count": 0,
            "prediction_count": 0,
            "subject_count": 0,
            "accuracy": math.nan,
        }
        for category in CATEGORY_ORDER
    }
    for summary in summaries:
        category = str(summary["category"])
        categories[category]["correct"] += int(summary["correct"])
        categories[category]["question_count"] += int(summary["total"])
        categories[category]["prediction_count"] += int(summary["prediction_count"])
        categories[category]["subject_count"] += 1

    for item in categories.values():
        total = int(item["question_count"])
        item["accuracy"] = float(item["correct"] / total) if total else math.nan
    return [categories[category] for category in CATEGORY_ORDER]


def write_outputs(
    output_dir: Path,
    summaries: list[dict[str, Any]],
    records: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    overall_total = sum(int(item["total"]) for item in summaries)
    overall_correct = sum(int(item["correct"]) for item in summaries)
    overall_predictions = sum(int(item["prediction_count"]) for item in summaries)
    overall_accuracy = overall_correct / overall_total if overall_total else math.nan
    payload = {
        "checkpoint": args.model_path,
        "dataset": args.dataset,
        "split": args.split,
        "n_shot": args.n_shot,
        "batch_size": args.batch_size,
        "scoring_method": "batched_mlm_cloze_option_token_log_probability",
        "normalize_by_length": args.normalize_by_length,
        "overall": {
            "prediction_count": overall_predictions,
            "total": overall_total,
            "correct": overall_correct,
            "accuracy": overall_accuracy,
        },
        "categories": build_category_summaries(summaries),
        "subjects": summaries,
    }

    with (output_dir / "ceval_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    with (output_dir / "ceval_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["subject", "id", "prediction", "answer", "correct"] + [f"score_{choice}" for choice in CHOICES]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)

    with (output_dir / "ceval_category_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["category", "correct", "question_count", "prediction_count", "accuracy", "subject_count"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(payload["categories"])


def parse_subjects(value: str, max_subjects: int | None) -> list[str]:
    if value == "all":
        subjects = list(ALL_SUBJECTS)
    else:
        subjects = [subject.strip() for subject in value.split(",") if subject.strip()]
    unknown = [subject for subject in subjects if subject not in SUBJECT_CATEGORIES]
    if unknown:
        raise ValueError(f"Unknown C-Eval subject(s): {', '.join(unknown)}")
    if max_subjects is not None:
        subjects = subjects[: int(max_subjects)]
    return subjects


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run C-Eval for an encoder-only MLM checkpoint with only a model path required."
    )
    parser.add_argument("model_path", help="Local checkpoint directory, for example runs/.../latest.")
    parser.add_argument("--dataset", default=os.environ.get("CEVAL_DATASET", "ceval/ceval-exam"))
    parser.add_argument("--split", default=os.environ.get("CEVAL_SPLIT", "val"), choices=["dev", "val", "test"])
    parser.add_argument("--n-shot", type=int, default=int(os.environ.get("CEVAL_N_SHOT", "5")))
    parser.add_argument("--subjects", default=os.environ.get("CEVAL_SUBJECTS", "all"))
    parser.add_argument("--limit", type=int, default=None, help="Optional per-subject row limit for smoke tests.")
    parser.add_argument("--max-subjects", type=int, default=None, help="Optional subject cap for smoke tests.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=os.environ.get("CEVAL_DEVICE", "auto"))
    parser.add_argument("--dtype", default=os.environ.get("CEVAL_DTYPE", "bf16"), choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("CEVAL_BATCH_SIZE", "16")))
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--normalize-by-length", action="store_true")
    parser.add_argument(
        "--dataset-local-files-only",
        action="store_true",
        help="Use only the local Hugging Face dataset cache instead of checking the network.",
    )
    args = parser.parse_args()

    if MISSING_DEPENDENCY_ERROR is not None:
        raise SystemExit(
            "Missing evaluation dependency. Activate the project environment first, for example:\n"
            "  conda activate chatlm-encoder\n"
            "or install the package dependencies with:\n"
            '  pip install -e ".[deepspeed]"\n'
            f"Original import error: {MISSING_DEPENDENCY_ERROR}"
        )

    checkpoint_path = Path(args.model_path).expanduser()
    model_source = str(checkpoint_path) if checkpoint_path.exists() else args.model_path
    output_dir = Path(
        args.output_dir
        or f"eval_results/ceval/{safe_name(args.model_path)}_{args.split}_{args.n_shot}shot_onepath"
    )
    subjects = parse_subjects(args.subjects, args.max_subjects)

    device = select_device(args.device)
    torch_dtype = dtype_for(args.dtype, device)
    log(f"[ceval] model={model_source}")
    log(f"[ceval] device={device} dtype={torch_dtype} batch_size={args.batch_size}")
    log(f"[ceval] split={args.split} n_shot={args.n_shot} subjects={len(subjects)} output={output_dir}")

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    log("[ceval] loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=False)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    log("[ceval] loading model")
    model = AutoModelForMaskedLM.from_pretrained(
        model_source,
        torch_dtype=torch_dtype,
        trust_remote_code=False,
    ).to(device)
    model.eval()

    max_length = args.max_length
    if max_length is None:
        config_max = getattr(model.config, "max_position_embeddings", None)
        tokenizer_max = getattr(tokenizer, "model_max_length", None)
        max_length = int(config_max or tokenizer_max or 512)
    log(f"[ceval] max_length={max_length}")

    summaries: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    started = time.time()
    for index, subject in enumerate(subjects, start=1):
        subject_started = time.time()
        log(f"[ceval] [{index}/{len(subjects)}] loading/evaluating {subject}")
        summary, subject_records = evaluate_subject(
            model=model,
            tokenizer=tokenizer,
            dataset_name=args.dataset,
            subject=subject,
            split=args.split,
            n_shot=args.n_shot,
            limit=args.limit,
            device=device,
            max_length=max_length,
            batch_size=args.batch_size,
            normalize_by_length=args.normalize_by_length,
            dataset_local_files_only=args.dataset_local_files_only,
        )
        summaries.append(summary)
        records.extend(subject_records)
        write_outputs(output_dir=output_dir, summaries=summaries, records=records, args=args)
        elapsed = time.time() - subject_started
        log(
            f"[ceval] {subject}: accuracy={summary['accuracy']:.4f} "
            f"({summary['correct']}/{summary['total']}) predictions={summary['prediction_count']} "
            f"time={elapsed:.1f}s"
        )
        log(f"[ceval] partial results written to {output_dir}")

    total = sum(int(item["total"]) for item in summaries)
    correct = sum(int(item["correct"]) for item in summaries)
    accuracy = correct / total if total else math.nan
    log(f"[ceval] done in {(time.time() - started) / 60:.1f} min")
    log(f"C-Eval {args.split} {args.n_shot}-shot MLM-cloze accuracy: {accuracy:.4f} ({correct}/{total})")
    log(f"Wrote results to {output_dir}")


if __name__ == "__main__":
    main()
