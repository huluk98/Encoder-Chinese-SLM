#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import torch
from datasets import get_dataset_config_names, load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

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


def normalize_subject(subject: str) -> str:
    return subject.replace("_", " ")


def clean_answer(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().upper()
    match = re.search(r"[ABCD]", text)
    return match.group(0) if match else None


def build_category_summaries(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    categories = {
        category: {"category": category, "correct": 0, "question_count": 0, "subject_count": 0, "accuracy": math.nan}
        for category in CATEGORY_ORDER
    }
    for summary in summaries:
        subject = str(summary["subject"])
        category = SUBJECT_CATEGORIES.get(subject)
        if category is None:
            raise KeyError(f"Missing C-Eval category mapping for subject: {subject}")
        categories[category]["correct"] += int(summary["correct"])
        categories[category]["question_count"] += int(summary["total"])
        categories[category]["subject_count"] += 1

    for item in categories.values():
        total = int(item["question_count"])
        item["accuracy"] = float(item["correct"] / total) if total else math.nan
    return [categories[category] for category in CATEGORY_ORDER]


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


@torch.no_grad()
def score_choices(
    model: AutoModelForMaskedLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    device: torch.device,
    max_length: int | None,
    normalize_by_length: bool,
) -> dict[str, float]:
    if tokenizer.mask_token_id is None:
        raise ValueError("Tokenizer must define mask_token_id for C-Eval MLM scoring.")

    prefix_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    encoded: list[tuple[str, list[int], list[int]]] = []
    for choice in CHOICES:
        candidate_ids = tokenizer(choice, add_special_tokens=False)["input_ids"]
        if not candidate_ids:
            candidate_ids = tokenizer(f" {choice}", add_special_tokens=False)["input_ids"]
        current_prefix_ids = trim_prefix(prefix_ids, len(candidate_ids), max_length)
        encoded.append((choice, current_prefix_ids, candidate_ids))

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    lengths = [len(current_prefix_ids) + len(candidate_ids) for _, current_prefix_ids, candidate_ids in encoded]
    max_len = max(lengths)
    input_ids = torch.full((len(encoded), max_len), int(pad_id), dtype=torch.long)
    attention_mask = torch.zeros((len(encoded), max_len), dtype=torch.long)
    label_positions: list[tuple[str, list[int], list[int]]] = []

    for row_idx, (choice, current_prefix_ids, candidate_ids) in enumerate(encoded):
        ids = current_prefix_ids + candidate_ids
        candidate_positions = list(range(len(current_prefix_ids), len(ids)))
        masked_ids = ids[:]
        for position in candidate_positions:
            masked_ids[position] = int(tokenizer.mask_token_id)
        input_ids[row_idx, : len(masked_ids)] = torch.tensor(masked_ids, dtype=torch.long)
        attention_mask[row_idx, : len(masked_ids)] = 1
        label_positions.append((choice, candidate_positions, candidate_ids))

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    log_probs = torch.log_softmax(logits, dim=-1)

    scores: dict[str, float] = {}
    for row_idx, (choice, positions, candidate_ids) in enumerate(label_positions):
        score = 0.0
        for position, token_id in zip(positions, candidate_ids, strict=True):
            score += float(log_probs[row_idx, position, int(token_id)].detach().cpu())
        if normalize_by_length and candidate_ids:
            score /= len(candidate_ids)
        scores[choice] = score
    return scores


def load_subject_rows(dataset_name: str, subject: str, split: str) -> list[dict[str, Any]]:
    dataset = load_dataset(dataset_name, name=subject, split=split)
    return [dict(row) for row in dataset]


def evaluate_rows(
    model: AutoModelForMaskedLM,
    tokenizer: AutoTokenizer,
    eval_rows: list[dict[str, Any]],
    subject: str,
    split: str,
    n_shot: int,
    fewshot_rows: list[dict[str, Any]],
    limit: int | None,
    device: torch.device,
    max_length: int | None,
    normalize_by_length: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if limit is not None:
        eval_rows = eval_rows[: int(limit)]

    records: list[dict[str, Any]] = []
    correct = 0
    answered = 0
    for row in tqdm(eval_rows, desc=subject, leave=False):
        answer = clean_answer(row.get("answer"))
        if answer is None:
            continue
        prompt = build_prompt(row, subject=subject, fewshot_rows=fewshot_rows)
        scores = score_choices(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            max_length=max_length,
            normalize_by_length=normalize_by_length,
        )
        prediction = max(scores, key=scores.get)
        is_correct = prediction == answer
        correct += int(is_correct)
        answered += 1
        records.append(
            {
                "subject": subject,
                "id": row.get("id"),
                "prediction": prediction,
                "answer": answer,
                "correct": is_correct,
                **{f"score_{choice}": scores[choice] for choice in CHOICES},
            }
        )

    accuracy = correct / answered if answered else math.nan
    summary = {
        "subject": subject,
        "split": split,
        "n_shot": n_shot,
        "total": answered,
        "correct": correct,
        "accuracy": accuracy,
    }
    return summary, records


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
    normalize_by_length: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    eval_rows = load_subject_rows(dataset_name, subject, split)
    fewshot_rows: list[dict[str, Any]] = []
    if n_shot > 0:
        try:
            fewshot_rows = load_subject_rows(dataset_name, subject, "dev")[: int(n_shot)]
        except Exception:
            fewshot_rows = []
    return evaluate_rows(
        model=model,
        tokenizer=tokenizer,
        eval_rows=eval_rows,
        subject=subject,
        split=split,
        n_shot=n_shot,
        fewshot_rows=fewshot_rows,
        limit=limit,
        device=device,
        max_length=max_length,
        normalize_by_length=normalize_by_length,
    )


def write_outputs(output_dir: Path, summaries: list[dict[str, Any]], records: list[dict[str, Any]], args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    overall_total = sum(int(item["total"]) for item in summaries)
    overall_correct = sum(int(item["correct"]) for item in summaries)
    overall_accuracy = overall_correct / overall_total if overall_total else math.nan
    category_summaries = build_category_summaries(summaries)
    payload = {
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "split": args.split,
        "n_shot": args.n_shot,
        "scoring_method": "mlm_cloze_option_token_log_probability",
        "normalize_by_length": args.normalize_by_length,
        "overall": {
            "total": overall_total,
            "correct": overall_correct,
            "accuracy": overall_accuracy,
        },
        "categories": category_summaries,
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
        fieldnames = ["category", "correct", "question_count", "accuracy", "subject_count"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(category_summaries)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an encoder-only MLM checkpoint on C-Eval by cloze-scoring A/B/C/D.")
    parser.add_argument(
        "--checkpoint",
        "--model-path",
        dest="checkpoint",
        required=True,
        help="Model checkpoint directory, for example runs/h20-8gpu-bert-0p2b-mlm-deepspeed/latest.",
    )
    parser.add_argument("--dataset", default="ceval/ceval-exam", help="Hugging Face dataset name.")
    parser.add_argument("--subjects", default="all", help="Comma-separated C-Eval subjects, or all.")
    parser.add_argument("--split", default="val", choices=["dev", "val", "test"], help="Evaluation split.")
    parser.add_argument("--n-shot", type=int, default=5, help="Number of dev examples to include as few-shot context.")
    parser.add_argument("--limit", type=int, default=None, help="Optional per-subject row limit for debugging.")
    parser.add_argument("--max-subjects", type=int, default=None, help="Optional subject cap for debugging.")
    parser.add_argument("--output-dir", default=None, help="Output directory for C-Eval files.")
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0, cpu, or mps.")
    parser.add_argument("--dtype", default="bf16", choices=["auto", "bf16", "fp16", "fp32"], help="Model load dtype.")
    parser.add_argument("--max-length", type=int, default=None, help="Override max sequence length for prompt trimming.")
    parser.add_argument("--normalize-by-length", action="store_true", help="Average option log-probability by token count.")
    args = parser.parse_args()

    device = select_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=False)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForMaskedLM.from_pretrained(
        args.checkpoint,
        torch_dtype=dtype_for(args.dtype, device),
        trust_remote_code=False,
    ).to(device)
    model.eval()

    max_length = args.max_length
    if max_length is None:
        config_max = getattr(model.config, "max_position_embeddings", None)
        tokenizer_max = getattr(tokenizer, "model_max_length", None)
        max_length = int(config_max or tokenizer_max or 512)

    if args.subjects == "all":
        subjects = get_dataset_config_names(args.dataset)
    else:
        subjects = [subject.strip() for subject in args.subjects.split(",") if subject.strip()]
    if args.max_subjects is not None:
        subjects = subjects[: int(args.max_subjects)]

    summaries: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for subject in tqdm(subjects, desc="subjects"):
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
            normalize_by_length=args.normalize_by_length,
        )
        summaries.append(summary)
        records.extend(subject_records)
        print(f"{subject}: {summary['accuracy']:.4f} ({summary['correct']}/{summary['total']})")

    output_dir = Path(args.output_dir or Path(args.checkpoint) / "eval" / f"ceval_{args.split}_{args.n_shot}shot").expanduser()
    write_outputs(output_dir=output_dir, summaries=summaries, records=records, args=args)

    total = sum(int(item["total"]) for item in summaries)
    correct = sum(int(item["correct"]) for item in summaries)
    accuracy = correct / total if total else math.nan
    print(f"C-Eval {args.split} {args.n_shot}-shot MLM-cloze accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"Wrote results to {output_dir}")


if __name__ == "__main__":
    main()
