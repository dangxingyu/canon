from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


def select_examples(dataset, max_samples: int):
    if max_samples < 0:
        return dataset
    return dataset.select(range(min(max_samples, len(dataset))))


def make_prompt(problem: str) -> str:
    return f"Problem: {problem}\nSolution:"


def parse_braced_content(text: str, open_brace: int) -> str | None:
    depth = 0
    for idx in range(open_brace, len(text)):
        char = text[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[open_brace + 1 : idx]
    return None


def extract_last_boxed(text: str) -> str | None:
    values: list[str] = []
    for match in re.finditer(r"\\boxed\s*{", text):
        content = parse_braced_content(text, match.end() - 1)
        if content is not None:
            values.append(content)
    return values[-1] if values else None


def extract_answer(text: str) -> tuple[str, str]:
    boxed = extract_last_boxed(text)
    if boxed is not None:
        return boxed.strip(), "boxed"

    math_spans = re.findall(r"\$([^$]+)\$", text)
    if math_spans:
        return math_spans[-1].strip(), "last_dollar_math"

    answer_patterns = [
        r"(?:final answer|answer is|therefore)[^:\n.]*[:\s]+(.+)",
        r"(?:=)\s*([^\n.]+)\.?\s*$",
    ]
    for pattern in answer_patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return matches[-1].strip(), "answer_phrase"

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        return lines[-1], "last_line"
    return "", "empty"


def strip_outer_braces(text: str) -> str:
    changed = True
    while changed and len(text) >= 2 and text[0] == "{" and text[-1] == "}":
        changed = False
        depth = 0
        balanced_outer = True
        for idx, char in enumerate(text):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0 and idx != len(text) - 1:
                    balanced_outer = False
                    break
        if balanced_outer:
            text = text[1:-1].strip()
            changed = True
    return text


def normalize_answer(text: str) -> str:
    text = text.strip()
    boxed = extract_last_boxed(text)
    if boxed is not None:
        text = boxed
    return normalize_latex_text(text)


def normalize_latex_text(text: str) -> str:
    text = text.strip().strip("$").strip()
    text = re.sub(r"\\(?:left|right)", "", text)
    text = re.sub(r"\\(?:dfrac|tfrac)", r"\\frac", text)
    text = re.sub(r"\\(?:,|!|;|:)", "", text)
    text = re.sub(r"\\text\s*{([^{}]*)}", r"\1", text)
    text = re.sub(r"\\mathrm\s*{([^{}]*)}", r"\1", text)
    text = text.replace("\\ ", "")
    text = text.replace(" ", "")
    text = text.replace("\n", "")
    text = text.replace("\t", "")
    text = text.rstrip(".,;:")
    text = strip_outer_braces(text)
    return text.lower()


def summarize_groups(records: list[dict], key: str) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[dict]] = {}
    for record in records:
        grouped.setdefault(str(record[key]), []).append(record)
    summary = {}
    for value, group in sorted(grouped.items()):
        n = len(group)
        summary[value] = {
            "num_samples": n,
            "exact_match": sum(record["exact_match"] for record in group) / n if n else 0.0,
            "gold_in_generation": sum(record["gold_in_generation"] for record in group) / n if n else 0.0,
            "boxed_rate": sum(record["extraction_method"] == "boxed" for record in group) / n if n else 0.0,
        }
    return summary


@torch.inference_mode()
def generate_batch(model, tokenizer, prompts: list[str], max_input_tokens: int, max_new_tokens: int) -> list[str]:
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_tokens,
    )
    encoded = {key: value.to(model.device) for key, value in encoded.items()}
    generated = model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    prompt_width = encoded["input_ids"].shape[1]
    continuations = generated[:, prompt_width:]
    return tokenizer.batch_decode(continuations, skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-input-tokens", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--torch-dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=dtype_map[args.torch_dtype],
    )
    model.config.use_cache = True
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    dataset = select_examples(load_dataset("HuggingFaceH4/MATH-500")["test"], args.max_samples)
    records: list[dict] = []
    exact_matches = 0
    gold_in_generation = 0
    boxed_count = 0

    for start in range(0, len(dataset), args.batch_size):
        batch = dataset[start : start + args.batch_size]
        prompts = [make_prompt(problem) for problem in batch["problem"]]
        generations = generate_batch(model, tokenizer, prompts, args.max_input_tokens, args.max_new_tokens)
        for offset, generated_text in enumerate(generations):
            idx = start + offset
            gold = batch["answer"][offset]
            pred, method = extract_answer(generated_text)
            gold_norm = normalize_answer(gold)
            pred_norm = normalize_answer(pred)
            gen_norm = normalize_latex_text(generated_text)
            exact = pred_norm == gold_norm
            contains_gold = bool(gold_norm) and gold_norm in gen_norm
            exact_matches += int(exact)
            gold_in_generation += int(contains_gold)
            boxed_count += int(method == "boxed")
            records.append(
                {
                    "index": idx,
                    "unique_id": batch["unique_id"][offset],
                    "subject": batch["subject"][offset],
                    "level": batch["level"][offset],
                    "problem": batch["problem"][offset],
                    "gold_answer": gold,
                    "pred_answer": pred,
                    "gold_norm": gold_norm,
                    "pred_norm": pred_norm,
                    "extraction_method": method,
                    "exact_match": exact,
                    "gold_in_generation": contains_gold,
                    "generation": generated_text,
                }
            )

    n = len(records)
    metrics = {
        "model_path": args.model_path,
        "dataset": "HuggingFaceH4/MATH-500",
        "num_samples": n,
        "max_samples": args.max_samples,
        "batch_size": args.batch_size,
        "max_input_tokens": args.max_input_tokens,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "torch_dtype": args.torch_dtype,
        "exact_match": exact_matches / n if n else 0.0,
        "gold_in_generation": gold_in_generation / n if n else 0.0,
        "boxed_rate": boxed_count / n if n else 0.0,
        "num_exact_match": exact_matches,
        "num_gold_in_generation": gold_in_generation,
        "num_boxed": boxed_count,
        "by_subject": summarize_groups(records, "subject"),
        "by_level": summarize_groups(records, "level"),
    }
    (out_dir / "generation_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    with (out_dir / "predictions.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
