from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from activation_gradient_smoke import collect_stats
from canon.transforms import (
    apply_gqa_value_output_covariance_balance,
    apply_gqa_value_output_norm_balance,
    apply_gqa_value_output_orthogonal_balance,
    apply_swiglu_mlp_activation_gradient_balance,
    apply_swiglu_mlp_norm_balance,
    summarize_records,
)


TARGET_MODULES = {
    "all_linear": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "vo": ["v_proj", "o_proj"],
    "v_only": ["v_proj"],
    "mlp": ["gate_proj", "up_proj", "down_proj"],
}


def format_example(example: dict[str, str]) -> tuple[str, str]:
    prompt = f"Question: {example['question']}\nAnswer:"
    answer = " " + example["answer"]
    return prompt, answer


def tokenize_example(example, tokenizer, max_length: int):
    prompt, answer = format_example(example)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full = tokenizer(prompt + answer + tokenizer.eos_token, truncation=True, max_length=max_length, padding="max_length")
    labels = list(full["input_ids"])
    prompt_len = min(len(prompt_ids), max_length)
    for i in range(prompt_len):
        labels[i] = -100
    labels = [(-100 if tok == tokenizer.pad_token_id else lab) for tok, lab in zip(full["input_ids"], labels)]
    full["labels"] = labels
    return full


CANONICALIZER_CHOICES = [
    "original",
    "vo_norm",
    "mlp_norm",
    "combined",
    "vo_orth",
    "combined_orth",
    "actgrad_vo",
    "actgrad_mlp",
    "actgrad_combined",
    "actgrad_mlp_gmean",
    "actgrad_combined_gmean",
]


def apply_canonicalizer(model, tokenizer, args) -> dict[str, object]:
    name = args.canonicalizer
    if name not in CANONICALIZER_CHOICES:
        raise ValueError(f"unknown canonicalizer: {name}")
    records = []
    if name in {"vo_norm", "combined"}:
        records.extend(apply_gqa_value_output_norm_balance(model))
    if name in {"vo_orth", "combined_orth"}:
        records.extend(apply_gqa_value_output_orthogonal_balance(model))
    if name in {"mlp_norm", "combined", "combined_orth"}:
        # MLP canon is restricted to positive diagonal P regardless of attention
        # gauge family (elementwise gate * up product forbids non-diagonal P), so
        # combined_orth reuses the same MLP scaling as combined.
        records.extend(apply_swiglu_mlp_norm_balance(model))
    if name in {"actgrad_vo", "actgrad_mlp", "actgrad_combined", "actgrad_mlp_gmean", "actgrad_combined_gmean"}:
        cov_z, cov_g, h2, gh2, stats_summary = collect_stats(
            model,
            tokenizer,
            max_samples=args.actgrad_samples,
            max_length=args.actgrad_max_length or args.max_length,
            device="cuda" if torch.cuda.is_available() else "cpu",
            seed=args.seed,
        )
        if name in {"actgrad_vo", "actgrad_combined", "actgrad_combined_gmean"}:
            records.extend(apply_gqa_value_output_covariance_balance(model, cov_z, cov_g, trace_normalize=True))
        if name in {"actgrad_mlp", "actgrad_combined", "actgrad_mlp_gmean", "actgrad_combined_gmean"}:
            records.extend(
                apply_swiglu_mlp_activation_gradient_balance(
                    model,
                    h2,
                    gh2,
                    normalize_geomean=name in {"actgrad_mlp_gmean", "actgrad_combined_gmean"},
                )
            )
        return {"summary": summarize_records(records) | {"actgrad_proxy_loss_mean": stats_summary["proxy_loss_mean"]}, "records": [r.to_dict() for r in records]}
    if name == "original":
        return {"summary": {"count": 0}, "records": []}
    return {"summary": summarize_records(records), "records": [r.to_dict() for r in records]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/scratch/gpfs/ARORA/xd7812/models/Qwen3-0.6B")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--canonicalizer",
        choices=CANONICALIZER_CHOICES,
        default="original",
    )
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lr-scheduler-type", default="linear")
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--target-modules", choices=sorted(TARGET_MODULES), default="all_linear")
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--train-samples", type=int, default=512)
    parser.add_argument("--eval-samples", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--actgrad-samples", type=int, default=8)
    parser.add_argument("--actgrad-max-length", type=int, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    canon_info = apply_canonicalizer(model, tokenizer, args)

    lora_cfg = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES[args.target_modules],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    ds = load_dataset("openai/gsm8k", "main")
    train = ds["train"].shuffle(seed=args.seed).select(range(args.train_samples))
    eval_ds = ds["test"].shuffle(seed=args.seed).select(range(args.eval_samples))
    train_tok = train.map(lambda ex: tokenize_example(ex, tokenizer, args.max_length), remove_columns=train.column_names)
    eval_tok = eval_ds.map(lambda ex: tokenize_example(ex, tokenizer, args.max_length), remove_columns=eval_ds.column_names)

    training_args = TrainingArguments(
        output_dir=str(out_dir / "trainer"),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=args.lr,
        max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=5,
        save_strategy="no",
        eval_strategy="no",
        bf16=True,
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=2,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=train_tok, eval_dataset=eval_tok)
    train_result = trainer.train()
    eval_result = trainer.evaluate()

    first_grad_norm = next((entry["grad_norm"] for entry in trainer.state.log_history if "grad_norm" in entry), None)
    metrics = {
        "model_path": args.model_path,
        "canonicalizer": args.canonicalizer,
        "rank": args.rank,
        "lora_alpha": args.lora_alpha,
        "lr": args.lr,
        "lr_scheduler_type": args.lr_scheduler_type,
        "warmup_ratio": args.warmup_ratio,
        "target_modules": args.target_modules,
        "max_steps": args.max_steps,
        "train_samples": args.train_samples,
        "eval_samples": args.eval_samples,
        "max_length": args.max_length,
        "seed": args.seed,
        "canonicalizer_summary": canon_info["summary"],
        "train_metrics": train_result.metrics,
        "eval_metrics": eval_result,
        "train_loss": train_result.metrics.get("train_loss"),
        "eval_loss": eval_result.get("eval_loss"),
        "train_runtime": train_result.metrics.get("train_runtime"),
        "first_grad_norm": first_grad_norm,
        "log_history": trainer.state.log_history,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (out_dir / "canon_records.jsonl").write_text(
        "\n".join(json.dumps(r) for r in canon_info["records"]) + ("\n" if canon_info["records"] else "")
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
