from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gsm8k_lora_smoke import apply_canonicalizer, tokenize_example


def select_prefix(split, count: int):
    if count < 0:
        return split
    return split.select(range(min(count, len(split))))


def count_parameters(model) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/scratch/gpfs/ARORA/xd7812/models/Qwen3-0.6B")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--canonicalizer",
        choices=[
            "original",
            "vo_norm",
            "mlp_norm",
            "combined",
            "actgrad_vo",
            "actgrad_mlp",
            "actgrad_combined",
            "actgrad_mlp_gmean",
            "actgrad_combined_gmean",
        ],
        default="original",
    )
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lr-scheduler-type", default="constant_with_warmup")
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--train-samples", type=int, default=4096)
    parser.add_argument("--eval-samples", type=int, default=-1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--dataloader-num-workers", type=int, default=2)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-final-model", action=argparse.BooleanOptionalAction, default=False)
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
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    canon_info = apply_canonicalizer(model, tokenizer, args)
    trainable_params, total_params = count_parameters(model)
    print(f"trainable params: {trainable_params:,} || all params: {total_params:,}")

    ds = load_dataset("openai/gsm8k", "main")
    train = select_prefix(ds["train"].shuffle(seed=args.seed), args.train_samples)
    eval_ds = select_prefix(ds["test"].shuffle(seed=args.seed), args.eval_samples)
    train_tok = train.map(lambda ex: tokenize_example(ex, tokenizer, args.max_length), remove_columns=train.column_names)
    eval_tok = eval_ds.map(lambda ex: tokenize_example(ex, tokenizer, args.max_length), remove_columns=eval_ds.column_names)

    training_args = TrainingArguments(
        output_dir=str(out_dir / "trainer"),
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        save_strategy="no",
        eval_strategy="no",
        bf16=True,
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_num_workers,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=train_tok, eval_dataset=eval_tok)
    train_result = trainer.train()
    eval_result = trainer.evaluate()

    if args.save_final_model:
        trainer.save_model(str(out_dir / "final_model"))
        tokenizer.save_pretrained(out_dir / "final_model")

    first_grad_norm = next((entry["grad_norm"] for entry in trainer.state.log_history if "grad_norm" in entry), None)
    peak_cuda_memory_gb = None
    if torch.cuda.is_available():
        peak_cuda_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)

    metrics = {
        "model_path": args.model_path,
        "adaptation": "full_finetune",
        "canonicalizer": args.canonicalizer,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "lr_scheduler_type": args.lr_scheduler_type,
        "warmup_ratio": args.warmup_ratio,
        "max_steps": args.max_steps,
        "train_samples": len(train),
        "eval_samples": len(eval_ds),
        "max_length": args.max_length,
        "seed": args.seed,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "gradient_checkpointing": args.gradient_checkpointing,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "canonicalizer_summary": canon_info["summary"],
        "train_metrics": train_result.metrics,
        "eval_metrics": eval_result,
        "train_loss": train_result.metrics.get("train_loss"),
        "eval_loss": eval_result.get("eval_loss"),
        "train_runtime": train_result.metrics.get("train_runtime"),
        "first_grad_norm": first_grad_norm,
        "peak_cuda_memory_gb": peak_cuda_memory_gb,
        "log_history": trainer.state.log_history,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (out_dir / "canon_records.jsonl").write_text(
        "\n".join(json.dumps(r) for r in canon_info["records"]) + ("\n" if canon_info["records"] else "")
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
