from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from datasets import concatenate_datasets, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments, set_seed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gsm8k_lora_smoke import CANONICALIZER_CHOICES, apply_canonicalizer


def parse_step_list(spec: str) -> list[int]:
    if not spec:
        return []
    return sorted({int(token.strip()) for token in spec.split(",") if token.strip()})


class MidTrainingCanonCallback(TrainerCallback):
    """Re-apply a canonicalizer at given global steps.

    In "lazy" mode, only model weights are gauged; the Adam state is left in the old basis.
    In "reset" mode, the optimizer state is additionally cleared so the next step's first/second
    moments rebuild in the new basis. This separates gauge effect from optimizer-restart effect.
    """

    def __init__(self, canon_steps, canon_name, canon_mode, base_args, journal, tokenizer=None):
        self.canon_steps = set(canon_steps)
        self.canon_name = canon_name
        self.canon_mode = canon_mode
        self.base_args = base_args
        self.journal = journal
        self.tokenizer = tokenizer

    def _apply(self, model, tokenizer):
        sub_args = SimpleNamespace(
            canonicalizer=self.canon_name,
            actgrad_samples=getattr(self.base_args, "actgrad_samples", 8),
            actgrad_max_length=getattr(self.base_args, "actgrad_max_length", None),
            max_length=getattr(self.base_args, "max_length", 1024),
            seed=getattr(self.base_args, "seed", 1),
        )
        return apply_canonicalizer(model, tokenizer, sub_args)

    def on_step_end(self, args, state, control, **kwargs):
        step = int(state.global_step)
        if step not in self.canon_steps:
            return control
        model = kwargs.get("model")
        tokenizer = kwargs.get("processing_class") or kwargs.get("tokenizer") or self.tokenizer
        optimizer = kwargs.get("optimizer")
        info = self._apply(model, tokenizer)
        cleared = False
        if self.canon_mode == "reset" and optimizer is not None:
            optimizer.state.clear()
            cleared = True
        self.journal.append({
            "step": step,
            "mode": self.canon_mode,
            "name": self.canon_name,
            "optimizer_state_cleared": cleared,
            "summary": info["summary"],
            "records": info["records"],
        })
        return control


DEFAULT_MATH_CONFIGS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]


def select_prefix(split, count: int):
    if count < 0:
        return split
    return split.select(range(min(count, len(split))))


def format_example(example: dict[str, str]) -> tuple[str, str]:
    prompt = f"Problem: {example['problem']}\nSolution:"
    answer = " " + example["solution"]
    return prompt, answer


def tokenize_example(example, tokenizer, max_length: int, min_answer_tokens: int):
    prompt, answer = format_example(example)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
    if len(prompt_ids) + len(answer_ids) > max_length:
        answer_budget = min(len(answer_ids), max(1, min_answer_tokens), max_length)
        prompt_budget = max(0, max_length - answer_budget)
        prompt_ids = prompt_ids[:prompt_budget]
        answer_ids = answer_ids[: max_length - len(prompt_ids)]

    input_ids = (prompt_ids + answer_ids)[:max_length]
    labels = [-100] * len(prompt_ids) + answer_ids[: max(0, max_length - len(prompt_ids))]
    labels = labels[: len(input_ids)]
    attention_mask = [1] * len(input_ids)

    pad_len = max_length - len(input_ids)
    if pad_len > 0:
        input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
        attention_mask = attention_mask + [0] * pad_len
        labels = labels + [-100] * pad_len

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def load_hendrycks_math(configs: list[str], split: str):
    parts = [load_dataset("EleutherAI/hendrycks_math", config)[split] for config in configs]
    return concatenate_datasets(parts)


def load_eval_dataset(configs: list[str], eval_source: str):
    if eval_source == "math500":
        return load_dataset("HuggingFaceH4/MATH-500")["test"]
    if eval_source == "hendrycks_test":
        return load_hendrycks_math(configs, "test")
    raise ValueError(f"unknown eval source: {eval_source}")


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
        choices=CANONICALIZER_CHOICES,
        default="original",
    )
    parser.add_argument("--math-configs", default=",".join(DEFAULT_MATH_CONFIGS))
    parser.add_argument("--eval-source", choices=["math500", "hendrycks_test"], default="math500")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lr-scheduler-type", default="constant_with_warmup")
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--train-samples", type=int, default=-1)
    parser.add_argument("--eval-samples", type=int, default=-1)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--min-answer-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--dataloader-num-workers", type=int, default=2)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-final-model", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--actgrad-samples", type=int, default=8)
    parser.add_argument("--actgrad-max-length", type=int, default=None)
    parser.add_argument(
        "--mid-canon-steps",
        default="",
        help="Comma-separated global steps at which to re-apply canonicalization (empty = none).",
    )
    parser.add_argument(
        "--mid-canon-mode",
        choices=["lazy", "reset"],
        default="lazy",
        help="lazy: only transform weights; reset: also clear optimizer state.",
    )
    parser.add_argument(
        "--mid-canon-name",
        default=None,
        help="Canonicalizer name for mid-training canon; defaults to --canonicalizer.",
    )
    parser.add_argument(
        "--eval-strategy-mode",
        choices=["none", "steps"],
        default="none",
        help="'steps' enables periodic eval and step-0 baseline eval for trajectory tracking.",
    )
    parser.add_argument("--eval-steps-interval", type=int, default=250)
    parser.add_argument("--eval-on-start", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    configs = [config.strip() for config in args.math_configs.split(",") if config.strip()]

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

    train = select_prefix(load_hendrycks_math(configs, "train").shuffle(seed=args.seed), args.train_samples)
    eval_ds = select_prefix(load_eval_dataset(configs, args.eval_source).shuffle(seed=args.seed), args.eval_samples)
    train_tok = train.map(
        lambda ex: tokenize_example(ex, tokenizer, args.max_length, args.min_answer_tokens),
        remove_columns=train.column_names,
    )
    eval_tok = eval_ds.map(
        lambda ex: tokenize_example(ex, tokenizer, args.max_length, args.min_answer_tokens),
        remove_columns=eval_ds.column_names,
    )

    mid_canon_steps = parse_step_list(args.mid_canon_steps)
    mid_canon_name = args.mid_canon_name or args.canonicalizer
    mid_canon_journal: list[dict] = []

    training_args_kwargs = dict(
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
        bf16=True,
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_num_workers,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    if args.eval_strategy_mode == "steps":
        training_args_kwargs.update(
            eval_strategy="steps",
            eval_steps=args.eval_steps_interval,
            eval_on_start=args.eval_on_start,
        )
    else:
        training_args_kwargs.update(eval_strategy="no")
    training_args = TrainingArguments(**training_args_kwargs)
    callbacks = []
    if mid_canon_steps:
        callbacks.append(
            MidTrainingCanonCallback(
                canon_steps=mid_canon_steps,
                canon_name=mid_canon_name,
                canon_mode=args.mid_canon_mode,
                base_args=args,
                journal=mid_canon_journal,
                tokenizer=tokenizer,
            )
        )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        callbacks=callbacks or None,
    )
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
        "dataset": "hendrycks_math",
        "math_configs": configs,
        "eval_source": args.eval_source,
        "canonicalizer": args.canonicalizer,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "lr_scheduler_type": args.lr_scheduler_type,
        "warmup_ratio": args.warmup_ratio,
        "max_steps": args.max_steps,
        "train_samples": len(train),
        "eval_samples": len(eval_ds),
        "max_length": args.max_length,
        "min_answer_tokens": args.min_answer_tokens,
        "seed": args.seed,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "gradient_checkpointing": args.gradient_checkpointing,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "canonicalizer_summary": canon_info["summary"],
        "mid_canon_steps": mid_canon_steps,
        "mid_canon_mode": args.mid_canon_mode,
        "mid_canon_name": mid_canon_name,
        "mid_canon_journal": [
            {k: v for k, v in entry.items() if k != "records"} for entry in mid_canon_journal
        ],
        "eval_strategy_mode": args.eval_strategy_mode,
        "eval_steps_interval": args.eval_steps_interval,
        "eval_on_start": args.eval_on_start,
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
    if mid_canon_journal:
        with (out_dir / "mid_canon_journal.jsonl").open("w") as handle:
            for entry in mid_canon_journal:
                handle.write(json.dumps(entry) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
