from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gsm8k_lora_smoke import CANONICALIZER_CHOICES, apply_canonicalizer


def parse_step_list(spec: str) -> list[int]:
    if not spec:
        return []
    return sorted({int(token.strip()) for token in spec.split(",") if token.strip()})


class MidTrainingCanonCallback(TrainerCallback):
    """Same callback shape as math_full_finetune's: re-applies canonicalizer at
    requested global steps, optionally clearing Adam state."""

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


def select_prefix(split, count: int):
    if count < 0:
        return split
    return split.select(range(min(count, len(split))))


def _is_text_only(messages) -> bool:
    """Return True iff every message's content is plain text (no image/audio dicts)."""
    if not isinstance(messages, list):
        return False
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            # multimodal content list -- skip
            return False
        if not isinstance(content, str):
            return False
    return True


def build_tokenize_fn(tokenizer, max_length: int):
    """Return a map function that emits {input_ids, attention_mask, labels} for one example.

    Tülu-3 SFT mixture examples have ``messages = [{"role", "content"}, ...]``. We apply the
    tokenizer's chat template once to the full conversation (training target) and once to the
    conversation truncated before the *last* assistant turn (the prompt portion) so we can mask
    everything except the final assistant response from the loss.
    """
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def _tokenize(example):
        messages = example["messages"]
        # find the last assistant turn; train on its content only
        last_assistant_idx = max(
            (i for i, m in enumerate(messages) if m.get("role") == "assistant"),
            default=-1,
        )
        if last_assistant_idx < 0:
            # no assistant turn -- skip by returning a dummy 1-token mask (filtered later)
            return {
                "input_ids": [pad_id] * max_length,
                "attention_mask": [0] * max_length,
                "labels": [-100] * max_length,
            }
        prompt_msgs = messages[:last_assistant_idx]

        prompt_text = tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True
        )
        full_text = tokenizer.apply_chat_template(
            messages[: last_assistant_idx + 1], tokenize=False, add_generation_prompt=False
        )
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

        prompt_len = min(len(prompt_ids), max_length)
        full_ids = full_ids[:max_length]
        labels = [-100] * prompt_len + full_ids[prompt_len:]
        labels = labels[: len(full_ids)]
        attention_mask = [1] * len(full_ids)

        # pad
        pad_n = max_length - len(full_ids)
        if pad_n > 0:
            full_ids = full_ids + [pad_id] * pad_n
            attention_mask = attention_mask + [0] * pad_n
            labels = labels + [-100] * pad_n

        return {"input_ids": full_ids, "attention_mask": attention_mask, "labels": labels}

    return _tokenize


def count_parameters(model) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def maybe_freeze_modules(model, *, freeze_vision_audio: bool, freeze_per_layer_emb: bool) -> dict:
    """Freeze multimodal towers / per-layer embedding tables on Gemma-4. Returns a report."""
    report = {
        "frozen_vision_params": 0,
        "frozen_audio_params": 0,
        "frozen_per_layer_emb_params": 0,
    }
    inner = getattr(model, "model", model)  # Gemma4ForConditionalGeneration -> .model

    if freeze_vision_audio:
        for name in ("vision_tower", "audio_tower"):
            mod = getattr(inner, name, None)
            if mod is not None:
                for p in mod.parameters():
                    p.requires_grad = False
                    report[f"frozen_{'vision' if name == 'vision_tower' else 'audio'}_params"] += p.numel()

    if freeze_per_layer_emb:
        lm = getattr(inner, "language_model", None)
        if lm is not None:
            for cand in ("embed_tokens_per_layer", "per_layer_embed_tokens", "per_layer_input_embedding"):
                mod = getattr(lm, cand, None)
                if mod is not None:
                    for p in mod.parameters():
                        p.requires_grad = False
                        report["frozen_per_layer_emb_params"] += p.numel()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--canonicalizer", choices=CANONICALIZER_CHOICES, default="original")
    parser.add_argument("--dataset", default="allenai/tulu-3-sft-mixture")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--train-samples", type=int, default=8192)
    parser.add_argument("--eval-samples", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--dataloader-num-workers", type=int, default=2)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-final-model", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--freeze-vision-audio", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--freeze-per-layer-emb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--actgrad-samples", type=int, default=8)
    parser.add_argument("--actgrad-max-length", type=int, default=None)
    parser.add_argument("--mid-canon-steps", default="")
    parser.add_argument("--mid-canon-mode", choices=["lazy", "reset"], default="lazy")
    parser.add_argument("--mid-canon-name", default=None)
    parser.add_argument("--eval-strategy-mode", choices=["none", "steps"], default="none")
    parser.add_argument("--eval-steps-interval", type=int, default=250)
    parser.add_argument("--eval-on-start", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Gemma-4 uses AutoProcessor (multimodal); fall back to AutoTokenizer for non-MM models.
    try:
        tokenizer = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
        if hasattr(tokenizer, "tokenizer"):
            tok = tokenizer.tokenizer
        else:
            tok = tokenizer
    except (OSError, ValueError):
        tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    if hasattr(model, "config"):
        model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    freeze_report = maybe_freeze_modules(
        model,
        freeze_vision_audio=args.freeze_vision_audio,
        freeze_per_layer_emb=args.freeze_per_layer_emb,
    )

    canon_info = apply_canonicalizer(model, tok, args)
    trainable_params, total_params = count_parameters(model)
    print(f"trainable params: {trainable_params:,} || all params: {total_params:,}")
    print("freeze report:", json.dumps(freeze_report))

    # Load dataset and tokenize
    ds_full = load_dataset(args.dataset, split=args.dataset_split)
    # Text-only filter
    ds_full = ds_full.filter(lambda ex: _is_text_only(ex.get("messages", [])), num_proc=4)
    ds_full = ds_full.shuffle(seed=args.seed)

    n_train = args.train_samples if args.train_samples > 0 else len(ds_full)
    n_eval = args.eval_samples if args.eval_samples > 0 else 0
    train_ds = ds_full.select(range(min(n_train, len(ds_full))))
    eval_ds = ds_full.select(range(n_train, min(n_train + n_eval, len(ds_full)))) if n_eval > 0 else None

    tok_fn = build_tokenize_fn(tok, args.max_length)
    keep_cols = [c for c in train_ds.column_names]
    train_tok = train_ds.map(tok_fn, remove_columns=keep_cols, num_proc=4)
    eval_tok = eval_ds.map(tok_fn, remove_columns=eval_ds.column_names, num_proc=4) if eval_ds is not None else None

    mid_canon_steps = parse_step_list(args.mid_canon_steps)
    mid_canon_name = args.mid_canon_name or args.canonicalizer
    mid_canon_journal: list[dict] = []

    ta_kwargs = dict(
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
    if args.eval_strategy_mode == "steps" and eval_tok is not None:
        ta_kwargs.update(
            eval_strategy="steps",
            eval_steps=args.eval_steps_interval,
            eval_on_start=args.eval_on_start,
        )
    else:
        ta_kwargs.update(eval_strategy="no")
    training_args = TrainingArguments(**ta_kwargs)

    callbacks = []
    if mid_canon_steps:
        callbacks.append(
            MidTrainingCanonCallback(
                canon_steps=mid_canon_steps,
                canon_name=mid_canon_name,
                canon_mode=args.mid_canon_mode,
                base_args=args,
                journal=mid_canon_journal,
                tokenizer=tok,
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
    eval_result = trainer.evaluate() if eval_tok is not None else {}

    if args.save_final_model:
        trainer.save_model(str(out_dir / "final_model"))
        try:
            tok.save_pretrained(out_dir / "final_model")
        except Exception:
            pass

    first_grad_norm = next((entry["grad_norm"] for entry in trainer.state.log_history if "grad_norm" in entry), None)
    peak_cuda_memory_gb = None
    if torch.cuda.is_available():
        peak_cuda_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)

    metrics = {
        "model_path": args.model_path,
        "adaptation": "full_finetune",
        "dataset": args.dataset,
        "canonicalizer": args.canonicalizer,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "lr_scheduler_type": args.lr_scheduler_type,
        "warmup_ratio": args.warmup_ratio,
        "max_steps": args.max_steps,
        "train_samples": len(train_ds),
        "eval_samples": len(eval_ds) if eval_ds is not None else 0,
        "max_length": args.max_length,
        "seed": args.seed,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "gradient_checkpointing": args.gradient_checkpointing,
        "freeze_report": freeze_report,
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
        "eval_loss": eval_result.get("eval_loss") if eval_result else None,
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
    print(json.dumps({k: v for k, v in metrics.items() if k != "log_history"}, indent=2))


if __name__ == "__main__":
    main()
