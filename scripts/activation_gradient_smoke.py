from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from canon.model_utils import get_decoder_layers
from canon.transforms import (
    apply_gqa_value_output_covariance_balance,
    apply_swiglu_mlp_activation_gradient_balance,
    summarize_records,
)


PROMPTS = [
    "Question: If Alice has 3 apples and buys 5 more, how many apples does she have?\nAnswer:",
    "Solve for x: 2x + 7 = 19.\nAnswer:",
    "Write a Python function that returns the factorial of n.\n",
    "The capital of France is",
]


def format_example(example: dict[str, str]) -> tuple[str, str]:
    return f"Question: {example['question']}\nAnswer:", " " + example["answer"]


def tokenize_lm(example, tokenizer, max_length: int) -> dict[str, torch.Tensor]:
    prompt, answer = format_example(example)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full = tokenizer(prompt + answer + tokenizer.eos_token, truncation=True, max_length=max_length, padding="max_length")
    labels = list(full["input_ids"])
    prompt_len = min(len(prompt_ids), max_length)
    for i in range(prompt_len):
        labels[i] = -100
    labels = [(-100 if tok == tokenizer.pad_token_id else lab) for tok, lab in zip(full["input_ids"], labels)]
    return {
        "input_ids": torch.tensor(full["input_ids"], dtype=torch.long).unsqueeze(0),
        "attention_mask": torch.tensor(full["attention_mask"], dtype=torch.long).unsqueeze(0),
        "labels": torch.tensor(labels, dtype=torch.long).unsqueeze(0),
    }


@torch.no_grad()
def logits_for_prompts(model, tokenizer, device: str, max_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    batch = tokenizer(PROMPTS, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    batch = {k: v.to(device) for k, v in batch.items()}
    logits = model(**batch).logits.detach().float().cpu()
    mask = batch["attention_mask"].detach().cpu().bool()
    return logits, mask


def compare_logits(reference: tuple[torch.Tensor, torch.Tensor], candidate: tuple[torch.Tensor, torch.Tensor]) -> dict[str, float]:
    ref, mask = reference
    cand, cand_mask = candidate
    if not torch.equal(mask, cand_mask):
        raise ValueError("attention masks differ")
    diff = cand[mask].float() - ref[mask].float()
    last_idx = mask.long().sum(dim=1) - 1
    batch_idx = torch.arange(ref.shape[0])
    ref_last = ref[batch_idx, last_idx, :].float()
    cand_last = cand[batch_idx, last_idx, :].float()
    ref_logp = F.log_softmax(ref_last, dim=-1)
    cand_logp = F.log_softmax(cand_last, dim=-1)
    kl = F.kl_div(cand_logp, ref_logp.exp(), reduction="batchmean", log_target=False)
    return {
        "mse_valid_logits": float(torch.mean(diff**2).item()),
        "max_abs_valid_logits": float(torch.max(torch.abs(diff)).item()),
        "last_token_kl_ref_to_candidate": float(kl.item()),
        "last_token_top1_agreement": float((torch.argmax(ref_last, dim=-1) == torch.argmax(cand_last, dim=-1)).float().mean().item()),
    }


def collect_stats(model, tokenizer, *, max_samples: int, max_length: int, device: str, seed: int):
    cfg = model.config
    layers = get_decoder_layers(model)
    num_layers = len(layers)
    num_heads = int(cfg.num_attention_heads)
    num_kv_heads = int(cfg.num_key_value_heads)
    head_dim = int(getattr(cfg, "head_dim", cfg.hidden_size // num_heads))
    heads_per_kv = num_heads // num_kv_heads
    intermediate = int(cfg.intermediate_size)

    cov_z = [torch.zeros(num_kv_heads, head_dim, head_dim, dtype=torch.float64) for _ in range(num_layers)]
    cov_g = [torch.zeros(num_kv_heads, head_dim, head_dim, dtype=torch.float64) for _ in range(num_layers)]
    cov_count = [torch.zeros(num_kv_heads, dtype=torch.float64) for _ in range(num_layers)]
    h2 = [torch.zeros(intermediate, dtype=torch.float64) for _ in range(num_layers)]
    gh2 = [torch.zeros(intermediate, dtype=torch.float64) for _ in range(num_layers)]
    h_count = [0 for _ in range(num_layers)]

    current = []
    hooks = []

    def make_o_hook(layer_idx: int):
        def hook(_module, inputs):
            x = inputs[0]
            x.retain_grad()
            current.append(("o", layer_idx, x))
        return hook

    def make_mlp_hook(layer_idx: int):
        def hook(_module, inputs):
            x = inputs[0]
            x.retain_grad()
            current.append(("mlp", layer_idx, x))
        return hook

    for layer_idx, layer in enumerate(layers):
        hooks.append(layer.self_attn.o_proj.register_forward_pre_hook(make_o_hook(layer_idx)))
        hooks.append(layer.mlp.down_proj.register_forward_pre_hook(make_mlp_hook(layer_idx)))

    ds = load_dataset("openai/gsm8k", "main")["train"].shuffle(seed=seed).select(range(max_samples))
    losses = []
    model.train()
    try:
        for example in ds:
            current.clear()
            batch = tokenize_lm(example, tokenizer, max_length)
            batch = {k: v.to(device) for k, v in batch.items()}
            model.zero_grad(set_to_none=True)
            out = model(**batch)
            loss = out.loss
            loss.backward()
            losses.append(float(loss.detach().cpu().item()))
            valid = batch["attention_mask"].detach().cpu().bool()

            for kind, layer_idx, tensor in current:
                grad = tensor.grad
                if grad is None:
                    continue
                x = tensor.detach().float().cpu()
                g = grad.detach().float().cpu()
                if kind == "o":
                    x = x[valid].reshape(-1, num_heads, head_dim)
                    g = g[valid].reshape(-1, num_heads, head_dim)
                    for kv_idx in range(num_kv_heads):
                        h0 = kv_idx * heads_per_kv
                        h1 = h0 + heads_per_kv
                        z = x[:, h0:h1, :].reshape(-1, head_dim).double()
                        gz = g[:, h0:h1, :].reshape(-1, head_dim).double()
                        cov_z[layer_idx][kv_idx] += z.T @ z
                        cov_g[layer_idx][kv_idx] += gz.T @ gz
                        cov_count[layer_idx][kv_idx] += z.shape[0]
                else:
                    x = x[valid].reshape(-1, intermediate).double()
                    g = g[valid].reshape(-1, intermediate).double()
                    h2[layer_idx] += (x * x).sum(dim=0)
                    gh2[layer_idx] += (g * g).sum(dim=0)
                    h_count[layer_idx] += x.shape[0]
            model.zero_grad(set_to_none=True)
    finally:
        for hook in hooks:
            hook.remove()

    for layer_idx in range(num_layers):
        for kv_idx in range(num_kv_heads):
            denom = max(float(cov_count[layer_idx][kv_idx].item()), 1.0)
            cov_z[layer_idx][kv_idx] /= denom
            cov_g[layer_idx][kv_idx] /= denom
        denom = max(float(h_count[layer_idx]), 1.0)
        h2[layer_idx] /= denom
        gh2[layer_idx] /= denom

    return cov_z, cov_g, h2, gh2, {"proxy_loss_mean": sum(losses) / len(losses), "proxy_loss_count": len(losses)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/scratch/gpfs/ARORA/xd7812/models/Qwen3-0.6B")
    parser.add_argument("--out-dir", default="results/activation_gradient_smoke/qwen3_0_6b")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True, torch_dtype=dtype)
    model.to(args.device)
    base_logits = logits_for_prompts(model.eval(), tokenizer, args.device, args.max_length)

    cov_z, cov_g, h2, gh2, stats_summary = collect_stats(
        model,
        tokenizer,
        max_samples=args.max_samples,
        max_length=args.max_length,
        device=args.device,
        seed=args.seed,
    )
    model.eval()
    vo_records = apply_gqa_value_output_covariance_balance(model, cov_z, cov_g, trace_normalize=True)
    mlp_records = apply_swiglu_mlp_activation_gradient_balance(model, h2, gh2)
    transformed_logits = logits_for_prompts(model, tokenizer, args.device, args.max_length)

    metrics = {
        "model_path": args.model_path,
        "dtype": args.dtype,
        "max_samples": args.max_samples,
        "max_length": args.max_length,
        "seed": args.seed,
        "stats_summary": stats_summary,
        "actgrad_combined_vs_original": compare_logits(base_logits, transformed_logits),
        "vo_summary": summarize_records(vo_records),
        "mlp_summary": summarize_records(mlp_records),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (out_dir / "vo_records.jsonl").write_text("\n".join(json.dumps(r.to_dict()) for r in vo_records) + "\n")
    (out_dir / "mlp_records.jsonl").write_text("\n".join(json.dumps(r.to_dict()) for r in mlp_records) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
