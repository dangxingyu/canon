from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from canon.transforms import (
    apply_gqa_value_output_norm_balance,
    apply_gqa_value_output_orthogonal_balance,
    apply_swiglu_mlp_norm_balance,
    summarize_records,
)


PROMPTS = [
    "Question: If Alice has 3 apples and buys 5 more, how many apples does she have?\nAnswer:",
    "Solve for x: 2x + 7 = 19.\nAnswer:",
    "Write a Python function that returns the factorial of n.\n",
    "The capital of France is",
]


@torch.no_grad()
def logits_for_prompts(model, tokenizer, prompts: list[str], device: str, max_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    batch = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    batch = {k: v.to(device) for k, v in batch.items()}
    logits = model(**batch).logits.detach().float().cpu()
    mask = batch["attention_mask"].detach().cpu().bool()
    return logits, mask


def compare_logits(reference: tuple[torch.Tensor, torch.Tensor], candidate: tuple[torch.Tensor, torch.Tensor]) -> dict[str, float]:
    reference_logits, reference_mask = reference
    candidate_logits, candidate_mask = candidate
    if not torch.equal(reference_mask, candidate_mask):
        raise ValueError("reference and candidate attention masks differ")
    ref = reference_logits.float()
    cand = candidate_logits.float()
    valid = reference_mask
    valid_diff = cand[valid] - ref[valid]

    last_idx = valid.long().sum(dim=1) - 1
    batch_idx = torch.arange(ref.shape[0])
    ref_last = ref[batch_idx, last_idx, :]
    cand_last = cand[batch_idx, last_idx, :]
    ref_logp = F.log_softmax(ref_last, dim=-1)
    cand_logp = F.log_softmax(cand_last, dim=-1)
    kl = F.kl_div(cand_logp, ref_logp.exp(), reduction="batchmean", log_target=False)
    top1_ref = torch.argmax(ref_last, dim=-1)
    top1_cand = torch.argmax(cand_last, dim=-1)
    return {
        "mse_valid_logits": float(torch.mean(valid_diff ** 2).item()),
        "max_abs_valid_logits": float(torch.max(torch.abs(valid_diff)).item()),
        "last_token_kl_ref_to_candidate": float(kl.item()),
        "last_token_top1_agreement": float(torch.mean((top1_ref == top1_cand).float()).item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/scratch/gpfs/ARORA/xd7812/models/Qwen3-0.6B")
    parser.add_argument("--out-dir", default="results/static_smoke/qwen3_0_6b")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--lambda-scale", type=float, default=1e-6)
    parser.add_argument("--min-sv", type=float, default=0.25)
    parser.add_argument("--max-sv", type=float, default=4.0)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument(
        "--vo-canon",
        choices=["spd", "orthogonal"],
        default="spd",
        help="V/O canon family: SPD (Frobenius balance) or orthogonal (Procrustes).",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    model.eval().to(args.device)

    base_logits = logits_for_prompts(model, tokenizer, PROMPTS, args.device, args.max_length)

    if args.vo_canon == "spd":
        vo_records = apply_gqa_value_output_norm_balance(
            model,
            lambda_scale=args.lambda_scale,
            min_sv=args.min_sv,
            max_sv=args.max_sv,
        )
    else:
        vo_records = apply_gqa_value_output_orthogonal_balance(
            model,
            lambda_scale=args.lambda_scale,
        )
    vo_logits = logits_for_prompts(model, tokenizer, PROMPTS, args.device, args.max_length)

    mlp_records = apply_swiglu_mlp_norm_balance(model, min_scale=args.min_sv, max_scale=args.max_sv)
    combined_logits = logits_for_prompts(model, tokenizer, PROMPTS, args.device, args.max_length)

    metrics = {
        "model_path": args.model_path,
        "dtype": args.dtype,
        "vo_canon": args.vo_canon,
        "prompts": PROMPTS,
        "vo_vs_original": compare_logits(base_logits, vo_logits),
        "combined_vs_original": compare_logits(base_logits, combined_logits),
        "vo_summary": summarize_records(vo_records),
        "mlp_summary": summarize_records(mlp_records),
    }

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (out_dir / "vo_records.jsonl").write_text("\n".join(json.dumps(r.to_dict()) for r in vo_records) + "\n")
    (out_dir / "mlp_records.jsonl").write_text("\n".join(json.dumps(r.to_dict()) for r in mlp_records) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
