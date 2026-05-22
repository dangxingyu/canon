from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, set_seed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from third_party.ifeval import evaluation_lib  # noqa: E402


def make_prompt(tokenizer, prompt_text: str) -> str:
    """Apply chat template if available; else fall back to ``prompt_text + '\\n'``."""
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_text}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass
    return prompt_text


@torch.inference_mode()
def generate_batch(model, tokenizer, prompts, max_input_tokens: int, max_new_tokens: int):
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_tokens,
    )
    encoded = {k: v.to(model.device) for k, v in encoded.items()}
    generated = model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    prompt_width = encoded["input_ids"].shape[1]
    return tokenizer.batch_decode(generated[:, prompt_width:], skip_special_tokens=True)


def score(inputs, prompt_to_response):
    strict = [evaluation_lib.test_instruction_following_strict(inp, prompt_to_response) for inp in inputs]
    loose = [evaluation_lib.test_instruction_following_loose(inp, prompt_to_response) for inp in inputs]
    return strict, loose


def summarize(outputs):
    prompt_total = 0
    prompt_correct = 0
    instr_total = 0
    instr_correct = 0
    tier0_total = defaultdict(int)
    tier0_correct = defaultdict(int)
    tier1_total = defaultdict(int)
    tier1_correct = defaultdict(int)
    for ex in outputs:
        prompt_total += 1
        if all(ex.follow_instruction_list):
            prompt_correct += 1
        instr_total += len(ex.instruction_id_list)
        instr_correct += sum(ex.follow_instruction_list)
        for iid, ok in zip(ex.instruction_id_list, ex.follow_instruction_list):
            head = iid.split(":")[0]
            tier0_total[head] += 1
            tier0_correct[head] += int(ok)
            tier1_total[iid] += 1
            tier1_correct[iid] += int(ok)
    return {
        "prompt_total": prompt_total,
        "prompt_correct": prompt_correct,
        "prompt_acc": prompt_correct / prompt_total if prompt_total else 0.0,
        "instr_total": instr_total,
        "instr_correct": instr_correct,
        "instr_acc": instr_correct / instr_total if instr_total else 0.0,
        "tier0": {k: tier0_correct[k] / tier0_total[k] for k in sorted(tier0_total)},
        "tier1": {k: tier1_correct[k] / tier1_total[k] for k in sorted(tier1_total)},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dataset", default="google/IFEval")
    parser.add_argument("--split", default="train")  # IFEval ships its test set under "train"
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-input-tokens", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--torch-dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.torch_dtype]

    # Tokenizer / processor for chat template; fall back to plain tokenizer.
    try:
        proc = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
        tokenizer = getattr(proc, "tokenizer", proc)
    except (OSError, ValueError):
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_path, dtype=dtype, trust_remote_code=True)
    model.config.use_cache = True
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    ds = load_dataset(args.dataset, split=args.split)
    if args.max_samples > 0:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    # Build InputExample list compatible with evaluation_lib. The HF dataset
    # `google/IFEval` packs every possible instruction kwarg into each row with
    # None values for fields that don't apply; the IFEval scorer rejects unknown
    # kwargs, so strip None entries per instruction.
    inputs = []
    raw_prompts = []
    for row in ds:
        clean_kwargs = [
            {k: v for k, v in (kw or {}).items() if v is not None}
            for kw in row["kwargs"]
        ]
        inp = evaluation_lib.InputExample(
            key=row["key"],
            instruction_id_list=row["instruction_id_list"],
            prompt=row["prompt"],
            kwargs=clean_kwargs,
        )
        inputs.append(inp)
        raw_prompts.append(row["prompt"])

    # Generate
    formatted = [make_prompt(tokenizer, p) for p in raw_prompts]
    responses: list[str] = []
    for start in range(0, len(formatted), args.batch_size):
        batch_prompts = formatted[start : start + args.batch_size]
        gens = generate_batch(model, tokenizer, batch_prompts, args.max_input_tokens, args.max_new_tokens)
        responses.extend(gens)

    prompt_to_response = {raw: resp for raw, resp in zip(raw_prompts, responses)}
    strict_out, loose_out = score(inputs, prompt_to_response)

    metrics = {
        "model_path": args.model_path,
        "dataset": args.dataset,
        "num_samples": len(inputs),
        "max_new_tokens": args.max_new_tokens,
        "torch_dtype": args.torch_dtype,
        "seed": args.seed,
        "strict": summarize(strict_out),
        "loose": summarize(loose_out),
    }

    (out_dir / "ifeval_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    with (out_dir / "predictions.jsonl").open("w") as handle:
        for inp, resp, s_out, l_out in zip(inputs, responses, strict_out, loose_out):
            handle.write(json.dumps({
                "key": inp.key,
                "instruction_id_list": inp.instruction_id_list,
                "prompt": inp.prompt,
                "kwargs": inp.kwargs,
                "response": resp,
                "strict_follow_all": s_out.follow_all_instructions,
                "strict_follow_list": s_out.follow_instruction_list,
                "loose_follow_all": l_out.follow_all_instructions,
                "loose_follow_list": l_out.follow_instruction_list,
            }) + "\n")
    print(json.dumps({k: v for k, v in metrics.items() if k not in ("tier0", "tier1")}, indent=2))
    print(f"prompt-strict: {metrics['strict']['prompt_acc']:.4f}")
    print(f"instr-strict:  {metrics['strict']['instr_acc']:.4f}")
    print(f"prompt-loose:  {metrics['loose']['prompt_acc']:.4f}")
    print(f"instr-loose:   {metrics['loose']['instr_acc']:.4f}")


if __name__ == "__main__":
    main()
