# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Research code for **function-preserving checkpoint canonicalization** to make LLM finetuning more basis-robust. The primary target is the local Qwen3-0.6B checkpoint at `/scratch/gpfs/ARORA/xd7812/models/Qwen3-0.6B`. The full proposal is in `canonicalization_project_proposal.pdf` / `.tex`.

A "canonicalization" here is a function-preserving gauge transform applied to weights. Two are implemented:

- **GQA value/output (V/O) balance** — for each KV head `j`, solve for an SPD `P_j ∈ R^{head_dim × head_dim}` that balances `A = V_j V_j^T` against `B = Σ_h O_h^T O_h` (sum over the `heads_per_kv` query heads sharing this KV head). Then `V_j ← P_j V_j` and each `O_h ← O_h P_j^{-1}`. Exactly preserves attention output up to numerical error.
- **SwiGLU MLP channel scaling** — positive diagonal `s` on the hidden dimension: `up_proj ← diag(s) · up_proj`, `down_proj ← down_proj · diag(s)^{-1}`. `gate_proj` is untouched. Exact gauge for SiLU·linear gating.

Both have *weight-only* (`_norm_balance`) and *activation-gradient* (`_covariance_balance`, `_activation_gradient_balance`) variants. Weight-only needs no data; activation-gradient hooks `o_proj`/`down_proj` inputs over a calibration batch and uses input second moments vs. gradient second moments.

The SPD balancer (`solve_spd_balancer` in `canon/transforms.py`) computes `P` from the closed form `X = L^{-1/2} (L^{1/2} R L^{1/2})^{1/2} L^{-1/2}`, `P = X^{1/2}`, then spectrum-clips to `[min_sv, max_sv]`. Always run in fp64 on CPU, then cast back to the model's dtype/device.

## Layout

- `canon/` — the importable package.
  - `model_utils.py` — `summarize_config(model_path)` and `get_decoder_layers(model)` (handles `model.model.layers` and `model.transformer.h`).
  - `transforms.py` — all four canonicalizers + `solve_spd_balancer` + `BalanceRecord`/`summarize_records`.
- `scripts/` — one entry point per smoke/experiment. Each inserts the repo root onto `sys.path` so `from canon.* import …` works when run as a plain script.
  - `inspect_architecture.py` — dump config to `results/architecture/`.
  - `static_canon_smoke.py` — fp32/bf16 logit-drift check after weight-only canonicalization.
  - `gsm8k_lora_smoke.py` — LoRA training comparison across canonicalizers. `--canonicalizer ∈ {original, vo_norm, mlp_norm, combined}` and `--target-modules ∈ {all_linear, attention, vo, v_only, mlp}` (see `TARGET_MODULES` dict).
  - `activation_gradient_smoke.py` — data-aware variant, collects `o_proj`/`down_proj` input + grad second moments via forward-pre hooks with `retain_grad()`.
- `slurm/` — sbatch wrappers on partition `pli-c`. Each `cd`s into the repo, sources conda, activates `rllm`, sets `PYTHONPATH=$PWD`, then runs a script. Array jobs index a bash array of canonicalizers/ranks.
- `results/` — per-experiment outputs: `metrics.json` (one summary blob) plus `*_records.jsonl` (per-layer `BalanceRecord`s).
- `logs/` — Slurm stdout/stderr (`%x-%j.out`, `%x-%A_%a.out`).
- `experiment-plan.md`, `experiment-log.md`, `thoughts.md` — append-only research journal; each entry is timestamped (`YYYY-MM-DD HH:MM EDT`). Read these for stage status before starting new work; append a new entry after each meaningful run rather than rewriting history.

## Environment

The cluster has no `uv`/`venv` workflow set up yet for this repo, and every slurm script activates **conda env `rllm`** (Python 3.12, torch 2.7.1+cu126, transformers 4.55.3, datasets 4.0.0, peft 0.17.1, accelerate 1.10.0). When running locally, activate the same env:

```bash
source /usr/licensed/anaconda3/2025.6/etc/profile.d/conda.sh && conda activate rllm
export PYTHONPATH=/scratch/gpfs/ARORA/xd7812/canon:${PYTHONPATH:-}
export HF_HOME=/scratch/gpfs/ARORA/xd7812/.cache/huggingface
```

Note: the parent `/scratch/gpfs/ARORA/xd7812/CLAUDE.md` says prefer uv+venv. Existing slurm scripts predate that rule and use `rllm`. Don't silently rewrite them — if you migrate, do it as an explicit task. `bitsandbytes` and `trl` are not installed in `rllm`, so QLoRA isn't available.

## Common commands

Two-step LoRA debug (validates env + data path quickly, ~3s):

```bash
python scripts/gsm8k_lora_smoke.py \
  --out-dir results/gsm8k_lora_smoke/local_debug_original \
  --canonicalizer original --rank 4 --max-steps 2 \
  --train-samples 8 --eval-samples 4 --max-length 128
```

fp32 static logit-drift smoke (should give MSE near 1e-10, top-1 agreement 1.0):

```bash
python scripts/static_canon_smoke.py \
  --out-dir results/static_smoke/qwen3_0_6b_fp32 --dtype float32
```

Submit a Slurm job / array:

```bash
sbatch slurm/static_qwen_smoke.sbatch
sbatch slurm/gsm8k_lora_smoke.sbatch        # 4-task array over canonicalizers
sbatch slurm/gsm8k_vo_lowrank.sbatch        # 6-task array: {original,vo_norm,combined} × {rank 2, rank 4}
sbatch slurm/activation_gradient_smoke.sbatch
```

Monitor: `squeue -u $USER`; logs land in `logs/`.

## Working conventions

- **Smoke first, then submit.** Every script has a small-args mode (low `--max-samples`, `--max-steps 2`, `--train-samples 8`). Run that on the login/local GPU before sbatch.
- **fp32 for exactness checks, bf16 for training.** fp32 is the floor: if a transform isn't exact to ~1e-10 logit MSE in fp32, the implementation is wrong. bf16 drift is measured, not zero (bf16 V/O ran ~0.004 MSE — log it, don't gate on it).
- **Restrict logit-drift metrics to valid (non-padded) positions and the true last non-padding token.** A prior bug compared padded positions and made drift look catastrophic; see `compare_logits` in `static_canon_smoke.py:37`.
- **Apply transforms in `torch.no_grad()`, in fp64 on CPU, cast back to weight dtype/device when writing.** The decorators are already there — keep them.
- **GQA indexing is the easy thing to get wrong.** Qwen3-0.6B: 16 Q heads, 8 KV heads, `head_dim=128`, so each KV head is shared by 2 Q heads. `v_proj` is `[1024, 1024]`, `o_proj` is `[1024, 2048]` — `o_proj` columns are indexed by Q-head, not KV-head. The loop in `apply_gqa_value_output_norm_balance` is the reference; mirror it.
- **Don't add `bitsandbytes`/`trl`/`flash-attn` deps without checking — they're not in `rllm`.**
