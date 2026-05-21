# Canon Progress Report

As of: 2026-05-14 17:02 EDT

## Executive Summary

The project is set up and running end-to-end on the `pli-c` Slurm partition. We implemented exact function-preserving canonicalizers for the local Qwen3-0.6B checkpoint, validated their numerical exactness, ran LoRA smoke tests, then pivoted to full fine-tuning after LoRA gains proved small. The current result is scientifically useful but not yet a strong benchmark win:

- The canonicalizers are implemented correctly enough for fp32 exactness tests: V/O and combined transforms preserve logits to numerical precision.
- LoRA shows a small, repeatable signal only in constrained V/O-only low-rank settings. Broad all-linear LoRA is mostly neutral.
- Full fine-tuning is the right priority. It reaches substantially lower losses than LoRA and exposes a consistent optimizer-scale effect.
- Combined V/O+MLP canonicalization lowers first gradient norms in full fine-tuning and gives a small 1000-step MATH-500 loss improvement across four seeds.
- That improvement does not persist in the 2000-step MATH check, where combined has lower train loss but worse eval loss.

Current conclusion: canonicalization is affecting optimization geometry, especially early training scale, but the current static transforms do not yet give a robust final-quality improvement. The next stage should measure training trajectories, steps-to-loss-threshold, and generation accuracy rather than adding more final-only loss sweeps.

Current cluster state: no active Slurm jobs.

## Implemented Infrastructure

Core code:

- `canon/model_utils.py`: model inspection and architecture helpers.
- `canon/transforms.py`: exact static canonicalizers and activation-gradient variants.
- `scripts/static_canon_smoke.py`: exactness and drift tests.
- `scripts/gsm8k_lora_smoke.py`: GSM8K LoRA runner.
- `scripts/gsm8k_full_finetune.py`: GSM8K full fine-tuning runner.
- `scripts/math_full_finetune.py`: Hendrycks MATH full fine-tuning runner with MATH-500 evaluation.
- `scripts/summarize_runs.py`: metrics table helper.

Tracking documents:

- `thoughts.md`: research reasoning and interpretation.
- `experiment-plan.md`: staged plan and next directions.
- `experiment-log.md`: reverse-chronological experiment log.

Slurm coverage:

- Static smoke tests.
- GSM8K LoRA sweeps.
- GSM8K full fine-tuning pilot, seed extension, and LR sweep.
- MATH full fine-tuning pilot, seed extension, and 2000-step check.

There are currently 113 saved `metrics.json` files under `results/`.

## Model and Canonicalization Setup

Primary checkpoint:

- Path: `/scratch/gpfs/ARORA/xd7812/models/Qwen3-0.6B`
- Architecture: `Qwen3ForCausalLM`
- Layers: 28
- Hidden size: 1024
- MLP intermediate size: 3072
- Attention heads: 16
- KV heads: 8
- Head dim: 128
- Query heads per KV head: 2
- Tied embeddings: yes

Implemented exact gauges:

- GQA V/O gauge: per-KV-head `128 x 128` transform applied to the corresponding `v_proj` rows and inverse-applied to the two associated `o_proj` query-head column blocks.
- SwiGLU MLP gauge: positive hidden-channel scaling on `up_proj` rows and inverse scaling on `down_proj` columns.

The local Qwen3 shape is a good starting point because the GQA grouping is explicit and the checkpoint is already available locally.

## Exactness Results

The fp32 static exactness test validates the implementation.

Representative fp32 metrics:

| Transform | Valid-logit MSE | Max valid-logit drift | Last-token KL | Top-1 agreement |
| --- | ---: | ---: | ---: | ---: |
| V/O norm vs original | `2.17e-11` | `6.58e-5` | `2.82e-9` | `1.0` |
| Combined V/O+MLP vs original | `2.27e-11` | `5.44e-5` | `3.05e-8` | `1.0` |

V/O geometry improved as intended:

- Mean balance residual before: `0.3227`
- Mean balance residual after: `7.61e-15`
- Transform singular values: `0.473` to `2.091`
- Clipped groups: `0`

Interpretation: the Qwen3 indexing and exact transform algebra are correct. Any training differences are not caused by large functional drift in fp32.

## LoRA Experiments

The first all-linear LoRA smoke on GSM8K was mostly neutral.

All-linear rank-8 LoRA, GSM8K, 80 steps, seed 1:

| Canonicalizer | Eval loss |
| --- | ---: |
| original | `0.531589` |
| V/O norm | `0.531242` |
| MLP norm | `0.531685` |
| combined | `0.530640` |

This was not strong enough to justify broad LoRA claims.

The better LoRA signal came from deliberately constrained V/O-only adapters.

V/O-only rank-4 LoRA, GSM8K, 160 steps, seeds 1-4:

| Canonicalizer | Mean eval loss | Wins vs original |
| --- | ---: | ---: |
| original | `0.519497` | n/a |
| V/O norm | `0.518256` | `4/4` |
| combined | `0.518396` | `4/4` |

Fuller V/O-only LoRA validation, GSM8K, 500 steps, seeds 1-4:

| Rank | Original mean eval | V/O norm mean eval | Mean delta | Wins |
| ---: | ---: | ---: | ---: | ---: |
| 1 | `0.506362` | `0.506122` | `-0.000240` | `3/4` |
| 2 | `0.497717` | `0.497378` | `-0.000339` | `3/4` |
| 4 | `0.488946` | `0.488826` | `-0.000121` | `3/4` |

Interpretation: LoRA results support a narrow hypothesis that canonical basis matters when adaptation is low-rank and module-constrained. The effect is small, so LoRA is not the main evidence path right now.

Activation-gradient V/O did not beat static V/O in the rank-4 V/O-only setting:

- Original mean eval: `0.519497`
- Static V/O norm mean eval: `0.518256`
- Actgrad V/O mean eval: `0.519600`

Interpretation: the current activation-gradient estimator is not yet better than the static weight-geometry transform. It may need more calibration samples, better covariance regularization, or a different scale normalization.

## Full Fine-Tuning Experiments

The user requested focusing on full training if LoRA did not show significant gains. We pivoted accordingly.

### GSM8K Full Fine-Tuning

Setup:

- Qwen3-0.6B
- Full bf16 fine-tuning
- All `596,049,920` parameters trainable
- GSM8K
- 500 optimizer steps
- 4096 train examples
- Full GSM8K test eval
- Gradient checkpointing enabled

LR `2e-5`, seeds 1-4:

| Canonicalizer | Mean eval loss | Mean first grad norm | Wins vs original |
| --- | ---: | ---: | ---: |
| original | `0.464975` | `23.8438` | n/a |
| V/O norm | `0.465022` | `23.7500` | `1/4` |
| combined | `0.464807` | `21.2188` | `2/4` |

LR sensitivity:

| LR | Original eval | Combined eval | Combined delta |
| ---: | ---: | ---: | ---: |
| `1e-5` | `0.494189` | `0.494957` | `+0.000768` |
| `2e-5` | `0.464975` | `0.464807` | `-0.000169` |
| `5e-5` | `0.468495` | `0.468221` | `-0.000274` |

Interpretation: GSM8K full training is too flat to be the primary diagnostic. Combined canonicalization often lowers gradient scale, but eval loss gains are small and not seed-robust.

### MATH Full Fine-Tuning

Setup:

- Train: cached `EleutherAI/hendrycks_math` categories.
- Eval: cached `HuggingFaceH4/MATH-500`.
- Qwen3-0.6B.
- Full bf16 fine-tuning.
- Original vs combined V/O+MLP.
- LR `2e-5`.
- Max length `1024`.
- Tokenization reserves answer-token budget to avoid all-masked labels.

1000 steps, seeds 1-4:

| Canonicalizer | Mean train loss | Mean eval loss | Mean first grad norm | Wins vs original |
| --- | ---: | ---: | ---: | ---: |
| original | `0.722940` | `0.686109` | `12.2500` | n/a |
| combined | `0.722550` | `0.685697` | `10.5547` | `4/4` |

Per-seed combined eval deltas:

- Seed 1: `-0.000077`
- Seed 2: `-0.000207`
- Seed 3: `-0.000505`
- Seed 4: `-0.000859`

2000 steps, seeds 1-2:

| Canonicalizer | Mean train loss | Mean eval loss | Mean first grad norm | Wins vs original |
| --- | ---: | ---: | ---: | ---: |
| original | `0.669999` | `0.679716` | `13.5000` | n/a |
| combined | `0.669222` | `0.680151` | `11.6250` | `0/2` |

Interpretation:

- The 1000-step MATH result is the cleanest current full-training signal: combined wins all four seeds and lowers first gradient norm.
- The 2000-step check weakens the final-quality claim: combined trains lower but evaluates worse.
- The likely effect is early optimization/preconditioning, not a durable final eval-loss improvement under this setup.

## Current Research Interpretation

What looks solid:

- Exact canonicalization for Qwen3-0.6B is implemented and validated.
- V/O norm balancing has a repeatable low-rank optimization signal in constrained LoRA.
- Combined V/O+MLP canonicalization lowers early gradient scale in full fine-tuning.
- Harder MATH training is more informative than GSM8K for this project.

What is not yet established:

- A robust final-loss improvement from canonicalization.
- A generation/exact-match improvement.
- A data-aware activation-gradient canonicalizer that beats static V/O norm.
- Transfer to a second model architecture.

Most defensible current claim:

> Exact function-preserving canonicalization changes optimizer geometry in measurable ways. Static V/O balancing improves constrained V/O-only LoRA slightly, and combined V/O+MLP balancing lowers gradient scale and improves short-run MATH full-finetuning loss, but the current transforms do not yet yield a robust long-run eval-loss gain.

## Recommended Next Experiments

Priority 0: strengthen generation evaluation.

- First generation-level check is complete.
- Saved 1000-step MATH full-finetuned original/combined models for seeds `1,2`.
- Combined normalized exact match on MATH-500: `20.7%`.
- Original normalized exact match on MATH-500: `19.3%`.
- The grader is lightweight string normalization, so the immediate next step is symbolic/equivalence re-scoring of saved predictions.

Priority 1: trajectory analysis for full fine-tuning.

- Add step-wise eval to `math_full_finetune.py`.
- Compare original vs combined at 250-step or 500-step intervals.
- Report loss curves, steps-to-threshold, and whether the 1000-step advantage is transient.

Priority 2: generation-level evaluation.

- Save selected full fine-tuned checkpoints.
- Run deterministic generation on MATH-500.
- Track exact-match or normalized answer match, even if initially approximate.
- This is necessary because small eval-loss deltas may not reflect answer quality.

Priority 3: improve data-aware canonicalization only after trajectory results.

- Revisit activation-gradient V/O with more calibration samples.
- Add better covariance shrinkage and determinant/trace normalization.
- Do not expand the transform zoo until the evaluation metric is sharper.

Priority 4: second architecture after the Qwen3 pipeline is stable.

- Use a second GQA model only once trajectory and generation evaluation are automated.
- The canonical transform must be architecture-specific because GQA grouping and output projection layouts differ.

## Immediate Plan

The next concrete step should be a MATH trajectory experiment:

- Original vs combined.
- Seeds 1-2 first.
- LR `2e-5`.
- 1000 or 2000 max steps.
- Eval every 250 or 500 steps.
- Save final model for generation evaluation.

This directly tests the current hypothesis that canonicalization helps early optimization but may wash out by longer training.
