# Canon Experiment Plan

## Stage 0: Environment, Architecture, and Exactness Smoke Tests

Status: complete for the Qwen3-0.6B preliminary target.

Goal: establish that the local Qwen3-0.6B checkpoint can be loaded, transformed, and compared without numerical or architectural mistakes.

Planned runs:

1. Inspect local Qwen3-0.6B architecture and write a JSON summary under `results/architecture/`.
2. Run fp32 logit-drift tests on fixed calibration prompts for:
   - original model;
   - V/O GQA norm-balanced model;
   - V/O + SwiGLU MLP norm-balanced model.
3. Record transform geometry:
   - per-layer/per-KV-head condition numbers for `A = VV^T` and `B = sum_h O_h^T O_h`;
   - chosen `P` singular range;
   - post-transform balance residual.
4. Submit the same static smoke test through Slurm on partition `pli-c` to confirm cluster reproducibility.

Success criteria:

- model loads in the `rllm` conda env;
- fp32 logit MSE is near numerical noise for exact transforms;
- max logit drift is small enough to proceed to bf16 and SFT smoke tests;
- transformation metadata is saved and inspectable.

## Stage 1: Cheap LoRA Training Signal

Status: first all-linear smoke and V/O-only low-rank stress test completed for seed 1.

Goal: compare training curves for original vs canonicalized representatives under an intentionally small budget.

Initial task:

- Dataset: cached `openai/gsm8k`, because it is locally available and small enough for fast SFT smoke tests.
- Model: local Qwen3-0.6B.
- Adaptation: PEFT LoRA, target modules `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`.
- Canonicalizers: original, V/O norm, MLP norm, combined.
- Metrics: train loss, validation loss on a held-out GSM8K subset, runtime, and any instability.

Initial Slurm shape:

- partition `pli-c`;
- 1 GPU per job;
- 4-job array for the four canonicalizers;
- conservative max steps first, then expand rank/steps if jobs are stable.

Success criteria:

- all variants train without divergence;
- logs and JSON metrics are produced;
- validation loss curves are sufficiently stable to justify a rank/LR sweep.

Current observations:

- all-linear rank-8 LoRA is mostly neutral; deltas are around `1e-3` eval loss or smaller;
- V/O-only rank-2 and rank-4 LoRA show a directional benefit from static V/O norm balancing;
- rank-4 V/O-only LoRA replicated across four seeds with V/O norm better than original on every seed;
- actgrad V/O replication is queued as Slurm job `8150484`.

## Stage 2: Data-Aware Canonicalization

Status: seed-1 all-linear actgrad had small positive variants, but rank-4 V/O-only actgrad did not beat static V/O norm across seeds.

Goal: implement activation-gradient balancing once weight-only exact transforms are validated.

Planned additions:

- replicate rank-4 V/O-only LoRA over seeds for original vs V/O norm;
- replicate V/O actgrad if static V/O seed replication remains positive;
- avoid raw combined actgrad as a default because it inflates gradient scale;
- if V/O methods remain neutral after seeds, move to a harder math subset rather than overfitting GSM8K smoke.

## Stage 4: Larger Static V/O Validation

Status: queued as Slurm job `8150738`.

Goal: test whether the four-seed static V/O signal survives a less tiny training/eval budget.

Planned run:

- Qwen3-0.6B;
- GSM8K;
- rank-4 LoRA on `v_proj,o_proj`;
- original vs static V/O norm;
- seeds `1,2,3,4`;
- `500` optimizer steps;
- `4096` train examples;
- full GSM8K test split for eval loss.

Success criteria:

- no training instability;
- V/O norm remains better than original in mean eval loss;
- per-seed deltas remain mostly positive rather than collapsing to a single lucky seed.

Current result:

- rank-4 fuller completed with a small mean gain for V/O norm (`0.488826` vs `0.488946`);
- V/O norm won 3/4 seeds, but the effect is much smaller than the 160-step stress test;
- rank-2 fuller completed with a larger mean gain for V/O norm (`0.497378` vs `0.497717`);
- V/O norm lowered first gradient norm on every rank-2 seed;
- rank-1 fuller completed with a small mean gain for V/O norm (`0.506122` vs `0.506362`);
- across ranks 1, 2, and 4, V/O norm wins 3/4 seeds and lowers first gradient norm consistently.

Next stage options:

- implement saved-adapter generation/equivalence evaluation for GSM8K exact match;
- run a harder OpenMathReasoning/Numina-style subset after choosing a manageable slice;
- test a second GQA architecture if access is available.

## Stage 5: Full Fine-Tuning

Status: pilot queued as Slurm job `8186444`.

Rationale: LoRA showed only small, consistent optimization signals. The next priority is to test whether exact canonical representatives matter more when all model weights are updated.

Pilot run:

- Qwen3-0.6B;
- GSM8K;
- full bf16 fine-tuning, all parameters trainable;
- original vs static V/O norm vs combined V/O+MLP;
- seeds `1,2`;
- LR `2e-5`;
- `500` optimizer steps;
- `4096` train examples;
- full GSM8K test split for eval loss;
- gradient checkpointing enabled.

Success criteria:

- no OOM or instability;
- full-training loss is sensitive enough to compare canonicalizers;
- if no meaningful delta appears, move full fine-tuning to a harder/larger math subset rather than continuing LoRA sweeps.

Current result:

- seeds `1,2` completed;
- combined V/O+MLP is best so far (`0.464540` mean eval vs `0.465161` original);
- seeds `1,2,3,4` completed at LR `2e-5`;
- combined is best on mean (`0.464807` vs `0.464975` original) but not robust across seeds;
- LR sensitivity sweep completed:
  - `1e-5`: combined worse than original;
  - `2e-5`: combined slightly better on mean, not seed-robust;
  - `5e-5`: combined slightly better on two seeds;
- GSM8K loss is too flat for the next diagnostic, so MATH full fine-tuning is queued as Slurm job `8189381`.

## Stage 6: Harder MATH Full Fine-Tuning

Status: pilot queued as Slurm job `8189381`.

Goal: test full-training canonicalization on a harder cached math task after GSM8K showed only tiny canonicalizer deltas.

Pilot run:

- train on cached `EleutherAI/hendrycks_math` train categories;
- evaluate on cached `HuggingFaceH4/MATH-500`;
- Qwen3-0.6B;
- full bf16 fine-tuning;
- original vs combined V/O+MLP;
- seeds `1,2`;
- LR `2e-5`;
- `1000` optimizer steps;
- max length `1024`;
- answer-token budget reservation enabled to avoid all-masked-label examples.

Current result:

- seeds `1,2` completed;
- combined is slightly better on both seeds (`0.686419` mean eval vs `0.686561` original);
- seeds `1,2,3,4` completed;
- combined wins all four seeds (`0.685697` mean eval vs `0.686109` original);
- longer 2000-step MATH check completed;
- combined trains lower and lowers first gradient norm, but is worse on 2000-step MATH-500 eval (`0.680151` vs `0.679716` original).

Next recommended direction:

- keep full training as the main path;
- add trajectory/checkpoint evaluation rather than only final loss;
- add generation/exact-match evaluation for selected full fine-tuned checkpoints;
- avoid returning to broad LoRA sweeps unless a new hypothesis specifically requires low-rank constraints.

## Stage 7: MATH Generation-Level Evaluation

Status: complete for the first 1000-step original-vs-combined check.

Goal: check whether the small full-training loss differences correspond to answer-level behavior on MATH-500.

Setup:

- rerun 1000-step MATH full fine-tuning with final model saving;
- original vs combined V/O+MLP;
- seeds `1,2`;
- LR `2e-5`;
- deterministic generation on MATH-500;
- answer extraction via last `\boxed{...}` when available, with lightweight LaTeX normalization;
- record exact normalized answer match, boxed-answer rate, and whether the gold normalized answer appears in the generation.

Reasoning:

- existing full-training jobs did not save final model directories;
- the 1000-step MATH loss signal was the cleanest current result;
- the 2000-step result suggests the effect may be transient, so a first generation pass should evaluate the exact 1000-step setting before spending more compute on longer saved checkpoints.

Success criteria:

- all four saved final models are produced;
- full MATH-500 generation metrics are written;
- exact-match and boxed-rate differences are interpretable enough to decide whether to scale generation evaluation to 2000-step checkpoints or trajectory checkpoints.

Current result:

- saved-model rerun completed as Slurm job `8247464`;
- generation evaluation completed as Slurm job `8247465`;
- original mean normalized exact match: `0.193`;
- combined mean normalized exact match: `0.207`;
- combined delta: `+0.014`;
- original mean boxed rate: `0.662`;
- combined mean boxed rate: `0.674`.

Next:

- re-score saved predictions with a stronger symbolic/equivalence verifier;
- then decide between 2000-step saved checkpoint generation and trajectory checkpoint generation.

## Stage 3: Broader Model and Dataset Choice

Status: pending initial signal.

Candidate models:

- Qwen3-0.6B-Base/local Qwen3-0.6B: primary;
- Llama-3.2-1B: useful transfer target if local access is valid;
- OLMo-1B or continuation twins: later hard-to-adapt evidence.

Candidate datasets:

- GSM8K: preliminary smoke task;
- OpenMathReasoning or NuminaMath subset: main math SFT task after infrastructure works;
- MBPP/MBPP+ or small code instruction data: second domain after math signal.
