# Canon Generation-Level Evaluation Report

As of: 2026-05-14 17:38 EDT

## Setup

Goal: test whether the 1000-step MATH full-finetuning loss signal corresponds to answer-level behavior.

Training rerun:

- Model: local Qwen3-0.6B.
- Train data: cached Hendrycks MATH train categories.
- Eval data: MATH-500.
- Full bf16 fine-tuning.
- Canonicalizers: original and combined V/O+MLP.
- Seeds: `1,2`.
- LR: `2e-5`.
- Max steps: `1000`.
- Final models saved under `results/math_full_finetune_for_generation/`.

Generation evaluation:

- Script: `scripts/math_generation_eval.py`.
- Dataset: full MATH-500, `500` problems.
- Prompt: `Problem: ...\nSolution:`.
- Decoding: deterministic greedy generation.
- Max new tokens: `256`.
- Batch size: `8`.
- Metrics:
  - normalized exact match on extracted answer;
  - gold answer appears anywhere in normalized generation;
  - boxed-answer extraction rate.

The evaluator extracts the last `\boxed{...}` answer when present, otherwise falls back to the last dollar-math span or final line. Normalization is lightweight LaTeX string normalization, not symbolic equivalence.

## Saved-Model Training Check

| Run | Train loss | MATH-500 eval loss | First grad norm |
| --- | ---: | ---: | ---: |
| original seed 1 | `0.721660` | `0.686522` | `12.0000` |
| original seed 2 | `0.722716` | `0.687182` | `13.3125` |
| combined seed 1 | `0.721322` | `0.686805` | `10.5625` |
| combined seed 2 | `0.722247` | `0.686027` | `11.5625` |

Means:

| Canonicalizer | Mean train loss | Mean eval loss | Mean first grad norm |
| --- | ---: | ---: | ---: |
| original | `0.722188` | `0.686852` | `12.6563` |
| combined | `0.721784` | `0.686416` | `11.0625` |

This rerun preserves the earlier pattern: combined has lower training loss, lower first gradient norm, and lower mean MATH-500 eval loss at 1000 steps.

## Generation Results

| Canonicalizer | Seed | Exact match | Gold in generation | Boxed rate |
| --- | ---: | ---: | ---: | ---: |
| original | 1 | `0.190` | `0.384` | `0.640` |
| original | 2 | `0.196` | `0.370` | `0.684` |
| combined | 1 | `0.212` | `0.392` | `0.646` |
| combined | 2 | `0.202` | `0.382` | `0.702` |

Means:

| Canonicalizer | Exact match | Gold in generation | Boxed rate |
| --- | ---: | ---: | ---: |
| original | `0.193` | `0.377` | `0.662` |
| combined | `0.207` | `0.387` | `0.674` |

Combined improves normalized exact match by `+0.014` absolute over the two-seed mean.

Per-example exact-match comparison:

| Seed | Combined-only correct | Original-only correct | Both correct | Neither correct |
| ---: | ---: | ---: | ---: | ---: |
| 1 | `24` | `13` | `82` | `381` |
| 2 | `24` | `21` | `77` | `378` |

The generation gain is positive on both seeds, but seed 1 carries most of the margin.

## Breakdown

Exact-match deltas by subject, combined minus original:

| Subject | Delta |
| --- | ---: |
| Algebra | `+0.0081` |
| Counting & Probability | `+0.0395` |
| Geometry | `+0.0244` |
| Intermediate Algebra | `+0.0052` |
| Number Theory | `+0.0242` |
| Prealgebra | `+0.0183` |
| Precalculus | `+0.0000` |

Exact-match deltas by level:

| Level | Delta |
| ---: | ---: |
| 1 | `+0.0465` |
| 2 | `+0.0056` |
| 3 | `+0.0048` |
| 4 | `+0.0078` |
| 5 | `+0.0224` |

The subject/level breakdown is directionally nonnegative everywhere, but this should be interpreted cautiously because the evaluator is string-based and the sample is only MATH-500 x two seeds.

## Interpretation

This is the first answer-level evidence supporting the canonicalization direction. The effect is still modest, but it is more meaningful than a small loss delta because it appears in deterministic generation:

- combined exact match: `20.7%`;
- original exact match: `19.3%`;
- absolute gain: `+1.4` points.

The boxed rate also increases slightly, which suggests combined may be marginally better at following the supervised solution format.

This does not yet prove a robust quality improvement. The grader is not symbolic, exact-match normalization is incomplete, and only two seeds were evaluated. But the result justifies a more rigorous generation-evaluation branch rather than dropping the hypothesis after the 2000-step eval-loss washout.

## Recommended Next Step

Run one of these next:

1. Add a symbolic/equivalence-based MATH grader and re-score the saved predictions.
2. Save and evaluate 2000-step original/combined checkpoints to test whether the generation gain persists when the loss gain disappears.
3. Add trajectory checkpoints at 500, 1000, 1500, and 2000 steps and run generation on a fixed MATH-500 subset first.

The most efficient next step is re-scoring existing predictions with a stronger answer verifier, because it avoids another training run.
