# Canon Experiment Log

## 2026-05-14 17:38 EDT - MATH generation-level evaluation completed

- Slurm arrays completed:
  - saved-model full-finetuning job: `8247464`;
  - dependent generation-eval job: `8247465`.
- Current Slurm queue: empty.
- Saved final models under:
  - `results/math_full_finetune_for_generation/qwen3_0_6b/lr2e-5_steps1000/`.
- Generation metrics under:
  - `results/math_generation_eval/qwen3_0_6b/lr2e-5_steps1000_math500/`.
- Wrote focused report:
  - `generation-eval-report.md`.
- Saved-model rerun MATH-500 eval means:
  - original: `0.686852`;
  - combined: `0.686416`;
  - combined delta: `-0.000436`.
- Generation exact-match means:
  - original: `0.193`;
  - combined: `0.207`;
  - combined delta: `+0.014`.
- Other generation metrics:
  - gold-in-generation: original `0.377`, combined `0.387`;
  - boxed rate: original `0.662`, combined `0.674`.
- Per-seed exact-match comparison:
  - seed `1`: combined-only correct `24`, original-only correct `13`;
  - seed `2`: combined-only correct `24`, original-only correct `21`.
- Interpretation:
  - this is the first answer-level evidence in favor of combined canonicalization;
  - the gain is modest and uses a lightweight string-normalization grader, so it is not yet a final MATH accuracy claim;
  - next best step is to re-score saved predictions with a stronger symbolic/equivalence verifier or save/evaluate 2000-step checkpoints.

## 2026-05-14 17:13 EDT - Submitted MATH generation-level evaluation stage

- Added generation evaluation runner:
  - `scripts/math_generation_eval.py`.
- Added generation metrics summarizer:
  - `scripts/summarize_generation.py`.
- Added Slurm wrappers:
  - `slurm/math_full_ft_save_for_generation.sbatch`;
  - `slurm/math_generation_eval.sbatch`.
- Local evaluator smoke passed on the base Qwen3-0.6B checkpoint with `2` MATH-500 samples.
- Submitted saved-model full-finetuning rerun:
  - original job ID `8247399` was canceled and resubmitted with a tighter walltime for better backfill;
  - intermediate job ID `8247429` was canceled during scheduler startup and replaced by the active 1-hour job;
  - active job ID: `8247464`;
  - partition: `pli-c`;
  - original vs combined;
  - seeds `1,2`;
  - train on cached Hendrycks MATH;
  - eval on MATH-500;
  - LR `2e-5`;
  - max steps `1000`;
  - saves `final_model` for each task.
- Submitted dependent generation-eval array:
  - original job ID `8247400` was canceled and resubmitted with a tighter walltime;
  - intermediate job ID `8247430` was canceled and replaced by the active 1-hour job;
  - active job ID: `8247465`;
  - dependency: `afterok:8247464`;
  - evaluates saved final models on full MATH-500;
  - deterministic generation;
  - max new tokens `256`;
  - batch size `8`;
  - outputs `generation_metrics.json` and `predictions.jsonl`.

## 2026-05-14 17:02 EDT - Progress report produced

- Produced `progress-report.md` summarizing the current state of the project.
- Current saved metrics:
  - `113` total `metrics.json` files under `results/`;
  - `35` full fine-tuning metric files under GSM8K and MATH result directories, including local debugs.
- Current Slurm queue: empty.
- Current interpretation:
  - exact Qwen3 canonicalizers are validated;
  - LoRA gains are too small for the main project direction;
  - full training shows a real optimizer-scale effect;
  - MATH 1000-step combined canonicalization wins all four seeds, but the 2000-step check loses on eval despite lower train loss;
  - next stage should measure full-training trajectories and generation/exact-match, not just final eval loss.

## 2026-05-13 18:39 EDT - 2000-step MATH full-training check completed

- Slurm array `8191472` completed all four 2000-step MATH tasks on `pli-c`.
- Setup:
  - train on cached Hendrycks MATH categories;
  - eval on MATH-500;
  - original vs combined;
  - seeds `1,2`;
  - LR `2e-5`;
  - max steps `2000`;
  - max length `1024`.
- Results:
  - seed `1`: original eval `0.679683`, combined eval `0.679712`;
  - seed `2`: original eval `0.679750`, combined eval `0.680590`.
- Two-seed means:
  - original eval: `0.679716`;
  - combined eval: `0.680151`;
  - original train loss: `0.669999`;
  - combined train loss: `0.669222`;
  - original first grad: `13.5000`;
  - combined first grad: `11.6250`.
- Interpretation:
  - combined trains slightly lower and has lower first gradient norm;
  - combined does not improve 2000-step MATH-500 eval loss;
  - the 1000-step benefit appears to be early optimization/preconditioning, not a durable final-loss gain in this setup.
- Current Slurm queue: empty.

## 2026-05-13 18:10 EDT - MATH four-seed result completed; submitted 2000-step MATH check

- Slurm array `8190711` completed all MATH seed-extension tasks on `pli-c`.
- Combined with job `8189381`, MATH full training now has seeds `1,2,3,4`.
- Four-seed MATH-500 eval means:
  - original: `0.686109`;
  - combined: `0.685697`.
- Per-seed combined minus original eval deltas:
  - seed `1`: `-0.000077`;
  - seed `2`: `-0.000207`;
  - seed `3`: `-0.000505`;
  - seed `4`: `-0.000859`.
- First gradient norm means:
  - original: `12.2500`;
  - combined: `10.5547`.
- Interpretation:
  - combined wins all four MATH seeds, unlike the GSM8K full-training sweep;
  - the loss gap is still small, but the harder task gives the cleanest full-training signal so far;
  - test longer full training before moving to generation/exact-match.
- Added and submitted longer MATH full-training check:
  - script: `slurm/math_full_ft_longer.sbatch`;
  - job ID: `8191472`;
  - original vs combined;
  - seeds `1,2`;
  - LR `2e-5`;
  - max steps `2000`;
  - train all cached Hendrycks MATH categories;
  - eval on MATH-500.

## 2026-05-13 17:44 EDT - MATH full-training pilot completed; submitted seed extension

- Slurm array `8189381` completed all four MATH full-training pilot tasks on `pli-c`.
- Setup:
  - train on all cached Hendrycks MATH train categories;
  - evaluate on cached MATH-500;
  - Qwen3-0.6B full fine-tuning;
  - original vs combined V/O+MLP;
  - seeds `1,2`;
  - LR `2e-5`;
  - max steps `1000`;
  - max length `1024`.
- Results:
  - seed `1`: original eval `0.686645`, combined eval `0.686567`;
  - seed `2`: original eval `0.686477`, combined eval `0.686270`.
- Two-seed means:
  - original eval: `0.686561`;
  - combined eval: `0.686419`;
  - original first grad: `13.0625`;
  - combined first grad: `11.0625`.
- Interpretation:
  - combined is slightly better on both MATH pilot seeds, but the absolute loss gap remains small;
  - first gradient norm is again lower for the canonicalized representative;
  - extend to seeds `3,4` before changing task or adding generation evaluation.
- Added and submitted MATH seed extension:
  - script: `slurm/math_full_ft_seed_sweep.sbatch`;
  - job ID: `8190711`;
  - seeds `3,4`;
  - original vs combined;
  - same MATH full-training hyperparameters.

## 2026-05-13 17:26 EDT - GSM8K full-training LR sweep completed; submitted MATH full-training pilot

- Slurm array `8187989` completed all LR-sweep tasks on `pli-c`.
- GSM8K full-training LR sweep results:
  - LR `1e-5`, original mean eval `0.494189`, combined mean eval `0.494957`;
  - LR `2e-5`, original mean eval `0.464975`, combined mean eval `0.464807`;
  - LR `5e-5`, original mean eval `0.468495`, combined mean eval `0.468221`.
- Interpretation:
  - combined has lower gradient norms but not a robust GSM8K eval-loss advantage;
  - LR `2e-5` remains the best tested GSM8K full-training LR;
  - GSM8K loss appears too flat for the next full-training diagnostic.
- Found cached harder math datasets:
  - `EleutherAI/hendrycks_math`;
  - `HuggingFaceH4/MATH-500`.
- Added MATH full fine-tuning runner:
  - `scripts/math_full_finetune.py`.
- Fixed MATH tokenization to reserve answer tokens after local debug produced `NaN` eval loss from all-masked labels.
- Verification:
  - `python -m py_compile canon/*.py scripts/*.py` passed;
  - `bash -n slurm/math_full_ft_pilot.sbatch` passed;
  - local fixed MATH debug wrote finite train/eval loss: `1.4035` / `1.1772`.
- Submitted MATH full-training pilot:
  - script: `slurm/math_full_ft_pilot.sbatch`;
  - job ID: `8189381`;
  - train: all cached Hendrycks MATH train categories;
  - eval: cached MATH-500;
  - canonicalizers: `original`, `combined`;
  - seeds: `1,2`;
  - LR: `2e-5`;
  - max steps: `1000`;
  - max length: `1024`.

## 2026-05-13 17:04 EDT - Full fine-tuning four-seed result; submitted LR sensitivity sweep

- Slurm array `8187243` completed all six seed-extension tasks on `pli-c`.
- Combined with job `8186444`, LR `2e-5` full fine-tuning now has seeds `1,2,3,4` for:
  - original;
  - V/O norm;
  - combined V/O+MLP.
- Four-seed eval means:
  - original: `0.464975`;
  - V/O norm: `0.465022`;
  - combined: `0.464807`.
- Per-seed deltas vs original:
  - V/O norm: `-0.000767`, `+0.000116`, `+0.000363`, `+0.000477`;
  - combined: `-0.000999`, `-0.000245`, `+0.000076`, `+0.000493`.
- Interpretation:
  - combined is best on mean but not seed-robust;
  - the LR `2e-5` full-training gain is too small to call significant;
  - continue full-training work through LR/task sensitivity, not more LoRA.
- Added and submitted LR sensitivity sweep:
  - script: `slurm/gsm8k_full_ft_lr_sweep.sbatch`;
  - job ID: `8187989`;
  - canonicalizers: `original`, `combined`;
  - LRs: `1e-5`, `5e-5`;
  - seeds: `1,2`;
  - same 500-step GSM8K full fine-tuning setup.

## 2026-05-13 16:52 EDT - Full fine-tuning pilot completed; submitted seed extension

- Slurm array `8186444` completed all six full fine-tuning pilot tasks on `pli-c`.
- Setup:
  - Qwen3-0.6B;
  - all parameters trainable;
  - GSM8K;
  - LR `2e-5`;
  - max steps `500`;
  - train samples `4096`;
  - eval samples full GSM8K test split;
  - seeds `1,2`;
  - canonicalizers `original`, `vo_norm`, `combined`.
- Eval losses:
  - seed `1`: original `0.466372`, V/O norm `0.465604`, combined `0.465373`;
  - seed `2`: original `0.463951`, V/O norm `0.464067`, combined `0.463706`.
- Two-seed means:
  - original eval: `0.465161`;
  - V/O norm eval: `0.464836`;
  - combined eval: `0.464540`.
- Interpretation:
  - full fine-tuning reaches substantially lower loss than LoRA under the same GSM8K framing;
  - combined V/O+MLP is the best current full-training representative and wins both seeds;
  - deltas remain small, so seed extension is needed before LR sweeps.
- Added and submitted seed extension:
  - script: `slurm/gsm8k_full_ft_seed_sweep.sbatch`;
  - job ID: `8187243`;
  - seeds `3,4`;
  - canonicalizers `original`, `vo_norm`, `combined`;
  - same full fine-tuning hyperparameters as pilot.

## 2026-05-13 16:42 EDT - Switched to full fine-tuning and submitted pilot

- User direction: focus on full training first because LoRA gains are not significant.
- Added full fine-tuning runner:
  - `scripts/gsm8k_full_finetune.py`.
- Added Slurm pilot:
  - `slurm/gsm8k_full_ft_pilot.sbatch`.
- Full fine-tuning setup:
  - Qwen3-0.6B;
  - all `596,049,920` parameters trainable;
  - GSM8K;
  - canonicalizers: `original`, `vo_norm`, `combined`;
  - seeds: `1,2`;
  - LR: `2e-5`;
  - scheduler: `constant_with_warmup`;
  - warmup ratio: `0.03`;
  - max steps: `500`;
  - train samples: `4096`;
  - eval samples: full GSM8K test split;
  - gradient accumulation: `8`;
  - bf16 + gradient checkpointing.
- Verification:
  - `python -m py_compile canon/*.py scripts/*.py` passed;
  - `bash -n slurm/gsm8k_full_ft_pilot.sbatch` passed;
  - local 2-step full-training debug completed and wrote `results/gsm8k_full_finetune/local_debug_original/metrics.json`;
  - local debug peak CUDA allocation: `5.61 GB`;
  - local debug train/eval loss: `1.6369` / `1.1263`.
- Submitted Slurm array:
  - job ID: `8186444`;
  - partition: `pli-c`;
  - array: 6 tasks, max concurrency 6.

## 2026-05-12 21:51 EDT - Rank-1 fuller validation completed

- Slurm array `8152081` completed all eight rank-1 fuller tasks on `pli-c`.
- Setup matched rank-2/rank-4 fuller except:
  - rank: `1`;
  - LoRA alpha: `2`.
- Eval losses:
  - seed `1`: original `0.505356`, V/O norm `0.505545`;
  - seed `2`: original `0.505530`, V/O norm `0.505186`;
  - seed `3`: original `0.507684`, V/O norm `0.507111`;
  - seed `4`: original `0.506876`, V/O norm `0.506645`.
- Four-seed means:
  - original eval: `0.506362`;
  - V/O norm eval: `0.506122`;
  - original first grad: `0.4559`;
  - V/O norm first grad: `0.4288`.
- Per-seed V/O minus original eval deltas:
  - seed `1`: `+0.000188`;
  - seed `2`: `-0.000344`;
  - seed `3`: `-0.000574`;
  - seed `4`: `-0.000232`.
- Interpretation:
  - V/O norm wins 3/4 seeds, as in rank 2 and rank 4;
  - the mean gain is smaller than rank 2 but larger than rank 4;
  - V/O norm lowers first gradient norm on every seed;
  - the next stage should move beyond GSM8K loss-only sweeps.
- Current Slurm queue: empty after completion.

## 2026-05-12 21:28 EDT - Rank-2 fuller validation completed; submitted rank-1 stress sweep

- Slurm array `8151319` completed all eight rank-2 fuller tasks on `pli-c`.
- Setup matched rank-4 fuller except:
  - rank: `2`;
  - LoRA alpha: `4`.
- Eval losses:
  - seed `1`: original `0.497212`, V/O norm `0.496567`;
  - seed `2`: original `0.496751`, V/O norm `0.497007`;
  - seed `3`: original `0.498408`, V/O norm `0.497918`;
  - seed `4`: original `0.498497`, V/O norm `0.498021`.
- Four-seed means:
  - original eval: `0.497717`;
  - V/O norm eval: `0.497378`;
  - original first grad: `0.6479`;
  - V/O norm first grad: `0.6054`.
- Per-seed V/O minus original eval deltas:
  - seed `1`: `-0.000646`;
  - seed `2`: `+0.000255`;
  - seed `3`: `-0.000491`;
  - seed `4`: `-0.000476`.
- Interpretation:
  - V/O norm wins 3/4 seeds and has a larger mean gain than rank 4 under the same fuller budget;
  - first gradient norm is lower for V/O norm on every seed;
  - this supports continuing the capacity stress test to rank 1.
- Added and submitted rank-1 fuller validation:
  - script: `slurm/gsm8k_vo_rank1_fuller_sweep.sbatch`;
  - job ID: `8152081`;
  - same setup as rank-2 fuller, except rank `1` and LoRA alpha `2`.

## 2026-05-12 21:04 EDT - Larger rank-4 static V/O validation completed; submitted rank-2 fuller sweep

- Slurm array `8150738` completed all eight tasks on `pli-c`.
- Setup:
  - Qwen3-0.6B;
  - GSM8K;
  - rank `4`;
  - target modules: `v_proj,o_proj`;
  - original vs static `vo_norm`;
  - seeds `1,2,3,4`;
  - max steps `500`;
  - train samples `4096`;
  - eval samples `1319` (full GSM8K test split).
- Eval losses:
  - seed `1`: original `0.488789`, V/O norm `0.488877`;
  - seed `2`: original `0.488001`, V/O norm `0.487927`;
  - seed `3`: original `0.488731`, V/O norm `0.488444`;
  - seed `4`: original `0.490264`, V/O norm `0.490055`.
- Four-seed means:
  - original eval: `0.488946`;
  - V/O norm eval: `0.488826`;
  - original first grad: `0.8878`;
  - V/O norm first grad: `0.8374`.
- Interpretation:
  - V/O norm still wins on mean eval and 3/4 seeds, but the effect is much smaller than in the 160-step stress test;
  - this points toward an early-optimization or low-capacity effect rather than a large asymptotic loss gain.
- Added and submitted rank-2 fuller validation:
  - script: `slurm/gsm8k_vo_rank2_fuller_sweep.sbatch`;
  - job ID: `8151319`;
  - same setup as rank-4 fuller, except rank `2` and LoRA alpha `4`.

## 2026-05-12 20:48 EDT - Actgrad V/O follow-up completed; submitted larger static V/O validation

- Slurm array `8150484` completed all four `actgrad_vo` tasks on `pli-c`.
- Rank-4 V/O-only LoRA eval results compared to existing baselines:
  - seed `1`: original `0.529978`, static V/O `0.528355`, actgrad V/O `0.529646`;
  - seed `2`: original `0.508444`, static V/O `0.507917`, actgrad V/O `0.508391`;
  - seed `3`: original `0.522318`, static V/O `0.520490`, actgrad V/O `0.522823`;
  - seed `4`: original `0.517246`, static V/O `0.516261`, actgrad V/O `0.517542`.
- Four-seed means:
  - original eval: `0.519497`;
  - static V/O norm eval: `0.518256`;
  - actgrad V/O eval: `0.519600`.
- Interpretation:
  - actgrad V/O is worse than static V/O norm on every seed in this restricted setting;
  - actgrad V/O is not robust enough with the current 8-sample proxy covariance;
  - static V/O norm is the best current method to scale.
- Added and submitted larger validation sweep:
  - script: `slurm/gsm8k_vo_rank4_fuller_sweep.sbatch`;
  - job ID: `8150738`;
  - tasks: original vs static `vo_norm` for seeds `1,2,3,4`;
  - rank: `4`;
  - target modules: `v_proj,o_proj`;
  - max steps: `500`;
  - train samples: `4096`;
  - eval samples: full GSM8K test split, `1319`;
  - LR/schedule: `3e-4`, `constant_with_warmup`.

## 2026-05-12 20:37 EDT - Rank-4 V/O seed sweep completed; submitted actgrad V/O follow-up

- Slurm array `8150111` completed all nine tasks on `pli-c`.
- Rank-4 V/O-only LoRA eval results:
  - seed `1`: original `0.529978`, V/O norm `0.528355`, combined `0.528986`;
  - seed `2`: original `0.508444`, V/O norm `0.507917`, combined `0.508071`;
  - seed `3`: original `0.522318`, V/O norm `0.520490`, combined `0.520098`;
  - seed `4`: original `0.517246`, V/O norm `0.516261`, combined `0.516430`.
- Four-seed means:
  - original eval: `0.519497`;
  - V/O norm eval: `0.518256`;
  - combined eval: `0.518396`;
  - original first grad: `1.0113`;
  - V/O norm first grad: `0.9717`;
  - combined first grad: `0.9774`.
- Per-seed deltas vs original:
  - V/O norm: `-0.00162`, `-0.00053`, `-0.00183`, `-0.00098`;
  - combined: `-0.00099`, `-0.00037`, `-0.00222`, `-0.00082`.
- Interpretation:
  - static V/O norm improved eval loss on all four seeds in the restricted V/O-only setting;
  - the mean gain is small but consistent enough to justify a focused next experiment;
  - V/O norm is slightly cleaner than combined on average, matching the module-restricted hypothesis.
- Added and submitted next-stage Slurm follow-up:
  - script: `slurm/gsm8k_vo_rank4_actgrad_sweep.sbatch`;
  - job ID: `8150484`;
  - task: `actgrad_vo` for seeds `1,2,3,4`;
  - same rank/LR/schedule/data budget as the static V/O seed sweep;
  - goal: compare data-aware V/O balancing against the completed original and static V/O baselines.

## 2026-05-12 20:32 EDT - Slurm act-gradient arrays completed

- Slurm jobs completed on `pli-c`:
  - `8149235`: original vs raw `actgrad_combined`;
  - `8149488`: original, `actgrad_vo`, raw `actgrad_mlp`, raw `actgrad_combined`;
  - `8149693`: original, `actgrad_mlp_gmean`, `actgrad_combined_gmean`.
- Shared setup:
  - Qwen3-0.6B;
  - GSM8K;
  - rank `8`;
  - target modules: all linear transformer projections;
  - LR: `2e-4`;
  - scheduler: `constant_with_warmup`;
  - max steps: `160`;
  - train/eval samples: `1024/256`;
  - proxy samples for actgrad: `8`.
- Matched Slurm results:
  - original: train `0.545385`, eval `0.512756`, first grad `1.8274`;
  - `actgrad_vo`: train `0.546163`, eval `0.511304`, first grad `1.7739`, clipped groups `8/224`;
  - raw `actgrad_mlp`: train `0.531017`, eval `0.512499`, first grad `3.0568`, clipped layers `28/28`;
  - raw `actgrad_combined`: train `0.531080`, eval `0.513330`, first grad `3.0655`, clipped records `36/252`;
  - `actgrad_mlp_gmean`: train `0.544601`, eval `0.511365`, first grad `1.8984`, clipped layers `18/28`;
  - `actgrad_combined_gmean`: train `0.545747`, eval `0.511403`, first grad `1.8742`, clipped records `26/252`.
- Interpretation:
  - raw combined actgrad remains unsafe because MLP scaling inflates the gradient scale and worsens eval;
  - V/O actgrad is the cleanest data-aware result in this batch;
  - geometric-mean MLP normalization fixes most of the raw MLP pathology and gives a small eval improvement;
  - the next claim-quality step is seed replication, not adding more actgrad variants.
- Current queue:
  - rank-4 V/O seed replication job `8150111` has started; one task completed and four tasks are running/pending.

## 2026-05-12 20:27 EDT - Submitted rank-4 V/O seed replication sweep

- Added `slurm/gsm8k_vo_rank4_seed_sweep.sbatch`.
- Added `scripts/summarize_runs.py` for quick TSV summaries of result directories.
- Updated `scripts/gsm8k_lora_smoke.py` so future metrics JSON includes top-level:
  - `train_loss`;
  - `eval_loss`;
  - `train_runtime`;
  - `first_grad_norm`.
- Verification:
  - `python -m py_compile canon/*.py scripts/*.py` passed in the `rllm` env;
  - `bash -n slurm/gsm8k_vo_rank4_seed_sweep.sbatch` passed.
- Submitted Slurm job:
  - job ID: `8150111`;
  - partition: `pli-c`;
  - array: 9 tasks;
  - seeds: `2,3,4`;
  - canonicalizers: `original`, `vo_norm`, `combined`;
  - rank: `4`;
  - target modules: `v_proj,o_proj`;
  - LR/schedule/steps: `3e-4`, `constant_with_warmup`, `160`.

## 2026-05-12 20:25 EDT - Slurm low-rank V/O stress test completed

- Slurm array `8149031` completed all six tasks on `pli-c`.
- Setup:
  - Qwen3-0.6B;
  - GSM8K;
  - target modules: `v_proj,o_proj`;
  - LR: `3e-4`;
  - scheduler: `constant_with_warmup`;
  - max steps: `160`;
  - train/eval samples: `1024/256`;
  - seed: `1`.
- Results:
  - rank `2`, original: train `0.621585`, eval `0.537316`, first grad `0.6418`;
  - rank `2`, V/O norm: train `0.620603`, eval `0.536613`, first grad `0.6085`;
  - rank `2`, combined: train `0.620848`, eval `0.536967`, first grad `0.6123`;
  - rank `4`, original: train `0.595661`, eval `0.529978`, first grad `0.8400`;
  - rank `4`, V/O norm: train `0.595100`, eval `0.528355`, first grad `0.8131`;
  - rank `4`, combined: train `0.595009`, eval `0.528986`, first grad `0.8283`.
- Interpretation:
  - static V/O norm balancing is directionally helpful in the restricted V/O-only LoRA setting;
  - the rank-4 eval improvement vs original is `0.00162`, larger than the earlier all-linear rank-8 deltas;
  - combined V/O+MLP norm is not better than V/O alone when LoRA does not target MLP modules;
  - this should be replicated across seeds before making any claim.
- Queue state after completion:
  - activation-gradient static job `8149145` completed successfully;
  - actgrad LoRA comparison `8149235` is running;
  - actgrad ablation `8149488` is partially running/pending;
  - gmean actgrad array `8149693` remains pending with reason `Priority`.

## 2026-05-12 20:52 EDT - Geometric-mean MLP actgrad normalization tested

- Added normalized MLP act-gradient variants:
  - `actgrad_mlp_gmean`;
  - `actgrad_combined_gmean`.
- Added `slurm/gsm8k_actgrad_gmean.sbatch` and submitted it:
  - job ID: `8149693`;
  - state: pending with reason `Priority`.
- Local 160-step all-linear LoRA results now:
  - `original`: train `0.546003`, eval `0.511679`, first grad `1.8420`;
  - `actgrad_vo`: train `0.546562`, eval `0.511837`, first grad `1.8501`;
  - raw `actgrad_mlp`: train `0.531106`, eval `0.513149`, first grad `3.0518`;
  - raw `actgrad_combined`: train `0.531376`, eval `0.514554`, first grad `3.0702`;
  - `actgrad_mlp_gmean`: train `0.545078`, eval `0.511652`, first grad `1.9072`;
  - `actgrad_combined_gmean`: train `0.546055`, eval `0.511424`, first grad `1.9062`.
- Interpretation:
  - geometric-mean normalization removes the raw MLP actgrad gradient inflation and eval degradation;
  - normalized combined actgrad is the best local 160-step result, but only by `0.00025` eval loss, so it is a promising smoke result, not evidence;
  - the next meaningful test is seeds and lower-rank/module-restricted settings.

## 2026-05-12 20:38 EDT - Local act-grad ablation results

- Local constant-LR 160-step all-linear LoRA comparison:
  - setup: Qwen3-0.6B, GSM8K, rank `8`, train/eval `1024/256`, LR `2e-4`, scheduler `constant_with_warmup`, seed `1`;
  - original:
    - train loss `0.546003`;
    - eval loss `0.511679`;
    - first logged grad norm `1.8420`;
  - actgrad V/O only:
    - train loss `0.546562`;
    - eval loss `0.511837`;
    - first logged grad norm `1.8501`;
    - clipped groups `8`;
  - actgrad MLP only:
    - train loss `0.531106`;
    - eval loss `0.513149`;
    - first logged grad norm `3.0518`;
    - clipped layers `28`;
  - actgrad combined:
    - train loss `0.531376`;
    - eval loss `0.514554`;
    - first logged grad norm `3.0702`;
    - clipped records `36`.
- Interpretation:
  - trace-normalized actgrad V/O is neutral in this setting;
  - raw MLP actgrad scaling lowers training loss but worsens held-out eval and inflates gradient norms;
  - combined degradation is mostly explained by MLP actgrad scaling;
  - next MLP actgrad variant should normalize or center the per-channel scale vector, not directly use raw `(g2/h2)^{1/4}`.

## 2026-05-12 20:21 EDT - Submitted act-grad LoRA comparison

- Extended `scripts/gsm8k_lora_smoke.py` with `actgrad_combined` canonicalizer.
- Added `slurm/gsm8k_actgrad_lora_smoke.sbatch`.
- Local two-step debug for actgrad LoRA passed:
  - proxy samples: `2`;
  - rank: `2`;
  - target modules: `all_linear`;
  - train loss: `1.5269`;
  - eval loss: `1.4335`;
  - note: high clipping under only two proxy samples is expected.
- Submitted two-task Slurm array:
  - job ID: `8149235`;
  - tasks: `original`, `actgrad_combined`;
  - rank: `8`;
  - all-linear LoRA;
  - LR: `2e-4`;
  - scheduler: `constant_with_warmup`;
  - max steps: `160`;
  - train/eval samples: `1024/256`;
  - actgrad proxy: `8` samples, length `256`.

## 2026-05-12 20:11 EDT - Activation-gradient smoke implemented and submitted

- Added activation-gradient canonicalization support:
  - `apply_gqa_value_output_covariance_balance`;
  - `apply_swiglu_mlp_activation_gradient_balance`;
  - `scripts/activation_gradient_smoke.py`;
  - `slurm/activation_gradient_smoke.sbatch`.
- Local two-sample debug without trace normalization:
  - ran successfully;
  - V/O mean balance residual before/after: `0.9997 -> 0.9750`;
  - all 224 V/O groups clipped;
  - conclusion: raw activation-gradient balancing is dominated by activation/gradient scale mismatch.
- Added trace normalization of gradient covariance before solving `X C X = F`.
- Local two-sample trace-normalized debug:
  - output: `results/activation_gradient_smoke/local_debug_s2_len128_trace_norm/metrics.json`;
  - proxy loss mean: `1.6769`;
  - V/O mean balance residual before/after: `0.6876 -> 0.0338`;
  - V/O clipped groups: `185/224`;
  - bf16 valid-logit MSE: `0.00377`;
  - last-token KL: `0.00176`;
  - top-1 agreement: `1.0`.
- Submitted 8-sample activation-gradient Slurm smoke:
  - job ID: `8149145`;
  - state: pending with reason `Priority`.

## 2026-05-12 20:02 EDT - Submitted VO-only low-rank stress test

- Updated `scripts/gsm8k_lora_smoke.py` with:
  - `--target-modules` choices: `all_linear`, `attention`, `vo`, `v_only`, `mlp`;
  - `--lr-scheduler-type`;
  - `--warmup-ratio`.
- Added `slurm/gsm8k_vo_lowrank.sbatch`.
- Ran local two-step debug for `vo_norm`, target modules `vo`, rank `2`, constant-with-warmup LR: passed.
- Submitted six-task Slurm array:
  - job ID: `8149031`;
  - tasks: original/vo_norm/combined at ranks `2` and `4`;
  - target modules: `v_proj,o_proj`;
  - LR: `3e-4`;
  - scheduler: `constant_with_warmup`;
  - max steps: `160`;
  - train/eval samples: `1024/256`.
- Initial queue state: pending with reason `Priority`.

## 2026-05-12 19:52 EDT - Local four-way GSM8K LoRA smoke completed

- Ran local 80-step rank-8 all-linear LoRA smoke on Qwen3-0.6B/GSM8K for four representatives:
  - original;
  - V/O norm;
  - MLP norm;
  - combined V/O+MLP norm.
- Shared setup:
  - train samples: `512`;
  - eval samples: `128`;
  - max length: `512`;
  - LR: `2e-4`;
  - default Trainer linear LR decay to zero;
  - seed: `1`.
- Results:
  - `original`: train loss `0.624332`, eval loss `0.531297`, runtime `68.96s`;
  - `vo_norm`: train loss `0.624208`, eval loss `0.531186`, runtime `75.30s`;
  - `mlp_norm`: train loss `0.624918`, eval loss `0.531503`, runtime `72.26s`;
  - `combined`: train loss `0.624653`, eval loss `0.530629`, runtime `69.54s`.
- Interpretation:
  - all transformed models train normally;
  - differences are very small and not claim-worthy from one seed;
  - combined is slightly best in this smoke, but the margin is only about `6.7e-4` eval loss vs original;
  - next experiments should stress basis sensitivity with lower rank, restricted target modules, LR schedule changes, or activation-gradient balancing.

## 2026-05-12 19:45 EDT - Submitted first Slurm jobs

- Submitted `slurm/static_qwen_smoke.sbatch` to `pli-c`:
  - job ID: `8148822`;
  - purpose: reproduce architecture + fp32 static canonicalization smoke through Slurm.
- Submitted `slurm/gsm8k_lora_smoke.sbatch` to `pli-c`:
  - array job ID: `8148823`;
  - tasks: `0=original`, `1=vo_norm`, `2=mlp_norm`, `3=combined`;
  - purpose: first 80-step GSM8K LoRA comparison across canonicalizers.
- Initial queue state: all pending with reason `Priority`.

## 2026-05-12 19:44 EDT - Local GSM8K LoRA debug passed

- Ran local two-step GSM8K LoRA debug:
  - script: `scripts/gsm8k_lora_smoke.py`;
  - canonicalizer: `original`;
  - rank: `4`;
  - max steps: `2`;
  - train samples: `8`;
  - eval samples: `4`;
  - max length: `128`;
  - output: `results/gsm8k_lora_smoke/local_debug_original/metrics.json`.
- Result:
  - trainable params: `2,523,136`;
  - train loss: `1.5260`;
  - eval loss: `1.4556`;
  - runtime: `2.81s`.
- Interpretation: cached GSM8K, PEFT LoRA, Hugging Face Trainer, and bf16 training work in the `rllm` environment. Safe to submit the four-canonicalizer Slurm smoke array.

## 2026-05-12 19:39 EDT - Corrected drift metrics and bf16 smoke

- Fixed `scripts/static_canon_smoke.py` so logit drift is measured only on valid prompt tokens and last-token KL uses the true last non-padding token.
- Reran fp32 static smoke:
  - V/O valid-logit MSE: `2.119e-11`;
  - V/O max valid-logit drift: `6.294e-5`;
  - V/O last-token KL: `2.16e-8`;
  - combined valid-logit MSE: `2.311e-11`;
  - combined max valid-logit drift: `6.294e-5`;
  - top-1 agreement: `1.0`.
- Reran bf16 static smoke:
  - V/O valid-logit MSE: `0.00433`;
  - V/O max valid-logit drift: `0.6875`;
  - V/O last-token KL: `0.00137`;
  - combined valid-logit MSE: `0.00455`;
  - combined max valid-logit drift: `0.75`;
  - combined last-token KL: `0.00226`;
  - top-1 agreement: `1.0`.
- Interpretation: fp32 transformation is exact to numerical noise; bf16 rewrite has measurable but acceptable first-smoke drift. Keep bf16 drift in safety metrics.

## 2026-05-12 19:30 EDT - Local fp32 exactness smoke passed

- Added initial `canon` package and scripts:
  - `canon/model_utils.py`;
  - `canon/transforms.py`;
  - `scripts/inspect_architecture.py`;
  - `scripts/static_canon_smoke.py`;
  - `scripts/gsm8k_lora_smoke.py`;
  - `slurm/static_qwen_smoke.sbatch`;
  - `slurm/gsm8k_lora_smoke.sbatch`.
- Fixed script import path issue by inserting the repo root into `sys.path` and exporting `PYTHONPATH` in Slurm scripts.
- Ran `python -m py_compile canon/*.py scripts/*.py`: passed.
- Ran local architecture inspection in `rllm`: passed and wrote `results/architecture/qwen3_0_6b.json`.
- Ran local fp32 static smoke:
  - output: `results/static_smoke/qwen3_0_6b_fp32/metrics.json`;
  - V/O vs original:
    - all-logit MSE: `3.159e-10`;
    - max abs logit drift: `3.252e-4`;
    - last-token top-1 agreement: `1.0`;
  - combined V/O+MLP vs original:
    - all-logit MSE: `2.407e-10`;
    - max abs logit drift: `3.071e-4`;
    - last-token top-1 agreement: `1.0`;
  - V/O balance records:
    - groups transformed: `224`;
    - mean balance residual before: `0.3227`;
    - mean balance residual after: `7.61e-15`;
    - no singular-value clipping.
- Interpretation: exact Qwen3 GQA V/O norm balancing and SwiGLU MLP scaling are implemented correctly enough to proceed to bf16 drift and LoRA smoke tests.

## 2026-05-12 19:24 EDT - Initial setup and target choice

- Read proposal before this run; project goal is finetuning-friendly function-preserving checkpoint canonicalization.
- Confirmed cluster partition `pli-c` is available and has mixed/allocated H100-class nodes; no current jobs under this user.
- Local node exposes an H100 PCIe with 81 GB memory, so local smoke tests are feasible before Slurm submission.
- Found local Qwen3 checkpoint at `/scratch/gpfs/ARORA/xd7812/models/Qwen3-0.6B`.
- Confirmed local config:
  - architecture: `Qwen3ForCausalLM`;
  - layers: 28;
  - hidden size: 1024;
  - attention heads: 16;
  - KV heads: 8;
  - head dim: 128;
  - `v_proj`: `[1024, 1024]`;
  - `o_proj`: `[1024, 2048]`;
  - `gate_proj/up_proj/down_proj`: `[3072,1024]`, `[3072,1024]`, `[1024,3072]`.
- Environment probe:
  - default Python: torch works, datasets works, transformers/peft are slow or timeout under short probes, bitsandbytes/trl missing;
  - `rllm` env: Python 3.12, torch 2.7.1+cu126, transformers 4.55.3, datasets 4.0.0, peft 0.17.1, accelerate 1.10.0, no bitsandbytes/trl.
- Web/model-card notes:
  - Qwen3-0.6B-Base is listed as a 0.6B pretraining-stage causal LM with 28 layers, GQA 16 Q / 8 KV heads, and 32k context;
  - Llama-3.2-1B is useful later but gated on Hugging Face;
  - OpenMathReasoning is relevant for main math SFT but too large for first smoke tests; cached GSM8K is better for initial LoRA plumbing.
