# Canon Research Thoughts

## 2026-05-14 17:38 EDT

The first generation-level result is directionally positive. On saved 1000-step MATH full-finetuned models, combined reaches `20.7%` normalized exact match on MATH-500 versus `19.3%` for original over seeds 1-2. It also has slightly higher gold-in-generation and boxed-answer rates.

This matters because it converts the small 1000-step eval-loss gap into an answer-level signal. The result is still not strong enough to be a headline claim: the evaluator is string-based, not symbolic; two seeds are not enough; and the earlier 2000-step loss result warns that the effect may be transient. But it changes the next priority. Before spending more GPU time, the best move is to improve the grader or re-score the existing predictions with symbolic equivalence. If the `+1.4` point exact-match gain survives a better verifier, then the project has a real preliminary story: exact canonicalization gives a small early full-training quality gain on MATH generation, not just lower gradient norm.

## 2026-05-14 17:13 EDT

I moved Priority 2 into execution. Because the previous MATH full-training jobs used `save_strategy="no"` and did not save final models, generation evaluation requires a targeted rerun. I submitted the smallest version that can answer the immediate question: original vs combined, seeds 1-2, 1000 MATH steps, save final models, then run deterministic MATH-500 generation.

The first generation grader is deliberately simple. It should not be treated as a final MATH evaluator because it lacks symbolic equivalence checking. But it does measure useful surface behavior: whether the model emits a boxed answer, whether the normalized extracted answer exactly matches the MATH-500 answer, and whether the normalized gold appears anywhere in the generation. If these metrics are completely flat, the project should not oversell the 1000-step loss gap. If combined improves boxed rate or exact match, that justifies a more rigorous symbolic grader and saved trajectory checkpoints.

## 2026-05-14 17:02 EDT

The current project state is now clearer after consolidating the report. The most important evidence is not a single eval-loss win; it is the pattern across adaptation regimes:

- exact canonicalization works numerically for Qwen3;
- constrained V/O-only LoRA shows small but coherent static V/O gains;
- all-linear LoRA mostly washes out the effect;
- full fine-tuning shows consistent gradient-scale reduction;
- MATH 1000-step full training gives the cleanest short-run eval signal;
- MATH 2000-step training removes the eval advantage despite lower combined train loss.

This points to canonicalization as an optimizer-geometry intervention, not yet as a final-quality recipe. The project should now make that precise. A trajectory experiment is more informative than another final checkpoint sweep: if combined reaches the same validation loss in fewer steps, then the result is a speed/preconditioning result; if it briefly improves then overfits earlier, that also matters; if generation exact match does not move, then loss-level claims should stay modest.

The report in `progress-report.md` should be treated as the current baseline summary. The next code work should add step-wise MATH eval and saved checkpoints for generation evaluation.

## 2026-05-13 18:39 EDT

The 2000-step MATH check changes the conclusion. Combined still has lower training loss and lower first gradient norm, but its MATH-500 eval loss is worse than original:

- 1000 steps, four seeds: combined better by `0.00041` mean eval;
- 2000 steps, two seeds: combined worse by `0.00044` mean eval.

So the canonicalization effect is real enough to affect optimization scale, but not yet a reliable final-quality gain. It looks like an early-training/preconditioning effect that may wash out or overfit under longer full fine-tuning. That is still useful scientifically: it tells us where to focus the project. The next rigorous step should be trajectory analysis and generation/exact-match, not just more final-loss sweeps. In particular, compare losses at intermediate checkpoints or log steps-to-loss-threshold; the existing final-only Trainer setup is hiding the transient advantage.

Current working conclusion: full training is the right priority over LoRA, but the present static transforms need either a better objective/metric or a trajectory-based claim. The most defensible result so far is: combined canonicalization reduces early gradient scale and modestly improves short-run MATH full-finetuning loss, but does not survive as a clear 2000-step eval-loss improvement.

## 2026-05-13 18:10 EDT

The four-seed MATH full-training result is the cleanest full-training signal so far. Combined beats original on all four seeds:

- seed 1: `-0.00008`;
- seed 2: `-0.00021`;
- seed 3: `-0.00050`;
- seed 4: `-0.00086`.

Mean MATH-500 eval is `0.68611` for original and `0.68570` for combined. First gradient norm is also consistently lower (`12.25` original mean vs `10.55` combined mean). The absolute loss gap is still small, but this is more coherent than GSM8K and supports the user's request to focus on full training.

I queued a 2000-step MATH run for seeds 1-2. This is an important check: if the gap persists or grows, canonicalization is affecting full-training optimization beyond the first epoch-ish pass. If the gap disappears, the result is likely an early optimization/preconditioning effect rather than a final-quality effect.

## 2026-05-13 17:44 EDT

The MATH pilot gives the same qualitative picture as full GSM8K, but on a harder task: combined is slightly better and has lower first gradient norm, yet the eval-loss gap is still small. On seeds 1-2:

- original MATH-500 eval: `0.68656`;
- combined MATH-500 eval: `0.68642`;
- combined deltas: `-0.00008`, `-0.00021`.

This is not significant, but unlike GSM8K LR `2e-5`, both MATH seeds move in the same direction. I queued seeds 3-4 before making any claim. If the four-seed MATH result remains consistently positive, the next step should be a generation/exact-match evaluation or a larger/harder full-training run. If it collapses, the current canonicalizers probably only provide small optimizer-scale changes, not meaningful task gains at this model/task scale.

## 2026-05-13 17:26 EDT

The GSM8K full-training LR sweep does not reveal a strong canonicalizer effect. Combined is worse than original at `1e-5`, slightly better at `2e-5` on mean but not seed-robust, and slightly better at `5e-5` on two seeds. The gradient scale reduction is consistent, but eval loss is too flat for GSM8K to be the primary full-training benchmark.

I found cached harder math datasets locally: Hendrycks MATH categories and MATH-500. This is a better next step than continuing to tune GSM8K. I added a full-training MATH runner that trains on the seven Hendrycks MATH training categories and evaluates on MATH-500. A tokenization bug appeared in the first debug because long MATH prompts could consume the entire sequence and leave no answer labels; I fixed this by reserving answer-token budget.

The submitted MATH full-training pilot compares original vs combined for seeds 1-2 at LR `2e-5`, 1000 steps, max length 1024. This should test the user's requested full-training direction on a harder task where the loss signal has more room to separate.

## 2026-05-13 17:04 EDT

The four-seed full-finetuning result at LR `2e-5` is not significant. Combined still has the best mean eval loss, but the pattern is not seed-robust:

- original mean eval: `0.46498`;
- V/O norm mean eval: `0.46502`;
- combined mean eval: `0.46481`.

Combined beats original on seeds 1 and 2, then loses on seeds 3 and 4. The mean improvement is only `0.00017`, smaller than normal experimental noise for this setup. The interesting secondary signal is still optimizer scale: combined and V/O norm have lower first gradient norms than original.

I am keeping the focus on full training, but the right next step is not more seeds at the same LR. I queued an LR sensitivity sweep comparing original vs combined at `1e-5` and `5e-5` for seeds 1-2. If canonicalization has a useful optimizer effect, it may show up as better robustness at a larger LR or different early-loss behavior. If the LR sweep is also flat, GSM8K loss is probably exhausted as a diagnostic and we should move full fine-tuning to a harder math subset.

## 2026-05-13 16:52 EDT

The first full-finetuning pilot is more encouraging than the LoRA loss sweeps, although still not a claim. Full fine-tuning reaches much lower eval loss than the rank-restricted LoRA runs, as expected. On seeds 1-2 at LR `2e-5`, 500 steps:

- original mean eval: `0.46516`;
- V/O norm mean eval: `0.46484`;
- combined mean eval: `0.46454`.

Combined V/O+MLP beats original on both seeds, while V/O norm is mixed. This differs from the V/O-only LoRA story and makes sense because full fine-tuning updates the MLP as well. The full-parameter optimizer can interact with the MLP scale gauge, so combined is now worth keeping.

The differences are still small, roughly `6e-4` mean eval for combined vs original. I queued seeds 3-4 with the same setup before touching LR. If combined remains consistently better across four seeds, the next full-training axis should be LR sensitivity (`1e-5`, `2e-5`, maybe `5e-5`) or a harder math subset.

## 2026-05-13 16:42 EDT

The LoRA results are too small to be the main evidence path, so I am switching the next stage to full fine-tuning as requested. This changes the question: with all weights trainable, canonicalization is no longer about whether a low-rank adapter can express an update in a better basis. It is about whether optimizer dynamics, Adam moments, weight decay choices, and finite-precision updates behave better from one exact representative than another.

That means the best first full-training comparison is not a broad transform zoo. It should be original vs the cleanest exact transform found so far, static V/O norm, plus combined V/O+MLP as a secondary control. I am starting with Qwen3-0.6B on GSM8K because the infrastructure is already validated and the checkpoint is local. The first pilot uses full bf16 fine-tuning, all `596,049,920` parameters trainable, LR `2e-5`, 500 steps, 4096 train samples, and the full GSM8K test split for loss.

The two-step local debug passed with gradient checkpointing and only about `5.6 GB` peak CUDA allocation for the tiny sequence/batch smoke, so the H100 Slurm jobs should have comfortable memory headroom. The key risk is not OOM; it is that GSM8K loss remains too insensitive. If this pilot shows no material difference, the next move should be full fine-tuning on a harder/larger math slice rather than returning to more LoRA sweeps.

## 2026-05-12 21:51 EDT

Rank 1 did not produce a dramatic amplification, but it did preserve the pattern. Mean eval loss is `0.50636` for original and `0.50612` for V/O norm. V/O norm wins seeds 2, 3, and 4, loses seed 1 by `0.00019`, and lowers first gradient norm on every seed.

Across fuller sweeps, the summary is now:

- rank 4: V/O norm mean gain `0.00012`, wins 3/4 seeds;
- rank 2: mean gain `0.00034`, wins 3/4 seeds;
- rank 1: mean gain `0.00024`, wins 3/4 seeds.

This is not a large benchmark result, but it is a coherent preliminary optimization signal. The strongest consistent metric is not eval loss; it is first gradient norm reduction, which appears across all ranks. That suggests V/O norm balancing is functioning as a mild adapter preconditioner. The next useful experiment should not be another GSM8K loss sweep. It should either:

1. add generation/exact-match evaluation with saved adapters;
2. run a harder math subset where loss gaps may spread;
3. test another architecture with a different GQA gauge, once local access is available.

## 2026-05-12 21:28 EDT

The rank-2 fuller sweep supports the constrained-adaptation interpretation more than the rank-4 fuller sweep. With 500 steps and full-test eval, mean eval loss is `0.49772` for original and `0.49738` for V/O norm, a mean gain of `0.00034`. V/O norm wins 3 of 4 seeds, and its first logged gradient norm is lower on every seed by roughly `0.04-0.047`.

The loss gain is still modest, but the gradient-norm effect is consistent across rank 4 and rank 2. This strengthens the idea that V/O balancing is changing the local optimization scale in a useful direction, even when final eval loss differences are small.

I queued rank 1 as the next capacity stress point. If rank 1 shows a larger or cleaner V/O norm advantage, the preliminary story becomes much more coherent: the canonical basis matters when the low-rank adapter has very little capacity. If rank 1 is noisy or worse, rank 2 may be the practical lower bound where the adapter can exploit the balanced basis.

## 2026-05-12 21:04 EDT

The larger rank-4 validation weakens but does not erase the static V/O signal. With 500 steps, 4096 train examples, and full GSM8K test loss, mean eval loss is `0.48895` for original and `0.48883` for V/O norm. Per-seed deltas are `+0.00009`, `-0.00007`, `-0.00029`, `-0.00021`; V/O norm wins 3 of 4 seeds, but the mean gain is only `0.00012`.

This is a classic preliminary result: robust enough to keep investigating, too small to claim. The short 160-step sweep showed a larger effect, so canonicalization may mainly help early optimization or tighter bottlenecks. To test the bottleneck explanation, I queued the same fuller validation at rank 2. If rank 2 recovers a larger consistent gain, the story becomes "canonicalization helps most when adaptation capacity is severely constrained." If rank 2 is also tiny, we should pivot to a harder task, a generation accuracy metric, or a model where the original gauge is less benign.

## 2026-05-12 20:48 EDT

The actgrad V/O follow-up is a useful boundary result. In the same rank-4 V/O-only setting, actgrad V/O does not beat static V/O norm. Across seeds 1-4, mean eval losses are:

- original: `0.51950`;
- static V/O norm: `0.51826`;
- actgrad V/O: `0.51960`.

Actgrad V/O improves over original on seeds 1 and 2 by tiny margins, but is worse on seeds 3 and 4. It is worse than static V/O norm on every seed. This suggests the 8-example proxy gradient estimate is too noisy or too mismatched for the restricted V/O adapter, even though it looked promising in the all-linear actgrad smoke.

The working hypothesis should shift: static weight-geometry canonicalization is currently the strongest method, and actgrad should be treated as a future refinement requiring better covariance estimation, more proxy samples, or a different normalization. The next validation should scale the static V/O result to more training examples, more steps, and the full GSM8K test split for eval loss.

## 2026-05-12 20:37 EDT

The rank-4 V/O-only seed replication is now the strongest preliminary result. Across seeds 1-4, static V/O norm beats original on every seed:

- seed 1: `-0.00162` eval loss;
- seed 2: `-0.00053`;
- seed 3: `-0.00183`;
- seed 4: `-0.00098`.

Mean eval loss is `0.51950` for original and `0.51826` for V/O norm, a mean improvement of about `0.00124`. Combined V/O+MLP norm also improves over original on every seed, but V/O norm is slightly better on average (`0.51826` vs `0.51840`). This supports the narrower claim that canonicalizing the adapted GQA value/output subspace can improve low-rank adaptation geometry.

The effect is still small in absolute loss, and it is not yet an accuracy result. But it is consistent across four seeds under a deliberately constrained adapter. That is enough to justify a next-stage actgrad V/O sweep in the same setting, reusing these original/static baselines. If actgrad V/O wins there too, the project has two plausible canonicalization levels: a data-free architecture-aware gauge and a data-aware Fisher-shape gauge.

## 2026-05-12 20:32 EDT

The Slurm act-gradient result changes the interpretation from "actgrad is risky" to "actgrad is useful only after separating the gauges." Matched Slurm baseline eval is `0.51276`. V/O actgrad reaches `0.51130` with a slightly lower first gradient norm (`1.83 -> 1.77`). Geometric-mean MLP actgrad reaches `0.51136`, and combined gmean reaches `0.51140`. Raw combined actgrad remains worse at `0.51333` with first grad norm `3.07`.

The likely explanation is that the full raw combined transformation mixes a good V/O shape correction with an MLP global-scale distortion. The gmean fix removes most of that distortion but leaves many MLP channels clipped, so the data-aware MLP transform is still not as clean as the exact V/O matrix gauge. For the next serious data-aware branch, prioritize V/O actgrad and treat MLP actgrad as an ablation rather than the default.

The static V/O norm and actgrad V/O are now the two best-aligned methods with the proposal. Static V/O norm has a clean exact geometry and showed a low-rank seed-1 signal. Actgrad V/O is more expensive and data-dependent, but it produced the best Slurm all-linear actgrad result. The key next comparison is not more variants; it is whether either V/O method survives seed replication.

## 2026-05-12 20:27 EDT

I queued a rank-4 V/O-only seed sweep rather than expanding the all-linear setting. The reason is statistical and mechanistic: the seed-1 low-rank result is the first one that reflects the proposal's predicted failure mode, namely that low-rank adaptation sees a different optimization geometry under different exact representatives. Replicating that setting across seeds is more informative than adding more one-off canonicalizers.

The seed sweep includes combined V/O+MLP norm as a control, but the main comparison is original vs V/O norm. If V/O norm wins consistently while combined is mixed, the paper story should focus narrowly on architecture-aware gauge fixing for the adapted subspace, not on applying every available exact transform by default. That would be a better result scientifically: it identifies when canonicalization matters and prevents us from selling a broad but brittle recipe.

## 2026-05-12 20:25 EDT

The Slurm low-rank V/O-only stress test gives the first directional result that is more aligned with the proposal's hypothesis. With LoRA restricted to `v_proj,o_proj`, static V/O norm balancing improved eval loss at both ranks:

- rank 2: original `0.53732`, V/O norm `0.53661`, combined `0.53697`;
- rank 4: original `0.52998`, V/O norm `0.52836`, combined `0.52899`.

The rank-4 V/O norm improvement is about `0.00162`, larger than the earlier all-linear rank-8 smoke deltas. This is still one seed and a short SFT proxy, so it is not evidence by itself. But it is a useful sign that the coordinate system matters more when the adapter is rank-constrained and module-constrained. This matches the project thesis better than the all-linear smoke, where LoRA had enough freedom to wash out a basis effect.

Combined V/O+MLP norm was not better than V/O alone in this restricted target setting. That makes sense because the adapter is not touching MLP modules, so MLP canonicalization only changes frozen forward numerics and any indirect attention/MLP interaction. For V/O-only LoRA, V/O norm should be the cleaner representative.

The gradient norms also move in the expected direction: V/O norm slightly lowers the first logged gradient norm at both ranks (`0.642 -> 0.608` at rank 2, `0.840 -> 0.813` at rank 4). This is consistent with preconditioning rather than pure random noise, but it needs seed replication. The immediate next experiment should be a seed sweep for rank-4 V/O-only LoRA, comparing original vs V/O norm and maybe combined only as a control.

## 2026-05-12 19:24 EDT

The first preliminary decision is to start with Qwen3-0.6B rather than Llama-3.2-1B. The proposal already suggested this, and the local environment strengthens that choice: `/scratch/gpfs/ARORA/xd7812/models/Qwen3-0.6B` is already present, while Llama may involve gated access. Qwen3-0.6B is also the cleaner exact-gauge target because the local config shows a compact GQA layout: 28 layers, hidden size 1024, 16 Q heads, 8 KV heads, head dimension 128, and SwiGLU MLP with 3072 hidden channels.

The exact value-output gauge for this model is straightforward but slightly nonstandard because `num_attention_heads * head_dim = 2048`, larger than `hidden_size = 1024`. The value projection has shape `[1024, 1024]`, corresponding to 8 KV heads times 128 channels. The output projection has shape `[1024, 2048]`, corresponding to 16 query-head output blocks. Each KV head is shared by 2 query heads. Therefore a per-KV-head matrix `P_j in R^{128 x 128}` should left-multiply the corresponding 128 rows of `v_proj`, while `P_j^{-1}` right-multiplies the two corresponding 128-column blocks of `o_proj`.

For first experiments, weight-only V/O norm balancing is safer than activation-gradient balancing because it needs no task batch, no autograd hooks, and should expose implementation errors through logit drift immediately. If this exact transformation cannot preserve logits to near fp32 precision, the more complex data-aware method is not worth launching.

SwiGLU hidden-channel balancing is also easy to test because Qwen3 exposes `gate_proj`, `up_proj`, and `down_proj`. The exact positive scaling acts on `up_proj` rows and `down_proj` columns while leaving `gate_proj` unchanged. This provides a second independent exact canonicalizer with a much cheaper diagonal transform.

The default Python environment is not ideal: `torch` works, `datasets` works, but `transformers`/`peft` imports are slow in Python 3.13 and `bitsandbytes`/`trl` are missing. The `rllm` conda env is the best current candidate: Python 3.12, torch 2.7.1+cu126, transformers 4.55.3, datasets 4.0.0, peft 0.17.1, accelerate 1.10.0. There is no bitsandbytes there, so QLoRA should wait until a suitable env is installed or found.

Web/model-card research supports the local choice: Hugging Face lists Qwen3-0.6B-Base as a pretraining-stage 0.6B causal LM with 28 layers, GQA with 16 Q heads and 8 KV heads, and 32k context. The Llama-3.2-1B card confirms GQA and 128k context, but it requires accepting Meta access terms. OpenMathReasoning is a strong long-term math SFT dataset, but it has 5.68M rows; for a preliminary cluster smoke test, cached GSM8K is a more practical short-run sanity task.

Immediate experimental risk: preliminary LoRA on GSM8K may be too weak/noisy to show a meaningful benchmark improvement in a short run. That is acceptable for stage 0 because the first goal is to validate canonicalization implementation, measure drift, and check whether training curves are at least comparable across original vs canonicalized representatives.

## 2026-05-12 19:30 EDT

The fp32 exactness smoke strongly supports moving forward. V/O norm balancing produced all-logit MSE `3.16e-10`, max logit drift `3.25e-4`, last-token KL around numerical zero, and top-1 agreement `1.0` on calibration prompts. Combined V/O + MLP norm balancing was similar. This is the behavior expected from a true gauge transform with small floating-point roundoff.

The V/O geometry result is more interesting than the drift alone: the mean relative gap between `P A P^T` and `P^{-T} B P^{-1}` drops from `0.323` before balancing to `7.6e-15` after balancing, with no singular-value clipping. This suggests the solver and the Qwen GQA block indexing are correct. The per-group transform spectrum is also mild: global min singular value about `0.473`, max about `2.09`. This is not an aggressive or sabotage-like gauge; it is a plausible static preconditioner.

The MLP norm balancing implementation is exact but the current summary metric is not ideal. It reports the log-scale spread of the scaling factors, not a before/after matched norm residual. That is sufficient for the smoke test but should be refined before using the metric in a paper-quality table.

## 2026-05-12 19:39 EDT

The original bf16 drift looked alarming, but that was partly a metric bug: the script compared padded positions and used the padded last column as the "last token." After restricting to valid prompt tokens and the true last non-padding token, fp32 max valid-logit drift is only `6.29e-5`. bf16 V/O drift is still real but moderate: valid-logit MSE `0.00433`, max valid-logit drift `0.6875`, last-token KL `0.00137`, and top-1 agreement `1.0`.

This suggests bf16 training from a directly rewritten bf16 checkpoint is probably acceptable for smoke tests, but final experiments should be more careful. Better options to compare:

1. apply transforms in fp32, save fp32 master weights, let mixed-precision training cast as needed;
2. apply transforms then save bf16, matching typical Hugging Face checkpoint storage;
3. add a stricter drift threshold and condition-number clipping grid for bf16.

For now, proceed with LoRA smoke because the exactness and indexing are validated. The bf16 drift is a measurement to log, not a blocker.

## 2026-05-12 19:52 EDT

The first local LoRA comparison is deliberately small and should not be overread. With Qwen3-0.6B, GSM8K, rank-8 all-linear LoRA, 512 train examples, 128 eval examples, and 80 optimizer steps, all four representatives produce essentially the same curve. Eval losses:

- original: `0.53130`;
- V/O norm: `0.53119`;
- MLP norm: `0.53150`;
- combined: `0.53063`.

The combined result is numerically best, but the margin is only `6.7e-4` eval loss from original. That is below the threshold for a claim. The more important observation is that none of the canonicalizers destabilize LoRA, and the gradient norms look similar. This means the exact transforms are safe enough to include in a larger sweep.

This neutral smoke result also suggests a useful next-stage adjustment: plain weight-norm balancing may be too weak on a short, easy GSM8K loss objective. To see a real effect we probably need at least one of:

1. lower-rank LoRA where basis choice matters more, such as rank 2 or 4;
2. a more constrained module subset, especially `v_proj,o_proj` only;
3. a larger LR or no decay-to-zero schedule to expose stability differences;
4. data-aware activation-gradient balancing;
5. a harder or more mismatched checkpoint/task pair.

The current Trainer default uses a linear learning-rate schedule to zero over 80 steps. That is fine as a controlled smoke, but it is not ideal for measuring steps-to-threshold. The next jobs should add constant-with-warmup or a small LR grid.

## 2026-05-12 20:02 EDT

The next queued experiment directly targets the V/O hypothesis instead of spreading LoRA capacity across every linear layer. If LoRA is all-linear at rank 8, the adapter can route around basis changes in many modules. By restricting LoRA to `v_proj,o_proj` and using ranks 2 and 4, the exact GQA gauge should have a better chance of changing the effective low-rank geometry. This is still not the full activation-gradient method, but it is a better stress test for whether static gauge choice matters.

I changed the LR schedule to `constant_with_warmup` for this branch because the previous 80-step smoke decayed to almost zero, which compresses the last half of the learning curve and makes steps-to-threshold analysis less meaningful.

## 2026-05-12 20:11 EDT

Activation-gradient balancing has a scale-identifiability issue that the proposal did not make explicit enough. The activation covariance `C` and gradient covariance `F` have different units, and the absolute scale of `F` depends on loss normalization. Solving `X C X = F` directly can choose a huge global rescaling unrelated to useful geometry. With only two GSM8K proxy examples, raw data-aware V/O balancing clipped every group at the singular-value bounds.

The first fix is trace normalization: rescale `F` to match `trace(C)` before solving. This turns the experiment into shape balancing instead of loss-scale balancing. With trace normalization, V/O mean balance residual improved from `0.688` to `0.0338` after clipping on the two-sample debug. Many groups still clipped, likely because the covariance estimates are very low-rank with too few samples. The 8-sample Slurm job should show whether this stabilizes.

This is a useful design lesson: final data-aware canonicalization probably needs explicit choices among:

1. full scale balancing, which changes effective step sizes strongly;
2. trace-normalized shape balancing, which is safer;
3. determinant-normalized balancing, which removes volume changes;
4. a global scale budget tuned on calibration drift.

## 2026-05-12 20:21 EDT

The 8-sample activation-gradient static result is promising as an implementation milestone: V/O clipping drops to only `8/224` groups and the V/O balance residual reaches `3.5e-4`. This is much healthier than the two-sample estimate. The data-aware MLP scaling still clips all layers on the lower scale bound, with scales roughly `0.25` to `0.379`; that may indicate the gradient-vs-activation scale issue remains for the diagonal SwiGLU case. It may need its own trace/geometric-mean normalization rather than using raw `(g2/h2)^{1/4}`.

The first act-gradient training run should be treated as a smoke test, not a benchmark. It collects proxy gradients on the same broad GSM8K distribution and then trains on GSM8K, so if it helps, the next question is whether that transfers to held-out tasks or just tunes the gauge to the proxy loss.

## 2026-05-12 20:38 EDT

The first act-gradient training ablation is a useful negative/diagnostic result. V/O actgrad alone is basically neutral: eval `0.51184` vs original `0.51168`, with the same gradient scale. MLP actgrad alone lowers training loss substantially (`0.5311` vs `0.5460`) but worsens eval (`0.51315`) and raises early gradient norm from `1.84` to `3.05`. Combined inherits this pattern and is worse (`0.51455`).

This points to an MLP scaling pathology, not a general failure of V/O balancing. The raw MLP formula `c_i=(E[g_i^2]/E[h_i^2])^{1/4}` drove every layer to the lower clipping boundary. That is likely acting as a global effective LR/regularization change on MLP LoRA rather than a clean channel-basis improvement. For the next MLP variant, use a normalized scale vector, e.g. divide `c` by its geometric mean per layer before clipping, so the transform changes relative channel conditioning without globally shrinking all up-projection rows and expanding all down-projection columns.

## 2026-05-12 20:52 EDT

Geometric-mean normalization did what it should. Raw MLP actgrad had first grad norm `3.05` and eval `0.51315`; gmean-normalized MLP actgrad has first grad norm `1.91` and eval `0.51165`, essentially tied with original. Combined gmean is the best local 160-step run so far at eval `0.51142` versus original `0.51168`, but this is only a `0.00025` loss difference. The right conclusion is not "we improved GSM8K"; it is "the data-aware machinery can be made numerically sane, and the unsafe MLP scaling failure has a clear fix."

The low-rank V/O-only jobs remain important. If canonicalization has a real effect, it should be more visible when rank and module freedom are restricted. The all-linear rank-8 setting has enough capacity that many coordinate effects may be washed out.
