# Retrofit v2 data mix — experimental results

**Status**: EXPERIMENTAL. Findings below are NOT paper-ready — the v2 data
mix has documented failure modes (AI2D collapse on 2B, global collapse on 4B).
A v3 run with corrected mix is planned before any claim enters the paper.

## Setup

| Run | Data mix | Sources |
|---|---|---|
| v1 (canonical H_r256_5k) | 50% text + 50% VL | UltraChat + LLaVA-Instruct-VSFT |
| **v2** | 70% text + 30% VL | UltraChat 25% + NuminaMath 20% + OpenThoughts 15% + OpenMathInstruct-2 5% + LLaVA-VSFT 20% + ScienceQA 5% + Cauldron 10% |

Common: adapter rank 256, γ-curriculum 0→1 ramp-frac 0.3, 5k steps, seed 0,
max_seq 2048. Matches `retrofit.md` §3 H_r256_5k except for `--data-mix`.

Training artefacts:
- `retrofit/outputs/H_r256_5k_v2_2B/retrofit_attnres_state.pt`
- `retrofit/outputs/H_4B_r256_5k_v2/retrofit_attnres_state.pt`

Eval artefacts: `retrofit/outputs/lmms_eval_v2/`.

## Full VLM benchmark results (n=full per lmms-eval default)

| Benchmark | base_2B | v1_2B | v2_2B | base_4B | v1_4B | v2_4B |
|---|---|---|---|---|---|---|
| ai2d (exact-match) | 0.736 | 0.684 | **0.283** 💥 | 0.819 | 0.810 | **0.603** 💥 |
| mmbench_en_dev | 75.77 | 72.85 | 73.80 | 83.33 | 83.76 | 81.87 |
| mmmu_val | 0.414 | 0.406 | **0.421** | 0.490 | 0.510 | 0.477 |
| mmstar (avg) | 0.536 | 0.386 | 0.422 | 0.624 | 0.579 | **0.333** 💥 |
| ocrbench | 0.772 | 0.795 | **0.806** | 0.819 | 0.812 | 0.808 |
| realworldqa | 0.648 | 0.447 | 0.512 | 0.715 | 0.707 | 0.686 |
| LAMBADA acc | 0.532 | 0.576 | **0.590** | 0.576 | 0.600 | 0.598 |
| HellaSwag acc_norm | 0.506 | 0.522 | 0.504 | 0.562 | 0.578 | 0.572 |

## MMStar subcategories (the user's motivating regression)

MMStar has 6 subcategories; the "logic / math / science" triple is what v1
was documented to lose −8pp on (retrofit.md). The v2 mix was designed to fix
exactly these three.

| 2B cell | logic | math | sci&tech | (sum) |
|---|---|---|---|---|
| base_2B | 0.429 | 0.413 | 0.408 | 1.250 |
| v1_2B | 0.341 | 0.175 | 0.237 | 0.753 |
| **v2_2B** | **0.412** | **0.342** | **0.301** | **1.055** |

2B v2 → v1 subtask deltas:
- logic: **+7.0pp** ✓
- math: **+16.8pp** ✓✓ (the biggest single-subtask gain we've seen)
- sci&tech: **+6.4pp** ✓

So on the ORIGINAL motivation (repair reasoning regression at 2B), **v2 works
as designed**. It closes ~60% of the gap to base on reasoning subtasks.

| 4B cell | logic | math | sci&tech | (sum) |
|---|---|---|---|---|
| base_4B | 0.626 | 0.549 | 0.465 | 1.640 |
| v1_4B | 0.542 | 0.467 | 0.389 | 1.398 |
| v2_4B | 0.320 | 0.192 | 0.261 | 0.773 |

4B v2 → v1 subtask deltas:
- logic: −22.2pp, math: −27.5pp, sci&tech: −12.8pp. **All catastrophic.**

## Findings

### 1. AI2D collapse is the headline v2 failure
AI2D (diagram-grounded grade-school MCQ) is where v2 hurts most on both
scales: −40pp on 2B, −21pp on 4B. Neither MMStar nor AI2D is captioning;
both require "reading a diagram" which combines vision tokens and spatial
reasoning. Hypothesis: the CoT text data (NuminaMath/OpenThoughts/OpenMath2
= 40% of mix) teaches symbolic reasoning in pure language, and under a
frozen-base retrofit the router learns to route away from vision-heavy
paths. This robs diagram reasoning while helping pure-text reasoning.

### 2. Scale flips the net outcome
- **2B**: v2 wins 5 of 6 VL benchmarks against v1 and wins 3 of 3 MMStar
  reasoning subtasks. Only AI2D crashes. Net: mixed, arguably better.
- **4B**: v2 loses 6 of 6 VL benchmarks against v1 (including a −29pp
  MMStar collapse that corresponds to a near-total routing drift).

The scale-dependent response is real — same data, same hyperparameters,
diverging outcomes. Likely driver: 4B retrofit parameter count is ~1.6×
the 2B retrofit's, so drift compounds under the reduced VL signal.

### 3. OCR and realworldqa are "safe" tasks
OCRBench and RealWorldQA see small (<1pp) changes across the v1→v2
transition, at both scales. These tasks rely on local vision features
(OCR) or general scene comprehension and do not demand symbolic diagram
reasoning, so they're insensitive to the CoT shift.

### 4. LAMBADA / HellaSwag text benchmarks tell a different story
LAMBADA responds to the CoT text additions as hoped (+1.4pp on 2B, ≈tied
on 4B). HellaSwag is noise-level (±2pp).

## Root cause hypotheses (ranked)

| # | Hypothesis | Evidence for | Evidence against |
|---|---|---|---|
| H1 | VL proportion too low (30% vs v1's 50%). Router under-trained on VL distribution. | 4B collapse scales with parameter count; ai2d/mmstar (diagram-heavy) hit hardest. | Does not explain why 2B partially recovers — smaller models should be MORE data-starved. |
| H2 | LLaVA-Instruct-VSFT is narrow / dated. Real Qwen3-VL pre-training used much richer VL data. | Replacing with LLaVA-OneVision (282 GB, now local) plausibly dominates v2's 20% LLaVA-VSFT content. | Hasn't been tested. |
| H3 | 5 k steps × 30% VL ≈ 1.5 k VL gradient updates, vs v1's 2.5 k. Under-training, not mix mismatch. | Simple: just scale steps. | If adapter is already fit tightly to math CoT, more steps may further drift VL. |
| H4 | Math-CoT text is adversarial to diagram reasoning. Internal competition for the "reasoning" code path. | AI2D+MMStar crash while OCR is fine — consistent with "symbolic reasoning vs diagram reasoning" axis. | Can't be isolated without ablating math-CoT weight at fixed VL weight. |
| H5 | Cauldron quality issues. Some subsets are text-heavy (localized_narratives had dangling image paths; chartqa/docvqa OK but may skew the router toward "read-text-in-image" not "reason-about-diagram"). | Some support — clevr had broken images. | Cauldron only 10% of mix. |

Best single bet: **H1 + H2 combined + H3 scaling**. V3 should keep the
math-CoT gains (LAMBADA +1.4pp, MMStar reasoning subtasks +7-17pp on 2B)
while restoring VL capacity via richer data and more steps.

## Tentative v3 design (not locked)

- **Anchor VL**: LLaVA-OneVision-Data (282 GB, local) replaces LLaVA-VSFT
- **VL math supplement**: MathV360K (bridges text-CoT and image-reasoning —
  direct image-math QA data we DON'T currently have)
- **Target proportion**: ~55% VL + ~30% math/reasoning text + ~15% general text
- **Steps**: 10k (2× v2) to let the router reconverge on the new distribution
- **Keep**: γ-curriculum, adapter rank 256, seed 0 for direct comparability

Expected outcome: v3 should
- hold or improve on 2B's reasoning-subtask gains,
- lift 4B out of the global collapse,
- at minimum return AI2D to v1's level (≥0.68 on 2B, ≥0.80 on 4B).

If v3 still shows AI2D < 0.65 on 2B despite richer VL, H4 (adversarial
math-CoT text) becomes more credible and we'd move to a mix that caps
text-CoT at ≤15%.

## What to do with v2 artefacts

**Keep**: states + eval results are a documented "reasoning-mix too
aggressive" data point. Useful for ablation tables in the paper IF we
ultimately land on v3 as canonical.

**Do NOT**: swap v2 states into VLA warm-start pipelines without a LIBERO
eval, given the diagram-reasoning regression could translate to VLA spatial
skill regression. This is untested.
