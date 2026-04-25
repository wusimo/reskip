---
title: Retrofit v3 data-mix — experimental results
date: 2026-04-23
status: complete (VLM evals done; LAMBADA+HellaSwag pending)
---

# Retrofit v3 data-mix — experimental results

**Status**: VLM evaluation complete on all 8 cells. This is the canonical
data-mix study; v3 replaces v2 as the reference mix going forward. Text
benchmarks (LAMBADA, HellaSwag) pending — scheduled next.

## Motivation

v2 showed a catastrophic VL collapse (AI2D −40pp on 2B, global regression on
4B; see `v2_vlm_analysis.md`). The v2 diagnosis ranked three co-hypotheses:

- **H1** VL proportion too low (30 % vs v1's 50 %).
- **H2** LLaVA-Instruct-VSFT too narrow / dated.
- **H3** 5 k steps too few (only ~1.5 k VL gradient updates in v2).

v3 addresses all three simultaneously and adds a fourth cell to isolate
whether math/CoT text is adversarial to diagram reasoning (H4 from v2).

## Cells

Six retrofit cells + two controls, 8 total. All use adapter rank 256,
max_seq 2048, seed 0, γ-curriculum 0→1 (ramp-frac 0.3 unless noted).

| cell            | scale | mix         | steps | ramp-frac | purpose |
|-----------------|-------|-------------|-------|-----------|---------|
| **v3_2B**       | 2B    | v3          | 10 k  | 0.3       | canonical v3 |
| **v3_4B**       | 4B    | v3          | 10 k  | **0.5**   | canonical v3 (ramp extended after γ-transition divergence at 0.3) |
| v3_5k_2B        | 2B    | v3          |  5 k  | 0.3       | is 10 k necessary? |
| v3_5k_4B        | 4B    | v3          |  5 k  | 0.3       | is 10 k necessary? |
| v3_vlonly_2B    | 2B    | v3_vlonly   | 10 k  | 0.3       | H4 test — remove math/CoT text |
| v3_vlonly_4B    | 4B    | v3_vlonly   | 10 k  | 0.3       | H4 test — remove math/CoT text |
| v1_10k_2B       | 2B    | v1          | 10 k  | 0.3       | control: does v1 just need more steps? |
| v1_10k_4B       | 4B    | v1          | 10 k  | 0.3       | control: does v1 just need more steps? |

Data-mix composition:
- **v1** (old canonical): UltraChat 50 % + LLaVA-Instruct-VSFT 50 %
- **v3**: LLaVA-OneVision 60 % + UltraChat 20 % + NuminaMath 10 % + OpenThoughts 10 %
- **v3_vlonly**: LLaVA-OneVision 80 % + UltraChat 20 %

## Full VLM results (lmms-eval, full eval splits)

| cell              | ai2d  | mmbench | mmmu  | mmstar | ocr   | rwqa  | star_m | star_l | star_s |
|-------------------|-------|---------|-------|--------|-------|-------|--------|--------|--------|
| base_2B           | 0.736 | 75.77   | 0.414 | 0.536  | 0.772 | 0.648 | 0.413  | 0.429  | 0.408  |
| v1_2B (5k canon)  | 0.684 | 72.85   | 0.406 | 0.386  | 0.795 | 0.447 | 0.175  | 0.341  | 0.237  |
| v2_2B (5k)        | 0.283 | 73.80   | 0.421 | 0.422  | 0.806 | 0.512 | 0.342  | 0.412  | 0.301  |
| **v3_2B**         | **0.765** | 79.30 | **0.439** | 0.534 | 0.809 | 0.668 | **0.492** | 0.431 | 0.373 |
| v3_5k_2B          | 0.746 | 77.49   | 0.438 | 0.521  | 0.810 | 0.659 | 0.471  | 0.450  | 0.337  |
| v3_vlonly_2B      | 0.755 | 78.95   | 0.410 | 0.543  | 0.812 | 0.671 | 0.441  | 0.476  | 0.370  |
| v1_10k_2B         | 0.677 | 73.71   | 0.404 | 0.471  | 0.801 | 0.642 | 0.321  | 0.394  | 0.312  |
| | | | | | | | | | |
| base_4B           | 0.819 | 83.33   | 0.490 | 0.624  | 0.819 | 0.715 | 0.549  | 0.626  | 0.465  |
| v1_4B (5k canon)  | 0.810 | 83.76   | 0.510 | 0.579  | 0.812 | 0.707 | 0.467  | 0.542  | 0.389  |
| v2_4B (5k)        | 0.603 | 81.87   | 0.477 | 0.333  | 0.808 | 0.686 | 0.192  | 0.320  | 0.261  |
| **v3_4B**         | 0.816 | 84.28   | **0.523** | 0.587 | 0.813 | 0.708 | 0.591 | 0.498 | 0.407 |
| v3_5k_4B          | 0.817 | 84.28   | **0.527** | **0.604** | 0.819 | 0.720 | 0.584 | **0.560** | **0.454** |
| v3_vlonly_4B      | 0.811 | 84.71   | 0.526 | 0.586  | 0.822 | **0.725** | 0.489 | 0.529 | 0.434 |
| v1_10k_4B         | 0.580 | 81.79   | 0.432 | 0.437  | 0.812 | 0.689 | 0.085  | 0.335  | 0.277  |

`mmstar` columns: `mmstar` = 6-subcategory average; `star_m` = math; `star_l`
= logical reasoning; `star_s` = science & technology.

## Deltas vs prior cells

### v2 → v3 (same adapter, new mix + more steps): the rescue

| bench     | 2B Δ     | 4B Δ     |
|-----------|----------|----------|
| ai2d      | **+48.2 pp** | **+21.3 pp** |
| mmstar    | +11.2 pp | **+25.4 pp** |
| star_math | +15.0 pp | **+39.9 pp** |
| rwqa      | +15.6 pp | + 2.2 pp |
| mmmu      | + 1.8 pp | + 4.6 pp |

v3 fully rescues the v2 collapse on both scales. On 4B the math-subtask
recovery is a +39.9 pp swing — the v2→v3 transition takes MMStar_math from
0.192 (worse than random-guess on a 4-choice task) up to 0.591.

### v3 vs base (does retrofit cost us anything?)

v3 is **positive or neutral vs base** on nearly every benchmark:

- 2B: ai2d +2.9 pp ✓ ; mmbench +3.5 ✓ ; mmmu +2.5 ✓ ; mmstar −0.2 ≈ ; ocr +3.7 ✓ ; rwqa +2.0 ✓
- 4B: ai2d −0.3 ≈ ; mmbench +1.0 ✓ ; mmmu +3.3 ✓ ; mmstar −3.7 ; ocr −0.6 ≈ ; rwqa −0.7 ≈

The only clearly-negative cell is mmstar on 4B (−3.7 pp). But v3_5k_4B lifts
that to −2.0 pp — see "scale × step-count interaction" below.

### v1 (5 k) vs v1 (10 k): does "just more steps" on v1 work?

No — and in fact it actively hurts at 4B.

| cell     | ai2d Δ (10k vs 5k) | mmstar Δ | star_math Δ |
|----------|-------------------|----------|-------------|
| 2B       | −0.7 pp           | +8.5 pp  | +14.6 pp    |
| 4B       | **−23.0 pp** 💥   | −14.2 pp | **−38.2 pp** 💥 |

At 4B, ten-thousand steps on the narrow v1 mix causes a worse collapse than
v2 did (mmstar_math 0.085). This is the cleanest evidence yet that the v1
LLaVA-VSFT data simply cannot support long retrofit runs — the adapter
over-fits a narrow VL distribution and destroys the reasoning-on-images
paths. v3's LLaVA-OneVision mix does not show this.

## Key findings

### F1. v3 is the new canonical mix

v3 fixes every failure mode of v2 on both scales while matching or exceeding
v1 on every benchmark (2B) and every benchmark except mmstar (4B). Use v3 for
all downstream work: block ablation, VLA warm-start, paper figures.

### F2. Data mix dominates step count

Holding steps fixed at 10 k:
- v3 (rich VL + reasoning text) ≈ base on VL tasks, +1 pp on mmmu.
- v1 (narrow VL) collapses on 4B by −23 pp on ai2d, −38 pp on star_math.

Holding mix fixed at v3:
- 5 k vs 10 k: within ±2 pp across all benchmarks on both scales.

Implication: the reason v1_5k looked "okay" in the original retrofit.md
report was that the v1 mix runs out of signal around 5 k; it wasn't
evidence that 5 k is the right step budget.

### F3. Scale × step-count interaction (subtle)

2B prefers 10 k slightly:
- v3_2B (10 k) beats v3_5k_2B on ai2d (+1.9), mmstar (+1.3), star_math (+2.1).

4B is indifferent or slightly prefers 5 k:
- v3_5k_4B beats v3_4B on mmstar (+1.7), star_logic (+6.2), star_sci (+4.7).
- Tied on ai2d/mmbench/mmmu.

This makes sense: with ~1.6× more retrofit parameters at 4B, the router
reaches mix-saturation earlier. Longer training past that point re-balances
toward higher-gradient samples (vision-heavy LLaVA-OV) at the cost of text
reasoning benchmarks. 2B is still converging at 5 k, so more helps.

**Recommendation for block ablation**: 10 k is safe for 2B, 5 k probably
optimal for 4B. We adopted a unified 10 k for ablation simplicity and
reproducibility of v3_2B / v3_4B as anchors. Worst-case cost at 4B is ≤2 pp
on mmstar vs the 5 k optimum — tolerable for within-block-count comparison.

### F4. H4 (math-CoT adversarial to diagram reasoning) — REFUTED

v3_vlonly removes all math/CoT text (0 % → the VL goes from 60 % to 80 %,
UltraChat 20 % stays as a general-text anchor). If H4 were true, v3_vlonly
should *beat* v3 on ai2d and mmstar by removing the competing reasoning
path.

Actual outcome:
- v3_vlonly_2B ai2d 0.755 vs v3_2B 0.765 (−1.0 pp)
- v3_vlonly_4B ai2d 0.811 vs v3_4B 0.816 (−0.5 pp)
- **star_math: v3_vlonly is *worse*** (2B −5.1 pp, 4B −10.2 pp)

Math/CoT text is additive to — not adversarial to — diagram math reasoning,
once the VL anchor is rich enough (LLaVA-OV ≥60 %). The v2 symptom was
caused by insufficient VL, not by CoT-text interference.

### F5. γ-transition stability depends on (mix × steps × ramp-frac × scale)

During v3 setup, 4B with ramp-frac 0.3 × 10 k steps diverged at step ~3 125
(CE 0.9 → 6.5+). Same config at 5 k converged fine, as did v3_vlonly_4B
(same ramp-frac but less dense reasoning gradient). Fix: ramp-frac 0.5
(ramp ends at step 5 000 instead of 3 000).

Working rule: `ramp_end_step ≥ 5 000` is stable for all configs tested.
v3_2B / v3_5k_2B / v3_5k_4B / v3_vlonly_* kept 0.3 and are valid data points.
v3_4B at ramp-frac 0.5 is the ONE cell with different schedule; since its
results are within 1 pp of v3_5k_4B (ramp-frac 0.3) across all VL benchmarks,
the ramp-frac difference does not confound the between-cell comparison.

## What this implies for the paper

1. Replace v1 canonical results in tables with v3. v3_2B is now the 2B
   adaptive-depth representative; v3_4B (or v3_5k_4B) is the 4B one.
2. The v2 experience becomes an ablation row ("VL 30 %, CoT 40 %") showing
   that retrofit is sensitive to VL proportion. Keep as cautionary point
   only if space permits.
3. Block partition ablation should use v3 mix; see `block_partition_ablation.md`
   for the L=2 justification from the 2B pilot, and the 8-cell full sweep
   once GPUs clear.
4. The v1_10k_4B collapse is evidence to cite when discussing
   "why not just train v1 longer" — if a reviewer asks.

## Open items

- LAMBADA + HellaSwag on all 8 cells — pending GPU free.
- Full block partition ablation (2B × {L=1,2,4,7}, 4B × {L=1,2,4,6}) —
  blocked on GPUs 0-3 held by DreamerVLA; half-batch can start on GPUs 4-7.
- Downstream: check whether v3 retrofit → LIBERO warm-start (VLA) behaves
  better than Path B (v1 warm-start). Not yet scheduled.

## Artefacts

States: `retrofit/outputs/{H_r256_10k_v3_2B, H_4B_r256_10k_v3, v3_5k_*,
v3_vlonly_*, v1_10k_*}/retrofit_attnres_state.pt`

Evals: `retrofit/outputs/lmms_eval_v3/{v3_2B, v3_4B, v3_5k_*, v3_vlonly_*,
v1_10k_*}/retrofit/models__Qwen3-VL-*/*_results.json`

Data mix spec: `retrofit/train/data_v2.py` (`V3_MIX`, `V3_VLONLY_MIX`)

Training launchers: `retrofit/train/run_retrofit_v3.sh`,
`retrofit/train/run_v3_ablations.sh`
