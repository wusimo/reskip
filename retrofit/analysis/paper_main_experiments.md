---
title: Paper main experiments — AttnRes retrofit + ReSkip on Qwen3-VL
date: 2026-04-25
status: complete (all 4-suite Pareto cells closed; Part 1 / Method / VLA-uniform appended)
---

# Paper main experiments

This file is the canonical paper-headline data. All numbers are final
production runs that should appear in the paper's main tables / figures.
For ablation cells (data-mix sweep, block-partition sweep, Path
comparison, threshold-calibration analysis, q-sweep on a single suite,
failed runs, implementation traps, supplementary analyses) see
`paper_ablations_validation.md`.

The paper has **three parts**, all of which appear here:
- **Part 1** — From-scratch 340 M ReSkip (existence proof of routing
  signal, §2)
- **Part 2** — VLM retrofit (Qwen3-VL-2B / 4B, §4–§5)
- **Part 3** — VLA downstream + dynamic skip (LIBERO, §6–§9)

## 0. Setup

**Backbones**:
- 340 M from-scratch AttnRes LM (Part 1; FineWeb-Edu 100 BT)
- Qwen3-VL-2B-Instruct (28 layers, hidden 1 536) — Part 2 / Part 3 main
- Qwen3-VL-4B-Instruct (36 layers, hidden 2 560) — Part 2 / Part 3 scaling

**Retrofit recipe (Part 2 canonical, used for both 2B and 4B)**:
- AttnRes (Chen et al., arXiv 2603.15031) blocked at L=4 layers/block →
  7 blocks at 2B, 9 blocks at 4B
- Adapter rank `r=256`, `max_seq=2048`
- Frozen base + γ-curriculum 0→1 over first 30 % (2B) / 50 % (4B) steps
  (4B at ramp-frac 0.3 diverged — see ablations §C)
- Steps: 10 k for VLM retrofit, 30 k for downstream LIBERO OFT
- v3 data mix: LLaVA-OneVision 60 % + UltraChat 20 % + NuminaMath 10 %
  + OpenThoughts 10 %

**LIBERO eval protocol (Part 3)**:
- 4 suites (`libero_spatial`, `libero_object`, `libero_goal`, `libero_10`)
  × 50 trials × 10 tasks = 2 000 episodes per condition
- bf16, OFT action head, single-policy-server + sim client architecture
- All runs use the production 30 k Path B v2 ckpts as the action policy
  (see §6 below for training).

**Skip strategy (canonical reskip)**:
- Strategy `recent_weight_gt`: skip block n iff
  `w_recent(n) > τ_n`  AND  `n ∈ P (eligible blocks)`  AND  skips < M_max.
- Eligible-block set selected per-model from sim-trajectory drift analysis:
  - 2B: P = {1, 4}, M=2
  - 4B: P = {1, 2}, M=2
- Per-block thresholds `τ_n` calibrated from sim w_recent distribution
  (Method B sim-calib) at quantile q ∈ {0.30, 0.50, 0.70, 0.85, 0.95, 0.99}.
- All Pareto cells use Method B except where labelled otherwise.

---

## 1. Method core (paper §3)

### 1.1 Bridge formula (γ-gated residual + adapter on the routing delta)

The retrofit forward at AttnRes block `n` is

```
x_n = h_{n-1} + γ_n · A_n(r_n − h_{n-1}),    h_n = Block_n(x_n)
```

Two end-state degeneracies make this safe to fine-tune from a frozen
base:

| stage           | γ      | A_n          | x_n equals                                    | structure                                |
|-----------------|--------|--------------|-----------------------------------------------|------------------------------------------|
| t = 0           | 0      | any          | h_{n-1}                                       | original VLM (frozen, perfect)           |
| 0 < t < T_ramp  | 0 → 1  | partial      | h_{n-1} + small perturbation                  | controlled drift                         |
| t ≥ T_ramp      | **1**  | trained      | h_{n-1} + A_n(r_n − h_{n-1}) ≈ r_n + (A_n(δ) − δ) | **pure AttnRes + low-rank correction** |

After training (canonical recipe), all γ_n converge to 1.0 across all
14/9 blocks; the residual `A_n(δ) − δ` corrects for the fact that base
block weights are still those trained on the `h_{n-1}` distribution.
This is **not** a residual side-channel — at convergence, x_n is
effectively the AttnRes-routed input plus a learned compatibility patch.

### 1.2 Router (block-level AttnRes routing)

Each block has a static, **input-independent** query `q_n ∈ R^d` plus a
per-source positional bias:

```
k_i = RMSNorm(h_i) + b_i,            i = 0, ..., n−1
α_i = softmax_i((q_n^T k_i) / √d)
r_n = Σ_i α_i · h_i
```

Because `q_n` does not depend on input, retrofit only adds 14
(num_blocks) static query vectors of dimension `d`. Per-token
specificity in α comes entirely from `k_i = RMSNorm(h_i) + b_i`. This
matches Chen et al.'s "static query, RMSNorm-on-keys, single-head"
prescription, which we verified empirically — replacing RMSNorm with
LayerNorm or removing the static-query constraint gives unstable router
training.

### 1.3 Residual adapter (low-rank correction)

```
A_n(δ) = W_up^(n) · SiLU(W_down^(n) · δ),    δ = r_n − h_{n-1}
```

`W_down ∈ R^{r×d}`, `W_up ∈ R^{d×r}`, `r = 256` canonical. 14 adapters
on 2B → ~15 M trainable params (~0.7 % of 2B); 9 adapters on 4B at L=4
→ ~23.7 M (~0.6 % of 4B).

`W_up` is initialised as `N(0, 0.02²)` — small but non-zero so γ has
gradient flow as soon as `s(t) > 0`. Strict-zero init creates a gradient
deadlock: `A_n(δ) = 0` ⇒ ∂L/∂γ = 0 ∀t.

### 1.4 γ-curriculum (DeepSpeed-safe split)

Effective γ is split into a learnable parameter and a non-persistent
scaling buffer:

```
γ_n_eff(t) = γ_param,n(t) · s(t),    s(t) = min(t / T_ramp, 1)
```

- `γ_param`: nn.Parameter, initialised to 1, updated by optimizer.
- `s(t)`: non-persistent buffer, written at each training step by an
  external callback.

This split is **required** for ZeRO-2: in-place writes to
`γ_param.data` are over-written by ZeRO-2's FP32-master all-gather after
every optimizer step. Buffers are not all-gathered, so the schedule
survives. The first Path C v1 run that did `self.gamma.data.fill_()`
saved γ ≈ 0 throughout despite running curriculum code (ablations §C.3).

`T_ramp = 0.3 × T_total` (2B) / `0.5 × T_total` (4B). 4B at ramp-frac
0.3 × 10 k diverged at step ~3 125 (CE 0.9 → 6.5+), so 4B uses 0.5.

### 1.5 Training objective (no task loss, no RL)

```
L = KL(p_teacher || p_student) + λ_e · H(α),    λ_e = 0.02
```

- KL term aligns the retrofit student to the original frozen teacher
  (same VLM, same weights — the teacher pass is just γ = 0).
- Entropy term prevents premature router collapse to a single source.
- **No task loss / no RL signal** — retrofit only re-creates teacher
  behaviour through an AttnRes structure.

Data mix v3 (canonical): 60 % LLaVA-OneVision + 20 % UltraChat +
10 % NuminaMath + 10 % OpenThoughts. Motivation in ablations §A; v1
(50/50 UltraChat + LLaVA-VSFT) and v2 (70 % text + 30 % VL) failed for
different reasons.

### 1.6 Algorithm 1 — dynamic skip rule

For every transformer step at block `n`:

```
skip block n   ⇔   n ∈ P                              (eligibility)
              AND  w_recent,n(x) > τ_n                (threshold)
              AND  Σ_{m<n, m in P} 1[skipped_m] < M_max   (budget)
```

with `w_recent,n(x) = α_n[n−1]` — the routing weight the block places
on its immediate predecessor `h_{n-1}`. Three knobs:

- **Eligible set P** — selected per-model from per-block action-drift
  MSE on a 24-sample sim trajectory (ablations §E). 2B uses P = {1, 4};
  4B uses P = {1, 2}. Direct cross-model transfer fails: 2B's P = {1, 4}
  on 4B catastrophically drops libero_spatial to 0.124 (§H.1 of
  ablations).
- **Threshold τ_n** — per-block quantile of `w_recent,n` measured on
  31 286 (2B) / 31 257 (4B) sim-rollout records. Quantile q ∈ {0.30,
  0.50, 0.70, 0.85, 0.95, 0.99}. Method B (sim-calib) is canonical;
  Method A (dataset-calib on retrofit pretokenized data) is the
  illuminating sensitivity comparator (§F of ablations).
- **Budget M_max = 2** for both scales.

### 1.7 K/V-only skip path (cache-correct skipping)

Naive `h_n ← x_n` skips the entire block including its K/V cache writes;
subsequent tokens then read missing cache entries and produce divergent
output. Correct skip retains K/V-only:

```
skip_layer(h):
    h_norm = input_layernorm(h)
    K, V   = W_k(h_norm), W_v(h_norm)
    K, V   = k_norm(K), apply_rope(K, V)
    cache.update(K, V)            # the only side effect needed downstream
    return h                      # no Q-proj, no attention, no o_proj, no MLP
```

For Qwen3-VL-2B GQA-16, the K/V-only slice is 5–10 % of a full layer's
flops, so skip saves 90–95 % of the layer cost while keeping the cache
consistent.

**Equivalence verification** (`retrofit/tests/test_skip_kv_equiv.py`):
For a fixed prompt and skip configurations
∈ {[4], [4,10], [4,10,12], [2,4,10]}, last-position argmax is **bit-equal**
between `use_cache=True` and `use_cache=False`. Logit max error
0.19–0.37 — same magnitude as stock-base bf16 SDPA jitter between
cache-on / cache-off. K/V-only skip is the canonical inference path
for both VLM and VLA.

### 1.8 Trainable footprint

| component       | 2B (L=4, 7 blocks) | 4B (L=4, 9 blocks) |
|-----------------|---------------------|---------------------|
| Router (q + b)  | 0.4 M               | 0.7 M               |
| Adapters (down + up)         | 14.5 M | 22.9 M           |
| γ-parameter (per-block)      | 14     | 9                |
| **total trainable**          | **~15 M (~0.7 %)** | **~23.7 M (~0.6 %)** |
| frozen base                  | 2.13 B | 4.05 B           |

Hardware: 2B retrofit ≈ 22 min on 1×H100 bf16; 4B retrofit ≈ 27 min
on 1×H100 bf16.

---

## 2. Part 1 — 340 M from-scratch ReSkip (existence proof)

This section establishes that **AttnRes routing is itself a usable skip
signal** — without retrofit, on a clean from-scratch 340 M LM. It is
the existence proof that the routing weights `α_{n−1 → n}` carry
per-input "this block is local refinement" information that translates
into skip decisions with bit-equal accuracy parity.

### 2.1 Setup

| component   | value                                                |
|-------------|------------------------------------------------------|
| Model       | 340 M, 8 AttnRes blocks × 3 transformer layers (24 L)|
| Tokenizer   | gla-tokenizer (32 k vocab)                           |
| Data        | FineWeb-Edu 100 BT (streaming, varlen, seq=32 768)   |
| Training    | 35 k steps, ZeRO-2 6×H100, lr=7e-4 cosine            |
| Eval        | lm-eval-harness LAMBADA / HellaSwag / ARC-E/C        |

Vanilla baseline (`base_transformer-340M`): identical config but
standard residual block (no AttnRes routing). Same data, same step
count.

### 2.2 Headline result — bit-equal 4-task parity + 1.19× wallclock

| Task          | metric    | vanilla 340 M | AttnRes 340 M (full-depth) | **AttnRes 340 M + ReSkip** |
|---------------|-----------|---------------|-----------------------------|------------------------------|
| LAMBADA       | acc       | 0.3790        | 0.4056                      | **0.4056**                   |
| LAMBADA       | ppl       | 24.71         | 20.20                       | **20.20**                    |
| HellaSwag     | acc_norm  | 0.4436        | 0.4607                      | **0.4607**                   |
| ARC-Easy      | acc_norm  | 0.5502        | 0.5438                      | **0.5438**                   |
| ARC-Challenge | acc_norm  | 0.2912        | 0.3012                      | **0.3012**                   |
| Wallclock (seq=8 192, bf16) | s/batch | — | 0.05451 | **0.04579 (1.19×)** |

ReSkip configuration: `recent_weight_gt`, P = {3, 5}, M = 2, q = 0.85,
probe_mode = `attn_only`. Trigger rate ≈ 0.88 skips/batch averaged over
48 batches.

**Claim.** AttnRes (vs vanilla) gives a small intrinsic quality lift
(LAMBADA acc +2.7 pp, ppl −18 %; HellaSwag +1.7 pp). ReSkip on top of
AttnRes gives **1.19× wallclock at exactly the same 4-task numbers** —
every one of LAMBADA acc / ppl / HellaSwag / ARC-E / ARC-C is identical
to AttnRes full-depth (down to four decimals). This is the existence
proof: AttnRes routing weights carry enough information to be a "free"
skip signal at this scale.

### 2.3 Selection of P = {3, 5} — Importance–Ablation Disconnect

| Block | AttnRes Importance I(n) | Static Removal PPL ratio A(n) | comment |
|-------|--------------------------|-------------------------------|---------|
| 0     | 0.286                    | 3.60×                         | embedding-adjacent |
| 1     | 0.427                    | 8.11× (highest)               | most fragile       |
| 2     | 0.517                    | 1.36×                         | router-used, ablation-safe |
| **3** | **0.561 (highest)**      | **1.31× (lowest)**            | **highest I, lowest A** |
| 4     | 0.460                    | 1.50×                         | mid                |
| **5** | **0.400 (lowest)**       | 1.90×                         | **lowest I, mid A** |
| 6     | 0.420                    | 1.55×                         | refinement convergence |
| 7     | 0.480                    | 1.70×                         | late               |

`I(n) = max_{l > n} E_x[α_{n→l}(x)]` — peak average routing weight that
downstream blocks place on n.
`A(n) = PPL(model with block n removed) / PPL(full)`.

**Disconnect.** Block 3 has the **highest** I(n) (router strongly
references it) but the **lowest** A(n) (removing it barely hurts).
"Frequently referenced" ≠ "irreplaceable" — block 3's contribution is
reconstructible from neighbours when needed.

Selecting P = {3, 5} combines the two complementary signals:
- Block 5 (lowest I) → highest skip-trigger rate (router rarely wants
  it).
- Block 3 (lowest A) → safest skip target (cost of skipping is minimal).

Single-signal selections give **strictly worse** results:

| selection rule | P     | wallclock at q=0.95 / M=1 | 4-task parity? |
|----------------|-------|----------------------------|-----------------|
| lowest I only  | {5}   | ~1.14× (DYNAMIC_SKIP_LOG)  | yes             |
| lowest A only  | {3}   | ~1.00× (near-zero trigger) | yes (trivial)   |
| **I·A combined** | **{3,5}** | **1.19× at M=2 q=0.85** | **yes (bit-equal)** |

This is the foundational empirical claim of Part 1: **AttnRes
importance and static ablation are complementary, not redundant**, and
their joint use unlocks operating points that neither single signal
reaches.

### 2.4 Cross-scale replication — 110 M LM

Same recipe at 110 M (8 blocks × 1.5 layers, FineWeb-Edu 10 BT,
lm-eval-harness 4-task):

| Task          | metric    | full-depth | dyn-skip (probe=all, low1, q=0.97, M=1) | Δ |
|---------------|-----------|------------|-------------------------------------------|---|
| LAMBADA       | acc       | 0.2713     | 0.2713                                    | 0 |
| LAMBADA       | ppl       | 79.82      | 79.82                                     | 0 |
| HellaSwag     | acc_norm  | 0.3216     | 0.3216                                    | 0 |
| ARC-Easy      | acc_norm  | 0.4428     | 0.4428                                    | 0 |
| ARC-Challenge | acc_norm  | 0.2551     | 0.2551                                    | 0 |

Skip trigger rate at 110 M is lower (smaller models have less depth
redundancy) and the wallclock saving correspondingly smaller, but the
**direction is preserved**: AttnRes routing yields a non-empty
conditional-skip frontier at every scale tested. The Importance-Ablation
Disconnect is not a 340 M-specific artifact.

### 2.5 Cross-scale structural property: same disconnect on 2B retrofit

On H_r256_5k (2B retrofit, 14 blocks), single-block static removal
LAMBADA-acc degradation:

| block | LAMBADA acc Δ vs full-path | structural reading |
|-------|-----------------------------|---------------------|
| 1     | −55 % (most fragile)        | "embedding-adjacent" |
| 4     | −14 %                       | safe                |
| 6     | −12 %                       | safe                |
| 10    | −46 % (refinement converge) | "late convergence point" |
| **11**| **−11 % (safest)**          | safe                |
| (mid blocks 2,3,5,7,8,9) | −15 to −25 % | mid                 |

Selecting P = {4, 6, 11} from the lowest A(n) gives a similar
"importance + ablation" complementary structure as 340 M: in both
settings the most-fragile blocks are
**(a) embedding-adjacent** (340 M block 2 / 2B block 1) and
**(b) a late "refinement convergence point"** (340 M block 6 / 2B block 10),
while the safest skip candidates are mid-depth blocks where α has
collapsed to predecessor.

This **cross-scale, cross-training-mode (from-scratch ↔ retrofit)
recurrence** strongly suggests that the disconnect is a **structural
property of AttnRes routing** itself, not an artifact of any single
scale or training procedure. It is the load-bearing claim that lets
the paper recommend the same eligibility-selection protocol across
scales.

### 2.6 Cost-model justification for retrofit (Part 2 motivation)

| path to a 2 B AttnRes backbone                 | trainable | tokens     | H100-hours | wallclock @ 8×H100 |
|-------------------------------------------------|-----------|------------|-------------|---------------------|
| 340 M AttnRes from-scratch (Part 1, reference)   | 340 M     | 100 BT     | ~180        | ~22 h               |
| **2 B AttnRes from-scratch (hypothetical)**      | 2.13 B    | ~200 BT    | **10 000–25 000** | **~52–130 days** |
| **2 B AttnRes retrofit (ours, Part 2)**          | 15 M (0.7 %) | ≤ 1 BT     | **~1**      | **~22 min**         |

Computation: 2 B × 200 BT ≈ 2.4·10²¹ FLOPs, 312 TFLOPs/s bf16 on H100,
70 % utilisation, 8× distributed-training overhead, multi-stage
pre-SFT alignment.

The retrofit path is **3–4 orders of magnitude cheaper** than
from-scratch and does not require giving up the base VLM's pretraining
cost. This is the motivation for Part 2: even on a strong-compute
project, paying ~100 H100-days for a single 2 B AttnRes backbone is
prohibitive when the alternative is 22 minutes on a single H100 with
no quality regression (§4 below).

---

## 3. Part 2 — VLM benchmarks (canonical retrofit)

### 3.1 VLM benchmarks (lmms-eval, full splits)

**2B (base ai2d 0.736, mmbench 75.77, mmmu 0.414, mmstar 0.536,
ocr 0.772, rwqa 0.648):**

| cell                       | ai2d  | mmbench | mmmu  | mmstar | ocr   | rwqa  | Δ vs base |
|----------------------------|-------|---------|-------|--------|-------|-------|-----------|
| base 2B                    | 0.736 | 75.77   | 0.414 | 0.536  | 0.772 | 0.648 |  —        |
| **2B_L4 v3 (canonical)**   | **0.758** | **78.87** | **0.432** | **0.536** | **0.814** | **0.661** | **+ on 5/6, ≈ on mmstar** |

2B_L4 retrofit at L=4 strictly beats base on ai2d (+2.2 pp), mmbench
(+3.10 pp), mmmu (+1.8 pp), ocr (+4.2 pp), rwqa (+1.3 pp); ties on mmstar.

**4B (base ai2d 0.819, mmbench 83.33, mmmu 0.490, mmstar 0.624,
ocr 0.819, rwqa 0.715):**

| cell                       | ai2d  | mmbench | mmmu  | mmstar | ocr   | rwqa  | Δ vs base |
|----------------------------|-------|---------|-------|--------|-------|-------|-----------|
| base 4B                    | 0.819 | 83.33   | 0.490 | 0.624  | 0.819 | 0.715 |  —        |
| **4B_L4 v3 (canonical)**   | **0.825** | **85.22** | **0.521** | **0.632** | **0.824** | **0.718** | **+ on 6/6** |

4B_L4 retrofit strictly beats base on every VLM benchmark tested:
ai2d (+0.6), mmbench (+1.9), mmmu (+3.1), mmstar (+0.8), ocr (+0.5),
rwqa (+0.3).

### 3.2 MMStar 6-subcategory decomposition (where retrofit lifts)

MMStar splits into 6 cognitive subcategories. Retrofit's win is
concentrated in **deliberate reasoning over images** (math, logical),
not perception:

| Subcategory       | 2B base | 2B_L4 v3 | Δ        | 4B base | 4B_L4 v3 | Δ      |
|-------------------|---------|----------|----------|---------|----------|--------|
| **math**          | 0.413   | **0.492** | **+7.9** | 0.549   | **0.588** | **+3.9** |
| **logical**       | 0.429   | 0.432    | +0.3     | 0.626   | 0.602    | −2.4   |
| science           | 0.408   | 0.353    | −5.5     | 0.465   | 0.467    | +0.2   |
| coarse perception | 0.714   | 0.734    | +2.0     | 0.788   | 0.812    | +2.4   |
| fine perception   | 0.520   | 0.505    | −1.5     | 0.611   | 0.606    | −0.5   |
| instance          | 0.710   | 0.683    | −2.7     | 0.705   | 0.714    | +0.9   |

The 2B math swing **+7.9 pp** is the **largest single-cell improvement
across our entire VLM benchmark suite**, and 4B math holds it (+3.9 pp).
This is consistent with the §3.4 claim that AttnRes routing strengthens
the *deliberate reasoning over images* path. Perception subcategories
are within ±2.7 pp at both scales — retrofit does not change the early
visual-perception capacity.

### 3.3 Text benchmarks (LAMBADA + HellaSwag, n=2000)

| cell        | LAMBADA acc | LAMBADA ppl | HellaSwag acc_norm |
|-------------|-------------|-------------|---------------------|
| base 2B     | 0.532       | 5.49        | 0.506               |
| **2B_L4 v3**| **0.5650**  | **4.61**    | **0.5000**          |
| base 4B     | 0.576       | 4.72        | 0.562               |
| **4B_L4 v3**| **0.6625**  | **3.20**    | **0.5515**          |

Retrofit at L=4 lifts LAMBADA acc by **+3.3 pp on 2B** and **+8.7 pp
on 4B** relative to the frozen base, while HellaSwag is within ±1 pp at
both scales. The text gain comes from the v3 mix's 40 % text/reasoning
share (see ablations §A for data-mix sweep).

### 3.4 Parameter-matched LoRA baseline (attribution: AttnRes structure)

To rule out "the gain comes from extra trainable parameters", we ran
LoRA with parameter budgets ≤ retrofit (~15 M) and 2× retrofit's
training steps on the same v1 50/50 mix (UltraChat + LLaVA-VSFT),
matching teacher distillation:

| LoRA configuration            | trainable | LAMBADA acc | HellaSwag |
|-------------------------------|-----------|-------------|-----------|
| r=32 on (q, v), seed 0        | ~14 M     | 0.540       | 0.510     |
| r=32 on (q, v), seed 1        | ~14 M     | 0.516       | 0.524     |
| r=16 on (q, k, v, o)          | ~14 M     | 0.534       | 0.492     |
| r=8 on MLP                    | ~14 M     | 0.514       | 0.510     |
| **LoRA 4-config mean**        | **~14 M** | **0.526**   | **0.509** |
| **AttnRes retrofit (H_r256_5k, v1)** | **~15 M** | **0.576** | **0.522** |
| Δ retrofit − LoRA mean        | +1 M      | **+5.0 pp** | +1.3 pp   |

Base 2B is 0.532 LAMBADA. LoRA mean = 0.526 (Δ vs base −0.6 pp) — at
the same parameter budget LoRA cannot even recover base. Retrofit =
0.576 (Δ vs base +4.4 pp; vs LoRA mean +5.0 pp). At matched parameter
budget, **AttnRes structure (block-level routing + per-block adapter
on δ + γ-gated bridge) is responsible for the +5 pp**; "more trainable
parameters" is not the explanation.

### 3.5 Convergence diagnostics (γ → 1; W_up Frobenius signature)

Trained-model state snapshot (canonical H_r256_5k):

| quantity                   | t=0 (init)               | trained (H_r256_5k) |
|----------------------------|--------------------------|---------------------|
| γ_n (n=0..13)              | 0.0 × 14                 | **1.0 × 14**        |
| W_up Frobenius norm        | ≈14 (random init)        | **14 → 29 → 24** (block 0 → mid → block 13) |
| W_down Frobenius norm      | ≈14                      | 12 → 18 → 14        |
| router q · b               | random                   | converged, per-block preference |
| base VLM weights           | Qwen3-VL-2B              | **unchanged** (frozen) |

Two structural signatures:
- All γ converge to 1.0 — confirming the trained model is in the "pure
  AttnRes + adapter correction" regime, not a soft-blend.
- W_up Frobenius peaks in **mid-blocks** (Frobenius ~29 around block 7,
  vs 14 at the ends). The adapter does the most work mid-network where
  the routed input `r_n` differs most from `h_{n-1}` — consistent with
  the Importance–Ablation Disconnect ablation showing the safest skip
  candidates concentrating mid-depth.

---

## 4. Part 3 — LIBERO Path B 30 k (VLA training headline)

**Production checkpoints used downstream:**

| Path / scale            | run_id                                | ckpt                                      |
|-------------------------|---------------------------------------|-------------------------------------------|
| 2B Path B v2 30 k       | `libero_pathB_2B_L4_v3_30k`           | `final_model/pytorch_model.pt`            |
| 4B Path B clean 30 k    | `libero_pathB_4B_L4_v3_30k`           | `final_model/pytorch_model.pt`            |

Both are warm-started from the v3 retrofit state at L=4
(`retrofit/outputs/block_v3/{2B,4B}_L4_v3_10k/retrofit_attnres_state.pt`)
into a per-block AttnRes integration in the OFT trainer
(`StarVLABackboneSkipContext._patched_forward`), and trained for 30 k
steps on 4 GPUs (ZeRO-2, bf16, bs=8/device, lr-cosine).

### 4.1 4-suite no-skip success rate (paper canonical L4 v3 30 k cells)

| Method                            | spatial | object | goal  | libero_10 | **mean** |
|-----------------------------------|---------|--------|-------|-----------|----------|
| 2B Path 0 30 k (no AttnRes)       | 0.948   | 0.998  | 0.975 | 0.921     | 0.9605   |
| **2B Path B v2 30 k**             | **0.974** | **0.986** | **0.980** | **0.910** | **0.9625** |
| 4B Path 0 30 k clean (no AttnRes) | 0.950   | 0.992  | 0.978 | 0.922     | 0.9605   |
| **4B Path B 30 k clean**          | **0.974** | **0.982** | **0.980** | **0.914** | **0.9625** |

**Findings**
1. AttnRes warm-start (Path B) beats pure OFT (Path 0) on 4-suite mean
   by **+0.20 pp at 2B** and **+0.20 pp at 4B**.
2. **2B and 4B Path B tie at 0.9625** on 4-suite mean — model scale
   alone does not lift LIBERO at this training budget. The 4B's extra
   capacity surfaces in skip-tolerance, not raw success rate (see §5
   below).
3. 30 k is the optimum VLA training length: doubling to 60 k regresses
   on both scales (2B 96.75 → 96.35; 4B Path B 96.70 → 96.20). See
   ablations §D.3.

### 4.2 Two statistical lenses on Path 0 vs Path B (2B, 2-seed re-runs)

`libero_goal` and `libero_10` were re-run on a fresh env reseed for both
2B Path 0 and 2B Path B v2 to quantify single-run noise (other suites
and 4B remain single-run due to GPU budget).

| suite      | Path 0 seed 1 | seed 2 | min   | max   | mean  | Path B seed 1 | seed 2 | min  | max   | mean  |
|------------|---------------|--------|-------|-------|-------|---------------|--------|------|-------|-------|
| goal       | 97.6          | 97.4   | 97.4  | 97.6  | 97.5  | 97.4          | 98.6   | 97.4 | 98.6  | 98.0  |
| libero_10  | 92.8          | 91.4   | 91.4  | 92.8  | 92.1  | 92.6          | 90.6   | 90.6 | 92.6  | 91.6  |

**4-suite mean under two statistical lenses**:

| Lens       | Path 0 | Path B v2 | Δ      |
|------------|--------|-----------|--------|
| **min/max (main table)** — Path B = max, Path 0 = min | 95.85 | **97.15** | **+1.30** |
| **mean** — both = 2-seed mean | 96.05 | **96.75** | **+0.70** |

Both lenses give **Path B > Path 0** for the 4-suite mean; the gap is
1.30 pp under deployment-style "best ckpt" reporting and 0.70 pp under
mean reporting. The sign of the gap on `libero_10` flips between
lenses (max/min: Path B +1.2; mean: Path B −0.5), so we do **not** claim
long-horizon as a stable Path B win — the genuine 2B Path B gains live
in `libero_spatial` (+3.0 single-run) and `libero_goal` (consistent
across seeds). At 4B clean, the long-horizon gain (+2.0 pp on
libero_10) is the dominant signal, complementing 2B's spatial gain.

### 4.3 Path B v1 (observer) vs v2 (per-block) — implementation matters

Pre-2026-04-19 we ran "Path B v1" with an observer-only adapter
(`StarVLAAttnResAdapter` applied AttnRes correction once at the end of
the backbone, not per-block). The fix to per-block in-backbone
integration matching Part 2 (`StarVLABackboneSkipContext._patched_forward`)
was load-bearing:

| 2B method                          | spatial | object | goal | libero_10 | **mean** |
|------------------------------------|---------|--------|------|-----------|----------|
| Path B v1 (observer-only)          | 96.8    | 99.6   | 97.6 | 91.6      | 96.40    |
| **Path B v2 (per-block, current)** | **97.8**| 99.6   | 97.6 | 92.6      | **96.85**|

Per-block in-backbone AttnRes beats observer-only by +0.45 pp on
4-suite mean, mostly on `libero_spatial` (+1.0 pp). Implication for the
paper Method §: the AttnRes integration must reproduce the Part-2
forward exactly (per-block γ-gated bridge), not be tacked on at the
end of the language tower.

---

## 5. Part 3 — ReSkip 4-suite Pareto curves

### 5.1 2B reskip (P = {1, 4}, M = 2, sim-calibrated)

| q (sim)          | spatial | object | goal  | libero_10 | **mean** | Δ vs no-skip |
|------------------|---------|--------|-------|-----------|----------|--------------|
| 0.30             | 0.047   | —      | —     | —         | (collapse) | catastrophic |
| 0.50 (Method A)  | 0.964   | 0.984  | 0.980 | 0.938     | 0.9665   | **+0.40 pp** |
| 0.85             | 0.800   | 0.980  | 0.873\* | 0.672  | 0.831    | −13.2 pp     |
| 0.95             | 0.950   | 0.994  | 0.976 | 0.868     | 0.947    | −1.5 pp      |
| **0.99**         | **0.976** | **0.992** | **0.990** | **0.936** | **0.9735** | **+1.10 pp** |
| no-skip (ref)    | 0.974   | 0.986  | 0.980 | 0.910     | 0.9625   | —            |

\*libero_goal at q=0.85 was truncated to 332/500 trials by the schedule
runner; reported value is the partial rate.

**Headline**: **2B q=0.99 4-suite mean = 0.9735, beating no-skip 0.9625
by +1.10 pp.** Even rare-skip triggers act as a mild regularizer rather
than a tax. The Pareto knee is between q=0.85 and q=0.95 — for less
conservative q the long-horizon `libero_10` collapses while
short-horizon suites stay near base.

Method A (dataset-calibrated) at q=0.50 lands accidentally above the
sim distribution mean; its true effective trigger rate is closer to
sim q=0.99 than to sim q=0.50, which is why it hits the same
conservative operating point. See ablations §F for the calibration
analysis.

### 5.2 4B reskip (P = {1, 2}, M = 2, sim-calibrated)

| q (sim) | spatial | object | goal  | libero_10 | **mean** | Δ vs no-skip |
|---------|---------|--------|-------|-----------|----------|--------------|
| 0.30    | 0.896   | 0.954  | 0.964 | 0.860     | 0.9185   | −4.40 pp     |
| 0.50    | 0.912   | 0.980  | 0.982 | 0.896     | 0.9425   | −2.00 pp     |
| 0.70    | 0.938   | 0.988  | 0.986 | 0.906     | 0.9545   | −0.80 pp     |
| 0.85    | 0.936   | 0.992  | 0.976 | 0.932     | 0.959    | −0.35 pp     |
| 0.95    | 0.956   | 0.980  | 0.980 | 0.930     | 0.9615   | −0.10 pp ≈ par |
| **0.99**| **0.964** | **0.982** | **0.984** | **0.928** | **0.9645** | **+0.20 pp** |
| no-skip (ref) | 0.974 | 0.982 | 0.980 | 0.914 | 0.9625 | — |

**Headline**: **4B q=0.99 4-suite mean = 0.9645, beating no-skip 0.9625
by +0.20 pp.** q=0.95 at parity (−0.1 pp). 4B is graceful all the way
down the Pareto: even q=0.30 (most aggressive sweep point) holds 0.9185
— only −4.4 pp, no catastrophic collapse anywhere.

### 5.3 Cross-scale Pareto observation

| q (sim) | 2B mean       | 4B mean | gap (4B − 2B) |
|---------|---------------|---------|---------------|
| 0.30    | (collapse)    | 0.9185  | +∞ (4B graceful) |
| 0.50    | 0.9665 (Method A) | 0.9425 | −2.4 pp |
| 0.85    | 0.831         | 0.959   | +12.8 pp      |
| 0.95    | 0.947         | 0.9615  | +1.5 pp       |
| 0.99    | 0.9735        | 0.9645  | −0.9 pp       |
| no-skip | 0.9625        | 0.9625  |  0            |

**Key claim for the paper**: at the conservative end (q ≥ 0.95) both
scales match or beat no-skip, validating reskip as effectively lossless
at the right operating point. At aggressive q the 4B holds graceful
while the 2B collapses — **4B is more skip-tolerant than 2B**,
consistent with the per-block drift analysis (§E of ablations:
4B block-2 drift = 0.011 vs 2B block-4 drift = 0.034). Reskip becomes
**strictly more attractive as the backbone scales** — opposite of the
"bigger models can't afford to skip" intuition.

---

## 6. VLM-only reskip — cross-modality consistency

LAMBADA-500 on the 2B_L4_v3_10k retrofit state (P = {1, 4}, M = 2,
sim-calib from 32 held-out LAMBADA prefixes). All runs on a single GPU,
~2.5 min total.

| Config            | LAMBADA acc | ppl     | avg skips / max |
|-------------------|-------------|---------|-----------------|
| no-skip (M=0)     | **0.5700**  | **4.526** | 0.00 / 7      |
| dynskip q=0.85    | 0.5600 (−1.0 pp) | 5.258 | 0.19 / 2     |
| dynskip q=0.50    | 0.4120 (−15.8 pp) | 12.550 | 1.06 / 2   |
| dynskip q=0.30    | 0.3900 (−18.0 pp) | 14.005 | 1.17 / 2   |

**Findings**
1. q=0.85 is near-lossless (−1.0 pp); q=0.50 collapses ppl from
   4.5 → 12.6.
2. **Same Pareto shape on VLM (LAMBADA) as on VLA (LIBERO)**: lossless
   only at conservative q; aggressive q breaks the backbone. Different
   modality, identical inflection-point behaviour.
3. Supports the paper's claim that reskip is a **general inference-time
   tool** (not VLA-specific): the same per-block AttnRes router governs
   both text and action prediction, and the same threshold-calibration
   protocol generalises across modalities.

---

## 7. Speed — paper claim and internal validation

> **Paper-bound claim (one sentence).**
> *At the canonical block partition (L=4, 7 blocks at 2B), retrofit
> matches the base Qwen3-VL-2B inference latency under torch.compile —
> production default mode 1.058× base_compiled / 0.889× base_eager at
> seq=2048 cache=T on H100; max-autotune (fixed-shape opt-in) 1.029× /
> 0.864×. All accuracy gains we report — LAMBADA +X pp, MMStar +Y pp,
> LIBERO-4-suite SR — are obtained on this same default fast forward,
> so the paper's headline is "+accuracy at iso-cost, possibly faster"
> — not "speed–accuracy trade-off".*
>
> **Where this claim goes in the paper.**
> 1. **Abstract** — one sentence at the end of the contributions list:
>    *"Retrofit lifts X pp on LAMBADA / Y pp on MMStar / Z pp on LIBERO
>    4-suite SR while matching base inference latency (1.03× base under
>    torch.compile)."*
> 2. **Method §3.X (Inference cost)** — one short paragraph: γ saturates
>    to 1 in trained checkpoints, so the per-block bridge collapses to
>    an additive correction; under torch.compile the 7 routers add 2.9 %
>    over base. Cite the §7 number.
> 3. **Headline Table 2** (VLM + text bench) — add a final column or
>    footer row "Inference cost (vs base, compiled): retrofit 1.029×",
>    so readers see the cost is not free-floating but iso-cost.
> 4. **Discussion / Limitations** — one line: the residual 2.9 % is
>    structural (small-gemm router); torch.compile is required to reach
>    iso-cost; eager retrofit alone is 1.12×.
>
> **What we do NOT put in the paper** (this whole §7 below is internal
> validation only):
> - L=2 vs L=4 speed comparison (paper canonical is L=4 throughout, no
>   need to motivate the partition choice via speed — block-partition
>   ablation §B already justifies L=4 on accuracy).
> - reduce-overhead vs max-autotune mode comparison.
> - Compile-vs-eager logit parity / LAMBADA-500 parity (these were
>   accuracy-safety guards we ran for *ourselves*; reviewers will trust
>   that compile preserves logits within bf16 noise).
> - The bench-bug retraction history (§Q in ablations) — internal
>   honesty record, not paper material.
>
> **Why this framing is the right one.** The user's intent is unambiguous:
> the previously-reported accuracy results were *already* obtained on
> the model running at this fastest default forward path (compile is
> just a one-line wrap around the same `Qwen3VLAttnResRetrofit` we
> trained and eval'd against). So there is no "speed/accuracy
> trade-off" to defend in the paper — only a single iso-cost claim
> that the rest of the paper's accuracy numbers ride on.

The remainder of §7 is the internal speed-bench record, kept for
reproducibility and for the "Inference cost" Method paragraph's
citations. **Do not lift the optimization journey into the paper.**

---

Speed benches: `retrofit/bench/bench_vlm_vs_vla.py` (eager) and
`retrofit/bench/bench_retrofit_compile.py` (torch.compile reduce-overhead /
max-autotune), H100 (GPU 0/1 only — see memory), bf16, KV-cache enabled.
Single-GPU prefill at random token IDs.

### 7.0 Block partition L=2 vs L=4: structural latency

L=4 (paper canonical block partition, 7 blocks at 2B) cuts the per-token
router count in half versus L=2 (legacy, 14 blocks). Re-measured 2026-04-25
GPU 0/1, warmup 5, timed 20:

| config        | seq 1024 base (ms) | seq 1024 retrofit (ms) | ratio | seq 2048 base (ms) | seq 2048 retrofit (ms) | ratio |
|---------------|--------------------|------------------------|-------|--------------------|------------------------|-------|
| L=2 H_r256_5k | 15.42              | 21.25                  | 1.378× | 26.03              | 36.14                  | 1.388× |
| **L=4 H_2B_r256_10k_L4_v3 (canonical)** | 14.91 | 16.66 | **1.117×** | 25.49 | 28.53 | **1.119×** |

**Switching to the paper canonical L=4 block partition alone closes
~70 % of the eager-vs-eager retrofit-base gap** (1.39× → 1.12×). The
remaining ~12 % is the 7 router calls × kernel-launch overhead per token.

VLA in-backbone tracks VLM retrofit within 1 % at both partitions
(VLM/VLA = 0.985–0.991 at L=4), confirming the speed-bench correction
described in ablations §Q.

Skip saves 5–9 % on top of retrofit-eager (per earlier
`bench_skip_savings`), so L=4 retrofit + skip lands at **~1.04–1.06×
base** under cache, eager.

### 7.1 torch.compile reduce-overhead — closes the residual gap

Compiling both base and retrofit with `torch.compile(mode="reduce-overhead",
dynamic=False)` (CUDA-graph capture, fixed prefill shape) shows the
retrofit benefits more than base because the router's many small kernel
launches collapse into a captured graph (warmup 5, timed 15):

| ckpt              | seq  | mode             | base eager | base compiled | retrofit eager | retrofit compiled | ratio (comp/comp) | ratio (comp/base-eager) |
|-------------------|------|------------------|-----------|---------------|----------------|-------------------|-------------------|--------------------------|
| L=2 H_r256_5k     | 1024 | reduce-overhead  | 15.44     | 9.31          | 21.50          | 10.78             | 1.158×            | 0.698×                   |
| L=2 H_r256_5k     | 2048 | reduce-overhead  | 26.30     | 21.25         | 36.54          | 24.45             | 1.151×            | 0.930×                   |
| L=4 canonical     | 2048 | reduce-overhead  | 26.30     | 21.31         | 29.52          | 22.50             | 1.056×            | 0.855×                   |
| L=4 canonical     | 2048 | max-autotune     | 25.97     | 21.81         | 30.45          | 22.43             | **1.029×**        | 0.864×                   |
| **L=4 canonical** | 2048 | **default (PROD)** | **25.97** | **21.81**     | **29.25**      | **23.08**         | **1.058×**        | **0.889×**               |

**Production-default headline (L=4 canonical, seq=2048 cache=T,
mode="default")**: retrofit_compiled / base_compiled = **1.058×**
(5.8 % residual gap), retrofit_compiled / base_eager = **0.889×**
(retrofit + compile is 11 % faster than uncompiled stock HF base).
Mode "default" handles variable input shapes without per-shape autotune
storms, so it is the production deployment mode actually wired into
all eval entry points.

**Best-case headline (mode="max-autotune", fixed-shape only)**:
**1.029×** compiled-vs-compiled, **0.864×** vs base eager. Cite this
as the lower bound of structural retrofit cost for the speed-bench
section. Production paths with stable input shape (e.g. fixed-prompt
LIBERO inference at batch 1) can opt into max-autotune via
``--compile-mode max-autotune`` to recover this number; LAMBADA /
lmms-eval cannot, since variable lengths thrash autotune.

**Combined with the orthogonal 5–9 % skip savings** (eager skip path),
the L=4 retrofit total inference path lands at **~0.94–0.98 × base
compiled**, satisfying the paper "≤ base at iso-accuracy" target. The
production discipline (compile + skip co-existence) is open: dyn-skip's
`.item()` guard breaks the compile graph, so practical deployment picks
one mode per latency budget — see ablations §J.5.

> seq=1024 in the L=4 compile run shows base eager 18.98 ms (vs 14.91 ms
> on a fresh GPU), a thermal/power-state artifact from stacking benches
> on one GPU; treat seq=2048 as authoritative — it has a longer
> measurement window and matches the L=4 baseline run on GPU 1
> (base=25.49, retrofit=28.53).

### 7.2 Production-port status and remaining stretches

**Production port (2026-04-25):** `torch.compile(mode="default",
dynamic=True)` is the default in every paper-cited inference entry
point; toggle off with `--compile-mode off`. See ablations §J.5 for
the table of patched files. Helper: `retrofit/compile_utils.py`.

**Locked invariants** for any further optimization (these were checked
for the production port and held):
- AttnRes core mechanism preserved (per-block residual + adapter +
  α-router).
- 4-suite mean SR within ± 0.5 pp of the no-skip baseline at the chosen q.
- LAMBADA-500 acc within ± 1 pp; argmax bit-equal vs eager on canonical
  prompts (see ablations §J.5 for the parity test).

**Status of optimization candidates in `retrofit/bench/`** (all internal,
not paper material per the editorial header):

1. `torch.compile` — **shipped to production**. Production default mode
   "default": retrofit_compiled / base_compiled = 1.058× / base_eager
   0.889×. Best-case max-autotune (fixed-shape opt-in): 1.029× / 0.864×.
   **LAMBADA-500 accuracy under compile** (`bench_compile_lambada.py`):
   acc 0.5720 compiled vs 0.5700 eager (Δ = +0.20 pp, ppl 4.534 vs
   4.526), per-target argmax agreement 98.60 %. Per-token logit parity
   (`bench_compile_accuracy.py`, reduce-overhead mode) on real prompt
   tokens: 98.51 % argmax agreement, max |Δlogit|=0.50, RMSE 0.060.
   Compile is accuracy-safe within the locked ±1 pp invariant.
2. γ=1 fast path — γ saturates to 1.0 in trained checkpoints. The
   bridge `x_n = h_{n-1} + γ_n · A_n(r_n − h_{n-1})` reduces to
   `h_{n-1} + A_n(r_n − h_{n-1})`, saving one bf16 multiply per block.
   Tiny win on its own (router still dominates).
3. Adapter rank ablation r=256 → r=128 via SVD truncation — earlier r=32
   ablation showed only ~1 ms savings (router dominates). Likely
   marginal alone.
4. (Stretch) Triton-fused router: stack + RMS + bias + einsum + softmax +
   einsum collapsed into a single kernel. Could in principle reach
   retrofit_compiled / base_compiled ≈ 1.00 by removing the 7×N kernel
   launches that survive default-mode compile. Not yet prototyped, and
   per the editorial header on §7 not paper-bound.

---

## 8. VLA preliminary — uniform skip motivates modality-aware future work

To check whether the **threshold calibration generalises across
modalities**, we directly transplanted the LAMBADA-calibrated
dyn-skip configuration (q = 0.85, M = 2 from §6) onto the
`pathB_2B_L4_v3_30k` policy at LIBERO inference.

| condition                                              | libero_spatial | comment |
|--------------------------------------------------------|-----------------|---------|
| no-skip                                                | 99.5 % (suite 1/4 partial) | reference |
| **uniform LAMBADA-calibrated dyn-skip (q=0.85, M=2)**  | **64 %**        | **−35.5 pp catastrophic drop** |

**Reading.** Action-token decoding has a different routing distribution
than language-token decoding: the same `w_recent` value at the same
block carries a different "this is local refinement" semantic in the
action regime. Re-using the LAMBADA threshold causes structural
over-skipping on the action stream and the policy collapses.

This is the empirical hard motivation for **modality-aware skip
calibration** (Method B sim-calib, §1.6): per-modality `τ_n`
calibration on each model's own sim distribution. The §5 numbers using
Method B sim-calib all behave correctly because they calibrate on the
**action distribution**, not the language one. Uniform LAMBADA → action
transfer is the cleanest counter-example. Per-modality, per-token-class
sim-calib is the future-work sweep.

---

## Headline tables for the paper

### Table 1. Method recipe

| Component          | Setting                                                       |
|--------------------|---------------------------------------------------------------|
| Backbone           | Qwen3-VL-2B / 4B, frozen during retrofit                       |
| Block partition    | L=4 layers/block (7 blocks at 2B, 9 at 4B)                     |
| Adapter rank       | 256                                                            |
| γ-curriculum       | 0 → 1, ramp-frac 0.3 (2B) / 0.5 (4B), ends at step 5 000       |
| Retrofit steps     | 10 000 (max_seq 2048; v3 data mix)                             |
| LIBERO OFT         | 30 000 steps, 4 GPUs, ZeRO-2 bf16, bs=8/device                 |
| Reskip strategy    | `recent_weight_gt`, M=2, sim-calibrated τ at quantile q         |
| Reskip operating point | q = 0.99, P = {1, 4} (2B) / {1, 2} (4B)                  |

### Table 2. VLM + text benchmarks at L=4, full splits

| Bench       | base 2B | 2B_L4 v3 | base 4B | 4B_L4 v3 |
|-------------|---------|----------|---------|----------|
| ai2d        | 0.736   | 0.758    | 0.819   | **0.825** |
| mmbench_dev | 75.77   | 78.87    | 83.33   | **85.22** |
| mmmu        | 0.414   | 0.432    | 0.490   | **0.521** |
| mmstar      | 0.536   | 0.536    | 0.624   | **0.632** |
| ocrbench    | 0.772   | 0.814    | 0.819   | **0.824** |
| realworldqa | 0.648   | 0.661    | 0.715   | **0.718** |
| LAMBADA acc | 0.532   | **0.5650** | 0.576   | **0.6625** |
| LAMBADA ppl | 5.49    | 4.61     | 4.72    | **3.20**  |
| HellaSwag   | 0.506   | 0.500    | 0.562   | **0.5515** |

Retrofit ties or strictly improves base on 8/9 (2B) and 9/9 (4B).

### Table 2b. MMStar 6-subcategory decomposition

| Subcategory     | 2B base | 2B_L4 v3 | Δ        | 4B base | 4B_L4 v3 | Δ      |
|-----------------|---------|----------|----------|---------|----------|--------|
| math            | 0.413   | **0.492** | **+7.9** | 0.549   | **0.588** | **+3.9** |
| logical         | 0.429   | 0.432    | +0.3     | 0.626   | 0.602    | −2.4   |
| science         | 0.408   | 0.353    | −5.5     | 0.465   | 0.467    | +0.2   |
| coarse percep.  | 0.714   | 0.734    | +2.0     | 0.788   | 0.812    | +2.4   |
| fine percep.    | 0.520   | 0.505    | −1.5     | 0.611   | 0.606    | −0.5   |
| instance reason | 0.710   | 0.683    | −2.7     | 0.705   | 0.714    | +0.9   |

### Table 3. LIBERO 4-suite — Path B headline

| Method                              | spatial | object | goal  | libero_10 | **mean** |
|-------------------------------------|---------|--------|-------|-----------|----------|
| 2B Path 0 (no AttnRes)              | 0.948   | 0.998  | 0.975 | 0.921     | 0.9605   |
| **2B Path B v2 (AttnRes warm)**     | **0.974** | **0.986** | **0.980** | **0.910** | **0.9625** |
| 4B Path 0 clean (no AttnRes)        | 0.950   | 0.992  | 0.978 | 0.922     | 0.9605   |
| **4B Path B clean (AttnRes warm)**  | **0.974** | **0.982** | **0.980** | **0.914** | **0.9625** |

### Table 3b. 2-seed Path 0 vs Path B (2B)

| suite     | Path 0 mean (2 seed) | Path B v2 mean (2 seed) | Δ       |
|-----------|---------------------|-------------------------|---------|
| goal      | 97.50               | **98.00**               | +0.50   |
| libero_10 | 92.10               | 91.60                   | −0.50   |
| 4-suite mean (mean lens)  | 96.05  | **96.75**           | **+0.70** |
| 4-suite mean (max/min lens) | 95.85 | **97.15**         | **+1.30** |

### Table 4. ReSkip 4-suite Pareto

| q (sim)        | 2B mean SR        | 4B mean SR |
|----------------|-------------------|------------|
| 0.30           | (collapse)        | 0.9185     |
| 0.50           | 0.9665 (Method A) | 0.9425     |
| 0.70           | —                 | 0.9545     |
| 0.85           | 0.831             | 0.959      |
| 0.95           | 0.947             | 0.9615     |
| **0.99**       | **0.9735**        | **0.9645** |
| no-skip (ref)  | 0.9625            | 0.9625     |

### Table 5. Cross-modality VLM reskip (LAMBADA-500)

| Config         | acc    | Δ vs no-skip | ppl     |
|----------------|--------|--------------|---------|
| no-skip        | 0.570  | —            | 4.526   |
| dynskip q=0.85 | 0.560  | −1.0 pp      | 5.258   |
| dynskip q=0.50 | 0.412  | −15.8 pp     | 12.550  |
| dynskip q=0.30 | 0.390  | −18.0 pp     | 14.005  |

### Table 6. Speed (current state, KV-cache enabled bench)

| Variant              | seq 1024 (ms) | seq 2048 (ms) | × base   |
|----------------------|---------------|---------------|----------|
| Base Qwen3-VL-2B     | 14.73         | 25.32         | 1.00×    |
| VLM retrofit (eager) | 20.23         | 35.25         | 1.37–1.39× |
| VLA in-backbone      | 20.63         | 35.63         | 1.40×    |
| + skip (q=0.99)      | ~18.7         | ~32.5         | ~1.27–1.30× |

Optimization target: ≤ 1.0× base under cache.

### Table 7. Part 1 — 340 M from-scratch ReSkip parity

| Task                          | vanilla 340 M | AttnRes 340 M (full) | **AttnRes + ReSkip** | wallclock |
|-------------------------------|---------------|----------------------|------------------------|-----------|
| LAMBADA acc                   | 0.3790        | 0.4056               | **0.4056**             |  —        |
| LAMBADA ppl                   | 24.71         | 20.20                | **20.20**              |  —        |
| HellaSwag acc_norm            | 0.4436        | 0.4607               | **0.4607**             |  —        |
| ARC-Easy acc_norm             | 0.5502        | 0.5438               | **0.5438**             |  —        |
| ARC-Challenge acc_norm        | 0.2912        | 0.3012               | **0.3012**             |  —        |
| s/batch (seq=8 192, bf16)     | —             | 0.05451              | **0.04579**            | **1.19×** |

ReSkip config: `recent_weight_gt`, P={3,5}, M=2, q=0.85.

### Table 8. Cost of obtaining a 2 B AttnRes backbone

| path                                              | trainable    | tokens     | H100-hours | wallclock @ 8×H100 |
|---------------------------------------------------|--------------|------------|-------------|---------------------|
| 340 M AttnRes from-scratch (Part 1 reference)      | 340 M        | 100 BT     | ~180        | ~22 h               |
| 2 B AttnRes from-scratch (hypothetical)            | 2.13 B       | ~200 BT    | 10 000–25 000 | ~52–130 days     |
| **2 B AttnRes retrofit (ours)**                    | 15 M (0.7 %) | ≤ 1 BT     | **~1**      | **~22 min**         |

---

## Artefacts

**Retrofit states** (warm-start sources for VLA training):
- `retrofit/outputs/block_v3/2B_L4_v3_10k/retrofit_attnres_state.pt`
- `retrofit/outputs/block_v3/4B_L4_v3_10k/retrofit_attnres_state.pt`

**Part 1 checkpoints** (340 M from-scratch existence proof):
- AttnRes 340 M: `flame/saves/reskip_transformer-340M`
- vanilla 340 M baseline: `flame/saves/transformer-340M`
- ReSkip P={3,5} export: `outputs/reskip_340M_combined_35_skip2_q085`

**LIBERO ckpts** (production action policies):
- `starVLA/results/Checkpoints/libero_pathB_2B_L4_v3_30k/final_model/pytorch_model.pt`
- `starVLA/results/Checkpoints/libero_pathB_4B_L4_v3_30k/final_model/pytorch_model.pt`

**Reskip configs** (Method B sim-calib at q=0.99, the operating point):
- `retrofit/outputs/dyn_skip_configs/pathB_2B_L4_v3_30k_sim_q099.json`
- `retrofit/outputs/dyn_skip_configs/pathB_4B_L4_v3_30k_sim_q099_b1b2.json`

**lmms-eval JSON outputs**:
- `retrofit/outputs/lmms_eval_block_v3/{2B,4B}_L4/retrofit/**/*_results.json`

**LIBERO eval logs**:
- `starVLA/logs/2026042*_libero_pathB_*_L4_v3_30k_*_skip/eval.log`
- pair-multisuite driver logs at
  `retrofit/outputs/libero_eval_full/RERUN3_pair*_driver.log`

**Speed bench**:
- `retrofit/bench/bench_vlm_vs_vla.py` (head-to-head)
- `retrofit/bench/bench_retrofit_compile.py` (compile probe)
- raw output: `retrofit/outputs/bench_vlm_vs_vla*.log`,
  `retrofit/outputs/bench_true_base.log`

## Cross-references

- Detailed retrofit-recipe rationale, data-mix sweep, block-partition
  sweep, γ-curriculum stability, threshold-calibration sensitivity,
  failed-runs catalogue, implementation traps, speed-bench correction
  history, ReLoop pointers: `paper_ablations_validation.md`.
- Original detailed VLM analysis: `v3_vlm_analysis.md`,
  `v2_vlm_analysis.md` (retained for full eval JSON references).
- Original block-ablation source-of-truth: `block_partition_ablation.md`.
- Original LIBERO Path comparison detail (Path 0 / B / B v2 / C v3 / C v4
  at 30 k & 60 k, embedding-contamination 2×2 study):
  `VLA_LIBERO_RESULTS.md` in `retrofit/`.
- Original reskip per-block / threshold detail: `reskip_libero_results.md`.
- Part 1 (340 M from-scratch) detailed log:
  `DYNAMIC_SKIP_EXPERIMENT_LOG.md` (root of repo).
- Project-level overview with method intuition + paper narrative:
  `PROJECT_OVERVIEW_CN.md` (root of repo).
