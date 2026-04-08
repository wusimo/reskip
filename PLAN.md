# ReSkip: Roadmap to Paper-Quality Experiments

## Overview of Issues and Solutions

The prototype validates the core mechanism (AttnRes weights as routing signals).
The reviewer feedback identifies 7 concrete gaps between current state and publishable quality.
This plan addresses each systematically.

---

## Issue 1: Scale Too Small (Critical)

**Problem:** 55M params on synthetic data. Need 300M+ on natural language.

**Plan:**

### Phase 1: 300M Model on SlimPajama (Primary target)
- **Model:** d=1024, n_heads=16, n_layers=24, n_blocks=8 (~350M params)
- **Data:** SlimPajama-627B (sample ~10B tokens for training, standard split)
- **Tokenizer:** GPT-NeoX tokenizer (vocab 50257) or LLaMA tokenizer (vocab 32000)
- **Training:**
  - ~20K steps with batch_size=256, seq_len=2048 (matches Chinchilla-optimal for 350M)
  - BF16 mixed precision, gradient accumulation 8x
  - Hardware: 4x A100-80GB or 8x A6000-48GB (estimate ~3-4 days)
  - Cosine schedule, lr=3e-4, warmup 2000 steps
- **Eval benchmarks:**
  - WikiText-103 perplexity
  - LAMBADA accuracy
  - HellaSwag (0-shot)
  - ARC-Easy/Challenge (0-shot)

### Phase 2: Scaling Law Suite (Nice-to-have)
- Train at 3 scales: 125M, 350M, 1.3B with matched training tokens
- Show that optimal skip thresholds and routing patterns scale predictably
- This directly addresses the "scaling laws" future direction and makes the paper much stronger

### Implementation Changes Needed
1. Replace `StructuredSyntheticLM` with HuggingFace streaming dataloader:
   ```python
   from datasets import load_dataset
   ds = load_dataset("cerebras/SlimPajama-627B", split="train", streaming=True)
   ```
2. Switch from learned positional embeddings to RoPE (standard for modern LLMs)
3. Add gradient accumulation to `train_lm.py`
4. Add multi-GPU DDP support (torch.distributed)
5. Add proper eval harness (integrate with `lm-eval-harness` by EleutherAI)

---

## Issue 2: Synthetic Data is a Red Flag (Critical)

**Problem:** Controllable difficulty makes routing results tautological.

**Plan:**
- **Drop synthetic data entirely** from the main paper. Move to appendix as "controlled experiment demonstrating mechanism" at most.
- **All main results on natural language** (SlimPajama/C4)
- For input-dependent routing analysis on natural language:
  - Stratify by perplexity: compute per-sequence PPL, bin into easy/medium/hard, analyze routing
  - Stratify by domain: use RedPajama metadata (Wikipedia vs. code vs. books vs. CommonCrawl)
  - Stratify by sequence length complexity: measure effective depth on short factual sentences vs. long reasoning chains
  - Use GSM8K with difficulty grading (by number of steps) to show depth correlates with reasoning complexity

---

## Issue 3: VLA Table Shows Identical Results (Critical)

**Problem:** All skip modes produce exactly the same L1 errors. The thresholds are too conservative.

**Plan:**

### A. Push thresholds until modes diverge
- Sweep vision_skip_threshold: [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
- Sweep action_skip_threshold: [0.001, 0.005, 0.01, 0.02, 0.05]
- Keep language threshold fixed at moderate value
- The table should show: at aggressive thresholds, uniform skipping hurts action prediction while modality-aware preserves it (because it protects action tokens)

### B. Report the operating regime
- Show "blocks actually skipped" per mode (currently all modes keep 5/6 blocks)
- Plot L1 error vs. effective FLOPs for each mode as separate curves
- The modality-aware curve should be a better Pareto frontier than uniform

### C. Investigate the "zero contribution" case
- If blocks truly contribute zero: that's actually interesting
- Analyze: is this because the model never learned to use those blocks, or because it learned to route around them?
- Compare: train same model WITHOUT AttnRes (standard residuals), then try removing the same blocks. If PPL degrades, it means AttnRes is learning to make blocks skippable.

---

## Issue 4: VLA Architecture is Toy-Scale (Critical)

**Problem:** No real VLA backbone, no standard benchmark, no baselines.

**Plan:**

### Phase 1: Integrate with StarVLA
The user's existing StarVLA codebase at `~/Documents/starVLA/` has all the infrastructure:
- Replace standard residual connections in the StarVLA backbone with BlockAttnRes
- Keep the existing vision encoder (Qwen2.5-VL ViT) and action heads (flow-matching)
- Minimal code changes: modify the backbone's forward pass to use AttnRes

### Phase 2: LIBERO Benchmark
- **Benchmark:** LIBERO-Long (most challenging, 10 tasks, 50 demos each)
- **Metrics:** Task success rate (%), average actions to completion
- **Baselines:**
  1. StarVLA (standard residuals, full depth)
  2. StarVLA + AttnRes (full depth, no skipping) — shows AttnRes doesn't hurt
  3. StarVLA + AttnRes + uniform skip — baseline skip method
  4. StarVLA + AttnRes + modality-aware skip (ReSkip-VLA) — our method
  5. OpenVLA (if compute allows) — external baseline
- **Key measurement:** Success rate vs. inference latency curve
  - At what speedup does uniform skip start failing while modality-aware maintains?
  - This is the money figure for the VLA section

### Phase 3: SimplerEnv (Optional, strengthens paper)
- Simulated manipulation benchmark with Google Robot embodiment
- Can run many trials quickly (no real robot needed)
- Supports multiple difficulty levels

### Implementation Changes
1. Add `src/starvla_integration.py` that patches StarVLA's backbone with AttnRes
2. Modify `benchmark_vla.py` to load real LIBERO datasets
3. Add rollout evaluation loop (not just L1 prediction, but actual closed-loop success)

---

## Issue 5: ReLoop Results Too Thin (Major)

**Problem:** Only 2 configurations compared, nearly identical PPL.

**Plan:**

### A. K x M Ablation Grid
| K (unique blocks) | M (max loops) | Effective depth | Unique params | PPL |
|---|---|---|---|---|
| 2 | 6 | 12 | ~90M | ? |
| 3 | 4 | 12 | ~130M | ? |
| 4 | 3 | 12 | ~175M | ? |
| 6 | 2 | 12 | ~260M | ? |
| 8 | 2 | 16 | ~350M (full) | baseline |

All at matched total effective depth, showing the tradeoff between parameter efficiency and quality.

### B. Universal Transformer Comparison
- Implement naive UT (same weights, no AttnRes, standard residuals) at K=4, M=3
- Show that AttnRes routing is what makes weight sharing work
- Expected: UT degrades significantly because it can't differentiate loop iterations; ReLoop maintains quality because pseudo-queries are position-specific

### C. Depth Distribution Histograms
- On natural language data, plot histogram of effective depth per sequence
- Show that depth varies meaningfully (not all sequences use the same depth)
- Stratify by: sequence length, domain, perplexity
- This is the key evidence that adaptive computation is happening

### D. Per-Iteration Routing Analysis
- For K=4, M=3: show what the model learns in each iteration
- Iteration 1: expected to capture syntax/local patterns
- Iteration 2: expected to capture semantics/longer-range dependencies
- Iteration 3: expected to be used only for hard inputs (and skipped for easy ones)
- Visualize with attention weight heatmaps per iteration

---

## Issue 6: Missing Baselines (Critical)

**Problem:** No comparison to any existing method.

**Plan:**

### Methods to Compare Against
All on the same 350M model, same data, same compute budget:

1. **Standard Transformer** (no AttnRes, no skipping) — upper bound on quality
2. **Static Pruning (Gromov et al.)** — prune lowest-BI layers post-training
   - Implementation: train standard model, compute Block Influence (BI) scores, remove N layers, evaluate
   - This is the simplest baseline and readily implementable
3. **CALM (Confident Adaptive Language Modeling)** — early exit with classifiers
   - Implementation: add MLP classifier at each layer, train with auxiliary confidence loss
   - Compare Pareto curves (quality vs. FLOPs)
4. **Mixture-of-Depths (MoD)** — per-token routing with capacity constraint
   - Implementation: add router at each layer, top-k token selection
   - Most relevant comparison: also learns routing, but requires auxiliary losses
5. **LayerSkip** — early exit + self-speculative decoding
   - Can cite numbers from their paper at matched scale if reimplementation is too costly

### Comparison Table Format
| Method | PPL | FLOPs (%) | Params | Auxiliary Components | Training Cost |
|--------|-----|-----------|--------|---------------------|---------------|

Key selling points to demonstrate:
- ReSkip matches or beats CALM/MoD quality at same FLOPs without auxiliary losses
- ReSkip's routing signal is free (no additional parameters or training objectives)
- ReSkip's Pareto curve is at least competitive, ideally dominant

### Practical Implementation Priority
1. **Static Pruning** — easiest, just train a standard model and prune (1 day)
2. **Standard Transformer** — already have this, just remove AttnRes (0.5 days)
3. **CALM** — moderate effort, add exit classifiers (~3 days)
4. **MoD** — significant effort, per-token routing changes architecture (~1 week)
5. **LayerSkip** — cite from paper, don't reimplement unless needed

---

## Issue 7: Pareto Curve Too Sparse (Moderate)

**Problem:** Only 5 threshold values, sharp transition between "fine" and "catastrophic."

**Plan:**

### A. Dense Threshold Sweep
- Sweep 20+ thresholds: [0, 0.001, 0.002, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12, 0.15, 0.20, 0.30, 0.50]
- With 8 blocks (instead of 6), there are more possible operating points

### B. Per-Block Threshold
- Instead of a single global ε, sweep individual block thresholds
- This reveals which specific blocks are dispensable at what cost
- Results in a much smoother Pareto frontier

### C. Calibration-Based Skipping
- Collect α statistics over a calibration set (1000 sequences)
- For each block, compute percentile importance scores
- Skip based on percentile (e.g., "skip if this block's importance is below its 10th percentile")
- This is more robust than a fixed threshold

### D. Compare Pareto Curves Across Methods
- Plot ReSkip, static pruning, CALM on the same PPL vs. FLOPs axes
- This is the most compelling visualization in the paper

---

## Additional Experiment: EVA/Physics Connection (Strengthens Paper)

**Problem:** Section 6 is purely speculative.

**Plan:**
- On LIBERO tasks, compare AttnRes depth utilization:
  - Simple pick-and-place (rigid objects) vs. cloth folding or deformable manipulation
  - If depth patterns differ, this supports the physics-iterative-inference interpretation
- Even one figure showing "rigid objects use depth 4.2 avg, deformable use depth 6.8 avg" would be powerful
- Can also correlate with BiX-VLA's left/right arm depth patterns if bimanual data is available

---

## Timeline (Aggressive, for NeurIPS 2026 deadline)

### Week 1-2: Infrastructure
- [ ] Multi-GPU training support (DDP)
- [ ] SlimPajama data pipeline with streaming
- [ ] RoPE integration
- [ ] lm-eval-harness integration
- [ ] StarVLA AttnRes integration

### Week 3-4: Main LM Experiments
- [ ] Train 350M standard transformer baseline on SlimPajama
- [ ] Train 350M AttnRes model
- [ ] Static pruning baseline (from standard model)
- [ ] CALM baseline
- [ ] Full skip threshold sweep (20+ points)
- [ ] Input-dependent routing analysis on natural language

### Week 5-6: ReLoop + Baselines
- [ ] K x M ablation grid (at 350M scale)
- [ ] Vanilla Universal Transformer comparison
- [ ] Depth distribution histograms
- [ ] MoD comparison (if time allows)

### Week 7-8: VLA Experiments
- [ ] StarVLA + AttnRes integration + LIBERO training
- [ ] LIBERO-Long benchmark (all baselines)
- [ ] Modality-aware vs. uniform skip divergence experiments
- [ ] Latency profiling on real hardware

### Week 9-10: Paper Revision
- [ ] Replace all tables with real results
- [ ] Dense Pareto curves
- [ ] Per-domain/difficulty routing analysis figures
- [ ] Scaling law analysis (if additional compute available)

---

## Compute Requirements Estimate

| Experiment | GPU-Hours | Hardware |
|-----------|-----------|----------|
| 350M standard baseline | ~100 | 4x A100 |
| 350M AttnRes model | ~100 | 4x A100 |
| 350M ReLoop (K x M grid, 5 configs) | ~400 | 4x A100 |
| CALM baseline | ~120 | 4x A100 |
| Static pruning (eval only) | ~10 | 1x A100 |
| StarVLA + AttnRes (LIBERO) | ~200 | 4x A100 |
| Scaling law suite (125M, 1.3B) | ~800 | 8x A100 |
| **Total (minimum for main results)** | **~930** | |
| **Total (with scaling laws)** | **~1730** | |

At ~$2/GPU-hour (cloud), minimum cost ~$1,860, full suite ~$3,460.

---

## Key Code Changes Needed

### `src/adaptive_transformer.py`
- Add RoPE support
- Add gradient checkpointing for large models
- Add DDP wrapper

### `src/data.py`
- Add SlimPajama/C4 streaming dataloader
- Add lm-eval-harness integration for downstream benchmarks

### `src/baselines/`
- `static_pruning.py` — Gromov et al.'s Block Influence method
- `calm.py` — CALM early exit with auxiliary classifiers
- `mod.py` — Mixture-of-Depths per-token routing
- `universal_transformer.py` — Vanilla UT without AttnRes

### `src/starvla_integration.py`
- Patch StarVLA backbone with BlockAttnRes
- Add LIBERO dataset loader and rollout evaluation

### `experiments/train_lm.py`
- Multi-GPU DDP support
- Gradient accumulation
- Streaming data
- lm-eval integration at each epoch

### `experiments/benchmark_vla.py`
- LIBERO rollout evaluation (closed-loop, not just L1)
- Real vision encoder integration
- Latency profiling on real hardware
