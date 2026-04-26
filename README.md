# ReSkip: Attention Residuals as Adaptive Computation Routers

**Input-Dependent Depth for LLMs and Vision-Language-Action Models**

ReSkip uses [Attention Residuals (AttnRes)](https://arxiv.org/abs/2603.15031) — learned, input-dependent weighted combinations over transformer depth — as zero-cost routing signals for adaptive computation. Instead of requiring auxiliary exit classifiers or routing networks, the AttnRes weights themselves tell you which blocks matter.

## Core Idea

Standard residual connections treat all layers equally: `x_l = x_{l-1} + f_l(x_{l-1})`. AttnRes replaces this with learned attention over depth:

```
x_l = Σ α_{i→l} · h_i    where α = softmax(w_l^T · k_i / √d)
```

These α weights are a natural routing signal:
- **ReSkip**: Skip blocks where downstream layers don't attend to them. Supports both static (calibration-based keep-mask) and dynamic (runtime, input-dependent) skip modes.
- **ReLoop**: Share weights across K blocks applied M times; AttnRes pseudo-queries differentiate each loop iteration.
- **VLA Adaptive Depth**: Different modalities (vision/language/action) get different effective depths.

---

## Current Status (April 2026)

### Implementation

The codebase has three components, each contributing one part of the paper:

| Component | Location | Paper part |
|-----------|----------|------------|
| ReSkip model (LM, from-scratch) | `flash-linear-attention/fla/models/reskip_transformer/` | Part 1 |
| ReLoop model (LM, weight-shared loops) | `flash-linear-attention/fla/models/reloop_transformer/` | (separate study) |
| Training framework | `flame/` | Part 1 |
| Qwen3-VL retrofit (block-level AttnRes injection) | `retrofit/` | Part 2 |
| Production compile policy (one source of truth) | `retrofit/compile_utils.py` | Part 2 §Inference cost |
| VLA fine-tune + LIBERO eval | `starVLA/` + `retrofit/eval/` | Part 3 |

### Key Experimental Results

**Part 1 — 340M from-scratch AttnRes** (`reskip_transformer-340M`, 8 blocks, 24 layers, FineWeb-Edu 100BT):

| Configuration | LAMBADA acc | HellaSwag | ARC-Easy | ARC-Challenge | Wall-clock speedup |
|---|---:|---:|---:|---:|---:|
| Full-depth | 0.4056 | 0.4607 | 0.5438 | 0.3012 | 1.00x |
| Dynamic skip: `attn_only + {3,5} + q=0.85 + max_skips=2` | **0.4056** | **0.4607** | **0.5438** | **0.3012** | **~1.19x** |

> Config: `strategy=recent_weight_gt`, `probe=attn_only`, `positions={3,5}`, `q=0.85`, `max_skips=2`. Position selection uses **ablation-informed + importance-informed** combination: block 3 has the lowest static-removal PPL impact; block 5 has the lowest AttnRes importance score. Together they provide both high skip frequency and safe skip space. See [DYNAMIC_SKIP_EXPERIMENT_LOG.md](DYNAMIC_SKIP_EXPERIMENT_LOG.md) for full experiment details.

**Part 2 — Qwen3-VL-{2B,4B} retrofit** (`retrofit/`) — paper-canonical L=4 v3, 2026-04-23/25:

γ-gated block-level AttnRes injection, **L=4 block partition** (7 blocks at 2B, 9 blocks at 4B), γ-curriculum 0→1, adapter rank 256, **10k SFT steps** on the v3 mix (LLaVA-OneVision 60 % + UltraChat 20 % + NuminaMath 10 % + OpenThoughts 10 %), Qwen3-VL backbone frozen. Every block converges to **γ=1** — the retrofitted model is structurally pure AttnRes. Block partition L=4 is the paper canonical (justified by the L∈{1,2,4,6/7} sweep in [retrofit/analysis/block_partition_ablation.md](retrofit/analysis/block_partition_ablation.md)) and matches the Chen et al. AttnRes paper's S∈{2,4,8} sweet plateau.

VLM benchmarks (lmms-eval, full splits):

| Config | ai2d | mmbench | mmmu | mmstar | ocr | rwqa |
|---|---:|---:|---:|---:|---:|---:|
| Base 2B | 0.736 | 75.77 | 0.414 | 0.536 | 0.772 | 0.648 |
| **2B_L4 v3** | **0.758** | **78.87** | **0.432** | **0.536** | **0.814** | **0.661** |
| Base 4B | 0.819 | 83.33 | 0.490 | 0.624 | 0.819 | 0.715 |
| **4B_L4 v3** | **0.825** | **85.22** | **0.521** | **0.632** | **0.824** | **0.718** |

Text benchmarks (LAMBADA + HellaSwag, n=2000):

| Config | LAMBADA acc | LAMBADA ppl | HellaSwag |
|---|---:|---:|---:|
| Base 2B | 0.532 | 5.49 | 0.506 |
| **2B_L4 v3** | **0.5650** | **4.61** | 0.5000 |
| Base 4B | 0.576 | 4.72 | 0.562 |
| **4B_L4 v3** | **0.6625** | **3.20** | 0.5515 |

> 4B_L4 retrofit strictly beats base on **all 6 VLM benchmarks**; 2B_L4 beats base on 5/6 and ties mmstar. The MMStar 6-subcategory split shows the gain is concentrated in *deliberate reasoning over images*: math **+7.9 pp at 2B / +3.9 pp at 4B**. LAMBADA lifts +3.3 pp (2B) / +8.7 pp (4B). At matched parameter budget (~14 M trainable), LoRA cannot recover base LAMBADA (−0.6 pp) — the AttnRes structure is responsible for the +5 pp gain. See [retrofit/analysis/paper_main_experiments.md](retrofit/analysis/paper_main_experiments.md) §3 and [retrofit/analysis/v3_vlm_analysis.md](retrofit/analysis/v3_vlm_analysis.md).

**Part 3 — VLA fine-tune + ReSkip on LIBERO** (`starVLA/` + `retrofit/`) — 2026-04-25:

LIBERO 4-suite mean SR (50 trials × 10 tasks × 4 suites), 30k OFT fine-tune from L=4 v3 retrofit warm-start (Path B v2: per-block AttnRes integration via `StarVLABackboneSkipContext`):

| Method | spatial | object | goal | libero_10 | **mean** |
|---|---:|---:|---:|---:|---:|
| 2B Path 0 (no AttnRes) | 0.948 | 0.998 | 0.975 | 0.921 | 0.9605 |
| **2B Path B v2 (no-skip)** | **0.974** | **0.986** | **0.980** | **0.910** | **0.9625** |
| **2B Path B v2 + ReSkip q=0.99 (P={1,4}, M=2)** | **0.976** | **0.992** | **0.990** | **0.936** | **0.9735** |
| 4B Path 0 (no AttnRes) | 0.950 | 0.992 | 0.978 | 0.922 | 0.9605 |
| **4B Path B (no-skip)** | **0.974** | **0.982** | **0.980** | **0.914** | **0.9625** |
| **4B Path B + ReSkip q=0.99 (P={1,2}, M=2)** | **0.964** | **0.982** | **0.984** | **0.928** | **0.9645** |

> AttnRes warm-start (Path B) beats pure OFT (Path 0) by **+0.20 pp at both scales**, and ReSkip on top adds **+1.10 pp at 2B / +0.20 pp at 4B**. ReSkip Pareto is graceful all the way down to q=0.30 on 4B (still 0.9185, no catastrophic collapse anywhere). Path B v2 (per-block) beats Path B v1 (observer-only adapter) by +0.45 pp 4-suite mean — the AttnRes integration must reproduce the Part 2 forward exactly. See [retrofit/analysis/paper_main_experiments.md](retrofit/analysis/paper_main_experiments.md) §4–§5 and [retrofit/analysis/reskip_libero_results.md](retrofit/analysis/reskip_libero_results.md).

**Inference cost (paper iso-cost claim)**:

All Part 2 / Part 3 accuracy numbers above are obtained on the production
default forward path (`torch.compile(mode="default", dynamic=True)` via
[retrofit/compile_utils.py](retrofit/compile_utils.py)). At seq=2048,
cache=True, H100, the L=4 canonical retrofit + compile runs at **1.058× base_compiled / 0.889× base_eager** under the production-default mode, and **1.029× / 0.864×** under fixed-shape `max-autotune`. The headline is **"+accuracy at iso-cost (or faster), no trade-off"**, not a speed-optimization narrative. See [retrofit/analysis/paper_main_experiments.md](retrofit/analysis/paper_main_experiments.md) §7 (internal validation only — paper has one Method paragraph + one table footer).

### Resolved Issues

- **Checkpoint consistency bug fixed**: TorchTitan `ModelWrapper.state_dict()` caching caused saved checkpoints to lag behind the live model. Fixed in `flame/flame/train.py`; checkpoints now replay to training loss within `1e-4`.
- **Static skip deprecated as main approach**: Globally deleting a fixed block causes significant PPL degradation for most configurations. Dynamic runtime skip is now the preferred path.
- **Dynamic skip is the main approach**: AttnRes block execution is naturally two-phase (phase-1 routing before the block runs + phase-2 merge). Phase-1 statistics allow per-input skip decisions at near-zero extra cost — no auxiliary network, no extra training.
- **v2 mix VL collapse → v3 mix**: v2 (LLaVA-Instruct-VSFT 30 % + math/CoT) regressed AI2D by 40 pp on 2B. v3 (LLaVA-OneVision 60 % + UltraChat 20 % + NuminaMath/OpenThoughts 20 %) is the new canonical mix; postmortem in [retrofit/analysis/v2_vlm_analysis.md](retrofit/analysis/v2_vlm_analysis.md), recovery in [retrofit/analysis/v3_vlm_analysis.md](retrofit/analysis/v3_vlm_analysis.md).
- **Path B v1 (observer) → v2 (per-block)**: an observer-only adapter (single end-of-tower correction) under-performed per-block in-backbone integration by 0.45 pp on 4-suite LIBERO. Path B v2 reproduces the Part 2 retrofit forward exactly inside the OFT trainer.
- **Speed bench bug retraction**: earlier "retrofit ≈ base under cache" numbers used an incorrect fast-path on the base side. After the fix, eager retrofit is 1.12× base (L=4) / 1.39× base (legacy L=2); torch.compile closes this to iso-cost. See [retrofit/analysis/paper_ablations_validation.md](retrofit/analysis/paper_ablations_validation.md) §J.5 / §Q.

---

## Document Map

| Document | Language | Purpose |
|---|---|---|
| [PLAN.md](PLAN.md) | English | Roadmap to paper quality: 7 critical gaps, timeline, compute budget |
| [FLAME_LM_PLAYBOOK.md](FLAME_LM_PLAYBOOK.md) | English | Part 1 training and evaluation quick reference |
| [EXPERIMENTS_CN.md](EXPERIMENTS_CN.md) | Chinese | Comprehensive training/eval guide with current commands and workflow |
| [DYNAMIC_SKIP_MECHANISM.md](DYNAMIC_SKIP_MECHANISM.md) | Chinese | Why and how dynamic skip works; paper update suggestions |
| [DYNAMIC_SKIP_EXPERIMENT_LOG.md](DYNAMIC_SKIP_EXPERIMENT_LOG.md) | Chinese/English | Chronological log of dynamic skip experiments (2026-04-09 to 04-14) |
| [ATTNRES_SKIP_LOOP_PLAN_CN.md](ATTNRES_SKIP_LOOP_PLAN_CN.md) | Chinese | Algorithm improvement proposals: Direction 2 (better skip scoring) and Direction 3 (unified skip+loop architecture) |
| [retrofit/README.md](retrofit/README.md) | English | Part 2/3 code map: train/eval/bench/tests/analysis layout, canonical pointers, compile policy |
| [retrofit/retrofit.md](retrofit/retrofit.md) | English/Chinese | Part 2 retrofit experiment log — full ablation sweep, dated entries |
| [retrofit/VLA_LIBERO_RESULTS.md](retrofit/VLA_LIBERO_RESULTS.md) | English | Part 3 VLA-side LIBERO eval log |
| [retrofit/analysis/paper_main_experiments.md](retrofit/analysis/paper_main_experiments.md) | English | ★ Canonical paper §3–§7 record (numbers cited from here) |
| [retrofit/analysis/paper_ablations_validation.md](retrofit/analysis/paper_ablations_validation.md) | English | ★ Paper Appendix: ablations + validation + bench-bug retraction history |
| [retrofit/analysis/block_partition_ablation.md](retrofit/analysis/block_partition_ablation.md) | English | L∈{1,2,4,6/7} × {2B,4B} sweep — justifies L=4 canonical |
| [retrofit/analysis/v3_vlm_analysis.md](retrofit/analysis/v3_vlm_analysis.md) | English | v3 mix study — justifies LLaVA-OneVision + math/CoT |
| [retrofit/analysis/v2_vlm_analysis.md](retrofit/analysis/v2_vlm_analysis.md) | English | v2 mix VL-collapse postmortem |
| [retrofit/analysis/reskip_libero_results.md](retrofit/analysis/reskip_libero_results.md) | English | LIBERO 4-suite ReSkip Pareto raw tables |
| [retrofit/analysis/brain_motivation_design.md](retrofit/analysis/brain_motivation_design.md) | English | Paper §1 brain-inspired framing notes |
| [paper/MOTIVATION_EXPERIMENTS.md](paper/MOTIVATION_EXPERIMENTS.md) | English | Motivation-section experiment plan (latency claims) |
| [paper/main.tex](paper/main.tex) | English | NeurIPS 2026 paper draft (Part 1 + Part 2 + Part 3) |

---

## Repository Structure

```
reskip/
├── flash-linear-attention/           # FLA library — Part 1 from-scratch ReSkip / ReLoop
│   └── fla/models/
│       ├── reskip_transformer/       # ReSkip model: dynamic/static block skipping
│       └── reloop_transformer/       # ReLoop model: weight-shared loops + ACT halting
├── flame/                            # Part 1 training framework
│   ├── configs/
│   │   ├── reskip_transformer_340M.json    # 340M ReSkip pretraining config
│   │   └── reloop_transformer_340M.json    # 340M ReLoop config
│   └── saves/                        # Part 1 training checkpoints
├── experiments/                      # Part 1 analysis and evaluation scripts
│   ├── flame_analyze_reskip.py       # Routing analysis + skip export (static & dynamic)
│   ├── flame_lm_eval.py              # lm-eval-harness wrapper
│   ├── flame_generate.py             # Generation with optional skip override
│   └── compare_fsdp_loss.py          # Checkpoint consistency verification
├── retrofit/                         # ★ Part 2 — Qwen3-VL retrofit (block-level AttnRes injection)
│   ├── qwen3vl_attnres_retrofit.py   #     core: Qwen3VLAttnResRetrofit + router + adapter
│   ├── compile_utils.py              #     production torch.compile policy + --compile-mode toggle
│   ├── train/ eval/ bench/ tests/    #     organized by role (see retrofit/README.md)
│   ├── analysis/                     #     paper §3-§7 + appendix writeups + ablation logs
│   └── outputs/                      #     L=4 v3 canonicals: H_{2B,4B}_r256_10k_L4_v3/ + block_v3/
├── starVLA/                          # ★ Part 3 — VLA fine-tune (Path B per-block AttnRes via OFT trainer)
│   └── results/Checkpoints/libero_pathB_{2B,4B}_L4_v3_30k/   # paper canonical VLA ckpts
├── paper/                            # NeurIPS 2026 draft (main.tex, main.pdf)
├── src/                              # Original 55M prototype (kept for reference)
└── train.sh                          # Part 1 training entry point
```

---

## Quick Start (Current Framework)

```bash
# Training — 340M ReSkip, 6 GPUs (see EXPERIMENTS_CN.md or FLAME_LM_PLAYBOOK.md for full args)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 bash train.sh \
    --model.config flame/configs/reskip_transformer_340M.json \
    --model.tokenizer_path /path/to/gla-tokenizer \
    --training.dataset /path/to/fineweb_edu_100BT \
    # ... (full command in EXPERIMENTS_CN.md §4.2)

# Routing analysis + dynamic skip export (auto-searches ablation-informed positions)
python experiments/flame_analyze_reskip.py \
    --model_path flame/saves/reskip_transformer_340M \
    --dataset /path/to/fineweb_edu_100BT \
    --streaming --seq_len 8192 --batch_size 1 --num_batches 32 \
    --dynamic_skip_strategy recent_weight_gt \
    --dynamic_skip_probe_modes attn_only,first_attn \
    --dynamic_skip_position_modes auto \
    --dynamic_skip_quantiles 0.8,0.85,0.9,0.93,0.95,0.97 \
    --dynamic_skip_max_skips_options 1,2,3 \
    --output_dir outputs/reskip_analysis \
    --export_best_dynamic_model_dir outputs/reskip_dynamic_best \
    --device cuda

# Evaluation (skip policy is saved in config.json, auto-applied)
python experiments/flame_lm_eval.py \
    --model_path outputs/reskip_dynamic_best \
    --tasks lambada_openai,hellaswag,arc_easy,arc_challenge \
    --device cuda:0
```

See [FLAME_LM_PLAYBOOK.md](FLAME_LM_PLAYBOOK.md) for complete training and evaluation commands.

---

## How Dynamic Skip Works

ReSkip block execution is naturally **two-phase**:
1. **Phase 1** (`batch_attend_completed_blocks`): Attend over all completed block states — produces routing weights `α` *before* the current block runs.
2. **Phase 2** (`merge_with_partial_block`): Merge the phase-1 result with the current block's partial output online.

**Dynamic skip exploits phase 1**: Since phase-1 routing statistics are available before the block executes, we can make a per-input skip decision at near-zero cost. The current best strategy (`recent_weight_gt`): if the current block places excessive weight on the most recent completed block, it is likely doing local refinement and can be skipped for that input.

Key properties:
- No auxiliary network or training loss
- Position-specific calibration thresholds (not a single global value)
- **Ablation-informed position selection**: blocks are ranked by static-removal PPL impact, not just AttnRes importance. Block 3 (highest importance) turns out to have the lowest removal impact — "frequently referenced" does not mean "irreplaceable"
- `max_skips=2` with combined ablation + importance positions gives the best speed/quality tradeoff

See [DYNAMIC_SKIP_MECHANISM.md](DYNAMIC_SKIP_MECHANISM.md) for the full mechanism description and [DYNAMIC_SKIP_EXPERIMENT_LOG.md](DYNAMIC_SKIP_EXPERIMENT_LOG.md) for experiment details.

---

## Prototype Results (Legacy, 55M)

> The following results are from the original 55M prototype (`src/` codebase). Current work is at 340M scale on FineWeb-Edu.

| Experiment | Key Finding |
|---|---|
| **ReSkip** (55M, 6 blocks) | eps=0.05 skips 1/6 blocks (83% FLOPs) with **zero PPL degradation** |
| **ReLoop** (59M, K=4, M=3) | Matches standard PPL (6.93 vs 6.92) with 14% fewer unique params |
| **VLA** (76M, 6 blocks) | Pick-place 5× harder than reach; skip modes preserve quality |

---

## Citation

```bibtex
@article{reskip2026,
  title={ReSkip: Attention Residuals as Adaptive Computation Routers
         for LLMs and Vision-Language-Action Models},
  author={TBD},
  year={2026}
}

@article{chen2026attnres,
  title={Attention as Residuals: Stronger Residual Connections for
         Pre-Norm Transformers},
  author={Chen and others (Kimi Team)},
  journal={arXiv preprint arXiv:2603.15031},
  year={2026}
}
```

## License

MIT
