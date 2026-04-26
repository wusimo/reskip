# ReSkip: Attention Residuals as Adaptive Computation Routers

**Input-Dependent Depth for LLMs and Vision-Language-Action Models**

ReSkip uses [Attention Residuals (AttnRes)](https://arxiv.org/abs/2501.xxxxx) — learned, input-dependent weighted combinations over transformer depth — as zero-cost routing signals for adaptive computation. Instead of requiring auxiliary exit classifiers or routing networks, the AttnRes weights themselves tell you which blocks matter.

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

The main implementation now lives in the **flame/FLA** framework, and ReSkip and ReLoop are now two separate FLA model modules (split April 2026 to allow ReLoop to evolve independently):

| Component | Location |
|-----------|----------|
| ReSkip model | `flash-linear-attention/fla/models/reskip_transformer/` |
| ReLoop model | `flash-linear-attention/fla/models/reloop_transformer/` |
| Training framework | `flame/` |
| ReSkip 340M config | `flame/configs/reskip_transformer_340M.json` (`model_type=reskip_transformer`) |
| ReLoop 340M config | `flame/configs/reloop_transformer_340M.json` (`model_type=reloop_transformer`) |

Current training target: **340M model on FineWeb-Edu 100BT**, 6-GPU training with streaming + varlen.

### Key Experimental Results

**Part 1 — 340M from-scratch AttnRes** (`reskip_transformer-340M`, 8 blocks, 24 layers, FineWeb-Edu 100BT) — 2026-04-16:

| Configuration | LAMBADA acc | HellaSwag | ARC-Easy | ARC-Challenge | Wall-clock speedup |
|---|---:|---:|---:|---:|---:|
| Full-depth | 0.4056 | 0.4607 | 0.5438 | 0.3012 | 1.00x |
| Dynamic skip: `attn_only + {3,5} + q=0.85 + max_skips=2` | **0.4056** | **0.4607** | **0.5438** | **0.3012** | **~1.19x** |

> Config: `strategy=recent_weight_gt`, `probe=attn_only`, `positions={3,5}`, `q=0.85`, `max_skips=2`. Position selection uses **ablation-informed + importance-informed** combination: block 3 has the lowest static-removal PPL impact; block 5 has the lowest AttnRes importance score. Together they provide both high skip frequency and safe skip space. See [DYNAMIC_SKIP_EXPERIMENT_LOG.md](DYNAMIC_SKIP_EXPERIMENT_LOG.md) for full experiment details.

**Part 2 — Qwen3-VL-2B retrofit** (`retrofit/`) — 2026-04-18:

γ-gated block-level AttnRes injection, 14 blocks × 2 layers, γ-curriculum 0→1 (first 30% of steps), adapter rank 256, **5k SFT steps** on 50/50 UltraChat + LLaVA-Instruct mix, Qwen3-VL-2B backbone frozen. Every block converges to **γ=1** — the retrofitted model is structurally pure AttnRes.

| Configuration | LAMBADA acc | HellaSwag | MMBench | Notes |
|---|---:|---:|---:|---|
| Base Qwen3-VL-2B | 0.5320 | 0.5060 | 0.7267 | — |
| LoRA (r=32 q,v, 4-seed avg ≈ 7M trainable) | 0.526 | 0.508 | — | +0 LAMBADA (noise) |
| Retrofit A (γ-free, r=128, 10k) | 0.576 | 0.512 | 0.720 | γ-free ablation: adapter does the work |
| **Retrofit H_r256_5k (γ→1, r=256, 5k, canonical)** | **0.576** | **0.522** | **0.710** | **pure AttnRes, half the cost of A** |
| Retrofit H (γ→1, r=256, 10k, over-trained) | 0.586 | — | 0.587 ⚠ | over-training breaks MMBench |
| Retrofit H (γ→1, r=256, 20k, severely over-trained) | 0.568 | 0.520 | 0.513 ⚠⚠ | monotonic γ=1 over-training signature |

> **Canonical H_r256_5k**: the retrofitted model runs structurally pure AttnRes (every γ=1, every block consumes the routed sum). Same LAMBADA gain as A (+4.4 pp) with stronger HellaSwag (+1.6 pp vs A's +0.6 pp) and half the training budget. MMBench −1.7 pp is within the n=300 noise floor (±1 question). Over-training at γ=1 is the binding constraint; 5k steps is exactly the right budget for Qwen3-VL-2B. See [retrofit/retrofit.md](retrofit/retrofit.md) for the full 20-config ablation (LoRA + A-sweep + H-sweep over rank/steps/data/ramp) and [paper/main.pdf](paper/main.pdf) for the paper draft.

### Resolved Issues

- **Checkpoint consistency bug fixed**: TorchTitan `ModelWrapper.state_dict()` caching caused saved checkpoints to lag behind the live model. Fixed in `flame/flame/train.py`; checkpoints now replay to training loss within `1e-4`.
- **Static skip deprecated as main approach**: Globally deleting a fixed block causes significant PPL degradation for most configurations. Dynamic runtime skip is now the preferred path.
- **Dynamic skip is the main approach**: AttnRes block execution is naturally two-phase (phase-1 routing before the block runs + phase-2 merge). Phase-1 statistics allow per-input skip decisions at near-zero extra cost — no auxiliary network, no extra training.

---

## Document Map

| Document | Language | Purpose |
|---|---|---|
| [PLAN.md](PLAN.md) | English | Roadmap to paper quality: 7 critical gaps, timeline, compute budget |
| [FLAME_LM_PLAYBOOK.md](FLAME_LM_PLAYBOOK.md) | English | Training and evaluation quick reference |
| [EXPERIMENTS_CN.md](EXPERIMENTS_CN.md) | Chinese | Comprehensive training/eval guide with current commands and workflow |
| [DYNAMIC_SKIP_MECHANISM.md](DYNAMIC_SKIP_MECHANISM.md) | Chinese | Why and how dynamic skip works; paper update suggestions |
| [DYNAMIC_SKIP_EXPERIMENT_LOG.md](DYNAMIC_SKIP_EXPERIMENT_LOG.md) | Chinese/English | Chronological log of dynamic skip experiments (2026-04-09 to 04-14) |
| [ATTNRES_SKIP_LOOP_PLAN_CN.md](ATTNRES_SKIP_LOOP_PLAN_CN.md) | Chinese | Algorithm improvement proposals: Direction 2 (better skip scoring) and Direction 3 (unified skip+loop architecture) |
| [retrofit/retrofit.md](retrofit/retrofit.md) | English/Chinese | Part 2 retrofit experiment log — Qwen3-VL-2B γ-gated AttnRes injection, full ablation sweep |
| [paper/MOTIVATION_EXPERIMENTS.md](paper/MOTIVATION_EXPERIMENTS.md) | English | Motivation-section experiment plan (latency claims) |
| [paper/main.tex](paper/main.tex) | English | NeurIPS 2026 paper draft (Part 1 + Part 2 + VLA plan) |

---

## Repository Structure

```
reskip/
├── flash-linear-attention/           # FLA library
│   └── fla/models/
│       ├── reskip_transformer/       # ReSkip model: dynamic/static block skipping
│       │   ├── modeling_reskip_transformer.py
│       │   └── configuration_reskip_transformer.py
│       └── reloop_transformer/       # ReLoop model: weight-shared loops + ACT halting
│           ├── modeling_reloop_transformer.py
│           └── configuration_reloop_transformer.py
├── flame/                            # Training framework
│   ├── configs/
│   │   ├── reskip_transformer_340M.json    # 340M ReSkip pretraining config
│   │   └── reloop_transformer_340M.json    # 340M ReLoop config
│   └── saves/                        # Training checkpoints
├── experiments/                      # Analysis and evaluation scripts
│   ├── flame_analyze_reskip.py       # Routing analysis + skip export (static & dynamic)
│   ├── flame_lm_eval.py              # lm-eval-harness wrapper
│   ├── flame_generate.py             # Generation with optional skip override
│   └── compare_fsdp_loss.py          # Checkpoint consistency verification
├── starVLA/                          # VLA framework integration target
├── src/                              # Original 55M prototype (kept for reference)
├── paper/                            # NeurIPS 2026 draft (main.tex, main.pdf)
├── outputs/                          # Experiment outputs, exported skip-ready models
└── train.sh                          # Main training entry point
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
  title={Attention Residuals as Adaptive Computation Routers:
         Input-Dependent Depth for LLMs and Vision-Language-Action Models},
  author={TBD},
  year={2026}
}
```

## License

MIT
