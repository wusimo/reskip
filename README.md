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

The main implementation now lives in the **flame/FLA** framework:

| Component | Location |
|-----------|----------|
| ReSkip/ReLoop model | `flash-linear-attention/fla/models/reskip_transformer/` |
| Training framework | `flame/` |
| Main 340M config | `flame/configs/reskip_transformer_340M.json` |
| ReLoop config | `flame/configs/reloop_transformer_340M.json` |

Current training target: **340M model on FineWeb-Edu 100BT**, 6-GPU training with streaming + varlen.

### Key Experimental Results

Results from the `test3` intermediate checkpoint (8 blocks, 24 layers):

| Configuration | LAMBADA acc | HellaSwag | ARC-Easy | ARC-Challenge | Long-context speedup |
|---|---:|---:|---:|---:|---:|
| Full-depth baseline | 0.2630 | 0.3189 | 0.4457 | 0.2602 | 1.00x |
| Dynamic skip: `attn_only + low1 + q=0.9` | **0.2630** | **0.3189** | **0.4457** | **0.2602** | **1.24x** |
| Dynamic skip: `prev_recent + low1 + q=0.95` | 0.2626 | 0.3172 | 0.4453 | 0.2585 | 1.26x |

> The `attn_only + low1 + q=0.9` configuration achieves full benchmark parity at 1.24x speedup on long-context inputs. This is the current paper mainline candidate. See [DYNAMIC_SKIP_EXPERIMENT_LOG.md](DYNAMIC_SKIP_EXPERIMENT_LOG.md) for full experiment details.

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
| [DYNAMIC_SKIP_EXPERIMENT_LOG.md](DYNAMIC_SKIP_EXPERIMENT_LOG.md) | Chinese/English | Chronological log of dynamic skip experiments (2026-04-09 to 04-10) |
| [ATTNRES_SKIP_LOOP_PLAN_CN.md](ATTNRES_SKIP_LOOP_PLAN_CN.md) | Chinese | Algorithm improvement proposals: Direction 2 (better skip scoring) and Direction 3 (unified skip+loop architecture) |

---

## Repository Structure

```
reskip/
├── flash-linear-attention/           # FLA library — contains ReSkip/ReLoop model implementations
│   └── fla/models/reskip_transformer/
│       ├── modeling_reskip_transformer.py   # Forward pass, dynamic skip, routing trace
│       └── configuration_reskip_transformer.py
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

# Routing analysis + dynamic skip export
python experiments/flame_analyze_reskip.py \
    --model_path flame/saves/reskip_transformer_340M \
    --dataset /path/to/fineweb_edu_100BT \
    --dynamic_skip_strategy recent_weight_gt \
    --dynamic_skip_probe_modes all,attn_only \
    --dynamic_skip_quantiles 0.9,0.95,0.97 \
    --output_dir outputs/reskip_analysis \
    --export_best_dynamic_model_dir outputs/reskip_340M_dynamic_skip_ready \
    --device cuda

# Evaluation
python experiments/flame_lm_eval.py \
    --model_path outputs/reskip_340M_dynamic_skip_ready \
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
- `max_skips=1` gives the best stability

See [DYNAMIC_SKIP_MECHANISM.md](DYNAMIC_SKIP_MECHANISM.md) for the full mechanism description and paper update suggestions.

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
