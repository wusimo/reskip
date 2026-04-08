# ReSkip: Attention Residuals as Adaptive Computation Routers

**Input-Dependent Depth for LLMs and Vision-Language-Action Models**

ReSkip uses [Attention Residuals](https://arxiv.org/abs/2501.xxxxx) (AttnRes) — learned, input-dependent weighted combinations over transformer depth — as zero-cost routing signals for adaptive computation. Instead of requiring auxiliary exit classifiers or routing networks, the AttnRes weights themselves tell you which blocks matter.

## Key Idea

Standard residual connections treat all layers equally: `x_l = x_{l-1} + f_l(x_{l-1})`. AttnRes replaces this with learned attention over depth:

```
x_l = Σ α_{i→l} · h_i    where α = softmax(w_l^T · k_i / √d)
```

We show these α weights are a natural routing signal:
- **ReSkip**: Skip blocks where no downstream layer attends to them (`I(n) = max_{l>n} α_{n→l} < ε`)
- **ReLoop**: Share weights across K blocks applied M times; AttnRes pseudo-queries differentiate each application
- **VLA Adaptive Depth**: Different modalities (vision/language/action) get different effective depths

## Results (Prototype Scale)

| Experiment | Key Finding |
|-----------|------------|
| **ReSkip** (55M, 6 blocks) | eps=0.05 skips 1/6 blocks (83% FLOPs) with **zero PPL degradation** |
| **ReLoop** (59M, K=4, M=3) | Matches standard PPL (6.93 vs 6.92) with 14% fewer unique params |
| **VLA** (76M, 6 blocks) | Pick-place 5x harder than reach (0.071 vs 0.014 L1); skip modes preserve quality |
| **Routing** | Block importance hierarchy emerges naturally; easy inputs use fewer blocks than hard |

> **Note**: Current results are prototype-scale (55M params, synthetic data). See [PLAN.md](PLAN.md) for the roadmap to paper-quality experiments (350M+ on SlimPajama, LIBERO VLA benchmarks, head-to-head baselines).

## Project Structure

```
reskip/
├── src/
│   ├── attn_residual.py          # Core: LayerAttnRes, BlockAttnRes, OnlineSoftmaxMerge
│   ├── adaptive_transformer.py    # ReSkip: transformer with AttnRes-guided block skipping
│   ├── looping_transformer.py     # ReLoop: weight-shared blocks with adaptive halting (ACT)
│   ├── vla_adaptive.py           # VLA: modality-aware adaptive depth
│   ├── data.py                   # Data pipelines (synthetic + HuggingFace)
│   └── utils.py                  # FLOP counting, latency profiling, logging
├── experiments/
│   ├── train_lm.py               # Train standard/looping AttnRes + skip sweep eval
│   ├── analyze_routing.py         # Routing visualization (heatmaps, importance, Pareto)
│   ├── benchmark_vla.py          # VLA benchmark with modality-adaptive depth
│   └── export_results_to_latex.py # Export results to paper tables
├── paper/
│   ├── main.tex                  # NeurIPS 2026 draft
│   └── main.pdf                  # Compiled paper (10 pages, 7 figures)
├── configs/                       # YAML configs for different scales
├── tests/                         # 20 unit tests
├── run_all.py                    # Master script: run all experiments end-to-end
└── PLAN.md                       # Roadmap for paper-quality experiments
```

## Quick Start

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install torch numpy matplotlib pyyaml tqdm

# Run tests
pytest tests/ -v

# Quick smoke test (CPU, ~3 min)
python run_all.py --quick --device cpu

# Full prototype experiments (GPU, ~1 hour)
python run_all.py --device cuda

# Individual experiments
python experiments/train_lm.py --mode standard --device cuda \
  --d_model 512 --n_heads 8 --n_layers 12 --n_blocks 6

python experiments/train_lm.py --mode looping --device cuda \
  --n_unique_blocks 4 --max_loops 3

python experiments/benchmark_vla.py --device cuda

# Analyze routing patterns
python experiments/analyze_routing.py --checkpoint outputs/standard/best.pt

# Compile paper
bash paper/compile.sh
```

## How It Works

### ReSkip: AttnRes-Guided Layer Skipping

```
Block importance: I(n) = max_{l > n} α_{n→l}

If I(n) < ε:  skip block n (zero compute cost)
              online softmax merge naturally excludes it
```

The routing signal is **free** — no auxiliary classifiers (vs CALM), no per-token routing losses (vs MoD), and it's **input-dependent** (vs static pruning).

### ReLoop: AttnRes-Routed Block Looping

K shared-weight blocks applied up to M times. The key insight: pseudo-queries `w_l` are per-depth-position, not per-block. So even with shared weights, each application has unique routing. The model learns "apply block 2 three times for math, once for factual recall."

### VLA Adaptive Depth

Different token modalities need different effective depths:
- **Vision tokens**: perceptual features captured early → aggressive skipping
- **Language tokens**: task specification → moderate depth
- **Action tokens**: compositional motor planning → full depth

A block is skipped only when ALL modalities agree it's unimportant.

## Generated Outputs

After running experiments, `outputs/` contains:

```
outputs/
├── standard/
│   ├── best.pt                    # Best model checkpoint
│   ├── final_results.json         # Skip sweep, FLOPs, latency
│   └── analysis/
│       ├── routing_heatmap.png    # α_{i→l} weight matrix
│       ├── block_importance.png   # I(n) per block with thresholds
│       ├── difficulty_comparison.png  # Routing by input difficulty
│       ├── pareto_curve.png       # PPL vs FLOPs tradeoff
│       └── training_curves.png    # Loss, PPL, entropy over training
├── looping/
│   ├── best.pt
│   └── final_results.json         # PPL, depth, parameter sharing
└── vla/
    ├── vla_best.pt
    ├── vla_results.json
    ├── vla_benchmark.png          # L1 by mode and task
    └── modality_routing.png       # Per-modality AttnRes weights
```

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
