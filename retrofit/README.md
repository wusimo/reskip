# retrofit/ — Part 2 (VLM) retrofit code

Organized by role. Core module + the production compile helper stay at
the top level; everything else is grouped into `train/`, `eval/`, `bench/`,
`tests/`, `analysis/`.

```
retrofit/
├── qwen3vl_attnres_retrofit.py     ★ CORE: Qwen3VLAttnResRetrofit + router + adapter
├── compile_utils.py                ★ production torch.compile policy (default mode + --compile-mode toggle)
├── retrofit.md                     # running experiment log
├── VLA_LIBERO_RESULTS.md           # VLA-side LIBERO eval log
├── outputs/                        # all training checkpoints + eval logs
│
├── train/                          # training entry points + sweep launchers
│   ├── train_qwen3vl_attnres_retrofit.py   — AttnRes retrofit SFT (γ-curriculum, KD)
│   ├── train_qwen3vl_lora.py                — LoRA baseline (paper comparison)
│   ├── data_v2.py                           — v3 data mix (LLaVA-OneVision 60 / UltraChat 20 / NuminaMath 10 / OpenThoughts 10)
│   ├── run_retrofit_v3.sh                   — canonical L=4 v3 launcher (2B / 4B)
│   ├── run_block_ablation_v3{,_half1,_half2}.sh — 8-cell L∈{1,2,4,6/7} × {2B,4B} block-partition ablation
│   ├── run_v3_ablations.sh                  — v3 6-cell mix ablation (5k / 10k / vlonly / v1-10k)
│   ├── run_libero_pathB_block_v3.sh         — Path B 30k VLA fine-tune from L=4 v3 winners
│   ├── run_pair_multisuite.sh               — paired LIBERO eval driver (no-skip + dynskip on same env reseed)
│   ├── run_qsweep_q085_parallel.sh / run_2B_qsweep_q095_q099.sh / run_4B_qsweep_*.sh
│   │                                        — q-sweep Pareto launchers (Experiment 2 / 1)
│   ├── run_continuation_4suite.sh           — restart helper for paired runs
│   ├── run_exp_pipeline.sh / run_after_pid.sh / run_vlm_reskip_exp3.sh
│   │                                        — orchestration scripts
│   └── run_block_ablation.sh / run_retrofit_v2.sh — legacy v2 launchers (kept for parity)
│
├── eval/                          # accuracy evaluation
│   ├── eval_qwen3vl_attnres_retrofit.py    — LAMBADA + HellaSwag + MMMU (--compile-mode aware)
│   ├── eval_dynamic_skip.py                — dyn-skip eval (compute-saving)
│   ├── eval_lora.py                        — LoRA baseline eval
│   ├── eval_mmbench.py / eval_mmstar.py / eval_mmmu_attnres.py — standalone single-bench evals
│   ├── lmms_eval_retrofit.py               — lmms-eval plugin (broader VLM bench, --compile-mode aware)
│   ├── calibrate_vla_thresholds.py         — dataset-calibrated dynamic_skip_config JSON for VLA
│   ├── calibrate_sim_thresholds.py         — sim-trajectory-calibrated thresholds (paper canonical)
│   ├── analyze_vla_reskip_2b_l4.py         — VLA reskip behaviour analysis (block / suite drift)
│   ├── run_h_family_evals.sh / run_lmms_eval.sh — H-family + lmms-eval dispatchers
│   ├── run_mmstar_lmms_6way.sh / run_mmstar_v2_compare.sh — MMStar 6-subcategory drilldowns
│   └── run_v2_vlm_8gpu.sh                   — 8-GPU lmms-eval helper
│
├── bench/                         # wall-clock speed benchmarks
│   ├── benchmark_speed.py                  — cacheless prefill (legacy)
│   ├── bench_cache_regime.py               — prefill + decode with use_cache=True
│   ├── bench_vlm_vs_vla.py                 — eager VLM-retrofit vs VLA in-backbone parity
│   ├── bench_retrofit_compile.py           — torch.compile speed bench (default / reduce-overhead / max-autotune)
│   ├── bench_compile_accuracy.py           — per-token logit parity eager vs compiled
│   ├── bench_compile_lambada.py            — LAMBADA-500 acc parity eager vs compiled
│   └── benchmark_340M_latency.py           — 340M LLM reference (Part 1 motivation)
│
├── tests/                         # correctness smoke + unit tests
│   ├── smoke_test_qwen3vl_attnres_retrofit.py  — identity-at-init + skip path sanity
│   ├── test_e8_use_cache.py                    — use_cache=True + generate parity
│   └── test_skip_kv_equiv.py                   — skip K/V cache vs no-cache argmax
│
└── analysis/                      # paper-bound writeups + ad-hoc analysis
    ├── paper_main_experiments.md           ★ canonical paper §3-§7 record (numbers cited from here)
    ├── paper_ablations_validation.md       ★ ablations + bench-bug retraction history (paper Appendix)
    ├── block_partition_ablation.md         — L∈{1,2,4,6/7} × {2B,4B} sweep, justifies L=4 canonical
    ├── v3_vlm_analysis.md                  — v3 mix study, justifies LLaVA-OneVision + math/CoT
    ├── v2_vlm_analysis.md                  — v2 catastrophic-VL-collapse postmortem (kept as failure record)
    ├── reskip_libero_results.md            — LIBERO 4-suite reskip Pareto raw tables
    ├── brain_motivation_design.md          — paper-§1 brain-inspired framing notes
    ├── brain_skip_difficulty.py            — alpha-vs-difficulty motivation plot
    ├── alpha_by_modality.py                — VLA modality-α inspection
    ├── gromov_baseline.py                  — Block Influence baseline (Gromov 2024)
    ├── prune_qwen3vl.py                    — α-guided pruning experiment
    └── review_structure.py                 — structural review of trained retrofit
```

## Experiment cookbook

All commands run from `/home/user01/Minko/reskip2/reskip`. Examples below
use these env vars — paste once at the start of a session:

```bash
cd /home/user01/Minko/reskip2/reskip
export STATE_2B=retrofit/outputs/H_2B_r256_10k_L4_v3/retrofit_attnres_state.pt   # L=4, num-blocks=7
export STATE_4B=retrofit/outputs/H_4B_r256_10k_L4_v3/retrofit_attnres_state.pt   # L=4, num-blocks=9
export CKPT_2B=starVLA/results/Checkpoints/libero_pathB_2B_L4_v3_30k/final_model/pytorch_model.pt
export CKPT_4B=starVLA/results/Checkpoints/libero_pathB_4B_L4_v3_30k/final_model/pytorch_model.pt
```

All Python entry points use `sys.path.insert(0, "retrofit")` so
`from qwen3vl_attnres_retrofit import …` and `from compile_utils import …`
work from any subdir.

### Part 2 — Retrofit training

```bash
# Canonical L=4 v3 (paper headline) — 2B + 4B back-to-back on GPUs 0/1
bash retrofit/train/run_retrofit_v3.sh

# Block-partition ablation: 4 block configs × 2 scales = 8 cells
# Half-1 (GPUs 0-3, 2B_L1/L2/L4/L7), half-2 (GPUs 4-7, 4B_L1/L2/L4/L6).
bash retrofit/train/run_block_ablation_v3_half1.sh
bash retrofit/train/run_block_ablation_v3_half2.sh
# (or both halves on a single 8-GPU host:)
bash retrofit/train/run_block_ablation_v3.sh

# v3 mix ablation: 5k vs 10k, vlonly vs full mix, v1-10k control (6 cells, GPUs 2-7)
bash retrofit/train/run_v3_ablations.sh

# Direct python (custom args) — L=4 example
CUDA_VISIBLE_DEVICES=0 python retrofit/train/train_qwen3vl_attnres_retrofit.py \
    --model-path /home/user01/Minko/models/Qwen3-VL-2B \
    --num-blocks 7 --adapter-rank 256 --steps 10000 --max-seq 2048 \
    --gamma-schedule --gamma-start 0 --gamma-end 1 --gamma-ramp-frac 0.3 \
    --data-mix v3 --kl-weight 1.0 \
    --output-dir retrofit/outputs/my_run

# LoRA baseline (paper §3.4 parameter-matched control)
python retrofit/train/train_qwen3vl_lora.py --help
```

### Part 2 — VLM accuracy eval

```bash
# Headline LAMBADA + HellaSwag + MMMU on the canonical (compile=default by default)
python retrofit/eval/eval_qwen3vl_attnres_retrofit.py \
    --model-type trained --state-path $STATE_2B --num-blocks 7

# Disable compile (eager fallback for accuracy reproduction)
python retrofit/eval/eval_qwen3vl_attnres_retrofit.py \
    --model-type trained --state-path $STATE_2B --num-blocks 7 \
    --compile-mode off

# Full lmms-eval suite (paper VLM table)
bash retrofit/eval/run_lmms_eval.sh retrofit 0 \
    ai2d,mmbench_en_dev,mmmu_val,mmstar,ocrbench,realworldqa - full

# MMStar 6-subcategory drilldown (paper Table 2b)
bash retrofit/eval/run_mmstar_lmms_6way.sh

# Per-bench standalone scripts
python retrofit/eval/eval_mmbench.py      --model-type trained --state-path $STATE_2B --num-blocks 7
python retrofit/eval/eval_mmmu_attnres.py --model-type trained --state-path $STATE_2B --num-blocks 7
# eval_mmstar.py is broken (gives identical 0.278 for every model);
# always use the lmms-eval harness (run_lmms_eval.sh / run_mmstar_lmms_6way.sh).

# Batch eval over the H-family / v3 cells
bash retrofit/eval/run_h_family_evals.sh all
```

### Part 2 — VLM dynamic-skip eval (Experiment 3)

```bash
# Single quantile point
python retrofit/eval/eval_dynamic_skip.py \
    --state-path $STATE_2B --num-blocks 7 \
    --quantile 0.95 --eligible 1,4 --max-skips 1 --lambada-n 500

# Full Experiment-3 sweep (baseline + q ∈ {0.30, 0.50, 0.85})
bash retrofit/train/run_vlm_reskip_exp3.sh
```

### Part 3 — VLA Path B 30k fine-tune

```bash
# Default: launch 2B (GPUs 0-3) and 4B (GPUs 4-7) concurrently
bash retrofit/train/run_libero_pathB_block_v3.sh both

# 2B only
bash retrofit/train/run_libero_pathB_block_v3.sh 2b
# 4B only
bash retrofit/train/run_libero_pathB_block_v3.sh 4b
```

### Part 3 — LIBERO eval (no-skip + ReSkip)

```bash
# Sim-trajectory threshold calibration. Step 1: start a policy server (e.g.
# port 7040 on GPU 0), then run the calibration eval — skip is OFF, but
# every forward's phase-1 w_recents are appended to ROUTING_DUMP_PATH.
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=starVLA \
  python starVLA/deployment/model_server/server_policy.py \
    --ckpt_path $CKPT_2B --port 7040 --use_bf16 &
ROUTING_DUMP_PATH=/tmp/sim_dump_2B_L4.jsonl \
  bash starVLA/examples/LIBERO/eval_files/eval_libero_calibrate.sh \
       $CKPT_2B libero_spatial 7040 5

# Step 2: derive a sim-calibrated dyn_skip_config JSON for any quantile
python retrofit/eval/calibrate_sim_thresholds.py \
    --dump   /tmp/sim_dump_2B_L4.jsonl \
    --output retrofit/outputs/dyn_skip_configs/pathB_2B_L4_v3_30k_sim_q099.json \
    --quantile 0.99 --eligible 1,4 --max-skips 2

# Dataset-calibrated thresholds (legacy / Method A — kept for ablation parity)
python retrofit/eval/calibrate_vla_thresholds.py \
    --state-path $STATE_2B \
    --output retrofit/outputs/vla_thresholds/h_r256_5k_q085.json \
    --quantile 0.85 --eligible 4,6,11 --max-skips 2

# Single (suite, config) eval — paired no-skip + dynskip on one server
bash retrofit/train/run_pair_multisuite.sh \
    0 1 7040 $CKPT_2B pathB_2B_L4_v3_30k_paired \
    libero_spatial:none \
    libero_spatial:retrofit/outputs/dyn_skip_configs/pathB_2B_L4_v3_30k_sim_q099.json

# 4-suite ReSkip Pareto sweep at 2B (q=0.95 + q=0.99 on libero_spatial, GPUs 2-3)
bash retrofit/train/run_2B_qsweep_q095_q099.sh

# 4-suite ReSkip Pareto sweep at 4B (paper canonical P={1,2})
bash retrofit/train/run_4B_qsweep_q070.sh           # q=0.70 only
bash retrofit/train/run_4B_qsweep_pareto_0_1.sh     # GPUs 0-1, q ∈ {0.30, 0.50}
bash retrofit/train/run_4B_qsweep_pareto_6_7.sh     # GPUs 6-7, q ∈ {0.85, 0.95, 0.99}

# Parallel q=0.85 sweep on multiple GPUs
bash retrofit/train/run_qsweep_q085_parallel.sh
```

### Part 1/2 — Speed & accuracy parity benches

```bash
# torch.compile speed bench (paper iso-cost claim cites this; default mode = production).
# num_blocks is read from the state file's cfg, not a CLI flag.
python retrofit/bench/bench_retrofit_compile.py \
    --state-path $STATE_2B --compile-mode default        --seq-lens 2048
# Best-case mode (fixed-shape opt-in): max-autotune
python retrofit/bench/bench_retrofit_compile.py \
    --state-path $STATE_2B --compile-mode max-autotune   --seq-lens 2048

# Eager VLM-retrofit vs VLA in-backbone parity (sanity, paper §7.0)
python retrofit/bench/bench_vlm_vs_vla.py --state-path $STATE_2B --vla-n-blocks 7

# Compile-vs-eager accuracy guards (state-path is required)
python retrofit/bench/bench_compile_accuracy.py --state-path $STATE_2B   # per-token logit parity
python retrofit/bench/bench_compile_lambada.py  --state-path $STATE_2B   # LAMBADA-500 acc parity

# Cacheless prefill bench (legacy)
python retrofit/bench/benchmark_speed.py     --state-path $STATE_2B
# Cache-on prefill + decode
python retrofit/bench/bench_cache_regime.py  --state-path $STATE_2B
# 340M reference (Part 1 motivation)
python retrofit/bench/benchmark_340M_latency.py
```

### Correctness tests

```bash
python retrofit/tests/smoke_test_qwen3vl_attnres_retrofit.py
python retrofit/tests/test_skip_kv_equiv.py --state-path $STATE_2B --num-blocks 7
python retrofit/tests/test_e8_use_cache.py  --state-path $STATE_2B --num-blocks 7
```

### Analysis utilities

```bash
# VLA reskip block-drift analysis (paper §5 supporting). Uses LIBERO trajectory
# data from --data-root (defaults baked in; no --suite flag).
python retrofit/eval/analyze_vla_reskip_2b_l4.py \
    --ckpt $CKPT_2B --n-episodes 12 --frame-per-episode 2 \
    --output retrofit/outputs/analysis/vla_reskip_drift.json

# AttnRes-importance per-modality (VLA): vision / language / action α split
python retrofit/analysis/alpha_by_modality.py \
    --vlm-state-path $STATE_2B --num-blocks 7 --gpu 0

# Brain-skip difficulty motivation plot
python retrofit/analysis/brain_skip_difficulty.py \
    --state-path $STATE_2B --num-blocks 7 --eligible 1,4 --quantile 0.85

# Block Influence baseline (Gromov 2024). Uses base Qwen3-VL internally;
# no --state-path needed.
python retrofit/analysis/gromov_baseline.py --num-seqs 8 --gpu 0

# α-guided pruning experiment (full short-train recovery; needs --output-dir)
python retrofit/analysis/prune_qwen3vl.py \
    --skip-layers 9,15 --recover-tokens 100000000 \
    --output-dir retrofit/outputs/prune_layers_9_15

# Structural review of a trained retrofit (γ values, W_up Frobenius)
python retrofit/analysis/review_structure.py --state-path $STATE_2B
```

## Production compile policy

`compile_utils.py` is the single source of truth for `torch.compile` use
in retrofit + VLA inference. Default mode is `"default"` (handles variable
input shapes; production-safe). All paper-cited entry points wrap the
model through `wrap_compile()` and accept `--compile-mode <off|default|reduce-overhead|max-autotune>`:

| Entry point                                                         | Default | Toggle off                       |
|---------------------------------------------------------------------|---------|----------------------------------|
| `retrofit/eval/eval_qwen3vl_attnres_retrofit.py`                    | default | `--compile-mode off`             |
| `retrofit/eval/lmms_eval_retrofit.py`                               | default | `--model_args ...,compile_mode=off` |
| `starVLA/deployment/model_server/server_policy.py`                  | default | `--compile-mode off`             |

Override via env: `RETROFIT_COMPILE_MODE=off`. The off-switch exists so
that accuracy reproductions can fall back to eager if a future torch
version regresses on this code path.

## Canonical pointers

- **Trained retrofit state (paper canonical)**:
  - 2B: `outputs/H_2B_r256_10k_L4_v3/retrofit_attnres_state.pt` — L=4 (7 blocks), v3 mix, 10 k steps, ramp-frac 0.3.
  - 4B: `outputs/H_4B_r256_10k_L4_v3/retrofit_attnres_state.pt` — L=4 (9 blocks), v3 mix, 10 k steps, ramp-frac 0.5.
- **Block-partition ablation cells** (paper §B): `outputs/block_v3/{2B,4B}_L{1,2,4,6,7}_v3_10k/`.
- **VLA Path B 30k checkpoints** (paper §4): `starVLA/results/Checkpoints/libero_pathB_{2B,4B}_L4_v3_30k/final_model/pytorch_model.pt`.
  Per-suite eval logs and reskip Pareto runs live under `retrofit/outputs/libero_eval_full/pathB_{2B,4B}_L4_v3_30k_*`.
- **VLA dyn-skip thresholds (sim-calibrated, paper-canonical)**:
  `outputs/dyn_skip_configs/pathB_{2B,4B}_L4_v3_30k_sim_q*.json` —
  Pareto sweep q ∈ {0.30, 0.50, 0.70, 0.85, 0.95, 0.99}.
- **Legacy L=2 canonical (v1 mix, kept for ablation parity)**:
  `outputs/H_r256_5k/retrofit_attnres_state.pt` and dataset-calibrated
  `outputs/vla_thresholds/h_r256_5k_q085.json`.
- **Experiment log**: `retrofit.md` (dated entries, most recent at bottom).
- **VLA LIBERO log**: `VLA_LIBERO_RESULTS.md`.
- **Paper writeups**: `analysis/paper_main_experiments.md` (§3–§7),
  `analysis/paper_ablations_validation.md` (§A–§J appendix).
