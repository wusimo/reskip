# retrofit/ — Part 2 (VLM) retrofit code

Organized by role. Core module stays at the top level; everything else is
grouped into `train/`, `eval/`, `bench/`, `tests/`, `analysis/`.

```
retrofit/
├── qwen3vl_attnres_retrofit.py     ★ CORE: Qwen3VLAttnResRetrofit + router + adapter
├── retrofit.md                     # running experiment log
├── VLA_LIBERO_RESULTS.md           # VLA-side LIBERO eval log
├── outputs/                        # all training checkpoints + eval logs
│
├── train/                          # training entry points
│   ├── train_qwen3vl_attnres_retrofit.py   — AttnRes retrofit SFT
│   └── train_qwen3vl_lora.py                — LoRA baseline (paper comparison)
│
├── eval/                           # accuracy evaluation
│   ├── eval_qwen3vl_attnres_retrofit.py    — LAMBADA + HellaSwag + MMMU
│   ├── eval_dynamic_skip.py                — dyn-skip eval (compute-saving)
│   ├── eval_lora.py                        — LoRA baseline eval (matches above)
│   ├── eval_mmbench.py                     — MMBench (standalone)
│   ├── eval_mmstar.py                      — MMStar (standalone)
│   ├── eval_mmmu_attnres.py                — MMMU val (standalone)
│   ├── lmms_eval_retrofit.py               — lmms-eval plugin (broader VLM bench)
│   ├── calibrate_vla_thresholds.py         — produce dynamic_skip_config JSON for VLA
│   ├── run_h_family_evals.sh               — batch eval for the 8 H-family variants
│   └── run_lmms_eval.sh                    — dispatcher for lmms-eval
│
├── bench/                          # wall-clock speed benchmarks
│   ├── benchmark_speed.py                  — cacheless prefill (classic)
│   ├── bench_cache_regime.py               — prefill + decode with use_cache=True
│   └── benchmark_340M_latency.py           — 340M LLM reference (Part 1 Motivation)
│
├── tests/                          # correctness smoke + unit tests
│   ├── smoke_test_qwen3vl_attnres_retrofit.py  — identity-at-init + skip path sanity
│   ├── test_e8_use_cache.py                    — use_cache=True + generate parity
│   └── test_skip_kv_equiv.py                   — skip K/V cache vs no-cache argmax
│
└── analysis/                       # ad-hoc analysis / auxiliary baselines
    ├── gromov_baseline.py                  — Block Influence baseline (Gromov 2024)
    ├── prune_qwen3vl.py                    — α-guided pruning experiment
    └── review_structure.py                 — structural review of trained retrofit
```

## Running things

Everything is invoked from `/home/user01/Minko/reskip2/reskip` with explicit
subdir prefix, e.g.:

```bash
# training
python retrofit/train/train_qwen3vl_attnres_retrofit.py --help

# accuracy eval on H_r256_5k
python retrofit/eval/eval_qwen3vl_attnres_retrofit.py \
    --model-type trained --state-path retrofit/outputs/H_r256_5k/retrofit_attnres_state.pt

# dynamic-skip eval (LAMBADA, q=0.95, M=1)
python retrofit/eval/eval_dynamic_skip.py \
    --state-path retrofit/outputs/H_r256_5k/retrofit_attnres_state.pt \
    --quantile 0.95 --max-skips 1 --eligible 4,6,11

# speed bench (cache-off prefill)
python retrofit/bench/benchmark_speed.py \
    --state-path retrofit/outputs/H_r256_5k/retrofit_attnres_state.pt

# speed bench (cache-on prefill + decode)
python retrofit/bench/bench_cache_regime.py \
    --state-path retrofit/outputs/H_r256_5k/retrofit_attnres_state.pt

# correctness tests
python retrofit/tests/smoke_test_qwen3vl_attnres_retrofit.py
python retrofit/tests/test_skip_kv_equiv.py \
    --state-path retrofit/outputs/H_r256_5k/retrofit_attnres_state.pt
python retrofit/tests/test_e8_use_cache.py \
    --state-path retrofit/outputs/H_r256_5k/retrofit_attnres_state.pt

# produce dynamic_skip_config JSON for VLA eval
python retrofit/eval/calibrate_vla_thresholds.py \
    --state-path retrofit/outputs/H_r256_5k/retrofit_attnres_state.pt \
    --output retrofit/outputs/vla_thresholds/h_r256_5k_q085.json \
    --quantile 0.85 --eligible 4,6,11 --max-skips 2

# batch eval for H-family
bash retrofit/eval/run_h_family_evals.sh all

# lmms-eval dispatcher
bash retrofit/eval/run_lmms_eval.sh retrofit 0 mmbench_en_dev,mmstar,mmmu_val - full
```

All Python entry points use `sys.path.insert(0, "/home/user01/Minko/reskip2/reskip/retrofit")`
so moving scripts between subdirs does not break the `from qwen3vl_attnres_retrofit import …`
pattern — the core module stays at the retrofit top level.

## Canonical pointers

- **Trained retrofit state**: `outputs/H_r256_5k/retrofit_attnres_state.pt`
  (γ→1 curriculum, r=256, 5k steps, 50/50 UltraChat+LLaVA).
- **VLA dyn-skip thresholds**: `outputs/vla_thresholds/h_r256_5k_q085.json`
  (q=0.85 LAMBADA calibration, eligible={4,6,11}, max_skips=2).
- **Experiment log**: `retrofit.md` (dated entries, most recent at bottom).
- **VLA LIBERO log**: `VLA_LIBERO_RESULTS.md`.
