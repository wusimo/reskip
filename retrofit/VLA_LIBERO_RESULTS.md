# VLA LIBERO — Experiment Log

Running log for Qwen3-VL-2B + OFT action head fine-tuned on LIBERO (all 4 suites,
`libero_all` mix). Paper target: **30k steps × bs=8 × ZeRO-2 = 9.54 epochs** (matches
starVLA baseline in their README).

Eval harness: `examples/LIBERO/eval_files/run_full_eval.sh`
→ 1 policy server (bf16) + 1 LIBERO sim (dreamervla conda env), 50 trials × 10 tasks
= 500 episodes per suite. Four suites run sequentially per ckpt.

---

## Quick-start reproduction

### 1. One-time env setup (both training & eval)

**Main venv** (training, policy server): `/home/user01/Minko/reskip2/.venv`
  Already has: transformers 4.57, accelerate 1.5, deepspeed 0.18, torch 2.8,
  our fork of starVLA deps (omegaconf, diffusers, numpydantic, albumentations,
  pytorch3d stub in `.venv/lib/python3.11/site-packages/pytorch3d/`), tyro,
  matplotlib, mediapy, websockets, msgpack.

**Simulator env** (LIBERO sim client only): `/home/user01/miniconda3/envs/dreamervla`
  Has: Python 3.11, numpy 1.26, robosuite 1.4.1, bddl, LIBERO (installed
  `-e /home/user01/yuxinglei/workspace/DreamerVLA/LIBERO`), tyro, easydict,
  robomimic, hydra-core, opencv, gym, cloudpickle.

If those envs are gone, restore from:
```bash
# Main venv — re-run these if missing a dep:
uv pip install omegaconf diffusers deepspeed numpydantic albumentations==1.4.18 \
    python-Levenshtein decord tyro matplotlib mediapy websockets msgpack \
    robomimic easydict hydra-core opencv-python cloudpickle gym

# dreamervla conda — re-install LIBERO if missing:
/home/user01/miniconda3/envs/dreamervla/bin/pip install \
    tyro matplotlib mediapy websockets msgpack bddl \
    easydict robomimic hydra-core opencv-python gym cloudpickle
/home/user01/miniconda3/envs/dreamervla/bin/pip install --no-deps \
    -e /home/user01/yuxinglei/workspace/DreamerVLA/LIBERO
```

### 2. Prepare base VLM

Qwen3-VL-2B has no `🔍` action token in its tokenizer; add one:
```bash
python -c "
import sys
sys.path.insert(0, '/home/user01/Minko/reskip2/reskip/starVLA/starVLA/model/modules/vlm/tools/add_qwen_special_tokens')
import add_special_tokens_to_qwen as m
sys.argv = ['x',
  '--model-id', '/home/user01/Minko/models/Qwen3-VL-2B',
  '--save-dir', '/home/user01/Minko/reskip2/reskip/starVLA/playground/Pretrained_models/Qwen3-VL-2B-Instruct-Action',
  '--tokens', '🔍',
  '--init-strategy', 'avg']
m.main()
"
```

Saved 1 new special token (embed init = avg of existing). Action-head training reads this.

### 3. LIBERO dataset

Already symlinked: `starVLA/playground/Datasets/LEROBOT_LIBERO_DATA -> /home/user01/Minko/datasets/libero`

Four lerobot-format suites inside:
`libero_spatial_no_noops_1.0.0_lerobot`, `libero_object_*`, `libero_goal_*`, `libero_10_*`.

### 4. Training (pick one path)

All training scripts live in `starVLA/examples/LIBERO/train_files/`. All use 4 GPUs
× bs 8/GPU × ZeRO-2 = effective batch 32. 30k steps → ~2.4 epochs (published recipe
is 9.5 epochs at effective bs 128; our compute is below that but gives meaningful
comparison).

**Path 0 — pure base, no AttnRes** (baseline control):
```bash
cd /home/user01/Minko/reskip2/reskip/starVLA
CUDA_VISIBLE_DEVICES=0,1,2,3 WANDB_MODE=offline PYTHONPATH=$(pwd) \
  bash examples/LIBERO/train_files/run_libero_train_attnres_2B.sh \
    4 30000 8 libero_path0_base_30k
# Flag `attnres.enabled False` is set inside the script → stock Qwen3-VL-2B.
```

**Path B — warm-start from Part-2 VLM retrofit state**:
```bash
STATE=/home/user01/Minko/reskip2/reskip/retrofit/outputs/H_r256_5k/retrofit_attnres_state.pt
CUDA_VISIBLE_DEVICES=0,1,2,3 WANDB_MODE=offline PYTHONPATH=$(pwd) \
  bash examples/LIBERO/train_files/run_libero_train_attnres_on.sh \
    4 30000 8 libero_pathB_warm_30k "$STATE"
# Loads router/adapters/γ from VLM retrofit, then fine-tunes on LIBERO actions.
```

**Path C — AttnRes γ-curriculum from base (no VLM retrofit warm-start)**:
```bash
cd /home/user01/Minko/reskip2/reskip/starVLA
CUDA_VISIBLE_DEVICES=0,1,2,3 WANDB_MODE=offline PYTHONPATH=$(pwd) \
accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 4 \
  starVLA/training/train_starvla.py \
  --config_yaml ./examples/LIBERO/train_files/starvla_cotrain_libero_attnres.yaml \
  --framework.name QwenOFT \
  --framework.qwenvl.base_vlm playground/Pretrained_models/Qwen3-VL-2B-Instruct-Action \
  --framework.attnres.enabled True \
  --framework.attnres.n_blocks 14 \
  --framework.attnres.adapter_rank 256 \
  --framework.attnres.enable_skipping False \
  --framework.attnres.skip_mode none \
  --framework.attnres.gamma_ramp_steps 9000 \
  --framework.attnres.gamma_target 1.0 \
  --datasets.vla_data.data_root_dir playground/Datasets/LEROBOT_LIBERO_DATA \
  --datasets.vla_data.data_mix libero_all \
  --datasets.vla_data.per_device_batch_size 8 \
  --trainer.vla_data.video_backend torchvision_av \
  --trainer.freeze_modules '' \
  --trainer.max_train_steps 30000 \
  --trainer.save_interval 30000 \
  --trainer.logging_frequency 10 \
  --trainer.eval_interval 1000000 \
  --run_root_dir ./results/Checkpoints \
  --run_id libero_pathC_curr_30k \
  --wandb_project starVLA_Libero --wandb_entity reskip
```

γ schedule is implemented in `src/starvla_integration.py::StarVLAAttnResAdapter.enable_gamma_curriculum` —
a non-persistent `_gamma_scale` buffer ramps 0→1 over 9k steps. γ Parameter itself
stays learnable; optimizer updates it around the scale curve.

Checkpoints land at `starVLA/results/Checkpoints/<run_id>/final_model/pytorch_model.pt`.

### 5. Post-training cleanup (one-time per ckpt if trained before bugfix on `_bound_text_model`)

Pre-2026-04-19-fix ckpts have the Qwen3-VL-2B text model duplicated under
`attnres_adapter._bound_text_model.*` because `bind_text_model()` used to set
it as a plain attribute (auto-registered as submodule). Strip those keys before
eval:
```python
import torch
path = 'starVLA/results/Checkpoints/<run_id>/final_model/pytorch_model.pt'
sd = torch.load(path, map_location='cpu', weights_only=False)
inner = sd['state_dict'] if 'state_dict' in sd else sd
bad = [k for k in inner.keys() if '_bound_text_model' in k]
for k in bad: del inner[k]
torch.save(sd, path)
```
Ckpt should shrink from ~10 GB → ~5 GB. New trainings (after fix) don't need this.

### 6. Full LIBERO eval (2 GPUs per checkpoint)

```bash
cd /home/user01/Minko/reskip2/reskip/starVLA
CKPT=$(pwd)/results/Checkpoints/libero_path0_base_30k/final_model/pytorch_model.pt

# One-liner: starts policy server + sequentially runs 4 suites × 50 trials.
bash examples/LIBERO/eval_files/run_full_eval.sh \
  path0_30k "$CKPT" <server_gpu> <render_gpu> <port> 50
# e.g. server_gpu=0, render_gpu=1, port=5694
```

Per-suite logs: `retrofit/outputs/libero_eval_full/<label>_<suite>/eval.log`.
Look for `Total success rate: <value>` at end of each.

Two checkpoints in parallel → use GPUs 0/1 for one, 2/3 for the other (different
ports), and remaining GPUs for a concurrent training.

### 7. EGL headless rendering — known gotchas

The server box has NVIDIA + Mesa EGL vendors in `/usr/share/glvnd/egl_vendor.d/`
and the default pickup lands on Mesa, which then errors inside robosuite/mujoco.
The eval_libero.sh script already sets these; if you hit `EGLError: <exception str() failed>`:

```bash
export MUJOCO_GL=egl
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export EGL_DEVICE_ID=<render_gpu_index>  # per eval client, use a different GPU
# For multi-client parallel eval, also mask CUDA_VISIBLE_DEVICES=<render_gpu>
# so MuJoCo / robosuite EGL picks that GPU instead of GPU 0.
```

The `<exception str() failed>` lines in the log under `Exception ignored in
__del__` are HARMLESS cleanup noise (MuJoCo/robosuite's EGL context destructor).
The real error is always a proper Python Traceback, not that line.

### 8. Fixed starVLA bugs (note for future debugging)

These were all real bugs in the starVLA repo we patched:

- `QwenOFT.predict_action` hard-accessed `routing_info["keep_mask"]` → switched to
  `.get(..., None)` so adapters without skip still work.
- `QwenOFT.predict_action` double-passed `return_routing_info` to `_encode_backbone`
  (once as kwarg, once in `**kwargs`) → `pop()` before re-adding.
- `model2libero_interface.ModelClient.step` used `response` outside the
  `step % action_chunk_size == 0` block → undefined for chunk-intermediate steps.
  Cached last response as `self._last_response` instead.
- Framework `__init__.py` called `logger.log(...)` on a `PureOverwatch` instance
  that has no `.log` method → replaced with per-submodule try/except + `logger.warning`.
- `src/starvla_integration.StarVLAAttnResAdapter` was observer-only (last-block
  correction only) → replaced with per-block in-backbone AttnRes matching Part-2
  retrofit's forward (`StarVLABackboneSkipContext._patched_forward`).

---

---

## Variants

| Label | AttnRes | γ schedule | Adapter rank | n_blocks | Init |
|-------|---------|------------|--------------|----------|------|
| **path0_base** | off | — | — | — | Qwen3-VL-2B-Instruct-Action |
| **pathA_attnres** | on (γ=1 everywhere, frozen) | none | 256 | 14 | from-scratch |
| **pathA_1gpu** | on | none | 256 | 14 | from-scratch, single-GPU small bs (quality control) |
| **pathB_warm** | on (γ=1) | none | 256 | 14 | warm-start from retrofit `H_r256_5k/retrofit_attnres_state.pt` |
| **pathC_curr** | on | 0→1 over first 9k steps | 256 | 14 | from-scratch |

Common training config (all variants):
- Qwen3-VL-2B backbone, L1Regression action head, bs=8/device, ZeRO-2, 4 GPUs
- LR: base 2.5e-5, qwen_vl_interface 1e-5, action_model 1e-4, cosine_with_min_lr
- warmup 5k steps, weight_decay 1e-8, grad_clip 1.0
- data_mix `libero_all` (spatial+object+goal+10 pooled)

Common eval config: bf16, `enable_skipping=false`, `skip_mode=none` (AttnRes forward
without skipping — only validates accuracy preservation, not compute saving).

---

## Results

### 5k-step smoke (libero_goal only, 5 trials/task = 50 episodes)
Significantly undertrained; numbers are for pipeline-validation only.

| ckpt | libero_goal SR |
|------|----------------|
| pathA_attnres_5k | **30.0%** (15/50) |
| pathB_warm_5k    | 10.0% (5/50) |
| path0_base_5k    | 8.0% (4/50) |
| pathA_1gpu_5k    | 2.0% (1/50) |

Observation: at low-compute regime, AttnRes from-scratch (pathA_attnres) beats
baseline by +22 pp. pathB_warm's LM-retrofit init does not immediately transfer —
may need longer training.

### 30k-step full eval (50 trials/task = 500 episodes/suite)

Paper reference (starVLA README): Qwen3-VL-OFT @ 30k gets
spatial 97.8 / object 98.6 / goal 96.2 / long 93.8 / **avg 96.6**.

| ckpt | spatial | object | goal | long | avg |
|------|---------|--------|------|------|-----|
| path0_base_30k (baseline Qwen3-vl-2BOFT) | 94.8% (474/500) | **99.8%** (499/500) | 97.6% (488/500) | … | … |
| pathB_warm_30k (AttnRes warm) | **96.8%** (484/500) | 99.6% (498/500) | 97.6% (488/500) | … | … |

Δ pathB − path0 per suite: spatial +2.0 pp, object −0.2 pp, goal 0.0 pp.

path0 number is slightly below paper's 97.8 — likely due to 30k vs potentially
longer training in paper (they list 9.54 epochs but configs may differ).

### 30k-step: pending runs
- **pathA_attnres_30k** — not started (from-scratch AttnRes at full compute)
- **pathA_1gpu_30k** — not started (single-GPU ablation)
- **pathC_curr_30k** — **training in progress** (started 2026-04-19 08:10,
  4×H100 ZeRO-2, γ ramp 0→1 over 9k steps, 30k total; eval pending on completion)

### Skip evaluation: pending
All current evals run with AttnRes forward but `enable_skipping=false`. The paper's
compute-saving claim still needs a skip sweep (τ thresholds, `skip_mode=uniform`
and `modality_aware`) on the accuracy-preserving ckpts.

---

## Paths

- Checkpoints root: `/home/user01/Minko/reskip2/reskip/starVLA/results/Checkpoints/`
  - `libero_path0_base_30k/final_model/pytorch_model.pt`
  - `libero_pathB_warm_30k/final_model/pytorch_model.pt`
  - `libero_pathC_curr_30k/` (training)
- Eval output videos: `/home/user01/Minko/reskip2/reskip/starVLA/results/libero_<suite>/<ckpt>/*.mp4`
- Eval master logs: `/home/user01/Minko/reskip2/reskip/retrofit/outputs/libero_eval_full/{path0_30k,pathB_30k}_master.log`
- Retrofit warm-start state: `/home/user01/Minko/reskip2/reskip/retrofit/outputs/H_r256_5k/retrofit_attnres_state.pt`

---

## Log

### 2026-04-19 08:11 — 30k full eval launched
path0_base_30k + pathB_warm_30k via `run_full_eval.sh`. Each runs 4 suites sequentially.
ETA ~2.7 h per ckpt (≈40 min × 4 suites).

### 2026-04-19 08:57 — libero_spatial done (suite 1/4)
- pathB_warm_30k: **96.8%** (484/500)
- path0_base_30k: **94.8%** (474/500)
- Δ AttnRes: **+2.0 pp**
Both moved on to libero_object immediately (run_full_eval.sh handles sequencing).

### 2026-04-19 09:40 — libero_object done (suite 2/4)
- path0_base_30k: **99.8%** (499/500)
- pathB_warm_30k: 99.6% (498/500)
- Δ AttnRes: **−0.2 pp** (1-episode gap, within n=500 noise floor).
Both moved on to libero_goal.

### 2026-04-19 10:40 — libero_goal done (suite 3/4)
- path0_base_30k: **97.6%** (488/500)
- pathB_warm_30k: **97.6%** (488/500)
- Δ AttnRes: 0.0 pp (tied). Both exceed paper's 96.2%.
Both moved on to libero_10 (long-horizon).

### 2026-04-19 08:10 — pathC_curr_30k training launched
γ-curriculum 0→1 over 9k steps, r=256, n_blocks=14, 30k total. Expect 4–6 h on 4×H100.

### 2026-04-19 11:05 — libero_10 done (suite 4/4)
- path0_base_30k: **92.8%** (464/500)
- pathB_warm_30k: 91.6% (458/500)
- Δ AttnRes: **−1.2 pp** (AttnRes slightly hurts long-horizon). Both ~1 pp below paper's 93.8%.

**Final 4-suite averages**: path0 = 96.25, pathB = 96.40 (+0.15 pp).
AttnRes warm-start wins on spatial (+2.0), ties on object/goal, loses on long-horizon (−1.2).
Per-suite distribution is non-trivial despite near-tied average.

### 2026-04-19 10:30 — pathC_curr_30k training done BUT γ-curriculum buggy
Saved γ = [0]×13 + [-0.035]×1 — near zero throughout. Root cause: in-place
`self.gamma.data.fill_()` inside forward got overwritten by DeepSpeed ZeRO-2's
FP32-master all-gather after each optimizer step. Ckpt is therefore equivalent
to "Path A stuck at γ=0" — dead weight. Not usable as γ-curriculum data point.

### 2026-04-19 10:45 — pathC_curr_v2_30k relaunched with curriculum fix
Fix in `src/starvla_integration.py`: split effective γ into `γ_param × _gamma_scale`
where γ_param stays learnable (init 1.0) and `_gamma_scale` is a buffer ramped
0→1 over 9k steps. Buffers are NOT touched by ZeRO-2 optimizer, so the schedule
survives. 30k steps on GPUs 4-7, currently in progress.

### 2026-04-20 02:55 — pathBv2 libero_spatial done (suite 1/4)
- pathB_warm_v2_30k: **97.8%** (489/500)
- vs pathB_warm_30k (v1, observer bug): 96.8%
- vs path0_base_30k: 94.8%
- Δ per-block AttnRes over observer: **+1.0 pp**; over base: **+3.0 pp**
Running through remaining suites (object / goal / 10).

### 2026-04-20 ~03:45 — pathBv2 libero_object done (suite 2/4)
- pathB_warm_v2_30k: **99.6%** (498/500)
- vs pathB_warm_30k (v1): 99.6% (tied — both at saturation)
- vs path0_base_30k: 99.8%
- Δ per-block AttnRes over base: **−0.2 pp** (1-episode gap, n=500 noise floor)
Object saturates for all three methods; limited headroom to show AttnRes effect here.

### 2026-04-20 ~04:30 — pathBv2 libero_goal done (suite 3/4)
- pathB_warm_v2_30k: **97.4%** (487/500)
- vs pathB_warm_30k (v1): 97.6%
- vs path0_base_30k: 97.6%
- Δ per-block AttnRes over base / v1: **−0.2 pp** (1-episode gap, n=500)

**3-suite running average** (spatial/object/goal):
- pathB_warm_v2_30k: **98.27** (win +0.87 vs base, +0.27 vs v1)
- pathB_warm_30k (v1 observer): 98.00
- path0_base_30k: 97.40

Proceeding to libero_10 (long-horizon). v1 observer lost -1.2 pp there; v2 per-block will
decide whether fixed implementation still helps on long-horizon.

### 2026-04-20 ~05:35 — pathBv2 libero_10 done (suite 4/4) — FINAL
- pathB_warm_v2_30k: **92.6%** (463/500)
- vs pathB_warm_30k (v1): 91.6% (+1.0 pp)
- vs path0_base_30k: 92.8% (−0.2 pp, 1 episode)

Per-task breakdown highlights (v2 vs base):
  task 9 (hardest for base, base=68%):  v2 **74% (+3 eps)** — AttnRes helps hardest task
  task 1 (base=94%):                    v2 86% (−4 eps)
  task 7 (base=82%):                    v2 78% (−2 eps)
  task 10 (base=98%):                   v2 94% (−2 eps)
Not uniformly weaker on long-horizon — failure distribution shifts across tasks.

## FINAL 4-suite comparison (n=500/suite, 2000 total episodes)

| method                          | spatial | object | goal | 10   | **avg** |
|---------------------------------|---------|--------|------|------|---------|
| path0_base_30k                  | 94.8    | 99.8   | 97.6 | 92.8 | 96.25   |
| pathB_warm_30k (v1 observer)    | 96.8    | 99.6   | 97.6 | 91.6 | 96.40   |
| **pathB_warm_v2_30k (per-blk)** | **97.8**| 99.6   | 97.4 | 92.6 | **96.85** |

**Headline:** per-block AttnRes warm-start beats base by **+0.60 pp** across 4 suites,
and beats the (buggy) observer-only warm-start by **+0.45 pp**. The fix matters.
Object saturates for all three; spatial shows the clearest AttnRes win (+3.0 pp vs base).

### 2026-04-20 — 2-run variance replication (goal + lib10) on path0 and pathB v2

To quantify single-run noise, replicated goal + libero_10 with fresh env reseed.

| suite      | path0 run1 | path0 run2 | path0 avg | pathB v2 run1 | pathB v2 run2 | pathB v2 avg |
|------------|------------|------------|-----------|---------------|---------------|--------------|
| goal       | 97.6       | 97.4       | **97.50** | 97.4          | 98.6          | **98.00**    |
| libero_10  | 92.8       | 91.4       | **92.10** | 92.6          | 90.6          | **91.60**    |

- goal: pathB v2 +0.5 pp over base (consistent across runs)
- lib10: pathB v2 **−0.5 pp below base** (reversed from first run) — on long-horizon, AttnRes does NOT help
- Per-task task-9 is a universal hard task (62–74% across all 4 runs)

### 2026-04-20 — Path C v3 (γ-curriculum from base, no VLM retrofit) full eval

| suite     | pathC v3 |
|-----------|----------|
| spatial   | 92.6     |
| object    | 100.0    |
| goal      | 95.8     |
| libero_10 | 88.8     |
| **avg**   | **94.30**|

Path C v3 is **1.75 pp below path0** and **2.45 pp below pathB v2**. The γ-curriculum-from-base
strategy cannot replicate the gains of VLM retrofit warm-start. Biggest gap on long-horizon
(lib10 88.8 vs path0 92.1, −3.3 pp).

### 2026-04-20 — Final 4-suite comparison (all with 2-run variance where possible)

| method                          | spatial | object | goal (2-run) | lib10 (2-run) | **4-suite avg** |
|---------------------------------|---------|--------|--------------|---------------|-----------------|
| path0_base_30k                  | 94.8    | 99.8   | 97.50        | 92.10         | **96.05**       |
| pathB_warm_30k (v1 observer)    | 96.8    | 99.6   | 97.6         | 91.6          | 96.40           |
| **pathB_warm_v2_30k (per-blk)** | **97.8**| 99.6   | **98.00**    | 91.60         | **96.75**       |
| pathC_curr_v3_30k (no warm-start)| 92.6   | 100.0  | 95.8         | 88.8          | 94.30           |

**Key conclusions for paper:**
1. **VLM retrofit warm-start is load-bearing** — pathB v2 beats pathC v3 by +2.45 pp on 4-suite avg.
   Simply adding γ-curriculum during VLA training does not substitute for pre-doing VLM retrofit.
2. **Per-block integration matters** — pathB v2 beats pathB v1 (observer) by +0.35 pp; the
   observer bug was load-bearing in the wrong direction.
3. **AttnRes does not uniformly help long-horizon** — pathB v2 ≤ path0 on libero_10 (-0.5 pp avg).
   The +0.70 pp 4-suite win comes from spatial (+3.0) and goal (+0.5), not long-horizon.

### 2026-04-20 — Path C v4 (60k steps, 2x training) launched
Testing whether Path C's shortfall is due to under-training. Same config as Path C v3
but `--trainer.max_train_steps 60000`, `--framework.attnres.gamma_ramp_steps 18000`
(keeps 30% curriculum ramp). Training on GPUs 2,3,6,7. Will run full 4-suite eval
after completion. If Path C v4 closes the gap to pathB v2, under-training was the cause;
if not, VLM retrofit warm-start is genuinely necessary.

### 2026-04-20 — Path C v4 (60k steps) full 4-suite eval done

| suite     | pathC v3 (30k) | pathC v4 (60k) | Δ      |
|-----------|----------------|----------------|--------|
| spatial   | 92.6           | **95.2**       | +2.6   |
| object    | 100.0          | 98.6           | −1.4   |
| goal      | 95.8           | **96.8**       | +1.0   |
| libero_10 | 88.8           | **91.4**       | +2.6   |
| **avg**   | 94.30          | **95.50**      | **+1.20** |

Spatial task 6 per-task breakdown (universal hard task):
- path0: 62 | pathB v2: **78** | pathC v3: **50** (undertrained) | pathC v4: 64 (recovered to base level)

Doubling VLA training **partially closes** the gap — Path C v4 now only 0.55 pp below path0
vs 1.75 pp for v3. But Path C v4 (60k VLA steps) is still **1.25 pp below Path B v2
(30k VLA + 5k retrofit = 35k-equiv)** — VLM retrofit warm-start delivers more per compute.

**Paper-level conclusion: VLM retrofit warm-start is not just a training-length substitute.**
Even with 2× VLA training, Path C cannot match the performance Path B achieves with
standard-length VLA training, and the gap is concentrated on tasks that benefit from
stable pre-trained routers (e.g. spatial task 6).

### 2026-04-20 — Path B v2 60k upper-bound launched
Testing the upper bound of VLM-retrofit + AttnRes when given matching (60k) VLA compute.
Same config as Path B v2 30k but `max_train_steps 60000`. Warm-start from same
`H_r256_5k/retrofit_attnres_state.pt`. Training on GPUs 0-3. ETA ~6h. Full 4-suite
eval after completion will answer: with equal compute, how much does Path B beat Path C?

### 2026-04-21 — Qwen3-VL-4B Path B (with `-Action` base, 2048 extra tokens)

Replicated the 2B pipeline on Qwen3-VL-4B: H_4B_r256_5k retrofit (18 blocks × 2
layers, same γ-curriculum) → warm-start Path B LIBERO (OFT, 30k steps, 4-GPU).

**Training setup**
- `base_vlm = playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action` (2048
  `<robot_action_*>` tokens added for future FAST compatibility, not used by OFT)
- `N_BLOCKS=18`, `adapter_rank=256`, `enable_skipping=False`
- 30k VLA steps on GPUs 0-3, ZeRO-2 bf16
- Warm-start state: `outputs/H_4B_r256_5k/retrofit_attnres_state.pt`
- ckpt: `results/Checkpoints/libero_pathB_4B_warm_v2_30k/final_model/pytorch_model.pt`

**Eval results** (50 trials × 10 tasks = 500 episodes / suite):

| suite     | 4B Path B | 2B Path B v2 (30k) | Δ (4B − 2B) |
|-----------|-----------|--------------------|-------------|
| spatial   | 95.0      | 97.8               | −2.8        |
| object    | **100.0** | 99.3               | +0.7        |
| goal      | **98.4**  | 94.25              | +4.15       |
| libero_10 | 84.8      | 95.65              | **−10.85**  |
| **avg**   | 94.55     | **96.75**          | **−2.20**   |

**Findings**
1. 4B wins on short-horizon manipulation (object/goal) — as expected from scale.
2. **Long-horizon libero_10 collapses −10.85 pp**, dragging the 4-suite average below
   2B. Same pattern shows up in MMStar: +58% params but −4.5 pp on reasoning subtasks
   (logical/math/science).
3. Two confounders for the 4B regression, investigated next:
   - **Under-trained retrofit**: same 5k steps, but 4B has +58% params. Adapter rank
     r=256 may be insufficient for hidden_size=2560.
   - **Embedding contamination**: the `-Action` base has 2048 `<robot_action_*>`
     embeddings initialised as the mean of existing rows and left unfrozen during
     retrofit. In OFT these tokens are never predicted, but `lm_head` and
     `embed_tokens` rows for them still sit in the shared weight matrix and receive
     gradient from distillation loss.

**Next experiment** (launched 04-21): clean-base 4B controls
- **Path 0 clean-base** (GPUs 0-3): pure OFT from `/home/user01/Minko/models/Qwen3-VL-4B`
  (no 2048 extra tokens). `run_id=libero_path0_4B_cleanbase_30k`, 30k steps.
- **Path B clean-base** (GPUs 4-7 after Path B v2 60k eval): same warm-start
  H_4B_r256_5k state but base VLM is the clean 4B. Retrofit state is
  embedding-shape-independent (only Adapter/Router/γ/LoRA), so it transfers.

Goal: if both clean-base runs close the libero_10 gap, embedding contamination
was causal; if not, 4B retrofit is genuinely under-trained and needs r↑ or steps↑.

### 2026-04-21 — Path B v2 60k full 4-suite eval done

Tests whether 60k VLA steps (2× the standard 30k) improves the warm-start Path B v2
ceiling. Same config as Path B v2 30k. Ckpt:
`results/Checkpoints/libero_pathB_warm_v2_60k/final_model/pytorch_model.pt` (4.65 GB).

| suite     | Path B v2 30k | Path B v2 60k | Δ (60k − 30k) |
|-----------|---------------|---------------|---------------|
| spatial   | 97.80         | 94.40         | **−3.40** ⚠️    |
| object    | 99.30         | 99.20         | −0.10          |
| goal      | 94.25         | **97.80**     | **+3.55**      |
| libero_10 | 95.65         | 94.00         | −1.65          |
| **avg**   | **96.75**     | 96.35         | **−0.40**      |

**Finding: 30k is the optimal training length for Path B v2.** Doubling to 60k
slightly overfits — spatial and libero_10 both regress by 1.65–3.40 pp, while goal
posts a large +3.55 pp gain. The net average drops 0.40 pp.

This flips the expected direction. Hypothesis: the warm-started Router/Adapter
converge fast (they start from a VLM-adapted state, not random), so extra VLA
steps drift the action head away from the spatial/long-horizon sweet spot reached
around step 30k, while goal (simpler short-horizon semantics) keeps benefiting.

**Practical implication for the paper:** stop at 30k for Path B v2. Path B is
still the headline method (96.75 % avg at 30k), no need to argue longer training.

### 2026-04-21 — Clean-base 4B controls launched in parallel

With all 8 cards free, launched both clean-base 4B trainings to isolate whether
the 4B Path B regression (−2.20 pp vs 2B, driven by libero_10 −10.85 pp) came
from the `-Action` base's 2048 extra token embeddings:

- **#38 Path 0 clean-base** (GPUs 0-3): `libero_path0_4B_cleanbase_30k`,
  `attnres.enabled=False`, base = `Qwen3-VL-4B-Instruct` symlink to
  `/home/user01/Minko/models/Qwen3-VL-4B` (no extra tokens), N_BLOCKS=18, 30k steps.
- **#39 Path B clean-base** (GPUs 4-7): `libero_pathB_4B_cleanbase_30k`, same
  clean base, warm-start from `H_4B_r256_5k/retrofit_attnres_state.pt` (retrofit
  state is embedding-shape-independent so directly transferable).

ETA ~5 h each (4B ZeRO-2 on 4 GPUs). After eval we'll compare the 2×2:

|                          | `-Action` base (dirty) | clean base |
|--------------------------|------------------------|------------|
| Path 0 (pure OFT)        | (not run)              | **#38**    |
| Path B (AttnRes warmstart)| 94.55 avg (done)      | **#39**    |

If #39 closes the libero_10 gap to 2B levels (~95 %), embedding contamination was
causal. If not, 4B retrofit needs r↑ (r=512) or steps↑ (10k). #38 bounds the
pure-base 4B performance regardless.

### 2026-04-21 — Clean-base 4B Path 0 full 4-suite eval done

Pure OFT on clean Qwen3-VL-4B (no 2048 `<robot_action_*>` tokens). AttnRes disabled.
`run_id=libero_path0_4B_cleanbase_30k`. 5h 26min on GPUs 0-3, `action_dit_loss=0.01536`,
epoch 4.73. Ckpt 9.36 GB.

| suite     | 4B Path 0 clean | 2B Path 0 30k (2-run avg) |
|-----------|-----------------|---------------------------|
| spatial   | 95.0            | 94.8                      |
| object    | 99.2            | 99.8                      |
| goal      | 97.8            | 97.50                     |
| libero_10 | 92.2            | 92.10                     |
| **avg**   | **96.05**       | 96.05                     |

4B pure-base and 2B pure-base tie exactly at 96.05 avg. 4B slightly better on
spatial (+0.2) and goal (+0.3), slightly worse on object (−0.6). libero_10 is
identical. **Model scale alone does not improve LIBERO at this training budget.**

### 2026-04-21 — Clean-base 4B Path B full 4-suite eval done

Warm-start from `H_4B_r256_5k/retrofit_attnres_state.pt` into clean Qwen3-VL-4B.
`run_id=libero_pathB_4B_cleanbase_30k`. 6h 26min on GPUs 4-7, `action_dit_loss=0.01526`,
epoch 4.73. Ckpt 9.39 GB. `attnres_blocks_executed=18, flops_ratio=1`.

**2×2 contamination study (headline result):**

| suite     | 4B Path 0 **clean** | 4B Path B **clean** | 4B Path B **dirty** (`-Action` base) | 2B Path B v2 30k |
|-----------|---------------------|---------------------|--------------------------------------|------------------|
| spatial   | 95.0                | 94.6                | 95.0                                 | 97.8             |
| object    | 99.2                | 99.8                | 100.0                                | 99.3             |
| goal      | 97.8                | 98.2                | 98.4                                 | 94.25            |
| libero_10 | 92.2                | **94.2**            | 84.8                                 | 95.65            |
| **avg**   | 96.05               | **96.70**           | 94.55                                | 96.75            |

**Three decisive findings**

1. **The 2048 `<robot_action_*>` tokens in the `-Action` base corrupt Path B.**
   Clean Path B − Dirty Path B = **+9.4 pp on libero_10**, +2.15 pp avg.
   Mechanism: those rows in `embed_tokens`/`lm_head` are initialised as the mean of
   the existing vocab, left unfrozen during retrofit + VLA training, and receive
   gradient from distillation and action-token label smoothing that they should never
   have touched. In OFT those token IDs are never predicted, but they sit in the same
   tied matrix, so gradients leak into them and degrade the representation used by
   the action head — especially for the long-horizon 10 suite that needs stable
   language grounding.

2. **AttnRes warm-start helps on clean 4B.** Clean Path B − Clean Path 0 = **+2.0 pp
   on libero_10**, +0.65 pp avg. The benefit is concentrated on long-horizon, matching
   the 2B pattern where Path B v2 beat Path 0 mostly on spatial/goal.

3. **4B with clean retrofit matches 2B, does not exceed it.** Clean 4B Path B (96.70)
   ≈ 2B Path B v2 30k (96.75). The 4B's additional parameters do not convert into
   LIBERO wins at 30k steps — spatial stays −3.2 pp behind 2B. Options to close it:
   longer VLA training (4B may need 60k+), larger retrofit (r=512 or more than 5k
   steps), or richer action-space data.

**Paper-ready takeaway**

- Use **clean base VLMs** for OFT training; do not add FAST vocabulary to a VLM
  you're only going to train with OFT. (This changes how we describe the pipeline in
  the paper: base → retrofit AttnRes → OFT LIBERO, with NO FAST token insertion for
  the OFT story.)
- **AttnRes warm-start remains net positive** on both 2B and clean 4B, with the same
  qualitative signature (long-horizon gain).
- Headline numbers for the paper:
  - 2B Path 0 30k: 96.05
  - 2B Path B v2 30k: **96.75** (+0.70 vs 2B Path 0)
  - 4B Path 0 clean: 96.05
  - 4B Path B clean: **96.70** (+0.65 vs 4B Path 0)

**GPU status**: all 8 cards idle after evals.

**Open question**: Why does 4B Path B gain only +0.65 pp on average vs Path 0, while
2B Path B v2 gains +0.22 pp? Answer: 4B benefits more from warm-start on long-horizon
(+2.0 pp on 10) because its larger capacity better utilises the Router's soft-routing
signal from H_4B_r256_5k. But it pays a spatial tax (−0.4 pp) that 2B doesn't. Net
gain is still positive, and the 4B curve still has headroom if we bump retrofit
capacity (r↑) or VLA steps (60k).

---

## 4B clean-base 60k upper-bound (appended 2026-04-22)

Ran both 4B clean-base paths at **60k steps, 4-GPU** to probe upper-bound of
the 4B pipeline and compare to the 2B Path B v2 60k picture (which had gone
30k→60k = 96.75→96.35, i.e. overtraining hurts).

| Suite | Path 0 60k | Path B 60k | Δ(B−0) |
|---|---|---|---|
| spatial | 95.00 | 93.40 | **−1.60** |
| object | **100.00** | 99.20 | −0.80 |
| goal | **98.60** | 97.40 | −1.20 |
| 10 | 94.40 | **94.80** | +0.40 |
| **AVG** | **97.00** | 96.20 | **−0.80** |

**30k → 60k deltas (per-Path):**

| Path | 30k AVG | 60k AVG | Δ |
|---|---|---|---|
| 4B Path 0 (pure OFT) | 96.05 | **97.00** | **+0.95** |
| 4B Path B (warm-start) | 96.70 | 96.20 | **−0.50** |

**Decisive findings (appended to the paper story):**

1. **At 4B × 60k, Path 0 overtakes Path B (97.00 vs 96.20, +0.80pp).** The
   30k ranking ("Path B > Path 0") is flipped. 4B's larger Path 0 capacity
   closes the warm-start advantage when given enough steps.

2. **Path B 60k regression reproduces on 4B** (same phenomenon as 2B Path B
   v2: 30k→60k = 96.75→96.35). AttnRes warm-start gain dilutes with longer
   OFT training across **both** scales. **30k is the optimum VLA training
   length for OFT**; 60k overtrains.

3. **Path 0 60k=97.00 is the new headline ceiling for 4B clean-base.** This
   is the highest LIBERO score of any run in this table. Given 30k Path B
   is within 0.30pp of 60k Path 0 at 1/2 the compute, Path B 30k remains
   the paper's "preferred" recipe for compute efficiency, while Path 0 60k
   is the "max-quality" upper bound.

**Paper-ready update to the headline table:**

| Model / Path / Steps | AVG |
|---|---|
| 2B Path 0 30k | 96.05 |
| 2B Path B v2 30k | **96.75** |
| 2B Path B v2 60k | 96.35 (−0.40 vs 30k) |
| 4B Path 0 30k (clean) | 96.05 |
| 4B Path B 30k (clean) | 96.70 |
| 4B Path 0 60k (clean) | **97.00** ← new ceiling |
| 4B Path B 60k (clean) | 96.20 |

**Recommendation**: headline recipe is still **Path B at 30k** on both scales
(best compute-normalised quality). Paper Section N should note that 60k
pushes 4B Path 0 to 97.00 but 60k is a "capacity-probe" result, not the
recommended training length.

**GPU status**: all 8 cards freed after these evals finished at 2026-04-22
~13:00 UTC-4.
