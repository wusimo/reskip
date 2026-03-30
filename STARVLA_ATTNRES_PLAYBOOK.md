# StarVLA AttnRes Playbook

This document covers the real VLA path in this repo:

- `StarVLA` training on LIBERO
- `AttnRes` routing over Qwen hidden-state blocks
- rollout evaluation through the native websocket policy server

## What Is Implemented

The current integration is real StarVLA training and real LIBERO rollout evaluation.

It adds:

- `QwenOFT + AttnRes`
- `QwenGR00T + AttnRes`
- full-depth training
- representation-level AttnRes routing during training
- true decoder-block skipping during inference and rollout
- uniform skip evaluation
- modality-aware skip evaluation
- training/eval logging for executed blocks and effective block ratio

Important:

- training still runs full-depth Qwen-VL for stability
- inference-time skip now patches `self.model.language_model.layers` and skips whole decoder blocks
- the online skip decision currently uses `mean_abs_delta(hidden_states, routed_summary)` as the block score
- this gives real decoder-layer compute savings during rollout
- if `attnres_backbone_compute_preserved=false`, the server actually skipped backbone blocks

## Key Files

- `src/starvla_integration.py`
- `starVLA/starVLA/model/framework/QwenOFT.py`
- `starVLA/starVLA/model/framework/QwenGR00T.py`
- `starVLA/starVLA/training/train_starvla.py`
- `starVLA/deployment/model_server/server_policy.py`
- `starVLA/examples/LIBERO/eval_files/model2libero_interface.py`
- `starVLA/examples/LIBERO/eval_files/eval_libero.py`
- `starVLA/examples/LIBERO/train_files/starvla_cotrain_libero_attnres.yaml`

## Training

Run from the outer repo root:

```bash
bash starVLA/examples/LIBERO/train_files/run_libero_train_attnres.sh
```

This script launches:

- framework: `QwenOFT`
- dataset: `LIBERO`
- routing: `AttnRes enabled`
- skip during training: `off`

You can also launch directly:

```bash
accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  starVLA/training/train_starvla.py \
  --config_yaml ./starVLA/examples/LIBERO/train_files/starvla_cotrain_libero_attnres.yaml \
  --framework.attnres.enabled True \
  --framework.attnres.enable_skipping False
```

## Policy Server

Run from the nested `starVLA/` repo or use the helper script:

```bash
bash examples/LIBERO/eval_files/run_policy_server_attnres.sh
```

Direct launch:

```bash
python deployment/model_server/server_policy.py \
  --ckpt_path results/Checkpoints/libero_qwenoft_attnres/checkpoints/steps_50000_pytorch_model.pt \
  --port 5694 \
  --use_bf16 \
  --skip_mode none
```

To enable uniform skip:

```bash
python deployment/model_server/server_policy.py \
  --ckpt_path results/Checkpoints/libero_qwenoft_attnres/checkpoints/steps_50000_pytorch_model.pt \
  --port 5694 \
  --use_bf16 \
  --enable_skipping \
  --skip_mode uniform \
  --uniform_skip_threshold 0.01
```

To enable modality-aware skip:

```bash
python deployment/model_server/server_policy.py \
  --ckpt_path results/Checkpoints/libero_qwenoft_attnres/checkpoints/steps_50000_pytorch_model.pt \
  --port 5694 \
  --use_bf16 \
  --enable_skipping \
  --skip_mode modality_aware \
  --vision_skip_threshold 0.02 \
  --language_skip_threshold 0.01 \
  --action_skip_threshold 0.005
```

## LIBERO Rollout

From the nested `starVLA/` repo:

```bash
bash examples/LIBERO/eval_files/eval_libero_attnres.sh
```

Direct launch:

```bash
python examples/LIBERO/eval_files/eval_libero.py \
  --args.pretrained-path results/Checkpoints/libero_qwenoft_attnres/checkpoints/steps_50000_pytorch_model.pt \
  --args.host 127.0.0.1 \
  --args.port 5694 \
  --args.task-suite-name libero_goal \
  --args.num-trials-per-task 50 \
  --args.video-out-path results/libero_goal/attnres_eval \
  --args.enable-skipping false \
  --args.skip-mode none
```

Uniform skip rollout:

```bash
python examples/LIBERO/eval_files/eval_libero.py \
  --args.pretrained-path results/Checkpoints/libero_qwenoft_attnres/checkpoints/steps_50000_pytorch_model.pt \
  --args.host 127.0.0.1 \
  --args.port 5694 \
  --args.task-suite-name libero_goal \
  --args.num-trials-per-task 50 \
  --args.video-out-path results/libero_goal/uniform_skip \
  --args.enable-skipping true \
  --args.skip-mode uniform \
  --args.uniform-skip-threshold 0.01
```

Modality-aware rollout:

```bash
python examples/LIBERO/eval_files/eval_libero.py \
  --args.pretrained-path results/Checkpoints/libero_qwenoft_attnres/checkpoints/steps_50000_pytorch_model.pt \
  --args.host 127.0.0.1 \
  --args.port 5694 \
  --args.task-suite-name libero_goal \
  --args.num-trials-per-task 50 \
  --args.video-out-path results/libero_goal/modality_skip \
  --args.enable-skipping true \
  --args.skip-mode modality_aware \
  --args.vision-skip-threshold 0.02 \
  --args.language-skip-threshold 0.01 \
  --args.action-skip-threshold 0.005
```

## Recommended Experiment Order

1. Train `QwenOFT + AttnRes` with skipping disabled.
2. Run LIBERO rollout with `skip_mode=none`.
3. Run LIBERO rollout with `skip_mode=uniform`.
4. Run LIBERO rollout with `skip_mode=modality_aware`.
5. Compare:
   - success rate
   - mean effective block ratio
   - failure cases

## Remaining Work

The main remaining gap is experimental calibration rather than plumbing:

- tune skip thresholds per task suite
- benchmark actual latency on your deployment GPU
- compare `uniform` vs `modality_aware` against the same checkpoint
