# Flame/FLA LM Playbook

## New Implementation

The new LM implementation now lives in FLA itself:

- `flash-linear-attention/fla/models/reskip_transformer/`
- `flame/configs/reskip_transformer_340M.json`
- `flame/configs/reloop_transformer_340M.json`

This keeps the architecture change inside FLA’s transformer family, while flame remains the trainer/orchestrator.

Implemented behavior:

- Block AttnRes now follows the paper implementation more closely: routing uses the paper's learned pseudo-query plus RMS-normalized block states, without the extra router-side `key_proj` that the paper does not use.
- ReSkip now uses the paper's two-phase block execution path during training: completed-block routing for all sites in a block is batched first, then each site merges that result with the current `partial_block` online.
- ReSkip uses the draft’s block importance definition `I(n) = max_{l>n} α_{n→l)` aggregated from downstream block-routing events, while deployment/eval still uses the calibration-based keep-mask path.
- ReLoop now uses shared block groups, depth-position-specific AttnRes routers, and ACT-style halting with optional ponder-cost regularization.
- Skip-ready checkpoints can be exported after routing analysis and then used directly for generation or `lm_eval`.

## Training

Use the repo-root `train.sh`, which forwards into flame and resolves the monorepo paths correctly.

### 340M AttnRes

```bash
bash train.sh \
  --job.config_file flame/flame/models/fla.toml \
  --job.dump_folder exp/reskip-transformer-340M \
  --model.config flame/configs/reskip_transformer_340M.json \
  --model.tokenizer_path /home/user01/Minko/models/gla-tokenizer \
  --optimizer.name AdamW \
  --optimizer.eps 1e-15 \
  --optimizer.lr 0.0007 \
  --lr_scheduler.warmup_steps 3500 \
  --lr_scheduler.lr_min 0.00007 \
  --lr_scheduler.decay_type cosine \
  --lr_scheduler.decay_ratio 0.2 \
  --training.batch_size 1 \
  --training.context_len 2048 \
  --training.gradient_accumulation_steps 2 \
  --training.steps 35000 \
  --training.skip_nan_inf \
  --training.seq_len 65536 \
  --training.dataset /home/user01/Minko/datasets/fineweb_edu_100BT \
  --training.dataset_split train \
  --training.seed 0 \
  --checkpoint.interval 5000 \
  --metrics.log_freq 1 \
  --checkpoint.folder /home/user01/Minko/reskip/flame/saves/reskip-340M/checkpoint \
  --training.num_workers 8 \
  --training.prefetch_factor 2 \
  --checkpoint.export_dtype bfloat16 \
  --checkpoint.enable_checkpoint \
  --checkpoint.load_step -1 \
  --training.data_parallel_shard_degree 8 \
  --activation_checkpoint.mode none \
  --training.streaming \
  --training.varlen \
  --metrics.enable_wandb
```

### 340M ReLoop

Replace:

```bash
--model.config flame/configs/reskip_transformer_340M.json
```

with:

```bash
--model.config flame/configs/reloop_transformer_340M.json
```

## Routing Analysis And Skip Export

After `train.sh` finishes, flame will export a HuggingFace-format checkpoint under the dump folder.

Run routing analysis and export a calibrated skip-ready checkpoint:

```bash
python experiments/flame_analyze_reskip.py \
  --model_path exp/reskip-transformer-340M \
  --dataset /home/user01/Minko/datasets/fineweb_edu_100BT \
  --dataset_split train \
  --seq_len 2048 \
  --context_len 2048 \
  --batch_size 1 \
  --num_batches 128 \
  --num_workers 4 \
  --streaming \
  --output_dir outputs/reskip_analysis \
  --export_best_model_dir outputs/reskip_340M_skip_ready \
  --device cuda
```

This produces:

- `outputs/reskip_analysis/routing_analysis.json`
- `outputs/reskip_340M_skip_ready/` as a new HF checkpoint with `skip_keep_mask` written into config

## Inference

Full-depth or skip-ready generation:

```bash
python experiments/flame_generate.py \
  --model_path outputs/reskip_340M_skip_ready \
  --prompt "The capital of France is" \
  --max_new_tokens 32 \
  --device cuda
```

If you want to override the keep mask manually:

```bash
python experiments/flame_generate.py \
  --model_path exp/reskip-transformer-340M \
  --prompt "The capital of France is" \
  --keep_mask 1,1,1,0,0,1,1,1 \
  --max_new_tokens 32 \
  --device cuda
```

## lm_eval

Use the wrapper so `fla` gets imported before `lm_eval` instantiates the HF model:

```bash
python experiments/flame_lm_eval.py \
  --model_path outputs/reskip_340M_skip_ready \
  --tasks lambada_openai,hellaswag,arc_easy,arc_challenge \
  --batch_size auto \
  --device cuda:0 \
  --output_path outputs/lm_eval_reskip_340M
```

## Notes

- `reskip_transformer_340M.json` is the main pretraining config. The filename is kept for continuity with earlier experiments, but after removing the non-paper router projection the exact parameter count is lower than the historical name suggests; use the training log as the source of truth.
- `reloop_transformer_340M.json` is the weight-shared loop variant for the loop ablation line and enables ACT halting.
- In looping mode, KV cache is disabled intentionally because shared blocks reuse logical layer positions.
- Activation checkpointing, per-block compile, and FSDP for `reskip_transformer` now follow the block-group boundary, because the two-phase AttnRes work is executed at the block level rather than at individual layer-call boundaries.
- The skip path used for deployment/eval is calibrated keep-mask skipping, which matches the draft’s practical implementation section.
- `ponder_loss_weight` controls the ACT depth penalty in looping mode.
