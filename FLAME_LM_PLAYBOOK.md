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



训练 340M ReSkip。

你的训练参数模板就按 111.txt 来，把其中 --model.config 指到：
flame/configs/reskip_transformer_340M.json

命令形式是：
bash train.sh \
    --model.config /home/user01/Minko/reskip2/reskip/flame/configs/reskip_transformer_340M.json \
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
    --training.gradient_accumulation_steps 4 \
    --training.steps 35000 \
    --training.skip_nan_inf \
    --training.seq_len 32768 \
    --training.dataset /home/user01/Minko/datasets/fineweb_edu_100BT \
    --training.dataset_split train \
    --training.seed 0 \
    --checkpoint.interval 5000 \
    --metrics.log_freq 1 \
    --checkpoint.folder /home/user01/Minko/reskip2/reskip/flame/saves/reskip_transformer_340M/checkpoint \
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

#6卡
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 bash train.sh \
    --model.config /home/user01/Minko/reskip2/reskip/flame/configs/reskip_transformer_340M.json \
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
    --training.gradient_accumulation_steps 6 \
    --training.steps 35000 \
    --training.skip_nan_inf \
    --training.seq_len 32768 \
    --training.dataset /home/user01/Minko/datasets/fineweb_edu_100BT \
    --training.dataset_split train \
    --training.seed 0 \
    --checkpoint.interval 3500 \
    --metrics.log_freq 1 \
    --checkpoint.folder /home/user01/Minko/reskip2/reskip/flame/saves/reskip_transformer_340M-3/checkpoint \
    --training.num_workers 8 \
    --training.prefetch_factor 2 \
    --checkpoint.export_dtype bfloat16 \
    --checkpoint.enable_checkpoint \
    --checkpoint.load_step -1 \
    --training.data_parallel_shard_degree 6 \
    --activation_checkpoint.mode none \
    --training.streaming \
    --training.varlen \
    --metrics.enable_wandb


CUDA_VISIBLE_DEVICES=7,0,1,2,3,4,5,6  bash train.sh \
    --model.config /home/user01/Minko/reskip2/reskip/flame/configs/reskip_transformer_340M.json \
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
    --training.gradient_accumulation_steps 4 \
    --training.steps 32768 \
    --training.skip_nan_inf \
    --training.seq_len 32768 \
    --training.dataset /home/user01/Minko/datasets/fineweb_edu_100BT \
    --training.dataset_split train \
    --training.seed 0 \
    --checkpoint.interval 5000 \
    --metrics.log_freq 1 \
    --checkpoint.folder /home/user01/Minko/reskip2/reskip/flame/saves/reskip_transformer_340M-4/checkpoint \
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
python -m flame.utils.convert_dcp_to_hf --path /home/user01/Minko/reskip2/reskip/flame/saves/reskip_transformer_340M --step 35000 --config /home/user01/Minko/reskip2/reskip/flame/configs/reskip_transformer_340M.json  --tokenizer /home/user01/Minko/models/gla-tokenizer

python -m flame.utils.convert_dcp_to_hf --path /home/user01/Minko/reskip2/reskip/flame/saves/reskip_transformer_340M-2 --step 15000 --config /home/user01/Minko/reskip2/reskip/flame/configs/reskip_transformer_340M.json  --tokenizer /home/user01/Minko/models/gla-tokenizer

训练完做 routing analysis，并导出 skip-ready checkpoint：
python experiments/flame_analyze_reskip.py \
  --model_path /home/user01/Minko/reskip2/reskip/flame/saves/reskip_transformer_340M \
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
生成测试：
python experiments/flame_generate.py \
  --model_path outputs/reskip_340M_skip_ready \
  --prompt "The capital of France is" \
  --max_new_tokens 32 \
  --device cuda
跑 lm_eval：
python experiments/flame_lm_eval.py \
  --model_path outputs/reskip_340M_skip_ready \
  --tasks lambada_openai,hellaswag,arc_easy,arc_challenge \
  --batch_size auto \
  --device cuda:1 \
  --output_path outputs/lm_eval_reskip_340M

python experiments/flame_lm_eval.py \
  --model_path /home/user01/Minko/reskip2/reskip/flame/saves/reskip_transformer_340M-2 \
  --tasks lambada_openai,hellaswag,arc_easy,arc_challenge \
  --batch_size auto \
  --device cuda:6 \
  --output_path outputs/lm_eval_base_attnres_340M-2
如果跑 ReLoop

训练时把配置改成：

flame/configs/reloop_transformer_340M.json
其他流程一样。

现在这版你可以直接上服务器跑首轮实验。最稳妥的顺序是先短训验证 train -> analyze -> generate -> lm_eval 全链路，再开正式长训。



CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NCCL_DEBUG=INFO \
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
torchrun --standalone --nproc_per_node=8 repro_nccl_peer_fault.py --iters 200 --rows-per-rank 32768 --hidden-size 2048  --dtype bf16
