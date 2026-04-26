# ReSkip 1.3B 八卡全流程

## 1. 目标

本文件给出当前 `reskip` 主线的 `1.3B` 八卡完整流程，覆盖：

- 训练
- checkpoint 一致性检查
- ReSkip analysis
- full-depth `lm-eval`
- dynamic ReSkip `lm-eval`

默认目录：

- 配置文件：[reskip_transformer_1.3B.json](/home/user01/Minko/reskip2/reskip/flame/configs/reskip_transformer_1.3B.json)
- 训练输出目录：`/home/user01/Minko/reskip2/reskip/flame/saves/reskip_transformer_1.3B`
- 动态导出目录：`/home/user01/Minko/reskip2/reskip/outputs/reskip_1p3B_dynamic_ready`
- analysis 输出目录：`/home/user01/Minko/reskip2/reskip/outputs/reskip_analysis_1p3B`

## 2. 模型规格

当前 `1.3B` 版本直接沿用仓库现有 `transformer_1B.json` 这一套大模型基线尺寸，并切到 `reskip_transformer`：

- `hidden_size = 2048`
- `num_hidden_layers = 24`
- `num_heads = 32`
- `attn_res_num_blocks = 8`

这里的 `1.3B` 命名与当前 Flame/FLA 项目自身习惯对齐，不再另外发明一套尺寸。

## 3. 八卡训练

建议训练环境：

- Python 环境：`/home/user01/Minko/reskip2/.venv`
- 工作目录：`/home/user01/Minko/reskip2/reskip/flame`
- 数据集：`/home/user01/Minko/datasets/fineweb_edu_100BT`
- tokenizer：`/home/user01/Minko/models/gla-tokenizer`

推荐起始训练命令：

```bash
cd /home/user01/Minko/reskip2/reskip/flame

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash train.sh \
    --model.config /home/user01/Minko/reskip2/reskip/flame/configs/reskip_transformer_1.3B.json \
    --model.tokenizer_path /home/user01/Minko/models/gla-tokenizer \
    --optimizer.name AdamW \
    --optimizer.eps 1e-15 \
    --optimizer.lr 0.0003 \
    --lr_scheduler.warmup_steps 4000 \
    --lr_scheduler.lr_min 0.00003 \
    --lr_scheduler.decay_type cosine \
    --lr_scheduler.decay_ratio 0.1 \
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
    --checkpoint.folder /home/user01/Minko/reskip2/reskip/flame/saves/reskip_transformer_1.3B/checkpoint \
    --training.num_workers 3 \
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

说明：

- `training.num_workers=3` 是为了避免 streaming dataset 再次出现 `dataset.num_shards=3` 的 worker 截断告警。
- 这套命令是当前 `340M` 主线的八卡放大版，重点是保持训练链路稳定，而不是先追极限吞吐。

## 4. checkpoint 一致性检查

正式训练后，先验证 live loss 和离线 checkpoint replay loss 是否一致。

推荐在单卡上做：

```bash
cd /home/user01/Minko/reskip2/reskip

CUDA_VISIBLE_DEVICES=7 torchrun --nproc_per_node=1 \
  experiments/compare_fsdp_loss.py \
  --checkpoint_dir /home/user01/Minko/reskip2/reskip/flame/saves/reskip_transformer_1.3B/checkpoint \
  --config_path /home/user01/Minko/reskip2/reskip/flame/configs/reskip_transformer_1.3B.json \
  --tokenizer_path /home/user01/Minko/models/gla-tokenizer \
  --dataset_path /home/user01/Minko/datasets/fineweb_edu_100BT \
  --step 5000 \
  --seq_len 32768 \
  --context_len 2048 \
  --data_parallel_degree 8 \
  --dataset_split train \
  --streaming \
  --varlen
```

建议至少检查：

- `step 5000`
- `step 10000`
- 最终 step

## 5. ReSkip Analysis

训练完成后，对 full-depth HF 导出模型做 routing analysis。

如果你的 HF 导出模型目录就是：

- `/home/user01/Minko/reskip2/reskip/flame/saves/reskip_transformer_1.3B`

那么推荐命令：

```bash
cd /home/user01/Minko/reskip2/reskip

CUDA_VISIBLE_DEVICES=7 python experiments/flame_analyze_reskip.py \
  --model_path /home/user01/Minko/reskip2/reskip/flame/saves/reskip_transformer_1.3B \
  --dataset /home/user01/Minko/datasets/fineweb_edu_100BT \
  --dataset_split train \
  --seq_len 8192 \
  --context_len 2048 \
  --batch_size 1 \
  --num_workers 2 \
  --num_batches 128 \
  --streaming \
  --varlen \
  --dynamic_skip_strategy recent_weight_gt \
  --dynamic_skip_granularity block \
  --dynamic_skip_probe_modes all,attn_only,first_attn \
  --dynamic_skip_position_modes auto \
  --dynamic_skip_quantiles 0.90,0.92,0.94,0.95,0.97 \
  --dynamic_skip_max_skips_options 1 \
  --ppl_tolerance 0.02 \
  --dynamic_skip_speed_ppl_tolerance 0.05 \
  --dynamic_skip_latency_num_batches 48 \
  --output_dir /home/user01/Minko/reskip2/reskip/outputs/reskip_analysis_1p3B \
  --export_best_dynamic_model_dir /home/user01/Minko/reskip2/reskip/outputs/reskip_1p3B_dynamic_ready
```

analysis 输出里重点关注：

- `best_ppl_metrics`
- `best_tolerated_metrics`
- `best_recommended_metrics`
- `best_speed_metrics`

当前主线建议优先看：

- 质量优先：`best_dynamic_tolerated`
- 部署优先：`best_dynamic_recommended`
- 速度优先：`best_dynamic_fast`

## 6. Full-depth lm-eval

先评 full-depth 基线：

```bash
cd /home/user01/Minko/reskip2/reskip

CUDA_VISIBLE_DEVICES=7 python experiments/flame_lm_eval.py \
  --model_path /home/user01/Minko/reskip2/reskip/flame/saves/reskip_transformer_1.3B \
  --tasks lambada_openai,hellaswag,arc_easy,arc_challenge \
  --batch_size auto \
  --device cuda:0 \
  --output_path /home/user01/Minko/reskip2/reskip/outputs/lm_eval_reskip_1p3B_full
```

## 7. Dynamic ReSkip lm-eval

### 7.1 直接评已经导出的动态模型

```bash
cd /home/user01/Minko/reskip2/reskip

CUDA_VISIBLE_DEVICES=7 python experiments/flame_lm_eval.py \
  --model_path /home/user01/Minko/reskip2/reskip/outputs/reskip_1p3B_dynamic_ready \
  --tasks lambada_openai,hellaswag,arc_easy,arc_challenge \
  --batch_size auto \
  --device cuda:0 \
  --output_path /home/user01/Minko/reskip2/reskip/outputs/lm_eval_reskip_1p3B_dynamic
```

### 7.2 从 analysis 自动准备并评测

如果想在跑分时明确选择方案，而不是依赖默认导出目录，直接用：

质量优先：

```bash
CUDA_VISIBLE_DEVICES=7 python experiments/flame_lm_eval.py \
  --model_path /home/user01/Minko/reskip2/reskip/flame/saves/reskip_transformer_1.3B \
  --analysis_json /home/user01/Minko/reskip2/reskip/outputs/reskip_analysis_1p3B/routing_analysis.json \
  --dynamic_mode tolerated \
  --prepared_model_dir /home/user01/Minko/reskip2/reskip/outputs/reskip_1p3B_dynamic_tolerated \
  --tasks lambada_openai,hellaswag,arc_easy,arc_challenge \
  --batch_size auto \
  --device cuda:0 \
  --output_path /home/user01/Minko/reskip2/reskip/outputs/lm_eval_reskip_1p3B_dynamic_tolerated
```

推荐部署配置：

```bash
CUDA_VISIBLE_DEVICES=7 python experiments/flame_lm_eval.py \
  --model_path /home/user01/Minko/reskip2/reskip/flame/saves/reskip_transformer_1.3B \
  --analysis_json /home/user01/Minko/reskip2/reskip/outputs/reskip_analysis_1p3B/routing_analysis.json \
  --dynamic_mode recommended \
  --prepared_model_dir /home/user01/Minko/reskip2/reskip/outputs/reskip_1p3B_dynamic_recommended \
  --tasks lambada_openai,hellaswag,arc_easy,arc_challenge \
  --batch_size auto \
  --device cuda:0 \
  --output_path /home/user01/Minko/reskip2/reskip/outputs/lm_eval_reskip_1p3B_dynamic_recommended
```

速度优先：

```bash
CUDA_VISIBLE_DEVICES=7 python experiments/flame_lm_eval.py \
  --model_path /home/user01/Minko/reskip2/reskip/flame/saves/reskip_transformer_1.3B \
  --analysis_json /home/user01/Minko/reskip2/reskip/outputs/reskip_analysis_1p3B/routing_analysis.json \
  --dynamic_mode fast \
  --prepared_model_dir /home/user01/Minko/reskip2/reskip/outputs/reskip_1p3B_dynamic_fast \
  --tasks lambada_openai,hellaswag,arc_easy,arc_challenge \
  --batch_size auto \
  --device cuda:0 \
  --output_path /home/user01/Minko/reskip2/reskip/outputs/lm_eval_reskip_1p3B_dynamic_fast
```

## 8. 建议记录的结果

`1.3B` 这条线至少记录下面几组结果：

- full-depth `lm-eval`
- `best_dynamic_tolerated`
- `best_dynamic_recommended`
- `best_dynamic_fast`

同时记录 analysis 里的：

- `perplexity`
- `avg_blocks`
- `speedup_vs_full`
- `probe_mode`
- `position_mode`
- `quantile`
- `max_skips`

## 9. 当前执行顺序

推荐严格按下面顺序走：

1. 八卡训练
2. 用 `compare_fsdp_loss.py` 验证 checkpoint 一致性
3. 用 `flame_analyze_reskip.py` 做 routing analysis 并导出 dynamic model
4. 跑 full-depth `lm-eval`
5. 跑 dynamic `lm-eval`
6. 对比 full / tolerated / recommended / fast 三组结果，再决定后续是否继续调 dynamic policy
