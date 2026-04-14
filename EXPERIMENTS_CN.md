# 实验总览与执行说明

> **Document summary (English)**: This is the primary Chinese-language guide for training and evaluating ReSkip/ReLoop. It covers training commands (340M, 6-GPU), checkpoint consistency checks, lm-eval workflow (full-depth / static skip / dynamic skip), and the StarVLA+AttnRes experiment line. For a concise English quick-reference, see [FLAME_LM_PLAYBOOK.md](FLAME_LM_PLAYBOOK.md). For the dynamic skip mechanism explanation, see [DYNAMIC_SKIP_MECHANISM.md](DYNAMIC_SKIP_MECHANISM.md).

## 1. 文档定位

本文件用于统一整理当前仓库里真正需要执行和记录的实验内容，重点覆盖：

- `ReSkip` 语言模型训练
- `ReLoop` 语言模型训练
- checkpoint 导出与一致性验证
- `lm-eval` 评测
- `StarVLA + AttnRes` 实验线
- 当前已经确认的关键问题与修复结论

这个文件是执行层面的中文实验说明，不替代原有的 [PLAN.md](/home/user01/Minko/reskip2/reskip/PLAN.md) 和 [README.md](/home/user01/Minko/reskip2/reskip/README.md)。

---

## 2. 当前仓库结论

### 2.1 建议使用哪套训练框架

现在正式训练统一使用当前仓库里的：

- [flame](/home/user01/Minko/reskip2/reskip/flame)
- [flash-linear-attention](/home/user01/Minko/reskip2/reskip/flash-linear-attention)

不再使用单独拉下来的 `baseflame`。

### 2.2 已确认修复的问题

之前 `ReSkip` 训练里最关键的问题是：

- 训练日志的 loss 明显下降
- 但导出的 checkpoint 拉出来做离线前向时，loss 对不上

这个问题已经确认根因是 checkpoint 保存链路，而不是模型前向公式本身。

根因：

- TorchTitan 的 `ModelWrapper.state_dict()` 在初始化时缓存了一份 model state
- 后续周期性保存 checkpoint 时继续使用旧缓存
- 导致训练时看的 live model 与磁盘里保存的 checkpoint 不是同一份权重

修复位置：

- [flame/train.py](/home/user01/Minko/reskip2/reskip/flame/flame/train.py)

修复方式：

- 在训练启动时 monkey-patch `ModelWrapper.state_dict()`
- 每次保存前都重新从 live model 抓最新 state dict

### 2.3 修复后的验证结论

已经做过 fresh run 验证：

- 训练后同批次 replay 的 loss
- 从同一步 checkpoint 重新加载后的 replay loss

可以对齐到 `1e-4` 量级。

因此：

- 新训练生成的 checkpoint 可以正常用于离线检查、HF 导出、`lm-eval`
- 旧的异常 checkpoint 不再作为正式结果依据

### 2.4 当前 ReSkip analysis 结论

目前 `ReSkip` 的分析结论已经更新：

- 不再把“全局静态删一个 block”当成唯一 skip 方案
- 当前更推荐的部署/评测路径是：
  - 先做 `reskip analysis`
  - 再导出带默认动态 skip 配置的 HF 模型
  - 再直接对这个动态模型跑 `lm-eval`

已经验证过：

- 静态删除任意单个中间 block，通常都会让 PPL 恶化较多
- 但基于 AttnRes 运行时 `phase1` 统计的动态 skip，可以在更接近 full-depth 的质量下减少平均 block 数
- 这条动态 skip 路径不需要额外训练，只依赖 AttnRes 自身的运行时信号

---

## 3. 当前主要实验线

建议按下面顺序推进。

### 3.1 主线 A：ReSkip 340M

目标：

- 训练主模型
- 验证 checkpoint 与训练态一致
- 跑 `lm-eval`

### 3.2 主线 B：ReLoop 340M

目标：

- 比较共享块循环后的训练稳定性
- 比较 loss、评测与参数复用效果

### 3.3 主线 C：StarVLA + AttnRes

目标：

- 在真实 VLA 骨干中接入 AttnRes
- 比较 full depth / uniform skip / modality-aware skip

---

## 4. ReSkip 训练

### 4.1 配置文件

- [reskip_transformer_340M.json](/home/user01/Minko/reskip2/reskip/flame/configs/reskip_transformer_340M.json)

### 4.2 推荐正式训练命令

```bash
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
    --checkpoint.interval 5000 \
    --metrics.log_freq 1 \
    --checkpoint.folder /home/user01/Minko/reskip2/reskip/flame/saves/reskip_transformer_340M/checkpoint \
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
```

### 4.3 当前建议

- 正式训练优先使用 `6 卡`
- `seq_len=32768`
- `context_len=2048`
- `streaming + varlen`

---

## 5. ReLoop 训练

### 5.1 配置文件

- [reloop_transformer_340M.json](/home/user01/Minko/reskip2/reskip/flame/configs/reloop_transformer_340M.json)

### 5.2 推荐训练命令

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 bash train.sh \
    --model.config /home/user01/Minko/reskip2/reskip/flame/configs/reloop_transformer_340M.json \
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
    --checkpoint.interval 5000 \
    --metrics.log_freq 1 \
    --checkpoint.folder /home/user01/Minko/reskip2/reskip/flame/saves/reloop_transformer_340M/checkpoint \
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
```

### 5.3 当前配置已支持的能力

- 共享块循环
- halting 相关配置
- routing 统计

---

## 6. checkpoint 一致性检查

### 6.1 目标

每个正式 run 都应该验证：

- 训练态 loss 是否正常下降
- raw checkpoint 的离线前向 loss 是否与训练趋势一致
- HF 导出后的 loss 是否与 raw checkpoint 一致

### 6.2 使用脚本

- [compare_fsdp_loss.py](/home/user01/Minko/reskip2/reskip/experiments/compare_fsdp_loss.py)

### 6.3 推荐命令

```bash
CUDA_VISIBLE_DEVICES=6 torchrun --nproc_per_node=1 \
  experiments/compare_fsdp_loss.py \
  --checkpoint_dir /home/user01/Minko/reskip2/reskip/flame/saves/reskip_transformer_340M/checkpoint \
  --config_path /home/user01/Minko/reskip2/reskip/flame/configs/reskip_transformer_340M.json \
  --tokenizer_path /home/user01/Minko/models/gla-tokenizer \
  --dataset_path /home/user01/Minko/datasets/fineweb_edu_100BT \
  --step 5000 \
  --seq_len 32768 \
  --context_len 2048 \
  --data_parallel_degree 6 \
  --data_parallel_rank 0
```

### 6.4 推荐检查步点

- `step 200`
- `step 1000`
- `step 5000`
- `step 10000`

---

## 7. HF 导出

`train.sh` 会在训练完成后自动导出 HF 权重。

典型导出结果包括：

- `model.safetensors`
- `config.json`
- `tokenizer.json`

如果只想手动导出，也可以使用 flame 自带转换脚本。

---

## 8. lm-eval

### 8.1 使用脚本

- [flame_lm_eval.py](/home/user01/Minko/reskip2/reskip/experiments/flame_lm_eval.py)
- [flame_analyze_reskip.py](/home/user01/Minko/reskip2/reskip/experiments/flame_analyze_reskip.py)

### 8.2 推荐流程

现在推荐把 `ReSkip` 的评测分成三种：

1. full-depth 基线
2. 静态 calibrated skip
3. 动态 runtime skip

其中第三种是当前最推荐的 `ReSkip` 跑分方式。

### 8.3 full-depth 跑分命令

```bash
CUDA_VISIBLE_DEVICES=6 python experiments/flame_lm_eval.py \
  --model_path /home/user01/Minko/reskip2/reskip/flame/saves/reskip_transformer_340M \
  --tasks lambada_openai,hellaswag,arc_easy,arc_challenge \
  --batch_size auto \
  --device cuda:0 \
  --output_path /home/user01/Minko/reskip2/reskip/outputs/lm_eval_reskip_340M
```

### 8.4 先做 dynamic analysis，再导出动态模型

```bash
CUDA_VISIBLE_DEVICES=6 python experiments/flame_analyze_reskip.py \
  --model_path /home/user01/Minko/reskip2/reskip/flame/saves/reskip_transformer_340M \
  --dataset /home/user01/Minko/datasets/fineweb_edu_100BT \
  --dataset_split train \
  --seq_len 8192 \
  --context_len 2048 \
  --batch_size 1 \
  --num_workers 2 \
  --num_batches 32 \
  --streaming \
  --varlen \
  --device cuda \
  --dtype bf16 \
  --output_dir /home/user01/Minko/reskip2/reskip/outputs/reskip_analysis_dynamic \
  --dynamic_skip_strategy recent_weight_gt \
  --dynamic_skip_quantiles 0.9,0.95,0.97 \
  --dynamic_skip_max_skips_options 1 \
  --export_best_dynamic_model_dir /home/user01/Minko/reskip2/reskip/outputs/reskip_340M_dynamic_skip_ready
```

这个目录导出后，`config.json` 会自带：

- `dynamic_skip_strategy`
- `dynamic_skip_position_thresholds`
- `dynamic_skip_max_skips`

因此后续直接拿这个目录跑分即可。

注意这里有 3 种 dynamic 选择：

- `best_dynamic_ppl`
  质量最优，但通常跳得不深
- `best_dynamic_skip`
  跳得最深，但可能明显掉分
- `best_dynamic_tolerated`
  在允许的 `ppl_tolerance` 以内，跳得最深

现在推荐把第三种作为默认部署/跑分方案。

### 8.5 动态 skip 直接跑分

```bash
CUDA_VISIBLE_DEVICES=6 python experiments/flame_lm_eval.py \
  --model_path /home/user01/Minko/reskip2/reskip/outputs/reskip_340M_dynamic_skip_ready \
  --tasks lambada_openai,hellaswag,arc_easy,arc_challenge \
  --batch_size auto \
  --device cuda:0 \
  --output_path /home/user01/Minko/reskip2/reskip/outputs/lm_eval_reskip_340M_dynamic
```

### 8.6 让 lm-eval 自动读取 analysis 并准备动态模型

如果不想手动维护单独的动态导出目录，也可以直接：

```bash
CUDA_VISIBLE_DEVICES=6 python experiments/flame_lm_eval.py \
  --analysis_json /home/user01/Minko/reskip2/reskip/outputs/reskip_analysis_dynamic/routing_analysis.json \
  --dynamic_mode tolerated \
  --prepared_model_dir /tmp/reskip_eval_dynamic \
  --tasks lambada_openai,hellaswag,arc_easy,arc_challenge \
  --batch_size auto \
  --device cuda:0 \
  --output_path /home/user01/Minko/reskip2/reskip/outputs/lm_eval_reskip_340M_dynamic
```

这个流程会：

1. 读取 `routing_analysis.json`
2. 自动准备一个带默认动态 skip 配置的 HF 模型目录
3. 直接调用 `lm-eval`

其中：

- `--dynamic_mode quality`
  选择质量最好的 dynamic 配置
- `--dynamic_mode tolerated`
  选择在 `ppl_tolerance` 内跳得最深的 dynamic 配置
- `--dynamic_mode deepest`
  选择跳得最深的 dynamic 配置，不保证质量
- `--dynamic_mode static`
  走静态 keep-mask 配置，但仍复用同一个评测入口

### 8.7 建议顺序

1. 先做 smoke test
2. 确认 checkpoint 一致性没问题
3. 先比较 full-depth 和 dynamic skip
4. 再跑完整任务集
5. 静态 keep-mask 结果只保留作对照，不再作为主方案

---

## 9. StarVLA + AttnRes

### 9.1 当前已实现方向

- `QwenOFT + AttnRes`
- `QwenGR00T + AttnRes`
- inference/rollout 阶段的 skip
- uniform skip 与 modality-aware skip

### 9.2 关键路径

- [src/starvla_integration.py](/home/user01/Minko/reskip2/reskip/src/starvla_integration.py)
- [starVLA](/home/user01/Minko/reskip2/reskip/starVLA)

### 9.3 推荐实验顺序

1. 先训 full-depth AttnRes backbone
2. rollout 跑 `skip_mode=none`
3. rollout 跑 `skip_mode=uniform`
4. rollout 跑 `skip_mode=modality_aware`
5. 比较 success rate、effective block ratio、时延

---

## 10. 结果记录建议

每条正式实验至少记录：

- 模型配置文件
- 训练命令全文
- GPU 编号
- checkpoint 路径
- 离线 loss 检查结果
- `lm-eval` 结果
- 是否为正式 run

建议：

- 长期记录写进这个文件
- 临时说明写在 [1232.txt](/home/user01/Minko/reskip2/reskip/1232.txt)

---

## 11. 当前推荐执行顺序

1. 启动新的正式 `ReSkip 340M` 长训
2. 在 `200 / 1000 / 5000 / 10000 step` 做 checkpoint 一致性检查
3. 对稳定 checkpoint 跑 `lm-eval`
4. 再推进 `ReLoop`
5. 最后做 `StarVLA + AttnRes`
