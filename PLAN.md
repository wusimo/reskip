# ReSkip 统一实验计划

## 1. 文档目的

本文件是当前仓库唯一保留的实验规划文档，统一整理以下内容：

- 已实现的模型与训练路径
- 已确认的问题与修复结论
- 当前推荐的训练、导出、评测命令
- 后续实验顺序与记录规范

根目录其余 playbook 已合并进本文件。

---

## 2. 当前仓库状态

### 2.1 主要实现

- `ReSkip LM`
  路径：`flash-linear-attention/fla/models/reskip_transformer/`
- `ReLoop LM`
  路径：`flash-linear-attention/fla/models/reskip_transformer/`
- `flame + FLA` 训练链路
  路径：`flame/`
- `StarVLA + AttnRes`
  路径：`starVLA/`

### 2.2 当前建议使用的训练框架

- 正式训练统一使用当前仓库内的 `flame`
- 不再使用单独拉下来的 `baseflame`

原因：

- 当前仓库已经包含 `ReSkip/ReLoop` 的适配与验证结果
- 训练态与 checkpoint 态不一致的问题已经在当前仓库内修复
- `baseflame` 只作为对照与定位工具，不再需要继续保留

---

## 3. 已解决的关键问题

### 3.1 历史问题

之前 `ReSkip` 训练中出现过明显异常：

- 训练日志中的 loss 很低
- 导出的 checkpoint 拿出来离线前向时，loss 明显更高
- 导致无法判断训练是否真实有效

典型表现是：

- 训练过程中日志 loss 接近 `2.x ~ 4.x`
- checkpoint 离线前向仍在 `6.x ~ 7.x`

### 3.2 最终定位

最终确认根因不在：

- `reskip` 的 forward/loss 公式
- `HF 导出`
- `lm-eval` 脚本本身

真正根因在 **TorchTitan checkpoint 保存路径**：

- `ModelWrapper.state_dict()` 在构造时缓存了一次 model state
- 后续定期保存 checkpoint 时继续使用旧缓存
- 导致训练时看的 live model 与磁盘里保存的 checkpoint 不是同一份权重

### 3.3 最终修复

修复位置：

- [train.py](/home/user01/Minko/reskip2/reskip/flame/flame/train.py)

修复方式：

- 训练启动时 monkey-patch `torchtitan.components.checkpoint.ModelWrapper.state_dict`
- 改为每次保存 checkpoint 前都重新从 live model 抽取最新 state dict

### 3.4 修复验证结论

已做 fresh run 验证。

在同一份训练后 microbatch 上：

- 训练后 replay loss
- 从刚保存的 checkpoint 重新加载后的 replay loss

已经精确对齐，误差约为 `1e-4` 量级。

结论：

- 对 **新训练产生的 checkpoint**，训练态与 checkpoint 态已对齐
- 旧的异常 checkpoint 不会被自动修复，不再作为正式结果使用

---

## 4. 当前推荐实验主线

建议按下面顺序推进。

### 4.1 主线 A：ReSkip 340M 训练

目标：

- 验证 `ReSkip` 在当前 flame/FLA 链路下的稳定训练
- 周期性抽 checkpoint 做一致性与 `lm-eval` 检查

### 4.2 主线 B：ReLoop 340M 训练

目标：

- 验证共享块循环版本在相同训练框架下是否稳定
- 对比 `ReSkip` 与 `ReLoop` 的 loss、评测、导出链路

### 4.3 主线 C：StarVLA + AttnRes

目标：

- 在真实 VLA 骨干中验证 AttnRes 路由与跳块
- 后续重点放在 rollout 成功率、时延与 skip 策略比较

---

## 5. ReSkip 训练方案

### 5.1 配置文件

- [reskip_transformer_340M.json](/home/user01/Minko/reskip2/reskip/flame/configs/reskip_transformer_340M.json)

### 5.2 推荐正式训练命令

下面这条命令是当前建议的标准模板。

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

### 5.3 说明

- 当前主线更推荐 `6 卡`
- `seq_len=32768`
- `context_len=2048`
- `streaming + varlen`
- `checkpoint.interval=5000`

---

## 6. ReLoop 训练方案

### 6.1 配置文件

- [reloop_transformer_340M.json](/home/user01/Minko/reskip2/reskip/flame/configs/reloop_transformer_340M.json)

### 6.2 推荐训练命令

与 `ReSkip` 相同，只需要替换配置文件：

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

### 6.3 当前配置内的额外项

`ReLoop` 目前已经支持：

- halting / depth regularization 相关配置
- 共享块循环
- routing 统计

---

## 7. checkpoint 导出与一致性检查

### 7.1 训练结束后的导出

`train.sh` 会自动把最后一步 checkpoint 导出成 HF 格式。

导出的典型内容包括：

- `model.safetensors`
- `config.json`
- `tokenizer.json`

### 7.2 单卡离线 loss 检查

脚本：

- [compare_fsdp_loss.py](/home/user01/Minko/reskip2/reskip/experiments/compare_fsdp_loss.py)

用途：

- 对 raw DCP checkpoint 做单卡或指定 rank 的离线前向
- 检查 checkpoint loss 是否跟训练趋势一致

示例：

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

### 7.3 推荐检查节奏

每次正式训练建议在以下 checkpoint 做检查：

- `step 200`
- `step 1000`
- `step 5000`
- `step 10000`

检查内容：

- 训练日志 loss
- raw checkpoint 离线 loss
- HF 导出后 loss
- `lm-eval` 小样本 smoke test

---

## 8. lm-eval 评测

脚本：

- [flame_lm_eval.py](/home/user01/Minko/reskip2/reskip/experiments/flame_lm_eval.py)

示例：

```bash
CUDA_VISIBLE_DEVICES=6 python experiments/flame_lm_eval.py \
  --model_path /home/user01/Minko/reskip2/reskip/flame/saves/reskip_transformer_340M \
  --tasks lambada_openai,hellaswag,arc_easy,arc_challenge \
  --batch_size auto \
  --device cuda:0 \
  --output_path /home/user01/Minko/reskip2/reskip/outputs/lm_eval_reskip_340M
```

建议：

- 先做小样本 smoke test
- 确认训练态 / checkpoint 态一致后，再跑完整任务集

---

## 9. StarVLA + AttnRes 实验线

### 9.1 目标

把 AttnRes / skip 机制迁移到真实 VLA 骨干，比较：

- full depth
- uniform skip
- modality-aware skip

### 9.2 主要路径

- [src/starvla_integration.py](/home/user01/Minko/reskip2/reskip/src/starvla_integration.py)
- [starVLA](/home/user01/Minko/reskip2/reskip/starVLA)

### 9.3 推荐顺序

1. 先训 full-depth AttnRes backbone
2. 跑 `skip_mode=none` rollout
3. 跑 `skip_mode=uniform`
4. 跑 `skip_mode=modality_aware`
5. 比较 success rate、effective block ratio 与时延

---

## 10. 实验记录规范

建议每条正式实验至少记录以下信息：

- 模型配置文件
- 训练命令全文
- 训练起止时间
- 使用 GPU 编号
- checkpoint 路径
- 离线 loss 检查结果
- `lm-eval` 结果
- 是否为正式 run 或 smoke run

推荐保存位置：

- 正式结果写入 `PLAN.md` 的对应章节
- 临时说明写入 [1232.txt](/home/user01/Minko/reskip2/reskip/1232.txt)

---

## 11. 当前结论

### 11.1 已确认

- `ReSkip` 当前训练链路可正常训练
- checkpoint 保存态与训练态不一致的问题已修复
- 新训练生成的 checkpoint 可用于离线前向与 HF 导出

### 11.2 不再建议

- 不再使用旧的异常 checkpoint 作为结果依据
- 不再继续维护根目录多份重复 playbook
- 不再继续使用单独的 `baseflame`

### 11.3 下一步建议

优先级最高的是：

1. 启动新的正式 `ReSkip 340M` 长训
2. 在固定 checkpoint 做一致性检查
3. 再进行完整 `lm-eval`
4. 然后推进 `ReLoop` 与 `StarVLA` 线
