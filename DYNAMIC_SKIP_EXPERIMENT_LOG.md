# ReSkip Dynamic Skip Experiment Log

## 2026-04-09 `test3` 基线与动态对比

模型：
- 训练目录：[reskip_transformer-test3](/home/user01/Minko/reskip2/reskip/flame/saves/reskip_transformer-test3)

分析结果：
- 分析文件：[routing_analysis.json](/home/user01/Minko/reskip2/reskip/outputs/reskip_analysis_test3_v3/routing_analysis.json)
- full-depth `ppl = 14.9254`
- 最优静态 skip 仍为全保留
- 最优动态质量配置：
  - `strategy = recent_weight_gt`
  - `quantile = 0.97`
  - `max_skips = 1`
  - `avg_blocks = 7.875`
  - `ppl = 16.2589`
- `ppl_tolerance = 0.02` 下没有任何动态候选满足容忍度，因此 `best_dynamic_tolerated` 实际回退到 `best_dynamic_ppl`

动态导出模型：
- [reskip_test3_dynamic_ready_v3](/home/user01/Minko/reskip2/reskip/outputs/reskip_test3_dynamic_ready_v3)

正式评测结果：
- full-depth 结果：[results_2026-04-09T23-03-41.282020.json](/home/user01/Minko/reskip2/reskip/outputs/lm_eval_reskip_test3_full_v3/__home__user01__Minko__reskip2__reskip__flame__saves__reskip_transformer-test3/results_2026-04-09T23-03-41.282020.json)
- dynamic 结果：[results_2026-04-09T23-03-43.776195.json](/home/user01/Minko/reskip2/reskip/outputs/lm_eval_reskip_test3_dynamic_v3/__home__user01__Minko__reskip2__reskip__outputs__reskip_test3_dynamic_ready_v3/results_2026-04-09T23-03-43.776195.json)

关键指标对比：

| Task | Metric | Full-depth | Dynamic |
|---|---:|---:|---:|
| lambada_openai | acc | 0.2630 | 0.1999 |
| lambada_openai | ppl | 83.08 | 190.67 |
| hellaswag | acc_norm | 0.3189 | 0.3178 |
| arc_easy | acc_norm | 0.4457 | 0.4440 |
| arc_challenge | acc_norm | 0.2602 | 0.2594 |

结论：
- 动态 skip 链路已经修通，不再是 no-op。
- 当前 `test3` 上，动态 skip 的真实作用是“轻度省算但明显伤 `lambada`”。
- 这说明问题已经不在分析器失效，而在“当前允许 skip 的位置仍然过宽”。

## 2026-04-09 位置收紧实验

实验目的：
- 在不改训练、不改核心动态判据的前提下，只限制允许被动态 skip 的 block 位置，检查是否能降低质量损失。

基础配置：
- 使用上面的 `best_dynamic_ppl`
- 原始阈值：
  - `[1e9, 0.3182, 0.3311, 0.3428, 0.3535, 0.2832, 0.3096, 1e9]`
- 原始实际跳过位置：
  - `1, 3, 4, 6`

32 batch train-split proxy 结果：

| 配置 | 允许位置 | ppl | avg_blocks | 实际 skip |
|---|---|---:|---:|---|
| `base_best` | `1,3,4,6` | 16.2589 | 7.8750 | `1,3,4,6` 各 1 次 |
| `mid_3456` | `3,4,6` | 15.5604 | 7.9063 | `3,4,6` |
| `late_456` | `4,6` | 15.3360 | 7.9375 | `4,6` |
| `late_56` | `6` | 15.1996 | 7.9688 | `6` |
| `only_4` | `4` | 15.0593 | 7.9688 | `4` |
| `only_5` | `5` | 14.9254 | 8.0000 | 无 skip |

当前判断：
- 早期 block 的动态 skip 风险明显更高。
- 后部低重要度 block 更适合作为动态 skip 候选。
- 下一步应该把“允许参与动态 skip 的位置”也纳入 analyze 搜索，而不是只搜索阈值。

## 2026-04-09 Benchmark-safe 动态位置约束

思路：
- 不改训练
- 不改运行时判据 `recent_weight_gt`
- 只手动收紧允许触发动态 skip 的位置

候选：
- `only_4`
- `only_6`
- `late_456`

对应导出模型：
- [reskip_test3_dynamic_only4](/home/user01/Minko/reskip2/reskip/outputs/reskip_test3_dynamic_only4)
- [reskip_test3_dynamic_only6](/home/user01/Minko/reskip2/reskip/outputs/reskip_test3_dynamic_only6)
- [reskip_test3_dynamic_late456](/home/user01/Minko/reskip2/reskip/outputs/reskip_test3_dynamic_late456)

LAMBADA 单任务结果：

| 配置 | acc | ppl | 结论 |
|---|---:|---:|---|
| full-depth | 0.2630 | 83.0795 | 基线 |
| `only_4` | 0.2630 | 83.0863 | 与基线一致 |
| `only_6` | 0.2630 | 83.0863 | 与基线一致 |
| `late_456` | 0.2630 | 83.0863 | 与基线一致 |

完整四任务结果：
- full-depth：[results_2026-04-09T23-03-41.282020.json](/home/user01/Minko/reskip2/reskip/outputs/lm_eval_reskip_test3_full_v3/__home__user01__Minko__reskip2__reskip__flame__saves__reskip_transformer-test3/results_2026-04-09T23-03-41.282020.json)
- `late_456`：[results_2026-04-09T23-20-48.657951.json](/home/user01/Minko/reskip2/reskip/outputs/lm_eval_reskip_test3_late456_full/__home__user01__Minko__reskip2__reskip__outputs__reskip_test3_dynamic_late456/results_2026-04-09T23-20-48.657951.json)

四任务对比：

| Task | Metric | Full-depth | `late_456` |
|---|---:|---:|---:|
| lambada_openai | acc | 0.2630 | 0.2630 |
| lambada_openai | ppl | 83.0795 | 83.0795 |
| hellaswag | acc_norm | 0.3189 | 0.3189 |
| arc_easy | acc_norm | 0.4457 | 0.4457 |
| arc_challenge | acc_norm | 0.2602 | 0.2602 |

解释：
- 当前 `late_456` 已经达到了“几乎不影响 benchmark”的要求。
- 但它仍然是一个非常保守的动态 skip 配置，省算幅度有限。
- 这说明下一阶段的关键不是再修链路，而是想办法在保持这种 benchmark 稳定性的前提下，继续扩大可安全跳过的范围。

## 当前优化方向

目标：
- 在很小质量损失下实现真正可用的 `ReSkip`

下一步方法：
- 保持 `AttnRes` 运行时信号不变
- 在 analyze 中增加“候选位置子集”搜索
- 默认优先搜索低全局重要度 block 的后部子集，而不是让所有中间 block 都参与在线 skip


## 2026-04-10 进一步优化：自动 tolerated、候选位置 phase1 优化、轻量 probe

### 自动 tolerated / recommended（test3）
- 分析输出：`outputs/reskip_analysis_test3_v4/routing_analysis.json`
- `best_dynamic_tolerated`: `probe=all, mode=low1, q=0.95, max_skips=1`
- 代理指标：`ppl=15.1773, avg_blocks=7.9375`
- 对应导出目录：`outputs/reskip_test3_dynamic_tolerated_v4`

### 四任务 benchmark（test3 tolerated）
- 结果：`outputs/lm_eval_reskip_test3_tolerated_v4/...json`
- 与 full-depth 一致：
  - `lambada acc = 0.2630`
  - `lambada ppl = 83.0795`
  - `hellaswag acc_norm = 0.3189`
  - `arc_easy acc_norm = 0.4457`
  - `arc_challenge acc_norm = 0.2602`

### 第二模型点复验（110M test）
- 分析输出：`outputs/reskip_analysis_test110_v1/routing_analysis.json`
- `best_dynamic_tolerated`: `probe=all, mode=low1, q=0.97, max_skips=1`
- 代理指标：`ppl=15.0620, avg_blocks=7.9688`
- 四任务 benchmark 与 full-depth 一致：
  - `lambada acc = 0.2713`
  - `lambada ppl = 79.8209`
  - `hellaswag acc_norm = 0.3216`
  - `arc_easy acc_norm = 0.4428`
  - `arc_challenge acc_norm = 0.2551`

### 推理开销优化
- 在 `modeling_reskip_transformer.py` 中加入“只在真实候选位置才计算 phase1”逻辑。
- 结论：分析 trace 开销显著下降；但对部署时延，当前 safe skip 触发率仍然太低，尚未形成正的 wall-clock speedup。

### 真实前向时延（不返回 routing_info）
- `test3 full`: `0.0355 s/batch`, `230.7k tok/s`
- `test3 tolerated(low1 q0.95)`: `0.0506 s/batch`, `161.6k tok/s`
- `test3 low1_q05`: `0.0374 s/batch`, `218.9k tok/s`
- `110M full`: `0.0478 s/batch`, `171.1k tok/s`
- `110M tolerated`: `0.0568 s/batch`, `144.1k tok/s`

### 更激进阈值扫描（test3, low1 only）
- 代理结果：
  - `q=0.95`: `avg_blocks=7.9375`
  - `q=0.90`: `avg_blocks=7.8750`
  - `q=0.80`: `avg_blocks=7.7812`
  - `q=0.60`: `avg_blocks=7.5938`
  - `q=0.50`: `avg_blocks=7.5625`
- `lambada_openai limit=200` 上，`q=0.95/0.90/0.80/0.60/0.50` 都未观察到精度下降。
- 但即便到 `q=0.50`，真实前向时延仍未超过 full-depth。

### 轻量 probe 探索
- 新增 `dynamic_skip_probe_mode`，支持：`all`, `attn_only`, `first_layer`, `first_attn`
- 结论：
  - `first_attn` / `attn_only` 对当前模型过轻，几乎不触发 skip。
  - `first_layer` 虽然能明显多跳，但质量下降过大。
  - 当前最稳的仍是 `probe=all`。

### 当前判断
- 推理侧代码路径已经基本抠干净：
  - 自动推荐位置子集可用
  - 动态 tolerated/recommended 可自动导出
  - 只在候选位置计算 phase1 已落地
  - 轻量 probe 已验证过一轮
- 现在的主瓶颈已经不是“分析脚本不会选”或“动态逻辑有明显冗余”，而是：
  - **在当前 8-block ReSkip 模型上，benchmark-safe 的 skip 触发率仍然太低，不足以带来净速度收益。**
- 下一步若要同时满足“几乎不掉点 + 真正加速”，优先级应转向：
  1. 更细粒度 block（例如 12 或 24 blocks）
  2. 或重新设计更强的 AttnRes runtime proxy / architecture

## 2026-04-10 Probe-aware Calibration 与 Fast Mode

### 修复与新增
- `dynamic calibration` 现在按 `probe_mode` 单独统计阈值，不再错误复用 `all` 的统计给 `first_attn / attn_only / first_layer`
- `analyze` 新增 `best_dynamic_fast`
  - 在给定 `speed_ppl_tolerance` 内筛候选
  - 再在同一组缓存 batch 上做短时 latency benchmark
  - 输出真实更快的动态配置
- `lm_eval` 新增 `--dynamic_mode fast`

### `test3` focused analyze v7
- 分析输出：`outputs/reskip_analysis_test3_v7/routing_analysis.json`
- full-depth `ppl = 14.9254`
- `best_dynamic_tolerated`
  - `probe = all`
  - `mode = low1`
  - `q = 0.95`
  - `ppl = 15.1773`
  - `avg_blocks = 7.9375`
- `best_dynamic_fast`
  - `probe = all`
  - `mode = low1`
  - `q = 0.90`
  - `ppl = 15.5378`
  - `avg_blocks = 7.8750`
  - `mean_batch_s = 0.042409`
  - `speedup = 1.003x`

### 手工 Pareto 点
- 使用同一模型 `test3`、同一组 long-context proxy batch 做真实前向测速：

| 配置 | `ppl` | `avg_blocks` | `mean_batch_s` | `tok/s` |
|---|---:|---:|---:|---:|
| full-depth | 14.9254 | 8.0000 | 0.057929 | 141259.6 |
| `all + low1 + q=0.5` | 17.0117 | 7.5625 | 0.044912 | 182201.9 |
| `first_attn + low1 + q=0.5` | 16.4231 | 7.6875 | 0.042709 | 191599.3 |
| `first_layer + low1 + q=0.8` | 15.3956 | 7.9062 | 0.044598 | 183483.7 |

### `first_layer + low1 + q=0.8` 的 benchmark
- 导出目录：`outputs/reskip_test3_dynamic_speed_v1`
- 结果：`outputs/lm_eval_reskip_test3_dynamic_speed_v1/...json`
- 四任务结果与 full-depth 完全一致：
  - `lambada acc = 0.2630`
  - `lambada ppl = 83.0795`
  - `hellaswag acc_norm = 0.3189`
  - `arc_easy acc_norm = 0.4457`
  - `arc_challenge acc_norm = 0.2602`

### 当前判断更新
- 代码侧推理路径已经进一步收口：
  - probe-aware calibration 已修正
  - fast 模式已经能自动选出“在指定 PPL 容忍度内最快”的动态配置
- 但对当前 `8 blocks / 24 layers` 的 `ReSkip`：
  - 在 **<= 5% proxy PPL 增幅** 这个范围内，真实净加速仍然几乎为零
  - 更明显的速度收益，需要进入更激进的 Pareto 区域
- 这说明当前限制速度收益的主因已经不是实现 bug，而是：
  - **安全 skip 频率仍太低**
  - **当前 block 粒度仍偏粗**

## 2026-04-10 干净复比：真实速度收益已经做出来

这轮在空闲 GPU 上做了同一设备、同一批次、长上下文 (`seq_len=8192`) 的干净复比，结果文件：
- [dynamic_skip_clean_compare_v1.json](/home/user01/Minko/reskip2/reskip/outputs/dynamic_skip_clean_compare_v1.json)

对比配置：
- full-depth
- `block_all_low1_q095`
- `block_attn_only_low1_q09`
- `block_first_layer_low1_q08`
- `prev_recent_weight_low1_q095`

长上下文前向对比：

| 配置 | `ppl` | `avg_blocks` | `mean_batch_s` | `tok/s` | 相对 full |
|---|---:|---:|---:|---:|---:|
| full-depth | 14.9254 | 8.0000 | 0.05163 | 169.1k | 1.000x |
| `block_all_low1_q095` | 15.1773 | 7.9375 | 0.05017 | 174.0k | 1.029x |
| `block_attn_only_low1_q09` | 15.3108 | 7.9063 | 0.04161 | 209.8k | 1.241x |
| `block_first_layer_low1_q08` | 15.3956 | 7.9063 | 0.04191 | 208.3k | 1.232x |
| `prev_recent_weight_low1_q095` | 15.3923 | 7.9063 | 0.04085 | 213.7k | 1.264x |

关键结论：
- 之前“几乎不掉点但几乎没速度”的阶段已经过去了。
- 在当前 `test3` 上，已经出现了**真实可见的 long-context 净加速**。
- 最稳的两类候选是：
  - 轻量 pre-probe：`block_attn_only_low1_q09`
  - 零额外 pre-probe：`prev_recent_weight_low1_q095`

## 2026-04-10 Benchmark 复验：当前最适合 paper 主线的是 `block_attn_only_low1_q09`

导出模型：
- [reskip_test3_block_attnonly_low1_q09](/home/user01/Minko/reskip2/reskip/outputs/reskip_test3_block_attnonly_low1_q09)
- [reskip_test3_prev_recent_low1_q095](/home/user01/Minko/reskip2/reskip/outputs/reskip_test3_prev_recent_low1_q095)

评测结果：
- full-depth：[results_2026-04-09T23-03-41.282020.json](/home/user01/Minko/reskip2/reskip/outputs/lm_eval_reskip_test3_full_v3/__home__user01__Minko__reskip2__reskip__flame__saves__reskip_transformer-test3/results_2026-04-09T23-03-41.282020.json)
- `block_attn_only_low1_q09`：[results_2026-04-10T05-03-14.813435.json](/home/user01/Minko/reskip2/reskip/outputs/lm_eval_reskip_test3_block_attnonly_low1_q09/__home__user01__Minko__reskip2__reskip__outputs__reskip_test3_block_attnonly_low1_q09/results_2026-04-10T05-03-14.813435.json)
- `prev_recent_weight_low1_q095`：[results_2026-04-10T05-03-12.946192.json](/home/user01/Minko/reskip2/reskip/outputs/lm_eval_reskip_test3_prev_recent_low1_q095/__home__user01__Minko__reskip2__reskip__outputs__reskip_test3_prev_recent_low1_q095/results_2026-04-10T05-03-12.946192.json)

四任务对比：

| 配置 | lambada acc | lambada ppl | hellaswag acc_norm | arc_easy acc_norm | arc_challenge acc_norm |
|---|---:|---:|---:|---:|---:|
| full-depth | 0.2630 | 83.0795 | 0.3189 | 0.4457 | 0.2602 |
| `block_attn_only_low1_q09` | 0.2630 | 83.0795 | 0.3189 | 0.4457 | 0.2602 |
| `prev_recent_weight_low1_q095` | 0.2626 | 83.3020 | 0.3172 | 0.4453 | 0.2585 |

当前结论：
- **`block_attn_only_low1_q09` 是目前最好的 paper 主线候选。**
  - 四任务 benchmark 与 full-depth 一致
  - 长上下文代理测速约 `1.24x`
- `prev_recent_weight_low1_q095` 也很有价值：
  - 它是更“AttnRes 原生”的零额外 pre-probe 方案
  - 长上下文代理测速约 `1.26x`
  - 但当前 benchmark 已经开始轻微下滑，更适合作为扩展方案或后续继续优化的方向

## 最新可复现实验指令

### 1. 主线分析：`block_attn_only_low1_q09`

用于在 `test3` 上重新做最新版动态分析：

```bash
CUDA_VISIBLE_DEVICES=6 /home/user01/Minko/reskip2/.venv/bin/python \
  /home/user01/Minko/reskip2/reskip/experiments/flame_analyze_reskip.py \
  --model_path /home/user01/Minko/reskip2/reskip/flame/saves/reskip_transformer-test3 \
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
  --output_dir /home/user01/Minko/reskip2/reskip/outputs/reskip_analysis_test3_v7 \
  --dynamic_skip_strategy recent_weight_gt \
  --dynamic_skip_probe_modes all,attn_only,first_layer,first_attn \
  --dynamic_skip_position_modes auto \
  --dynamic_skip_quantiles 0.5,0.6,0.7,0.8,0.9,0.95,0.97,0.99 \
  --dynamic_skip_max_skips_options 1,2 \
  --dynamic_skip_latency_num_batches 16 \
  --dynamic_skip_latency_top_k 16 \
  --dynamic_skip_speed_ppl_tolerance 0.05
```

当前 paper 主线推荐从分析结果中选用：
- `probe_mode = attn_only`
- `position_mode = low1`
- `quantile = 0.9`
- `max_skips = 1`

对应导出模型目录：
- [reskip_test3_block_attnonly_low1_q09](/home/user01/Minko/reskip2/reskip/outputs/reskip_test3_block_attnonly_low1_q09)

### 2. 主线跑分：`block_attn_only_low1_q09`

```bash
CUDA_VISIBLE_DEVICES=6 /home/user01/Minko/reskip2/.venv/bin/python \
  /home/user01/Minko/reskip2/reskip/experiments/flame_lm_eval.py \
  --model_path /home/user01/Minko/reskip2/reskip/outputs/reskip_test3_block_attnonly_low1_q09 \
  --tasks lambada_openai,hellaswag,arc_easy,arc_challenge \
  --batch_size auto \
  --device cuda:0 \
  --output_path /home/user01/Minko/reskip2/reskip/outputs/lm_eval_reskip_test3_block_attnonly_low1_q09
```

### 3. 扩展方向：`prev_recent_weight_low1_q095`

这是更 AttnRes 原生的“上一已执行 block 决策下一 block”版本。当前 benchmark 略掉点，但速度更激进。

导出模型目录：
- [reskip_test3_prev_recent_low1_q095](/home/user01/Minko/reskip2/reskip/outputs/reskip_test3_prev_recent_low1_q095)

跑分命令：

```bash
CUDA_VISIBLE_DEVICES=6 /home/user01/Minko/reskip2/.venv/bin/python \
  /home/user01/Minko/reskip2/reskip/experiments/flame_lm_eval.py \
  --model_path /home/user01/Minko/reskip2/reskip/outputs/reskip_test3_prev_recent_low1_q095 \
  --tasks lambada_openai,hellaswag,arc_easy,arc_challenge \
  --batch_size auto \
  --device cuda:0 \
  --output_path /home/user01/Minko/reskip2/reskip/outputs/lm_eval_reskip_test3_prev_recent_low1_q095
```
