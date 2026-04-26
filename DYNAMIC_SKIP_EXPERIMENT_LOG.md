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

## 最新可复现实验指令（2026-04-16 更新）

对任意已训练好的 ReSkip 模型，完整流程为：**分析 → 导出 → 评测**。

### 前置条件

```bash
# 激活环境
source /home/user01/Minko/reskip2/.venv/bin/activate
cd /home/user01/Minko/reskip2/reskip

# 变量（按实际修改）
MODEL_PATH=flame/saves/reskip_transformer-340M
DATASET=/home/user01/Minko/datasets/fineweb_edu_100BT
GPU=6
```

### 第 1 步：路由分析 + 阈值校准

自动搜索最佳动态 skip 配置。分析脚本会：
1. 跑 full-depth 评测，得到 block importance 和 block ablation
2. 按 probe_mode × position_mode × quantile × max_skips 网格搜索
3. 输出 `routing_analysis.json`（包含所有候选配置的 PPL 和 skip 率）

```bash
CUDA_VISIBLE_DEVICES=$GPU python experiments/flame_analyze_reskip.py \
  --model_path $MODEL_PATH \
  --dataset $DATASET \
  --dataset_split train \
  --seq_len 8192 \
  --batch_size 1 \
  --num_batches 32 \
  --streaming \
  --device cuda \
  --dtype bf16 \
  --output_dir outputs/reskip_analysis \
  --dynamic_skip_strategy recent_weight_gt \
  --dynamic_skip_probe_modes attn_only,first_attn \
  --dynamic_skip_position_modes auto \
  --dynamic_skip_quantiles 0.8,0.85,0.9,0.93,0.95,0.97 \
  --dynamic_skip_max_skips_options 1,2,3 \
  --dynamic_skip_latency_num_batches 16 \
  --dynamic_skip_latency_top_k 8 \
  --dynamic_skip_speed_ppl_tolerance 0.05 \
  --export_best_dynamic_model_dir outputs/reskip_dynamic_best
```

`--dynamic_skip_position_modes auto` 会自动搜索以下位置集（含 ablation-informed）：
- `low1`/`low2`/`low3`：AttnRes importance 最低的 1/2/3 个 block
- `ablation1`/`ablation2`/`ablation3`：static removal PPL impact 最低的 1/2/3 个 block
- `recommended`/`all`/`late*`/`taillow*`：其他启发式组合

分析完成后查看推荐配置：

```bash
python -c “
import json
with open('outputs/reskip_analysis/routing_analysis.json') as f:
    d = json.load(f)
dsa = d['dynamic_skip_analysis']
for key in ['best_ppl_metrics', 'best_tolerated_metrics', 'best_speed_metrics']:
    m = dsa.get(key)
    if m:
        print(f'{key}: probe={m.get(\”probe_mode\”)} pos={m.get(\”position_mode\”)} '
              f'q={m.get(\”quantile\”)} max_skips={m.get(\”max_skips\”)} '
              f'ppl={m.get(\”perplexity\”,0):.4f} blocks={m.get(\”avg_blocks\”,0):.2f}')
“
```

### 第 2 步：手动导出指定配置（可选）

如果想导出分析脚本之外的自定义配置（如组合 ablation + importance 位置），使用导出脚本：

```bash
CUDA_VISIBLE_DEVICES=$GPU python experiments/export_ablation_skip_models.py \
  --model_path $MODEL_PATH \
  --dataset $DATASET \
  --seq_len 8192 \
  --cal_batches 32
```

默认导出三个配置：
- `reskip_340M_ablation1_skip1_q085`：block 3 only, max_skips=1（最安全）
- `reskip_340M_ablation2_skip2_q085`：blocks {2,3}, max_skips=2（更激进）
- `reskip_340M_ablation2_skip2_q093`：blocks {2,3}, max_skips=2, 更保守阈值

当前最佳配置 `combined {3,5}/skip2/q085` 的手动导出见本文后续章节。

### 第 3 步：四任务 Benchmark 评测

```bash
# 评测导出模型（dynamic skip 策略已保存在 config.json 中，自动生效）
CUDA_VISIBLE_DEVICES=$GPU python experiments/flame_lm_eval.py \
  --model_path outputs/reskip_dynamic_best \
  --tasks lambada_openai,hellaswag,arc_easy,arc_challenge \
  --batch_size auto \
  --device cuda:0 \
  --output_path outputs/lm_eval_reskip_dynamic_best

# 对比 full-depth 基线
CUDA_VISIBLE_DEVICES=$GPU python experiments/flame_lm_eval.py \
  --model_path $MODEL_PATH \
  --tasks lambada_openai,hellaswag,arc_easy,arc_challenge \
  --batch_size auto \
  --device cuda:0 \
  --output_path outputs/lm_eval_reskip_full
```

### 当前 340M 主线推荐配置

| 参数 | 值 |
|---|---|
| strategy | `recent_weight_gt` |
| probe_mode | `attn_only` |
| positions | `{3, 5}`（block 3 = ablation 最安全，block 5 = importance 最低） |
| quantile | `0.85` |
| max_skips | `2` |

对应已导出模型：[reskip_340M_combined_35_skip2_q085](/home/user01/Minko/reskip2/reskip/outputs/reskip_340M_combined_35_skip2_q085)

实测结果：
- 四任务 benchmark 与 full-depth **完全一致**
- Wall-clock **~1.19x** 加速
- 每 batch 平均跳 0.88 个 block

## 2026-04-14 340M 干净复比：benchmark 不掉且已有真实加速

模型与导出目录：
- full-depth：[reskip_transformer-340M](/home/user01/Minko/reskip2/reskip/flame/saves/reskip_transformer-340M)
- dynamic-safe：[reskip_340M_dynamic_tolerated](/home/user01/Minko/reskip2/reskip/outputs/reskip_340M_dynamic_tolerated)

这轮复比的目的：
- 不再讨论 proxy latency
- 直接在同一张 GPU7、同一批缓存样本、同一 `seq_len=8192` 条件下，对 full-depth 与 dynamic-safe 做干净 wall-clock 对比
- 再用同一张 GPU7 跑完整四任务 `lm-eval`

### 运行前提

- GPU：物理 `7` 卡
- 环境：`/home/user01/Minko/reskip2/.venv`
- 动态配置来源：
  - 分析文件：[routing_analysis.json](/home/user01/Minko/reskip2/reskip/outputs/reskip_analysis_340M_targeted/routing_analysis.json)
  - 导出模型目录：[reskip_340M_dynamic_tolerated](/home/user01/Minko/reskip2/reskip/outputs/reskip_340M_dynamic_tolerated)

当前 `dynamic-safe` 配置：
- `strategy = recent_weight_gt`
- `probe_mode = attn_only`
- `position_mode = low1`
- `quantile = 0.95`
- `max_skips = 1`

### 代码侧必要优化

针对 340M，这轮确认真正限制速度的不是 AttnRes 分 block 本身，而是动态 skip 的额外实现开销。为此在
[modeling_reskip_transformer.py](/home/user01/Minko/reskip2/reskip/flash-linear-attention/fla/models/reskip_transformer/modeling_reskip_transformer.py)
中只保留了必要的推理热路径优化：

- block dynamic skip 只在 `cached_prev` 或显式收集统计时计算 `block_phase1_summary`
- MLP dynamic skip 只在当前位置确实可能触发时才计算 `mlp_phase1_stats`
- 普通推理不再额外构造 `mlp_execution_trace`
- `router_entropies` 只在训练或显式统计时累计

### 同批次真实测速

测速设置：
- `seq_len = 8192`
- `num_batches = 48`
- `warmup = 4`
- 不开启 `return_routing_info`
- full 和 dynamic 使用完全相同的缓存 batch

结果：

| 配置 | mean_batch_s | tok/s | mean_loss |
|---|---:|---:|---:|
| full-depth | 0.04865 | 168395.9 | 9.2022 |
| dynamic-safe | 0.04275 | 191621.8 | 8.6621 |

结论：
- 当前 `340M dynamic-safe` 相对 full-depth 的真实前向速度提升约为 `1.14x`
- 这已经不是 proxy，而是同批次 wall-clock 测速结果

### 动态触发情况

在前 16 个 batch 上额外打开 `return_routing_info` 检查：
- 每个 batch 都稳定跳过 `1` 个 block
- `skipped_blocks_samples = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]`

这说明：
- 340M 不是“根本跳不动”
- 当前 safe 配置已经能稳定触发动态 skip

### 四任务 lm-eval

full-depth 命令：

```bash
CUDA_VISIBLE_DEVICES=7 /home/user01/Minko/reskip2/.venv/bin/python \
  /home/user01/Minko/reskip2/reskip/experiments/flame_lm_eval.py \
  --model_path /home/user01/Minko/reskip2/reskip/flame/saves/reskip_transformer-340M \
  --tasks lambada_openai,hellaswag,arc_easy,arc_challenge \
  --batch_size auto \
  --device cuda:0 \
  --output_path /home/user01/Minko/reskip2/reskip/outputs/lm_eval_reskip_340M_full_gpu7
```

dynamic-safe 命令：

```bash
CUDA_VISIBLE_DEVICES=7 /home/user01/Minko/reskip2/.venv/bin/python \
  /home/user01/Minko/reskip2/reskip/experiments/flame_lm_eval.py \
  --model_path /home/user01/Minko/reskip2/reskip/outputs/reskip_340M_dynamic_tolerated \
  --tasks lambada_openai,hellaswag,arc_easy,arc_challenge \
  --batch_size auto \
  --device cuda:0 \
  --output_path /home/user01/Minko/reskip2/reskip/outputs/lm_eval_reskip_340M_dynamic_safe_gpu7
```

结果文件：
- full-depth：[results_2026-04-14T12-37-58.313056.json](/home/user01/Minko/reskip2/reskip/outputs/lm_eval_reskip_340M_full_gpu7/__home__user01__Minko__reskip2__reskip__flame__saves__reskip_transformer-340M/results_2026-04-14T12-37-58.313056.json)
- dynamic-safe：[results_2026-04-14T12-40-01.423102.json](/home/user01/Minko/reskip2/reskip/outputs/lm_eval_reskip_340M_dynamic_safe_gpu7/__home__user01__Minko__reskip2__reskip__outputs__reskip_340M_dynamic_tolerated/results_2026-04-14T12-40-01.423102.json)

四任务对比：

| Task | Metric | Full-depth | Dynamic-safe |base Transformer|
|---|---:|---:|---:|
| lambada_openai | acc | 0.4056 | 0.4056 |0.3790|
| lambada_openai | ppl | 20.2022 | 20.2022 |24.7077|
| hellaswag | acc_norm | 0.4607 | 0.4607 |0.4436|
| arc_easy | acc_norm | 0.5438 | 0.5438 |0.5502|
| arc_challenge | acc_norm | 0.3012 | 0.3012 |0.2912|

### 最新结论

- 对当前 `340M / 8 blocks / 24 layers` 的模型，`dynamic-safe` 已经实现：
  - benchmark 与 full-depth 一致
  - 同批次真实前向测速约 `1.14x`
- 因此当前主线判断更新为：
  - `340M` 上的动态 ReSkip 并不是”只有精度没有速度”
  - 问题的关键点在于推理热路径实现是否足够收敛
  - 在收紧这部分额外开销后，当前 safe 动态 skip 已具备继续写入主实验线的价值

## 2026-04-16 基于 Ablation 的多 Block Skip：突破 max_skips=1 的瓶颈

### 动机

之前所有实验均使用 `max_skips=1`，最多只跳 1 个 block，速度提升上限被粒度硬性限制。此次实验的核心思路是：

1. **用 static block ablation 的 PPL impact 来选择 skip 候选位置**，而非仅依赖 AttnRes importance score
2. **放开 `max_skips=2`**，允许同时跳过多个 block
3. 使用 `recent_weight_gt` + `attn_only` probe + 按位置独立校准阈值

### Block Ablation 回顾

从 `reskip_340M_probe_sweep_fast_gpu7` 分析得到的单 block 静态移除 PPL：

| Block | AttnRes Importance | Static Removal PPL | PPL Ratio |
|---|---:|---:|---:|
| 0 | 0.286 | 36.91 | 3.60x |
| 1 | 0.427 | 83.23 | 8.11x |
| 2 | 0.517 | 13.99 | **1.36x** |
| **3** | **0.561** | **13.43** | **1.31x (最低)** |
| 4 | 0.460 | 15.40 | 1.50x |
| 5 | 0.400 | 19.54 | 1.90x |

关键洞察：Block 3 的 AttnRes importance 最高（0.561），但 static removal PPL impact 最低（1.31x）。这说明 **importance score（”被引用频率”）和实际可替代性是两回事**。Block 3 虽然被下游 block 频繁引用，但它的信息可以被其他 block 的组合补偿。

### 候选位置集设计

| 名称 | 位置 | 选择依据 |
|---|---|---|
| `low1` | {5} | 之前最佳，AttnRes importance 第二低 |
| `ablation1` | {3} | ablation impact 最低 |
| `ablation2` | {2, 3} | ablation impact 最低的两个 |

### 实验 A：多 Block Skip（max_skips=1,2）

proxy 评测设置：
- 模型：`flame/saves/reskip_transformer-340M`
- 数据：`fineweb_edu_100BT` train split, streaming, seed=1
- `seq_len=8192`, `batch_size=1`, `eval_batches=16`, `cal_batches=32`
- `strategy=recent_weight_gt`, `probe_mode=attn_only`
- 校准阈值按位置独立从 calibration set 中取 quantile

proxy PPL ratio 结果（相对 full-depth）：

| 配置 | max_skips | quantile | PPL ratio | Avg Blocks |
|---|---:|---:|---:|---:|
| `low1` {5} | 1 | 0.85 | 0.973x | 7.81 |
| **`ablation1` {3}** | **1** | **0.85** | **0.924x** | **7.75** |
| `ablation1` {3} | 1 | 0.95 | 0.959x | 7.88 |
| `ablation2` {2,3} | 1 | 0.85 | 0.996x | 7.62 |
| **`ablation2` {2,3}** | **2** | **0.85** | **0.996x** | **7.62** |
| **`ablation2` {2,3}** | **2** | **0.93** | **1.010x** | **7.69** |

注意：`ablation1 {3} q=0.85` 的 proxy PPL 反而比 full-depth **低 7.6%**。这意味着 block 3 在部分输入上不仅冗余，而且可能引入干扰。

### 实验 B/C 总结（负面结果，记录但不推荐）

**实验 B（Hybrid block+MLP skip）**：MLP-level skip 使用 `recent_weight_gt` 信号效果极差，总是增加 overhead 而非减少延迟。`recent_weight_gt` 衡量的是 block 间引用关系，不适合判断单层 MLP 是否可跳。结论：**放弃，MLP skip 需要专用的 routing signal。**

**实验 C（Sample-level quantile 聚合）**：在 `batch_size=1` 下与 mean 聚合完全一致（预期中），需要 `batch_size > 1` 才能体现差异。结论：**中性，待大 batch 场景验证。**

### 四任务 lm-eval Benchmark

导出模型：
- `ablation1 {3} skip=1 q=0.85`：[reskip_340M_ablation1_skip1_q085](/home/user01/Minko/reskip2/reskip/outputs/reskip_340M_ablation1_skip1_q085)
- `ablation2 {2,3} skip=2 q=0.85`：[reskip_340M_ablation2_skip2_q085](/home/user01/Minko/reskip2/reskip/outputs/reskip_340M_ablation2_skip2_q085)

跑分命令：

```bash
# ablation1
CUDA_VISIBLE_DEVICES=6 python experiments/flame_lm_eval.py \
  --model_path outputs/reskip_340M_ablation1_skip1_q085 \
  --tasks lambada_openai,hellaswag,arc_easy,arc_challenge \
  --batch_size auto --device cuda:0 \
  --output_path outputs/lm_eval_reskip_340M_ablation1_skip1_q085

# ablation2
CUDA_VISIBLE_DEVICES=7 python experiments/flame_lm_eval.py \
  --model_path outputs/reskip_340M_ablation2_skip2_q085 \
  --tasks lambada_openai,hellaswag,arc_easy,arc_challenge \
  --batch_size auto --device cuda:0 \
  --output_path outputs/lm_eval_reskip_340M_ablation2_skip2_q085
```

结果文件：
- `ablation1`：[results_2026-04-16T08-39-27.197962.json](/home/user01/Minko/reskip2/reskip/outputs/lm_eval_reskip_340M_ablation1_skip1_q085/outputs__reskip_340M_ablation1_skip1_q085/results_2026-04-16T08-39-27.197962.json)
- `ablation2`：[results_2026-04-16T08-40-03.755112.json](/home/user01/Minko/reskip2/reskip/outputs/lm_eval_reskip_340M_ablation2_skip2_q085/outputs__reskip_340M_ablation2_skip2_q085/results_2026-04-16T08-40-03.755112.json)

四任务对比：

| Task | Metric | Full-depth | prev best (low1/skip1) | ablation1 {3}/skip1 | ablation2 {2,3}/skip2 |
|---|---:|---:|---:|---:|---:|
| lambada_openai | acc | 0.4056 | 0.4056 | **0.4056** | 0.4048 |
| lambada_openai | ppl | 20.2022 | 20.2022 | **20.2022** | 20.6781 |
| hellaswag | acc_norm | 0.4607 | 0.4607 | **0.4607** | 0.4603 |
| arc_easy | acc_norm | 0.5438 | 0.5438 | **0.5438** | **0.5438** |
| arc_challenge | acc_norm | 0.3012 | 0.3012 | **0.3012** | **0.3012** |

### 新最佳配置：combined {3,5}/skip2/q085

通过把 ablation 发现的 block 3 和传统的 block 5 组合，同时放开 max_skips=2：

- **Block 3**：ablation impact 最低，skip 可提升质量
- **Block 5**：AttnRes importance 最低，skip 触发频率最高
- **max_skips=2**：两个位置在同一 forward 中同时可跳

导出模型：[reskip_340M_combined_35_skip2_q085](/home/user01/Minko/reskip2/reskip/outputs/reskip_340M_combined_35_skip2_q085)

### 四任务 lm-eval

结果文件：[results_2026-04-16T09-05-32.101070.json](/home/user01/Minko/reskip2/reskip/outputs/lm_eval_reskip_340M_combined_35_skip2_q085/outputs__reskip_340M_combined_35_skip2_q085/results_2026-04-16T09-05-32.101070.json)

| Task | Metric | Full-depth | prev best (low1/skip1) | **combined {3,5}/skip2** |
|---|---:|---:|---:|---:|
| lambada_openai | acc | 0.4056 | 0.4056 | **0.4056** |
| lambada_openai | ppl | 20.2022 | 20.2022 | **20.2022** |
| hellaswag | acc_norm | 0.4607 | 0.4607 | **0.4607** |
| arc_easy | acc_norm | 0.5438 | 0.5438 | **0.5438** |
| arc_challenge | acc_norm | 0.3012 | 0.3012 | **0.3012** |

四任务与 full-depth **完全一致**。

### 同批次 wall-clock 测速

设置：`seq_len=8192`, `num_batches=48`, `warmup=4`, GPU6 串行跑所有配置。

| 配置 | s/batch | tok/s | speedup | avg_skips/batch |
|---|---:|---:|---:|---:|
| full-depth | 0.05451 | 150273 | 1.000x | 0.00 |
| prev_best (low1/skip1/q095) | — | — | ~1.14x (2026-04-14) | ~1.00 |
| **combined {3,5}/skip2/q085** | **0.04579** | **178900** | **1.190x** | **0.88** |
| combined {2,3,5}/skip2/q080 | 0.04480 | 182850 | 1.217x | 1.12 |

注意：latency 因 GPU 调度波动会有 ±5% 偏差。以上为同一会话内连续测量值，相对排序可靠。

### 核心发现

1. **Block 3 是隐藏的最佳 skip 目标**：AttnRes importance 最高（0.561）但 static removal impact 最低（1.31x PPL ratio）。"被引用频率高"≠"不可替代"——block 3 的信息可被其他 block 补偿。

2. **组合 {3,5} + max_skips=2 同时提速和保质**：
   - Block 5 提供高 skip 触发率（低 importance → routing 信号频繁触发）
   - Block 3 提供安全 skip 空间（低 ablation impact → 跳过不损质量）
   - 两者互补，达到 0.88 skips/batch，speedup ~1.19x，benchmark 零退化

3. **Ablation-informed 选位 vs AttnRes importance 选位**：两种信号互补而非替代。importance 低的 block（如 5）skip 触发频率高；ablation impact 低的 block（如 3）skip 安全性高。组合使用效果最佳。

### 淘汰方向（代码已清理）

- MLP-level skip：`recent_weight_gt` 信号不适用于 MLP 级别决策
- Hybrid block+MLP skip：增加 overhead 而非减少
- Sample-level quantile 聚合：batch_size=1 下无效

### 代码清理

- `configuration_reskip_transformer.py`：移除 `dynamic_skip_granularity`、`mlp_position_thresholds`、`max_mlp_skips`、`sample_quantile`；legacy key 自动忽略
- `modeling_reskip_transformer.py`：移除 MLP skip 路径、hybrid 模式、sample_quantile 管道；BlockGroup.forward() 返回值从 10-tuple 简化为 6-tuple
- `flame_analyze_reskip.py`：新增 `ablation1`/`ablation2`/`ablation3` 位置模式（基于 block ablation PPL 排序），默认 `max_skips_options=1,2,3`

### 下一步

- 考虑更细粒度 block 划分（12 blocks × 2 layers），进一步释放 skip 空间
- 将 ablation-informed 选位写入 paper 的 method section
- 在 1.3B 模型上验证组合选位策略的可迁移性
