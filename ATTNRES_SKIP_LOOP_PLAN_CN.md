# 基于 AttnRes 的 ReSkip / ReLoop 后续改进方案

## 1. 文档目的

本文档只覆盖两个方向：

- 方向二：在不脱离 AttnRes 本身特性的前提下，重新设计一个更可靠的 skip 算法
- 方向三：基于 AttnRes 设计一个可训练的 unified skip+loop 架构

方向一“增加 block 数、降低单次 skip 粒度”已经很直接，这里不展开。

本文档基于以下事实与材料撰写：

- 当前代码实现
  - `flash-linear-attention/fla/models/reskip_transformer/modeling_reskip_transformer.py`
  - `flash-linear-attention/fla/models/reskip_transformer/configuration_reskip_transformer.py`
  - `experiments/flame_analyze_reskip.py`
  - `experiments/flame_reskip_common.py`
  - `flame/flame/train.py`
- 当前总计划：`PLAN.md`
- 论文草稿：`paper/main.tex`
- 最近真实实验现象：
  - full-depth LM 在持续变好
  - routing entropy 在下降
  - 但 skip quality 没有稳定同步变好
  - 到较后期 checkpoint 时，一旦开始跳层，PPL 仍可能恶化过快

---

## 2. 当前问题的准确定义

### 2.1 现在不是“没有 routing”

当前 `ReSkip` 并不是完全没有学出 AttnRes 路由。相反，已有证据表明：

- `routing_entropy` 会随训练下降
- `block_importance` 会形成一定层次
- `analyze` 中已经能看出个别 block 比其他 block 更像候选可裁块

问题在于：

- 现有 skip 决策过于粗、过于硬
- 现有 importance 定义过于保守
- 现有 skip policy 主要是“全局静态 keep mask”
- 现有训练目标没有直接鼓励“可平滑跳层”

因此当前现象更像：

- full-depth 模型在变好
- 路由也在变尖
- 但一旦真的裁 block，模型还不够鲁棒

### 2.2 当前实现中最核心的约束

当前 `ReSkip` 的核心机制在论文里的定义是：

- block importance: `I(n) = max_{l > n} alpha_{n -> l}`
- 推理时基于一个阈值 `epsilon` 决定是否跳过 block
- 对 input-dependent skip，草稿中提到“可用 current block self-importance 作为 proxy”

当前代码大体对应这一路径：

- 在 `modeling_reskip_transformer.py` 中聚合 routing event
- `_aggregate_routing()` 里把 `block_importance` 定义为所有 downstream target 上的最大值
- `build_keep_mask_from_importance()` 直接对 `block_importance` 做阈值裁剪
- `flame_analyze_reskip.py` 对 full-depth 模型先统计一次 `block_importance`，再做 threshold sweep

这套定义的优点是简单、论文叙事干净；缺点是太硬，且对真实大模型不够稳定。

---

## 3. 方向二：重新设计一个更可靠的 AttnRes Skip 算法

## 3.1 总体目标

方向二的目标不是重新发明一个外部 router，而是把 AttnRes 从“全局静态裁块评分器”升级成“更稳的 skip 决策器”。

这个方向必须满足四条约束：

1. 依然以 AttnRes 内部 routing signal 为核心
2. 不引入与主模型割裂的外部 early-exit classifier
3. 尽量把改动放在 analyze / calibration / inference policy
4. 论文主叙事仍然可以写成“routing signal 是 AttnRes 自带的”

## 3.2 当前 skip 方案的主要不足

### 不足 A：importance 只看 downstream max，过于保守

当前 `I(n) = max_{l>n} alpha_{n->l}` 有两个问题：

- 只要某个 block 对某个未来位置偶尔重要，就会长期被视为“不可裁”
- 对真实语言模型，max 很容易被少数 hard sequence 拉高

直接后果是：

- 全局 threshold 很难平滑地移动
- 一开始不跳，一旦开始跳就容易跳过猛

### 不足 B：全局静态 keep mask 太硬

当前策略本质是：

1. 先在 calibration set 上统计一个平均 importance
2. 生成一组全局 keep mask
3. 对所有样本一刀切

这更像 post-hoc pruning，而不是 input-dependent skipping。

### 不足 C：当前 analyze 对“边际伤害”建模太弱

当前 `block_importance` 只是 routing 强度，不等于“去掉该 block 后质量损失”。  
换句话说，现在的 score 更像“注意到它”，不是“依赖它”。

这也是为什么：

- entropy 在降
- importance 在分化
- 但真跳时 PPL 仍然恶化明显

## 3.3 新的算法目标：从阈值裁块改成分阶段校准 skip

建议把方向二定义为：

**AttnRes-Calibrated Progressive Skip, ACPS**

核心思想：

- AttnRes 仍提供原始 routing prior
- 但 skip 决策不再直接由单一 `max downstream importance` 决定
- 而是先做 richer calibration，再用更平滑的策略出最终 keep policy

## 3.4 建议的新 skip 方案

### 方案 2A：用复合 score 替代单一 max importance

对每个 block `n`，在 calibration set 上统计以下量：

- `I_max(n)`：当前实现中的 downstream max
- `I_p90(n)`：downstream importance 的 90 分位数
- `I_mean(n)`：downstream importance 的均值
- `S_self(n)`：self-importance
- `V(n)`：跨 batch 方差，表示该 block 的稳定性
- `DeltaPPL_single(n)`：单独 ablate 该 block 的相对 PPL 损失

建议最终使用一个更稳的 ranking score：

`score_skip(n) = w1 * I_p90 + w2 * I_mean + w3 * S_self + w4 * DeltaPPL_single - w5 * V`

第一阶段不需要把 `w1...w5` 做成训练参数，直接手工固定即可，目标只是让排序更稳。

推荐第一版先试这个简化式：

`score_skip(n) = 0.4 * I_p90 + 0.2 * I_mean + 0.2 * S_self + 0.2 * norm(DeltaPPL_single)`

这比只看 `I_max` 更接近“既看 routing，也看真实伤害”。

### 方案 2B：从 threshold sweep 改成 progressive removal

不再一次性对所有 block 做阈值裁剪，而改成逐块剔除：

1. full-depth 跑 calibration，拿到每个 block 的 `score_skip`
2. 先尝试删掉分数最低的 1 个 block
3. 重新评估 PPL
4. 若仍在容忍范围内，再删下一个
5. 直到越界为止

这个策略比 threshold 好解释，也更稳：

- 它直接优化“在给定容忍度下，最多能删几个 block”
- 能避免“阈值稍微提高一点就一下子删掉多个 block”的跳变

### 方案 2C：引入 sequence-level input-dependent skip

这是论文草稿里已经给出的方向，且最契合 AttnRes：

- 不直接做 token-level routing
- 先只做 sequence-level skip
- 对每个样本单独计算当前序列的 proxy score

建议第一版 proxy 用：

- `current block self-importance`
- `current block downstream p90`
- `当前 block routing entropy`

并在推理时采用简单规则：

- 若某 block 的局部 proxy 低于该 block 的 calibration threshold，则该样本跳过该 block
- 否则保留

这个版本仍然完全建立在 AttnRes 统计量上，不需要额外 router。

### 方案 2D：保留“论文主线”和“工程增强”两套口径

为了方便论文书写，建议把方向二拆成两个层次：

- 主方法：AttnRes + calibration + progressive skip
- 增强版：AttnRes + sequence-level input-dependent skip

这样主线足够干净，增强版也有明确增量。

## 3.5 基于当前代码的具体落地步骤

### 第一步：扩展 routing 统计接口

目标文件：

- `flash-linear-attention/fla/models/reskip_transformer/modeling_reskip_transformer.py`

需要新增的输出：

- per-block downstream distribution summary
  - `max`
  - `mean`
  - `p90`
  - `p95`
- per-block self-importance
- per-block routing variance 或 batch-level波动指标

建议做法：

- 现有 `_aggregate_routing()` 不要只返回 `block_importance`
- 增加结构化字段，如：
  - `block_importance_max`
  - `block_importance_mean`
  - `block_importance_p90`
  - `self_importance`
  - `block_importance_var`

### 第二步：把 analyze 改成“排序 + 逐块剔除”

目标文件：

- `experiments/flame_analyze_reskip.py`
- `experiments/flame_reskip_common.py`

新增流程：

1. full-depth 统计 richer routing stats
2. 单块 ablation，得到 `DeltaPPL_single`
3. 生成 block ranking
4. progressive removal
5. 输出新的 Pareto 结果

建议输出字段：

- `block_ranking`
- `single_block_ablation`
- `progressive_skip_curve`
- `best_progressive_keep_mask`
- `sequence_level_skip_curve`

### 第三步：支持 sequence-level skip policy

目标文件：

- `modeling_reskip_transformer.py`

当前只支持全局 `skip_keep_mask`。需要新增一个更细粒度的 runtime 接口：

- 输入一批样本时，为每个样本动态生成 block keep decision
- 第一版只做 sequence-level，不做 token-level

建议接口形态：

- `skip_policy="static" | "sequence_dynamic"`
- `skip_block_thresholds=[...]`

推理流程：

1. 先用当前序列的 routing proxy 估计每个 block 的重要度
2. 基于 calibration threshold 为该序列生成 keep mask
3. 对该序列执行实际 skip

### 第四步：建立清晰的评测 protocol

建议固定四组结果：

1. full-depth baseline
2. 当前旧版 threshold skip
3. progressive skip
4. sequence-level dynamic skip

统一报告：

- validation PPL
- avg blocks
- effective FLOPs
- per-bucket quality
  - easy / medium / hard
- per-domain quality

## 3.6 方向二的成功判据

满足以下任意一条，即可认为方向二有效：

- 在 `PPL +2%` 容忍内，`avg_blocks <= baseline - 1`
- progressive skip 明显优于当前 threshold skip
- input-dependent skip 在 hard/easy 样本上能表现出不同平均深度，且 overall PPL 优于静态 skip

失败判据：

- richer score 与 `I_max` 没有实质差别
- progressive skip 仍然一开始删就大幅掉点
- input-dependent skip 没有形成任何样本间深度差异

## 3.7 方向二的优先级判断

方向二应当作为当前主线的首选增强方向，因为它：

- 不需要重写训练架构
- 最符合论文当前叙事
- 可以直接复用现有 `ReSkip` 代码
- 能先回答“问题到底出在 score 还是出在训练目标”这个核心问题

---

## 4. 方向三：基于 AttnRes 的 unified skip + loop 架构

## 4.1 方向三的定位

方向三不应被视为“方向二的小补丁”，而应被视为：

**下一代 AttnRes 自适应深度架构**

它要解决的问题不是“怎样更稳地裁块”，而是：

- 哪些输入应该前进
- 哪些输入应该复用当前 block
- 哪些输入应该直接略过部分计算
- 哪些输入已经可以停止

因此方向三更接近论文后续工作或增强版模型，而不是当前主线的最小修补。

## 4.2 方向三的核心思想

当前代码里已经有一部分 unified depth 控制的基础设施：

- `enable_looping`
- `halt_head`
- `ponder_cost_tensor`
- `expected_depth_tensor`
- `exit_kl_tensor`

但现在的 loop 仍主要是：

- 按预先给定的 `block_schedule`
- 再通过 `halt_head` 决定是否提前停止

它没有真正统一：

- skip
- repeat
- advance
- halt

建议把方向三定义为：

**AttnRes Action-over-Depth, AAOD**

即：在每个逻辑深度位置，根据 AttnRes 提供的内部路由特征，选择一个 depth action。

## 4.3 建议的动作空间

第一版不要做 token-level action，而做 sequence-level action。动作空间定义为：

- `ADVANCE`
  - 执行当前 block 后，进入下一个 unique block
- `REPEAT`
  - 继续使用当前 unique block，再执行一次
- `BYPASS`
  - 直接跳过下一个逻辑位置，不执行 block
- `HALT`
  - 当前序列停止继续加深

这四个动作正好统一了：

- skip
- loop
- early halt

## 4.4 为什么这个方向必须依托 AttnRes，而不是外接一个 router

如果方向三额外接一个独立控制器，论文叙事会明显偏离“AttnRes 自带 routing signal”。

因此建议控制信号尽量由当前模型内部已有量构成：

- 当前 block self-importance
- 当前 block downstream importance
- 当前 routing entropy
- 当前 hidden state pooled summary
- 前后两次执行的 hidden delta
- 当前 halt probability

也就是说，AAOD 不是一个独立 MoD router，而是一个由 AttnRes statistics 驱动的动作层。

## 4.5 建议的控制器设计

### 方案 3A：最小版本，轻量动作头

在每个逻辑位置，对 pooled hidden 和 routing summary 做一个很小的动作头：

- 输入特征：
  - `pooled_hidden`
  - `self_importance`
  - `downstream_importance_p90`
  - `routing_entropy`
  - `hidden_delta_norm`
  - `halt_logit`
- 输出：
  - `p_advance`
  - `p_repeat`
  - `p_bypass`
  - `p_halt`

这个动作头可以是一个非常小的 MLP 或线性层。

这样做的理由：

- 可训练
- 容易接到当前 `enable_looping` 路径
- 保持控制器极小，避免方法重心从 AttnRes 转移到 router

### 方案 3B：更纯粹版本，不增加新头，只做规则控制

如果要更坚持“零额外参数”，可以先做规则版：

- `halt` 仍由 `halt_head` 决定
- `repeat` 主要由高 self-importance 和低 hidden-delta 触发
- `bypass` 主要由低 downstream importance 触发
- 否则 `advance`

这个版本更像研究原型，适合先验证 unified depth 思路是否成立。

建议实践顺序：

1. 先实现规则版
2. 若规则版有效，再实现轻量动作头版

## 4.6 基于当前代码的具体落地步骤

### 第一步：重构 block_schedule 为 runtime action loop

目标文件：

- `modeling_reskip_transformer.py`

当前执行方式更像：

- 预定义 `block_schedule`
- 逐位置循环

方向三需要改成：

- 维护 `logical_depth`
- 维护 `current_block_idx`
- 每步先执行当前 block
- 再根据动作决定下一步

需要新引入的 runtime state：

- `current_unique_block`
- `num_repeats_used`
- `logical_depth_position`
- `halted_mask`

### 第二步：把 skip 与 loop 统一进 routing_info

当前 `routing_info` 主要记录：

- `block_importance`
- `execution_trace`
- `effective_depth`
- `expected_depth`

方向三还需要记录：

- `action_trace`
- `num_advances`
- `num_repeats`
- `num_bypasses`
- `halt_step`
- `depth_path`

这样后续 analyze 才能真正看 unified adaptive depth。

### 第三步：引入新的训练目标

方向三是 trainable architecture，因此必须明确训练目标，而不能只靠后处理。

建议训练目标由三部分组成：

1. 标准 LM loss
2. 计算预算损失
   - 对 `expected_depth` 加约束
3. 动作分布正则
   - 防止模型退化成“永远 repeat”或“永远 bypass”

建议第一版保留当前已有的：

- `ponder_loss_weight`
- `halt_kl_weight`

并新增一个统一深度预算项，例如：

- `depth_budget_weight`

不要一开始引入太多损失，先保持最小化。

### 第四步：统一 skip / loop 的分析工具

目标文件：

- `flame_analyze_reskip.py`
- 未来可单独新增 `flame_analyze_reflow.py` 或 `flame_analyze_adaptive_depth.py`

需要新增的分析维度：

- 平均动作分布
- 不同难度样本的 action pattern
- 不同 domain 的 depth path
- `repeat` 和 `bypass` 的联合 Pareto

## 4.7 推荐的阶段性实验路线

### 阶段 A：规则版 unified depth

目标：

- 验证“AttnRes 统计量能否同时支持 repeat / bypass / halt”

要求：

- 不新增复杂控制器
- 不改主损失
- 仅在 inference 或轻量训练下先看 depth path 是否有意义

成功标志：

- easy/hard 样本的平均路径明显不同
- 相比单独 ReLoop 或 ReSkip，Pareto 更平滑

### 阶段 B：轻量动作头版 unified depth

目标：

- 把规则版中有效的启发式固化为可学习策略

要求：

- 动作头极小
- 仍以 AttnRes 统计量为输入
- 不引入大型独立 router

成功标志：

- 在同等 PPL 下，effective depth 低于纯 ReLoop
- 在同等 compute 下，PPL 低于纯 ReSkip

### 阶段 C：论文增强版

目标：

- 作为 follow-up 或扩展章节

输出应包含：

- unified adaptive depth 图
- action distribution 图
- 与 `ReSkip`、`ReLoop`、UT baseline 的对比

## 4.8 方向三的风险与边界

方向三最大的风险不是工程复杂，而是方法叙事会明显变重。

风险点：

- 很容易从“AttnRes 自带 routing”滑向“小控制器才是真正起作用”
- 训练目标如果过多，方法会变得像 ACT / MoD 混合体
- debug 成本高于方向二

因此建议把方向三明确定位为：

- 新架构
- 增强版
- 后续工作

而不是当前主 paper 的唯一主线。

---

## 5. 两个方向的建议优先级

当前建议顺序：

1. 先做方向二
2. 方向二若有效，再决定方向三是否立项

原因：

- 方向二能先回答“当前问题到底出在 skip score 还是训练目标”
- 方向三一旦开始做，问题空间会迅速扩大
- 若方向二已经能把 skip 做到足够好，方向三可转为 follow-up

---

## 6. 我建议的最小可执行版本

### 方向二最小可执行版本

1. 扩展 routing statistics
2. 做 single-block ablation
3. 做 progressive removal
4. 做 sequence-level dynamic skip

这是最应该立即开干的一条线。

### 方向三最小可执行版本

1. 先做规则版 unified depth
2. 观察 action path 是否合理
3. 只有规则版有效，再做轻量动作头版

---

## 7. 一句话结论

- 方向二的本质是：把当前“硬阈值静态裁块”升级成“AttnRes 驱动的渐进式、可校准、最好还是 input-dependent 的 skip policy”
- 方向三的本质是：把当前分离的 `skip` 和 `loop` 统一成一个 AttnRes 驱动的 depth action framework

如果目标是尽快把当前工作做成论文，优先做方向二。  
如果目标是做下一代方法，方向三才值得正式立项。
