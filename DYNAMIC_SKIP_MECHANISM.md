# ReSkip 动态 Skip 方案说明

> **本文档说明**：本文档解释了为什么 ReSkip 从静态 keep-mask 转向动态 runtime skip，动态 skip 如何利用 AttnRes 的两阶段结构，以及对论文写法的具体修改建议。对应实验数据见 [DYNAMIC_SKIP_EXPERIMENT_LOG.md](DYNAMIC_SKIP_EXPERIMENT_LOG.md)，训练/评测命令见 [EXPERIMENTS_CN.md](EXPERIMENTS_CN.md)。

## 1. 为什么要从静态 skip 改成动态 skip

我们最近在语言模型上的实验表明，一个很关键的现象是：

- 如果采用传统的静态 block 删除方式，即为所有样本固定删除同一个中间 block，
- 那么即使删除的是“看起来最弱”的 block，PPL 往往仍然恶化明显。

这说明当前 `ReSkip` 的主要问题并不只是“block 选得不准”，而是：

- 模型对“全局永久删除某个 block”并不鲁棒；
- 但是这不等于模型不存在条件性的深度冗余；
- 更合理的假设是：某些 block 只在某些输入、某些上下文、某些深度阶段是冗余的。

因此，paper 里的 `ReSkip` 需要从：

- “基于 calibration 的静态 keep-mask”

扩展成：

- “基于 AttnRes 运行时信号的动态、样本相关的 skip”

这一步仍然不改变论文主线，因为它依然完全依赖 AttnRes 自身的 routing signal，而不是引入外部 router 或额外训练目标。

---

## 2. 动态 skip 依赖 AttnRes 的哪一个核心特性

动态 skip 的关键，不是简单地“再看一次 block importance”，而是利用了 AttnRes 在当前实现里的一个非常重要的结构性特征：

### 2.1 ReSkip 的 block 执行本来就是两阶段的

当前 `ReSkip` block 的实现不是一次性完成，而是天然分成两步：

1. `phase1`: 对所有已经完成的 block state 做 routing
2. `phase2`: 再把 `phase1` 的结果和当前 block 内部逐层形成的 `partial_block` 在线合并

在代码里，对应的是：

- `batch_attend_completed_blocks(...)`
- `merge_with_partial_block(...)`

也就是说，在真正执行当前 block 之前，模型其实已经先算出了一次“当前 block 会如何从历史 completed blocks 中取信息”的 routing 结果。

这正是动态 skip 可以利用的时机。

### 2.2 动态 skip 使用的是 “进入当前 block 之前就已有的 routing 信息”

因此，动态 skip 并不需要额外跑一遍完整前向，也不需要单独的辅助网络。

它只需要在进入 block 之前，读取 `phase1` 已经天然产生的统计量，例如：

- 当前 block 对最近一个 completed block 的权重
- 当前 block 对 embedding/source-0 的权重
- 当前 block 当前时刻的 routing entropy

这些量本来就是 AttnRes 机制的内部产物。

换句话说，动态 skip 不是额外创造一个 routing signal，而是把 AttnRes 原本就有的 routing signal，真正用于运行时计算分配。

这和 paper 的核心叙事是一致的：

- routing signal 是 free 的
- routing signal 来自 AttnRes 本身
- 不需要额外 classifier 或 auxiliary router

---

## 3. 当前动态 skip 的具体实现

## 3.1 运行时接口

目前代码里新增了四个运行时参数：

- `dynamic_skip_strategy`
- `dynamic_skip_threshold`
- `dynamic_skip_position_thresholds`
- `dynamic_skip_max_skips`

它们已经接到：

- `flash-linear-attention/fla/models/reskip_transformer/modeling_reskip_transformer.py`
- `flash-linear-attention/fla/models/reskip_transformer/configuration_reskip_transformer.py`

并且可以写入导出后的 `config.json`，因此动态 skip 可以作为模型默认推理行为直接开启。

## 3.2 block 执行前的 runtime 统计

在每个 block position 开始前，当前实现会先调用：

- `ReSkipBlockGroup.prepare_phase1(...)`

它会基于已经完成的 blocks，批量得到各层 router 的 `phase1` 结果。

随后调用：

- `ReSkipBlockGroup.summarize_phase1(...)`

把 `phase1` 的细粒度权重压缩成几个轻量统计量：

- `avg_phase1_entropy`
- `avg_phase1_embed_weight`
- `avg_phase1_recent_weight`
- `num_completed_sources`

这些统计量被记录到 execution trace 里，也可直接用于运行时决策。

## 3.3 当前默认策略：`recent_weight_gt`

目前实验里效果最好、同时也最容易解释的策略是：

- `dynamic_skip_strategy = recent_weight_gt`

其含义是：

- 如果当前 block 在进入时，对“最近一个已完成 block”的权重过高，
- 说明当前 block 这一步更像是在重复依赖最近的局部状态，
- 而不是整合更广泛、更必要的深度信息，
- 则当前样本可以跳过该 block。

这个策略背后的直觉是：

- 当 AttnRes 强烈偏向最近一个已完成 block 时，
- 当前 block 更可能只是在做局部 refinement，
- 因而更可能存在条件性冗余。

这和静态 importance 完全不同。

静态 importance 问的是：

- “这个 block 全局看重不重要？”

动态 `recent_weight_gt` 问的是：

- “这次这个样本在这个时刻，当前 block 是否主要只是在重复利用上一层附近的信息？”

后者更适合运行时 skip。

## 3.4 为什么额外开销比较小

动态 skip 不是零开销，但它的额外开销很小，原因是：

1. `phase1` 本来就是当前 block 执行所必需的计算
2. 动态策略只是在 `phase1` 之后额外读几个标量统计
3. 若 block 最终被执行，已有 `phase1` 结果会直接复用，不会重算

因此额外成本主要是：

- 少量张量归约
- 一次简单阈值判断

而不是再跑一整层或再接一个外部控制器。

---

## 4. 为什么要用 calibration，而不是直接固定阈值

虽然动态 skip 是运行时决策，但当前最佳实践不是给一个全局手工阈值，而是：

- 先在 calibration set 上收集每个 block position 的 runtime 统计分布
- 再按分位数得到 `dynamic_skip_position_thresholds`

原因有两个：

### 4.1 不同 block position 的统计尺度不同

第 2 个 block 和第 6 个 block 的 `recent_weight` 分布不一定处于同一量纲。

如果对所有位置使用同一个阈值，很容易出现：

- 前面几个 block 永远不跳
- 后面几个 block 一跳就跳过猛

因此更稳的方案是：

- 每个 block position 单独校准阈值

### 4.2 需要一个“安全阀”来限制 skip 激进程度

现在代码中还加入了：

- `dynamic_skip_max_skips`

它的作用是限制每个样本最多跳过多少个 block。

这很重要，因为实验表明：

- `max_skips=1` 明显比 `max_skips=2` 更稳
- 当前语言模型阶段更适合“轻度 dynamic skip”，而不是激进裁剪

因此当前推荐配置是：

- position-specific thresholds
- `max_skips=1`

---

## 5. 当前实验结果说明了什么

在两个模型上都已经验证过：

### 5.1 旧的 `340M-2` checkpoint

对这个已经训练完成的模型，动态 skip 能做到：

- 比静态删一个固定 block 更平滑
- 在接近 full-depth 质量的同时，略微减少平均 block 数
- 并且推理耗时接近静态 skip

这说明动态 skip 不是训练分支偶然产物，而是对现有 AttnRes 模型也有效。

### 5.2 最新训练链路的 `step-2000` checkpoint

在新的训练链路上，结论同样成立：

- full-depth 仍然最好
- 但动态 skip 明显优于“固定删任意一个中间 block”
- 并且可以作为实际部署/评测的默认推理策略导出

这说明动态 skip 不是旧 checkpoint 的特例，而是当前 `ReSkip` 路线下可复现的趋势。

---

## 6. 现在的主结论

目前可以把 `ReSkip` 的结论更新为：

### 旧说法

- ReSkip 主要通过 calibration 后的静态 keep-mask 来实现 skip

### 新说法

- ReSkip 现在有两种 skip 模式：
  - static calibrated skip
  - dynamic runtime skip
- 其中 dynamic runtime skip 更符合当前语言模型实验现象，也更适合作为主评测方案

原因是：

- 静态 skip 等价于“对所有输入永久删某个 block”，过于强硬
- dynamic skip 则是“对某些输入、某些时刻条件性跳过某个 block”，更符合 AttnRes 的输入相关特性

---

## 7. 对 paper 表述的建议修改

在论文里，建议把 `ReSkip` 的实践路径写成两层：

### 7.1 保留原始静态版本作为基础定义

也就是继续保留：

- `I(n) = max_{l>n} alpha_{n->l}`
- calibration-based static skipping

因为这是最简洁的定义，也最容易和原始方法联系起来。

### 7.2 新增一个 runtime dynamic skip 小节

建议新增一个 subsection，说明：

- 静态 calibration 只是第一版 practical implementation
- 对真实大模型而言，更有效的是基于 AttnRes 运行时局部统计的 dynamic skip

可以表述为：

> Instead of applying a single global keep-mask to all inputs, we additionally consider a runtime dynamic skip rule based on the AttnRes phase-1 routing statistics already computed before each block. In particular, when the current block places excessive weight on the most recent completed block, the block is treated as a likely local refinement step and can be skipped for that input. This preserves the “free routing signal” property of AttnRes, because no auxiliary router or training loss is introduced.

### 7.3 在实验部分明确对比三种模式

建议 `ReSkip` 主实验至少同时报：

1. full-depth
2. static skip
3. dynamic skip

并把 dynamic skip 作为主方案，static skip 作为对照。

---

## 8. 这个动态 skip 对论文最大的价值

这个改动最重要的意义，不只是“指标更好”，而是它让论文叙事更完整：

1. AttnRes 不再只是离线 importance estimator
2. AttnRes 真正成为了运行时 computation router
3. ReSkip 从 post-hoc pruning 风格，转向 input-dependent adaptive computation 风格

这恰好更贴近论文一开始想讲的核心观点：

- AttnRes 提供了 natural, zero-cost routing signal for adaptive computation

如果只停留在静态 keep-mask，审稿人很容易把它理解成：

- “只是另一种 importance-based pruning”

而加入动态 runtime skip 之后，`ReSkip` 更像真正的 adaptive computation method。

---

## 9. 当前实现的边界

当前动态 skip 还不是最终形态，它有几个明确边界：

- 目前最稳定的是 `max_skips=1`
- 当前主要使用 sequence-level / sample-level 决策
- 还没有做到 token-level dynamic depth
- 还没有和 ReLoop 统一成一个完整的 action-over-depth 框架

因此，paper 里更适合把它写成：

- 当前 `ReSkip` 的强化版 practical implementation
- 或者主实验中更有效的 deployment strategy

而不是声称所有问题都已经彻底解决。

---

## 10. 一句话版本

当前动态 skip 的核心实现可以概括为：

> 在每个 block 真正执行之前，利用 AttnRes 已经自然产生的 phase-1 routing 统计，判断该 block 对当前输入是否更像局部重复 refinement；若是，则在运行时直接跳过该 block。这个策略不需要额外训练，不需要外部 router，并且在现有语言模型实验中明显优于静态删除固定 block。
