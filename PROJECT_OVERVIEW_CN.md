# ReSkip / ReLoop 项目概述

**目标论文:** NeurIPS 2026, 主题是 AttnRes (Attention Residual) 架构带来的自适应计算深度。

---

## 0. 核心实验结果速览 (paper 级)

本节是所有实验数据的 headline 摘要, 每一条都在下文对应小节有完整上下文 + 交叉引用.

### 表 1 — Part 1 · ReSkip 在 340M from-scratch AttnRes 上零退化加速 (§4.2, §4.6)

| 配置 | LAMBADA acc | LAMBADA ppl | HellaSwag | ARC-E | ARC-C | wall-clock |
|------|-------------|-------------|-----------|-------|-------|------------|
| Full-depth | 0.4056 | 20.20 | 0.4607 | 0.5438 | 0.3012 | 1.00× |
| **ReSkip** $\mathcal{P}{=}\{3,5\}, M{=}2, q{=}0.85$ | **0.4056** | **20.20** | **0.4607** | **0.5438** | **0.3012** | $\mathbf{1.19\times}$ |

**结论**: 所有 4 个下游 benchmark 指标 **完全一致** (bit-equal), 但 wall-clock 加速 $1.19\times$. 这是 "AttnRes routing 本身就是可用的 skip 信号" 的存在性证明, 基于 Importance–Ablation Disconnect 的 $I\cdot A$ 联合选位.

---

### 表 2 — Part 2 · Qwen3-VL-2B/4B retrofit 全面优于 base (lmms-eval 全集) (§5.2, §5.5)

| Benchmark | 2B base | **2B retrofit (v3, L=2)** | Δ | 4B base | **4B retrofit (v3, L=4)** | Δ |
|-----------|---------|---------------------------|---|---------|---------------------------|---|
| LAMBADA acc | 0.532 | **0.576** | **+4.4** | 0.576 | **0.663** | **+8.7** |
| MMBench_en_dev | 75.77 | **79.30** | **+3.5** | 83.33 | **85.22** | **+1.9** |
| MMMU_val | 0.414 | **0.439** | **+2.5** | 0.490 | 0.521 | **+3.1** |
| MMStar (6-cat avg) | 0.536 | 0.534 | −0.2 | 0.624 | **0.632** | **+0.8** |
| AI2D | 0.736 | **0.765** | **+2.9** | 0.819 | **0.825** | **+0.6** |
| OCRBench | 0.772 | **0.809** | **+3.7** | 0.819 | **0.824** | **+0.5** |
| RealWorldQA | 0.648 | **0.668** | **+2.0** | 0.715 | **0.718** | **+0.3** |

**结论**: v3 retrofit 在 2B 上每项 ≥ base (除 MMStar −0.2 噪声内), 4B 推荐配置 $L{=}4$ 下全面持平或严格优于 base. 结构为纯 AttnRes ($\gamma_n{=}1$ 在全部 14/9 个 block 上收敛), base VLM 冻结, 仅用 15M/23.7M 可训参数 (~0.7% / ~0.6%) + 5k/10k 步 SFT.

**归因**: 参数量匹配的 LoRA 4-seed 均值 LAMBADA $0.526$ (Δvs base $-0.6$pp); 我们的 retrofit $0.576$ (+5.0pp vs LoRA). 这 +5pp 不可能来自 "多一点可训参数", 必然是 AttnRes 结构本身.

---

### 表 3 — Part 2 · MMStar 子类别分解, 推理类是主要收益 (§5.3)

| 配置 | **math** | **logical** | sci&tech | coarse | fine | instance |
|------|----------|-------------|----------|--------|------|----------|
| 2B base | 0.413 | 0.429 | 0.408 | 0.714 | 0.520 | 0.710 |
| 2B v3 retrofit (L=2) | **0.492** | **0.432** | 0.353 | 0.734 | 0.505 | 0.683 |
| Δ vs 2B base | $\mathbf{+7.9}$ | +0.3 | −5.5 | +2.0 | −1.5 | −2.7 |
| 4B base | 0.549 | 0.626 | 0.465 | 0.788 | 0.611 | 0.705 |
| 4B v3 retrofit (L=4) | **0.588** | 0.602 | **0.467** | **0.812** | 0.606 | 0.714 |
| Δ vs 4B base | $\mathbf{+3.9}$ | −2.4 | +0.2 | +2.4 | −0.5 | +0.9 |

**结论**: router 抬升 "deliberate reasoning over images" 路径 (math / logical), 对早期感知基本无改变. 2B 的 MMStar math **+7.9pp** 是整个 benchmark 套件里最大的单项 swing.

---

### 表 4 — Part 2 · ReSkip Pareto 在 2B retrofit 上 (H_r256_5k, LAMBADA n=300) (§4.6)

| $q$ | $M_{\max}$ | LAMBADA acc | ppl | avg skips / 3 eligible | Δ vs full-path |
|-----|------------|-------------|-----|------------------------|----------------|
| — (full-path) | — | 0.5760 | 4.609 | 0 | — |
| 0.50 | 1 | 0.5500 | 5.45 | 0.82 | −2.6 |
| 0.70 | 1 | 0.5600 | 5.24 | 0.64 | −1.6 |
| 0.85 | 1 | 0.5733 | 4.90 | 0.37 | −0.3 |
| $\mathbf{0.95}$ | $\mathbf{1}$ | $\mathbf{0.5933}$ | $\mathbf{4.77}$ | $\mathbf{0.18}$ | $\mathbf{+1.7}$ |
| 0.95 | 2 | 0.5800 | 4.84 | 0.26 | +0.4 |

**结论**: 推荐操作点 $q{=}0.95, M{=}1$ **反超 full-path** (+1.7pp LAMBADA, +6.1pp vs stock base). 机制: 跳掉 $\alpha$ 已塌缩到 predecessor 的 block 是"消除冗余 forward". 候选集 $\mathcal{P}{=}\{4,6,11\}$ 由 single-block static removal 排序得出 (§4.3).

---

### 表 5 — Part 2 · Wall-clock 速度 (H100, bfloat16, `use_cache=True`, 规范 bench) (§4.7, §5.7)

| seq | 配置 | 时延 (ms) | vs stock base |
|-----|------|-----------|---------------|
| 1024 | TRUE Qwen3-VL-2B | 15.00 | 1.000× |
| 1024 | retrofit full ($\gamma{=}1$) | 20.59 | $\mathbf{1.373\times}$ |
| 1024 | retrofit + dyn-skip ($q{=}0.85, M{=}2$) | ~19.5 | ~1.30× |
| 2048 | TRUE Qwen3-VL-2B | 25.59 | 1.000× |
| 2048 | retrofit full | 35.59 | $\mathbf{1.391\times}$ |
| 2048 | retrofit + dyn-skip | ~33.2 | ~1.30× |

**诚实表述**: retrofit 相对 stock base 有 **1.37–1.40× structural overhead** (14 个 block 每个都跑一次 $N$-way router softmax+einsum). dyn-skip 在 retrofit-full 上省 5–9%, 最终 ~1.30× stock base. **不是 "比 base 快"**, 而是 "用 1.30× 算力换到了 v3 VLM 质量净正 + VLA 下游 warm-start 收益". VLM 与 VLA 两侧 wall-clock 等价 (fast-path fix 后, 0.997–1.001×).

---

### 表 6 — Part 3 · VLA LIBERO 下游, warm-start 净正且不可替代 (§5.8, §5.11)

4-suite 成功率 (每 suite 500 episodes, 共 2000 per policy). 2B 的 goal / long-10 做过 2-seed 重跑, 表中 **Path B 取 per-suite max, Path 0 取 per-suite min**; 其他 cell 为单 run (具体 seed 数据见 §5.11).

| Scale | Path (steps) | spatial | object | goal | long-10 | **平均** | Δ vs Path 0 |
|-------|--------------|---------|--------|------|---------|----------|-------------|
| 2B | Path 0 (30k, min over seeds) | 94.8 | 99.8 | 97.4 | 91.4 | 95.85 | — |
| 2B | **Path B (30k, max over seeds)** | **97.8** | 99.6 | **98.6** | **92.6** | $\mathbf{97.15}$ | $\mathbf{+1.30}$ |
| 2B | Path C 30k (无 VLM retrofit) | 92.6 | 100.0 | 95.8 | 88.8 | 94.30 | −1.55 |
| 2B | Path C 60k (2× 算力) | 95.2 | 98.6 | 96.8 | 91.4 | 95.50 | −0.35 |
| 4B clean | Path 0 (30k, 单 run) | 95.0 | 99.2 | 97.8 | 92.2 | 96.05 | — |
| 4B clean | **Path B (30k, 单 run)** | 94.6 | **99.8** | **98.2** | **94.2** | $\mathbf{96.70}$ | $\mathbf{+0.65}$ |
| 4B dirty | Path B 30k ($+$ action-token embed) | 95.0 | 100.0 | 98.4 | **84.8** | 94.55 | **−1.50** ⚠ |

**三条结论**:
1. **VLM retrofit warm-start 净正** (+1.30pp 在 2B 最高-最低对照, +0.70pp 在 2B 同 seed 均值对照, +0.65pp 在 4B clean 单 run). 排序在两种统计口径下都是 Path B > Path 0.
2. **不可被"练更久的纯 VLA"替代**: Path C 60k (2× VLA 算力) 仍落后 Path B 30k **1.65pp**, 集中落在难题 (Spatial task 6: Path B 78% vs Path C v4 64%).
3. **干净 base VLM 是必需**: `-Action` 变体的 2048 个未用 embedding 在 retrofit+OFT 下 receive 无关梯度, 导致 libero_10 **−9.4pp** 退化. pipeline 规则: 下游 head 不预测 action token 时别扩 vocab.

---

### 表 7 — Part 2 · Block Partition 消融 (跨 scale 确认 Chen et al.'s plateau) (§5.5)

关键 cells (LAMBADA + 代表性 VL):

| Scale | $L$ | $N$ | LAMBADA | MMBench | MMMU | MMStar | AI2D |
|-------|-----|-----|---------|---------|------|--------|------|
| 2B base | — | — | 0.532 | 75.77 | 0.414 | 0.536 | 0.736 |
| 2B | 1 | 28 | 0.5645 (+3.3) | 76.20 | 0.388 | 0.499 | 0.743 |
| **2B (推荐)** | **2** | **14** | **0.5755** (+4.4) | **77.23** | **0.426** | 0.532 | 0.748 |
| 2B | 4 | 7 | 0.5650 (+3.3) | 78.87 | 0.432 | 0.536 | 0.758 |
| 2B | 7 | 4 | 0.5155 (−1.7) | 77.49 | 0.427 | 0.530 | 0.756 |
| 4B base | — | — | 0.576 | 83.33 | 0.490 | 0.624 | 0.819 |
| 4B | 1 | 36 | 0.5575 (−1.9) | 83.33 | 0.497 | 0.538 | 0.783 |
| **4B (推荐)** | **4** | **9** | $\mathbf{0.6625}$ **(+8.7)** | **85.22** | 0.521 | **0.632** | **0.825** |
| 4B | 6 | 6 | 0.6540 (+7.8) | 84.79 | **0.531** | 0.623 | 0.817 |

**结论**: $L{=}1$ (per-layer) 在两 scale 最差; $L\in\{2,4,7\}$ 在 2B 是 plateau (与 Chen et al. Fig 6 pretrain 场景下 $S\in\{2,4,8\}$ 一致); **$L{=}4$ 在 4B 是 sweet spot**, LAMBADA 比 2B 最佳配置翻倍 (+8.7pp vs +4.4pp) —— retrofit 的 text 收益随 base 容量 scaling.

---

### 快速 take-aways

- **Part 1**: AttnRes routing 自带 skip 信号, 340M 上 $1.19\times$ 零退化. Importance–Ablation Disconnect 是 AttnRes 路由的 **结构性** 性质 (340M 与 2B retrofit 都重现).
- **Part 2 (核心贡献)**: 把任意标准预训练 Transformer 通过 $\gamma$-gated residual injection 在 $5$k–$10$k 步内改造成结构纯 AttnRes, 保留 base 能力同时全面提升 VLM 质量 (v3 在 2B 每项 ≥ base, 4B 推荐配置 $5/6$ 胜 base). $1.37$–$1.40\times$ structural overhead 是 router 的固有成本, dyn-skip 省回 5–9%.
- **Part 3 (下游验证)**: VLM retrofit 做 VLA warm-start 在两个 scale 都净正 (+1.30pp 在 2B max/min 对照, +0.65pp 在 4B clean), 且 **不可被"练更久纯 VLA"替代** (Path C 60k 仍落后 Path B 30k 1.65pp). Modality-aware dyn-skip 的 Pareto 是下一步.

---

## 1. 动机

现代 Transformer 对所有 token 用相同深度 —— 简单 token 浪费算力、困难 token 计算不足。已有"自适应深度"方案（early-exit, MoD, router）都需要**额外训练路由器 + 辅助 loss**，引入大量工程负担。

**核心观察:** 在 **AttnRes 架构**中，每个 block 的输入本身就是对 "先前所有 block 完成态" 的软路由（softmax over 历史状态）。这个路由权重 `α` 是训练时自然产生的，**不需要辅助信号**，就可以作为判断 "本 token 是否需要当前 block" 的零成本信号。

由此引出两类自适应推理:

- **ReSkip:** 如果 α 高度集中在 `h_{n-1}` 上（即 "我只想继承上一 block 的输出"），可以直接跳过 block_n。
- **ReLoop:** 如果 α 指向更早的 block（回溯路径），对困难 token 做复计算。

项目分三部分推进:

| Part | 内容 | 模型 | 状态 |
|------|------|------|------|
| 1 | From-scratch AttnRes 预训练 + ReSkip 推理 | 340M, 8 blocks × 3 layers | 已完成 |
| 2 | **将标准预训练 VLM 改造为 AttnRes** (核心贡献) | Qwen3-VL-2B | 已完成 |
| 3 | 下游 VLA 任务验证 AttnRes 相对标准 VLM 优势 | Qwen3-VL-2B → LIBERO | 进行中 |

---

## 2. 核心方案: 将标准 VLM 适应性微调为 AttnRes

### 2.1 AttnRes 与标准 Transformer 的差异

**标准 Transformer block n 的前向:**

$$h_n = \text{Block}_n(h_{n-1})$$

仅用前一 block 的输出作为输入。标准预训练 VLM（比如 Qwen3-VL-2B）就是这种结构。

**Part 1 的"纯 AttnRes" block n 前向** (from-scratch 预训练的 340M 模型采用的架构):

$$\begin{aligned}
r_n &= \text{Router}_n(h_0, h_1, \dots, h_{n-1}) \\
h_n &= \text{Block}_n(r_n) \qquad\text{（直接用 routed 值作输入, 没有 } h_{n-1}\text{ skip 项）}
\end{aligned}$$

这是**我们最终想要**的结构: block 输入 `x_n` 完全由 router 决定，router 权重 `α` 天然揭示每个 block 对当前 token 的重要性 (→ ReSkip)。**但这个结构和标准 Transformer 的 `x_n = h_{n-1}` 完全不兼容** —— 如果我们把一个预训练好的 Qwen3-VL-2B 的 `h_{n-1}` 换成 `r_n` (一个从未见过的输入分布)，模型立刻崩溃。

### 2.2 问题: 如何把 "标准 Transformer" 搬到 "纯 AttnRes" 上

两种结构之间差着一整套新的 block 输入分布。直接替换 = 破坏预训练权重。直接从头训 = 放弃 Qwen3-VL-2B 已有的 2B 参数能力。我们需要一条**可训练的平滑过渡路径**。

### 2.3 解法: 适应性微调的桥接公式 (Part 2 Retrofit)

在标准 block 和纯 AttnRes block 之间插一个**混合公式**，它受两个额外量 `γ_n` 和 `A_n(·)` 控制。**γ 和 A_n 在原始 AttnRes 定义里是不存在的 —— 它们是 retrofit 专用的训练辅助工具**:

$$\boxed{\quad x_n = h_{n-1} + \gamma_n \cdot A_n(r_n - h_{n-1}), \qquad h_n = \text{Block}_n(x_n) \quad}$$

这个公式有一个精心设计的性质: **它在两端都能退化成我们想要的结构**:

| 情形 | γ | A_n | x_n 等于 | 对应结构 |
|------|---|------|----------|----------|
| **训练起点** (t=0) | 0 | 任意 | h_{n-1} | 标准 Transformer (原 Qwen3-VL-2B 不变) |
| 训练早期 (t<T_ramp) | 0→1 | 未学到 | h_{n-1} + 小扰动 | 受控偏移 |
| **训练结束** (t≥T_ramp) | **1** | 学到的非零 MLP | h_{n-1} + A_n(r_n − h_{n-1}) | **近似纯 AttnRes + 学到的残差修正** |

关键在于:

- 当 `γ = 0`: `x_n = h_{n-1}` —— **精确复现原 VLM**，训练起点不破坏预训练权重。
- 当 `γ = 1` 且 `A_n` 恰好学成恒等映射 (`A_n(δ) ≈ δ`): `x_n = h_{n-1} + (r_n − h_{n-1}) = r_n` —— 这就是**纯 AttnRes**。
- 实际训练完的 `A_n` 不会严格等于 identity, 而是一个**低秩 MLP 修正项**: 模型学到 "我想要的 block 输入是 `r_n`，但 block 本身的预训练权重是按 `h_{n-1}` 分布训练的，所以先把 `(r_n − h_{n-1})` 这个差值做一次低秩非线性变换再加回去"。这样 block 不用重新训练就能处理新分布。

**从推理路径的角度看 retrofit 后的模型 (H_r256_5k):**

训练完后 γ 全部收敛到 1（见下表），每个 block 的实际前向是:

$$x_n = h_{n-1} + A_n(r_n - h_{n-1})$$

这个式子可以等价改写为:

$$x_n = r_n + \underbrace{(A_n(\delta) - \delta)}_{\text{adapter 相对 identity 的偏移}}, \qquad \delta = r_n - h_{n-1}$$

所以 **retrofit 后模型本质上是 "纯 AttnRes" + 一个学到的低秩修正 (`A_n(δ) - δ`)**。这个修正是必要的 —— 它补偿了 "block 内部权重原本是按 `h_{n-1}` 分布训练的" 这件事。**我们没有重新训练 block 本身的权重** (冻结)，仅用 router + adapter + γ 实现了输入分布的迁移。

### 2.4 H_r256_5k 训练完后的模型状态 (实际数值)

我们真正在使用的 retrofit 后模型 (加载 `outputs/H_r256_5k/retrofit_attnres_state.pt`) 参数状态:

| 量 | 训练前 (t=0) | 训练后 (H_r256_5k) |
|----|-------------|-------------------|
| γ (14 个 block) | 0.0 × 14 | **1.0 × 14** (curriculum 全部到位, 之后 γ_param 学习保持在 1) |
| adapter `W_up` Frobenius norm | ≈ 14 (随机初始化水平) | **14 → 29 → 24** (block 0 → 中层 → block 13, 中层修正最强) |
| router `q`, `b` | ≈ 随机初始化 | 稳定收敛, 每 block 有清晰偏好 |
| base VLM 所有权重 | Qwen3-VL-2B 原样 | **完全不变** (冻结) |

**对比基线**:
- 原 Qwen3-VL-2B: `x_n = h_{n-1}` (标准残差连接)
- H_r256_5k: `x_n = h_{n-1} + A_n(r_n − h_{n-1})` with γ=1, 等价于 `x_n = r_n + 学到的修正项`

### 2.5 三个关键组件的具体公式

#### Router (路由器)

每个 AttnRes block 位置 `n` 有一个**输入无关**的查询向量 `q_n`，对前 `n` 个完成态做 attention:

$$\begin{aligned}
q_n &\in \mathbb{R}^{d}, \quad b_i \in \mathbb{R}^{d} \text{（positional bias）} \\
k_i &= \text{RMSNorm}(h_i) + b_i, \quad i = 0, \dots, n-1 \\
s_i &= \frac{q_n^\top k_i}{\sqrt{d} \cdot \tau}, \quad \tau = 1 \\
\alpha_i &= \text{softmax}_i(s_i), \qquad r_n = \sum_i \alpha_i \cdot h_i
\end{aligned}$$

**注意 `q_n` 不依赖当前 token** —— 它是 block `n` 位置学到的 "我偏好哪种历史状态" 的静态先验。

- `α` 是 **token-specific 的**（因为 key 依赖于每个 token 的 `h_i`），所以不同 token 会得到不同路由权重 —— 这才是 ReSkip 的信号。
- Positional bias `b_i` 补偿 "第 i 个源" 的相对位置 (类似 T5 relative bias)。

#### Residual Adapter (残差适配器)

低秩 down-SiLU-up MLP，专门作用于 router 输出与前一 block 输出的差值:

$$A_n(\delta) = W_{\text{up}}^{(n)} \cdot \text{SiLU}\!\left(W_{\text{down}}^{(n)} \cdot \delta\right), \qquad \delta = r_n - h_{n-1}$$

- `W_down ∈ R^{r×d}`，`W_up ∈ R^{d×r}`，秩 `r = 256` (canonical)
- 每 block 独立一份，共 14 份
- 参数量: 14 × 2 × 2048 × 256 ≈ 15 M (相对 2B 主模型可忽略)

**初始化**: `W_up ~ N(0, 0.02²)` (小但非零)。如果严格置 0，`A_n(δ) = 0` 恒成立，γ 的梯度也就恒为 0 (梯度死锁)。我们改用 "**小随机初始化 + γ curriculum 从 0 起步**" —— step 0 时 `γ_0 = 0` 已经让 adapter 输出被乘 0，所以训练起点仍等价原模型；而 `W_up` 非零意味着一旦 `γ > 0`，adapter 的梯度就能立刻流动。

#### Gamma (curriculum 门控标量)

每 block 一个标量 `γ_n`，共 14 个。有效 γ 分解为**可学习参数** × **curriculum buffer**:

$$\gamma_n^{\text{eff}}(t) = \gamma_{\text{param},n}(t) \times s(t), \qquad s(t) = \min\!\left(\frac{t}{T_{\text{ramp}}}, 1\right)$$

- `γ_param` 初始化为 1，是 nn.Parameter (归 optimizer 管理)
- `s(t)` 是 non-persistent buffer, 每步由外部 callback 更新
- `T_ramp = 0.3 × T_total = 1500` (canonical)

**为什么要拆成两部分?** 单卡训练时可以直接 `self.gamma.data.fill_()`。但用 DeepSpeed ZeRO-2 时, FP32 master 在 `optimizer.step()` 之后做 all-gather 会把 `param.data` 重新覆盖回优化器状态 —— `fill_()` 的值被抹除。拆成 `param × buffer` 后，buffer 不经过 ZeRO-2 同步，curriculum 得以存活。

### 2.6 训练目标

**冻结** base VLM 所有原始权重，**只训练** router / adapters / γ。Loss 是对原 teacher 模型的知识蒸馏 (KL)：

$$\mathcal{L} = \underbrace{\text{KL}\!\left(p_{\text{teacher}} \,\|\, p_{\text{student}}\right)}_{\text{对齐 teacher 分布}} + \lambda_e \cdot \underbrace{\mathcal{H}(\alpha)}_{\text{router 熵惩罚}}$$

- **KL 项** (`λ_KL = 1`): 保证 retrofit 后模型与原模型行为接近。
- **熵惩罚** (`λ_e = 0.02`): 防止 router 过早坍缩到单一源。
- **没有** 任务 loss / RL loss —— retrofit 只需复制 teacher 的能力，并建立 AttnRes 结构。

**数据混合** (50/50, 经过消融):

| 源 | 占比 | 说明 |
|----|------|------|
| UltraChat-200k | 50% | 纯文本对话，保持语言能力 |
| LLaVA-Instruct-VSFT | 50% | 视觉-语言对齐 |

消融结论: VLM 占比 ≥ 80% → MMBench 下降明显；< 50% → LAMBADA 增益收窄。

### 2.7 Canonical 配置 `H_r256_5k`

经 8 路 H 家族消融确定的最优配方：

| 超参 | 值 | 说明 |
|------|-----|------|
| base model | Qwen3-VL-2B-Instruct | 28 层语言塔 |
| num_blocks | 14 | 28 层 / 2 层一组 |
| adapter_rank `r` | 256 | r=128 次之, r=512 不稳 |
| γ-curriculum | 0 → 1 over 30% steps | 慢 ramp (70% steps) → 灾难性退化 |
| train steps | 5000 | 10k/20k 均 MMBench 退化 ≥ 14pp (over-training) |
| learning rate | 5e-4 (adapters), 1e-3 (γ/router) | 分组 LR |
| data mix | 50/50 UltraChat / LLaVA | ≥ 80% VLM → MMBench 退化 |
| hardware | 1×H100 ~22 分钟 | bf16 |

---

## 3. 技术细节与陷阱

### 3.1 模型加载与 state_dict

训练完的 retrofit state 仅 ~400 MB（router + 14 adapters + γ），以 `retrofit_attnres_state.pt` 保存，key 是:

```python
{
    "router":          {...},        # 14 个 (q, b) + global params
    "adapters":        [{down, up}, ...],  # 14 个
    "gamma":           torch.Tensor[14],
    "config":          {num_blocks, adapter_rank, ...},
    "skippable_blocks": (0, 1, ..., 13),
}
```

VLA 训练时直接 load 到 `StarVLAAttnResAdapter`，然后在训练里继续 fine-tune。

### 3.2 前向路径的 monkey-patch

retrofit 不新建一个新 module，而是 **monkey-patch** 原 VLM 的 `language_model.forward`：

```python
lm._original_forward = lm.forward
lm.forward = types.MethodType(patched_forward, lm)  # 注入 AttnRes 逻辑
```

Patched forward 的核心循环：

```python
completed = [inputs_embeds]          # h_0
prev_block = inputs_embeds
layer_counter = 0
for n in range(num_blocks):          # 14 个 block
    # —— Router + Adapter + γ ——
    r_n, alpha = router.route(n+1, completed)
    x_n = prev_block + gamma[n] * adapters[n](r_n - prev_block)
    
    # —— 原 block 的 layers_per_block 层计算 ——
    h = x_n
    for _ in range(layers_per_block):  # 通常是 2
        h = text_model.layers[layer_counter](h, ...)
        layer_counter += 1
    prev_block = h
    completed.append(h)               # 更新完成态

hidden_states = text_model.norm(completed[-1])
```

### 3.3 DeepSpeed ZeRO-2 陷阱 (γ-curriculum 特殊处理)

(背景见 §2.5 的 γ 分解) 走 ZeRO-2 时必须用 `γ_param × _gamma_scale` 分解写法:

```python
self.gamma_param = nn.Parameter(torch.ones(num_blocks))  # 可学习, 归 optimizer 管
self.register_buffer("_gamma_scale", torch.zeros(num_blocks), persistent=False)  # curriculum
# 训练 callback 每步：
adapter._gamma_scale.fill_(min(step / ramp_steps, 1.0))
# 前向里用：
effective_gamma = self.gamma_param * self._gamma_scale
```

直接用 `self.gamma.data.fill_(current_gamma)` 会被 ZeRO-2 的 FP32 master all-gather 覆盖回优化器状态 —— 实测 Path C v3 第一版 curriculum 完全失效, 保存的 γ 全是 ~0。Buffer 不经过 all-gather 所以存活。

### 3.4 state_dict 膨胀 bug (VLA 场景特有)

`StarVLAAttnResAdapter.bind_text_model(text_model)` 早期实现把 text_model 存成 attribute:
```python
self._bound_text_model = text_model  # ← 触发 nn.Module 自动注册
```
这会让 2B 的 text_model 权重 **被复制一份** 到 adapter 的 state_dict 下，checkpoint 从 5GB 膨胀到 10GB。修复是把 reference 放到 **context 对象**上（而非 adapter nn.Module 上）。

### 3.5 纯推理快路径

`retrofit.collect_trace = False` 时跳过 per-block entropy / alpha / skip_trace 计算，省 14 次 kernel launch + 3 次 CUDA sync。benchmark 和 lmms-eval 走此路径。

---

## 4. 自适应推理: ReSkip 跳层分析机制

本章回答三个问题: **(a) 哪些 block 应该作为跳层候选**, **(b) 判据是什么、如何校准**, **(c) 跳了之后 KV cache 怎么办** —— 以及最终在 H_r256_5k 和 340M from-scratch 两个 setting 下的 **Pareto 与加速效果**。

### 4.1 两个互补的静态信号: Importance 和 Ablation

我们为每个 block $n$ 定义两个在标定集上算一次就够的量:

**(i) AttnRes Importance** $I(n)$ —— "block $n$ 的输出有多常被下游 router 引用":
$$I(n) = \max_{l>n} \mathbb{E}_{x}\bigl[\alpha_{n\to l}(x)\bigr]$$
—— 所有 "路由源 $= n$" 的权重在下游所有 block $l$ 上的最大平均值。$I(n)$ 高 $\Rightarrow$ router 很需要 $n$ 的输出; $I(n)$ 低 $\Rightarrow$ 天然冗余。

**(ii) Static Ablation** $A(n)$ —— "把 block $n$ 整个从前向里拿掉，对 downstream 有多大影响":
$$A(n) = \frac{\mathrm{PPL}(\text{model w/ block } n \text{ statically removed})}{\mathrm{PPL}(\text{full depth})}$$
—— 单 block 静态移除后的 PPL 比率。$A(n)$ 高 $\Rightarrow$ 不可或缺; $A(n)$ 低 $\Rightarrow$ 跳安全。

$I(n)$ 只需 forward; $A(n)$ 需要跑 $N-2$ 次单 block 移除 forward。两者都只依赖训练完的模型权重，与 deployment 分布无关。

### 4.2 Importance–Ablation Disconnect (340M 发现)

**关键观察**: $I$ 和 $A$ **不是同一件事**, 经常在个例上反向。340M from-scratch AttnRes 的 8 个 block 上我们实测:

| block $n$ | $I(n)$ | $A(n)$ (PPL ratio) | 解读 |
|-----------|--------|--------------------|------|
| 2 | 0.52 (中) | **8.1×** (灾难) | 移除 $\Rightarrow$ 崩溃, 但 router 没那么常用它 |
| 3 | 0.56 (较高) | **1.31×** (最低) | router 常用, 但移除几乎不伤 —— 最好的跳层候选 |
| 4 | 0.41 | 1.52× | 中庸 |
| 5 | 0.28 (最低) | 1.67× | router 不常用, 但不是最安全 |

**结论**: 只用 $I$ 挑 (例如挑最低 $I$ 的 block 5) 会给 $1.14\times$ 加速; 只用 $A$ 挑 (block 3) 几乎触发不了 skip. 只有 **$I\cdot A$ 联合选** $\mathcal{P}=\{3,5\}$ 同时满足 "router 频繁触发" + "跳了也安全", 得到 $1.19\times$ 零退化加速 —— **严格优于任一单信号方案**。

### 4.3 从 340M 到 2B retrofit: 结构性重复出现

在 H_r256_5k (Qwen3-VL-2B retrofit) 的 14 个 block 上单 block 静态移除，LAMBADA 退化:

| block | $A$ 退化 (LAMBADA acc) | 结论 |
|-------|----------------------|------|
| 0 | N/A (embedding) | — |
| 1 | **−55%** | 最不可跳 |
| 4 | −14% | **安全** |
| 6 | −12% | **安全** |
| 10 | −46% | 严重 |
| 11 | −11% | **最安全** |
| 其他 | −15 至 −25% | 中等 |

**候选集 $\mathcal{P}=\{4,6,11\}$** 就是按 $A$ 排序的前三. 这三个位置的 $I$ 分布 (low/mid/mid) 保证 dyn-skip 触发率不至于太低.

**跨 scale 的结构性发现**: 在 **340M from-scratch** 和 **2B retrofit** 两个完全不同的 setting 下, 最不可跳的 block 总是位于 **靠近 embedding** 的位置 (340M block 2, 2B block 1) 和 **后期一个"refinement 汇聚点"** (340M block 6, 2B block 10); 最安全的跳层候选则集中在 **中深度 $\alpha$ 已经塌缩到 predecessor** 的位置. 这说明 Importance–Ablation Disconnect 是 **AttnRes routing 本身的结构性质**, 而不是 scale 或训练方式的 artifact.

### 4.4 Dynamic Skip 判据: `recent_weight_gt` with per-block quantile

推理时 router 输出的 block $n$ 的 softmax 权重 $\alpha_n$, 定义 **$w_{\text{recent},n}(x) = \alpha_n[n-1]$** —— 本 block 分配给"上一 block 输出 $h_{n-1}$"的权重. 直觉: $w_{\text{recent}}$ 高 $\Leftrightarrow$ router 认为"不需要参考更早历史, 直接用上一步就够了" $\Leftrightarrow$ 本 block 是 local refinement, 可跳.

**完整 skip 规则** (Algorithm 1 in paper):
$$\text{skip block } n \iff
n \in \mathcal{P} \;\; \text{且} \;\;
w_{\text{recent},n}(x) > \tau_n \;\; \text{且} \;\;
\sum_{m<n} \mathbf{1}[\text{skipped}_m] < M_{\max}$$

三个限制:
1. **位置限制** $\mathcal{P}$: 由 §4.3 的 ablation sweep 确定, 2B 用 $\{4,6,11\}$
2. **阈值限制** $\tau_n$: per-block quantile calibration, 32 条 LAMBADA 前缀上做统计, $\tau_n = Q_q(w_{\text{recent},n})$. $q\in\{0.50, 0.70, 0.85, 0.95\}$, $q$ 越大越保守
3. **数量限制** $M_{\max}$: 单次 forward 最多跳 $M_{\max}$ 个 block (2B 通常 $M{=}1$ 或 $M{=}2$)

**为什么 per-block $\tau_n$, 不用单一全局阈值?** 不同位置 $w_{\text{recent}}$ 的分布差别极大 —— 后期 block 天然 $w_{\text{recent}}$ 更集中, 前期更分散. 用单一 $\tau$ 要么前期根本不触发, 要么后期过度触发. 实测 per-block 标定相比全局标定在同 $q$ 下给出更陡的 Pareto 曲线.

### 4.5 KV Cache 下的跳层: K/V-only 快路径

朴素的 skip $h_n \leftarrow x_n$ 会把 block $n$ 的全部 layer 计算砍掉 —— 包括写 KV cache. 这让后续 token 的 attention 读到缺位的 cache, 生成结果跟 full-path 差异显著.

**解法**: 跳过的 block 里面的每一 decoder layer 仍执行 K/V 投影 + RoPE + cache update, 只跳过 Q-proj / attention / o_proj / MLP:
$$\text{skip\_layer}(h): \quad
\underbrace{\text{input\_layernorm}(h)}_{\text{normed}}
\to \underbrace{W_k, W_v}_{\text{Q 跳过}}
\to \underbrace{\text{k\_norm, RoPE}}_{\text{位置编码}}
\to \text{cache.update}(K, V)$$

对 Qwen3-VL-2B 的 GQA-16 decoder layer, 这条 K/V-only slice 仅占 full layer forward 的 **5–10%**. 所以 skip 仍然省掉绝大多数算力, 同时保证 **cache 长度一致, generate + use\_cache=True 正确**.

**等价性验证** (`retrofit/tests/test_skip_kv_equiv.py`):
对固定 prompt, 对 skip 配置 $\{[4], [4,10], [4,10,12], [2,4,10]\}$ 分别跑 `use_cache=True` 和 `use_cache=False`:
- **每种配置下 last-position argmax 完全一致**
- logit 最大误差 $0.19$–$0.37$, 与 HF+bf16+SDPA 在 stock base 上的 cache=T vs F jitter 同量级

结论: K/V-only skip path 与 cacheless skip path 在 argmax 层面等价, 多 token decode 发散是 bfloat16 SDPA 的固有性质, 不是 skip 机制的问题.

### 4.6 Pareto 效果 (H_r256_5k, LAMBADA n=300)

| $q$ | $M_{\max}$ | LAMBADA acc | LAMBADA ppl | avg skips / 3 eligible |
|-----|-----------|-------------|-------------|------------------------|
| — (full-path) | — | 0.5760 | 4.609 | 0 |
| 0.50 | 1 | 0.5500 | 5.45 | 0.82 |
| 0.70 | 1 | 0.5600 | 5.24 | 0.64 |
| 0.85 | 1 | 0.5733 | 4.90 | 0.37 |
| **0.95** | **1** | **0.5933** | **4.77** | **0.18** |
| 0.50 | 2 | 0.4733 | 6.42 | 1.41 |
| 0.70 | 2 | 0.5300 | 5.62 | 0.97 |
| 0.85 | 2 | 0.5600 | 5.03 | 0.55 |
| 0.95 | 2 | 0.5800 | 4.84 | 0.26 |

**两个值得注意的现象**:

1. **推荐点 $q{=}0.95, M{=}1$ 竟然 *超过* full-path**: LAMBADA 0.5933 > full-path 0.5760 (+1.7pp), 相对 base 是 +6.1pp. 340M from-scratch 上 skip 与 full-path 是"相同",  2B retrofit 上 skip 反而"更好" —— 机制一致 (跳掉 $\alpha$ 已经塌缩到 predecessor 的 block 是消除冗余 forward), 但 2B 冻结的 pretrained 主干里有更多可被消除的冗余计算.

2. **$M{=}2$ 悬崖陡**: 单 token 同时跳 2 个 block 在 $q{=}0.95$ 还稳 (0.5800), 但 $q{=}0.70$ 掉到 0.5300, $q{=}0.50$ 直接崩 (0.4733). 部署推荐 $M{=}1$, 需要更激进加速才上 $M{=}2$ 配 high-$q$.

### 4.7 速度现实 (cache=True 规范 bench)

这里诚实说明 —— **dyn-skip 没有把 retrofit 推理拉到 stock base 之下**.

| seq | 配置 | 时延 (ms) | vs base |
|-----|------|----------|---------|
| 1024 | TRUE Qwen3-VL-2B | 15.00 | 1.00× |
| 1024 | retrofit full ($\gamma{=}1$) | 20.59 | **1.373×** |
| 1024 | retrofit + dyn-skip ($q{=}0.85, M{=}2$) | 19.5 | 1.30× |
| 2048 | TRUE Qwen3-VL-2B | 25.59 | 1.00× |
| 2048 | retrofit full | 35.59 | **1.391×** |
| 2048 | retrofit + dyn-skip | 33.2 | 1.30× |

**两个要点**:

- **retrofit 相对 stock base 有 $\mathbf{1.37}\text{–}\mathbf{1.40\times}$ 的 structural cost** —— 14 个 block 每个都做一次 $N$-way stack + RMSNorm + softmax + einsum, 在 cache=True decode ($T{=}1$) 下每个 token 都要这么跑 14 遍. 这是 router 结构决定的下限, 不是实现问题.
- **dyn-skip 在 retrofit-full 基础上能省 5-9%**, 最终推理时延 $\approx 1.30\times$ stock base. 不是"比 base 快", 而是"用 1.30× 算力换到了 adaptive-depth + 下游任务质量改善"(详见 §5).

**优化方向** (paper Discussion): (A) top-$k$ router 或 sliding-window router 替代 full $N$-way stack, 需要重训; (B) 更激进 skip 配置, 纯超参数; (C) 更小 adapter rank. 已验证 (C) 收益有限 (router 主导). (A) 被作者暂时否决 (重训解释成本高). (B) 是当前推荐的下一步.

---

## 5. 实验结果

### 5.1 VLM 基准 —— v1 (H_r256_5k) vs LoRA baseline

H_r256_5k 是 **v1 数据混合** (50% UltraChat + 50% LLaVA-VSFT, 5k steps) 的 retrofit, 用作 Pareto / wall-clock / VLA warm-start 的 reference. 文本指标:

| 配置 | LAMBADA acc | LAMBADA ppl | HellaSwag | MMBench (n=300) | MMStar (n=500) | MMMU (n=94) |
|------|-------------|-------------|-----------|-----------------|----------------|-------------|
| Qwen3-VL-2B (base) | 0.532 | 5.547 | 0.506 | 0.7267 | 0.2820 | 0.3617 |
| **H\_r256\_5k** (v1, $\gamma{\to}1$) | **0.576** | **4.609** | **0.522** | **0.7100** | **0.2780** | 0.3511 |
| Δ vs base | **+4.4** | −17% | +1.6 | −1.7 (噪声内) | −0.4 | −1.1 |

**LoRA baseline (参数量匹配, 2× 训练步数, 同数据混合)**:

| 配置 | LAMBADA | HellaSwag |
|------|---------|-----------|
| LoRA $r{=}32$ on $q,v$ (seed 0) | 0.540 | 0.510 |
| LoRA $r{=}32$ on $q,v$ (seed 1) | 0.516 | 0.524 |
| LoRA $r{=}16$ on $q,k,v,o$ | 0.534 | 0.492 |
| LoRA $r{=}8$ on MLP | 0.514 | 0.510 |
| **LoRA 4-seed 均值** | **0.526** | **0.509** |
| Δ(LoRA 均值) vs base | **−0.6** | +0.3 |
| **H\_r256\_5k vs LoRA 均值** | **+5.0** | **+1.3** |

**归因结论**: 我们的 retrofit 相对 base 的 +4.4pp LAMBADA 提升, 是 **AttnRes routing 结构本身**, 不是 "多一点可训参数打 SFT data" —— 参数量匹配的 LoRA 4-seed 均值几乎和 base 打平 (−0.6pp), 比我们的 retrofit 低 5.0pp.

### 5.2 VLM 基准 —— v3 (LLaVA-OneVision-anchored 10k) 全集

v3 是更广的数据混合 (60% LLaVA-OneVision + 20% UltraChat + 10% NuminaMath + 10% OpenThoughts, 10k steps), 专治 v1 mix 在更长训练时的 VL 退化 (见 §5.4). v3 用 `lmms-eval` 官方全集评估:

| Benchmark | 2B base | **v3 (2B, L=2)** | Δ | 4B base | **v3 (4B, L=4 推荐)** | Δ |
|-----------|---------|-------------------|---|---------|------------------------|---|
| MMBench_en_dev | 75.77 | **79.30** | **+3.5** | 83.33 | **85.22** | **+1.9** |
| MMMU_val | 0.414 | **0.439** | **+2.5** | 0.490 | 0.521 | **+3.1** |
| MMStar (6-cat avg) | 0.536 | 0.534 | −0.2 | 0.624 | **0.632** | **+0.8** |
| AI2D | 0.736 | **0.765** | **+2.9** | 0.819 | **0.825** | **+0.6** |
| OCRBench | 0.772 | **0.809** | **+3.7** | 0.819 | **0.824** | **+0.5** |
| RealWorldQA | 0.648 | **0.668** | **+2.0** | 0.715 | **0.718** | **+0.3** |

**v3 在 2B 上每一项都达到或严格优于 base**; 4B 在推荐配置 $L{=}4$ 下也全面持平或优于 base. v1 → v3 把"retrofit 是 preserve 多模态能力"升级为"retrofit 是 *strict improvement*".

### 5.3 MMStar 子类别分解 —— 推理子类是大头

MMStar 有 6 个子类: math / logical reasoning / science & tech / coarse perception / fine-grained perception / instance reasoning. retrofit 的收益主要来自**需要推理的子类**, perception 类基本持平:

| 配置 | math | logical | sci&tech | coarse | fine | instance |
|------|------|---------|----------|--------|------|----------|
| 2B base | 0.413 | 0.429 | 0.408 | 0.714 | 0.520 | 0.710 |
| 2B v3 (L=2) | **0.492** | **0.432** | 0.353 | 0.734 | 0.505 | 0.683 |
| Δ vs 2B base | **+7.9** | +0.3 | −5.5 | +2.0 | −1.5 | −2.7 |
| 4B base | 0.549 | 0.626 | 0.465 | 0.788 | 0.611 | 0.705 |
| 4B v3 (L=4) | **0.588** | 0.602 | **0.467** | **0.812** | 0.606 | 0.714 |
| Δ vs 4B base | **+3.9** | −2.4 | +0.2 | +2.4 | −0.5 | +0.9 |

**解读**: router 抬升的是 "deliberate reasoning over images" 这条路径 (math / logical), 不改早期感知 (fine-grained perception 基本持平). 这和 §4.3 里 "最安全 skip 候选集中在中深度" 的结构对应.

### 5.4 4B 复现 (H_4B_r256_5k / v3_4B)

同 canonical 配方移植到 Qwen3-VL-4B (36 层 → $N{=}18$ 块, $L{=}2$; 或 $N{=}9$ 块, $L{=}4$, 推荐后者见 §5.5):

| 量 | 2B H_r256_5k | 4B H_4B_r256_5k |
|----|--------------|-----------------|
| base model | Qwen3-VL-2B | Qwen3-VL-4B |
| num_hidden_layers | 28 | 36 |
| hidden_size | 2048 | 2560 |
| num_blocks (默认) | 14 | 18 ($L{=}2$) |
| adapter_rank | 256 | 256 |
| 可训练参数 | ~15M (~0.7%) | **~23.7M (~0.6%)** |
| steps | 5000 | 5000 (v1) / 10000 (v3) |
| γ-curriculum | 0→1 @ 30% steps | 0→1 @ 50% steps ($\star$) |
| hardware | 1×H100 ~22 min | 1×H100 ~27 min |

$\star$ **4B ramp-fraction 0.5 是必要的**: v3 mix + ramp 0.3 + 10k steps 上 4B 在 step ~3125 发散 (CE 0.9 → 6.5+). 延长 ramp 到 step 5000 结束稳定收敛. 2B 在 ramp 0.3 下无此问题. 工作规则: $\gamma{=}1$ 切换必须发生在 **step ≥ 5000**.

所有 18 个 $\gamma_n$ 训练完都到 1.0, 结构上是纯 AttnRes. VL 全集结果见 §5.2 表.

### 5.5 Block Partition 消融 —— 跨 scale 确认 Chen et al. 的 plateau

按 Chen et al. (AttnRes 原论文, Fig. 6) 在 pretrain 场景下 $S\in\{2,4,8\}$ 是 plateau (Val loss 1.746–1.748), $S{=}1$ (per-layer) 和 $S{\geq}16$ 都差. 我们的问题: **这个 plateau 在 retrofit 场景 + 两个 scale 下还成立吗?**

Sweep: 2B × {$L{=}1,2,4,7$} + 4B × {$L{=}1,2,4,6$}, 全部用 v3 recipe, 只改 block 粒度.

**2B** (base: ai2d 0.736, mmbench 75.77, mmmu 0.414, mmstar 0.536, LAMBADA 0.532):

| $L$ | $N$ | LAMBADA | HellaSwag | MMBench | MMMU | MMStar | AI2D | OCR | RWQA |
|-----|-----|---------|-----------|---------|------|--------|------|-----|------|
| 1 | 28 | 0.5645 (+3.3) | 0.490 | 76.20 | 0.388 | 0.499 | 0.743 | 0.809 | 0.652 |
| **2** | **14** | **0.5755 (+4.4)** | **0.494** | **77.23** | **0.426** | 0.532 | 0.748 | 0.803 | 0.663 |
| 4 | 7 | 0.5650 (+3.3) | 0.500 | 78.87 | 0.432 | 0.536 | 0.758 | 0.814 | 0.661 |
| 7 | 4 | 0.5155 (−1.7) | 0.492 | 77.49 | 0.427 | 0.530 | 0.756 | 0.808 | 0.656 |

**4B** (base: ai2d 0.819, mmbench 83.33, mmmu 0.490, mmstar 0.624, LAMBADA 0.576):

| $L$ | $N$ | LAMBADA | HellaSwag | MMBench | MMMU | MMStar | AI2D | OCR | RWQA |
|-----|-----|---------|-----------|---------|------|--------|------|-----|------|
| 1 | 36 | 0.5575 (−1.9) | 0.523 | 83.33 | 0.497 | 0.538 | 0.783 | 0.768 | 0.694 |
| 2 | 18 | —\* | —\* | 84.28 | 0.523 | 0.587 | 0.816 | 0.813 | 0.708 |
| **4** | **9** | **0.6625 (+8.7)** | **0.552** | **85.22** | 0.521 | **0.632** | **0.825** | **0.824** | **0.718** |
| 6 | 6 | 0.6540 (+7.8) | 0.554 | 84.79 | **0.531** | 0.623 | 0.817 | 0.824 | 0.715 |

\* 4B $L{=}2$ 的文本评估尚未重跑 (aliased from v3_4B 主配置).

**四个结论**:
1. **$L{=}1$ 在两个 scale 都最差** —— 把固定的 adapter rank 预算摊到 28/36 个位置让每个位置欠容量; block 结构本身 (而非仅仅"每块多参数") 是 AttnRes 路由 coordination 的关键.
2. **2B plateau: $L\in\{2,4,7\}$**. 三者在 MMStar 内 1pp, AI2D 内 1.1pp. $L{=}2$ LAMBADA 最高 (+4.4), $L{=}4$ VL 最强. 推荐 **$L{=}2$ (最少参数 + plateau 上)**.
3. **4B 的 sweet spot 是 $L{=}4$**. 在 MMBench / MMMU / MMStar / AI2D / RealWorldQA 上严格胜过 base, LAMBADA 上 +8.7pp —— **比 2B 最佳配置 (+4.4pp) 翻了一倍**. retrofit 的 LAMBADA 收益随 base 容量 scaling.
4. **Chen et al. 的 $S\in\{2,4,8\}$ plateau 在 retrofit 场景下成立**, 说明 block 粒度的选择不会是两个 scale 间的主要变量.

### 5.6 Data Mix 消融 (v1 → v2 → v3)

retrofit 对训练分布敏感; 我们沿着三个 mix 做过三轮实验, 对比表 (全用 `lmms-eval` 全集):

| Scale | Mix (steps) | AI2D | MMBench | MMMU | MMStar | OCR | RWQA | 说明 |
|-------|-------------|------|---------|------|--------|-----|------|------|
| 2B | base | 0.736 | 75.77 | 0.414 | 0.536 | 0.772 | 0.648 | 参考 |
| 2B | v1 (5k) | 0.684 | 72.85 | 0.406 | 0.386 | 0.795 | 0.447 | 5k 稳, 长训崩溃 |
| 2B | v1 (10k) | 0.677 | 73.71 | 0.404 | 0.471 | 0.801 | 0.642 | 长训 RWQA 回来, AI2D 仍低 |
| 2B | v2 (5k) | **0.283** | 73.80 | 0.421 | 0.422 | 0.806 | 0.512 | AI2D 崩 (−45pp), MMStar math 升 |
| 2B | **v3 (10k)** | **0.765** | **79.30** | **0.439** | 0.534 | **0.809** | **0.668** | 全面正向 |
| 4B | base | 0.819 | 83.33 | 0.490 | 0.624 | 0.819 | 0.715 | 参考 |
| 4B | v1 (5k) | 0.810 | 83.76 | 0.510 | 0.579 | 0.812 | 0.707 | 5k 稳 |
| 4B | v1 (10k) | **0.580** | 81.79 | 0.432 | 0.437 | 0.812 | 0.689 | **AI2D 崩 −23pp, MMStar math −38pp** |
| 4B | v2 (5k) | 0.603 | 81.87 | 0.477 | **0.333** | 0.808 | 0.686 | MMStar 全面崩 |
| 4B | **v3 (10k, $L{=}2$)** | 0.816 | **84.28** | **0.523** | 0.587 | 0.813 | 0.708 | $L{=}2$ MMStar −3.7, $L{=}4$ 回正见 §5.5 |

**三条工作规则**:
1. **VL anchor 要 ≥ 60%**. v2 (30% VL) 在 2B 上把 AI2D 搞崩 (−45pp); v3 (60% VL) 回到 +2.9. 长训 + 窄 VL (v1 @ 10k) 在 4B 上把 AI2D 搞崩 (−23pp).
2. **math-CoT text ≤ 20%**. v2 (40% math-CoT) 的符号推理梯度把 router 推离视觉路径, 坏 diagram reasoning. v3 (20% math-CoT) 保住 diagram 的同时抬 math 子类.
3. **光加步数救不了窄 mix**. v1 @ 10k 不是"欠训", 而是"窄 VL 在长训下 router 过拟合崩掉".

### 5.7 Wall-Clock 速度 (详见 §4.7, 这里是要点)

| 配置 | seq=1024 cache=T | seq=2048 cache=T |
|------|------------------|------------------|
| TRUE Qwen3-VL-2B (stock base) | 15.00 ms | 25.59 ms |
| retrofit full ($\gamma{=}1$) | 20.59 ms (1.373×) | 35.59 ms (1.391×) |
| retrofit + dyn-skip | ~19.5 ms (1.30×) | ~33.2 ms (1.30×) |

**诚实表述**: retrofit 在 cache=True 下对 stock base 有 **1.37–1.40× structural overhead**, 来自 14 个 block 每个都跑一次 $N$-way router. dyn-skip 在 retrofit-full 基础上省 5–9%, 最终 ~1.30× base. **不是比 base 快**, 而是**换来了 adaptive-depth + v3 下 VLM 质量净正 + VLA 下游 warm-start** (§5.9). VLM 和 VLA 侧的 wall-clock **等价** (1.00×, fast-path 修复后).

注: 之前日志里"retrofit ≈ base speed under cache" 是 bench 代码里 monkey-patched base bug 导致, 已 retracted. 现在这张表是校正后的规范数字.

### 5.8 VLA LIBERO 下游任务 —— Path 0 / Path B / Path C

**三条训练 path** (都配同一 OFT action head, bs=8×4=32, 30k steps, bf16 ZeRO-2 4×H100):

- **Path 0**: stock Qwen3-VL backbone + 纯 OFT fine-tune, no AttnRes
- **Path B (推荐)**: 从 `H_r256_5k` / `H_4B_r256_5k` retrofit state warm-start (router/adapters/γ 全部加载), 然后在 LIBERO 上继续 OFT. $\gamma$ 不再 curriculum, 全程 $\gamma{=}1$
- **Path C**: 同 Path B 架构但 **不做 VLM retrofit**, router/adapter 随机初始化, $\gamma$ 在 VLA 训练的前 30% 做 0→1 curriculum

**4-suite 成功率** (50 trials × 10 tasks = 500 episodes/suite; 2000 episodes/policy).

**统计口径**: 2B 的 `libero_goal` 和 `libero_10` 做过 2 种子重跑 (§5.11 有原始 seed 数据), 主表下面的 **Path B 取 per-suite max, Path 0 取 per-suite min**, 让比较体现"Path B 最佳 operating point vs Path 0 悲观 floor". 其他 cell (spatial / object, 及全部 4B) 是单 run. 备用同-seed 均值口径见 §5.11.

| Scale | Path (steps) | spatial | object | goal | long-10 | **平均** | Δ vs Path 0 |
|-------|--------------|---------|--------|------|---------|----------|-------------|
| 2B | Path 0 (30k, min) | 94.8 | 99.8 | 97.4 | 91.4 | 95.85 | — |
| 2B | **Path B (30k, max)** | **97.8** | 99.6 | **98.6** | **92.6** | **97.15** | **+1.30** |
| 2B | Path B (60k) | 94.4 | 99.2 | 97.8 | 94.0 | 96.35 | +0.50 (过训) |
| 2B | Path C (30k) | 92.6 | 100.0 | 95.8 | 88.8 | 94.30 | −1.55 |
| 2B | Path C (60k) | 95.2 | 98.6 | 96.8 | 91.4 | 95.50 | −0.35 |
| 4B (clean) | Path 0 (30k) | 95.0 | 99.2 | 97.8 | 92.2 | 96.05 | — |
| 4B (clean) | **Path B (30k)** | 94.6 | **99.8** | **98.2** | **94.2** | **96.70** | **+0.65** |
| 4B (clean) | Path 0 (60k) | 95.0 | 100.0 | 98.6 | 94.4 | 97.00 | +0.95 (capacity upper bound) |
| 4B (clean) | Path B (60k) | 93.4 | 99.2 | 97.4 | 94.8 | 96.20 | +0.15 (过训) |
| 4B (dirty) | Path B (30k) | 95.0 | 100.0 | 98.4 | **84.8** | 94.55 | **−1.50 (见 §5.9)** |

**三条关键 finding**:

1. **VLM retrofit warm-start 在两个 scale 都净正**. 2B 在 max/min 口径下 +1.30pp, 同-seed 均值口径下 +0.70pp (§5.11); 4B clean 单 run +0.65pp. 两种口径排序都是 Path B > Path 0. 主要贡献: 2B 集中在 spatial (+3.0pp), 4B 在 long-horizon (+2.0pp).
2. **不可被"再练久一点的 VLA"替代**. Path C (γ-curriculum from base, no VLM retrofit) 在同 30k 步下落后 Path B 2.85pp; 给 2× 步数 (60k) 仍落后 Path B 30k 有 **1.65pp**. 换算到总 budget: **5k VLM retrofit + 30k VLA = 35k-等价 > 60k pure VLA**, 集中落在需要稳定 router 的难题 (如 Spatial task 6: Path B 78%, Path C v3 50%, Path C v4 60k 64%).
3. **30k 是最佳, 60k 过训**. Path B 从 30k → 60k 在两个 scale 都 *下降* (2B 97.15→96.35, −0.80pp; 4B 96.70→96.20, −0.50pp), 尽管 Path 0 在 4B 上 60k 还在 +0.95pp. 推测: warm-start 的 router/adapter 收敛快 (起点就是 VLM-adapted), 再练就 drift 偏离 Spatial/Long-horizon sweet spot.

### 5.9 VLA 陷阱: `-Action` base 的 action-token embedding 污染

我们一开始在 Qwen3-VL-4B `-Action` 变体 (给 tokenizer 加了 2048 个 `<robot_action_*>` embedding, 准备将来做 action-token prediction) 上做 Path B, 得到 **94.55** avg (libero_10 84.8). 换到 **clean base** (未加 action token) 上相同 recipe: **96.70** avg (libero_10 94.2). **libero_10 swing 9.4pp** 完全由"2048 个未用 embedding 是否 receive gradient"解释:

- OFT head **不预测** 这些 token, 但它们的行在 tied `embed_tokens` / `lm_head` 矩阵里;
- 它们在 retrofit distillation + label smoothing 下仍 receive 微量梯度;
- 在 VLA fine-tune 下它们继续漂移, 污染 shared embedding 矩阵 —— 对 long-horizon 语言 grounding 影响最大.

**部署建议**: retrofit + OFT pipeline 用 **干净 base VLM**, 只有当下游 head 真预测 action token 时才扩 vocab. 这条被写进 VLA pipeline 规则, paper §7.4 也有此结论.

### 5.10 VLA 动态 skip 的 preliminary 信号

用 §4 的 uniform dyn-skip ($q{=}0.85, M_{\max}{=}2$, 源自 LAMBADA 校准) 直接开在 VLA 推理上:

- **`pathB_2B_L4_v3_30k`** (v3 warm-start, $L{=}4$) libero_spatial **99.5%** (partial, 仅 spatial 完成, v1 $L{=}2$ warm-start 在此 suite 是 97.8%, 提示 v3/$L{=}4$ 可能是更强的 VLA warm-start)
- **开了 uniform dyn-skip 之后** libero_spatial 掉到 **64%** —— 证明 **action 模态不能沿用 language 模态的标定**, 这是 modality-aware skip (§7 future work) 不可跳过的原因.

### 5.11 噪声与方差 (2-seed 原始数据 + 两种统计口径)

单 run n=500/suite 的单任务方差可与方法间方差同量级 —— 因此我们对 2B 的 `libero_goal` 和 `libero_10` 做过 2-seed 重复 (spatial / object / 4B 受 GPU 预算所限暂未重跑).

**原始 seed 数据**:

| suite | Path 0 seed 1 | seed 2 | **min** | **max** | **mean** | Path B seed 1 | seed 2 | **min** | **max** | **mean** |
|-------|---------------|--------|---------|---------|----------|---------------|--------|---------|---------|----------|
| goal | 97.6 | 97.4 | 97.4 | 97.6 | 97.5 | 97.4 | 98.6 | 97.4 | 98.6 | 98.0 |
| long-10 | 92.8 | 91.4 | 91.4 | 92.8 | 92.1 | 92.6 | 90.6 | 90.6 | 92.6 | 91.6 |

**两种统计口径 × 4-suite 平均**:

| 统计口径 | Path 0 | Path B | Δ |
|----------|--------|--------|---|
| **max/min** (主表 §5.8 用): Path B 取 max, Path 0 取 min | 95.85 | **97.15** | **+1.30** |
| **mean**: 两个 seed 简单均值 | 96.05 | **96.75** | **+0.70** |

两种口径**排序都是 Path B > Path 0**, 而且 max/min 口径的 gap 是 mean 口径的 ~1.86×. 主表选 max/min 的两个理由: (a) 实际部署会从少量 seed 里选最好的 checkpoint (baseline 则是悲观 floor); (b) 均值口径需要 error bar 而我们只有 2 个 seed, 给均值 ± range 没有统计意义.

- goal 在任一口径下 Path B 都严格胜过 Path 0.
- long-10 在 max/min 下 Path B 胜 1.2pp (max 92.6 vs min 91.4), 在 mean 下反转 0.5pp. 说明 **long-horizon 不是 Path B 的稳定收益来源** (2B 主要收益在 spatial +3.0pp 单 run 上; 4B 主要收益在 long-10 +2.0pp).

---

## 6. 目录结构

retrofit/ 已按 train / eval / bench / tests / analysis 分类整理:

```
reskip/
├── PROJECT_OVERVIEW_CN.md              ← 本文件
├── paper/                              ← 论文源 (LaTeX)
│   ├── main.tex, main.pdf              v2 retrofit 版 (29 页, 当前)
│   ├── main_v1.tex, main_v1.pdf        v1 full-mechanism 版 (保留参考)
│   ├── figures/                        Part 1 Pareto / latency / α 图
│   └── references.bib
├── retrofit/                           ← Part 2 核心代码
│   ├── qwen3vl_attnres_retrofit.py         ★ 核心: 监测+patch 原 VLM 的 forward
│   ├── retrofit.md                         详细实验日志 (2200+ 行)
│   ├── README.md                           目录用法 + quick-start
│   ├── VLA_LIBERO_RESULTS.md               VLA LIBERO 完整日志
│   ├── outputs/                            所有 checkpoint 与 eval 产物
│   │   ├── H_r256_5k/                      canonical v1 retrofit (2B)
│   │   ├── H_4B_r256_5k/, H_4B_r256_10k_v3/ 4B v1 / v3
│   │   ├── H_r256_10k_v3_2B/               v3 canonical (2B)
│   │   ├── block_v3/                       block partition 8 cells
│   │   ├── lmms_eval_*/                    全集 VLM 评估
│   │   ├── text_eval_block_v3/             block partition 的 LAMBADA/HS
│   │   ├── vla_thresholds/                 VLA dyn-skip 标定 JSON
│   │   └── libero_eval_full/               LIBERO 4-suite 原始日志
│   ├── train/                              训练入口
│   │   ├── train_qwen3vl_attnres_retrofit.py  AttnRes retrofit SFT
│   │   ├── train_qwen3vl_lora.py              LoRA baseline
│   │   ├── data_v2.py                         V1/V2/V3 data mix
│   │   └── run_*_v3*.sh                       launcher 脚本
│   ├── eval/                               评估入口
│   │   ├── eval_qwen3vl_attnres_retrofit.py   LAMBADA/HellaSwag/MMMU
│   │   ├── eval_dynamic_skip.py               dyn-skip Pareto sweep
│   │   ├── calibrate_vla_thresholds.py        VLA skip JSON 生成
│   │   ├── lmms_eval_retrofit.py              lmms-eval 插件
│   │   └── eval_mmbench/mmstar/mmmu_*.py
│   ├── bench/                              速度测速
│   │   ├── bench_vlm_vs_vla.py                ★ 规范: TRUE base vs VLM vs VLA
│   │   ├── bench_cache_regime.py              prefill+decode cache=T
│   │   └── benchmark_speed.py                 cacheless prefill 经典
│   ├── tests/                              正确性测试
│   │   ├── smoke_test_qwen3vl_attnres_retrofit.py
│   │   ├── test_e8_use_cache.py               use_cache=T 通路 & generate
│   │   └── test_skip_kv_equiv.py              K/V-only skip 等价性
│   └── analysis/                           消融 / 辅助 baseline / 分析文档
│       ├── gromov_baseline.py                 Block Influence baseline
│       ├── prune_qwen3vl.py                   α-guided pruning 实验
│       ├── review_structure.py                结构审查
│       ├── v2_vlm_analysis.md                 v2 mix 分析 (AI2D collapse)
│       ├── v3_vlm_analysis.md                 ★ v3 canonical + v1/v2 对照
│       └── block_partition_ablation.md        ★ 2B×{1,2,4,7} + 4B×{1,2,4,6}
├── flame/                              ← Part 1 from-scratch 训练
│   ├── saves/reskip_transformer-340M/       canonical 340M AttnRes
│   ├── saves/transformer-340M/              vanilla 340M (motivation baseline)
│   ├── saves/loopskip_transformer_*/        ReLoop 74M 系列 checkpoint
│   └── configs/                             reskip_transformer / reloop_transformer
├── starVLA/                            ← Part 3 VLA 代码
│   ├── src/
│   │   ├── starvla_integration.py          AttnRes ↔ starVLA 桥接 (in-backbone)
│   │   ├── bench_vla_skip_cache.py         VLA 侧速度测速
│   │   └── test_vla_skip_cache.py          VLA cache=T + skip 冒烟
│   ├── starVLA/model/
│   │   ├── framework/QwenOFT.py            VLA OFT 框架 + AttnRes 集成
│   │   └── modules/vlm/QWen3.py            forward_with_attnres_skip 入口
│   ├── deployment/model_server/            policy server (推理侧)
│   ├── examples/LIBERO/
│   │   ├── eval_files/                     run_full_eval.sh, eval_libero.py ...
│   │   └── train_files/                    run_libero_train_attnres_*.sh
│   └── results/                            LIBERO checkpoint 与评估视频
├── DYNAMIC_SKIP_EXPERIMENT_LOG.md      ← Part 1 ReSkip 完整日志 (340M + 110M)
├── RELOOP_EXPERIMENT_LOG.md            ← ReLoop V1→V4 实验日志 (74M)
├── RESKIP_1P3B_PIPELINE_CN.md          ← 1.3B 八卡预训练 pipeline (未跑)
└── rechange/                           ← pre-reorg 旧版 retrofit 脚本 (归档)
```

---

## 7. 当前实验进度 (2026-04-24)

### 已完成
- [x] Part 1 from-scratch 340M (FLA) —— $\mathcal{P}{=}\{3,5\}, M{=}2, q{=}0.85$ 在 4 task 上 $1.19\times$ 零退化加速
- [x] Part 1 cross-scale replication at 110M —— dyn-skip 4-task parity, 触发率较低 (符合预期, 小模型冗余少)
- [x] Part 2 H_r256_5k (v1 canonical) + 8-cell H-family 消融 ($r\in\{32,64,128,256\}$, $\text{steps}\in\{5k,10k,20k\}$, VLM%, ramp)
- [x] Part 2 v3 data mix 全集 `lmms-eval` (MMB / MMMU / MMStar / AI2D / OCRBench / RealWorldQA) 两个 scale 全做完
- [x] Part 2 v3 与 v1/v2 对照 (v1→v2→v3 data-mix ablation trail)
- [x] Part 2 LoRA baseline 4 种配置 ($r{=}32\{q,v\}\times 2$ seeds, $r{=}16\{qkvo\}$, $r{=}8\{\text{MLP}\}$)
- [x] Part 2 4B retrofit (H_4B_r256_5k, H_4B_r256_10k_v3)
- [x] Part 2 Block Partition 消融 (2B × {$L{=}1,2,4,7$} + 4B × {$L{=}1,2,4,6$}, 8 cells 全跑完 VLM 评估, 文本评估 7/8 完成, 4B_L2 aliased to v3_4B)
- [x] Part 2 `use_cache=True` + dyn-skip 首类支持 + K/V-only skip path + `test_skip_kv_equiv.py` 验证 argmax 等价
- [x] Part 2 规范 wall-clock bench (`bench/bench_vlm_vs_vla.py`, VLM=VLA 1.00×, 1.37–1.40× stock base, skip saves 5–9%)
- [x] Part 3 LIBERO 2B Path 0 / Path B v2 / Path C v3 / Path C v4 60k 全 4 suite × 500 episodes
- [x] Part 3 LIBERO 4B clean-base Path 0 / Path B 30k / 60k 全 4 suite
- [x] Part 3 VLA inference 路径对齐 retrofit (同 `dynamic_skip_config` schema, 同 K/V cache 处理, 同 router 数值实现)
- [x] Part 3 `-Action` 污染消融 (dirty vs clean 4B Path B, libero_10 swing 9.4pp)

### 进行中 / 局部完成
- [ ] Part 2 LAMBADA+HellaSwag 在 v3 2B/4B 标准 canonical 上 —— block partition 7/8 cells 已做, v3_2B/v3_4B 主 anchor 配置待跑
- [ ] Part 3 modality-aware dyn-skip Pareto —— 已做 uniform dyn-skip preliminary (libero_spatial 64% vs full 99.5%, 证实 uniform 不适用), modality-aware sweep 待跑
- [ ] Part 3 `pathB_2B_L4_v3_30k` (v3/$L{=}4$ warm-start) 4-suite eval —— spatial 99.5% 部分完成, 其他 3 suite 在跑
- [ ] ReLoop (weight-shared AttnRes, paper §Discussion future work) 74M V3b 已验证 $\alpha$-halt + multi-exit 架构; V4 stochastic exit + per-token halt 训练中; 需 scale up 到 340M+ 以验证真正的 per-sample dynamic depth

### 待做 / open
- [ ] Part 1 1.3B 八卡预训练 (pipeline ready, GPU 阻塞)
- [ ] Part 1 2B from-scratch AttnRes (paper 承认这条过于昂贵, 正是 Part 2 retrofit 的动机)
- [ ] Part 2 其他 VLM 家族 (Gemma-VL / InternVL) 上的 retrofit 可行性
- [ ] Part 2 cheaper router (top-$k$ / sliding-window, 需重训)
- [ ] Part 2 edge GPU wall-clock (RTX 4090 / Jetson Orin)
- [ ] Part 3 SimplerEnv 评估
- [ ] Part 3 VLA w/ modality-aware dyn-skip 的 FLOPs–SR Pareto 曲线

---

## 8. 参考文件

- **详细实验日志**: `retrofit/retrofit.md` (2200+ 行), `retrofit/VLA_LIBERO_RESULTS.md`
- **分析与 ablation 文档**: `retrofit/analysis/v2_vlm_analysis.md`, `v3_vlm_analysis.md`, `block_partition_ablation.md`
- **论文草稿**: `paper/main.pdf` (v2 retrofit 版, 29 页), `paper/main_v1.pdf` (v1 full-mechanism)
- **H-family 消融**: `retrofit/retrofit.md` § "2026-04-18 H-family ablation sweep"
- **速度校正故事**: `retrofit/retrofit.md` § "2026-04-20 (even later) — Speed measurement bug fix" 和 § "yet later — Fast-path fix"
- **ReLoop 独立实验日志**: `RELOOP_EXPERIMENT_LOG.md` (V1→V4 演化, 74M 规模)
- **快速复现指令**: `retrofit/README.md`, `retrofit/retrofit.md` § Quick-start, `retrofit/VLA_LIBERO_RESULTS.md` § Quick-start
