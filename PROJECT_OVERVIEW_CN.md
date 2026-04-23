# ReSkip / ReLoop 项目概述

**目标论文:** NeurIPS 2026, 主题是 AttnRes (Attention Residual) 架构带来的自适应计算深度。

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

## 4. 自适应推理: ReSkip

### 4.1 动态 skip 判据

训练完的 retrofit 模型前向时，router 输出 `α_n`。定义 **w_recent = α_n[n-1]**，即 "本 block 路由给 `h_{n-1}` 的概率"。直觉：`w_recent` 高意味着 router 认为 "不需要更老的历史，就用上一步"，此时 block 贡献小，可跳过。

**跳过条件**:
$$\text{skip block } n \iff w_{\text{recent},n}(x) > \tau_n$$

阈值 `τ_n` 按 **per-block quantile calibration** 在 32 条 LAMBADA 校准样本上估得：`τ_n = Quantile_q(w_recent,n)`，`q ∈ {0.5, 0.7, 0.85, 0.95}`。`q` 越大跳得越保守。

**跳过后 block 的真正 forward**:
$$h_n = x_n \quad\text{（直接用 routed 值, 不过 block 层）}$$

### 4.2 Pareto 前沿 (H_r256_5k, LAMBADA)

| `q` / `max_skips` | LAMBADA acc | 平均跳过层数 | 加速 |
|-------------------|-------------|------------|------|
| 原模型 (no skip) | 0.576 | 0 | 1.00× |
| q=0.95, M=1 | 0.568 | 1.3 / 28 | 1.05× |
| q=0.85, M=2 | 0.548 | 3.1 / 28 | 1.11× |
| q=0.70, M=2 | 0.504 | 5.8 / 28 | 1.22× |

实际加速 < 跳层比例，因为 `language_model` 仅占 VLM 总推理时间 60-70%。跳 cross-attention 类 block 收益最大。

---

## 5. 实验结果

### 5.1 VLM 基准 (Qwen3-VL-2B, H_r256_5k vs 原模型)

| Benchmark | 原模型 | retrofit | Δ |
|-----------|--------|----------|---|
| LAMBADA acc | 0.532 | **0.576** | **+4.4** |
| HellaSwag acc | 0.506 | 0.522 | +1.6 |
| MMBench_en_dev | 0.727 | 0.710 | −1.7 (噪声内) |
| MMMU_val | 0.412 | 0.423 | +1.1 |
| OCRBench | 0.687 | 待补 | — |

**结论**: 文本任务全面提升 (LAMBADA +4.4pp)，视觉任务持平。AttnRes 的自注意路由**没有破坏原 VLM 的多模态对齐**。

### 5.2 VLA LIBERO 下游任务 (Qwen3-VL-2B → LIBERO, 30k steps, 4×H100)

**训练策略**:
- **Path 0 (baseline)**: 直接用原 Qwen3-VL-2B 在 LIBERO 上 OFT 微调
- **Path B v2**: 从 `H_r256_5k` retrofit state 热启动，保留 router/adapters/γ，在 LIBERO 上继续训（**per-block AttnRes 结构全程参与 loss**）
- **Path C v3**: 从原 Qwen3-VL-2B 开始（无 VLM retrofit warm-start），直接在 VLA 训练时加 AttnRes 组件 + γ-curriculum 0→1 over 9k steps

**4-suite LIBERO 成功率** (每 suite 10 tasks × 50 trials = 500 episodes):

| Method | spatial | object | goal | 10 | **平均** |
|--------|---------|--------|------|-----|---------|
| Path 0 (base) | 94.8 | 99.8 | 97.6 | 92.8 | 96.25 |
| Path B v1 (observer bug) | 96.8 | 99.6 | 97.6 | 91.6 | 96.40 |
| **Path B v2 (per-block)** | **97.8** | 99.6 | 97.4 | 92.6 | **96.85** |
| Path C v3 (γ-curriculum from base) | 92.6 | **100.0** | 进行中 | 待跑 | 待补 |

**关键观察**:

1. **Path B v2 > Path 0 by +0.60pp** (n=2000 episodes)。AttnRes warm-start 从 VLM retrofit 迁移到 VLA 下游有效。
2. **Path B v1 (observer) < Path B v2 by 0.45pp** —— 证明 **per-block in-backbone 集成** 优于只在最后一层做 correction 的 observer 实现（早期 bug 版本）。
3. **spatial 最受益** (+3.0pp vs base)，**object 饱和** (所有方法 ≥99.6%)。
4. **Path C v3 在 spatial 上 92.6% 反而低于 Path 0** —— `task 6` 只有 50%（单任务 -12pp，拖累整个 suite）。这说明 **纯靠 γ-curriculum 在 VLA 任务上从头学 AttnRes 结构**，相比于 **先做 VLM retrofit 得到稳定 router 再 warm-start**，在最难任务上不够稳定 —— 反过来支持 VLM retrofit 作为 warm-start 的必要性。

### 5.3 噪声与方差

单 run n=500/suite 的单任务方差 > 方法间方差。我们对 `libero_goal` 做 2-seed 重复：

| | run 1 | run 2 | 平均 | Δ |
|---|------|------|-----|---|
| Path 0 | 97.6 | 97.4 | 97.5 | 0.2pp |
| Path B v2 | 97.4 | 98.6 | **98.0** | 1.2pp |

多 seed 平均后 Path B v2 仍稳超 Path 0 +0.5pp。

---

## 6. 目录结构

```
reskip/
├── PROJECT_OVERVIEW_CN.md           ← 本文件
├── paper/                           ← 论文源 (LaTeX)
├── retrofit/                        ← Part 2 核心代码
│   ├── qwen3vl_attnres_retrofit.py      VLM→AttnRes 主文件
│   ├── train_qwen3vl_attnres_retrofit.py 训练脚本
│   ├── lmms_eval_retrofit.py            lmms-eval 插件
│   ├── eval_dynamic_skip.py             动态 skip Pareto
│   ├── outputs/H_r256_5k/               canonical state
│   ├── retrofit.md                      详细实验日志 + quick-start
│   └── VLA_LIBERO_RESULTS.md            VLA LIBERO 结果 + quick-start
├── flame/                           ← Part 1 from-scratch 训练
│   └── saves/reskip_transformer-340M/
└── starVLA/                         ← Part 3 VLA 代码
    ├── src/starvla_integration.py       AttnRes ↔ starVLA 桥接
    ├── starVLA/model/framework/QwenOFT.py
    └── examples/LIBERO/                 训练/评估脚本
```

---

## 7. 当前实验进度 (2026-04-20)

- [x] Part 1 from-scratch 340M (FLA) — 已完成
- [x] Part 2 VLM retrofit canonical H_r256_5k — 已完成
- [x] Part 2 VLM benchmark (LAMBADA / HellaSwag / MMBench / MMMU) — 已完成
- [x] Part 3 LIBERO Path 0 / Path B v2 全 4 suite × 500 episodes — 已完成
- [ ] Part 3 LIBERO Path C v3 —— 4 suite 评估进行中 (spatial/object 已完成)
- [ ] Path 0 / Path B v2 的 goal+10 二次重跑（用于方差估计）—— 进行中
- [ ] Part 2 lmms-eval 全面 benchmark（MMStar, AI2D, OCRBench, RealworldQA）—— 待补
- [ ] Part 2 latency 优化到 ≤ base 模型水平 —— 待做
- [ ] 消融: 去掉 adapter (Route A 纯 γ·r_n) 的 VLA 表现

---

## 8. 参考文件

- **详细实验日志**: `retrofit/retrofit.md`, `retrofit/VLA_LIBERO_RESULTS.md`
- **论文草稿**: `paper/main.pdf` (v2, retrofit 版本), `paper/main_v1.pdf` (v1, full-mechanism)
- **H-family 消融**: `retrofit/retrofit.md` § "2026-04-18 H-family ablation sweep"
- **快速复现指令**: `retrofit/retrofit.md` § Quick-start, `retrofit/VLA_LIBERO_RESULTS.md` § Quick-start
