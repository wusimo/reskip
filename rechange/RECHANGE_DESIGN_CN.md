# ReChange 设计主线

## 目标

把一个已经成熟的 VLM，尤其是 `Qwen3-VL`，在**不做 CPT** 的前提下改造成：

1. full-depth 能力尽量无损。
2. 正常前向中真实包含 `AttnRes` 结构。
3. 运行时可以基于 `AttnRes phase-1` 做 `ReSkip`。
4. 最终不是“另一种剪枝器”，而是“retrofit 版 AttnRes 模型”。

---

## 核心判断

之前如果把路线定义成：

> 原模型不动，只外挂一个 surrogate/observer 给 skip 用

那工程上可能能用，但论文主线会变弱，因为：

- `AttnRes` 没有真正进入正常 full-depth 前向
- `ReSkip` 更像外挂的近似替代或剪枝策略
- 很难说模型已经变成了 `AttnRes-capable`

所以 `rechange` 主线必须改成更强的定义：

> `AttnRes` 支路真实进入正常前向；skip 只是同一条 AttnRes 支路的运行时特例。

---

## 新结构

对第 `n` 个 block，定义：

```text
r_n = AttnRes_n(h_0, ..., h_{n-1})
x_n = h_{n-1} + gamma_n * Adapter_n(r_n - h_{n-1})
h_n = Block_n(x_n)
```

其中：

- `h_0` 是 embedding 或 multimodal merged hidden state
- `r_n` 是标准 block-level AttnRes routed state
- `Adapter_n` 是很轻的校正头
- `gamma_n` 是可学习标量，初始化为 `0`

这意味着：

- 初始化时 `x_n = h_{n-1}`，模型严格退化回原模型
- 训练后 `AttnRes` 会以安全、渐进的方式进入正常前向
- 这是真正的 retrofit，不是外接旁路

---

## Skip 如何定义

对于可跳 block，运行时不再另做一套模拟器，而是直接复用上面的 `x_n`：

```text
if safe_to_skip(n):
    h_n = x_n
else:
    h_n = Block_n(x_n)
```

所以：

- full-depth 模式：`AttnRes` 真实参与 normal forward
- skip 模式：同一条 `AttnRes` 支路承担 block 替代
- skip 信号：仍来自 `AttnRes phase-1 alpha`

这才保住了 `ReSkip` 和 `AttnRes` 的内在关系。

---

## 为什么这比旧 Route A 更合理

旧 Route A 的问题是把主路径写成：

```text
(1 - beta) * prev + beta * routed
```

当 `beta` 被推高时，本质上是在强迫整个成熟 backbone 重新围绕 AttnRes 共适应。  
这在无 CPT、单次轻量 fine-tune 条件下风险很高，容易出现：

- benchmark 回退
- 单一分布过拟合
- alpha 退化成 copy-prev / copy-embed

新的 `zero-init residual AttnRes injection` 则不同：

- 初始化严格安全
- AttnRes 进入正常前向，但只作为小修正开始
- 是否进一步发挥作用，由数据和训练自动决定
- skip 时复用的也是同一条分支，不会和 full-depth 机制脱节

---

## 第一阶段只做什么

`rechange` 第一阶段只做：

1. 只改 `Qwen3-VL language_model`
2. 不动 vision tower
3. 不动 multimodal merge / mRoPE / deepstack 主逻辑
4. 只开放后半段 block 为 `AttnRes-injected + skippable`

推荐默认：

- Qwen3-VL 28 layers
- 14 blocks，每块 2 层
- 前 8 blocks 纯原始路径
- 后 6 blocks 启用 AttnRes injection，并可跳

这样做的目的不是保守，而是避免早层表示构造被破坏。

---

## 训练目标

训练目标不再是“把 beta 拉高”，而是：

> 让 full-depth 保持原模型能力，同时让 `x_n` 在 block 被跳过时尽量可用。

因此最合理的训练方式是：

### 1. Full-path task loss

正常 full-depth 前向计算：

```text
L_task
```

保证模型主能力不丢。

### 2. Teacher / full-path consistency

对被采样为 skip-candidate 的 block，构造 skip branch，并对齐：

```text
L_kl = KL(logits_skip || logits_teacher)
```

### 3. Local hidden regression

对齐 block 输出：

```text
L_hidden = || x_n - h_n^teacher ||^2
```

或者更保守地，对齐 skip 后若干层输出。

### 4. Light routing regularization

只保留很弱的熵约束，避免完全平均化即可。

---

## 参数开放策略

第一阶段优先只训练：

- pseudo-queries
- positional bias
- adapters
- gamma

第二阶段再考虑打开：

- 后半段 block 的 LoRA
- 或后半段 norm

不建议一上来全量放开 language backbone。

---

## 数据策略

既然不做 CPT，就必须承认：  
retrofit 训练本质上是 post-training，而不是重新预训练。

但这不代表可以只用单一 SFT 数据。

建议的起点是 **post-training mixture**：

- text-only 指令数据
- reasoning / code 指令数据
- multimodal instruction 数据

关键原则：

- 不能只用 UltraChat
- 不能只用 LLaVA
- 必须让 full-depth 仍然覆盖通用语言与多模态能力

---

## 实验路线

### E0: Exact-wrap smoke test

验证：

- `gamma = 0`
- full-depth 输出与原模型一致或近似一致

### E1: AttnRes-in-forward activation

验证：

- 注入后 full-depth benchmark 基本不掉
- alpha 不再退化为完全无意义的均匀分布

### E2: Single-block safe skip

验证：

- 对单个 late block，用 `x_n` 直接替代 `Block_n(x_n)`
- benchmark 基本不掉
- 有真实 wall-clock 收益

### E3: Dynamic ReSkip

验证：

- phase-1 alpha 能否提供稳定的 pre-execution skip 信号
- 多 block runtime skip 是否可行

---

## `rechange` 目录第一批文件

第一批只做最必要的 4 个文件：

- `qwen3vl_attnres_retrofit.py`
- `smoke_test_qwen3vl_attnres_retrofit.py`
- `train_qwen3vl_attnres_retrofit.py`
- `eval_qwen3vl_attnres_retrofit.py`

本轮先把模型 wrapper 和 smoke test 落下来，后续再接训练脚本。
