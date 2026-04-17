# AttnRes Retrofit Paper Plan

**Status**: Draft v1 (2026-04-17)
**Target venue**: ICLR 2027 (deadline ~2026-09)
**Total timeline**: 16 weeks
**Compute budget**: 150-400 GPU-days (8×H100)

---

## 0. 核心战略转向

### 原方案的 gap
- ReLoop 在 74M/74B tokens scale 上 **完全无法匹配 Standard AttnRes**（lambada ppl 579 vs 83，差 7×）
- Weight-sharing 的收益在小模型上被 expressive-capacity 损失抵消
- 从头 pretrain 2-4B AttnRes 模型算力勉强够，但和 MoD/CALM/LayerSkip 正面竞争胜算不大

### 新方向的优势
把 **ReLoop 作为 future work**，paper 主线改为：

> **"把现有 pretrained transformer 通过轻量 fine-tune 改造为 AttnRes-capable，获得 ReSkip 能力"**

三大优势：
1. **算力门槛显著降低**：不用从头 pretrain 大模型；fine-tune 1-5B tokens 即可
2. **VLA 难题自然解决**：直接 retrofit SOTA VLM（Qwen2-VL、LLaVA）→ 不回避现有工作
3. **Practical impact 强**：方法能应用到任何 off-the-shelf transformer，reviewer 加分

### Paper 新定位
- **Title（草拟）**: "Retrofitting Pretrained Transformers for Zero-Cost Adaptive Computation via Attention Residuals"
- **Core contribution**:
  1. **Insight**: AttnRes routing weights = free skip signal（继承自 Fu et al. 2025）
  2. **Technical novelty**: Retrofit recipe，把 pretrained 模型通过 light fine-tune 转成 AttnRes-capable
  3. **VLA application**: Retrofit SOTA VLM → modality-aware skip，提升 VLA inference 效率

---

## 1. 三阶段总览

| Phase | 时间 | 目标 | 核心产出 |
|-------|------|------|----------|
| Phase 1 | Week 1-3 | LM from-scratch 小规模验证 | AttnRes + ReSkip 在 110M-2B 的 Pareto 曲线 |
| **Phase 2** | **Week 4-9** | **Retrofit 方法（核心创新）** | **3 种 retrofit 方案对比 + Llama-2-7B 可 retrofit 的证明** |
| Phase 3 | Week 10-12 | VLA 应用 | Qwen2-VL retrofit → modality-aware skip 数字 |
| Writing | Week 13-16 | Paper 成型 + polish | ICLR submission |

---

## 2. Phase 1：LM from-scratch 小规模验证

### 目标
证明 **AttnRes + ReSkip 机制本身 work**，建立 scaling 趋势。这是 Phase 2 的"基础 validation"，不求压倒性 baseline。

### 模型 scale 梯度

| Model | Params | Tokens | Chinchilla? | 算力估算（8×H100）|
|-------|--------|--------|-------------|-------------------|
| AttnRes-110M | 110M | 10B | ~5× 过训 | 0.5 GPU-days |
| AttnRes-340M | 340M | 30B | ~4× 过训 | 2 GPU-days |
| AttnRes-1B | 1B | 100B | ~5× 过训 | 15 GPU-days |
| AttnRes-2B | 2B | 200B | ~5× 过训 | 40 GPU-days |

**共计 ~60 GPU-days**（Phase 1 轻量）

### 对比 baseline（最少）
- **Plain Transformer**：同 scale、同 tokens → 证明 AttnRes 带来质量提升
- **Static pruning (Gromov et al.)**：从 AttnRes 训好后 post-hoc 裁层 → 证明 input-dependent > static

**不做的对比**（留到 Phase 2）：CALM、MoD、LayerSkip

### 数据
- **主语料**：SlimPajama-627B 或 FineWeb-Edu-350B（选其一）
- **Tokenizer**：LLaMA-2 tokenizer（32k vocab）或 GPT-NeoX tokenizer
- **Seq len**：2048（context_len）

### 评测
- Wikitext-103 / C4 validation PPL
- lm-eval-harness: lambada, hellaswag, arc_easy, arc_challenge, openbookqa, mmlu（if ≥1B）

### 关键产出
- Table: AttnRes vs Plain Transformer across scales
- Figure: ReSkip Pareto curves（PPL vs FLOPs）at 4 scales
- Figure: Scaling trend of optimal skip ratio（**novel finding**：skip ratio 和 scale 的关系）

---

## 3. Phase 2：Retrofit 方法（paper 核心创新）

### 问题定义

**Standard residual**：
$$x_l = x_{l-1} + f_l(x_{l-1})$$

**AttnRes**：
$$\alpha_{i \to l} = \text{softmax}\left(\frac{w_l^\top k_i}{\sqrt{d}}\right), \quad x_l = \sum_{i=0}^{l-1} \alpha_{i \to l} h_i$$

**Retrofit 目标**：
- 从 pretrained model 出发（标准 residual）
- 通过添加 pseudo-query $w_l$ 和少量 fine-tune
- 得到 AttnRes-capable 模型
- **关键约束**：quality 不能明显下降（相对原模型 ≤ 5%）
- **期望性质**：routing weights 有意义稀疏，支持 ReSkip

### 核心难点
Pseudo-query $w_l$ 是单个向量，无法对所有输入都让 $\alpha_{l-1 \to l} = 1$（即复现 standard residual）。naive retrofit → 初始化随机、routing 混乱、quality 崩。

### 三条技术路线（paper 核心对比）

#### Route A: 插值型（Interpolation Gate）

**Forward**：
$$x_l = (1 - \beta_l) \cdot \underbrace{[x_{l-1} + f_l(x_{l-1})]}_{\text{standard}} + \beta_l \cdot \underbrace{\sum_i \alpha_{i \to l} h_i}_{\text{AttnRes}}$$

- $\beta_l$ 是 learnable scalar，初始 $\beta_l = 0$（完全 standard）
- 训练中 $\beta_l$ 可以自由调整
- $w_l$ 随机初始化（small normal），让 $\alpha$ 初始近似均匀

**优点**：
- 初始时严格等同原模型，不会 catastrophic fail
- 训练 stable

**缺点**：
- 最终模型是"standard + AttnRes 加权和"，不是纯 AttnRes
- Skip decision 基于 $\alpha$，但 forward 还依赖 standard path
- Inference 稍微复杂：还要维护 standard path

#### Route B: 辅助观察者（Auxiliary Observer + Distillation）

**Forward**：保持不变（standard residual）

**额外训练**：
$$\mathcal{L}_{\text{aux}} = \|\text{AttnRes}(h_0, ..., h_{l-1}; w_l) - x_l\|^2$$

- AttnRes 只作为"观察器"，训练 $w_l$ 去重建 standard 输出
- $\alpha$ 反映"standard model 里各层的实际贡献"
- Skip decision 用学到的 $\alpha$

**优点**：
- 原模型不受干扰，quality 严格保留
- Fine-tune 只训 pseudo-queries，参数量极小 → 可在很大模型上快速 retrofit
- 概念清晰：AttnRes 是"事后分析器"

**缺点**：
- $\alpha$ 质量取决于重建 loss 能学出的信号强度
- 没有闭环反馈（skip 决策不影响训练 loss）
- 可能需要额外 calibration step

#### Route C: Temperature Annealing + Informed Init

**Forward**：直接替换为 AttnRes

**关键：informed initialization**
1. 在 pretrained model 上跑 calibration set，收集每层 $k_i = W_K h_i$ 的平均
2. 初始化 $w_l = \bar{k}_{l-1}$（与上一层的平均 key 对齐）
3. 这样 $\alpha_{l-1 \to l}$ 在 calibration set 上平均接近 1
4. 训练用 temperature schedule：开始 $\tau$ 很小（锐利，保持初始行为），逐渐升到 1

**优点**：
- 最终模型是 pure AttnRes，机制干净
- Paper 故事最清晰

**缺点**：
- Informed init 依赖数据统计，可能在 OOD 上崩
- Temperature schedule 是超参 nightmare
- 最易崩的路线

#### Route D（可选）：LoRA + AttnRes 补充

**Forward**：
- 在 $f_l$ 上加 LoRA adapter
- 同时训 AttnRes pseudo-query
- 用 LoRA 让模型适应 AttnRes routing 的变化

**优点**：参数效率高
**缺点**：和 LoRA 的界限不清楚，可能被 reviewer 质疑"只是 LoRA"

*注：Route D 作为备选，如果 A/B/C 有问题再考虑*

### 实验设计：选出最佳 retrofit 方案

**Target 模型**（初步选择）：
- **Llama-2-7B**（广泛使用，参数量合适，有 open-source baselines）
- 或等你下载的 **2B VLM** 直接作为 target（取决于是哪个模型）

**对每个 Route (A, B, C)**：

1. **Fine-tune quality preservation**：
   - Fine-tune 1B / 5B / 10B tokens
   - 每个 checkpoint 测 lm-eval-harness 全套
   - 画 "quality vs fine-tune tokens" 曲线
   - **验收**：5B tokens 时 lm-eval 平均跌幅 < 2%

2. **Skip Pareto curve**：
   - 在 retrofit 后的模型上跑 threshold sweep
   - 画 "quality vs FLOPs" 曲线
   - **验收**：在某 FLOPs 区间能跌幅 < 5% 且 FLOPs 节省 > 20%

3. **Route 之间对比**：
   - 同一 target 模型、同 fine-tune 预算下
   - 哪个 route 的 Pareto frontier 最好
   - 结合"实现复杂度"、"inference 速度"决定 winner

**Ablation study**：
- Fine-tune 数据量：500M / 2B / 5B / 10B tokens
- Fine-tune 数据类型：原 pretrain 同分布 vs. shift
- Layer granularity：per-layer vs per-block AttnRes

### 对比外部 baseline
- **LayerSkip (ICML 2024)**：他们是"continued pretraining"做 skip，我们是"轻量 retrofit"——直接对比 Llama-2-7B 上的 Pareto
- **Static pruning (Gromov et al.)**：同一 target 模型 post-hoc 裁层
- **SparseGPT / Wanda** (structured)：公平对比就要加

**Phase 2 算力预估**：
- Retrofit fine-tune：1-10B tokens × 7B 模型 ≈ 20-100 GPU-days/run
- 3 个 route × 2-3 target 模型 × 3 fine-tune 预算 ≈ 10-20 runs
- **总计 100-300 GPU-days**

---

## 4. Phase 3：VLA 应用

### 目标
把 retrofit 方法应用到真实 SOTA VLM，证明 **modality-aware skip** 的实用价值。

### 架构选择

**推荐 target**（用户将提供 2B VLM）：
- **Qwen2-VL-2B** 或类似规模的 open VLM
- 或 LLaVA-1.5-7B / OpenFlamingo 系列

**流程**：
1. Retrofit VLM（用 Phase 2 选出的最佳 route）→ AttnRes-VLM
2. 加 action head（flow matching，参考 Pi0）
3. Fine-tune on LIBERO-90 + OpenX-Embodiment subset
4. 分析 modality-specific routing
5. Apply modality-aware skip

### 关键实验

**实验 1：Modality-specific routing 分析**
- Vision tokens、language tokens、action tokens 各自的 $\alpha$ 分布
- 画 heatmap：某 layer 对 vision token 的 α_{i→l} vs action token
- 预期：**vision 浅、action 深**

**实验 2：Modality-aware vs uniform skip**
- 在 LIBERO-90 上跑：
  - No skip（baseline）
  - Uniform skip（所有 token 同 threshold）
  - **Modality-aware skip**（我们的方法）
- 画 "success rate vs FLOPs" 的 Pareto

**实验 3：对比 SOTA VLA**
- 和 OpenVLA-7B、Pi0 比 success rate（不一定赢，但要 comparable）
- 主卖点：**我们的 inference FLOPs 更低**

### Benchmark
- **LIBERO-90**（全任务覆盖）
- **LIBERO-Long**（长程任务）
- 可选：**SimplerEnv**（Google Robot 模拟）

### 关键数字
> "在保持 success rate ≥ OpenVLA 水平的前提下，modality-aware skip 比 uniform skip 多省 X% FLOPs"

X 是 paper VLA 章节的卖点。X > 15% 是强故事。

**Phase 3 算力预估**：
- VLM retrofit: ~20 GPU-days
- VLA fine-tune: ~20 GPU-days
- Ablation: ~10 GPU-days
- **总计 50 GPU-days**

---

## 5. 全局算力和时间

### 算力汇总

| Phase | 算力（GPU-days on 8×H100）| 时间 |
|-------|------------------------|------|
| Phase 1（LM scales）| 60 | Week 1-3 |
| Phase 2（Retrofit）| 100-300 | Week 4-9 |
| Phase 3（VLA）| 50 | Week 10-12 |
| **总计** | **~210-410** | **12 周** |
| Writing | — | Week 13-16 |

即 **26-52 GPU-月** 的总算力。按你给的 "2-4B 模型，200-400B tokens" 的预算完全够。

### Gantt 精确版

```
Week:  1    2    3    4    5    6    7    8    9    10   11   12   13   14   15   16
─────────────────────────────────────────────────────────────────────────────────────
Infra: ██   ██
110M:    ██
340M:       ██
1B:              ███
2B:                   ████████
Route A retrofit:          ██████
Route B retrofit:              ██████
Route C retrofit:                  ██████
Ablation:                              ████
VLA retrofit:                              ████
VLA train:                                     ████
VLA eval:                                          ██
Writing:                                              ███████████████
```

### 关键 milestone

- **Week 3 末**：Phase 1 完结，AttnRes + ReSkip 在 2B 上 Pareto 曲线干净
- **Week 6 末**：Route A、B 至少一个成功 retrofit Llama-2-7B
- **Week 9 末**：Retrofit 最佳 route 确定，Pareto 对 LayerSkip 有优势
- **Week 12 末**：VLA modality-aware skip 数据出炉
- **Week 16**：Paper draft 完成，准备投 ICLR 2027

---

## 6. 风险和对策

| 风险 | 概率 | 影响 | 对策 |
|------|------|------|------|
| Retrofit 三条路线都不 work | 低 | 致命 | 提前在 110M 小模型上验证路线（Week 2） |
| Quality 掉太多（>5%） | 中 | 大 | 多种 fine-tune 数据源 / 更长 fine-tune |
| Routing 不稀疏（没有 skip 空间） | 中 | 大 | 加稀疏正则（L1 on $\alpha$），或更大 threshold range |
| VLA success rate 完全输给 OpenVLA | 高 | 中 | 叙事聚焦"skip 效率"而不是"绝对 SR" |
| 算力不够 | 中 | 中 | 可以裁到 110M-1B scale，或缩短 fine-tune 预算 |
| NeurIPS 被拒 | 中 | 小 | 备投 ICLR（deadline 更晚） |

---

## 7. Paper 预期结构

```
Title: Retrofitting Pretrained Transformers for Zero-Cost Adaptive Computation
       via Attention Residuals

Abstract
  - AttnRes routing as free skip signal
  - Retrofit method: 轻量 fine-tune 转换 pretrained → AttnRes-capable
  - VLA application: modality-aware skip

1. Introduction
   1.1 Adaptive computation 的 motivation
   1.2 现有方法的局限（都要 from-scratch 或重训）
   1.3 我们的 insight + contribution

2. Background: Attention Residuals
   2.1 Standard residual
   2.2 AttnRes mechanism (citing Fu et al. 2025)

3. ReSkip: AttnRes-Guided Layer Skipping
   3.1 Block importance definition
   3.2 Online softmax merge with skip
   3.3 Calibration-based vs dynamic thresholds

4. Retrofit Method
   4.1 Problem statement
   4.2 Route A: Interpolation
   4.3 Route B: Auxiliary observer + distillation
   4.4 Route C: Temperature annealing
   4.5 Empirical comparison (key figure)

5. LM Experiments
   5.1 From-scratch AttnRes across scales (110M-2B)
   5.2 Retrofit Llama-2-7B: quality preservation + skip Pareto
   5.3 Comparison: LayerSkip, static pruning, CALM (if time)
   5.4 Scaling trends

6. VLA Experiments
   6.1 Retrofit Qwen2-VL-2B
   6.2 Modality-specific routing analysis
   6.3 Modality-aware skip on LIBERO-90

7. Related Work

8. Discussion
   - Limitations
   - Future work: ReLoop extension, larger scales

Appendix
   - Implementation details of each route
   - Ablation studies
   - Additional routing visualizations
```

---

## 8. 下一步具体行动

### 用户侧（等你下载 2B VLM 后）
1. 把 VLM 路径告诉我
2. 决定 retrofit target：是 LM 用 Llama-2-7B 还是直接用你的 2B VLM？
3. 确认 ICLR 2027 作为主 deadline

### 开始就能做（不等 VLM）

**Week 1 任务**：
1. **[高优先]** Phase 1 infrastructure：多卡 SlimPajama streaming pipeline 跑通
2. **[高优先]** 训练 110M AttnRes 作为 end-to-end 管线验证（~12 小时）
3. **[中优先]** 在 110M 上快速测试 Route A/B/C 初始化是否 stable（不用长训，看 loss 是否爆炸）
4. **[中优先]** 用户 review 本文档，确定整体方向

**Week 1 不做**（避免分散精力）：
- 不要继续调 V4/V5 ReLoop
- 不要跑 VLA 相关（等 Phase 3）
- 不要写 paper（等实验数据）

---

## 9. ReLoop 的处理

**明确位置**：**Future Work**（Section 8）中一段话带过。

```
Future work. Our retrofit framework naturally extends to weight-shared 
architectures (ReLoop): the position-specific AttnRes pseudo-queries 
differentiate each application of a shared block. We leave the 
investigation of retrofit-enabled weight sharing, which is complementary 
to the skip mechanism studied here, to future work.
```

**保留的技术遗产**：
- `reloop_transformer` 模型代码保留在 repo，不删
- V4 checkpoint 保留，将来可能用
- ReLoop 章节的 tex 注释掉但不删，将来续作可用

---

## 10. 成功判定

**Paper 能投 ICLR 2027 的条件**：
- [ ] Phase 1：至少 2 个 scale 的 AttnRes + ReSkip Pareto 曲线（340M + 1B 或 2B）
- [ ] Phase 2：至少 2 个 route 成功 retrofit 了 Llama-2-7B（或等价大模型），quality preservation < 5%
- [ ] Phase 2：retrofit 后的 skip Pareto 优于 static pruning
- [ ] Phase 3：modality-aware skip 在 LIBERO 上比 uniform skip 节省 FLOPs ≥ 15%
- [ ] Paper 写作完成，figures polish

**ICLR 2027 accept 的关键因素**：
- Retrofit method 的 novelty（我们是第一个这么做 AttnRes retrofit 的，应该没有正面竞争）
- 大模型上验证（Llama-2-7B、Qwen2-VL-2B 都是业界标准）
- 实际的 VLA efficiency 数字

---

## Appendix A：关键技术细节

### A.1 Block-level vs Layer-level AttnRes
- Layer-level：每层独立 pseudo-query，routing 更细但 KV cache 更大
- Block-level：把多层打包，routing 粗但效率高
- Retrofit 推荐 block-level（减小改动量）

### A.2 Pseudo-query 初始化
- 默认：$\mathcal{N}(0, 0.02)$ small normal
- Route C 专用：$w_l = \bar{k}_{l-1}$（calibration average）

### A.3 Online softmax merge with skip
算法保持和原 paper 一致，skip 的 block 直接跳过更新。

### A.4 Skip threshold 选择
- Calibration-based：在 calibration set 上收集 $I(n) = \max_{l > n} \alpha_{n \to l}$ 的分布
- 选 threshold 使 skip ratio 达到目标（20%、40%、60%）
- 或按 block importance 由低到高排序，逐个 skip

---

**文档版本历史**
- v1 (2026-04-17): Initial draft after strategic pivot away from ReLoop
