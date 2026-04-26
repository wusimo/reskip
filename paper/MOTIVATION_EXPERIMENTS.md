动机：主要是要找到一个切入点，说明为什么我们要做这个reskip。我现在想的是：AttnRes 在 llm 领域有一定效果，但及时使用 block 方式，依旧会在小参数模型上产生较大的延迟、速度影响（咱们的 part 1部分实验可以说明），这对 vla 领域是十分敏感的。同时从 0 训练一个 AttnRes 的 vlm 或者 vla 成本巨大。
这个动机部分的问题是需要一章内容以及一些实验来证明的，证明确实有这些问题，而不是解决。解决的话是下一章我们的方法里来介绍，实验对比那就是更下一章了，这里只用提现问题的存在和解决必要性。

我们的工作：那么有没有什么好的方法让现有模型具有 AttnRes的特性，性能提升的同时速度几乎不下降甚至加快呢？
# Motivation-Proving Experiments

Goal: produce empirical evidence for the paper's opening claim — **"AttnRes is
valuable but has practical latency issues, especially for VLAs; training it
from scratch is prohibitive; a cheap retrofit path is needed."**

Every bullet in the introduction should point to a number in one of the tables
below.

---

## Claim 1: AttnRes has real latency overhead vs a matched vanilla transformer

**What we need to show**: on the exact same scale and the same training data,
the AttnRes-pretrained transformer is measurably slower than a vanilla-residual
transformer — even though we use the block-level variant. Readers should see
the overhead in milliseconds.

### Experiment A1 — 340M latency table (reuse Part-1 assets)

We already have, from Part 1:
- `reskip_transformer-340M` — AttnRes from-scratch, FineWeb-Edu 100BT
- `base_transformer-340M` — standard-residual transformer, same data (the one
  whose 0.3790 LAMBADA acc appears in `DYNAMIC_SKIP_EXPERIMENT_LOG.md`)

Measure forward-pass wall-clock on 1×H100, bfloat16, batch = 1, random-token
input, 5 warmup + 20 timed passes:

| seq_len | vanilla 340M | AttnRes 340M full-depth | AttnRes 340M + dyn-skip | Δ full vs vanilla | Δ skip vs vanilla |
|---------|--------------|-------------------------|-------------------------|-------------------|-------------------|
| 512     | ms           | ms                      | ms                      |  +x.x %           |   ±x.x %          |
| 1024    | ms           | ms                      | ms                      |                   |                    |
| 2048    | ms           | ms                      | ms                      |                   |                    |
| 8192    | ms           | ms                      | ms                      |                   |                    |

**Expected pattern**: AttnRes full > vanilla (net overhead). Dyn-skip partially
closes the gap — at long seq, probably closes it; at short seq (VLA-relevant),
probably not.

### Experiment A2 — per-block overhead breakdown (same 340M)

On seq 2048, profile one forward and report:
- Router (phase-1 attention over completed blocks) time per block
- Attention sub-block time
- MLP sub-block time
- Total per-block
- % of per-block time that is "new AttnRes machinery"

Deliverable: stacked-bar figure, one bar per block, or a summary table with
aggregate mean / max.

### Experiment A3 — absolute vs vanilla, per-seq-len plot

Line plot: latency (ms) vs seq_len for {vanilla-340M, AttnRes-340M, AttnRes-340M + skip}.
Read from the Part-1 paper that skip gives 1.19× over AttnRes-full; we want to
show the same two lines plus a third reference curve (vanilla) and the
shaded "AttnRes overhead" region between them.

---

## Claim 2: the overhead matters at VLA scale (2B-class, latency-critical)

### Experiment B1 — Qwen3-VL-2B reference latency at VLA-relevant seq lens

We already measured on 1×H100, bfloat16, batch 1. Expand the table we already
have to include additional seq lens that map to VLA use:

| seq_len | what it represents | Base Qwen3-VL-2B | retrofit full-path | retrofit + dyn-skip |
|---------|---------------------|------------------|---------------------|----------------------|
| 256     | short language instruction only | ms | ms | ms |
| 512     | instruction + 1 image (tokenised) | ms | ms | ms |
| 1024    | instruction + image + short context | ms | ms | ms |
| 2048    | multi-turn w/ images | (have) | (have) | (have) |
| 4096    | long-horizon trajectory context | (have) | (have) | (have) |

We already have 1024 / 2048 / 4096 from previous speed benchmark. Need to add
256 and 512.

### Data point to cite from literature (not an experiment)

Put this in the related-work / introduction citation:
- pi0~\citep{black2024pi0}: 50 Hz control → 20 ms per inference cycle
- OpenVLA~\citep{kim2024openvla}: reports 6 Hz on 7B at bfloat16 — i.e., ~166 ms
- General robotics: closed-loop manipulation typically 10–50 Hz → 20–100 ms budget

Paper narrative: "Qwen3-VL-2B at seq 1024 takes X ms on an H100; adding a
fixed 35 % AttnRes probe overhead pushes this to 1.35 X ms, which crosses the
20 Hz control threshold for real-time manipulation — unless skip pays for it."

---

## Claim 3: training AttnRes from scratch at VLA scale is prohibitive

### Experiment C1 — cost model, no new GPU runs needed

Compute, don't measure. **Scope**: motivation chapter is problem-statement
only, so the table has **two rows** (no retrofit row — retrofit cost appears
in the methods/experiments chapter).

- **340M AttnRes** (reference): 340M × 100B tokens ≈ 2.0e20 FLOPs. At 312
  TFLOPs/s bf16 on 1×H100 → ~180 H100-hours. Used to calibrate the 2B estimate.
- **2B AttnRes-VLM from scratch**: 2B × 200B tokens ≈ 2.4e21 FLOPs → ~2,100
  core H100-hours. At 70% utilisation → ~3,000. Multiply by ~8× for
  distributed-training overhead, multi-context-length stages, pre-SFT
  alignment. **Real-world estimate: 10,000–25,000 H100-hours per 2B AttnRes
  VLM.**

Deliverable: cost-only table for motivation (Section \ref{sec:motivation_cost}):

| path to 2B backbone | trainable params | training tokens | H100-hours |
|---------------------|------------------|-----------------|------------|
| Stock Qwen3-VL-2B (as-is, no AttnRes) | 0 (given) | 0 | 0 |
| 2B AttnRes-VLM from scratch | 2.13 B | ~200 B | 10,000–25,000 |

---

## NOT in motivation — moved to experiments chapter

The following were originally sketched as part of motivation but belong to the
methods + experiments chapters per the problem-statement-only scoping:

- Quality-vs-base comparison table for our retrofit (LAMBADA / MMMU / MMBench / MMStar) — lives in Section \ref{sec:experiments}
- Wall-clock table including `retrofit-full` and `retrofit + dyn-skip` — Section \ref{sec:experiments}
- Our retrofit row in the cost table — Section \ref{sec:experiments}
- Pareto figure with our retrofit on it — Section \ref{sec:experiments}

Keeping these out of motivation is deliberate: the motivation chapter is
allowed only to show (i) the problem exists and (ii) solving it is
necessary. Any row that reads like "we solved it with X" belongs further down
the paper.

---

## What we need to run new

Minimal set of commands to produce every missing number above:

### Run 1 — vanilla 340M latency (Claim 1 · A1 / A3)

Needs the vanilla baseline checkpoint. Point to it + run the existing
`benchmark_speed.py`-style harness on it.

```
benchmark_speed_vanilla.py --model-path <path_to_base_transformer_340M> \
  --seq-lens "512,1024,2048,8192" --warmup 5 --timed 20
```

### Run 2 — AttnRes 340M latency, full + skip (Claim 1 · A1)

Reuse Part-1 checkpoint + its skip config; same seq lens.

```
benchmark_speed_attnres340m.py \
  --model-path flame/saves/reskip_transformer_340M \
  --dynamic-skip-config "P={3,5}, q=0.85, M=2" \
  --seq-lens "512,1024,2048,8192"
```

### Run 3 — per-block profile (Claim 1 · A2)

Run a single forward under `torch.profiler` at seq 2048 on AttnRes-340M, export
to timeline JSON, extract per-block timings.

### Run 4 — Qwen3-VL-2B short-seq latency (Claim 2 · B1)

Extend existing `benchmark_speed.py` output to include seq 256 and 512. Already
have 1024 / 2048 / 4096.

### No GPU runs for Claims 3 and 4

- Claim 3: pure cost model, computed on paper.
- Claim 4: reuse numbers already in `retrofit.md`.

---

## Proposed figures

1. **Fig 1 (opening)** — latency vs seq_len for the three 340M variants. Shows
   the AttnRes overhead + skip's partial recovery on a log axis. One line is
   "vanilla standard residual" as the visual baseline, two other lines are
   "AttnRes full" and "AttnRes + phase-1 skip".
2. **Fig 2** — stacked-bar per-block breakdown of AttnRes overhead (which part
   is router, which is decoder, which is adapter). Helps justify the γ-gated
   residual design choice.
3. **Fig 3** — cost-quality Pareto at 2B scale: x-axis H100-hours (log),
   y-axis LAMBADA / MMMU. Three dots: stock Qwen3-VL-2B (free, no AttnRes),
   from-scratch AttnRes-VLM-2B (10k+ hr, hypothetical), our retrofit (~1 hr,
   present).

---

## Sequencing

1. Confirm this plan
2. Locate vanilla-340M checkpoint and the AttnRes-340M checkpoint + skip config
3. Write the two benchmarking scripts (small, will reuse code)
4. Run Runs 1, 2, 4 in parallel (4 GPUs; Run 3 wants a single GPU with profiler)
5. Compile tables / figures
6. Land them in `main.tex` Sec 1 / Sec 2

Open questions before running:
- Do we have the 340M vanilla base checkpoint, or do we need to (re)train it?
- Is our measurement methodology (random tokens, bf16, single H100) what we
  want for paper-grade numbers, or should we switch to realistic prompts and
  average across a real benchmark set?
- Do we want to include an A100 measurement to make the VLA latency claim less
  hardware-specific?
