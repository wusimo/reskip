# ReLoop Experiment Log

## 2026-04-15 单卡 `reloop_transformer_340M-test` 训练闭环验证

实验目的：
- 先不再纠结双卡通信稳定性。
- 用单卡把 `ReLoop` 当前训练链路完整跑通一次。
- 检查训练结果是否真的学到了我们要的动态 loop 行为。
- 对最终 checkpoint 做分析和 `lm-eval`，确认它不是“看起来训完了但实际上没学对”。

训练配置：
- 配置文件：[reloop_transformer_340M-test.json](/home/user01/Minko/reskip2/reskip/flame/configs/reloop_transformer_340M-test.json)
- 训练目录：[loopskip_transformer_test_single](/home/user01/Minko/reskip2/reskip/flame/saves/loopskip_transformer_test_single)
- 日志文件：[stderr.log](/home/user01/Minko/reskip2/reskip/logs/none_gd_52op8/attempt_0/0/stderr.log)

训练命令要点：
- 单卡 `GPU6`
- `steps = 10000`
- `seq_len = 65536`
- `gradient_accumulation_steps = 8`
- `data_parallel_shard_degree = 1`
- `activation_checkpoint.mode = none`

训练结束状态：
- 本次训练已完整跑到 `step 10000`
- 日志中明确出现 `Training completed`
- 最终训练日志：
  - `loss = 3.0979`
  - `effective_depth = 6.00`
  - `expected_depth = 6.00`
  - `depth_ratio = 0.75`
  - `ponder_cost = 6.00`

checkpoint 产物：
- [step-1](/home/user01/Minko/reskip2/reskip/flame/saves/loopskip_transformer_test_single/checkpoint/step-1)
- [step-2000](/home/user01/Minko/reskip2/reskip/flame/saves/loopskip_transformer_test_single/checkpoint/step-2000)
- [step-4000](/home/user01/Minko/reskip2/reskip/flame/saves/loopskip_transformer_test_single/checkpoint/step-4000)
- [step-6000](/home/user01/Minko/reskip2/reskip/flame/saves/loopskip_transformer_test_single/checkpoint/step-6000)
- [step-8000](/home/user01/Minko/reskip2/reskip/flame/saves/loopskip_transformer_test_single/checkpoint/step-8000)
- [step-10000](/home/user01/Minko/reskip2/reskip/flame/saves/loopskip_transformer_test_single/checkpoint/step-10000)

补充说明：
- 日志尾部出现的 `TCPStore Broken pipe` / `ProcessGroupNCCL` 警告发生在训练结束后的进程组收尾阶段。
- 这次单卡训练本体不是被这些警告打断的。

## 2026-04-15 `step-10000` HF 导出与行为分析

导出目的：
- `flame_analyze_reloop.py` 和 `flame_lm_eval.py` 都默认按 HF 目录加载模型。
- 因此先把 `step-10000` 的 DCP checkpoint 导成 HF 目录，再进行后续分析和跑分。

导出结果：
- 导出目录仍为：[loopskip_transformer_test_single](/home/user01/Minko/reskip2/reskip/flame/saves/loopskip_transformer_test_single)
- 导出后的 HF 文件：
  - [config.json](/home/user01/Minko/reskip2/reskip/flame/saves/loopskip_transformer_test_single/config.json)
  - [generation_config.json](/home/user01/Minko/reskip2/reskip/flame/saves/loopskip_transformer_test_single/generation_config.json)
  - [model.safetensors](/home/user01/Minko/reskip2/reskip/flame/saves/loopskip_transformer_test_single/model.safetensors)
  - [tokenizer.json](/home/user01/Minko/reskip2/reskip/flame/saves/loopskip_transformer_test_single/tokenizer.json)
  - [tokenizer_config.json](/home/user01/Minko/reskip2/reskip/flame/saves/loopskip_transformer_test_single/tokenizer_config.json)
  - [special_tokens_map.json](/home/user01/Minko/reskip2/reskip/flame/saves/loopskip_transformer_test_single/special_tokens_map.json)

分析命令：

```bash
CUDA_VISIBLE_DEVICES=6 /home/user01/Minko/reskip2/.venv/bin/python \
  /home/user01/Minko/reskip2/reskip/experiments/flame_analyze_reloop.py \
  --model_path /home/user01/Minko/reskip2/reskip/flame/saves/loopskip_transformer_test_single \
  --dataset /home/user01/Minko/datasets/fineweb_edu_100BT \
  --dataset_split train \
  --seq_len 65536 \
  --context_len 2048 \
  --batch_size 1 \
  --num_workers 2 \
  --num_batches 32 \
  --sample_trace_limit 8 \
  --streaming \
  --varlen \
  --device cuda:0 \
  --dtype bf16 \
  --output_path /home/user01/Minko/reskip2/reskip/outputs/reloop_test_single_step10000_analysis.json
```

分析结果：
- 分析文件：[reloop_test_single_step10000_analysis.json](/home/user01/Minko/reskip2/reskip/outputs/reloop_test_single_step10000_analysis.json)
- 分析集平均：
  - `loss = 3.3586`
  - `perplexity = 28.7490`
  - `effective_depth = 6.0`
  - `expected_depth = 6.0`
  - `compute_ratio = 0.75`
  - `ponder_cost = 6.0`

深度分布：
- `effective_depth histogram = {6.00: 32}`
- `expected_depth histogram = {6.00: 32}`
- `hard_exit_depth_from_trace = {6.00: 32}`
- `hard_exit_depth_from_halt_probs = {6.00: 32}`

位置级 halt 行为：
- `position 0-4`：
  - `mean_halt_probability` 基本为 `0`
  - `status = executed`
- `position 5`：
  - `mean_halt_probability = 1.0`
  - `status = executed`
- `position 6-7`：
  - `mean_executed_fraction = 0.0`
  - `status = halted`

代表性 trace：
- 分析样本里，halt probabilities 基本呈现：
  - 前 5 个位置接近 `0`
  - 第 6 个位置直接为 `1.0`
- 然后 `position 6` 与 `position 7` 不再执行。

当前解释：
- 这说明模型确实学会了“提前停止”。
- 但它学成的不是样本相关的动态 halting。
- 它几乎已经塌缩成一个固定策略：
  - 总是跑到第 `6` 个 block
  - 然后稳定停止

因此当前 `ReLoop` 的训练结果更接近：
- `固定 6-depth 截断`

而不是：
- `根据输入内容自主选择 loop 次数`

## 2026-04-15 `step-10000` 四任务 lm-eval

评测命令：

```bash
CUDA_VISIBLE_DEVICES=6 /home/user01/Minko/reskip2/.venv/bin/python \
  /home/user01/Minko/reskip2/reskip/experiments/flame_lm_eval.py \
  --model_path /home/user01/Minko/reskip2/reskip/flame/saves/loopskip_transformer_test_single \
  --tasks lambada_openai,hellaswag,arc_easy,arc_challenge \
  --batch_size auto \
  --device cuda:0 \
  --output_path /home/user01/Minko/reskip2/reskip/outputs/lm_eval_reloop_test_single_step10000
```

结果文件：
- [results_2026-04-15T00-42-14.371746.json](/home/user01/Minko/reskip2/reskip/outputs/lm_eval_reloop_test_single_step10000/__home__user01__Minko__reskip2__reskip__flame__saves__loopskip_transformer_test_single/results_2026-04-15T00-42-14.371746.json)

四任务结果：

| Task | Metric | Value |
|---|---:|---:|
| lambada_openai | acc | 0.1997 |
| lambada_openai | ppl | 196.3694 |
| hellaswag | acc_norm | 0.2936 |
| arc_easy | acc_norm | 0.4171 |
| arc_challenge | acc_norm | 0.2432 |

结果解读：
- 这组分数整体偏弱。
- 至少就当前 `10000 step` 的 test 配置来说，模型还没有达到一个可接受的语言建模质量水平。
- 结合上面的分析结果，可以把当前问题定性为：
  - 训练确实让模型学会了“降深度”
  - 但它没有学会“有区分度的动态 loop”
  - 同时语言建模质量也还不足

## 当前阶段结论

这轮单卡闭环验证已经回答了几个关键问题。

确认成立的部分：
- `ReLoop` 当前训练链路已经能完整跑通：
  - 训练
  - DCP checkpoint 保存
  - HF 导出
  - 离线行为分析
  - `lm-eval`
- 当前实现确实能让模型学会“少算一些 block”

当前暴露出的核心问题：
- 这版训练目标会把模型推向一个非常稳定的固定停止深度。
- 现在看到的不是“dynamic routing/halting”，而是“collapsed fixed-depth policy”。
- 当前 `step-10000` checkpoint 体现为：
  - 固定 `6 / 8` depth
  - 固定 `0.75` compute ratio
  - 几乎没有样本间差异

因此本轮实验最重要的结论不是“链路有 bug”，而是：
- **链路已经通了，但当前训练目标不够对，学出来的行为不是我们真正要的动态 loop。**

## 下一步方向

下一步不应该继续盲目放大训练规模，而应该优先处理训练目标本身：
- 抑制 halt 策略塌缩成固定深度
- 让 halt 决策必须对样本内容产生真实区分
- 重新审视：
  - `halt_kl`
  - `early_exit_penalty`
  - `focused_halt_loss`
  - `ponder target curriculum`
 之间的耦合方式

当前优先级判断：
- 首先解决“固定 6-depth 塌缩”
- 然后再重新做短训验证
- 只有在分析结果开始出现样本相关的 depth 分布后，才值得继续追加长训
