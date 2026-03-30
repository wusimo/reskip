# ReSkip Experiment Playbook

This document turns [PLAN.md](/Users/apple/Desktop/luxi/reskip/PLAN.md) into an execution-oriented experiment workflow.

It explicitly separates:

- `Implemented now`: can be run with the current repository state.
- `Planned next`: part of the paper plan, but still requires additional code.
- `Not in this repo yet`: depends on external integration work such as StarVLA/LIBERO.


## 1. Overall Structure

The paper plan has two major lines:

1. `LM main line`
   Goal: validate AttnRes/ReSkip on large-scale natural language pretraining.
2. `VLA line`
   Goal: port the same adaptive-depth mechanism to a real VLA backbone and benchmark it on embodied control.

The intended execution order is:

1. Train `350M baseline LM`
2. Train `350M AttnRes LM`
3. Evaluate skip tradeoffs and routing on natural language
4. Run additional LM baselines and ReLoop ablations
5. Port AttnRes into StarVLA
6. Run LIBERO VLA baselines and skip comparisons

## 2. Environment

Recommended Python packages for the current repo:

```bash
pip install torch numpy matplotlib pyyaml tqdm tiktoken pyarrow
```

Optional packages for future work:

```bash
pip install datasets lm-eval
```

Current local SlimPajama path on the server:

```text
/inspire/ssd/project/multimodal-brain-signal/liuzhenyang-240108540154/Simo/datasets/SlimPajama-627B/data
```

Before training, verify the parquet schema:

```bash
python - <<'PY'
import pyarrow.parquet as pq
path = "/inspire/ssd/project/multimodal-brain-signal/liuzhenyang-240108540154/Simo/datasets/SlimPajama-627B/data/train-00000-of-00250.parquet"
pf = pq.ParquetFile(path)
print(pf.schema.names)
PY
```

If the text column is not `text`, replace `--text_key text` in all commands below with the actual column name.

## 3. LM Main Line

### 3.1 Standard Transformer Baseline

Status: `Implemented now`

Purpose:

- Establish the full-depth language-modeling baseline on SlimPajama.
- Provide the quality reference for all later AttnRes/ReSkip comparisons.
- Serve as the source model for static pruning baselines.

Recommended single-node command:

```bash
torchrun --nproc_per_node=1 experiments/train_lm.py \
  --mode baseline \
  --dataset slimpajama \
  --data_path /inspire/ssd/project/multimodal-brain-signal/liuzhenyang-240108540154/Simo/datasets/SlimPajama-627B/data \
  --val_data_path /inspire/ssd/project/multimodal-brain-signal/liuzhenyang-240108540154/Simo/datasets/SlimPajama-627B/data \
  --text_key text \
  --tokenizer_name gpt2 \
  --d_model 896 \
  --n_heads 14 \
  --n_layers 24 \
  --n_blocks 8 \
  --seq_len 2048 \
  --batch_size 2 \
  --grad_accum_steps 8 \
  --max_steps 20000 \
  --warmup_steps 2000 \
  --lr 3e-4 \
  --weight_decay 0.1 \
  --use_rope \
  --amp_dtype bf16 \
  --num_train 5120000 \
  --num_val 4096 \
  --num_workers 4 \
  --log_every 1 \
  --device cuda \
  --output_dir outputs/350m_baseline
```

Notes:

- `d_model=896, n_heads=14, n_layers=24` is closer to the intended 350M scale than `1024 x 24`, which is ~454M.
- If you want the exact previously tested command, replace `896/14` with `1024/16`.

Outputs:

- `outputs/350m_baseline/best.pt`
- `outputs/350m_baseline/train_log.jsonl`
- `outputs/350m_baseline/final_results.json`

### 3.2 AttnRes Main LM Model

Status: `Implemented now`

Purpose:

- Train the main AttnRes language model on the same data and compute budget as the baseline.
- Test whether AttnRes preserves full-depth quality while enabling later skip-based acceleration.

Command:

```bash
torchrun --nproc_per_node=1 experiments/train_lm.py \
  --mode standard \
  --dataset slimpajama \
  --data_path /inspire/ssd/project/multimodal-brain-signal/liuzhenyang-240108540154/Simo/datasets/SlimPajama-627B/data \
  --val_data_path /inspire/ssd/project/multimodal-brain-signal/liuzhenyang-240108540154/Simo/datasets/SlimPajama-627B/data \
  --text_key text \
  --tokenizer_name gpt2 \
  --d_model 896 \
  --n_heads 14 \
  --n_layers 24 \
  --n_blocks 8 \
  --seq_len 2048 \
  --batch_size 2 \
  --grad_accum_steps 8 \
  --max_steps 20000 \
  --warmup_steps 2000 \
  --lr 3e-4 \
  --weight_decay 0.1 \
  --use_rope \
  --amp_dtype bf16 \
  --num_train 5120000 \
  --num_val 4096 \
  --num_workers 4 \
  --log_every 1 \
  --device cuda \
  --output_dir outputs/350m_attnres
```

Outputs:

- `outputs/350m_attnres/best.pt`
- `outputs/350m_attnres/train_log.jsonl`
- `outputs/350m_attnres/final_results.json`

### 3.3 Skip Sweep and Checkpoint Evaluation

Status: `Implemented now`

Purpose:

- Evaluate the trained AttnRes model under different skip thresholds.
- Measure validation loss, perplexity, and effective FLOPs.

Command:

```bash
python experiments/train_lm.py \
  --mode eval \
  --checkpoint outputs/350m_attnres/best.pt \
  --dataset slimpajama \
  --data_path /inspire/ssd/project/multimodal-brain-signal/liuzhenyang-240108540154/Simo/datasets/SlimPajama-627B/data \
  --val_data_path /inspire/ssd/project/multimodal-brain-signal/liuzhenyang-240108540154/Simo/datasets/SlimPajama-627B/data \
  --text_key text \
  --tokenizer_name gpt2 \
  --batch_size 2 \
  --num_val 4096 \
  --num_workers 4 \
  --device cuda \
  --output_dir outputs/350m_attnres_eval
```

### 3.4 Routing Analysis

Status: `Implemented now`

Purpose:

- Plot routing heatmaps and training curves from the AttnRes checkpoint.

Command:

```bash
python experiments/analyze_routing.py \
  --checkpoint outputs/350m_attnres/best.pt \
  --sweep_results outputs/350m_attnres/final_results.json \
  --train_log outputs/350m_attnres/train_log.jsonl \
  --device cuda \
  --output_dir outputs/350m_attnres/analysis
```

## 4. LM Baselines and Extensions

### 4.1 Static Pruning

Status: `Planned next`

Purpose:

- Start from the standard baseline checkpoint.
- Compute layer/block importance.
- Remove the least important blocks and evaluate perplexity vs FLOPs.

Current repo status:

- No `src/baselines/static_pruning.py` yet.
- Requires implementation before a final command can be provided.

Planned workflow:

1. Train `outputs/350m_baseline`
2. Compute block influence scores
3. Prune N blocks
4. Re-run evaluation on the pruned model

Placeholder command shape after implementation:

```bash
python src/baselines/static_pruning.py \
  --checkpoint outputs/350m_baseline/best.pt \
  --dataset slimpajama \
  --data_path /inspire/ssd/project/multimodal-brain-signal/liuzhenyang-240108540154/Simo/datasets/SlimPajama-627B/data \
  --text_key text \
  --device cuda \
  --output_dir outputs/static_pruning
```

### 4.2 CALM

Status: `Planned next`

Purpose:

- Add early-exit classifiers as a stronger adaptive-compute baseline.

Current repo status:

- Not implemented.

### 4.3 Mixture-of-Depths

Status: `Planned next`

Purpose:

- Compare AttnRes routing with a learned per-token routing baseline.

Current repo status:

- Not implemented.

### 4.4 Dense Threshold Sweep

Status: `Partially implemented`

Current implementation:

- `train_lm.py --mode eval` runs a sparse threshold set.

Planned next:

- Extend the threshold list to 20+ points.

Current command:

```bash
python experiments/train_lm.py \
  --mode eval \
  --checkpoint outputs/350m_attnres/best.pt \
  --dataset slimpajama \
  --data_path /inspire/ssd/project/multimodal-brain-signal/liuzhenyang-240108540154/Simo/datasets/SlimPajama-627B/data \
  --text_key text \
  --batch_size 2 \
  --num_val 4096 \
  --device cuda \
  --output_dir outputs/350m_attnres_eval_dense
```

## 5. ReLoop Line

### 5.1 ReLoop Training

Status: `Implemented now`

Purpose:

- Test weight sharing across depth with AttnRes-based differentiation of loop iterations.

Command:

```bash
torchrun --nproc_per_node=1 experiments/train_lm.py \
  --mode looping \
  --dataset slimpajama \
  --data_path /inspire/ssd/project/multimodal-brain-signal/liuzhenyang-240108540154/Simo/datasets/SlimPajama-627B/data \
  --val_data_path /inspire/ssd/project/multimodal-brain-signal/liuzhenyang-240108540154/Simo/datasets/SlimPajama-627B/data \
  --text_key text \
  --tokenizer_name gpt2 \
  --d_model 896 \
  --n_heads 14 \
  --n_layers 24 \
  --n_unique_blocks 4 \
  --max_loops 3 \
  --seq_len 2048 \
  --batch_size 2 \
  --grad_accum_steps 8 \
  --max_steps 20000 \
  --warmup_steps 2000 \
  --lr 3e-4 \
  --weight_decay 0.1 \
  --use_rope \
  --amp_dtype bf16 \
  --num_train 5120000 \
  --num_val 4096 \
  --num_workers 4 \
  --log_every 1 \
  --device cuda \
  --output_dir outputs/350m_reloop
```

### 5.2 K x M Grid

Status: `Implemented manually`

Purpose:

- Sweep `n_unique_blocks` and `max_loops`.

Suggested grid:

- `(2, 6)`
- `(3, 4)`
- `(4, 3)`
- `(6, 2)`

Example:

```bash
torchrun --nproc_per_node=1 experiments/train_lm.py \
  --mode looping \
  --dataset slimpajama \
  --data_path /inspire/ssd/project/multimodal-brain-signal/liuzhenyang-240108540154/Simo/datasets/SlimPajama-627B/data \
  --val_data_path /inspire/ssd/project/multimodal-brain-signal/liuzhenyang-240108540154/Simo/datasets/SlimPajama-627B/data \
  --text_key text \
  --tokenizer_name gpt2 \
  --d_model 896 \
  --n_heads 14 \
  --n_layers 24 \
  --n_unique_blocks 2 \
  --max_loops 6 \
  --seq_len 2048 \
  --batch_size 2 \
  --grad_accum_steps 8 \
  --max_steps 20000 \
  --warmup_steps 2000 \
  --lr 3e-4 \
  --weight_decay 0.1 \
  --use_rope \
  --amp_dtype bf16 \
  --num_train 5120000 \
  --num_val 4096 \
  --num_workers 4 \
  --log_every 1 \
  --device cuda \
  --output_dir outputs/reloop_k2_m6
```

### 5.3 Vanilla Universal Transformer

Status: `Planned next`

Current repo status:

- Not implemented.

## 6. VLA Line

### 6.1 Current Repo VLA Benchmark

Status: `Prototype only`

Purpose:

- Sanity-check the toy multimodal adaptive-depth setup.

Command:

```bash
python experiments/benchmark_vla.py --device cuda --output_dir outputs/vla_proto
```

This is not the paper-grade VLA experiment.

### 6.2 StarVLA + AttnRes Integration

Status: `Not in this repo yet`

Purpose:

- Patch a real VLA backbone with BlockAttnRes.
- Keep the existing vision encoder and action heads.

Required code work:

- Create `src/starvla_integration.py`
- Patch the StarVLA backbone residual path
- Add model loading/config glue

### 6.3 LIBERO Main VLA Benchmark

Status: `Not in this repo yet`

Target comparisons:

1. `StarVLA` full depth
2. `StarVLA + AttnRes` full depth
3. `StarVLA + AttnRes + uniform skip`
4. `StarVLA + AttnRes + modality-aware skip`

Target metrics:

- task success rate
- average actions to completion
- latency

Planned command shape after implementation:

```bash
python experiments/benchmark_vla.py \
  --backend starvla \
  --libero_root /path/to/LIBERO \
  --task_suite libero_long \
  --mode attnres_full \
  --device cuda \
  --output_dir outputs/libero_attnres_full
```

Uniform skip:

```bash
python experiments/benchmark_vla.py \
  --backend starvla \
  --libero_root /path/to/LIBERO \
  --task_suite libero_long \
  --mode uniform_skip \
  --skip_threshold 0.05 \
  --device cuda \
  --output_dir outputs/libero_uniform_skip
```

Modality-aware skip:

```bash
python experiments/benchmark_vla.py \
  --backend starvla \
  --libero_root /path/to/LIBERO \
  --task_suite libero_long \
  --mode modality_aware_skip \
  --vision_skip_threshold 0.1 \
  --language_skip_threshold 0.03 \
  --action_skip_threshold 0.005 \
  --device cuda \
  --output_dir outputs/libero_modality_skip
```

These commands are placeholders until the real StarVLA/LIBERO integration is implemented.

## 7. Recommended Execution Order

### 7.1 Immediate

Run in this order:

1. `350m_baseline`
2. `350m_attnres`
3. `350m_attnres eval`
4. `350m_attnres analysis`
5. `350m_reloop`

### 7.2 Next Development

Implement next:

1. `static pruning`
2. `dense threshold sweep`
3. `lm-eval-harness integration`
4. `StarVLA integration`
5. `LIBERO rollout evaluation`

### 7.3 Paper-Grade Completion

The paper plan is only truly complete when all of these exist:

- large-scale LM baseline
- large-scale LM AttnRes
- skip tradeoff curves
- missing LM baselines
- ReLoop ablations
- real VLA backbone integration
- LIBERO success-rate and latency curves

## 8. What Is Already True Right Now

The current repo can already support:

- local SlimPajama parquet pretraining
- baseline LM training
- AttnRes LM training
- ReLoop LM training
- checkpoint evaluation
- sparse skip sweep
- routing analysis
- toy VLA benchmark

The current repo does not yet support:

- paper-grade VLA experiments
- static pruning baseline
- CALM
- MoD
- Universal Transformer baseline
- lm-eval downstream benchmark suite
- StarVLA/LIBERO integration
