#!/bin/bash
# Experiment 3: VLM-only reskip LAMBADA on H_r256_5k-style 2B_L4_v3_10k.
# Runs baseline (max-skips=0) + dynskip q ∈ {0.30, 0.50, 0.85} on GPU 6.

set -u
RETROFIT=/home/user01/Minko/reskip2/reskip/retrofit
OUT_ROOT=$RETROFIT/outputs/libero_eval_full
PY=/home/user01/Minko/reskip2/.venv/bin/python
STATE=$RETROFIT/outputs/block_v3/2B_L4_v3_10k/retrofit_attnres_state.pt
LOG=$OUT_ROOT/vlm_reskip_exp3.log
GPU=6

: > "$LOG"
echo "=== Experiment 3: VLM-only reskip on 2B_L4_v3_10k ===" >> "$LOG"
echo "start $(date)" >> "$LOG"
echo "state: $STATE" >> "$LOG"
echo "" >> "$LOG"

echo "--- Baseline (max-skips=0, no skip) ---" >> "$LOG"
$PY $RETROFIT/eval/eval_dynamic_skip.py \
  --state-path "$STATE" --num-blocks 7 --max-skips 0 \
  --lambada-n 500 --gpu $GPU >> "$LOG" 2>&1

for Q in 0.30 0.50 0.85; do
  echo "" >> "$LOG"
  echo "--- Dynskip q=$Q P={1,4} M=2 ---" >> "$LOG"
  $PY $RETROFIT/eval/eval_dynamic_skip.py \
    --state-path "$STATE" --num-blocks 7 \
    --quantile $Q --eligible 1,4 --max-skips 2 \
    --lambada-n 500 --gpu $GPU >> "$LOG" 2>&1
done

echo "" >> "$LOG"
echo "done $(date)" >> "$LOG"
