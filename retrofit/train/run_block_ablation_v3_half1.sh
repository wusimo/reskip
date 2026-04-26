#!/bin/bash
# First half of the 8-cell block-partition ablation.
# Runs L=1 and L=4 on both 2B and 4B, using the 4 free GPUs (4-7).
# Second half (L=2 re-run + L=7/6) will launch when GPUs 0-3 free up.
#
# All cells share v3-optimal config: v3 mix, 10k steps, adapter rank 256,
# max_seq 2048, γ-ramp-frac 0.5.

set -u
PROJECT=/home/user01/Minko/reskip2/reskip
PY=/home/user01/Minko/reskip2/.venv/bin/python
cd "$PROJECT"

launch() {
  local gpu="$1" scale="$2" num_blocks="$3" L="$4"
  local model_path
  if [ "$scale" = "2B" ]; then
    model_path=/home/user01/Minko/models/Qwen3-VL-2B
  else
    model_path=/home/user01/Minko/models/Qwen3-VL-4B
  fi
  local label="${scale}_L${L}_v3_10k"
  local outdir=$PROJECT/retrofit/outputs/block_v3/$label
  mkdir -p "$outdir"
  local log="$outdir/run.log"
  echo "[$(date +%H:%M:%S)] gpu $gpu → $label (num_blocks=$num_blocks)"

  nohup $PY retrofit/train/train_qwen3vl_attnres_retrofit.py \
    --model-path "$model_path" \
    --num-blocks "$num_blocks" \
    --adapter-rank 256 \
    --steps 10000 \
    --max-seq 2048 \
    --gamma-schedule --gamma-start 0 --gamma-end 1 --gamma-ramp-frac 0.5 \
    --data-mix v3 \
    --kl-weight 1.0 \
    --entropy-weight 0.02 \
    --output-dir "$outdir" \
    --gpu "$gpu" \
    > "$log" 2>&1 &
  echo "  pid=$!"
}

# Half 1: GPUs 4-7 (0-3 held by DreamerVLA)
launch 4 2B 28 1
launch 5 4B 36 1
launch 6 2B  7 4
launch 7 4B  9 4

echo
echo "[$(date +%H:%M:%S)] 4/8 block ablation cells launched on GPUs 4-7"
echo "Remaining (GPUs 0-3 when free): 2B L=2 & L=7, 4B L=2 & L=6"
