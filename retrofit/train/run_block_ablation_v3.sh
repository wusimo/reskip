#!/bin/bash
# Full block-partition ablation on top of the v3 data/training config.
# 4 block configs × 2 model sizes = 8 cells on 8 GPUs, all in parallel.
#
# Training config (from v3 main findings):
#   --data-mix v3 (60% LLaVA-OV + 20% UltraChat + 10% NuminaMath + 10% OpenThoughts)
#   --steps 10000, --gamma-ramp-frac 0.5 (for 4B stability)
#   --adapter-rank 256, --max-seq 2048
#
# 2B (28 layers): L ∈ {1, 2, 4, 7} → num_blocks ∈ {28, 14, 7, 4}
# 4B (36 layers): L ∈ {1, 2, 4, 6} → num_blocks ∈ {36, 18, 9, 6}

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

# 2B: GPUs 0, 2, 4, 6
launch 0 2B 28 1
launch 2 2B 14 2
launch 4 2B  7 4
launch 6 2B  4 7
# 4B: GPUs 1, 3, 5, 7
launch 1 4B 36 1
launch 3 4B 18 2
launch 5 4B  9 4
launch 7 4B  6 6

echo
echo "[$(date +%H:%M:%S)] all 8 block ablation cells launched"
