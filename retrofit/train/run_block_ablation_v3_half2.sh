#!/bin/bash
# Second half of the 8-cell block-partition ablation.
# 2B L=2 & L=7, 4B L=2 & L=6 — the 4 cells not covered by half1.
#
# Default target GPUs: 0, 1, 2, 3 (after yuxinglei's DreamerVLA frees them).
# Override with: GPUS="a,b,c,d" bash run_block_ablation_v3_half2.sh
#
# Config matches half1 (v3 mix, 10k steps, adapter rank 256, ramp-frac 0.5).
# 4B L=2 uses max_seq 2048 (already proven fits in 4B_L2/v3_4B).
# 4B L=6 uses max_seq 2048 (fewer adapters than L=4).

set -u
PROJECT=/home/user01/Minko/reskip2/reskip
PY=/home/user01/Minko/reskip2/.venv/bin/python
cd "$PROJECT"

GPUS_CSV="${GPUS:-0,1,2,3}"
IFS=',' read -ra GPU_LIST <<< "$GPUS_CSV"

launch() {
  local gpu="$1" scale="$2" num_blocks="$3" L="$4" max_seq="$5"
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
  echo "[$(date +%H:%M:%S)] gpu $gpu → $label (num_blocks=$num_blocks max_seq=$max_seq)"

  nohup $PY retrofit/train/train_qwen3vl_attnres_retrofit.py \
    --model-path "$model_path" \
    --num-blocks "$num_blocks" \
    --adapter-rank 256 \
    --steps 10000 \
    --max-seq "$max_seq" \
    --gamma-schedule --gamma-start 0 --gamma-end 1 --gamma-ramp-frac 0.5 \
    --data-mix v3 \
    --kl-weight 1.0 \
    --entropy-weight 0.02 \
    --output-dir "$outdir" \
    --gpu "$gpu" \
    > "$log" 2>&1 &
  echo "  pid=$!"
}

# 4 cells: 2B L=2, 2B L=7, 4B L=2, 4B L=6
launch "${GPU_LIST[0]}" 2B 14 2 2048
launch "${GPU_LIST[1]}" 2B  4 7 2048
launch "${GPU_LIST[2]}" 4B 18 2 2048
launch "${GPU_LIST[3]}" 4B  6 6 2048

echo
echo "[$(date +%H:%M:%S)] 4/4 half2 cells launched on GPUs $GPUS_CSV"
