#!/bin/bash
# Retrofit v3: VL-heavy (LLaVA-OneVision 60%) + math/CoT text (20%) + general
# text (20%), 10k steps (2× v2/v1). Fixes the v2 diagnosis:
#   H1 too-low VL fraction → 60% VL
#   H2 LLaVA-VSFT too narrow → LLaVA-OneVision-Data (3.2M samples)
#   H3 5k too few → 10k
#
# Usage: bash retrofit/train/run_retrofit_v3.sh 2B|4B <gpu>

set -u
PROJECT=/home/user01/Minko/reskip2/reskip
PY=/home/user01/Minko/reskip2/.venv/bin/python
cd "$PROJECT"

SCALE="${1:?usage: $0 2B|4B [gpu]}"
GPU="${2:-}"

case "$SCALE" in
  2B)
    MODEL_PATH=/home/user01/Minko/models/Qwen3-VL-2B
    NUM_BLOCKS=14
    OUTDIR=$PROJECT/retrofit/outputs/H_r256_10k_v3_2B
    [ -z "$GPU" ] && GPU=0
    ;;
  4B)
    MODEL_PATH=/home/user01/Minko/models/Qwen3-VL-4B
    NUM_BLOCKS=18
    OUTDIR=$PROJECT/retrofit/outputs/H_4B_r256_10k_v3
    [ -z "$GPU" ] && GPU=1
    ;;
  *) echo "unknown scale: $SCALE"; exit 1 ;;
esac

mkdir -p "$OUTDIR"
cp "$0" "$OUTDIR/"
LOG=$OUTDIR/run.log
echo "[$(date)] retrofit v3 $SCALE on GPU $GPU -> $OUTDIR"

nohup $PY retrofit/train/train_qwen3vl_attnres_retrofit.py \
  --model-path "$MODEL_PATH" \
  --num-blocks "$NUM_BLOCKS" \
  --adapter-rank 256 \
  --steps 10000 \
  --max-seq 2048 \
  --gamma-schedule --gamma-start 0 --gamma-end 1 --gamma-ramp-frac 0.3 \
  --data-mix v3 \
  --kl-weight 1.0 \
  --entropy-weight 0.02 \
  --output-dir "$OUTDIR" \
  --gpu "$GPU" \
  > "$LOG" 2>&1 &
echo "pid=$!"
echo "log: $LOG"
