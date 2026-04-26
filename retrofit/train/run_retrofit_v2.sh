#!/bin/bash
# Retrofit v2 launcher: reasoning/math/long-context enriched data mix
# (retrofit/train/data_v2.py DEFAULT_V2_MIX).
#
# Usage:
#   bash retrofit/train/run_retrofit_v2.sh 2B|4B <gpu>
#
# Cells:
#   2B -> outputs/H_r256_5k_v2_2B (GPU default 0, max_seq=4096)
#   4B -> outputs/H_4B_r256_5k_v2 (GPU default 1, max_seq=2048 to fit memory)
#
# Matches H_r256_5k hyperparameters except for --data-mix v2 and --max-seq.

set -u
PROJECT=/home/user01/Minko/reskip2/reskip
PY=/home/user01/Minko/reskip2/.venv/bin/python
cd "$PROJECT"

MODEL_SCALE="${1:?usage: $0 2B|4B [gpu]}"
GPU="${2:-}"

case "$MODEL_SCALE" in
  2B)
    MODEL_PATH=/home/user01/Minko/models/Qwen3-VL-2B
    NUM_BLOCKS=14
    MAX_SEQ=2048
    OUTDIR=$PROJECT/retrofit/outputs/H_r256_5k_v2_2B
    [ -z "$GPU" ] && GPU=0
    ;;
  4B)
    MODEL_PATH=/home/user01/Minko/models/Qwen3-VL-4B
    NUM_BLOCKS=18
    MAX_SEQ=2048
    OUTDIR=$PROJECT/retrofit/outputs/H_4B_r256_5k_v2
    [ -z "$GPU" ] && GPU=1
    ;;
  *) echo "unknown scale: $MODEL_SCALE (want 2B or 4B)"; exit 1 ;;
esac

mkdir -p "$OUTDIR"
cp "$0" "$OUTDIR/"
LOG=$OUTDIR/run.log
echo "[$(date)] retrofit v2 $MODEL_SCALE on GPU $GPU, max_seq=$MAX_SEQ, out=$OUTDIR"

nohup $PY retrofit/train/train_qwen3vl_attnres_retrofit.py \
  --model-path "$MODEL_PATH" \
  --num-blocks "$NUM_BLOCKS" \
  --adapter-rank 256 \
  --steps 5000 \
  --max-seq "$MAX_SEQ" \
  --gamma-schedule --gamma-start 0 --gamma-end 1 --gamma-ramp-frac 0.3 \
  --data-mix v2 \
  --kl-weight 1.0 \
  --entropy-weight 0.02 \
  --output-dir "$OUTDIR" \
  --gpu "$GPU" \
  > "$LOG" 2>&1 &
echo "pid=$!"
echo "log: $LOG"
