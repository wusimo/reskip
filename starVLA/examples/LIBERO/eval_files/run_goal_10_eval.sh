#!/bin/bash
# Run libero_goal + libero_10 for a checkpoint. Used for variance / 2nd-seed replication.
# Usage:
#   bash run_goal_10_eval.sh <label> <ckpt> <server_gpu> <render_gpu> <port> [trials]
set -u
LABEL="${1:?usage: <label> <ckpt> <server_gpu> <render_gpu> <port> [trials]}"
CKPT="${2:?}"
SERVER_GPU="${3:?}"
RENDER_GPU="${4:?}"
PORT="${5:?}"
TRIALS="${6:-50}"

STARVLA=/home/user01/Minko/reskip2/reskip/starVLA
OUT_ROOT=/home/user01/Minko/reskip2/reskip/retrofit/outputs/libero_eval_full
mkdir -p "$OUT_ROOT"

cd "$STARVLA"
echo "[$LABEL] starting policy server on GPU $SERVER_GPU port $PORT"
CUDA_VISIBLE_DEVICES=$SERVER_GPU WANDB_MODE=offline PYTHONPATH=$(pwd) \
  python deployment/model_server/server_policy.py \
    --ckpt_path "$CKPT" --port "$PORT" --use_bf16 \
  > "$OUT_ROOT/${LABEL}_server.log" 2>&1 &
SERVER_PID=$!
echo "[$LABEL] server PID=$SERVER_PID"

for i in $(seq 1 60); do
  if grep -q "server listening" "$OUT_ROOT/${LABEL}_server.log" 2>/dev/null; then
    echo "[$LABEL] server listening"
    break
  fi
  sleep 5
done

for SUITE in libero_goal libero_10; do
  OUT="$OUT_ROOT/${LABEL}_${SUITE}"
  mkdir -p "$OUT"
  echo "[$LABEL] === ${SUITE} begin === (trials=$TRIALS, render_gpu=$RENDER_GPU)"
  CUDA_VISIBLE_DEVICES=$RENDER_GPU bash examples/LIBERO/eval_files/eval_libero.sh \
    "$CKPT" "$SUITE" "$PORT" "$TRIALS" \
    > "$OUT/eval.log" 2>&1
  SR=$(grep -oE "Total success rate: [0-9.]+" "$OUT/eval.log" | tail -1)
  echo "[$LABEL] === ${SUITE} done: $SR ==="
done

kill -9 $SERVER_PID 2>/dev/null
echo "[$LABEL] GOAL+10 DONE"
