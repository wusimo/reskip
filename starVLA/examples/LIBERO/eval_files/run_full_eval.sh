#!/bin/bash
# Run FULL LIBERO eval for a checkpoint across all 4 suites sequentially.
# Usage:
#   bash run_full_eval.sh <label> <ckpt> <server_gpu> <render_gpu> <port> [trials]
#
# Launches:
#   - 1 policy server on $server_gpu at port
#   - sequential eval on 4 suites: libero_spatial, libero_object, libero_goal, libero_10
#   - each suite: $trials trials per task (default 50)
#
# Logs in retrofit/outputs/libero_eval/<label>_<suite>/eval.log
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

# 1. Start policy server (background)
cd "$STARVLA"
echo "[$LABEL] starting policy server on GPU $SERVER_GPU port $PORT"
CUDA_VISIBLE_DEVICES=$SERVER_GPU WANDB_MODE=offline PYTHONPATH=$(pwd) \
  python deployment/model_server/server_policy.py \
    --ckpt_path "$CKPT" --port "$PORT" --use_bf16 \
  > "$OUT_ROOT/${LABEL}_server.log" 2>&1 &
SERVER_PID=$!
echo "[$LABEL] server PID=$SERVER_PID"

# 2. Wait for server ready
for i in $(seq 1 60); do
  if grep -q "server listening" "$OUT_ROOT/${LABEL}_server.log" 2>/dev/null; then
    echo "[$LABEL] server listening"
    break
  fi
  sleep 5
done

# 3. Sequentially eval 4 suites
for SUITE in libero_spatial libero_object libero_goal libero_10; do
  OUT="$OUT_ROOT/${LABEL}_${SUITE}"
  mkdir -p "$OUT"
  echo "[$LABEL] === ${SUITE} begin === (trials=$TRIALS, render_gpu=$RENDER_GPU)"
  CUDA_VISIBLE_DEVICES=$RENDER_GPU \
    ENABLE_SKIPPING="${ENABLE_SKIPPING:-0}" \
    DYN_SKIP_CONFIG_PATH="${DYN_SKIP_CONFIG_PATH:-}" \
    bash examples/LIBERO/eval_files/eval_libero.sh \
    "$CKPT" "$SUITE" "$PORT" "$TRIALS" \
    > "$OUT/eval.log" 2>&1
  SR=$(grep -oE "Total success rate: [0-9.]+" "$OUT/eval.log" | tail -1)
  echo "[$LABEL] === ${SUITE} done: $SR ==="
done

# 4. Kill server
kill -9 $SERVER_PID 2>/dev/null
echo "[$LABEL] ALL SUITES DONE"
