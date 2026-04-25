#!/bin/bash
# Skip-enabled 4-suite LIBERO eval. Mirrors run_full_eval.sh but calls
# eval_libero_skip.sh (which adds --args.enable-skipping and optional
# --args.dyn-skip-config-path to eval_libero.py).
#
# Usage:
#   DYN_SKIP_CONFIG_PATH=/abs/path/to/dyn_skip_cfg.json \
#   bash run_full_eval_skip.sh <label> <ckpt> <server_gpu> <render_gpu> <port> [trials]
#
# The policy server is unchanged; skip config is threaded per-request
# through vla_input.

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

# Capture wall-clock for speed comparison with the no-skip run.
TSTART=$(date +%s)

for SUITE in libero_spatial libero_object libero_goal libero_10; do
  OUT="$OUT_ROOT/${LABEL}_${SUITE}"
  mkdir -p "$OUT"
  echo "[$LABEL] === ${SUITE} begin === (trials=$TRIALS, render_gpu=$RENDER_GPU)"
  SS=$(date +%s)
  CUDA_VISIBLE_DEVICES=$RENDER_GPU \
    DYN_SKIP_CONFIG_PATH="${DYN_SKIP_CONFIG_PATH:-}" \
    bash examples/LIBERO/eval_files/eval_libero_skip.sh \
    "$CKPT" "$SUITE" "$PORT" "$TRIALS" \
    > "$OUT/eval.log" 2>&1
  SS_END=$(date +%s)
  SR=$(grep -oE "Total success rate: [0-9.]+" "$OUT/eval.log" | tail -1)
  echo "[$LABEL] === ${SUITE} done: $SR  (elapsed $((SS_END-SS))s) ==="
done

TEND=$(date +%s)
echo "[$LABEL] TOTAL elapsed: $((TEND-TSTART))s"

kill -9 $SERVER_PID 2>/dev/null
echo "[$LABEL] ALL SUITES DONE"
