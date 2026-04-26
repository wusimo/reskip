#!/bin/bash
# Generic 4-suite continuation launcher for a single GPU pair.
#
# Usage:
#   bash run_continuation_4suite.sh <wait-for-pid> \
#        <server-gpu> <eval-gpu> <port> <ckpt> <dyn-cfg> <label>
#
# Waits for <wait-for-pid> to exit (current server on this pair), then
# starts a fresh policy server on <server-gpu>/<port> and runs
# libero_object → libero_goal → libero_10 eval on <eval-gpu> using
# <dyn-cfg>.  Results land in $OUT_ROOT/<label>_libero_{object,goal,10}/.

set -u
WAIT_PID="${1:?pid}"
SRV_GPU="${2:?server gpu}"
EVAL_GPU="${3:?eval gpu}"
PORT="${4:?port}"
CKPT="${5:?ckpt}"
CFG="${6:?dyn cfg}"
LBL="${7:?label}"

STARVLA=/home/user01/Minko/reskip2/reskip/starVLA
RETROFIT=/home/user01/Minko/reskip2/reskip/retrofit
OUT_ROOT=$RETROFIT/outputs/libero_eval_full
SRV_LOG=$OUT_ROOT/${LBL}_server.log
DRV_LOG=$OUT_ROOT/${LBL}_driver.log

echo "[cont-$LBL] waiting for pid=$WAIT_PID to exit" | tee -a $DRV_LOG
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
echo "[cont-$LBL] pid $WAIT_PID exited at $(date); starting new server on GPU $SRV_GPU port $PORT" | tee -a $DRV_LOG

CUDA_VISIBLE_DEVICES=$SRV_GPU WANDB_MODE=offline PYTHONPATH=$STARVLA \
  nohup python $STARVLA/deployment/model_server/server_policy.py \
    --ckpt_path "$CKPT" --port $PORT --use_bf16 > $SRV_LOG 2>&1 &
SRV_PID=$!
for i in $(seq 1 60); do
  sleep 10
  grep -q "server listening" $SRV_LOG && break
done
grep -q "server listening" $SRV_LOG || { echo "[cont-$LBL] server failed to start" | tee -a $DRV_LOG; exit 1; }
echo "[cont-$LBL] server PID=$SRV_PID listening" | tee -a $DRV_LOG

for SUITE in libero_object libero_goal libero_10; do
  OUT=$OUT_ROOT/${LBL}_${SUITE}
  mkdir -p "$OUT"
  echo "[cont-$LBL] === $SUITE begin $(date) ===" | tee -a $DRV_LOG
  TSTART=$(date +%s)
  CUDA_VISIBLE_DEVICES=$EVAL_GPU EGL_DEVICE_ID=$EVAL_GPU \
    DYN_SKIP_CONFIG_PATH="$CFG" \
    bash $STARVLA/examples/LIBERO/eval_files/eval_libero_skip.sh \
      "$CKPT" $SUITE $PORT 50 > "$OUT/eval.log" 2>&1
  TEND=$(date +%s)
  SR=$(grep -oE "Total success rate: [0-9.]+" "$OUT/eval.log" | tail -1)
  echo "[cont-$LBL] $SUITE done $SR elapsed=$((TEND-TSTART))s" | tee -a $DRV_LOG
done

kill -KILL $SRV_PID 2>/dev/null
echo "[cont-$LBL] all 3 suites done; server shut down" | tee -a $DRV_LOG
