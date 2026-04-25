#!/bin/bash
# Generic pair launcher: start a policy server, run N (suite, config) pairs
# sequentially on it, then shut server down.
#
# Usage:
#   bash run_pair_multisuite.sh <server_gpu> <eval_gpu> <port> <ckpt> \
#        <label> <suite1>:<cfg1> [<suite2>:<cfg2> ...]

set -u
SRV_GPU="${1:?server gpu}"
EVAL_GPU="${2:?eval gpu}"
PORT="${3:?port}"
CKPT="${4:?ckpt}"
LBL="${5:?label}"
shift 5

STARVLA=/home/user01/Minko/reskip2/reskip/starVLA
OUT_ROOT=/home/user01/Minko/reskip2/reskip/retrofit/outputs/libero_eval_full
SRV_LOG=$OUT_ROOT/${LBL}_server.log
DRV_LOG=$OUT_ROOT/${LBL}_driver.log

echo "[pair-$LBL] start $(date) GPU $SRV_GPU/$EVAL_GPU port $PORT" | tee -a $DRV_LOG
CUDA_VISIBLE_DEVICES=$SRV_GPU WANDB_MODE=offline PYTHONPATH=$STARVLA \
  nohup python $STARVLA/deployment/model_server/server_policy.py \
    --ckpt_path "$CKPT" --port $PORT --use_bf16 > $SRV_LOG 2>&1 &
SRV_PID=$!
for i in $(seq 1 60); do
  sleep 10
  grep -q "server listening" $SRV_LOG && break
done
grep -q "server listening" $SRV_LOG || { echo "[pair-$LBL] server failed" | tee -a $DRV_LOG; exit 1; }
echo "[pair-$LBL] server PID=$SRV_PID listening" | tee -a $DRV_LOG

for arg in "$@"; do
  SUITE="${arg%%:*}"
  CFG="${arg#*:}"
  # Distinct dir per (suite, cfg) so the same suite can run twice with different
  # configs on one launcher without clobbering eval.log.
  CFGTAG=$(basename "$CFG" .json)
  OUT=$OUT_ROOT/${LBL}_${SUITE}_${CFGTAG}
  mkdir -p "$OUT"
  echo "[pair-$LBL] === $SUITE begin $(date) (cfg=$CFG) ===" | tee -a $DRV_LOG
  TSTART=$(date +%s)
  CUDA_VISIBLE_DEVICES=$EVAL_GPU EGL_DEVICE_ID=$EVAL_GPU \
    DYN_SKIP_CONFIG_PATH="$CFG" \
    bash $STARVLA/examples/LIBERO/eval_files/eval_libero_skip.sh \
      "$CKPT" $SUITE $PORT 50 > "$OUT/eval.log" 2>&1
  TEND=$(date +%s)
  SR=$(grep -oE "Total success rate: [0-9.]+" "$OUT/eval.log" | tail -1)
  echo "[pair-$LBL] $SUITE done $SR elapsed=$((TEND-TSTART))s" | tee -a $DRV_LOG
done

kill -KILL $SRV_PID 2>/dev/null
echo "[pair-$LBL] all done $(date)" | tee -a $DRV_LOG
