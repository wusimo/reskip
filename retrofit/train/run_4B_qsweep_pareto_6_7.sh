#!/bin/bash
# GPU 6-7: 4B server (port 7042) + q=0.30 then q=0.95 sequential spatial.
# Complements the q=0.85/q=0.99 runs on GPU 0-1 + q=0.50 retry + q=0.70.

set -u
STARVLA=/home/user01/Minko/reskip2/reskip/starVLA
RETROFIT=/home/user01/Minko/reskip2/reskip/retrofit
OUT_ROOT=$RETROFIT/outputs/libero_eval_full
DYN_CFGS=$RETROFIT/outputs/dyn_skip_configs
CKPT=$STARVLA/results/Checkpoints/libero_pathB_4B_L4_v3_30k/final_model/pytorch_model.pt

PORT=7042
SRV_LOG=$OUT_ROOT/4B_qsweep_gpu67_server.log

echo "[4B-pareto-67] starting 4B server on GPU 6 port $PORT"
CUDA_VISIBLE_DEVICES=6 WANDB_MODE=offline PYTHONPATH=$STARVLA \
  nohup python $STARVLA/deployment/model_server/server_policy.py \
    --ckpt_path "$CKPT" --port $PORT --use_bf16 > $SRV_LOG 2>&1 &
SRV_PID=$!
for i in $(seq 1 60); do
  sleep 10
  grep -q "server listening" $SRV_LOG && break
done
grep -q "server listening" $SRV_LOG || { echo "[4B-pareto-67] server failed"; exit 1; }
echo "[4B-pareto-67] server ready PID=$SRV_PID"

for Q in 0.30 0.95; do
  QLBL=$(echo $Q | tr -d .)
  CFG=$DYN_CFGS/pathB_4B_L4_v3_30k_sim_q${QLBL}_b1b2.json
  LBL=pathB_4B_L4_v3_30k_qsweep_simq${QLBL}_b1b2
  OUT=$OUT_ROOT/${LBL}_libero_spatial
  mkdir -p "$OUT"
  echo "[4B-pareto-67] === q=$Q begin $(date) ==="
  TSTART=$(date +%s)
  CUDA_VISIBLE_DEVICES=7 EGL_DEVICE_ID=7 \
    DYN_SKIP_CONFIG_PATH="$CFG" \
    bash $STARVLA/examples/LIBERO/eval_files/eval_libero_skip.sh \
      "$CKPT" libero_spatial $PORT 50 > "$OUT/eval.log" 2>&1
  TEND=$(date +%s)
  SR=$(grep -oE "Total success rate: [0-9.]+" "$OUT/eval.log" | tail -1)
  echo "[4B-pareto-67] q=$Q done $SR elapsed=$((TEND-TSTART))s" | tee -a $OUT_ROOT/4B_pareto_gpu67_driver.log
done

kill -KILL $SRV_PID 2>/dev/null
echo "[4B-pareto-67] done"
