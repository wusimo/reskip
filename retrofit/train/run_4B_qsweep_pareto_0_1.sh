#!/bin/bash
# GPU 0-1: 4B server (port 7041) + q=0.85 then q=0.99 sequential spatial.
# Complements the q=0.30/q=0.70/q=0.50 retry runs on other GPUs.

set -u
STARVLA=/home/user01/Minko/reskip2/reskip/starVLA
RETROFIT=/home/user01/Minko/reskip2/reskip/retrofit
OUT_ROOT=$RETROFIT/outputs/libero_eval_full
DYN_CFGS=$RETROFIT/outputs/dyn_skip_configs
CKPT=$STARVLA/results/Checkpoints/libero_pathB_4B_L4_v3_30k/final_model/pytorch_model.pt

PORT=7041
SRV_LOG=$OUT_ROOT/4B_qsweep_gpu01_server.log

echo "[4B-pareto-01] starting 4B server on GPU 0 port $PORT"
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline PYTHONPATH=$STARVLA \
  nohup python $STARVLA/deployment/model_server/server_policy.py \
    --ckpt_path "$CKPT" --port $PORT --use_bf16 > $SRV_LOG 2>&1 &
SRV_PID=$!
for i in $(seq 1 60); do
  sleep 10
  grep -q "server listening" $SRV_LOG && break
done
grep -q "server listening" $SRV_LOG || { echo "[4B-pareto-01] server failed"; exit 1; }
echo "[4B-pareto-01] server ready PID=$SRV_PID"

for Q in 0.85 0.99; do
  QLBL=$(echo $Q | tr -d .)
  CFG=$DYN_CFGS/pathB_4B_L4_v3_30k_sim_q${QLBL}_b1b2.json
  LBL=pathB_4B_L4_v3_30k_qsweep_simq${QLBL}_b1b2
  OUT=$OUT_ROOT/${LBL}_libero_spatial
  mkdir -p "$OUT"
  echo "[4B-pareto-01] === q=$Q begin $(date) ==="
  TSTART=$(date +%s)
  CUDA_VISIBLE_DEVICES=1 EGL_DEVICE_ID=1 \
    DYN_SKIP_CONFIG_PATH="$CFG" \
    bash $STARVLA/examples/LIBERO/eval_files/eval_libero_skip.sh \
      "$CKPT" libero_spatial $PORT 50 > "$OUT/eval.log" 2>&1
  TEND=$(date +%s)
  SR=$(grep -oE "Total success rate: [0-9.]+" "$OUT/eval.log" | tail -1)
  echo "[4B-pareto-01] q=$Q done $SR elapsed=$((TEND-TSTART))s" | tee -a $OUT_ROOT/4B_pareto_gpu01_driver.log
done

kill -KILL $SRV_PID 2>/dev/null
echo "[4B-pareto-01] done"
