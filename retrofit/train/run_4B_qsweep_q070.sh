#!/bin/bash
# 4B q=0.70 spatial point for Pareto (parallel with ongoing 4B q=0.50 retry).
# GPU 6 = 4B server port 7039, GPU 7 = eval client.

set -u
STARVLA=/home/user01/Minko/reskip2/reskip/starVLA
RETROFIT=/home/user01/Minko/reskip2/reskip/retrofit
OUT_ROOT=$RETROFIT/outputs/libero_eval_full
DYN_CFGS=$RETROFIT/outputs/dyn_skip_configs

PORT=7039
LBL=pathB_4B_L4_v3_30k_qsweep_simq070_b1b2
OUT=$OUT_ROOT/${LBL}_libero_spatial
mkdir -p "$OUT"

echo "[4B-q070] starting policy server on GPU 6 port $PORT"
CUDA_VISIBLE_DEVICES=6 WANDB_MODE=offline PYTHONPATH=$STARVLA \
  nohup python $STARVLA/deployment/model_server/server_policy.py \
    --ckpt_path $STARVLA/results/Checkpoints/libero_pathB_4B_L4_v3_30k/final_model/pytorch_model.pt \
    --port $PORT --use_bf16 > $OUT_ROOT/${LBL}_server.log 2>&1 &
SRV_PID=$!
echo "[4B-q070] server PID=$SRV_PID"

for i in $(seq 1 60); do
  sleep 10
  grep -q "server listening" $OUT_ROOT/${LBL}_server.log && break
done
grep -q "server listening" $OUT_ROOT/${LBL}_server.log || { echo "[4B-q070] server failed to start"; exit 1; }
echo "[4B-q070] server ready"

CFG=$DYN_CFGS/pathB_4B_L4_v3_30k_sim_q070_b1b2.json
CKPT=$STARVLA/results/Checkpoints/libero_pathB_4B_L4_v3_30k/final_model/pytorch_model.pt

TSTART=$(date +%s)
CUDA_VISIBLE_DEVICES=7 EGL_DEVICE_ID=7 \
  DYN_SKIP_CONFIG_PATH="$CFG" \
  bash $STARVLA/examples/LIBERO/eval_files/eval_libero_skip.sh \
    "$CKPT" libero_spatial $PORT 50 > "$OUT/eval.log" 2>&1
TEND=$(date +%s)
SR=$(grep -oE "Total success rate: [0-9.]+" "$OUT/eval.log" | tail -1)
echo "[4B-q070] done $SR elapsed=$((TEND-TSTART))s" | tee -a $OUT_ROOT/${LBL}_driver.log

kill -KILL $SRV_PID 2>/dev/null
echo "[4B-q070] server shut down"
