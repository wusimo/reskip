#!/bin/bash
# Parallel q=0.85 launcher on GPUs 2-3 port 7038.  Runs alongside the
# orchestrator's Phase E on GPUs 0-1 (currently doing q=0.70).

set -u
STARVLA=/home/user01/Minko/reskip2/reskip/starVLA
RETROFIT=/home/user01/Minko/reskip2/reskip/retrofit
OUT_ROOT=$RETROFIT/outputs/libero_eval_full
DYN_CFGS=$RETROFIT/outputs/dyn_skip_configs

PORT=7038
LBL=pathB_2B_L4_v3_30k_qsweep_simq085
OUT=$OUT_ROOT/${LBL}_libero_spatial
mkdir -p "$OUT"

echo "[qsweep-q085] starting policy server on GPU 2 port $PORT"
CUDA_VISIBLE_DEVICES=2 WANDB_MODE=offline PYTHONPATH=$STARVLA \
  nohup python $STARVLA/deployment/model_server/server_policy.py \
    --ckpt_path $STARVLA/results/Checkpoints/libero_pathB_2B_L4_v3_30k/final_model/pytorch_model.pt \
    --port $PORT --use_bf16 > $OUT_ROOT/${LBL}_server.log 2>&1 &
SRV_PID=$!
echo "[qsweep-q085] server PID=$SRV_PID"

# wait for server ready
for i in $(seq 1 60); do
  sleep 10
  grep -q "server listening" $OUT_ROOT/${LBL}_server.log && break
done
grep -q "server listening" $OUT_ROOT/${LBL}_server.log || { echo "[qsweep-q085] server failed to start"; exit 1; }
echo "[qsweep-q085] server ready"

CFG=$DYN_CFGS/pathB_2B_L4_v3_30k_sim_q085.json
CKPT=$STARVLA/results/Checkpoints/libero_pathB_2B_L4_v3_30k/final_model/pytorch_model.pt

TSTART=$(date +%s)
CUDA_VISIBLE_DEVICES=3 EGL_DEVICE_ID=3 \
  DYN_SKIP_CONFIG_PATH="$CFG" \
  bash $STARVLA/examples/LIBERO/eval_files/eval_libero_skip.sh \
    "$CKPT" libero_spatial $PORT 50 > "$OUT/eval.log" 2>&1
TEND=$(date +%s)
SR=$(grep -oE "Total success rate: [0-9.]+" "$OUT/eval.log" | tail -1)
echo "[qsweep-q085] done  $SR  elapsed=$((TEND-TSTART))s" | tee -a $OUT_ROOT/${LBL}_driver.log

# shut down server
kill -KILL $SRV_PID 2>/dev/null
echo "[qsweep-q085] server shut down"
