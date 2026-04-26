#!/bin/bash
# Pareto tail: 2B q=0.95 and q=0.99 sim-calib libero_spatial on GPUs 2-3.
# Sequential on one 2B server (port 7040). Expected: q=0.95 in [0.80, 0.96],
# q=0.99 near Method A's 0.964 (or higher).

set -u
STARVLA=/home/user01/Minko/reskip2/reskip/starVLA
RETROFIT=/home/user01/Minko/reskip2/reskip/retrofit
OUT_ROOT=$RETROFIT/outputs/libero_eval_full
DYN_CFGS=$RETROFIT/outputs/dyn_skip_configs
CKPT=$STARVLA/results/Checkpoints/libero_pathB_2B_L4_v3_30k/final_model/pytorch_model.pt

PORT=7040
SRV_LOG=$OUT_ROOT/2B_qsweep_q095_q099_server.log

echo "[qsweep-tail] starting 2B policy server on GPU 2 port $PORT"
CUDA_VISIBLE_DEVICES=2 WANDB_MODE=offline PYTHONPATH=$STARVLA \
  nohup python $STARVLA/deployment/model_server/server_policy.py \
    --ckpt_path "$CKPT" --port $PORT --use_bf16 > $SRV_LOG 2>&1 &
SRV_PID=$!
echo "[qsweep-tail] server PID=$SRV_PID"
for i in $(seq 1 60); do
  sleep 10
  grep -q "server listening" $SRV_LOG && break
done
grep -q "server listening" $SRV_LOG || { echo "[qsweep-tail] server failed"; exit 1; }
echo "[qsweep-tail] server ready"

for Q in 0.95 0.99; do
  QLBL=$(echo $Q | tr -d .)
  CFG=$DYN_CFGS/pathB_2B_L4_v3_30k_sim_q${QLBL}.json
  LBL=pathB_2B_L4_v3_30k_qsweep_simq${QLBL}
  OUT=$OUT_ROOT/${LBL}_libero_spatial
  mkdir -p "$OUT"
  echo "[qsweep-tail] === q=$Q begin $(date) ==="
  TSTART=$(date +%s)
  CUDA_VISIBLE_DEVICES=3 EGL_DEVICE_ID=3 \
    DYN_SKIP_CONFIG_PATH="$CFG" \
    bash $STARVLA/examples/LIBERO/eval_files/eval_libero_skip.sh \
      "$CKPT" libero_spatial $PORT 50 > "$OUT/eval.log" 2>&1
  TEND=$(date +%s)
  SR=$(grep -oE "Total success rate: [0-9.]+" "$OUT/eval.log" | tail -1)
  echo "[qsweep-tail] q=$Q done $SR elapsed=$((TEND-TSTART))s" | tee -a $OUT_ROOT/qsweep_tail_driver.log
done

kill -KILL $SRV_PID 2>/dev/null
echo "[qsweep-tail] server shut down"
