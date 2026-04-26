#!/bin/bash
# Wait for a training run's final ckpt to appear, then launch 2 parallel LIBERO
# eval groups covering all 4 suites.
#
# Usage:
#   bash wait_and_eval.sh <label> <ckpt_dir> <gpu_server_A> <gpu_render_A> <port_A> \
#                         <gpu_server_B> <gpu_render_B> <port_B>
# Example (Path 0 clean-base, GPUs 0-3):
#   bash wait_and_eval.sh path0_4B_cleanbase \
#       .../libero_path0_4B_cleanbase_30k/final_model \
#       0 1 5900 2 3 5910
set -u
LABEL="${1:?label}"
CKPT_DIR="${2:?ckpt dir containing pytorch_model.pt}"
GPU_A_S="${3:?}"; GPU_A_R="${4:?}"; PORT_A="${5:?}"
GPU_B_S="${6:?}"; GPU_B_R="${7:?}"; PORT_B="${8:?}"

STARVLA=/home/user01/Minko/reskip2/reskip/starVLA
OUT_ROOT=/home/user01/Minko/reskip2/reskip/retrofit/outputs/libero_eval_full
CKPT="$CKPT_DIR/pytorch_model.pt"

echo "[wait_and_eval:$LABEL] waiting for $CKPT"
while [ ! -f "$CKPT" ]; do sleep 60; done
echo "[wait_and_eval:$LABEL] ckpt appeared; sleeping 30s for write to finalize"
sleep 30

echo "[wait_and_eval:$LABEL] launching grp1 spatial+object on GPUs $GPU_A_S+$GPU_A_R port $PORT_A"
bash "$STARVLA/examples/LIBERO/eval_files/run_suites_eval.sh" \
    "${LABEL}_grp1" "$CKPT" "$GPU_A_S" "$GPU_A_R" "$PORT_A" \
    libero_spatial,libero_object 50 \
    > "$OUT_ROOT/${LABEL}_grp1_driver.log" 2>&1 &
G1=$!

echo "[wait_and_eval:$LABEL] launching grp2 goal+10 on GPUs $GPU_B_S+$GPU_B_R port $PORT_B"
bash "$STARVLA/examples/LIBERO/eval_files/run_suites_eval.sh" \
    "${LABEL}_grp2" "$CKPT" "$GPU_B_S" "$GPU_B_R" "$PORT_B" \
    libero_goal,libero_10 50 \
    > "$OUT_ROOT/${LABEL}_grp2_driver.log" 2>&1 &
G2=$!

wait $G1 $G2
echo "[wait_and_eval:$LABEL] ALL DONE"
