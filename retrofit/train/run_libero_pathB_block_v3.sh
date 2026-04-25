#!/bin/bash
# LIBERO Path B 30K from block-ablation winners (2B_L4 + 4B_L4), 4 GPUs each.
#
# Usage:  bash retrofit/train/run_libero_pathB_block_v3.sh [2b|4b|both]
#   default = both, launches two 4-GPU jobs concurrently (GPUs 0-3 / 4-7).
#
# Requires the host to be idle — GPU-holding scripts (e.g. gpu-zhanyong.py)
# must be stopped before launch.

set -u
WHICH="${1:-both}"
PROJECT=/home/user01/Minko/reskip2/reskip
SV=$PROJECT/starVLA
TMPL=$SV/results/Checkpoints/libero_pathB_4B_cleanbase_30k/run_libero_train_attnres_on.sh

S2B=$PROJECT/retrofit/outputs/block_v3/2B_L4_v3_10k/retrofit_attnres_state.pt
S4B=$PROJECT/retrofit/outputs/block_v3/4B_L4_v3_10k/retrofit_attnres_state.pt

LOGDIR=$SV/logs
mkdir -p "$LOGDIR"

launch_2b() {
  local rid=libero_pathB_2B_L4_v3_30k
  local log=$LOGDIR/${rid}_train.log
  cd "$SV"
  echo "[$(date +%H:%M:%S)] launch $rid on GPUs 0-3 → $log"
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
    WANDB_MODE=offline \
    N_BLOCKS=7 \
    PORT=29510 \
    BASE_VLM=playground/Pretrained_models/Qwen3-VL-2B-Instruct-Action \
    nohup bash "$TMPL" 4 30000 16 "$rid" "$S2B" > "$log" 2>&1 &
  echo "  pid=$!"
}

launch_4b() {
  local rid=libero_pathB_4B_L4_v3_30k
  local log=$LOGDIR/${rid}_train.log
  cd "$SV"
  echo "[$(date +%H:%M:%S)] launch $rid on GPUs 4-7 → $log"
  CUDA_VISIBLE_DEVICES=4,5,6,7 \
    WANDB_MODE=offline \
    N_BLOCKS=9 \
    PORT=29520 \
    BASE_VLM=playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action \
    nohup bash "$TMPL" 4 30000 16 "$rid" "$S4B" > "$log" 2>&1 &
  echo "  pid=$!"
}

case "$WHICH" in
  2b)   launch_2b ;;
  4b)   launch_4b ;;
  both) launch_2b; launch_4b ;;
  *) echo "usage: $0 [2b|4b|both]"; exit 1 ;;
esac

echo "[$(date +%H:%M:%S)] launched. Check logs in $LOGDIR/libero_pathB_{2B,4B}_L4_v3_30k_train.log"
