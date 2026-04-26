#!/bin/bash
# Phase B block-partition mini-ablation for NeurIPS2026 retrofit paper.
#
# Citation background: Chen et al. (arxiv 2603.15031, Kimi Team 2026) Figure 6
# swept block size S ∈ {2,4,8,16,32} on a 32-layer from-scratch pretrain
# and found S=2,4,8 all land near val-loss 1.746 (Full AttnRes = 1.737,
# PreNorm = 1.766). We verify this trend holds under the retrofit setting
# (frozen base + γ-curriculum + 5k step KD) on Qwen3-VL-2B (28 layers).
#
# Cells:
#   L=1  (28 blocks, finest) — GPU 0
#   L=2  (14 blocks) — already exists as outputs/H_r256_5k, not re-run
#   L=4  (7  blocks, coarser) — GPU 1
#
# All other hyperparameters match H_r256_5k exactly (retrofit.md §3).

set -u
PROJECT=/home/user01/Minko/reskip2/reskip
PY=/home/user01/Minko/reskip2/.venv/bin/python
MODEL=/home/user01/Minko/models/Qwen3-VL-2B
OUT_ROOT=$PROJECT/retrofit/outputs/block_ablation
mkdir -p "$OUT_ROOT"

cd "$PROJECT"

run_cell() {
  local label="$1"; local num_blocks="$2"; local gpu="$3"
  local outdir="$OUT_ROOT/${label}_r256_5k"
  mkdir -p "$outdir"
  local log="$outdir/run.log"
  echo "[$(date)] launching $label (num_blocks=$num_blocks, gpu=$gpu) -> $outdir" | tee -a "$OUT_ROOT/launch.log"
  nohup $PY retrofit/train/train_qwen3vl_attnres_retrofit.py \
    --model-path "$MODEL" \
    --num-blocks "$num_blocks" \
    --adapter-rank 256 \
    --steps 5000 \
    --gamma-schedule --gamma-start 0 --gamma-end 1 --gamma-ramp-frac 0.3 \
    --p-multimodal 0.5 \
    --kl-weight 1.0 \
    --entropy-weight 0.02 \
    --output-dir "$outdir" \
    --gpu "$gpu" \
    > "$log" 2>&1 &
  echo "  pid=$!" | tee -a "$OUT_ROOT/launch.log"
}

case "${1:-all}" in
  L1) run_cell L1_2B 28 "${2:-0}" ;;
  L4) run_cell L4_2B 7  "${2:-1}" ;;
  all)
    run_cell L1_2B 28 0
    run_cell L4_2B 7  1
    ;;
  *) echo "usage: $0 [L1|L4|all] [gpu]"; exit 1 ;;
esac

wait
echo "[$(date)] ALL CELLS DONE" | tee -a "$OUT_ROOT/launch.log"
