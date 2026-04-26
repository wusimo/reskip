#!/bin/bash
# Batch-eval the 8 H-family retrofit variants.
# Usage: bash run_h_family_evals.sh [lambada|mmbench|all]
# Default: all  (LAMBADA + HellaSwag via eval/eval_qwen3vl_attnres_retrofit.py, MMBench via eval/eval_mmbench.py)
#
# Run from anywhere; we cd to retrofit/ so relative outputs/ and eval/ paths resolve.
set -u
cd "$(dirname "$0")/.."  # retrofit/eval/ → retrofit/

MODE="${1:-all}"
RUNS=(H_r64 H_r32 H_r256_vlm80 H_r256_20k H_r256_5k H_r128_vlm80 H_r64_vlm80 H_r256_slowramp)
# Map to GPUs (same as training layout) — change if a GPU is still busy
GPUS=(0 1 2 3 4 5 6 7)

mkdir -p outputs/h_family_evals

# Build list of pending (state exists, eval missing)
PENDING_IDX=()
for i in "${!RUNS[@]}"; do
  r="${RUNS[$i]}"
  if [ ! -f "outputs/$r/retrofit_attnres_state.pt" ]; then
    echo "  skip $r (no state)"
    continue
  fi
  PENDING_IDX+=("$i")
done

run_lambada() {
  local i="$1"; local r="${RUNS[$i]}"; local g="${GPUS[$i]}"
  local out="outputs/h_family_evals/${r}_lambada_hs.log"
  if [ -f "$out" ] && grep -q "HellaSwag acc_norm" "$out" 2>/dev/null; then
    echo "[lam] $r already done -> skip"; return
  fi
  echo "[lam] launching $r on cuda:$g -> $out"
  nohup python eval/eval_qwen3vl_attnres_retrofit.py \
    --model-type trained \
    --state-path "outputs/$r/retrofit_attnres_state.pt" \
    --lambada-n 500 --hellaswag-n 500 \
    --gpu "$g" --label "$r" \
    > "$out" 2>&1 &
}

run_mmbench() {
  local i="$1"; local r="${RUNS[$i]}"; local g="${GPUS[$i]}"
  local out="outputs/h_family_evals/${r}_mmbench.log"
  if [ -f "$out" ] && grep -q "accuracy:" "$out" 2>/dev/null; then
    echo "[mmb] $r already done -> skip"; return
  fi
  echo "[mmb] launching $r on cuda:$g -> $out"
  nohup python eval/eval_mmbench.py \
    --model-type trained \
    --state-path "outputs/$r/retrofit_attnres_state.pt" \
    --n 300 --gpu "$g" --label "$r" \
    > "$out" 2>&1 &
}

case "$MODE" in
  lambada)
    for i in "${PENDING_IDX[@]}"; do run_lambada "$i"; done
    ;;
  mmbench)
    for i in "${PENDING_IDX[@]}"; do run_mmbench "$i"; done
    ;;
  all)
    for i in "${PENDING_IDX[@]}"; do run_lambada "$i"; done
    wait
    echo "[all] LAMBADA+HellaSwag done, starting MMBench"
    for i in "${PENDING_IDX[@]}"; do run_mmbench "$i"; done
    wait
    echo "[all] all evals done"
    ;;
  *)
    echo "Unknown mode: $MODE"; exit 1
    ;;
esac

wait
echo "[${MODE}] batch finished"

# Summarise
echo
echo "=== Summary: H-family evals ==="
printf "%-22s  %-8s  %-10s  %-8s\n" "run" "LAMBADA" "HellaSwag" "MMBench"
for r in "${RUNS[@]}"; do
  lam=$(grep -h "LAMBADA acc:" "outputs/h_family_evals/${r}_lambada_hs.log" 2>/dev/null | tail -1 | awk '{print $3}')
  hs=$(grep -h "HellaSwag acc_norm:" "outputs/h_family_evals/${r}_lambada_hs.log" 2>/dev/null | tail -1 | awk '{print $3}')
  mmb=$(grep -h "accuracy:" "outputs/h_family_evals/${r}_mmbench.log" 2>/dev/null | tail -1 | awk '{print $4}')
  printf "%-22s  %-8s  %-10s  %-8s\n" "$r" "${lam:-N/A}" "${hs:-N/A}" "${mmb:-N/A}"
done
