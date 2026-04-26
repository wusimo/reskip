#!/bin/bash
# MMStar 6-way compare: base / v1 / v2 on 2B and 4B.
# Targets the benchmark where v1 regressed -8pp on logic/math/science subcategories
# (per retrofit.md), so we can judge whether the v2 reasoning-mix fixes it.
#
# Usage: bash retrofit/eval/run_mmstar_v2_compare.sh
# Expects GPUs 0-5 free.

set -u
PROJECT=/home/user01/Minko/reskip2/reskip
PY=/home/user01/Minko/reskip2/.venv/bin/python
cd "$PROJECT"

OUT=$PROJECT/retrofit/outputs/mmstar_v2_compare
mkdir -p "$OUT"

N=500
MODEL2B=/home/user01/Minko/models/Qwen3-VL-2B
MODEL4B=/home/user01/Minko/models/Qwen3-VL-4B
STATE_V1_2B=$PROJECT/retrofit/outputs/H_r256_5k/retrofit_attnres_state.pt
STATE_V2_2B=$PROJECT/retrofit/outputs/H_r256_5k_v2_2B/retrofit_attnres_state.pt
STATE_V1_4B=$PROJECT/retrofit/outputs/H_4B_r256_5k/retrofit_attnres_state.pt
STATE_V2_4B=$PROJECT/retrofit/outputs/H_4B_r256_5k_v2/retrofit_attnres_state.pt

launch() {
  local label="$1"; local model_path="$2"; local model_type="$3"
  local state="$4"; local num_blocks="$5"; local gpu="$6"
  local args=(--model-type "$model_type" --model-path "$model_path"
              --n "$N" --gpu "$gpu" --label "$label")
  if [ "$model_type" = "trained" ]; then
    args+=(--state-path "$state" --num-blocks "$num_blocks")
  fi
  echo "[$(date)] launching $label on GPU $gpu -> $OUT/${label}.log"
  nohup $PY retrofit/eval/eval_mmstar.py "${args[@]}" \
    > "$OUT/${label}.log" 2>&1 &
}

launch base_2B "$MODEL2B" base    ""                0  0
launch v1_2B   "$MODEL2B" trained "$STATE_V1_2B"    14 1
launch v2_2B   "$MODEL2B" trained "$STATE_V2_2B"    14 2
launch base_4B "$MODEL4B" base    ""                0  3
launch v1_4B   "$MODEL4B" trained "$STATE_V1_4B"    18 4
launch v2_4B   "$MODEL4B" trained "$STATE_V2_4B"    18 5

wait
echo "[$(date)] all MMStar evals done"
echo
echo "=== SUMMARY ==="
for f in "$OUT"/*.log; do
  name=$(basename "$f" .log)
  acc=$(grep -E "MMStar acc|SUMMARY" "$f" 2>/dev/null | tail -1 | head -c 200)
  printf "%-12s  %s\n" "$name" "${acc}"
done
