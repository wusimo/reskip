#!/bin/bash
# MMStar via lmms-eval on 6 configs (base / v1 / v2 × 2B / 4B).
# Replaces the buggy direct eval_mmstar.py whose accuracy is stuck at 0.278
# across all models (images likely not reaching the model).

set -u
PROJECT=/home/user01/Minko/reskip2/reskip
LMMS=$PROJECT/retrofit/eval/run_lmms_eval.sh

MODEL2B=/home/user01/Minko/models/Qwen3-VL-2B
MODEL4B=/home/user01/Minko/models/Qwen3-VL-4B
S_V1_2B=$PROJECT/retrofit/outputs/H_r256_5k/retrofit_attnres_state.pt
S_V2_2B=$PROJECT/retrofit/outputs/H_r256_5k_v2_2B/retrofit_attnres_state.pt
S_V1_4B=$PROJECT/retrofit/outputs/H_4B_r256_5k/retrofit_attnres_state.pt
S_V2_4B=$PROJECT/retrofit/outputs/H_4B_r256_5k_v2/retrofit_attnres_state.pt

# Each cell: (label, mode, gpu, modeldir, state)
cells=(
  "base_2B|base|0|$MODEL2B|"
  "v1_2B|retrofit|1|$MODEL2B|$S_V1_2B"
  "v2_2B|retrofit|2|$MODEL2B|$S_V2_2B"
  "base_4B|base|3|$MODEL4B|"
  "v1_4B|retrofit|4|$MODEL4B|$S_V1_4B"
  "v2_4B|retrofit|5|$MODEL4B|$S_V2_4B"
)

for c in "${cells[@]}"; do
  IFS='|' read -r label mode gpu modeldir state <<< "$c"
  echo "[$(date)] $label  mode=$mode  gpu=$gpu  modeldir=$modeldir  state=$state"
  export MODELDIR="$modeldir"
  export STATE="$state"
  export OUT_ROOT=$PROJECT/retrofit/outputs/lmms_eval_v2
  nohup bash "$LMMS" "$mode" "$gpu" mmstar - "$label" \
    > $PROJECT/retrofit/outputs/lmms_eval_v2/${label}_driver.log 2>&1 &
done
wait
echo "[$(date)] all 6 lmms-eval MMStar done"
