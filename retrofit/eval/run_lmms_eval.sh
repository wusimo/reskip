#!/bin/bash
# Dispatch lmms-eval runs for base + retrofit on a curated set of VLM benchmarks.
# Usage:
#   bash run_lmms_eval.sh <base|retrofit|retrofit_skip> <gpu> <tasks_csv> [limit] [out_subdir]
#
# Examples:
#   bash run_lmms_eval.sh base      0 mmstar 20   smoke
#   bash run_lmms_eval.sh retrofit  0 mmbench_en_dev,mmstar,mmmu_val - full
#   bash run_lmms_eval.sh retrofit_skip 0 mmstar - dynskip

set -u
MODE="${1:?usage: <base|retrofit|retrofit_skip> <gpu> <tasks> [limit] [subdir]}"
GPU="${2:?gpu}"
TASKS="${3:?tasks}"
LIMIT="${4:--}"
SUBDIR="${5:-run}"

PROJECT=/home/user01/Minko/reskip2/reskip
MODELDIR="${MODELDIR:-/home/user01/Minko/models/Qwen3-VL-2B}"
STATE="${STATE:-$PROJECT/retrofit/outputs/H_r256_5k/retrofit_attnres_state.pt}"
OUT="${OUT_ROOT:-$PROJECT/retrofit/outputs/lmms_eval}/$SUBDIR/$MODE"
mkdir -p "$OUT"

case "$MODE" in
  base)
    MODEL=qwen3_vl
    MARGS="pretrained=$MODELDIR,max_pixels=1605632,min_pixels=200704"
    ;;
  retrofit)
    MODEL=qwen3_vl_retrofit
    MARGS="pretrained=$MODELDIR,retrofit_state_path=$STATE,max_pixels=1605632,min_pixels=200704"
    ;;
  retrofit_skip)
    MODEL=qwen3_vl_retrofit
    MARGS="pretrained=$MODELDIR,retrofit_state_path=$STATE,dynamic_skip=True,dyn_quantile=0.95,dyn_max_skips=1,dyn_positions=4|6|11,max_pixels=1605632,min_pixels=200704"
    ;;
  *)
    echo "bad mode: $MODE"; exit 1
    ;;
esac

EXTRA=""
if [ "$LIMIT" != "-" ] && [ -n "$LIMIT" ]; then
  EXTRA="$EXTRA --limit $LIMIT"
fi

cd $PROJECT
LOG=$OUT/eval.log
echo "[run_lmms_eval] MODE=$MODE GPU=$GPU TASKS=$TASKS LIMIT=$LIMIT OUT=$OUT" | tee -a "$LOG"
echo "[run_lmms_eval] model_args=$MARGS" | tee -a "$LOG"

CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$PROJECT/retrofit:$PROJECT/retrofit/eval \
  python -c "
import sys
sys.path.insert(0, '$PROJECT/retrofit')       # core module qwen3vl_attnres_retrofit.py
sys.path.insert(0, '$PROJECT/retrofit/eval')  # lmms_eval_retrofit.py plugin
import lmms_eval_retrofit  # register plugin
import lmms_eval.__main__ as m
sys.argv = ['lmms_eval',
  '--model', '$MODEL',
  '--model_args', '$MARGS',
  '--tasks', '$TASKS',
  '--batch_size', '1',
  '--output_path', '$OUT',
] + '$EXTRA'.split()
m.cli_evaluate()
" 2>&1 | tee -a "$LOG"

echo "[run_lmms_eval] done; log=$LOG"
