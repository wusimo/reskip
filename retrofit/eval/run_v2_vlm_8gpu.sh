#!/bin/bash
# Parallel v2 VLM eval across 8 GPUs:
#   GPU 0 — v2_2B ai2d
#   GPU 1 — v2_2B mmbench_en_dev
#   GPU 2 — v2_2B mmmu_val
#   GPU 3 — v2_2B ocrbench
#   GPU 4 — v2_2B realworldqa
#   GPU 5 — v2_4B ai2d        → then v2_4B ocrbench
#   GPU 6 — v2_4B mmbench_en_dev → then v2_4B realworldqa
#   GPU 7 — v2_4B mmmu_val
#
# Wallclock = max(2B single-task, 4B two-task chain).

set -u
PROJECT=/home/user01/Minko/reskip2/reskip
OUT=$PROJECT/retrofit/outputs/lmms_eval_v2
mkdir -p "$OUT"
M2B=/home/user01/Minko/models/Qwen3-VL-2B
M4B=/home/user01/Minko/models/Qwen3-VL-4B
S2B=$PROJECT/retrofit/outputs/H_r256_5k_v2_2B/retrofit_attnres_state.pt
S4B=$PROJECT/retrofit/outputs/H_4B_r256_5k_v2/retrofit_attnres_state.pt

cd "$PROJECT"

launch() {
  local gpu="$1" cell="$2" mdir="$3" state="$4" task="$5"
  local sub="${cell}_${task}"
  local log="$OUT/${sub}_driver.log"
  echo "[$(date +%H:%M:%S)] gpu $gpu → $cell / $task" | tee -a "$OUT/batch.log"
  OUT_ROOT="$OUT" MODELDIR="$mdir" STATE="$state" \
    bash retrofit/eval/run_lmms_eval.sh retrofit "$gpu" "$task" - "$sub" \
    > "$log" 2>&1
  echo "[$(date +%H:%M:%S)] gpu $gpu ← $cell / $task (done)" | tee -a "$OUT/batch.log"
}

# v2_2B: 5 tasks, each on its own GPU
launch 0 v2_2B "$M2B" "$S2B" ai2d &
launch 1 v2_2B "$M2B" "$S2B" mmbench_en_dev &
launch 2 v2_2B "$M2B" "$S2B" mmmu_val &
launch 3 v2_2B "$M2B" "$S2B" ocrbench &
launch 4 v2_2B "$M2B" "$S2B" realworldqa &

# v2_4B: 5 tasks on 3 GPUs; GPUs 5, 6 chain 2 tasks each
(
  launch 5 v2_4B "$M4B" "$S4B" ai2d
  launch 5 v2_4B "$M4B" "$S4B" ocrbench
) &
(
  launch 6 v2_4B "$M4B" "$S4B" mmbench_en_dev
  launch 6 v2_4B "$M4B" "$S4B" realworldqa
) &
launch 7 v2_4B "$M4B" "$S4B" mmmu_val &

wait
echo "[$(date +%H:%M:%S)] ALL v2 VLM evals finished" | tee -a "$OUT/batch.log"
