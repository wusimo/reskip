#!/bin/bash
# Master orchestrator for paper Exp1/2/3 (4B reskip / q-sweep / VLM reskip).
# Runs after the currently-running 2B no-skip + 2B dynskip evals and the
# already-launched 4B no-skip eval.  GPUs 4-5 handle 4B server+render;
# GPUs 6-7 host the 2B calibration dump already in flight.
#
# Sequence:
#   Phase A: wait for 4B no-skip (4-5) + 2B calib dump (6-7) + 2B dynskip (2-3)
#   Phase B: run 4B calibration dump (reuse 4B server on GPU 4, client on GPU 5)
#   Phase C: post-process 4B dump  ->  sim-calib config
#   Phase D: 4B dynskip 4-suite eval (GPUs 4-5)                         << Exp 1
#   Phase E: 2B q-sweep on libero_spatial {q=0.30, 0.70, 0.85}          << Exp 2
#   Phase F: VLM-only reskip (LAMBADA + HellaSwag on retrofit ckpt)     << Exp 3
#
# All results get appended to retrofit/analysis/reskip_libero_results.md.

set -u

STARVLA=/home/user01/Minko/reskip2/reskip/starVLA
RETROFIT=/home/user01/Minko/reskip2/reskip/retrofit
OUT_ROOT=$RETROFIT/outputs/libero_eval_full
DYN_CFGS=$RETROFIT/outputs/dyn_skip_configs
RESULTS_MD=$RETROFIT/analysis/reskip_libero_results.md
mkdir -p "$OUT_ROOT" "$DYN_CFGS"

PIPE_LOG=$OUT_ROOT/exp_pipeline.log
exec > "$PIPE_LOG" 2>&1
echo "[pipeline] start $(date)"

# ------ helpers ---------------------------------------------------------
wait_pid() {
  local pid="$1" label="$2"
  while kill -0 "$pid" 2>/dev/null; do sleep 60; done
  echo "[pipeline] $label (pid=$pid) exited $(date)"
}
wait_grep() {
  # wait until a grep pattern appears in a file (up to max_sec seconds)
  local pat="$1" file="$2" max_sec="${3:-7200}"
  local t0 ddl
  t0=$(date +%s); ddl=$((t0+max_sec))
  while ! grep -q "$pat" "$file" 2>/dev/null; do
    sleep 30
    [ "$(date +%s)" -gt "$ddl" ] && { echo "[pipeline] TIMEOUT waiting for '$pat' in $file"; return 1; }
  done
  echo "[pipeline] matched '$pat' in $file $(date)"
}

# ------ Phase A: wait for upstream work --------------------------------
echo "[pipeline] Phase A: wait for upstream evals + calib dump"
# 4B no-skip driver — triggers 'ALL SUITES DONE' at end
wait_grep "ALL SUITES DONE" "$OUT_ROOT/4B_noskip_driver.log" 36000
# 2B calib dump driver — triggers '2B calib dump ALL DONE'
wait_grep "2B calib dump ALL DONE" "$OUT_ROOT/2B_calib_driver.log" 18000
# 2B dynskip driver — triggers 'ALL SUITES DONE' at end
wait_grep "ALL SUITES DONE" "$OUT_ROOT/pathB_2B_L4_v3_30k_dynskip_b1b4_q50_M2_driver.log" 36000
echo "[pipeline] Phase A done $(date)"

# shut down 2B calib server (port 7033, GPU 6) so it stops burning memory
pkill -9 -f "server_policy.py.*port 7033" 2>/dev/null
sleep 3

# ------ Phase B: 4B calibration dump -----------------------------------
echo "[pipeline] Phase B: 4B calibration dump $(date)"
# (Re-)use 4B server at port 7032 (still alive until run_full_eval.sh kills it).
# Actually the run_full_eval.sh kills its server at the end. So start a new 4B server on GPU 4.
CUDA_VISIBLE_DEVICES=4 WANDB_MODE=offline PYTHONPATH=$STARVLA \
  nohup python $STARVLA/deployment/model_server/server_policy.py \
    --ckpt_path $STARVLA/results/Checkpoints/libero_pathB_4B_L4_v3_30k/final_model/pytorch_model.pt \
    --port 7034 --use_bf16 > $OUT_ROOT/4B_calib_server.log 2>&1 &
SERVER_4B_PID=$!
echo "[pipeline] 4B calib server PID=$SERVER_4B_PID"
wait_grep "server listening" "$OUT_ROOT/4B_calib_server.log" 600

DUMP_4B=$DYN_CFGS/pathB_4B_L4_v3_30k_sim_dump.jsonl
rm -f "$DUMP_4B"
(
  cd $STARVLA
  for SUITE in libero_spatial libero_object libero_goal libero_10; do
    echo "=== 4B calib $SUITE $(date) ==="
    CUDA_VISIBLE_DEVICES=5 EGL_DEVICE_ID=5 ROUTING_DUMP_PATH="$DUMP_4B" \
      bash examples/LIBERO/eval_files/eval_libero_calibrate.sh \
      $STARVLA/results/Checkpoints/libero_pathB_4B_L4_v3_30k/final_model/pytorch_model.pt \
      "$SUITE" 7034 5
  done
  echo "=== 4B calib dump ALL DONE ==="
) > "$OUT_ROOT/4B_calib_driver.log" 2>&1
echo "[pipeline] Phase B done $(date)"

# keep 4B server alive for next phases (we'll reuse for dynskip)

# ------ Phase C: post-process dumps ------------------------------------
echo "[pipeline] Phase C: post-process both dumps $(date)"
PY=${PY:-/home/user01/Minko/reskip2/.venv/bin/python}
# Prefer venv python with numpy; fall back to env python if script shebang handles it.
# 4B @ q=0.5  (P={1,4} pending — check drift after dump; default to {1,4})
$PY $RETROFIT/eval/calibrate_sim_thresholds.py \
  --dump "$DUMP_4B" \
  --output "$DYN_CFGS/pathB_4B_L4_v3_30k_sim_q50.json" \
  --quantile 0.50 --eligible 1,4 --max-skips 2 \
  --notes "Method B sim-calibration for 4B_L4, 200 trials × 4 suites"
# 2B @ q ∈ {0.30, 0.50, 0.70, 0.85} for sweep
DUMP_2B=$DYN_CFGS/pathB_2B_L4_v3_30k_sim_dump.jsonl
for Q in 0.30 0.50 0.70 0.85; do
  QLBL=$(echo $Q | tr -d .)
  $PY $RETROFIT/eval/calibrate_sim_thresholds.py \
    --dump "$DUMP_2B" \
    --output "$DYN_CFGS/pathB_2B_L4_v3_30k_sim_q${QLBL}.json" \
    --quantile $Q --eligible 1,4 --max-skips 2 \
    --notes "Method B sim-calibration for 2B_L4 q-sweep point q=$Q"
done
echo "[pipeline] Phase C done $(date)"

# ------ Phase D: 4B dynskip full eval (Experiment 1) --------------------
echo "[pipeline] Phase D: 4B dynskip full eval $(date)"
# Kill stale 4B server first and use run_full_eval_skip.sh which spins its own
pkill -9 -f "server_policy.py.*port 7034" 2>/dev/null
sleep 5
DYN_SKIP_CONFIG_PATH="$DYN_CFGS/pathB_4B_L4_v3_30k_sim_q50.json" \
  bash $STARVLA/examples/LIBERO/eval_files/run_full_eval_skip.sh \
  pathB_4B_L4_v3_30k_dynskip_simq50 \
  $STARVLA/results/Checkpoints/libero_pathB_4B_L4_v3_30k/final_model/pytorch_model.pt \
  4 5 7035 50
echo "[pipeline] Phase D done $(date)"

# ------ Phase E: q-sweep on 2B (Experiment 2) ---------------------------
echo "[pipeline] Phase E: 2B q-sweep $(date)"
# Start a fresh 2B server on GPU 0 (no-skip 2B should have finished by now on 0-1)
pkill -9 -f "server_policy.py.*port 7030" 2>/dev/null
pkill -9 -f "server_policy.py.*port 7031" 2>/dev/null
sleep 5
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline PYTHONPATH=$STARVLA \
  nohup python $STARVLA/deployment/model_server/server_policy.py \
    --ckpt_path $STARVLA/results/Checkpoints/libero_pathB_2B_L4_v3_30k/final_model/pytorch_model.pt \
    --port 7036 --use_bf16 > $OUT_ROOT/2B_qsweep_server.log 2>&1 &
SERVER_2B_PID=$!
echo "[pipeline] 2B qsweep server PID=$SERVER_2B_PID"
wait_grep "server listening" "$OUT_ROOT/2B_qsweep_server.log" 600

# for each q in {0.30, 0.70, 0.85} run libero_spatial only (q=0.50 is the
# already-completed dynskip run; we just reference its result).
for Q in 0.30 0.70 0.85; do
  QLBL=$(echo $Q | tr -d .)
  CFG="$DYN_CFGS/pathB_2B_L4_v3_30k_sim_q${QLBL}.json"
  LBL="pathB_2B_L4_v3_30k_qsweep_simq${QLBL}"
  OUT="$OUT_ROOT/${LBL}_libero_spatial"
  mkdir -p "$OUT"
  echo "[pipeline] === q=$Q ($LBL) begin ==="
  TSTART=$(date +%s)
  CUDA_VISIBLE_DEVICES=1 EGL_DEVICE_ID=1 \
    DYN_SKIP_CONFIG_PATH="$CFG" \
    bash $STARVLA/examples/LIBERO/eval_files/eval_libero_skip.sh \
    $STARVLA/results/Checkpoints/libero_pathB_2B_L4_v3_30k/final_model/pytorch_model.pt \
    libero_spatial 7036 50 > "$OUT/eval.log" 2>&1
  TEND=$(date +%s)
  SR=$(grep -oE "Total success rate: [0-9.]+" "$OUT/eval.log" | tail -1)
  echo "[pipeline] q=$Q done  SR=$SR  elapsed=$((TEND-TSTART))s"
done
pkill -9 $SERVER_2B_PID 2>/dev/null
echo "[pipeline] Phase E done $(date)"

# ------ Phase F: VLM-only reskip (Experiment 3) -------------------------
echo "[pipeline] Phase F: VLM-only reskip $(date)"
# Uses VLM retrofit checkpoint directly; eval_dynamic_skip.py runs LAMBADA end-to-end.
VLM_STATE=$RETROFIT/outputs/block_v3/2B_L4_v3_10k/retrofit_attnres_state.pt
VLM_LOG=$OUT_ROOT/vlm_reskip_driver.log
: > "$VLM_LOG"
echo "--- VLM baseline (max-skips 0) ---" >> "$VLM_LOG"
$PY $RETROFIT/eval/eval_dynamic_skip.py \
  --state-path "$VLM_STATE" --num-blocks 7 --max-skips 0 \
  --lambada-n 500 --gpu 6 >> "$VLM_LOG" 2>&1 || true
for Q in 0.30 0.50 0.85; do
  echo "--- VLM dynskip q=$Q, P={1,4}, M=2 ---" >> "$VLM_LOG"
  $PY $RETROFIT/eval/eval_dynamic_skip.py \
    --state-path "$VLM_STATE" --num-blocks 7 \
    --quantile $Q --eligible 1,4 --max-skips 2 \
    --lambada-n 500 --gpu 6 >> "$VLM_LOG" 2>&1 || true
done
echo "[pipeline] Phase F done $(date)"

echo "[pipeline] PIPELINE COMPLETE $(date)"
