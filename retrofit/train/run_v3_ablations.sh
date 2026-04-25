#!/bin/bash
# v3 ablations on GPUs 2-7, independent of main v3 runs on GPUs 0-1.
#
# GPU 2,3  — v3_5k      (v3 mix, 5k steps): isolates "is 10k steps necessary"
# GPU 4,5  — v3_vlonly  (LLaVA-OV 80% + UltraChat 20%, 10k): tests H4 (math
#            text adversarial to diagram reasoning). If AI2D/MMStar recover
#            here but LAMBADA falls back, H4 is real.
# GPU 6,7  — v1_10k     (v1 mix but 10k steps): isolates "is just more steps
#            enough with old data". If this matches v3_10k we were wrong
#            about the data mix mattering.

set -u
PROJECT=/home/user01/Minko/reskip2/reskip
PY=/home/user01/Minko/reskip2/.venv/bin/python
cd "$PROJECT"

launch() {
  local label="$1" gpu="$2" scale="$3" mix="$4" steps="$5"
  local model_path num_blocks
  if [ "$scale" = "2B" ]; then
    model_path=/home/user01/Minko/models/Qwen3-VL-2B; num_blocks=14
  else
    model_path=/home/user01/Minko/models/Qwen3-VL-4B; num_blocks=18
  fi
  local outdir=$PROJECT/retrofit/outputs/$label
  mkdir -p "$outdir"
  local log="$outdir/run.log"
  echo "[$(date +%H:%M:%S)] launch $label (gpu=$gpu, scale=$scale, mix=$mix, steps=$steps)"

  local extra="--data-mix $mix"
  if [ "$mix" = "v1" ]; then
    extra=""  # v1 uses default legacy mixer (ultrachat + llava_vsft)
  fi

  nohup $PY retrofit/train/train_qwen3vl_attnres_retrofit.py \
    --model-path "$model_path" \
    --num-blocks "$num_blocks" \
    --adapter-rank 256 \
    --steps "$steps" \
    --max-seq 2048 \
    --gamma-schedule --gamma-start 0 --gamma-end 1 --gamma-ramp-frac 0.3 \
    $extra \
    --kl-weight 1.0 \
    --entropy-weight 0.02 \
    --output-dir "$outdir" \
    --gpu "$gpu" \
    > "$log" 2>&1 &
  echo "  pid=$!"
}

launch v3_5k_2B       2 2B v3        5000
launch v3_5k_4B       3 4B v3        5000
launch v3_vlonly_2B   4 2B v3_vlonly 10000
launch v3_vlonly_4B   5 4B v3_vlonly 10000
launch v1_10k_2B      6 2B v1        10000
launch v1_10k_4B      7 4B v1        10000

echo
echo "[$(date +%H:%M:%S)] all 6 ablation cells launched"
