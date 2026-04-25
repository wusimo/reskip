#!/bin/bash
# Local-adapted LIBERO eval script.
# Usage:
#   bash eval_libero_local.sh <checkpoint_path> <task_suite> [num_trials] [port]
#
# Prereqs:
#   - Policy server running in main .venv (tradition: run_policy_server.sh)
#   - LIBERO deps installed into Minko .venv (uv-managed)
#   - LIBERO source at /home/user01/yuxinglei/workspace/DreamerVLA/LIBERO
#
# We use the LOCAL machine paths.

set -u
CKPT="${1:?usage: <ckpt_path> <task_suite> [trials] [port]}"
SUITE="${2:?}"
TRIALS="${3:-50}"
PORT="${4:-5694}"

export LIBERO_HOME=${LIBERO_HOME:-/home/user01/Minko/reskip2/LIBERO_local}
export LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH:-/home/user01/Minko/reskip2/libero_config}
export LIBERO_Python=${LIBERO_Python:-/home/user01/Minko/reskip2/.venv/bin/python}

export PYTHONPATH=${PYTHONPATH:-}:${LIBERO_HOME}

STARVLA_ROOT=/home/user01/Minko/reskip2/reskip/starVLA
cd "${STARVLA_ROOT}"
export PYTHONPATH=$(pwd):${PYTHONPATH}

HOST="127.0.0.1"

folder_name=$(echo "$CKPT" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')

LOG_DIR="logs/libero_eval_$(date +"%Y%m%d_%H%M%S")_${folder_name}_${SUITE}"
mkdir -p ${LOG_DIR}

video_out="results/${SUITE}/${folder_name}"
mkdir -p "${video_out}"

echo "[eval_libero_local] ckpt=$CKPT suite=$SUITE trials=$TRIALS port=$PORT"
echo "[eval_libero_local] LIBERO_Python=$LIBERO_Python"

${LIBERO_Python} ./examples/LIBERO/eval_files/eval_libero.py \
    --args.pretrained-path "${CKPT}" \
    --args.host "${HOST}" \
    --args.port "${PORT}" \
    --args.task-suite-name "${SUITE}" \
    --args.num-trials-per-task "${TRIALS}" \
    --args.video-out-path "${video_out}" 2>&1 | tee ${LOG_DIR}/eval.log
