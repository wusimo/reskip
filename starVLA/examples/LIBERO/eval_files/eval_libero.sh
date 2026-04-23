#!/bin/bash
# Local-adapted eval script. Args:
#   bash eval_libero.sh <ckpt_path> <task_suite> <port> [num_trials]

set -u
your_ckpt="${1:?usage: <ckpt> <suite> <port> [trials]}"
task_suite_name="${2:-libero_goal}"
base_port="${3:-5694}"
num_trials_per_task="${4:-50}"

cd /home/user01/Minko/reskip2/reskip/starVLA

###########################################################################################
export LIBERO_HOME=/home/user01/yuxinglei/workspace/DreamerVLA/LIBERO
export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero
export LIBERO_Python=/home/user01/miniconda3/envs/dreamervla/bin/python

# Force NVIDIA EGL vendor (mesa default fails on this box)
export MUJOCO_GL=egl
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export EGL_DEVICE_ID="${EGL_DEVICE_ID:-4}"

export PYTHONPATH=${PYTHONPATH:-}:${LIBERO_HOME}
export PYTHONPATH=$(pwd):${PYTHONPATH}
###########################################################################################

host="127.0.0.1"
unnorm_key="franka"

folder_name=$(echo "$your_ckpt" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')
LOG_DIR="logs/$(date +"%Y%m%d_%H%M%S")_${folder_name}_${task_suite_name}"
mkdir -p ${LOG_DIR}

video_out_path="results/${task_suite_name}/${folder_name}"
mkdir -p "${video_out_path}"

${LIBERO_Python} ./examples/LIBERO/eval_files/eval_libero.py \
    --args.pretrained-path "${your_ckpt}" \
    --args.host "$host" \
    --args.port "${base_port}" \
    --args.task-suite-name "${task_suite_name}" \
    --args.num-trials-per-task "${num_trials_per_task}" \
    --args.video-out-path "${video_out_path}" 2>&1 | tee ${LOG_DIR}/eval.log
