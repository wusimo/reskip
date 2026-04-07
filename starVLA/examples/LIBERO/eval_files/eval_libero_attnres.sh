#!/bin/bash

STARVLA_ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
cd "${STARVLA_ROOT}"

conda activate starVLA

export LIBERO_HOME=/mnt/petrelfs/share/yejinhui/Projects/LIBERO
export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero
export LIBERO_Python=/mnt/petrelfs/share/yejinhui/Envs/miniconda3/envs/lerobot/bin/python
export PYTHONPATH=$PYTHONPATH:${LIBERO_HOME}
export PYTHONPATH=$(pwd):${PYTHONPATH}

host="127.0.0.1"
base_port=5694
your_ckpt=./results/Checkpoints/libero_qwenoft_attnres/checkpoints/steps_50000_pytorch_model.pt
task_suite_name=libero_goal
num_trials_per_task=50
video_out_path="results/${task_suite_name}/attnres_eval"
enable_skipping=false
skip_mode=none
uniform_skip_threshold=0.01
vision_skip_threshold=0.02
language_skip_threshold=0.01
action_skip_threshold=0.005

${LIBERO_Python} ./examples/LIBERO/eval_files/eval_libero.py \
  --args.pretrained-path ${your_ckpt} \
  --args.host "$host" \
  --args.port $base_port \
  --args.task-suite-name "$task_suite_name" \
  --args.num-trials-per-task "$num_trials_per_task" \
  --args.video-out-path "$video_out_path" \
  --args.enable-skipping ${enable_skipping} \
  --args.skip-mode ${skip_mode} \
  --args.uniform-skip-threshold ${uniform_skip_threshold} \
  --args.vision-skip-threshold ${vision_skip_threshold} \
  --args.language-skip-threshold ${language_skip_threshold} \
  --args.action-skip-threshold ${action_skip_threshold}
