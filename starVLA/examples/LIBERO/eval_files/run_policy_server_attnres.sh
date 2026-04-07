#!/bin/bash

STARVLA_ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
cd "${STARVLA_ROOT}"

export PYTHONPATH=$(pwd):${PYTHONPATH}
export star_vla_python=/mnt/petrelfs/share/yejinhui/Envs/miniconda3/envs/starVLA/bin/python

your_ckpt=results/Checkpoints/libero_qwenoft_attnres/checkpoints/steps_50000_pytorch_model.pt
gpu_id=7
port=5694
enable_skipping=false
skip_mode=none
uniform_skip_threshold=0.01
vision_skip_threshold=0.02
language_skip_threshold=0.01
action_skip_threshold=0.005

server_args=()
if [ "${enable_skipping}" = "true" ]; then
  server_args+=(--enable_skipping)
fi

CUDA_VISIBLE_DEVICES=$gpu_id ${star_vla_python} deployment/model_server/server_policy.py \
  --ckpt_path ${your_ckpt} \
  --port ${port} \
  --use_bf16 \
  --skip_mode ${skip_mode} \
  --uniform_skip_threshold ${uniform_skip_threshold} \
  --vision_skip_threshold ${vision_skip_threshold} \
  --language_skip_threshold ${language_skip_threshold} \
  --action_skip_threshold ${action_skip_threshold} \
  "${server_args[@]}"
