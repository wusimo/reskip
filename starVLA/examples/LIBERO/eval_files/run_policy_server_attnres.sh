#!/bin/bash

STARVLA_ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
cd "${STARVLA_ROOT}"

export PYTHONPATH=$(pwd):${PYTHONPATH}
export star_vla_python=/mnt/petrelfs/share/yejinhui/Envs/miniconda3/envs/starVLA/bin/python

your_ckpt=results/Checkpoints/libero_qwenoft_attnres/checkpoints/steps_50000_pytorch_model.pt
gpu_id=7
port=5694

# Skip config is threaded per-request from eval_libero.py via vla_input;
# the server doesn't take skip args anymore.
CUDA_VISIBLE_DEVICES=$gpu_id ${star_vla_python} deployment/model_server/server_policy.py \
  --ckpt_path ${your_ckpt} \
  --port ${port} \
  --use_bf16
