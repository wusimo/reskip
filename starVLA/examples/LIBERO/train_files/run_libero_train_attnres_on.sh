#!/bin/bash
# LIBERO OFT training with AttnRes ENABLED (Path A or B).
#
# Usage:
#   bash run_libero_train_attnres_on.sh <num_gpus> <max_steps> <batch> <run_id> [init_state_path]
#
# Path A (from-scratch, no warm start):
#   bash run_libero_train_attnres_on.sh 4 30000 16 libero_pathA ''
# Path B (warm-start from H_r256_5k):
#   bash run_libero_train_attnres_on.sh 4 30000 16 libero_pathB /abs/path/retrofit_attnres_state.pt

set -u
NUM_GPUS="${1:-1}"
MAX_STEPS="${2:-200}"
PDB="${3:-4}"
RUN_ID="${4:-libero_attnres_smoke}"
INIT_STATE="${5:-}"

REPO_ROOT=$(cd "$(dirname "$0")/../../../.." && pwd)/starVLA
cd "${REPO_ROOT}"

export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000
export NCCL_SOCKET_TIMEOUT_MS=360000
export PYTHONPATH=$(pwd):${PYTHONPATH:-}

Framework_name=QwenOFT
base_vlm="${BASE_VLM:-playground/Pretrained_models/Qwen3-VL-2B-Instruct-Action}"
N_BLOCKS="${N_BLOCKS:-14}"
config_yaml=./examples/LIBERO/train_files/starvla_cotrain_libero_attnres.yaml
libero_data_root=playground/Datasets/LEROBOT_LIBERO_DATA
data_mix=libero_all
run_root_dir=./results/Checkpoints
output_dir=${run_root_dir}/${RUN_ID}
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/"

INIT_STATE_ARG=""
if [ -n "${INIT_STATE}" ]; then
  INIT_STATE_ARG="--framework.attnres.init_state_path ${INIT_STATE}"
fi

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes ${NUM_GPUS} \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --framework.attnres.enabled True \
  --framework.attnres.n_blocks ${N_BLOCKS} \
  --framework.attnres.adapter_rank 256 \
  --framework.attnres.enable_skipping False \
  --framework.attnres.skip_mode none \
  ${INIT_STATE_ARG} \
  --datasets.vla_data.data_root_dir ${libero_data_root} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size ${PDB} \
  --trainer.vla_data.video_backend torchvision_av \
  --trainer.freeze_modules '' \
  --trainer.max_train_steps ${MAX_STEPS} \
  --trainer.save_interval ${SAVE_INTERVAL:-${MAX_STEPS}} \
  --trainer.logging_frequency 10 \
  --trainer.eval_interval 1000000 \
  --run_root_dir ${run_root_dir} \
  --run_id ${RUN_ID} \
  --wandb_project starVLA_Libero \
  --wandb_entity reskip
