#!/usr/bin/env bash
# run_train_base.sh
# -----------------
# Launch script for the BASE 96×96 MRI diffusion model.
# Trains on AXT2 slices at 96×96 resolution (unconditional, cosine schedule).
#
# Multi-GPU support via torchrun:
#   GPUS=0,1 bash scripts/run_train_base.sh
#   GPUS=0,1,2,3 bash scripts/run_train_base.sh
#
# Single-GPU (default):
#   bash scripts/run_train_base.sh
#   GPUS=2 bash scripts/run_train_base.sh   # use GPU 2 only
#
# GPU memory:
#   batch_size=4  ≈  6 GB
#   batch_size=8  ≈ 12 GB

set -euo pipefail

PYTHON=/home/nvidia/miniconda/envs/fastmri/bin/python
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

BATCH=${BATCH:-8}
LR=${LR:-1e-4}
SAVE_INTERVAL=${SAVE_INTERVAL:-5000}
LOG_INTERVAL=${LOG_INTERVAL:-100}
GPUS="${GPUS:-0}"

PT_DIR="/data/Sahil_dataset/MRI_processed/train/AXT2_normalized"
SAVE_DIR="$REPO_DIR/checkpoints/base_96"

mkdir -p "$SAVE_DIR"


# Count GPUs
IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
NUM_GPUS=${#GPU_ARRAY[@]}

echo "====================================================="
echo " MRI Cascaded Diffusion — BASE MODEL (96×96)"
echo " GPUs   : $GPUS ($NUM_GPUS GPU(s))"
echo " Data   : $PT_DIR"
echo " Output : $SAVE_DIR"
echo " Batch  : $BATCH (per GPU)  |  LR: $LR"
echo "====================================================="

TRAIN_ARGS=(
    --pt_dir          "$PT_DIR"
    --save_dir        "$SAVE_DIR"
    --batch_size      "$BATCH"
    --lr              "$LR"
    --save_interval   "$SAVE_INTERVAL"
    --log_interval    "$LOG_INTERVAL"
    --ema_rate        "0.9999"
    --noise_schedule  "cosine"
    --diffusion_steps 1000
    --learn_sigma     "True"
    --in_channels     1
    --image_size      96
    --num_channels    64
    --num_res_blocks  2
    --num_heads       4
    --num_head_channels 32
    --attention_resolutions "16"
    --dropout         0.1
    --use_scale_shift_norm "True"
    --resblock_updown "True"
    --use_new_attention_order "True"
    --schedule_sampler "uniform"
    "$@"
)

if [ "$NUM_GPUS" -gt 1 ]; then
    CUDA_VISIBLE_DEVICES="$GPUS" torchrun \
        --nproc_per_node="$NUM_GPUS" \
        "$SCRIPT_DIR/train_base_96.py" \
        "${TRAIN_ARGS[@]}"
else
    CUDA_VISIBLE_DEVICES="$GPUS" $PYTHON "$SCRIPT_DIR/train_base_96.py" \
        "${TRAIN_ARGS[@]}"
fi
