#!/bin/bash
# run_train_afhq_sr.sh — Launch SR model training (32→64 AFHQ)
#
# Multi-GPU support via torchrun:
#   GPUS=0,1 bash scripts/run_train_afhq_sr.sh
#   GPUS=0,1,2,3 bash scripts/run_train_afhq_sr.sh
#
# Single-GPU (default):
#   bash scripts/run_train_afhq_sr.sh
#   GPUS=6 bash scripts/run_train_afhq_sr.sh   # use GPU 3 only
#
# Expected GPU memory: batch_size=16 ≈ 6–8 GB per GPU
# Can train in PARALLEL with the base model — they are independent.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON="${PYTHON:-python}"

# ── Configuration ────────────────────────────────────────────────────────────
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data/afhq_v2/train/combined_images}"
SAVE_DIR="${SAVE_DIR:-${PROJECT_ROOT}/checkpoints/afhq_sr_64_2}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LR="${LR:-1e-4}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
COND_AUG_PROB="${COND_AUG_PROB:-1.0}"
COND_AUG_MAX_TIMESTEP="${COND_AUG_MAX_TIMESTEP:-300}"
GPUS="${GPUS:- 4,5}"   

mkdir -p "$SAVE_DIR"
# Count GPUs
IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
NUM_GPUS=${#GPU_ARRAY[@]}

echo "=============================================="
echo "  AFHQ SR Model Training (32→64)"
echo "=============================================="
echo "  GPUs:               $GPUS ($NUM_GPUS GPU(s))"
echo "  Data:               $DATA_DIR"
echo "  Save:               $SAVE_DIR"
echo "  Batch size:         $BATCH_SIZE (per GPU)"
echo "  Learning rate:      $LR"
echo "  Save interval:      $SAVE_INTERVAL"
echo "  Cond aug prob:      $COND_AUG_PROB"
echo "  Cond aug S:         $COND_AUG_MAX_TIMESTEP"
echo "=============================================="

TRAIN_ARGS=(
    --data_dir            "$DATA_DIR"
    --save_dir            "$SAVE_DIR"
    --batch_size          "$BATCH_SIZE"
    --lr                  "$LR"
    --save_interval       "$SAVE_INTERVAL"
    --cond_aug_prob       "$COND_AUG_PROB"
    --cond_aug_max_timestep  "$COND_AUG_MAX_TIMESTEP"
)

MASTER_PORT="${MASTER_PORT:-29500}"

if [ "$NUM_GPUS" -gt 1 ]; then
    CUDA_VISIBLE_DEVICES="$GPUS" torchrun \
        --master_port="$MASTER_PORT" \
        --nproc_per_node="$NUM_GPUS" \
        "$SCRIPT_DIR/train_afhq_sr_64.py" \
        "${TRAIN_ARGS[@]}"

else
    # Single-GPU
    CUDA_VISIBLE_DEVICES="$GPUS" $PYTHON "$SCRIPT_DIR/train_afhq_sr_64.py" \
        "${TRAIN_ARGS[@]}"
fi
