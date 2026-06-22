#!/bin/bash
# run_train_afhq_base.sh — Launch base model training (32×32 AFHQ)
#
# Multi-GPU support via torchrun:
#   GPUS=0,1 bash scripts/run_train_afhq_base.sh
#   GPUS=0,1,2,3 bash scripts/run_train_afhq_base.sh
#
# Single-GPU (default):
#   bash scripts/run_train_afhq_base.sh
#   GPUS=5 bash scripts/run_train_afhq_base.sh   # use GPU 2 only
#
# Expected GPU memory: batch_size=32 ≈ 4–6 GB per GPU
# Train for 50k–100k steps for reasonable results on AFHQ.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON="${PYTHON:-python}"

# ── Configuration ────────────────────────────────────────────────────────────
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data/afhq_v2/train/combined_images}"
SAVE_DIR="${SAVE_DIR:-${PROJECT_ROOT}/checkpoints/afhq_base_32}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-1e-4}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
GPUS="${GPUS:-0}"    # comma-separated GPU IDs, e.g. "0,1" or "0,1,2,3"

mkdir -p "$SAVE_DIR"

# Count GPUs
IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
NUM_GPUS=${#GPU_ARRAY[@]}

echo "=============================================="
echo "  AFHQ Base Model Training (32×32)"
echo "=============================================="
echo "  GPUs:           $GPUS ($NUM_GPUS GPU(s))"
echo "  Data:           $DATA_DIR"
echo "  Save:           $SAVE_DIR"
echo "  Batch size:     $BATCH_SIZE (per GPU)"
echo "  Learning rate:  $LR"
echo "  Save interval:  $SAVE_INTERVAL"
echo "=============================================="

if [ "$NUM_GPUS" -gt 1 ]; then
    # Multi-GPU via torchrun
    CUDA_VISIBLE_DEVICES="$GPUS" torchrun \
        --nproc_per_node="$NUM_GPUS" \
        "$SCRIPT_DIR/train_afhq_base_32.py" \
        --data_dir      "$DATA_DIR" \
        --save_dir      "$SAVE_DIR" \
        --batch_size    "$BATCH_SIZE" \
        --lr            "$LR" \
        --save_interval "$SAVE_INTERVAL"
else
    # Single-GPU
    CUDA_VISIBLE_DEVICES="$GPUS" $PYTHON "$SCRIPT_DIR/train_afhq_base_32.py" \
        --data_dir      "$DATA_DIR" \
        --save_dir      "$SAVE_DIR" \
        --batch_size    "$BATCH_SIZE" \
        --lr            "$LR" \
        --save_interval "$SAVE_INTERVAL"
fi
