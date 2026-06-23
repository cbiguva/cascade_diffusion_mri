#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# run_train_edm_mri_base.sh — Train 96×96 base model for 2ch MRI using EDM
# ──────────────────────────────────────────────────────────────────────────────
#
# Usage:
#   GPUS=4,5 bash scripts/run_train_edm_mri_base.sh
#
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ── Conda environment ─────────────────────────────────────────────────────────
CONDA_ENV="${CONDA_ENV:-${HOME}/miniconda3/envs/edm}"
TORCHRUN="${CONDA_ENV}/bin/torchrun"

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_DIR="${DATA_DIR:-/CBIG-Standard-ECE/Sahil/stud_teach_fastmri/processed_data/AXT2_normalized}"
OUTDIR="${OUTDIR:-${PROJECT_ROOT}/checkpoints/edm_mri_base_96}"
BATCH="${BATCH:-64}"
BATCH_GPU="${BATCH_GPU:-64}"
LR="${LR:-1e-3}"
TOTAL_KIMG="${TOTAL_KIMG:-200000}"
MODEL_CHANNELS="${MODEL_CHANNELS:-128}"
DROPOUT="${DROPOUT:-0.13}"
TICK="${TICK:-50}"
SNAP="${SNAP:-50}"
SEED="${SEED:-0}"
FP16="${FP16:-false}"
GPUS="${GPUS:-0}"
RESUME_PKL="${RESUME_PKL:-}"
RESUME_STATE="${RESUME_STATE:-}"
RESUME_KIMG="${RESUME_KIMG:-0}"

# ── Parse GPUs ────────────────────────────────────────────────────────────────
IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
NUM_GPUS=${#GPU_ARRAY[@]}

echo "============================================================"
echo "  EDM MRI Base Model Training (2ch, 96×96)"
echo "============================================================"
echo "  GPUs:          $GPUS ($NUM_GPUS GPU(s))"
echo "  Data:          $DATA_DIR"
echo "  Output:        $OUTDIR"
echo "  Batch size:    $BATCH (batch_gpu=$BATCH_GPU)"
echo "  Learning rate: $LR"
echo "  Total kimg:    $TOTAL_KIMG"
echo "  model_channels:$MODEL_CHANNELS"
echo "  Dropout:       $DROPOUT"
echo "  FP16:          $FP16"
echo "  Seed:          $SEED"
echo "============================================================"

mkdir -p "$OUTDIR"

ARGS=(
    --data_dir      "$DATA_DIR"
    --outdir        "$OUTDIR"
    --batch         "$BATCH"
    --batch_gpu     "$BATCH_GPU"
    --lr            "$LR"
    --total_kimg    "$TOTAL_KIMG"
    --model_channels "$MODEL_CHANNELS"
    --dropout       "$DROPOUT"
    --tick          "$TICK"
    --snap          "$SNAP"
    --seed          "$SEED"
)

if [ "$FP16" = "true" ]; then
    ARGS+=(--fp16)
fi

if [ -n "$RESUME_PKL" ]; then
    ARGS+=(--resume_pkl "$RESUME_PKL" --resume_kimg "$RESUME_KIMG")
    echo "  Resuming from: $RESUME_PKL at kimg=$RESUME_KIMG"
fi
if [ -n "$RESUME_STATE" ]; then
    ARGS+=(--resume_state "$RESUME_STATE")
fi

# ── Launch ────────────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES="$GPUS"
MASTER_PORT="${MASTER_PORT:-29502}"

"$TORCHRUN" \
    --standalone \
    --master_port="$MASTER_PORT" \
    --nproc_per_node="$NUM_GPUS" \
    "${SCRIPT_DIR}/train_edm_mri_base.py" \
    "${ARGS[@]}"
