#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# run_train_edm_base.sh — Train a 32×32 base model using EDM's train.py
# ──────────────────────────────────────────────────────────────────────────────
#
# Usage:
#   GPUS=7 bash scripts/run_train_edm_base.sh
#   GPUS=4,5,6,7 bash scripts/run_train_edm_base.sh
#
# EDM uses its own ImageFolderDataset — point DATA to a directory of images.
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ── Conda environment ─────────────────────────────────────────────────────────
CONDA_ENV="${CONDA_ENV:-${HOME}/miniconda/envs/fastmri}"
TORCHRUN="${CONDA_ENV}/bin/torchrun"

# ── Configuration (override with env vars) ────────────────────────────────────
DATA="${DATA:-${PROJECT_ROOT}/data/afhq_32x32.zip}"
OUTDIR="${OUTDIR:-${PROJECT_ROOT}/checkpoints/edm_afhq_base_32}"
BATCH="${BATCH:-256}"
BATCH_GPU="${BATCH_GPU:-64}"         # per-GPU micro-batch
LR="${LR:-10e-4}"                    # EDM default 10e-4 = 1e-3
DURATION="${DURATION:-200}"          # total Mimg
DROPOUT="${DROPOUT:-0.13}"           # EDM default
AUGMENT="${AUGMENT:-0.12}"           # augment probability (EDM augment pipe)
XFLIP="${XFLIP:-True}"               # True = always apply horizontal flip
ARCH="${ARCH:-ddpmpp}"               # ddpmpp | ncsnpp | adm
PRECOND="${PRECOND:-edm}"            # edm | vp | ve
TICK="${TICK:-50}"                    # print every N kimg
SNAP="${SNAP:-50}"                   # snapshot every N ticks
CBASE="${CBASE:-128}"                # model_channels
SEED="${SEED:-0}"
FP16="${FP16:-false}"                # mixed-precision
GPUS="${GPUS:-2,3}"                    # comma-separated GPU IDs
RESUME="${RESUME:-}"                 # path to training-state-XXXXXX.pt to resume

# ── Parse GPUs ────────────────────────────────────────────────────────────────
IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
NUM_GPUS=${#GPU_ARRAY[@]}

echo "============================================================"
echo "  EDM Base Model Training (32×32)"
echo "============================================================"
echo "  GPUs:          $GPUS ($NUM_GPUS GPU(s))"
echo "  Data:          $DATA"
echo "  Output:        $OUTDIR"
echo "  Batch size:    $BATCH (batch_gpu=$BATCH_GPU)"
echo "  Learning rate: $LR"
echo "  Duration:      ${DURATION} Mimg"
echo "  Architecture:  $ARCH + $PRECOND"
echo "  model_channels:$CBASE"
echo "  Dropout:       $DROPOUT"
echo "  Augment:       $AUGMENT"
echo "  Horiz. flip:   $XFLIP"
echo "  FP16:          $FP16"
echo "  Seed:          $SEED"
echo "============================================================"

mkdir -p "$OUTDIR"

# ── Build command ─────────────────────────────────────────────────────────────
ARGS=(
    --outdir        "$OUTDIR"
    --data          "$DATA"
    --batch         "$BATCH"
    --batch-gpu     "$BATCH_GPU"
    --lr            "$LR"
    --duration      "$DURATION"
    --arch          "$ARCH"
    --precond       "$PRECOND"
    --cbase         "$CBASE"
    --dropout       "$DROPOUT"
    --augment       "$AUGMENT"
    --xflip        "$XFLIP"
    --tick          "$TICK"
    --snap          "$SNAP"
    --seed          "$SEED"
)

if [ "$FP16" = "true" ]; then
    ARGS+=(--fp16 True)
fi

if [ -n "$RESUME" ]; then
    ARGS+=(--resume "$RESUME")
fi

# ── Launch ────────────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES="$GPUS"
MASTER_PORT="${MASTER_PORT:-29500}"

"$TORCHRUN" \
    --standalone \
    --master_port="$MASTER_PORT" \
    --nproc_per_node="$NUM_GPUS" \
    "${PROJECT_ROOT}/edm_repo/train.py" \
    "${ARGS[@]}"
