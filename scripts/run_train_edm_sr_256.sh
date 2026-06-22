#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# run_train_edm_sr_256.sh — Train the EDM SR model (64→256)
# ──────────────────────────────────────────────────────────────────────────────
#
# Usage:
#   GPUS=7 bash scripts/run_train_edm_sr_256.sh
#   GPUS=4,5,6,7 bash scripts/run_train_edm_sr_256.sh
#
# CDM conditioning augmentation (§4.2):
#   Always applied.  s ~ Uniform{0,…,S},  S = 300 (30% of T=1000).
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ── Configuration (override with env vars) ────────────────────────────────────
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data/afhq/}"
OUTDIR="${OUTDIR:-${PROJECT_ROOT}/checkpoints/edm_afhq_sr_256_2}"
BATCH="${BATCH:-128}"
BATCH_GPU="${BATCH_GPU:-32}"         # per-GPU micro-batch
LR="${LR:-1e-4}"
TOTAL_KIMG="${TOTAL_KIMG:-200000}"   # total training kimg
COND_AUG_S="${COND_AUG_S:-300}"      # 30% of T=1000
MODEL_CHANNELS="${MODEL_CHANNELS:-64}"
DROPOUT="${DROPOUT:-0.00}"
TICK="${TICK:-5}"                    # print every N kimg
SNAP="${SNAP:-10}"                   # snapshot every N ticks
SEED="${SEED:-0}"
FP16="${FP16:-false}"
GPUS="${GPUS:-2,3}"                    # comma-separated GPU IDs
RESUME_PKL="${RESUME_PKL:-/data/Sahil/mri_cascaded_diffusion/checkpoints/edm_afhq_sr_256_2/network-snapshot-000153.pkl}"         # .pkl to transfer from
RESUME_STATE="${RESUME_STATE:-}"     # training-state-XXXXXX.pt
RESUME_KIMG="${RESUME_KIMG:-153}"      # kimg to resume from (must match pkl snapshot number)

# ── Parse GPUs ────────────────────────────────────────────────────────────────
IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
NUM_GPUS=${#GPU_ARRAY[@]}

echo "============================================================"
echo "  EDM SR Model Training (64→256)"
echo "  CDM Conditioning Augmentation: S=${COND_AUG_S} (always)"
echo "============================================================"
echo "  GPUs:          $GPUS ($NUM_GPUS GPU(s))"
echo "  Data:          $DATA_DIR"
echo "  Output:        $OUTDIR"
echo "  Batch size:    $BATCH (batch_gpu=$BATCH_GPU)"
echo "  Learning rate: $LR"
echo "  Total kimg:    $TOTAL_KIMG"
echo "  Cond aug S:    $COND_AUG_S / 1000"
echo "  model_channels:$MODEL_CHANNELS"
echo "  Dropout:       $DROPOUT"
echo "  FP16:          $FP16"
echo "  Seed:          $SEED"
echo "============================================================"

mkdir -p "$OUTDIR"

# ── Build command ─────────────────────────────────────────────────────────────
ARGS=(
    --data_dir      "$DATA_DIR"
    --outdir        "$OUTDIR"
    --batch         "$BATCH"
    --batch_gpu     "$BATCH_GPU"
    --lr            "$LR"
    --total_kimg    "$TOTAL_KIMG"
    --cond_aug_max_timestep "$COND_AUG_S"
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
    ARGS+=(--resume_pkl "$RESUME_PKL")
    ARGS+=(--resume_kimg "$RESUME_KIMG")
    echo "  Resuming from: $RESUME_PKL at kimg=$RESUME_KIMG"
fi
if [ -n "$RESUME_STATE" ]; then
    ARGS+=(--resume_state "$RESUME_STATE")
fi

# ── Launch ────────────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES="$GPUS"
MASTER_PORT="${MASTER_PORT:-29501}"

CONDA_ENV="${CONDA_ENV:-${HOME}/miniconda/envs/fastmri}"
TORCHRUN="${CONDA_ENV}/bin/torchrun"

"$TORCHRUN" \
    --standalone \
    --master_port="$MASTER_PORT" \
    --nproc_per_node="$NUM_GPUS" \
    "${SCRIPT_DIR}/train_edm_sr_256.py" \
    "${ARGS[@]}"
