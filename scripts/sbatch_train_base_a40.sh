#!/usr/bin/env bash
# =============================================================================
# sbatch_train_base_a40.sh
# -----------------------------------------------------------------------------
# SLURM batch script — BASE 96×96 MRI diffusion model on 3× A40 GPUs (EDM)
#
# Submit:  sbatch scripts/sbatch_train_base_a40.sh
# Resume:  RESUME_PKL=checkpoints/edm_mri_base_96/network-snapshot-XXXXXX.pkl \
#            sbatch scripts/sbatch_train_base_a40.sh
#
# A40 vs A100 notes:
#   A40:  48 GB VRAM   (half of A100 80 GB)
#   3×A40 = 144 GB total VRAM
#   batch_gpu=16 per GPU  →  global batch = 48  (safe headroom)
#   batch_gpu=24 per GPU  →  global batch = 72  (push if stable)
# =============================================================================

#SBATCH --job-name=mri_base_a40
#SBATCH --account=cbig_a40
#SBATCH --reservation=cbig_a40
#SBATCH --partition=dedicated
#SBATCH --gres=gpu:a40:3
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --chdir=/standard/CBIG-Standard-ECE/Sahil/mri_cascaded_diffusion
#SBATCH --output=logs/base_a40_%j.out
#SBATCH --error=logs/base_a40_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

# ---------------------------------------------------------------------------
set -euo pipefail

# Hardcoded: sbatch copies the script to a spool dir, so BASH_SOURCE[0] breaks relative paths
REPO_DIR="/standard/CBIG-Standard-ECE/Sahil/mri_cascaded_diffusion"
SCRIPT_DIR="$REPO_DIR/scripts"

# ── Environment ───────────────────────────────────────────────────────────────
# There is no ~/miniconda3; the working env is ~/.conda/envs/fastmri
CONDA_ENV="${HOME}/.conda/envs/fastmri"
TORCHRUN="${CONDA_ENV}/bin/torchrun"

# Fix: libtorch_cpu.so undefined symbol iJIT_NotifyEvent
# libiomp5.so (Intel OpenMP) lives in the conda env and provides this symbol
export LD_PRELOAD="${CONDA_ENV}/lib/libiomp5.so${LD_PRELOAD:+:${LD_PRELOAD}}"
export MKL_THREADING_LAYER=GNU

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR="${DATA_DIR:-/data/Sahil_dataset/MRI_processed/train/AXT2_normalized}"
OUTDIR="${OUTDIR:-$REPO_DIR/checkpoints/edm_mri_base_96}"

# ── Hyperparameters (A40-tuned) ───────────────────────────────────────────────
# Global batch is split across GPUs: batch_gpu=16 × 3 GPUs → global=48
BATCH="${BATCH:-48}"
BATCH_GPU="${BATCH_GPU:-16}"
LR="${LR:-1e-3}"
TOTAL_KIMG="${TOTAL_KIMG:-200000}"
MODEL_CHANNELS="${MODEL_CHANNELS:-128}"
DROPOUT="${DROPOUT:-0.13}"
TICK="${TICK:-50}"
SNAP="${SNAP:-50}"
SEED="${SEED:-0}"
FP16="${FP16:-false}"

# Optional resume
RESUME_PKL="${RESUME_PKL:-}"
RESUME_STATE="${RESUME_STATE:-}"
RESUME_KIMG="${RESUME_KIMG:-0}"

# ---------------------------------------------------------------------------
NUM_GPUS=3
mkdir -p "$OUTDIR"
mkdir -p "$REPO_DIR/logs"

echo "============================================================"
echo "  EDM MRI Base Model Training (96×96) — A40 × $NUM_GPUS"
echo "  Node    : $(hostname)"
echo "  GPUs    : $NUM_GPUS × A40 (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-auto/SLURM})"
echo "  Data    : $DATA_DIR"
echo "  Output  : $OUTDIR"
echo "  Batch   : $BATCH_GPU per GPU  →  global $BATCH"
echo "  LR      : $LR"
echo "  kimg    : $TOTAL_KIMG"
echo "  FP16    : $FP16"
echo "  Time    : $(date)"
echo "============================================================"

# Print SLURM GPU assignments
echo "SLURM_JOB_GPUS: ${SLURM_JOB_GPUS:-not set}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader 2>/dev/null || true

# ── Build args ────────────────────────────────────────────────────────────────
ARGS=(
    --data_dir       "$DATA_DIR"
    --outdir         "$OUTDIR"
    --batch          "$BATCH"
    --batch_gpu      "$BATCH_GPU"
    --lr             "$LR"
    --total_kimg     "$TOTAL_KIMG"
    --model_channels "$MODEL_CHANNELS"
    --dropout        "$DROPOUT"
    --tick           "$TICK"
    --snap           "$SNAP"
    --seed           "$SEED"
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
MASTER_PORT="${MASTER_PORT:-29502}"
echo "  torchrun : $TORCHRUN"

"$TORCHRUN" \
    --standalone \
    --master_port="$MASTER_PORT" \
    --nproc_per_node="$NUM_GPUS" \
    "$SCRIPT_DIR/train_edm_mri_base.py" \
    "${ARGS[@]}"
