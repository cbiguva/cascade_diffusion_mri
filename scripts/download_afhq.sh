#!/bin/bash
# download_afhq.sh — Download AFHQ dataset into data/afhq/
#
# AFHQ (Animal Faces HQ) dataset: ~500MB
# Contains train/ and val/ splits with cat/, dog/, wild/ subfolders.
# Original resolution: 512×512 RGB images.
#
# Usage:
#   bash scripts/download_afhq.sh
#
# Alternative: download manually from
#   https://www.kaggle.com/datasets/andrewmvd/animal-faces
#   or clone https://github.com/clovaai/stargan-v2 and use their download script.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_ROOT/data/afhq"

if [ -d "$DATA_DIR/train" ]; then
    echo "[download_afhq] Dataset already exists at $DATA_DIR"
    echo "  train/cat:  $(find $DATA_DIR/train/cat -type f 2>/dev/null | wc -l) images"
    echo "  train/dog:  $(find $DATA_DIR/train/dog -type f 2>/dev/null | wc -l) images"
    echo "  train/wild: $(find $DATA_DIR/train/wild -type f 2>/dev/null | wc -l) images"
    exit 0
fi

echo "[download_afhq] Downloading AFHQ dataset to $DATA_DIR ..."
mkdir -p "$DATA_DIR"

# Method 1: wget from the official Google Drive link (via gdown)
# Direct download using wget with cookie/redirect handling
echo "[download_afhq] Downloading AFHQ via wget..."
wget --save-cookies cookies.txt --keep-session-cookies --no-check-certificate \
    'https://drive.google.com/uc?export=download&id=1YbHlMUmMveurZDpb5TNzNOapfOZ4bjiC' -O- \
    | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p' > confirm.txt

wget --load-cookies cookies.txt \
    'https://drive.google.com/uc?export=download&confirm='$(cat confirm.txt)'&id=1YbHlMUmMveurZDpb5TNzNOapfOZ4bjiC' \
    -O "$DATA_DIR/afhq.zip"

rm -f cookies.txt confirm.txt
fi

echo "[download_afhq] Extracting..."
cd "$DATA_DIR"


unzip -q afhq.zip
rm -f afhq.zip

# Verify
echo "[download_afhq] Done! Dataset structure:"
echo "  train/cat:  $(find $DATA_DIR/train/cat -type f | wc -l) images"
echo "  train/dog:  $(find $DATA_DIR/train/dog -type f | wc -l) images"
echo "  train/wild: $(find $DATA_DIR/train/wild -type f | wc -l) images"
echo "  val/cat:    $(find $DATA_DIR/val/cat -type f 2>/dev/null | wc -l) images"
echo "  val/dog:    $(find $DATA_DIR/val/dog -type f 2>/dev/null | wc -l) images"
echo "  val/wild:   $(find $DATA_DIR/val/wild -type f 2>/dev/null | wc -l) images"
