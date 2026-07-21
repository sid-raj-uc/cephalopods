#!/usr/bin/env bash
# Download the trained caption-student model (Qwen3-VL-2B, MLX 4-bit, ~1.7 GB)
# from the GitHub release into models/qwen3vl2b_caption_v1_mlx_4bit/.
# The UI and src/local_pipeline.py look for it there automatically.
set -euo pipefail
cd "$(dirname "$0")"

URL="https://github.com/sid-raj-uc/cephalopods/releases/download/caption-student-v1/qwen3vl2b_caption_v1_mlx_4bit.zip"
DEST="models/qwen3vl2b_caption_v1_mlx_4bit"

if [ -f "$DEST/model.safetensors" ]; then
  echo "Model already present at $DEST — nothing to do."
  exit 0
fi

mkdir -p models
echo "Downloading caption student (~1.7 GB)..."
curl -L --fail --progress-bar -o models/_model.zip "$URL"
unzip -q -o models/_model.zip -d models/
rm models/_model.zip

[ -f "$DEST/model.safetensors" ] || { echo "ERROR: unzip did not produce $DEST"; exit 1; }
echo "Done: $DEST"
