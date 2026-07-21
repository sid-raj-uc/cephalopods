#!/usr/bin/env bash
# Download the trained caption-student model from the GitHub release.
#
#   Apple Silicon (arm64 macOS) -> MLX 4-bit model  (~1.7 GB)  -> models/qwen3vl2b_caption_v1_mlx_4bit/
#   everything else (Linux/Win/CUDA/CPU/Intel) -> LoRA adapter (~67 MB) -> models/qwen3vl2b_caption_v1_lora/
#     (the base Qwen/Qwen3-VL-2B-Instruct is fetched from the HF Hub on first run)
#
# src/local_pipeline.py + the UI auto-detect which backend to use and look here.
# Override the choice:  ./download_model.sh mlx   |   ./download_model.sh lora   |   ./download_model.sh both
set -euo pipefail
cd "$(dirname "$0")"

REL="https://github.com/sid-raj-uc/cephalopods/releases/download/caption-student-v1"
MLX_ZIP="qwen3vl2b_caption_v1_mlx_4bit.zip"
LORA_ZIP="qwen3vl2b_caption_v1_lora.zip"

# decide what to fetch
choice="${1:-auto}"
if [ "$choice" = "auto" ]; then
  if [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then choice="mlx"; else choice="lora"; fi
fi

mkdir -p models

fetch() {  # $1 = zip name, $2 = expected dir under models/, $3 = sentinel file, $4 = human size
  local zip="$1" dir="models/$2" sentinel="$3" size="$4"
  if [ -f "$dir/$sentinel" ]; then echo "Already present: $dir — skipping."; return; fi
  echo "Downloading $2 ($size)..."
  curl -L --fail --progress-bar -o "models/_dl.zip" "$REL/$zip"
  unzip -q -o "models/_dl.zip" -d models/
  rm -f "models/_dl.zip"
  [ -f "$dir/$sentinel" ] || { echo "ERROR: unzip did not produce $dir/$sentinel"; exit 1; }
  echo "Done: $dir"
}

case "$choice" in
  mlx)  fetch "$MLX_ZIP"  "qwen3vl2b_caption_v1_mlx_4bit" "model.safetensors"   "~1.7 GB" ;;
  lora) fetch "$LORA_ZIP" "qwen3vl2b_caption_v1_lora"     "adapter_model.safetensors" "~67 MB" ;;
  both) fetch "$MLX_ZIP"  "qwen3vl2b_caption_v1_mlx_4bit" "model.safetensors"   "~1.7 GB"
        fetch "$LORA_ZIP" "qwen3vl2b_caption_v1_lora"     "adapter_model.safetensors" "~67 MB" ;;
  *) echo "usage: $0 [auto|mlx|lora|both]"; exit 1 ;;
esac
