#!/usr/bin/env bash
set -euo pipefail

ASSETS="${ASSETS:-/content/assets}"
mkdir -p "$ASSETS/VGGT-1B" "$ASSETS/vkitti" "$ASSETS/hf"

echo "=== Installing VGGT training dependencies ==="
pip install -q hydra-core omegaconf iopath fvcore wcmatch tensorboard opencv-python-headless einops safetensors huggingface_hub pyyaml

echo "=== Downloading official public facebook/VGGT-1B checkpoint ==="
if [ ! -f "$ASSETS/VGGT-1B/model.pt" ]; then
  curl -L --fail --retry 3 -o "$ASSETS/VGGT-1B/model.pt" \
    "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
fi
ls -lh "$ASSETS/VGGT-1B/model.pt"

echo "=== Downloading official public Virtual KITTI 2.0.3 (per training/data/preprocess/vkitti.sh) ==="
cd "$ASSETS/vkitti"
if [ ! -d "$ASSETS/vkitti/Scene01" ]; then
  for f in vkitti_2.0.3_textgt.tar.gz vkitti_2.0.3_rgb.tar vkitti_2.0.3_depth.tar; do
    if [ ! -f "$f" ]; then
      curl -L --fail --retry 3 -o "$f" "https://download.europe.naverlabs.com/virtual_kitti_2.0.3/$f"
    fi
  done
  tar -xzf vkitti_2.0.3_textgt.tar.gz
  tar -xf vkitti_2.0.3_rgb.tar
  tar -xf vkitti_2.0.3_depth.tar
  # Flatten if extracted under vkitti_2.0.3_* subdirectories so Scene01/... sits directly in $ASSETS/vkitti
  for sub in vkitti_2.0.3_textgt vkitti_2.0.3_rgb vkitti_2.0.3_depth; do
    if [ -d "$sub" ]; then
      cp -rn "$sub"/* . || true
      rm -rf "$sub"
    fi
  done
  rm -f vkitti_2.0.3_textgt.tar.gz vkitti_2.0.3_rgb.tar vkitti_2.0.3_depth.tar
fi
ls -la "$ASSETS/vkitti"
find "$ASSETS/vkitti" -maxdepth 4 -path "*/*/*/rgb/*" | sort | head -n 20
