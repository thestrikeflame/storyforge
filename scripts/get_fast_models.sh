#!/usr/bin/env bash
# Downloads the two small files that make pictures much faster (about 145 MB together):
#   LCM-LoRA  lets Stable Diffusion 1.5 finish in about 6 steps instead of 15 to 20
#   TAESD     a tiny decoder that replaces the slow last stage of each picture
# Your existing model is kept and still used. Needs an internet connection on the Jetson.
# To undo: set SD_FAST=0 in settings.env (or delete the "fast" folder) and restart.
set -euo pipefail
source "$(dirname "$0")/_settings.sh"
LORA="${SD_LCM_LORA:-$JETSON_AI/models/images/fast/lcm-lora-sdv1-5.safetensors}"
TAESD="${SD_TAESD:-$JETSON_AI/models/images/fast/taesd.safetensors}"
mkdir -p "$(dirname "$LORA")" "$(dirname "$TAESD")"

fetch() {  # fetch <url> <destination> <minimum size in bytes>
  local url="$1" dest="$2" min="$3"
  if [ -f "$dest" ] && [ "$(stat -c %s "$dest")" -ge "$min" ]; then
    echo "Already there: $dest"; return
  fi
  echo "Downloading $(basename "$dest") ..."
  curl -L --fail --progress-bar -o "$dest.part" "$url"
  if [ "$(stat -c %s "$dest.part")" -lt "$min" ]; then
    rm -f "$dest.part"; echo "The download of $(basename "$dest") is too small to be right. Try again."; exit 1
  fi
  mv "$dest.part" "$dest"
}

fetch "https://huggingface.co/latent-consistency/lcm-lora-sdv1-5/resolve/main/pytorch_lora_weights.safetensors" "$LORA" 100000000
fetch "https://huggingface.co/madebyollin/taesd/resolve/main/diffusion_pytorch_model.safetensors" "$TAESD" 5000000

echo
if [ -x "${SD_CLI:-}" ]; then
  HELP="$("$SD_CLI" --help 2>&1)"
  for flag in --taesd --lora-model-dir; do
    grep -q -- "$flag" <<<"$HELP" && echo "sd-cli supports $flag" || echo "NOTE: this sd-cli does not list $flag, so that part will not be used."
  done
fi
echo
echo "Done. Restart StoryForge:   ./scripts/stop.sh; ./scripts/start.sh"
echo "The start-up log should then say:  picture mode: fast (lcm+taesd), 6 steps"
