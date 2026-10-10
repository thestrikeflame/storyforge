#!/usr/bin/env bash
# Measures how long pictures take on THIS machine with different settings, so you can pick
# the fastest one whose pictures you still like. Nothing is changed; results are only printed
# and saved. Run it while the Qwen server is running, because that is the real situation.
#
#   ./scripts/benchmark_sd.sh            the standard set (takes several minutes)
#   ./scripts/benchmark_sd.sh quick      only 20 and 12 steps
#
# Afterwards open the pictures in benchmark_results/<date>/ and set SD_STEPS (and, only if
# you want, SD_WIDTH/SD_HEIGHT) in settings.env.
set -uo pipefail
source "$(dirname "$0")/_settings.sh"
SD_CLI="${SD_CLI:?SD_CLI is not set in settings.env}"
SD_MODEL="${SD_MODEL:?SD_MODEL is not set in settings.env}"
EXTRA="${SD_EXTRA_ARGS:---diffusion-fa --vae-tiling --backend te=cpu --max-vram 0}"
DEFAULT_STYLE="children's storybook watercolor illustration"
PROMPT="${SD_STYLE:-$DEFAULT_STYLE}, cute purple baby dragon with large expressive eyes beside a small wooden door in a colorful enchanted forest"
OUT="$PROJECT_DIR/benchmark_results/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"

[ -x "$SD_CLI" ] || { echo "sd-cli not found at $SD_CLI"; exit 1; }
[ -f "$SD_MODEL" ] || { echo "Model not found at $SD_MODEL"; exit 1; }

free_mb() { awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo; }

echo "== What this sd-cli build supports =="
HELP="$("$SD_CLI" --help 2>&1)"
for flag in --diffusion-fa --vae-tiling --backend --max-vram --steps --seed --sampling-method --cfg-scale --taesd --lora-model-dir; do
  if grep -q -- "$flag" <<<"$HELP"; then echo "  yes  $flag"; else echo "  NO   $flag"; fi
done
for flag in $EXTRA; do
  case "$flag" in --*) grep -q -- "$flag" <<<"$HELP" || echo "  WARNING: SD_EXTRA_ARGS uses $flag, which --help does not list." ;; esac
done
if [ -x "$(dirname "$SD_CLI")/sd-server" ]; then
  echo "  note: this build also contains sd-server (keeps the model loaded between pictures). See README, 'Going faster'."
fi
if curl -s -o /dev/null --max-time 2 "$LLM_URL/health"; then echo "Qwen server: running (good, this is the realistic test)"; else echo "Qwen server: NOT running. Times and memory will look better than in real use."; fi
echo "Free memory now: $(free_mb) MB"
echo

printf "%-26s %9s %14s %s\n" "setting" "seconds" "lowest free MB" "result" | tee "$OUT/results.txt"

LORA="${SD_LCM_LORA:-$JETSON_AI/models/images/fast/lcm-lora-sdv1-5.safetensors}"
TAESD="${SD_TAESD:-$JETSON_AI/models/images/fast/taesd.safetensors}"
EXTRA_NO_TILING="${EXTRA//--vae-tiling/}"
BASE_PROMPT="$PROMPT"
BASE_EXTRA="$EXTRA"

run() {  # run <label> <width> <height> <steps> [extra args...]
  local label="$1" w="$2" h="$3" steps="$4"; shift 4
  local file="$OUT/${label// /_}.png" low start end status
  low="$(free_mb)"
  start="$(date +%s.%N)"
  # shellcheck disable=SC2086
  "$SD_CLI" -m "$SD_MODEL" -p "$PROMPT" -W "$w" -H "$h" --steps "$steps" $EXTRA --seed 42 "$@" -o "$file" > "$OUT/${label// /_}.log" 2>&1 &
  local pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    local now; now="$(free_mb)"; [ "$now" -lt "$low" ] && low="$now"
    sleep 1
  done
  wait "$pid"; status=$?
  end="$(date +%s.%N)"
  local seconds; seconds="$(awk -v a="$start" -v b="$end" 'BEGIN {printf "%.1f", b-a}')"
  if [ "$status" -eq 0 ] && [ -s "$file" ]; then
    printf "%-26s %9s %14s %s\n" "$label" "$seconds" "$low" "ok" | tee -a "$OUT/results.txt"
  else
    printf "%-26s %9s %14s %s\n" "$label" "$seconds" "$low" "FAILED (see ${label// /_}.log)" | tee -a "$OUT/results.txt"
  fi
  sleep 3   # let memory settle before the next run
}

if [ "${1:-}" = "quick" ]; then
  run "512 20 steps" 512 512 20
  run "512 12 steps" 512 512 12
else
  run "512 20 steps" 512 512 20     # the known-working baseline
  run "512 15 steps" 512 512 15
  run "512 12 steps" 512 512 12
  run "512 10 steps" 512 512 10
  run "448 15 steps" 448 448 15
  run "384 15 steps" 384 384 15
fi

# Fast mode, only when the files from ./scripts/get_fast_models.sh are present.
if [ -f "$TAESD" ] && grep -q -- "--taesd" <<<"$HELP"; then
  EXTRA="$EXTRA_NO_TILING"
  run "512 15 steps +taesd" 512 512 15 --taesd "$TAESD"
  EXTRA="$BASE_EXTRA"
fi
if [ -f "$LORA" ] && grep -q -- "--lora-model-dir" <<<"$HELP"; then
  PROMPT="$BASE_PROMPT<lora:$(basename "${LORA%.*}"):1>"
  LCM=(--lora-model-dir "$(dirname "$LORA")" --cfg-scale 1 --sampling-method lcm)
  run "512 lcm 6 steps" 512 512 6 "${LCM[@]}"
  if [ -f "$TAESD" ] && grep -q -- "--taesd" <<<"$HELP"; then
    EXTRA="$EXTRA_NO_TILING"
    run "512 lcm 6 +taesd" 512 512 6 "${LCM[@]}" --taesd "$TAESD"
    run "512 lcm 4 +taesd" 512 512 4 "${LCM[@]}" --taesd "$TAESD"
    EXTRA="$BASE_EXTRA"
  fi
  PROMPT="$BASE_PROMPT"
fi

echo
echo "Pictures and logs: $OUT"
echo "Compare the pictures, then put the steps (and size) you prefer into settings.env and restart."
