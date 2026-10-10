#!/usr/bin/env bash
# Runs StoryForge with the stand-in models so the website can be tried on any computer
# (no GPU, no Qwen, no Stable Diffusion). Open http://localhost:8010 and press Ctrl+C to stop.
# Needs:  pip install fastapi uvicorn pillow
set -euo pipefail
cd "$(dirname "$0")/.."
TMP="$(mktemp -d)"
python3 tests/fake_llama.py 18090 &
LLAMA=$!
trap 'kill $LLAMA 2>/dev/null; rm -rf "$TMP"' EXIT
sleep 0.5
curl -s -X POST localhost:18090/control -d '{"delay": 0.03}' >/dev/null || true
STORYFORGE_SETTINGS="$TMP/none.env" LLM_URL=http://127.0.0.1:18090 PORT="${PORT:-8010}" HTTPS_PORT=0 \
SD_CLI="$PWD/tests/fake_sd_cli" SD_MODEL="$PWD/tests/fake_sd_cli" SD_STEPS=12 SD_MIN_FREE_MB=0 \
IMAGE_DIR="$TMP/images" DATA_DIR="$TMP/data" LOG_DIR="$TMP/logs" CERT_DIR="$TMP/certs" TTS_MODE=browser \
FAKE_SD_STEP_SECONDS="${FAKE_SD_STEP_SECONDS:-0.6}" \
python3 -m uvicorn app:app --host 127.0.0.1 --port "${PORT:-8010}" --log-level warning
