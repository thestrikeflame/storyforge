#!/usr/bin/env bash
# Starts StoryForge. It also starts the Qwen server when that is not running yet.
#
#   ./scripts/start.sh                run in this terminal (press Ctrl+C to stop)
#   ./scripts/start.sh --background   keep running after the terminal is closed
set -euo pipefail
source "$(dirname "$0")/_settings.sh"
cd "$PROJECT_DIR"
mkdir -p logs

if ! "$PYTHON" -c "import fastapi, uvicorn" 2>/dev/null; then
  echo "FastAPI or Uvicorn is missing from $PYTHON."
  echo "Install them with:  $PYTHON -m pip install fastapi uvicorn"
  exit 1
fi

# 1. Is something already using the website port? (Usually the old app.py.)
if "$PYTHON" - "$PORT" <<'PY'
import socket, sys
with socket.socket() as s:
    s.settimeout(0.5)
    sys.exit(0 if s.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0 else 1)
PY
then
  echo "Port $PORT is already in use, probably by the previous StoryForge or the old app.py."
  echo "Stop it first:   ./scripts/stop.sh        (or find it with:  ss -ltnp | grep :$PORT )"
  exit 1
fi

# 2. Make sure the story writer (Qwen through llama-server) is running.
if curl -s -o /dev/null --max-time 2 "$LLM_URL/health"; then
  echo "Qwen server: already running at $LLM_URL"
else
  if [ ! -x "$LLAMA_SERVER" ] || [ ! -f "$LLAMA_MODEL" ]; then
    echo "The Qwen server is not running and could not be started automatically:"
    echo "  program: $LLAMA_SERVER"
    echo "  model:   $LLAMA_MODEL"
    echo "StoryForge will start anyway and show a friendly message until Qwen is available."
  else
    echo "Starting the Qwen server (output goes to logs/llama-server.log) ..."
    # The known-working launch command, unchanged.
    nohup "$LLAMA_SERVER" -m "$LLAMA_MODEL" -ngl 99 -c 2048 --host 0.0.0.0 --port 8080 \
      > logs/llama-server.log 2>&1 &
    echo $! > logs/llama-server.pid
    for _ in $(seq 1 90); do
      if curl -s --max-time 2 "$LLM_URL/health" | grep -q '"ok"'; then break; fi
      sleep 2
    done
    if curl -s --max-time 2 "$LLM_URL/health" | grep -q '"ok"'; then
      echo "Qwen server: ready"
    else
      echo "Qwen is still loading or failed to start. Check logs/llama-server.log"
    fi
  fi
fi

# 3. Start the website.
ADDRESS="$(hostname -I 2>/dev/null | awk '{print $1}')"
echo
echo "StoryForge:  http://${ADDRESS:-localhost}:$PORT"
if [ -f "${CERT_DIR:-$PROJECT_DIR/certs}/cert.pem" ] && [ "${HTTPS_PORT:-8443}" != "0" ]; then
  echo "With microphone (secure address):  https://${ADDRESS:-localhost}:${HTTPS_PORT:-8443}"
else
  echo "Microphone on other devices needs the secure address. Set it up once with: ./scripts/make_cert.sh"
fi
echo

if [ "${1:-}" = "--background" ]; then
  nohup "$PYTHON" -m uvicorn app:app --host "$HOST" --port "$PORT" --log-level warning > logs/web.log 2>&1 &
  echo $! > logs/storyforge.pid
  echo "Running in the background. Stop it with ./scripts/stop.sh   Log: logs/storyforge.log"
else
  echo $$ > logs/storyforge.pid
  exec "$PYTHON" -m uvicorn app:app --host "$HOST" --port "$PORT" --log-level warning
fi
