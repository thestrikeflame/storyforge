# Shared by the other scripts: finds the project folder and reads settings.env.
# Values already set in the environment win over the file.
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SETTINGS_FILE="${STORYFORGE_SETTINGS:-$PROJECT_DIR/settings.env}"
if [ -f "$SETTINGS_FILE" ]; then
  while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in ''|\#*) continue ;; esac
    key="${line%%=*}"; value="${line#*=}"
    case "$key" in *[!A-Za-z0-9_]*|'') continue ;; esac
    if [ -z "${!key+x}" ]; then export "$key=$value"; fi
  done < "$SETTINGS_FILE"
fi
JETSON_AI="${JETSON_AI:-/home/akw/jetson-ai}"
VENV="${VENV:-$JETSON_AI/.venv}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
LLM_URL="${LLM_URL:-http://127.0.0.1:8080}"
LLAMA_SERVER="${LLAMA_SERVER:-$JETSON_AI/llama.cpp/build/bin/llama-server}"
LLAMA_MODEL="${LLAMA_MODEL:-$JETSON_AI/models/qwen2.5-3b-instruct-q4_k_m.gguf}"
if [ -x "$VENV/bin/python" ]; then PYTHON="$VENV/bin/python"; else PYTHON="$(command -v python3)"; fi
