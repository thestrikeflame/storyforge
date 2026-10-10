#!/usr/bin/env bash
# Stops StoryForge.   ./scripts/stop.sh          stop the website only
#                     ./scripts/stop.sh --all    also stop the Qwen server this project started
set -uo pipefail
source "$(dirname "$0")/_settings.sh"
cd "$PROJECT_DIR"

stop_pid_file() {
  local file="$1" name="$2"
  if [ -f "$file" ]; then
    local pid; pid="$(cat "$file")"
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" && echo "Stopped $name (process $pid)."
      for _ in 1 2 3 4 5 6 7 8 9 10; do kill -0 "$pid" 2>/dev/null || break; sleep 0.5; done
    else
      echo "$name was not running."
    fi
    rm -f "$file"
  else
    echo "No record of a running $name."
  fi
}

stop_pid_file logs/storyforge.pid "StoryForge"
if [ "${1:-}" = "--all" ]; then stop_pid_file logs/llama-server.pid "the Qwen server"; fi
