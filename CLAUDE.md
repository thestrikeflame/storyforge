# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

StoryForge is a self-hosted interactive storybook running on a Jetson Orin Nano (8 GB shared CPU/GPU memory, no swap, aarch64, no Docker). A local Qwen2.5-3B served by llama-server writes one page at a time, stable-diffusion.cpp paints the picture afterwards, and Piper or the browser narrates.

## Live server — teammates are using it

- Never start, stop or restart the live StoryForge (port 8000) or llama-server (port 8080) without asking first. That includes `./scripts/start.sh` and `./scripts/stop.sh`, and `stop.sh --all` is especially risky.
- To test changes, run a separate instance on port 8001 that reuses the live llama-server and loads no model of its own. Give it its own data and log folders and turn HTTPS off, otherwise it collides with the live instance's sessions, pid files and port 8443:
  ```
  mkdir -p /tmp/sf-test/{data,logs}
  PORT=8001 HTTPS_PORT=0 IMAGES_ENABLED=0 DATA_DIR=/tmp/sf-test/data LOG_DIR=/tmp/sf-test/logs \
    /home/akw/jetson-ai/.venv/bin/python -m uvicorn app:app --host 127.0.0.1 --port 8001
  ```
  Stop only that process when you're done.
- Only one GPU job can run at a time, and a picture alone takes free memory down to about 800 MB. Keep `IMAGES_ENABLED=0` on the test instance. Ask before generating images (including `./scripts/benchmark_sd.sh`) while the live server is busy.
- Leave the old app files in `/home/akw/jetson-ai/` alone (`app.py`, `app_backup.py`, `storyforge_app_with_illustrations.py`, the zips). The old app also uses port 8000.

## Commands

- Python env: the shared venv `/home/akw/jetson-ai/.venv`. There is no requirements.txt or pyproject. Dependencies are fastapi, uvicorn and pydantic, plus pillow for tests and piper-tts (optional).
- Tests (plain unittest, using a fake llama-server and a fake `sd-cli`, no GPU, ports 18000/18080): `python3 tests/test_storyforge.py`. Single test: `python3 tests/test_storyforge.py Flow.test_02_only_the_opening_is_written`. The `Flow` tests are numbered and depend on running in order.
- Fake-model demo: `./tests/run_demo.sh` (port 8010, fake LLM on 18090).
- Health: `curl -s http://localhost:8000/api/health | python3 -m json.tool` (use 8001 for the test instance).
- No formatter, linter or CI is configured.

## Architecture notes

- `app.py` is a FastAPI app with routes under `/api/...`. An `asyncio.Lock` allows only one page to be written at a time.
- `storyforge/config.py` defines a frozen `Settings` dataclass. Values come from `settings.env`, env vars override them (`STORYFORGE_SETTINGS` points to a different file), and numbers are clamped. `scripts/_settings.sh` loads the same file for the shell scripts.
- `storyforge/llm.py` uses stdlib `http.client` to call llama-server's `/v1/chat/completions`. Only a running summary and the latest page are sent, because llama-server runs with `-c 2048`. Prompts and the validation and repair of model replies live here too.
- `storyforge/images.py` runs a single-worker queue. Each picture is a fresh `sd-cli` subprocess so memory is released afterwards (don't switch to a persistent sd-server). It checks free memory against `SD_MIN_FREE_MB` before starting, and fast mode (LCM-LoRA + TAESD) falls back to classic settings on failure.
- Keep `SD_EXTRA_ARGS` (`--diffusion-fa --vae-tiling --backend te=cpu --max-vram 0`) unless a benchmark shows a change is better.
- `data/` and `logs/` hold live runtime state (sessions, pid files, `timings.jsonl`), not source.

## Conventions

- Use `from __future__ import annotations` and type hints. Prefer the stdlib over new dependencies.
- User-facing text (UI, error messages, README) is written for non-developers: plain, friendly, no jargon.
- The frontend is vanilla JS/HTML/CSS in `static/` with no build step.
