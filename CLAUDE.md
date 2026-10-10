# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

StoryForge is a self-hosted interactive storybook running on a Jetson Orin Nano (8 GB shared CPU/GPU memory plus an 8 GB swap file, aarch64, no Docker). A local Qwen2.5-3B served by llama-server writes one page at a time, Hugging Face (or local stable-diffusion.cpp) paints a background for each page, and Piper or the browser narrates.

## Live server — teammates are using it

- Never start, stop or restart the live StoryForge (port 8000) or llama-server (port 8080) without asking first. That includes `./scripts/start.sh` and `./scripts/stop.sh`, and `stop.sh --all` is especially risky.
- To test changes, run a separate instance on port 8001 that reuses the live llama-server and loads no model of its own. Give it its own data and log folders and turn HTTPS off, otherwise it collides with the live instance's sessions, pid files and port 8443:
  ```
  mkdir -p /tmp/sf-test/{data,logs,images}
  PORT=8001 HTTPS_PORT=0 IMAGE_MODE=huggingface HF_FALLBACK_LOCAL=0 \
    DATA_DIR=/tmp/sf-test/data LOG_DIR=/tmp/sf-test/logs IMAGE_DIR=/tmp/sf-test/images \
    /home/akw/jetson-ai/.venv/bin/python -m uvicorn app:app --host 127.0.0.1 --port 8001
  ```
  Stop only that process when you're done.
- Only one GPU job can run at a time, and a local picture alone takes free memory down to about 800 MB. Keep `HF_FALLBACK_LOCAL=0` (or `IMAGE_MODE=off`) on the test instance so it never runs sd-cli. Ask before generating images (including `./scripts/benchmark_sd.sh`) while the live server is busy.
- Leave the old app files in `/home/akw/jetson-ai/` alone (`app.py`, `app_backup.py`, `storyforge_app_with_illustrations.py`, the zips). The old app also uses port 8000.

## Commands

- Python env: the shared venv `/home/akw/jetson-ai/.venv`. There is no requirements.txt or pyproject. Dependencies are fastapi, uvicorn and pydantic, plus pillow for tests and piper-tts (optional).
- Tests (plain unittest, using a fake llama-server, fake Hugging Face router and fake `sd-cli`; no GPU, no internet; ports 18000/18001/18080/18091): `/home/akw/jetson-ai/.venv/bin/python tests/test_storyforge.py` (system `python3` has no uvicorn, so the test apps never start). Single test: append e.g. `Flow.test_02_only_the_opening_is_written`. The `Flow` tests are numbered and depend on running in order.
- Fake-model demo: `./tests/run_demo.sh` (port 8010, fake LLM on 18090).
- Health: `curl -s http://localhost:8000/api/health | python3 -m json.tool` (use 8001 for the test instance).
- No formatter, linter or CI is configured.

## Architecture notes

- `app.py` is a FastAPI app with routes under `/api/...`. An `asyncio.Lock` allows only one page to be written at a time.
- `storyforge/config.py` defines a frozen `Settings` dataclass. Values come from `settings.env`, env vars override them (`STORYFORGE_SETTINGS` points to a different file), and numbers are clamped. `scripts/_settings.sh` loads the same file for the shell scripts.
- `storyforge/llm.py` uses stdlib `http.client` to call llama-server's `/v1/chat/completions`. Only a running summary and the latest page are sent, because llama-server runs with `-c 2048`. Prompts and the validation and repair of model replies live here too.
- Secrets (`HF_TOKEN`) live in `.env` (gitignored); everything else in `settings.env`. `config.py` reads both (env vars > `.env` > `settings.env`). Never log, print or return the token; `Settings.hf_token` is `repr=False`. Tests point `STORYFORGE_ENV_FILE` at a fake file so the real key is never used.
- Pictures are page *backgrounds only* (no characters). The child's hero drawing (`/api/stories/{id}/hero`, placeholder `static/art/hero-placeholder.png` until `/draw` exists) is layered on top in the browser. `IMAGE_MODE=huggingface|local|off`. If nothing can paint a page (and the mode isn't `off`), `ImageWorker.use_preset` shows the best-matching `static/art/preset-<place>.jpg` (castle, beach, village, space, underwater; matched by `PRESET_WORDS`) instead (`image.source == "preset"`); retrying a preset page tries real painting again.
- Cloud mode (`storyforge/hf_images.py`): FLUX.1-schnell via the `nscale` provider at `router.huggingface.co/nscale/v1/images/generations` (the `hf-inference` provider returns 410 for this model). Each page's JSON starts with a plan (`setting`, `visual_details` 2-3 items, `action` from `llm.HERO_ACTIONS`; a drawn hero also gets `hero_name` before the text), then the text is written from it. `llm.extract_plan` spots the finished plan mid-stream, so the background (setting + details) starts while Qwen is still writing; 20 s limit, then local fallback once the page is ready. The 3B model sometimes names characters in the plan; `images.without_characters`/`names_a_character` strip them from background prompts, so keep that filter. Backgrounds also drop every word of the hero's kind, features and name (`images.hero_words`, colours and filler excepted) and any detail about something worn or held (`WORN_OR_HELD_WORDS`), and the style is followed by `images.COMPOSITION` (an open, empty centre, said positively because FLUX ignores negatives); otherwise "Cowboy Bill's hat" becomes a giant hat in a field. Stories stay gentle, but never name weapons in general prompts: the 3B model adds whatever is named. Only when the vision description itself has a weapon word (`llm._WEAPON`, checked in Python) does `llm.weapon_rule` add one "toy or tool" line. The vision prompt describes only what is clearly drawn and has no examples of any kind (examples get drawn in). The vision input stays a 512 px, tightly cropped, white-flattened JPEG: compared on saved drawings, 768 px (with or without padding) made Llama-4-Scout invent more. Each request logs image size, full prompt, model and raw answer (`hero vision request`/`hero vision answer`). The optional "Who is this?" (`who`) goes to both the vision prompt and the story (`hero_who`) and wins over the model's guess.
- `/draw` (`static/draw.*`, vanilla canvas + pointer events) posts the cropped transparent PNG and a white-backed JPEG to `POST /api/stories/drawn`, which saves the hero, gets a one-sentence description from `storyforge/vision.py` (HF router chat completions, `VISION_MODELS` tried in order; falls back to a generic description so the story still starts), stores it as `hero_description`/`character_look`, starts page 1 and returns; the browser then opens `/?story=<id>`. Gemini is not wired in: the HF vision models work with our key. Node isn't installed, so check frontend JS in a browser.
- The hero layer is animated in the browser from each page's `hero_action`: CSS keyframes in `style.css` ("Hero animation", nested `.hero-stage > .hero-move > .hero-flip > .hero-act > .hero-breath > img`) plus `static/hero.js` (line-boil SVG filter re-seeding, particles, reduced motion). `llm.HERO_ACTIONS`, `hero.js` ACTIONS and the CSS must list the same actions (a test checks). Preview all of them at `/static/hero-demo.html`. Page 1 gets the drawn hero as `llm.hero_notes` (kind + features), never the full vision sentence, or the 3B model copies it word for word.
- The reader's choice (`choice`/`custom` kinds) is softened in Python (`llm.soften_choice`), forced as the plan's first key via a JSON-schema `const` `event`, repeated as the prompt's last line ("This page must show: ..."), and checked in Python (`llm.choice_followed`): retry once, then prepend `llm.bridge_sentence`. Later pages get only a 2-sentence summary plus the previous page's last sentence.
- `storyforge/images.py` runs a single-worker queue for local sd-cli. Each picture is a fresh `sd-cli` subprocess so memory is released afterwards (don't switch to a persistent sd-server). It checks free memory against `SD_MIN_FREE_MB` before starting, and fast mode (LCM-LoRA + TAESD) falls back to classic settings on failure.
- Keep `SD_EXTRA_ARGS` (`--diffusion-fa --vae-tiling --backend te=cpu --max-vram 0`) unless a benchmark shows a change is better.
- `data/` and `logs/` hold live runtime state (sessions, pid files, `timings.jsonl`), not source.

## Conventions

- Use `from __future__ import annotations` and type hints. Prefer the stdlib over new dependencies.
- User-facing text (UI, error messages, README) is written for non-developers: plain, friendly, no jargon.
- The frontend is vanilla JS/HTML/CSS in `static/` with no build step.
