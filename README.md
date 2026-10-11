# StoryForge

**Draw a hero, and watch an AI storybook bring them to life.**

StoryForge is an interactive picture book for kids, built at **Knight Hacks IX** on an
**NVIDIA Jetson Orin Nano**. A child draws a hero (or types an idea), and StoryForge writes a
choose-your-own-adventure story around them one page at a time: illustrated, animated, narrated,
and matched to the child's reading level. An LED strip in the book glows in the colors of each page.

---

## What it does

- **Draw your hero.** A finger-friendly drawing page turns the child's drawing into a transparent
  character that stars in every page of the story. An optional "Who is this?" box lets the child
  name and describe it.
- **A real adventure, not a wall of text.** The story model first plans an outline (a goal, a
  problem, friends and a happy ending), then writes each page as a step toward it. Every page ends
  with real choices, or the child can type their own idea.
- **Illustrated scenes.** Each page gets a generated storybook background. The child's drawing
  stays the hero, layered on top and animated to act out the page (walking, jumping, flying and more).
- **Read-aloud narration.** Every page is narrated with a warm storyteller voice.
- **Reading levels.** Pre-reader, Early reader (K-2) and Confident reader (3-5) change page length,
  sentence length and vocabulary.
- **Tap-a-word definitions.** Tricky words are picked by word-frequency difficulty scoring (not by
  the AI) and get kid-friendly definitions you can tap to see and hear.
- **Ambient lighting.** A 30-LED strip maps each page's background, top to bottom, into colors on
  the physical book.
- **Built for kids.** Gentle-content checks, structured model output validated in Python, and
  fallbacks at every step so a story never stalls.

## How it works

| Part | Technology |
|---|---|
| Story writing (on device) | Qwen 2.5 3B Instruct (Q4_K_M) via llama.cpp `llama-server` on the Jetson |
| Story structure and safety | Outline + page "beats", structured JSON, Python validation and filters |
| Understanding the drawing | Hugging Face vision models (Llama 4 Scout, Gemma 3 fallback) |
| Backgrounds | FLUX.1-schnell via Hugging Face Inference, with local Stable Diffusion and preset fallbacks |
| Narration | ElevenLabs (cached per page), browser speech as fallback |
| Reading support | `wordfreq` (word difficulty) and `textstat` (readability) |
| Animation | HTML, CSS, SVG filters and JavaScript |
| Lighting | ESP32 + FastLED driving WS2812B LEDs over USB serial; colors computed on the Jetson |
| Backend | Python, FastAPI, Uvicorn, Pydantic, async processing |
| Front end | Plain HTML/CSS/JS, used on an iPad inside a 3D-printed book enclosure |

```
iPad (browser) ──► FastAPI on Jetson ──► llama-server (Qwen, local)
                         │ ├──► Hugging Face (vision + backgrounds)
                         │ ├──► ElevenLabs (narration)
                         │ └──► ESP32 over USB ──► LED strip
```

## Run it

Requirements: a Jetson (or any Linux machine) with Python 3.10+, `llama.cpp` built with
`llama-server`, and the Qwen 2.5 3B Instruct GGUF model.

```bash
git clone https://github.com/thestrikeflame/storyforge.git
cd storyforge
python3 -m venv .venv && . .venv/bin/activate
pip install fastapi uvicorn pydantic pillow huggingface_hub wordfreq textstat pyserial numpy
```

Create a `.env` file (never committed) with your keys:

```bash
HF_TOKEN=hf_...
ELEVENLABS_API_KEY=...
ELEVENLABS_VOICE_ID=...
```

Check the model paths and options in `settings.env`, then start:

```bash
./scripts/start.sh          # starts llama-server if needed, then the app on port 8000
./scripts/stop.sh           # stop the app (add --all to stop llama-server too)
```

Open `http://<jetson-ip>:8000`. Everything degrades gracefully: without cloud keys it uses preset
backgrounds and browser narration.

**LED strip (optional):** flash `firmware/esp32_leds/esp32_leds.ino` to an ESP32 (FastLED, data on
GPIO 5, 30 WS2812B LEDs, separate 5V supply with a shared ground) and plug it into the Jetson over USB.

## Tests

```bash
python3 tests/test_storyforge.py    # unit tests with fake models, no GPU needed
./tests/run_demo.sh                 # full site with fake models on http://localhost:8010
```

## Team

Built in 36 hours at Knight Hacks IX (UCF, October 2026) by
Ahnaf Chowdhury, Ariel Kurelo Wilk, Maribel Garcia and Sara Benyamine.

Developed with help from AI coding tools (Claude Code).
