# StoryForge

An interactive storybook for the Jetson Orin Nano. You describe an idea, Qwen writes **one page
at a time**, the page is read aloud while Stable Diffusion paints its picture, and then you decide
what happens next.

This is a fresh build written from the project brief. It was **not** made by editing your old
`app.py`, because that file was not available. Your old app, models and backups are not touched:
StoryForge lives in its own folder and only reads the model files.

---

## 1. Install on the Jetson (about 5 minutes)

Copy `storyforge.zip` to the Jetson, then in a terminal:

```bash
cd /home/akw/jetson-ai
unzip storyforge.zip            # creates /home/akw/jetson-ai/storyforge
cd storyforge
```

Nothing new needs to be installed for the basic experience. It uses FastAPI and Uvicorn, which
your existing virtual environment already has. To check:

```bash
/home/akw/jetson-ai/.venv/bin/python -c "import fastapi, uvicorn; print('ok')"
```

If that prints an error instead of `ok`:

```bash
/home/akw/jetson-ai/.venv/bin/python -m pip install fastapi uvicorn
```

## 2. Start it

The old app uses port 8000, so stop it first (press Ctrl+C in its terminal). Then:

```bash
cd /home/akw/jetson-ai/storyforge
./scripts/start.sh
```

This starts the Qwen server with your known-working command if it is not already running, then
starts the website. Open **http://10.11.20.11:8000**. Press Ctrl+C to stop.

| Command | What it does |
|---|---|
| `./scripts/start.sh` | Start in this terminal |
| `./scripts/start.sh --background` | Keep running after the terminal closes |
| `./scripts/stop.sh` | Stop the website |
| `./scripts/stop.sh --all` | Stop the website and the Qwen server it started |

## 3. Turn on the microphone (one-time)

Browsers only allow the microphone on secure (`https`) pages. On the Jetson's own browser
`http://localhost:8000` already counts as secure. For tablets and other computers:

```bash
./scripts/make_cert.sh          # creates a certificate for this Jetson's address
./scripts/stop.sh; ./scripts/start.sh
```

Then open **https://10.11.20.11:8443**. The first time, the browser warns that the certificate is
self-made: choose *Advanced*, then *Proceed*. Typing always works, on either address.

Speech-to-text uses the browser's built-in recognition (Chrome, Edge, Safari). In Chrome that
needs an internet connection. Firefox has none, so it shows a message and you type instead, unless
you add the optional offline listener in section 6.

## 4. Check that it works

```bash
curl -s http://localhost:8000/api/health | python3 -m json.tool
```

`story_writer.ok` and `pictures.ok` should both be `true`. Then try a story in the browser and
watch the log in a second terminal:

```bash
tail -f /home/akw/jetson-ai/storyforge/logs/storyforge.log
```

The complete output of the most recent picture run is in `logs/sd-last.log`, and the log shows
the exact prompt used for every picture.

Each page and picture writes one line with its timings and free memory. The layout looks like this
(the numbers here are made up to show the format, they are not measurements):

```
page written     story=ab12 page=1  first words 1.4s, text complete 9.8s, all done 14.2s  tokens=310 (21.8 per second) ...
picture painted  story=ab12 page=1  71.3s  steps=15  512x512  free_mem before=3100MB lowest=1400MB
```

Averages are at `http://localhost:8000/api/metrics`.

### Tests that do not need the Jetson

These use stand-ins for Qwen and sd-cli, so they run on any computer and never touch the GPU or
your real stories (they need `pip install pillow`):

```bash
python3 tests/test_storyforge.py      # 17 automated tests
./tests/run_demo.sh                   # the full website with fake models on http://localhost:8010
```

## 5. Make pictures faster

### The big one: fast mode (recommended)

```bash
./scripts/get_fast_models.sh        # one-time download, about 145 MB
./scripts/stop.sh; ./scripts/start.sh
```

This adds two small helper files next to your existing model (which is kept and still used):
**LCM-LoRA**, so a picture needs about 6 steps instead of 15 and each step does half the work, and
**TAESD**, a tiny decoder for the last stage. The start-up log then says
`picture mode: fast (lcm+taesd), 6 steps`. If fast mode fails on your build, StoryForge repaints
that picture with the classic settings and switches fast mode off until the next restart, so a
story is never left without its picture. To turn it off yourself: `SD_FAST=0` in `settings.env`.

Fast-mode pictures are a little softer than 15 to 20 step ones. `SD_LCM_STEPS=8` is sharper and
slower, `4` is faster and rougher.

### Fine tuning

Picture time is almost all Stable Diffusion sampling, and it grows with the number of steps.
`settings.env` starts at **15 steps** instead of 20 (roughly a quarter less sampling work), with
the rest of your known-working command unchanged. To choose with real numbers:

```bash
./scripts/benchmark_sd.sh
```

It first prints which options your `sd-cli` build really supports, then times 20, 15, 12 and 10
steps at 512×512 and 15 steps at 448 and 384, recording seconds and lowest free memory, and saves
every picture in `benchmark_results/`. With the fast-mode files present it times those too. Look at the pictures, put the steps (and size, if you want)
you prefer into `settings.env`, and restart. To go back to the original: `SD_STEPS=20`.

Other things already done for speed and stability:

* Qwen finishes the page before the picture starts, so they do not compete for the GPU.
  (Choices appear when the picture is done, which keeps it that way. "Choose now" skips the wait.)
* The story text streams onto the page as it is written, and narration starts the moment the page
  text is complete, before the choices are written and long before the picture.
* One picture at a time, with a queue. Free memory is checked first (`SD_MIN_FREE_MB`): if there is
  too little, the picture is skipped with a message rather than risking a crash.
* Pictures for a story you have left are cancelled.
* **Reading level** (on the cover and on the drawing page): Pre-reader (30-45 words a page, very short
  sentences, narration a little slower), Early reader K-2 (45-60, the default) or Confident reader 3-5
  (60-80, richer words). It is saved with the story.
* **Tap a word:** once a page is finished, Qwen picks 3-4 tricky words on it and explains each in one
  sentence for the reading level. Those words get a dotted underline; tapping one shows the
  meaning and reads it aloud (with ElevenLabs, made once per word, or the browser's voice). This only
  runs while Qwen has nothing else to do: a new page always goes first, and the words simply
  appear a little later.

Going further (not done, because each needs a download or a measured decision on your hardware):

* **Keep the picture model loaded.** Each picture currently starts `sd-cli` fresh, which reloads
  the model. If your build has `sd-server` (the benchmark tells you), it can stay loaded, but then
  Qwen and Stable Diffusion hold memory at the same time. With 8 GB that may not fit.
* **A different picture model.** Plain Stable Diffusion 1.5 is weak at characters. A storybook or
  illustration fine-tune of SD 1.5 (converted to the same GGUF format) usually looks much better
  at the same speed. Point `SD_MODEL` at it; nothing else changes.
* **Free memory:** the desktop (GNOME) uses a lot. Running the Jetson without the desktop, or
  adding a swap file as a safety net, both help avoid out-of-memory failures.

## 6. Optional extras

**A natural narrator voice (ElevenLabs, online).** Put your key, and the voice you want, in `.env`
(never in `settings.env`, which is shared):

```bash
ELEVENLABS_API_KEY=sk_...
ELEVENLABS_VOICE_ID=JBFqnCBsd6RMkjVDRZzb   # "George", a warm storyteller; any voice ID from your ElevenLabs account works
```

and keep `TTS_MODE=elevenlabs` in `settings.env`. Each page is voiced once, as soon as its text is
finished, and kept in `data/audio`, so reading a page again costs nothing. If ElevenLabs fails,
takes more than 10 seconds or has no key, the browser's own voice reads that page instead.
Every voiced page writes a `narration ready ... characters=... credits=...` line to the log, so
you can keep an eye on your credits: `grep "voice=elevenlabs" logs/storyforge.log`.
On iPads and iPhones the first tap on the page turns the sound on (Safari only plays sound after a tap).
Set `TTS_MODE=browser` to go back to the browser's voice.

**A nicer narrator voice (Piper, offline).** Without it the browser's own voice reads the story.
Set `TTS_MODE=auto` (or `piper`) in `settings.env` to use it.

```bash
/home/akw/jetson-ai/.venv/bin/python -m pip install piper-tts
mkdir -p /home/akw/jetson-ai/models/voices && cd /home/akw/jetson-ai/models/voices
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
```

Restart StoryForge; `/api/health` should then show `"narration": "piper"`. If Piper ever fails,
the browser voice takes over automatically.

**Offline voice input (whisper.cpp).** Only needed for browsers without speech recognition or
without internet. Build whisper.cpp in `/home/akw/jetson-ai/whisper.cpp`, download `ggml-base.en.bin`
into its `models` folder, and restart. `/api/health` then shows `"offline_voice_input": true`.

**Your own cover picture.** Save an image as `static/art/landing.jpg` (or `.png`/`.webp`). It
replaces the drawn book on the first screen. Nothing is generated when the site opens.

## 7. Settings

Everything adjustable is in `settings.env`, with a comment on each line: page length, steps, picture
size, style words, memory limit, narration mode, ports. Edit, save, restart.

## 8. How stories are stored

Each story is a small file in `data/sessions/`, and its pictures are in
`/home/akw/jetson-ai/generated_images/` with names starting `sf_`. A story survives a browser
refresh and a restart ("Continue ..." appears under the cover buttons). Stories untouched for 14 days
are deleted at start-up, along with their `sf_` pictures. No other files in that folder are touched.

For each story the app keeps: your idea, the title, the main character's look, the setting, every
page, every decision, and a short running summary. Qwen is sent the summary and the latest page
only, never the whole history, so the 2048-token context is not exceeded on long stories.

## 9. Going back to the old app

Nothing of the old app was changed, so there is nothing to restore:

```bash
cd /home/akw/jetson-ai/storyforge && ./scripts/stop.sh
cd /home/akw/jetson-ai && source .venv/bin/activate
uvicorn app:app --host 0.0.0.0 --port 8000        # your previous app, as before
```

To remove StoryForge completely: `rm -r /home/akw/jetson-ai/storyforge`.

## 10. What was and was not tested

Tested (on a development machine, with stand-ins for the two models): the whole flow in a real
browser at desktop, tablet and phone sizes; one page per request; streaming text; narration
starting before the picture; background pictures with progress; choices, own ideas, continue and
finish; looking back through pages; double clicks; malformed model output; Qwen offline; failed
pictures and retry; leaving a story mid-generation; saving and resuming; the https address;
hold-to-talk and narration controls with simulated browser speech.

**Not tested, because it needs your Jetson:** real Qwen output quality and speed, real `sd-cli`
timings and memory, fast mode with the real LCM-LoRA and TAESD files, real microphones and voices,
Piper and whisper.cpp. No speed numbers are
claimed here. Section 4 and 5 are the checks to run.

Known limits:

* The old app's web addresses (routes) are not reproduced, since the old code was not available.
* Qwen 3B sometimes ignores length or repeats itself. Bad replies are retried once and repaired
  where possible; if choices are missing, three general ones are shown instead.
* Characters look similar from page to page (same description, style and seed) but not identical.
  SD 1.5 cannot guarantee that.
* The picture progress bar reads sd-cli's step counter. If your build prints progress differently,
  the bar stays near the start until the picture appears. Nothing else is affected.

## 11. Files

```
app.py                    the web application and its routes
settings.env              every adjustable setting
storyforge/config.py      reads settings.env
storyforge/llm.py         prompts, streaming from Qwen, validation of its replies
storyforge/store.py       story sessions and saving to disk
storyforge/images.py      picture queue around sd-cli
storyforge/speech.py      narration (ElevenLabs or Piper) and whisper voice input (all optional)
storyforge/elevenlabs.py  ElevenLabs text-to-speech client
storyforge/sysinfo.py     memory readings, logs and timings
storyforge/tls_proxy.py   the https address for the microphone
static/index.html, style.css, app.js     the website
static/fonts/             Fraunces and Literata (open licence, bundled so no internet is needed)
scripts/start.sh, stop.sh, make_cert.sh, benchmark_sd.sh, get_fast_models.sh
tests/                    automated tests and stand-in models
```
