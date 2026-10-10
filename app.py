"""StoryForge web application.

Start it with:  ./scripts/start.sh      (or: uvicorn app:app --host 0.0.0.0 --port 8000)

How a page is made:
  1. The browser sends the reader's idea (or their decision for the next page).
  2. Qwen writes ONE page. Its words stream in, so the browser can show them straight away.
  3. As soon as the page text is complete, narration can begin.
  4. The illustration is painted in the background by sd-cli; the browser polls for it.
  5. The reader decides what happens next, and the cycle repeats.
"""
from __future__ import annotations

import asyncio
import base64
import binascii
import os
import struct
import time
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from storyforge import llm, speech, tls_proxy, vision
from storyforge.config import settings
from storyforge.images import MESSAGES as IMAGE_MESSAGES
from storyforge.images import ImageWorker, preset_backgrounds
from storyforge.store import StoryStore, hero_path, new_segment, new_story, segment_view, story_view
from storyforge.sysinfo import log, memory_mb, setup_logging, timings

ERRORS = {
    "writer_offline": "The storyteller is not awake. Check that the Qwen server is running, then try again.",
    "writer_failed": "The storyteller lost the thread of this page. Try again.",
    "interrupted": "StoryForge was restarted while this page was being written. Try again.",
}

store = StoryStore()
narrator = speech.Narrator()
worker = ImageWorker(on_change=lambda story: store.save(story))
llm_lock: asyncio.Lock  # one page is written at a time; created at startup
active_streams: dict[str, llm.LLMStream] = {}
write_tasks: dict[str, asyncio.Task] = {}
MAX_WAITING_WRITES = 3


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global llm_lock
    setup_logging()
    llm_lock = asyncio.Lock()
    store.cleanup_old()
    await worker.start()
    https_server = await tls_proxy.start_https_proxy()
    ok, reason = worker.availability()
    mem = memory_mb()
    log.info(
        "StoryForge started  port=%d  pictures=%s  narration=%s  offline voice input=%s  free memory=%d of %d MB",
        settings.port, worker.describe_mode() if ok else reason, speech.tts_mode(),
        "yes" if speech.stt_available() else "no", mem["available"], mem["total"],
    )
    if mem["total"] and not mem["swap_total"]:
        log.info("note: no swap space is configured; see the README section on memory")
    try:
        yield
    finally:
        for stream in list(active_streams.values()):
            stream.cancel()
        for task in list(write_tasks.values()):
            task.cancel()
        await worker.stop()
        if https_server is not None:
            https_server.close()
        for story in store.stories.values():
            store.save(story)


app = FastAPI(title="StoryForge", lifespan=lifespan, docs_url=None, redoc_url=None)


# --------------------------------------------------------------------------
# Writing a page
# --------------------------------------------------------------------------

def _update_partial(story: dict, seg: dict, stream: llm.LLMStream, started: float) -> None:
    """Copy the story text that has arrived so far into the page, and announce the moment
    the page text is complete so narration can start before the choices are even written.
    The page plan (setting, visual details, hero action) streams in first: in cloud mode the
    background starts painting as soon as the plan is complete."""
    if not seg.get("setting"):
        plan = llm.extract_plan(stream.raw)
        if plan:
            seg["setting"] = plan["setting"]
            seg["visual_details"] = plan["visual_details"]
            seg["hero_action"] = plan["action"]
            if settings.image_mode == "huggingface":
                worker.enqueue(story, seg)
    if seg["text_complete"]:
        return
    text, complete = llm.extract_string_field(stream.raw, "segment")
    if not text:
        return
    cleaned = llm.clean_text(text, 1800)
    seg["partial"] = cleaned
    if complete and llm.word_count(cleaned) >= llm.minimum_words() and '"visual_prompt"' not in cleaned:
        seg["text"] = cleaned
        seg["text_complete"] = True
        seg["timings"]["text_seconds"] = round(time.monotonic() - started, 1)
        narrator.prepare(story["id"], seg["index"], cleaned)


def _fail(story: dict, seg: dict, code: str, detail: str) -> None:
    seg["status"] = "failed"
    seg["error"] = {"code": code, "message": ERRORS[code]}
    seg["partial"] = ""
    seg["text"] = ""
    seg["text_complete"] = False
    log.error("page failed  story=%s page=%d  %s: %s", story["id"], seg["index"] + 1, code, detail)
    timings.record("page", ok=False, reason=code)
    store.save(story)


async def write_segment(story: dict, seg: dict) -> None:
    first = seg["index"] == 0
    final = bool(seg["final"])
    started = time.monotonic()
    mem_before = memory_mb()["available"]
    page: llm.Page | None = None
    stream: llm.LLMStream | None = None
    try:
        async with llm_lock:
            if story.get("abandoned"):
                return
            for attempt in (1, 2):
                stream = llm.LLMStream(
                    messages=llm.build_messages(story, seg["action"], final),
                    schema=llm.response_schema(first, final, drawn_hero=bool(story.get("hero_description"))),
                    temperature=settings.llm_temperature if attempt == 1 else 0.5,
                )
                active_streams[story["id"]] = stream
                job = asyncio.create_task(asyncio.to_thread(stream.run))
                try:
                    while not job.done():
                        await asyncio.sleep(0.2)
                        _update_partial(story, seg, stream, started)
                    raw = job.result()
                except llm.WriterOffline as exc:
                    _fail(story, seg, "writer_offline", str(exc))
                    return
                except llm.WriterFailed as exc:
                    if story.get("abandoned"):
                        return
                    if attempt == 2 and not seg["text_complete"]:
                        _fail(story, seg, "writer_failed", str(exc))
                        return
                    if seg["text_complete"]:
                        raw = stream.raw  # the page itself arrived; salvage it below
                    else:
                        log.warning("page attempt %d failed (%s); trying once more", attempt, exc)
                        seg["partial"] = ""
                        continue
                _update_partial(story, seg, stream, started)
                try:
                    page = llm.parse_page(raw, first, final, last_try=attempt == 2)
                    break
                except llm.WriterFailed as exc:
                    if seg["text_complete"]:
                        # Narration may already be playing this text, so keep it and fill in the rest.
                        page = llm.Page(
                            segment=seg["text"], visual_prompt=llm.clean_line(seg["text"].split(".")[0], 200),
                            choices=[] if final else list(llm.FALLBACK_CHOICES), summary="",
                            used_fallback_choices=not final, repaired=True,
                        )
                        break
                    if attempt == 2:
                        _fail(story, seg, "writer_failed", f"{exc}; reply began: {raw[:200]!r}")
                        return
                    log.warning("page attempt %d unusable (%s); trying once more", attempt, exc)
                    seg["partial"] = ""
    except asyncio.CancelledError:
        if stream is not None:
            stream.cancel()
        raise
    finally:
        active_streams.pop(story["id"], None)

    if page is None or story.get("abandoned"):
        return

    # ---- the page is good: update the story state ----
    if not seg["text_complete"]:
        seg["text"] = page.segment
        seg["text_complete"] = True
        seg["timings"]["text_seconds"] = round(time.monotonic() - started, 1)
        narrator.prepare(story["id"], seg["index"], seg["text"])
    seg["partial"] = ""
    seg["visual_prompt"] = page.visual_prompt
    if page.setting and not seg.get("setting"):
        seg["setting"] = page.setting
        seg["visual_details"] = page.visual_details
    if not seg.get("hero_action"):
        seg["hero_action"] = page.hero_action
    seg["choices"] = page.choices
    seg["error"] = None
    seg["status"] = "ready"

    if first:
        story["title"] = page.title or _title_from_premise(story["premise"])
        # A drawn hero keeps the vision model's description; the writer never redescribes it.
        story["character_look"] = story.get("hero_description") or page.character_look
        if page.hero_name and story.get("hero_description"):
            story["hero_name"] = page.hero_name
        story["world"] = page.world
    if page.summary:
        story["summary"] = page.summary
    else:  # the model forgot the summary: extend the old one with this page's first sentence
        addition = seg["text"].split(". ")[0]
        story["summary"] = llm.clean_line(f"{story.get('summary', '')} {addition}.".strip(), 520)
    if seg["action"]:
        story["direction"] = seg["action"]
        story["decisions"].append({"page": seg["index"] + 1, "kind": seg["kind"], "action": seg["action"]})
    if final:
        story["ended"] = True

    total = round(time.monotonic() - started, 1)
    seg["timings"].update(
        first_token_seconds=round(stream.first_token_seconds, 1) if stream and stream.first_token_seconds else None,
        total_seconds=total,
        tokens=stream.tokens if stream else None,
    )
    rate = round(stream.tokens_per_second, 1) if stream and stream.tokens_per_second else None
    log.info(
        "page written  story=%s page=%d  first words %ss, text complete %ss, all done %ss  "
        "tokens=%s (%s per second)  words=%d  free_mem=%dMB%s%s",
        story["id"], seg["index"] + 1, seg["timings"].get("first_token_seconds"),
        seg["timings"].get("text_seconds"), total, seg["timings"].get("tokens"), rate,
        llm.word_count(seg["text"]), mem_before,
        "  [choices: fallback]" if page.used_fallback_choices else "",
        "  [reply repaired]" if page.repaired else "",
    )
    timings.record(
        "page", ok=True, first_token_seconds=seg["timings"].get("first_token_seconds"),
        text_seconds=seg["timings"].get("text_seconds"), total_seconds=total,
        tokens=seg["timings"].get("tokens"), tokens_per_second=rate,
        words=llm.word_count(seg["text"]), free_mb_before=mem_before,
        fallback_choices=page.used_fallback_choices, repaired=page.repaired,
    )
    store.save(story)
    # A local picture starts only now, with Qwen idle again: the two never fight for the GPU
    # unless the reader races ahead. (A cloud background usually started while writing.)
    worker.enqueue(story, seg)
    store.save(story)


def _title_from_premise(premise: str) -> str:
    words = premise.strip().rstrip(".!?").split()[:6]
    return " ".join(words).capitalize() if words else "A new story"


def _start_writing(story: dict, seg: dict) -> None:
    task = asyncio.create_task(write_segment(story, seg), name=f"write-{story['id']}-{seg['index']}")
    write_tasks[story["id"]] = task

    def done(finished: asyncio.Task) -> None:
        if write_tasks.get(story["id"]) is finished:
            write_tasks.pop(story["id"], None)
        if not finished.cancelled() and finished.exception() is not None:
            log.error("unexpected error while writing", exc_info=finished.exception())
            if seg["status"] == "writing":
                _fail(story, seg, "writer_failed", repr(finished.exception()))

    task.add_done_callback(done)


def _waiting_writes() -> int:
    return sum(1 for task in write_tasks.values() if not task.done())


# --------------------------------------------------------------------------
# Requests
# --------------------------------------------------------------------------

class NewStory(BaseModel):
    premise: str = Field(min_length=1, max_length=settings.max_premise_chars)


class DrawnHero(BaseModel):
    # Base64 (optionally a data: URL). The PNG keeps its transparency and becomes the hero layer;
    # the JPEG is the same drawing on white, for the vision model.
    hero_png: str = Field(min_length=100, max_length=6_000_000)
    preview_jpeg: str = Field(min_length=100, max_length=2_000_000)
    idea: str = Field(default="", max_length=settings.max_premise_chars)
    who: str = Field(default="", max_length=120)  # "Who is this?" on /draw, e.g. "my bunny Ricky"


def _decode_image(value: str, magic: bytes, limit: int) -> bytes:
    if value.startswith("data:"):
        value = value.partition(",")[2]
    try:
        data = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError):
        data = b""
    if not data.startswith(magic) or len(data) > limit:
        raise HTTPException(422, "The drawing could not be read. Try pressing Done again.")
    return data


def _png_size(data: bytes) -> tuple[int, int]:
    """Width and height from the PNG header (the first chunk is always IHDR)."""
    if len(data) < 24 or data[12:16] != b"IHDR":
        return 0, 0
    return struct.unpack(">II", data[16:24])


class NextPage(BaseModel):
    kind: Literal["choice", "custom", "continue", "finish"]
    action: str = Field(default="", max_length=settings.max_action_chars)
    after: int = Field(ge=0, le=10_000)  # the page the reader was on; makes double clicks harmless


def _story_or_404(story_id: str) -> dict:
    story = store.get(story_id)
    if story is None:
        raise HTTPException(404, "This story could not be found. It may have been cleared away.")
    return story


def _segment_or_404(story: dict, index: int) -> dict:
    if index < 0 or index >= len(story["segments"]):
        raise HTTPException(404, "That page does not exist.")
    return story["segments"][index]


def _no_store(payload: dict, status: int = 200) -> JSONResponse:
    return JSONResponse(payload, status_code=status, headers={"Cache-Control": "no-store"})


@app.post("/api/stories")
async def create_story(body: NewStory):
    premise = llm.clean_text(body.premise, settings.max_premise_chars)
    if len(premise) < 3:
        raise HTTPException(422, "Describe your story idea in a few words first.")
    if _waiting_writes() >= MAX_WAITING_WRITES:
        raise HTTPException(503, "The storyteller is busy with other stories. Try again in a moment.")
    story = new_story(premise)
    seg = new_segment(0, None, "opening")
    story["segments"].append(seg)
    store.add(story)
    log.info("story started  story=%s  idea=%r", story["id"], premise[:80])
    _start_writing(story, seg)
    return _no_store(story_view(story, worker.eta_seconds()), 201)


@app.post("/api/stories/drawn")
async def create_drawn_story(body: DrawnHero):
    """From /draw: a new story whose hero is the child's drawing. Saves the drawing, asks a vision
    model for a one-sentence description, then starts page 1 with that character as the hero."""
    if _waiting_writes() >= MAX_WAITING_WRITES:
        raise HTTPException(503, "The storyteller is busy with other stories. Try again in a moment.")
    hero_png = _decode_image(body.hero_png, b"\x89PNG\r\n\x1a\n", 4_000_000)
    preview = _decode_image(body.preview_jpeg, b"\xff\xd8\xff", 1_500_000)
    width, height = _png_size(hero_png)
    if not (8 <= width <= 2048 and 8 <= height <= 2048):
        raise HTTPException(422, "The drawing could not be read. Try pressing Done again.")
    idea = llm.clean_text(body.idea, settings.max_premise_chars)
    # The child's own words, kept as typed: only whitespace collapsed, double quotes made single
    # (it is quoted in prompts), control characters dropped and the length capped.
    who = " ".join("".join(c for c in body.who.replace('"', "'") if c.isprintable()).split())[:80]

    story = new_story(idea or "A story about the hero I drew")
    path = hero_path(story["id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    tmp.write_bytes(hero_png)
    os.replace(tmp, path)
    story["hero"] = int(time.time())

    started = time.monotonic()
    try:
        description, model = await asyncio.to_thread(vision.describe_hero, preview, who)
        log.info("hero described  story=%s  %.1fs  %s  who=%r: %s",
                 story["id"], time.monotonic() - started, model, who, description)
    except vision.VisionFailed as exc:
        description = vision.FALLBACK
        log.warning("hero could not be described (%s); using a general description", exc)
    story["hero_description"] = description
    story["character_look"] = description
    if who:
        story["hero_who"] = who  # the child's own words win over the model's guess
    if not idea:
        # Only the kind of character: the full description in the premise gets copied into page 1.
        story["premise"] = f"A story starring {who or llm.hero_notes(description)[0]}"

    seg = new_segment(0, None, "opening")
    story["segments"].append(seg)
    store.add(story)
    log.info("story started from a drawing  story=%s  hero=%dx%d", story["id"], width, height)
    _start_writing(story, seg)
    return _no_store({**story_view(story, worker.eta_seconds()), "hero_description": description}, 201)


@app.get("/draw")
async def draw_page():
    return FileResponse(settings.static_dir / "draw.html", headers={"Cache-Control": "no-cache"})


@app.get("/api/stories/{story_id}")
async def get_story(story_id: str):
    story = _story_or_404(story_id)
    return _no_store(story_view(story, worker.eta_seconds()))


@app.get("/api/stories/{story_id}/segments/{index}")
async def get_segment(story_id: str, index: int):
    story = _story_or_404(story_id)
    seg = _segment_or_404(story, index)
    view = segment_view(story, seg, worker.eta_seconds())
    view["title"] = story.get("title") or ""
    view["ended"] = bool(story.get("ended"))
    if view["image"]["error"]:
        view["image"]["message"] = IMAGE_MESSAGES.get(view["image"]["error"], IMAGE_MESSAGES["failed"])
    return _no_store(view)


@app.post("/api/stories/{story_id}/continue")
async def continue_story(story_id: str, body: NextPage):
    story = _story_or_404(story_id)
    if story.get("abandoned"):
        raise HTTPException(409, "This story was closed. Start a new journey.")
    last = story["segments"][-1]
    # A repeated click (or a retry after a lost reply) asks for a page that already exists:
    # answer with the current state instead of writing a second, competing page.
    if body.after < last["index"]:
        return _no_store({**story_view(story, worker.eta_seconds()), "duplicate": True})
    if body.after > last["index"]:
        raise HTTPException(409, "That page has not been written yet.")
    if last["status"] != "ready":
        raise HTTPException(409, "This page is still being written.")
    if story.get("ended"):
        raise HTTPException(409, "This story has reached its ending.")
    if _waiting_writes() >= MAX_WAITING_WRITES:
        raise HTTPException(503, "The storyteller is busy with other stories. Try again in a moment.")

    kind = body.kind
    action = llm.clean_line(body.action, settings.max_action_chars)
    if kind in ("choice", "custom") and len(action) < 2:
        raise HTTPException(422, "Say or type what should happen next.")
    if kind == "continue":
        action = "Continue the adventure"
    if kind == "finish":
        action = "Bring the story to its ending"
    if len(story["segments"]) + 1 >= settings.max_segments:
        kind = "finish"  # the book is full: the next page is the ending

    seg = new_segment(last["index"] + 1, action, kind)
    story["segments"].append(seg)
    store.save(story)
    _start_writing(story, seg)
    return _no_store(story_view(story, worker.eta_seconds()), 201)


@app.post("/api/stories/{story_id}/segments/{index}/retry")
async def retry_segment(story_id: str, index: int):
    story = _story_or_404(story_id)
    seg = _segment_or_404(story, index)
    if story.get("abandoned"):
        raise HTTPException(409, "This story was closed. Start a new journey.")
    if seg["status"] == "failed" and index == len(story["segments"]) - 1:
        if _waiting_writes() >= MAX_WAITING_WRITES:
            raise HTTPException(503, "The storyteller is busy with other stories. Try again in a moment.")
        seg.update(status="writing", error=None, partial="", text="", text_complete=False, timings={})
        _start_writing(story, seg)
    return _no_store(segment_view(story, seg, worker.eta_seconds()))


@app.post("/api/stories/{story_id}/segments/{index}/image/retry")
async def retry_image(story_id: str, index: int):
    story = _story_or_404(story_id)
    seg = _segment_or_404(story, index)
    image = seg["image"]
    if seg["status"] == "ready" and (image["status"] in ("failed", "none") or image.get("source") == "preset"):
        worker.enqueue(story, seg, retry=True)  # refuses duplicates by itself
        store.save(story)
    return _no_store(segment_view(story, seg, worker.eta_seconds()))


@app.get("/api/stories/{story_id}/segments/{index}/image")
async def get_image(story_id: str, index: int):
    story = _story_or_404(story_id)
    seg = _segment_or_404(story, index)
    if seg["image"]["status"] == "completed" and seg["image"].get("source") == "preset":
        # Only a name from the known list of preset files is ever served.
        preset = next((p for p in preset_backgrounds() if p.name == seg["image"].get("file")), None)
        if preset is None:
            raise HTTPException(404, "This page has no picture yet.")
        return FileResponse(preset, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=86400"})
    # The file name is rebuilt on the server from validated values; the browser never supplies a path.
    kind = "jpg" if str(seg["image"].get("file") or "").endswith(".jpg") else "png"
    path = settings.image_dir / f"sf_{story['id']}_{seg['index']}.{kind}"
    if seg["image"]["status"] != "completed" or not path.is_file():
        raise HTTPException(404, "This page has no picture yet.")
    media = "image/jpeg" if kind == "jpg" else "image/png"
    return FileResponse(path, media_type=media, headers={"Cache-Control": "private, max-age=86400"})


@app.get("/api/stories/{story_id}/hero")
async def get_hero(story_id: str):
    """The child's drawing of the hero (a transparent PNG), or the placeholder until /draw exists."""
    story = _story_or_404(story_id)
    path = hero_path(story["id"])
    if not story.get("hero") or not path.is_file():
        path = settings.static_dir / "art" / "hero-placeholder.png"
    return FileResponse(path, media_type="image/png", headers={"Cache-Control": "no-cache"})


@app.get("/api/stories/{story_id}/segments/{index}/audio")
async def get_audio(story_id: str, index: int):
    story = _story_or_404(story_id)
    seg = _segment_or_404(story, index)
    if not seg["text_complete"] or not seg["text"]:
        raise HTTPException(404, "This page has no narration yet.")
    path = await narrator.get(story["id"], seg["index"], seg["text"])
    if path is None:
        raise HTTPException(404, "Narration is not available from the server.")
    return FileResponse(path, media_type="audio/wav", headers={"Cache-Control": "private, max-age=86400"})


@app.post("/api/stories/{story_id}/abandon")
async def abandon_story(story_id: str):
    story = _story_or_404(story_id)
    if not story.get("abandoned"):
        story["abandoned"] = True
        stream = active_streams.get(story_id)
        if stream is not None:
            stream.cancel()
        await worker.cancel_story(story_id)
        for seg in story["segments"]:
            if seg["status"] == "writing":
                seg["status"] = "failed"
                seg["error"] = {"code": "interrupted", "message": "This story was closed."}
        store.save(story)
        log.info("story closed  story=%s  pages=%d", story_id, len(story["segments"]))
    return _no_store({"ok": True})


@app.post("/api/transcribe")
async def transcribe(request: Request):
    """Offline voice input: the browser sends a short WAV recording, whisper.cpp returns text."""
    if not speech.stt_available():
        raise HTTPException(404, "Offline voice input is not installed.")
    body = await request.body()
    if len(body) < 2000 or len(body) > 6_000_000 or body[:4] != b"RIFF" or body[8:12] != b"WAVE":
        raise HTTPException(422, "The recording was empty or too long. Hold the microphone and try again.")
    try:
        text = await speech.transcribe(body)
    except Exception:
        raise HTTPException(500, "The recording could not be understood. Try again, or type instead.")
    return _no_store({"text": text[: settings.max_premise_chars]})


@app.get("/api/config")
async def get_config():
    images_ok, images_reason = worker.availability()
    art = next(
        (f"/static/art/{p.name}" for ext in ("webp", "jpg", "jpeg", "png")
         for p in [settings.static_dir / "art" / f"landing.{ext}"] if p.is_file()),
        None,
    )
    return _no_store({
        "tts": speech.tts_mode(),
        "server_stt": speech.stt_available(),
        "images": images_ok,
        "images_reason": None if images_ok else IMAGE_MESSAGES.get(images_reason, images_reason),
        "landing_art": art,
        "https_port": settings.https_port if tls_proxy.https_ready() else None,
        "max_premise_chars": settings.max_premise_chars,
        "max_action_chars": settings.max_action_chars,
        "image_eta_seconds": worker.eta_seconds(),
    })


@app.get("/api/health")
async def health():
    writer_ok, writer_state = await asyncio.to_thread(llm.check_writer)
    images_ok, images_reason = worker.availability()
    return _no_store({
        "ok": writer_ok,
        "story_writer": {"ok": writer_ok, "state": writer_state, "url": settings.llm_url},
        "pictures": {
            "ok": images_ok, "state": images_reason, "mode": worker.describe_mode(),
            "hf_key_set": bool(settings.hf_token), "local_fallback": worker.local_fallback_ok(),
            "size": f"{settings.sd_width}x{settings.sd_height}", "waiting_or_running": worker.queue_length(),
            "options_checked_against_help": worker.help_checked,
        },
        "narration": speech.tts_mode(),
        "offline_voice_input": speech.stt_available(),
        "https_for_microphone": tls_proxy.https_ready(),
        "memory_mb": memory_mb(),
    })


@app.get("/api/metrics")
async def metrics():
    return _no_store(timings.summary())


@app.exception_handler(HTTPException)
async def http_error(_request: Request, exc: HTTPException):
    return JSONResponse({"message": str(exc.detail)}, status_code=exc.status_code)


@app.exception_handler(RequestValidationError)
async def validation_error(_request: Request, _exc: RequestValidationError):
    return JSONResponse(
        {"message": "That was empty or too long. Keep it to a few sentences and try again."}, status_code=422
    )


@app.exception_handler(Exception)
async def unexpected_error(_request: Request, exc: Exception):
    log.exception("unexpected error", exc_info=exc)  # full details go to the log, never to the browser
    return JSONResponse({"message": "Something went wrong inside StoryForge. Try again."}, status_code=500)


@app.get("/")
async def index():
    return FileResponse(settings.static_dir / "index.html", headers={"Cache-Control": "no-cache"})


app.mount("/static", StaticFiles(directory=settings.static_dir), name="static")
