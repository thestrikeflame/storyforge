"""Narration (ElevenLabs in the cloud, or Piper on the Jetson) and optional offline voice input
(whisper.cpp).

All are optional. Without a recording the browser's built-in voice reads the story.
Without whisper.cpp the browser's built-in speech recognition is used where it exists.
"""
from __future__ import annotations

import asyncio
import os
import shutil
import time
from pathlib import Path

from . import elevenlabs
from .config import settings
from .sysinfo import log, timings

AUDIO_DIR = settings.data_dir / "audio"


def _piper_path() -> str | None:
    if settings.piper_bin.is_file() and os.access(settings.piper_bin, os.X_OK):
        return str(settings.piper_bin)
    return shutil.which("piper")


def tts_mode() -> str:
    """'elevenlabs' (voice made in the cloud), 'piper' (voice made on the Jetson),
    'browser' (the browser speaks) or 'off'."""
    mode = settings.tts_mode
    if mode == "off":
        return "off"
    if mode == "browser":
        return "browser"
    if mode == "elevenlabs":
        return "elevenlabs" if settings.elevenlabs_api_key else "browser"
    piper_ready = _piper_path() is not None and settings.piper_voice.is_file()
    if piper_ready:
        return "piper"
    if mode == "piper":
        log.warning("TTS_MODE=piper but Piper or its voice file was not found; using the browser voice")
    return "browser"


def stt_available() -> bool:
    return (
        settings.whisper_cli.is_file()
        and os.access(settings.whisper_cli, os.X_OK)
        and settings.whisper_model.is_file()
    )


RECORDED = ("elevenlabs", "piper")  # modes where the server makes a recording for each page


class Narrator:
    """Creates one recording per page (an MP3 from ElevenLabs or a WAV from Piper) and reuses it on
    replays: a page that already has one is never voiced again. A page ElevenLabs could not voice
    is not tried again either (it may have used credits); the browser's voice reads it instead."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()  # one Piper synthesis at a time keeps CPU use predictable
        self._inflight: dict[str, asyncio.Task] = {}
        self._failed: set[str] = set()

    def path_for(self, story_id: str, index: int) -> Path:
        suffix = ".mp3" if tts_mode() == "elevenlabs" else ".wav"
        return AUDIO_DIR / f"sf_{story_id}_{index}{suffix}"

    def prepare(self, story_id: str, index: int, text: str) -> None:
        """Start synthesis in the background as soon as the page text exists."""
        if tts_mode() not in RECORDED:
            return
        key = f"{story_id}_{index}"
        if key in self._inflight or key in self._failed or self.path_for(story_id, index).is_file():
            return
        task = asyncio.create_task(self._synthesize(story_id, index, text))
        self._inflight[key] = task
        task.add_done_callback(lambda _t: self._inflight.pop(key, None))

    async def get(self, story_id: str, index: int, text: str) -> Path | None:
        """Return the narration file, waiting for it when it is still being made."""
        if tts_mode() not in RECORDED:
            return None
        path = self.path_for(story_id, index)
        if path.is_file():
            return path
        self.prepare(story_id, index, text)
        task = self._inflight.get(f"{story_id}_{index}")
        if task:
            wait = settings.elevenlabs_timeout + 5 if tts_mode() == "elevenlabs" else 90
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=wait)
            except (asyncio.TimeoutError, Exception):
                return None
        return path if path.is_file() else None

    async def _synthesize(self, story_id: str, index: int, text: str) -> None:
        if tts_mode() == "elevenlabs":
            await self._synthesize_elevenlabs(story_id, index, text)
        else:
            await self._synthesize_piper(story_id, index, text)

    async def _synthesize_elevenlabs(self, story_id: str, index: int, text: str) -> None:
        AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        final = self.path_for(story_id, index)
        line = " ".join(text.split())
        started = time.monotonic()
        try:
            audio, credits = await asyncio.to_thread(elevenlabs.synthesize, line)
        except elevenlabs.TTSFailed as exc:
            self._failed.add(f"{story_id}_{index}")
            log.error("narration failed  story=%s page=%d  voice=elevenlabs  characters=%d  (%s); "
                      "the browser's voice reads this page", story_id, index + 1, len(line), exc)
            return
        tmp = final.with_suffix(".part.mp3")
        tmp.write_bytes(audio)
        os.replace(tmp, final)
        seconds = round(time.monotonic() - started, 1)
        # Characters (and the credits ElevenLabs says it charged) for keeping an eye on the budget.
        log.info("narration ready  story=%s page=%d  voice=elevenlabs  characters=%d  credits=%s  %.1fs",
                 story_id, index + 1, len(line), "?" if credits is None else credits, seconds)
        timings.record("narration", seconds=seconds, characters=len(line), credits=credits, voice="elevenlabs")

    async def _synthesize_piper(self, story_id: str, index: int, text: str) -> None:
        piper = _piper_path()
        if not piper:
            return
        AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        final = self.path_for(story_id, index)
        tmp = final.with_suffix(".part.wav")
        async with self._lock:
            started = time.monotonic()
            try:
                proc = await asyncio.create_subprocess_exec(
                    piper, "--model", str(settings.piper_voice), "--output_file", str(tmp),
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.PIPE,
                )
                # Piper reads one utterance per line, so the page goes in as a single line.
                line = " ".join(text.split()) + "\n"
                _, err = await asyncio.wait_for(proc.communicate(line.encode("utf-8")), timeout=120)
            except (OSError, asyncio.TimeoutError) as exc:
                log.error("narration failed  story=%s page=%d: %s", story_id, index + 1, exc)
                tmp.unlink(missing_ok=True)
                return
            if proc.returncode != 0 or not tmp.is_file() or tmp.stat().st_size < 2000:
                log.error(
                    "narration failed  story=%s page=%d  exit=%s  %s",
                    story_id, index + 1, proc.returncode, err.decode("utf-8", "replace")[-400:],
                )
                tmp.unlink(missing_ok=True)
                return
            os.replace(tmp, final)
            seconds = round(time.monotonic() - started, 1)
            log.info("narration ready  story=%s page=%d  %.1fs", story_id, index + 1, seconds)
            timings.record("narration", seconds=seconds, characters=len(text))


_stt_lock = asyncio.Lock()


async def transcribe(wav_bytes: bytes) -> str:
    """Turn a short 16 kHz mono WAV recording into text with whisper.cpp."""
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    wav = AUDIO_DIR / f"mic_{os.getpid()}_{time.monotonic_ns()}.wav"
    wav.write_bytes(wav_bytes)
    try:
        async with _stt_lock:
            proc = await asyncio.create_subprocess_exec(
                str(settings.whisper_cli), "-m", str(settings.whisper_model),
                "-f", str(wav), "--no-timestamps", "--no-prints", "--language", "en",
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            out, err = await asyncio.wait_for(proc.communicate(), timeout=90)
        if proc.returncode != 0:
            log.error("voice input failed: %s", err.decode("utf-8", "replace")[-400:])
            raise RuntimeError("whisper failed")
        text = " ".join(out.decode("utf-8", "replace").split())
        # whisper marks silence like [BLANK_AUDIO] or (silence)
        if text.startswith(("[", "(")) and text.endswith(("]", ")")):
            return ""
        return text
    finally:
        wav.unlink(missing_ok=True)
