"""Voice input: the browser records (MediaRecorder) and POST /api/listen turns it into text.

1. ffmpeg turns whatever the browser recorded (webm, mp4...) into 16 kHz mono WAV.
2. whisper.cpp's whisper-server on the Jetson transcribes it. It is started once with the model
   (base.en, on the GPU) and kept warm, so a short clip takes well under a second.
3. If whisper is not there or fails: Hugging Face's Whisper, with HF_TOKEN.
If both fail the browser says "Type your idea instead."
"""
from __future__ import annotations

import asyncio
import http.client
import json
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse

from .config import settings
from .sysinfo import log, memory_mb, timings


class ListenFailed(Exception):
    pass


def ffmpeg_path() -> str | None:
    if settings.ffmpeg.is_file() and os.access(settings.ffmpeg, os.X_OK):
        return str(settings.ffmpeg)
    return shutil.which("ffmpeg")


async def to_wav(audio: bytes) -> bytes:
    """The recording as 16 kHz mono 16-bit WAV. From a file, not a pipe: Safari's mp4 recordings
    need seeking."""
    ffmpeg = ffmpeg_path()
    if not ffmpeg:
        raise ListenFailed("ffmpeg is not installed")
    with tempfile.TemporaryDirectory(prefix="sf-listen-") as folder:
        source = Path(folder) / "recording"
        source.write_bytes(audio)
        proc = await asyncio.create_subprocess_exec(
            ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(source),
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", "-f", "wav", "pipe:1",
            stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=20)
        except asyncio.TimeoutError:
            proc.kill()
            raise ListenFailed("ffmpeg took too long") from None
    if proc.returncode != 0 or len(out) < 1000:
        raise ListenFailed(f"ffmpeg could not read the recording: {err.decode('utf-8', 'replace')[-200:]}")
    return out


def wav_seconds(wav: bytes) -> float:
    return max(0.0, (len(wav) - 44) / 32000)  # 16 kHz, 16-bit, mono


def clean(text: str) -> str:
    text = " ".join(text.split())
    # whisper marks silence and noises like [BLANK_AUDIO] or (wind blowing)
    while text[:1] in "[(" and text.find("]" if text[0] == "[" else ")") > 0:
        text = text[text.find("]" if text[0] == "[" else ")") + 1:].strip()
    return text.strip()


def _post(url: str, body: bytes, headers: dict, timeout: float) -> tuple[int, bytes]:
    parts = urlparse(url)
    conn_cls = http.client.HTTPSConnection if parts.scheme == "https" else http.client.HTTPConnection
    conn = conn_cls(parts.hostname, parts.port, timeout=timeout)
    try:
        conn.request("POST", parts.path + (f"?{parts.query}" if parts.query else ""), body=body, headers=headers)
        response = conn.getresponse()
        return response.status, response.read()
    finally:
        conn.close()


class Whisper:
    """whisper.cpp's server, started once with the model and kept warm on 127.0.0.1."""

    def __init__(self) -> None:
        self.proc: asyncio.subprocess.Process | None = None
        self.ready = False
        self.state = "not started"
        self._starting: asyncio.Task | None = None
        self._last_try = 0.0

    def installed(self) -> bool:
        return (settings.whisper_server.is_file() and os.access(settings.whisper_server, os.X_OK)
                and settings.whisper_model.is_file())

    def running(self) -> bool:
        return self.ready and self.proc is not None and self.proc.returncode is None

    def start(self) -> None:
        """Start it in the background (loading the model takes a few seconds); never twice at once."""
        if not self.installed():
            self.state = "not installed"
            return
        if self.running() or (self._starting and not self._starting.done()):
            return
        self._last_try = time.monotonic()
        self._starting = asyncio.create_task(self._start())

    async def _start(self) -> None:
        self.ready = False
        mem = memory_mb()
        settings.log_dir.mkdir(parents=True, exist_ok=True)
        out = open(settings.log_dir / "whisper-server.log", "ab")
        try:
            self.proc = await asyncio.create_subprocess_exec(
                str(settings.whisper_server), "-m", str(settings.whisper_model),
                "--host", "127.0.0.1", "--port", str(settings.whisper_port), "-t", "4", "-l", "en",
                stdin=asyncio.subprocess.DEVNULL, stdout=out, stderr=out,
            )
        except OSError as exc:
            self.state = f"could not start: {exc}"
            log.error("voice input: whisper-server could not start (%s)", exc)
            return
        finally:
            out.close()
        started = time.monotonic()
        while time.monotonic() - started < 90:
            if self.proc.returncode is not None:
                self.state = f"stopped (exit {self.proc.returncode})"
                log.error("voice input: whisper-server stopped while loading (exit %s); see logs/whisper-server.log",
                          self.proc.returncode)
                return
            if await asyncio.to_thread(self._answers):
                self.ready = True
                self.state = "ready"
                after = memory_mb()
                log.info("voice input: whisper ready  %.1fs  model=%s  free memory %d MB before, %d MB after",
                         time.monotonic() - started, settings.whisper_model.name, mem["available"], after["available"])
                return
            await asyncio.sleep(0.5)
        self.state = "did not answer"
        log.error("voice input: whisper-server did not answer within 90 s")

    def _answers(self) -> bool:
        conn = http.client.HTTPConnection("127.0.0.1", settings.whisper_port, timeout=2)
        try:
            conn.request("GET", "/")
            return conn.getresponse().status < 500
        except OSError:
            return False
        finally:
            conn.close()

    async def stop(self) -> None:
        if self.proc and self.proc.returncode is None:
            self.proc.terminate()
            try:
                await asyncio.wait_for(self.proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                self.proc.kill()
        self.ready = False

    async def transcribe(self, wav: bytes) -> str:
        if not self.running():
            if self.installed() and time.monotonic() - self._last_try > 60:
                self.start()  # it stopped: start it again for next time
            raise ListenFailed(f"whisper is {self.state}")
        boundary = uuid.uuid4().hex
        body = b"".join([
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"speech.wav\"\r\n"
            "Content-Type: audio/wav\r\n\r\n".encode(), wav, b"\r\n",
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"response_format\"\r\n\r\njson\r\n".encode(),
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"temperature\"\r\n\r\n0.0\r\n".encode(),
            f"--{boundary}--\r\n".encode(),
        ])
        try:
            status, raw = await asyncio.to_thread(
                _post, f"http://127.0.0.1:{settings.whisper_port}/inference", body,
                {"Content-Type": f"multipart/form-data; boundary={boundary}"}, settings.listen_timeout)
        except OSError as exc:
            raise ListenFailed(f"whisper did not answer ({type(exc).__name__})") from None
        if status != 200:
            raise ListenFailed(f"whisper answered HTTP {status}")
        try:
            return clean(json.loads(raw).get("text", ""))
        except ValueError:
            raise ListenFailed("whisper's answer was not JSON") from None


async def hugging_face(audio: bytes, content_type: str) -> str:
    """Hugging Face's Whisper (inference API), with HF_TOKEN. The token is never logged."""
    if not settings.hf_token:
        raise ListenFailed("no Hugging Face key")
    url = f"{settings.hf_url}/hf-inference/models/{settings.hf_asr_model}"
    try:
        status, raw = await asyncio.to_thread(
            _post, url, audio, {"Authorization": f"Bearer {settings.hf_token}", "Content-Type": content_type,
                                "Accept": "application/json"}, settings.listen_timeout)
    except OSError as exc:
        raise ListenFailed(f"Hugging Face did not answer ({type(exc).__name__})") from None
    if status != 200:
        raise ListenFailed(f"Hugging Face answered HTTP {status}: {raw[:160].decode('utf-8', 'replace')}")
    try:
        return clean(json.loads(raw).get("text", ""))
    except (ValueError, AttributeError):
        raise ListenFailed("Hugging Face's answer was not JSON") from None


whisper = Whisper()


def available() -> bool:
    """Something can turn speech into text (whisper here, or Hugging Face as the fallback)."""
    return whisper.installed() or bool(settings.hf_token)


async def listen(audio: bytes, content_type: str) -> tuple[str, str]:
    """(transcript, engine). Raises ListenFailed when neither whisper nor Hugging Face could do it."""
    started = time.monotonic()
    problems = []
    try:
        wav = await to_wav(audio)
    except ListenFailed as exc:
        wav = audio if audio[:4] == b"RIFF" else b""
        problems.append(str(exc))
    converted = time.monotonic()
    if wav:
        try:
            text = await whisper.transcribe(wav)
            done = time.monotonic()
            log.info("voice input  engine=whisper  audio=%.1fs  convert=%.2fs  transcribe=%.2fs  words=%d",
                     wav_seconds(wav), converted - started, done - converted, len(text.split()))
            timings.record("listen", engine="whisper", audio_seconds=round(wav_seconds(wav), 1),
                           convert_seconds=round(converted - started, 2), seconds=round(done - converted, 2))
            return text, "whisper"
        except ListenFailed as exc:
            problems.append(str(exc))
    try:
        text = await hugging_face(wav or audio, "audio/wav" if wav else content_type)
    except ListenFailed as exc:
        problems.append(str(exc))
        log.warning("voice input failed  (%s)", "; ".join(problems))
        raise ListenFailed("; ".join(problems)) from None
    done = time.monotonic()
    log.info("voice input  engine=huggingface  audio=%.1fs  convert=%.2fs  transcribe=%.2fs  words=%d  (%s)",
             wav_seconds(wav) if wav else 0, converted - started, done - converted, len(text.split()), "; ".join(problems))
    timings.record("listen", engine="huggingface", seconds=round(done - converted, 2))
    return text, "huggingface"
