"""Narration voiced by ElevenLabs (text-to-speech in the cloud).

    POST {ELEVENLABS_URL}/v1/text-to-speech/{ELEVENLABS_VOICE_ID}?output_format=mp3_44100_128
The key (ELEVENLABS_API_KEY, from .env) only ever goes into the xi-api-key header. It is never
logged, never returned to the browser and never included in error messages.
"""
from __future__ import annotations

import http.client
import json
import socket
import time
from urllib.parse import quote, urlparse

from .config import settings


class TTSFailed(Exception):
    """A page could not be voiced; the browser's own voice reads it instead."""


def synthesize(text: str, speed: float = 1.0) -> tuple[bytes, int | None]:
    """Voice one page (or one word and its definition). Blocking; run it in a thread. Returns
    (MP3 bytes, credits charged when ElevenLabs reports them). `speed` below 1 reads more slowly.
    The whole exchange, connecting included, is limited to ELEVENLABS_TIMEOUT_SECONDS."""
    if not settings.elevenlabs_api_key:
        raise TTSFailed("no key")
    url = urlparse(settings.elevenlabs_url)
    conn_cls = http.client.HTTPSConnection if url.scheme == "https" else http.client.HTTPConnection
    deadline = time.monotonic() + settings.elevenlabs_timeout
    request: dict = {"text": text, "model_id": settings.elevenlabs_model}
    if speed != 1.0:
        request["voice_settings"] = {"speed": speed}
    body = json.dumps(request).encode("utf-8")
    path = (f"{url.path.rstrip('/')}/v1/text-to-speech/{quote(settings.elevenlabs_voice_id, safe='')}"
            "?output_format=mp3_44100_128")
    conn = conn_cls(url.hostname or "api.elevenlabs.io", url.port, timeout=settings.elevenlabs_timeout)
    try:
        conn.request("POST", path, body=body, headers={
            "xi-api-key": settings.elevenlabs_api_key,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        })
        conn.sock.settimeout(max(1.0, deadline - time.monotonic()))
        response = conn.getresponse()
        chunks: list[bytes] = []
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TTSFailed("timed out")
            conn.sock.settimeout(remaining)
            chunk = response.read(65536)
            if not chunk:
                break
            chunks.append(chunk)
            if sum(len(c) for c in chunks) > 20_000_000:
                raise TTSFailed("reply too large")
        raw = b"".join(chunks)
    except (socket.timeout, TimeoutError):
        raise TTSFailed("timed out") from None
    except OSError as exc:
        raise TTSFailed(f"offline ({type(exc).__name__})") from None
    finally:
        conn.close()

    if response.status != 200:
        raise TTSFailed(f"HTTP {response.status}: {_error_text(raw)}")
    # An MP3 starts with an ID3 tag or an MPEG frame header.
    if len(raw) < 1000 or not (raw.startswith(b"ID3") or raw[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2")):
        raise TTSFailed("reply was not an MP3")
    try:
        credits = int(response.getheader("character-cost") or "")
    except ValueError:
        credits = None
    return raw, credits


def _error_text(raw: bytes) -> str:
    """ElevenLabs' own error message, shortened, for the log."""
    try:
        value = json.loads(raw)
        detail = value.get("detail") if isinstance(value, dict) else None
        if isinstance(detail, dict):
            detail = detail.get("message") or detail.get("status")
        return str(detail or value)[:200]
    except ValueError:
        return raw[:200].decode("utf-8", "replace")
