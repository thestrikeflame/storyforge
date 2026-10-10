"""Page backgrounds painted in the cloud by Hugging Face (FLUX.1-schnell through an inference provider).

Uses the provider's OpenAI-style endpoint through the Hugging Face router:
    POST {HF_URL}/{HF_PROVIDER}/v1/images/generations
The token (HF_TOKEN, from .env) only ever goes into the Authorization header. It is never
logged, never returned to the browser and never included in error messages.
"""
from __future__ import annotations

import base64
import http.client
import json
import socket
import time
from urllib.parse import urlparse

from .config import settings


class HFFailed(Exception):
    """A background could not be painted. `reason` is a key of images.MESSAGES."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason


def image_kind(data: bytes) -> str | None:
    """'png' or 'jpg' from the file's first bytes, or None when it is not a picture."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data.startswith(b"\xff\xd8\xff"):
        return "jpg"
    return None


def generate(prompt: str, width: int, height: int, seed: int) -> tuple[bytes, str]:
    """Paint one background. Blocking; run it in a thread. Returns (image bytes, 'png' or 'jpg').
    The whole exchange, connecting included, is limited to HF_TIMEOUT_SECONDS."""
    if not settings.hf_token:
        raise HFFailed("hf_no_token")
    url = urlparse(settings.hf_url)
    conn_cls = http.client.HTTPSConnection if url.scheme == "https" else http.client.HTTPConnection
    deadline = time.monotonic() + settings.hf_timeout
    body = json.dumps({
        "model": settings.hf_image_model,
        "prompt": prompt,
        "size": f"{width}x{height}",
        "n": 1,
        "response_format": "b64_json",
        "seed": seed,
    }).encode("utf-8")
    path = f"{url.path.rstrip('/')}/{settings.hf_provider}/v1/images/generations"
    conn = conn_cls(url.hostname or "router.huggingface.co", url.port, timeout=settings.hf_timeout)
    try:
        conn.request("POST", path, body=body, headers={
            "Authorization": f"Bearer {settings.hf_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        })
        conn.sock.settimeout(max(1.0, deadline - time.monotonic()))
        response = conn.getresponse()
        chunks: list[bytes] = []
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise HFFailed("timeout")
            conn.sock.settimeout(remaining)
            chunk = response.read(65536)
            if not chunk:
                break
            chunks.append(chunk)
            if sum(len(c) for c in chunks) > 20_000_000:
                raise HFFailed("failed", "reply too large")
        raw = b"".join(chunks)
    except (socket.timeout, TimeoutError):
        raise HFFailed("timeout") from None
    except OSError as exc:
        raise HFFailed("hf_offline", type(exc).__name__) from None
    finally:
        conn.close()

    status = response.status
    if status in (401, 403):
        raise HFFailed("hf_unauthorized", f"HTTP {status}")
    if status == 429:
        raise HFFailed("hf_busy", "HTTP 429")
    if status != 200:
        raise HFFailed("failed", f"HTTP {status}: {_error_text(raw)}")
    try:
        data = base64.b64decode(json.loads(raw)["data"][0]["b64_json"])
    except (ValueError, KeyError, IndexError, TypeError):
        raise HFFailed("failed", "unexpected reply") from None
    kind = image_kind(data)
    if kind is None or len(data) < 1000:
        raise HFFailed("failed", "reply was not a picture")
    return data, kind


def _error_text(raw: bytes) -> str:
    """The provider's own error message, shortened, for the log."""
    try:
        value = json.loads(raw)
        message = value.get("error") if isinstance(value, dict) else None
        if isinstance(message, dict):
            message = message.get("message")
        return str(message or value)[:200]
    except ValueError:
        return raw[:200].decode("utf-8", "replace")
