"""Describes the child's hero drawing in one sentence, with a vision model on the Hugging Face
router (OpenAI-style chat completions, same HF_TOKEN as the backgrounds).

The models in VISION_MODELS are tried in order until one answers. The token only ever goes into
the Authorization header and is never logged.
"""
from __future__ import annotations

import base64
import http.client
import json
import re
import socket
import time
from urllib.parse import urlparse

from .config import settings
from .sysinfo import log

PROMPT = (
    "A child drew this character to be the hero of a picture book. In ONE short sentence of at most "
    "25 words, say what kind of character it is and what it looks like: colours, shape, size and any "
    "clothes or special features. Describe it as if it were real: say 'a blue fish', never "
    "'a blue-outlined fish' or anything about lines, strokes or sketching. Start with 'a' or 'an'. "
    "Describe only what is clearly drawn. Never guess at things it might be holding or wearing: if "
    "you are not sure what something is, leave it out. "
    "Do not mention that it is a drawing or a picture, and do not give it a name."
)
# Used when no model can be reached, so the story still starts.
FALLBACK = "a brave little hero in bright colours"


class VisionFailed(Exception):
    pass


def describe_hero(jpeg: bytes) -> tuple[str, str]:
    """Returns (one-sentence description, model that wrote it). Blocking; run it in a thread.
    Raises VisionFailed when no model gave a usable answer."""
    if not settings.hf_token:
        raise VisionFailed("no Hugging Face key")
    image = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")
    problems: list[str] = []
    deadline = time.monotonic() + settings.vision_timeout
    for model in settings.vision_models:
        remaining = deadline - time.monotonic()
        if remaining < 2:
            problems.append("out of time")
            break
        try:
            text = _ask(model, image, remaining)
        except VisionFailed as exc:
            problems.append(f"{model}: {exc}")
            log.warning("hero description failed with %s: %s", model, exc)
            continue
        sentence = clean_description(text)
        if sentence:
            return sentence, model
        problems.append(f"{model}: empty answer")
    raise VisionFailed("; ".join(problems) or "no vision model configured")


def clean_description(text: str) -> str:
    """One tidy sentence: no quotes, markdown or 'This drawing shows', lower-case start, capped."""
    text = re.sub(r"[*_`#\"“”]", "", text or "").strip()
    text = re.sub(r"^(?:the|this)\s+(?:drawing|picture|image|character)\s+(?:shows|is|depicts)\s+", "", text, flags=re.I)
    text = re.split(r"(?<=[.!?])\s", text, maxsplit=1)[0].strip().rstrip(".!?")
    words = text.split()[:30]
    if len(words) < 2:
        return ""
    text = " ".join(words)
    return text[0].lower() + text[1:]


def _ask(model: str, image: str, timeout: float) -> str:
    url = urlparse(settings.hf_url)
    conn_cls = http.client.HTTPSConnection if url.scheme == "https" else http.client.HTTPConnection
    body = json.dumps({
        "model": model,
        "max_tokens": 80,
        "temperature": 0.2,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": PROMPT},
            {"type": "image_url", "image_url": {"url": image}},
        ]}],
    }).encode("utf-8")
    conn = conn_cls(url.hostname or "router.huggingface.co", url.port, timeout=timeout)
    try:
        conn.request("POST", f"{url.path.rstrip('/')}/v1/chat/completions", body=body, headers={
            "Authorization": f"Bearer {settings.hf_token}",
            "Content-Type": "application/json",
        })
        response = conn.getresponse()
        raw = response.read(1_000_000)
    except (socket.timeout, TimeoutError):
        raise VisionFailed("timed out") from None
    except OSError as exc:
        raise VisionFailed(f"could not connect ({type(exc).__name__})") from None
    finally:
        conn.close()
    if response.status != 200:
        raise VisionFailed(f"HTTP {response.status}")
    try:
        return json.loads(raw)["choices"][0]["message"]["content"] or ""
    except (ValueError, KeyError, IndexError, TypeError):
        raise VisionFailed("unexpected reply") from None
