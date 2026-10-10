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

# No examples of any kind in here: the model adds whatever is named (it once gave a whiskered cat a
# kitchen knife, and an example list of body parts gave a person whiskers and a tail). Compared on
# saved drawings, this wording names the kind of character most reliably and invents least.
PROMPT = (
    "A child drew this character to be the hero of a picture book. In ONE sentence of at most 30 words, "
    "describe it: first what kind of creature or person it most looks like, then the colours of any parts "
    "that are filled in with colour, then two or three distinctive features that are clearly drawn. If it is "
    "only outlined in one pen colour, do not give its body a colour. Describe it as if it were real, never "
    "as lines, outlines, strokes or ink. Only mention things that are clearly drawn; if you cannot tell what "
    "something is, leave it out. Start with 'a' or 'an'. Do not mention that it is a drawing, and do not "
    "give it a name."
)


def build_prompt(who: str = "") -> str:
    """The prompt, plus what the child said the hero is ("my bunny Ricky"), which wins over guessing."""
    if not who:
        return PROMPT
    return (PROMPT + f' The child says this is: "{who}". Trust that for what kind of character it is, '
            "but do not repeat its name.")


# Used when no model can be reached, so the story still starts.
FALLBACK = "a brave little hero in bright colours"


class VisionFailed(Exception):
    pass


def describe_hero(jpeg: bytes, who: str = "") -> tuple[str, str]:
    """Returns (one-sentence description, model that wrote it). Blocking; run it in a thread.
    Raises VisionFailed when no model gave a usable answer. Logs the image size, the exact prompt
    and each model's raw answer, so recognition problems can be traced drawing by drawing."""
    if not settings.hf_token:
        raise VisionFailed("no Hugging Face key")
    image = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")
    prompt = build_prompt(who)
    log.info("hero vision request  image=%s  %d bytes  prompt: %s", _jpeg_size(jpeg), len(jpeg), prompt)
    problems: list[str] = []
    deadline = time.monotonic() + settings.vision_timeout
    for model in settings.vision_models:
        remaining = deadline - time.monotonic()
        if remaining < 2:
            problems.append("out of time")
            break
        try:
            text = _ask(model, image, remaining, prompt)
        except VisionFailed as exc:
            problems.append(f"{model}: {exc}")
            log.warning("hero description failed with %s: %s", model, exc)
            continue
        sentence = clean_description(text)
        log.info("hero vision answer  model=%s  raw=%r  cleaned=%r", model, text, sentence)
        if sentence:
            return sentence, model
        problems.append(f"{model}: empty answer")
    raise VisionFailed("; ".join(problems) or "no vision model configured")


def clean_description(text: str) -> str:
    """One tidy sentence: no quotes, markdown or 'This drawing shows', lower-case start, capped."""
    text = re.sub(r"[*_`#\"“”]", "", text or "").strip()
    text = re.sub(r"^(?:the|this)\s+(?:drawing|picture|image|character)\s+(?:shows|is|depicts)\s+", "", text, flags=re.I)
    # "..., drawn in dark blue, ..." is about the pen, not the character
    text = re.sub(r",?\s*(?:drawn|outlined|sketched)\s+(?:in|with)\s+[^,.;]+", "", text, flags=re.I)
    text = re.split(r"(?<=[.!?])\s", text, maxsplit=1)[0].strip().rstrip(".!?")
    words = text.split()[:30]
    if len(words) < 2:
        return ""
    text = " ".join(words)
    return text[0].lower() + text[1:]


def _jpeg_size(data: bytes) -> str:
    """'WxH' from a JPEG's start-of-frame marker, without an image library."""
    i = 2
    while i + 9 < len(data):
        if data[i] != 0xFF:
            i += 1
            continue
        marker, length = data[i + 1], int.from_bytes(data[i + 2:i + 4], "big")
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            return f"{int.from_bytes(data[i + 7:i + 9], 'big')}x{int.from_bytes(data[i + 5:i + 7], 'big')}"
        i += 2 + length
    return "?"


def _ask(model: str, image: str, timeout: float, prompt: str = PROMPT) -> str:
    url = urlparse(settings.hf_url)
    conn_cls = http.client.HTTPSConnection if url.scheme == "https" else http.client.HTTPConnection
    body = json.dumps({
        "model": model,
        "max_tokens": 80,
        "temperature": 0.2,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
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
