"""Talking to Qwen (llama-server) and turning its replies into safe, validated story pages.

Only the Python standard library is used, so nothing new has to be installed.
The model is asked for ONE page at a time and never for the whole story.
"""
from __future__ import annotations

import http.client
import json
import re
import socket
import threading
import time
from dataclasses import dataclass, field
from urllib.parse import urlparse

from .config import settings


class WriterOffline(Exception):
    """llama-server could not be reached."""


class WriterFailed(Exception):
    """llama-server answered, but not with something usable."""


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are StoryForge, a storyteller who writes an illustrated story one page at a time "
    "together with a reader. Unless the reader's idea clearly asks for an older audience, "
    "write for children aged 5 to 10: warm, vivid, simple sentences, never frightening or violent. "
    "Keep it gentle: nobody is hurt, attacked or fought. If the hero has something like a knife, sword "
    "or other weapon, it is a tool or a toy: for cooking, carving, building or a pretend quest, never for "
    "fighting. "
    "Each page has at least {min_words} and at most {max_words} words, in two or three short "
    "paragraphs, told in the past tense. Keep names, looks and places consistent with what came before. "
    "Never wrap up the whole story unless you are told it is the last page. "
    "Reply with one JSON object only, no other text."
)

# What the hero does on a page; the browser animates the hero layer for each (static/hero.js).
# The first one is the default for anything off the list.
HERO_ACTIONS = ("standing", "walking", "running", "jumping", "waving", "flying", "swimming", "sleeping",
                "cheering", "dancing", "spinning", "hiding")

# The page plan is asked for first, so the background is painted while the page is written.
_PLAN_KEYS = (
    '"setting": a plan for THIS page: the specific place it happens, chosen to fit the story, in at most '
    "15 words, for the background painter. Any kind of place: a kitchen, a castle hall, a sunny beach, "
    "a space station, a busy village square, an underwater reef, a snowy mountain, a city park. Do not "
    "default to a forest; use one only when the story truly happens in one. No people, animals or "
    "characters.\n"
    '"visual_details": a list of 2 or 3 concrete things visible in that place, each at most 6 words: '
    "objects, weather or time of day. Never characters. The page must happen in this place and "
    "mention every one of these details.\n"
    f'"action": what the hero is doing on this page, exactly one of: {", ".join(HERO_ACTIONS)}.\n'
)
_COMMON_KEYS = (
    '"visual_prompt": at most 20 words for the illustrator: ONE main character (say what it looks '
    "like, never its name), doing ONE thing, in ONE place. Simple and concrete, no feelings.\n"
)
_CHOICES_KEY = (
    '"choices": a list of exactly 3 different things the reader could make happen next. '
    "Each is at most 9 words and starts with a verb.\n"
)
_SUMMARY_KEY = (
    '"continuity_summary": at most 60 words holding everything worth remembering so far: '
    "names, goals, objects, promises and where everyone is."
)


def hero_notes(description: str) -> tuple[str, list[str]]:
    """Split the vision description into the kind of character and a few short features, so the
    storyteller gets notes to use rather than a sentence to copy.
    "a small blue fish with a yellow crown, one black eye, and a red tail"
      -> ("a small blue fish", ["a yellow crown", "one black eye", "a red tail"])"""
    kind, _, rest = description.partition(" with ")
    features = [f.strip(" .") for f in re.split(r",\s*(?:and\s+)?|\s+and\s+", rest) if f.strip(" .")]
    return kind.strip(" ,.") or description, features[:4]


def build_messages(story: dict, action: str | None, final: bool) -> list[dict]:
    """Build the chat messages for the next page. Sends a compact summary plus only the
    most recent page, never the whole history, so the 2048-token context is not exceeded."""
    system = SYSTEM_PROMPT.format(
        min_words=settings.segment_min_words, max_words=settings.segment_max_words
    )
    ready = [s for s in story["segments"] if s["status"] == "ready"]
    page_number = len(ready) + 1

    if not ready:
        hero = story.get("hero_description")
        hero_line = ""
        if hero:
            kind, features = hero_notes(hero)
            hero_line = (
                f"The hero is a character the reader drew: {kind}."
                + (f" Notes: {'; '.join(features)}." if features else "")
                + " Make it the main character. Choose its name in \"hero_name\" and begin the page with "
                "that name. Tell what the hero does, in fresh story words; never list its looks in one "
                "long sentence. Mention one of the notes naturally, later in the page.\n\n"
            )
        user = (
            f"The reader's idea for the story:\n\"{story['premise']}\"\n\n"
            + hero_line +
            "Write page 1 only. Introduce the main character and the place, and end on a moment "
            "of curiosity so the reader wants to decide what happens next. Do not resolve anything.\n\n"
            "Use these JSON keys in this order:\n"
            '"title": a title for the story, at most 6 words.\n'
            + _PLAN_KEYS
            + ('"hero_name": the hero\'s short, friendly name, 1 to 3 words.\n' if hero else "") +
            '"segment": the text of page 1, written from the plan above.\n'
            '"character_look": what the main character looks like in at most 20 words '
            "(kind of creature or person, colours, size, clothing).\n"
            '"world": the setting in at most 20 words.\n'
            + _COMMON_KEYS + _CHOICES_KEY + _SUMMARY_KEY
        )
    else:
        previous = ready[-1]["text"][-900:]
        if final:
            task = (
                f"Write page {page_number}, the LAST page. Bring the story to a warm, satisfying "
                'ending that fits everything so far. "choices" must be an empty list [].'
            )
            choices_key = '"choices": [].\n'
        else:
            task = (
                f"Write page {page_number} only. Show what happens because of the reader's decision, "
                "move the story forward with something new, and end on a moment where the reader "
                "could decide what happens next. Do not end the story."
            )
            choices_key = _CHOICES_KEY
        decision = action or "Continue the adventure in the most natural way."
        user = (
            f"Story title: {story.get('title') or 'Untitled'}\n"
            f"The reader's original idea: \"{story['premise']}\"\n"
            + (f"Main character's name: {story['hero_name']}\n" if story.get("hero_name") else "")
            + f"Main character looks like: {story.get('character_look') or 'not described yet'}\n"
            f"Setting: {story.get('world') or 'not described yet'}\n"
            f"Previous page's place: {ready[-1].get('setting') or 'not described'} "
            "(stay there or move somewhere new, as the story needs)\n"
            f"Story so far: {story.get('summary') or 'see the previous page'}\n\n"
            f"Previous page:\n{previous}\n\n"
            f"The reader decided: \"{decision}\"\n\n"
            f"{task}\n\n"
            "Use these JSON keys in this order:\n"
            + _PLAN_KEYS +
            f'"segment": the text of page {page_number}, written from the plan above.\n'
            + _COMMON_KEYS + choices_key + _SUMMARY_KEY
        )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def response_schema(first: bool, final: bool, drawn_hero: bool = False) -> dict:
    """JSON schema handed to llama-server so its grammar forces well-formed output."""
    props: dict = {}
    if first:
        props["title"] = {"type": "string"}
    props["setting"] = {"type": "string"}
    props["visual_details"] = {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 3}
    props["action"] = {"type": "string", "enum": list(HERO_ACTIONS)}
    if first and drawn_hero:
        props["hero_name"] = {"type": "string"}
    props["segment"] = {"type": "string"}
    if first:
        props["character_look"] = {"type": "string"}
        props["world"] = {"type": "string"}
    props["visual_prompt"] = {"type": "string"}
    count = 0 if final else 3
    props["choices"] = {"type": "array", "items": {"type": "string"}, "minItems": count, "maxItems": count}
    props["continuity_summary"] = {"type": "string"}
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


# --------------------------------------------------------------------------
# Streaming client
# --------------------------------------------------------------------------

# Ways of asking llama-server for JSON, best first. Older builds reject the first form with
# HTTP 400, so we fall back and remember which one worked.
_FORMAT_STYLES = ("json_schema", "json_object", "none")
_format_style_index = 0


@dataclass
class LLMStream:
    """One streaming request. `raw` grows as tokens arrive and may be read from another thread."""

    messages: list[dict]
    schema: dict
    temperature: float
    raw: str = ""
    first_token_seconds: float | None = None
    total_seconds: float | None = None
    tokens: int | None = None
    tokens_per_second: float | None = None
    _conn: http.client.HTTPConnection | None = field(default=None, repr=False)
    _cancelled: threading.Event = field(default_factory=threading.Event, repr=False)

    def cancel(self) -> None:
        """Stop generation. Closing the connection makes llama-server stop producing tokens."""
        self._cancelled.set()
        conn = self._conn
        if conn is not None:
            try:
                if conn.sock is not None:
                    # shutdown (not just close) wakes up the thread that is waiting to read
                    conn.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def _payload(self, style: str) -> bytes:
        body: dict = {
            "messages": self.messages,
            "temperature": self.temperature,
            "top_p": 0.9,
            "max_tokens": settings.llm_max_tokens,
            "stream": True,
            "cache_prompt": True,
        }
        if style == "json_schema":
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "story_page", "strict": True, "schema": self.schema},
            }
        elif style == "json_object":
            body["response_format"] = {"type": "json_object"}
        return json.dumps(body).encode("utf-8")

    def run(self) -> str:
        """Blocking. Call it through asyncio.to_thread so the web server stays responsive."""
        global _format_style_index
        url = urlparse(settings.llm_url)
        started = time.monotonic()
        last_error = "no attempt made"
        for index in range(_format_style_index, len(_FORMAT_STYLES)):
            style = _FORMAT_STYLES[index]
            if self._cancelled.is_set():
                raise WriterFailed("cancelled")
            conn_cls = http.client.HTTPSConnection if url.scheme == "https" else http.client.HTTPConnection
            conn = conn_cls(url.hostname or "127.0.0.1", url.port or 8080, timeout=settings.llm_timeout)
            self._conn = conn
            try:
                try:
                    conn.request(
                        "POST",
                        (url.path.rstrip("/") or "") + "/v1/chat/completions",
                        body=self._payload(style),
                        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
                    )
                    response = conn.getresponse()
                except (ConnectionError, TimeoutError, OSError) as exc:
                    if self._cancelled.is_set():
                        raise WriterFailed("cancelled") from exc
                    raise WriterOffline(str(exc)) from exc

                if response.status == 400 and style != "none":
                    last_error = response.read(600).decode("utf-8", "replace")
                    continue  # this build does not understand that response_format; try the next
                if response.status == 503:
                    raise WriterOffline("llama-server is still loading the model")
                if response.status != 200:
                    detail = response.read(600).decode("utf-8", "replace")
                    raise WriterFailed(f"llama-server returned HTTP {response.status}: {detail}")

                try:
                    self._read_stream(response, started)
                finally:
                    # Closing the response releases the socket straight away, which is what
                    # tells llama-server to stop generating when a page is cancelled.
                    response.close()
                _format_style_index = index
                self.total_seconds = time.monotonic() - started
                return self.raw
            except (TimeoutError, OSError, http.client.HTTPException) as exc:
                if self._cancelled.is_set():
                    raise WriterFailed("cancelled") from exc
                if isinstance(exc, TimeoutError):
                    raise WriterFailed("llama-server took too long to answer") from exc
                raise WriterOffline(str(exc)) from exc
            finally:
                try:
                    conn.close()
                except OSError:
                    pass
        raise WriterFailed(f"llama-server rejected the request: {last_error}")

    def _read_stream(self, response: http.client.HTTPResponse, started: float) -> None:
        deadline = started + settings.llm_timeout
        while True:
            if self._cancelled.is_set():
                raise WriterFailed("cancelled")
            if time.monotonic() > deadline:
                raise WriterFailed("llama-server took too long to answer")
            line = response.readline()
            if not line:
                return
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                return
            try:
                event = json.loads(data)
            except ValueError:
                continue
            if isinstance(event.get("error"), dict):
                raise WriterFailed(str(event["error"].get("message", "unknown error")))
            timings = event.get("timings")
            if isinstance(timings, dict):
                self.tokens = timings.get("predicted_n", self.tokens)
                self.tokens_per_second = timings.get("predicted_per_second", self.tokens_per_second)
            usage = event.get("usage")
            if isinstance(usage, dict) and self.tokens is None:
                self.tokens = usage.get("completion_tokens")
            for choice in event.get("choices") or []:
                piece = (choice.get("delta") or {}).get("content")
                if piece:
                    if self.first_token_seconds is None:
                        self.first_token_seconds = time.monotonic() - started
                    self.raw += piece


def check_writer() -> tuple[bool, str]:
    """Quick health probe used by /api/health."""
    url = urlparse(settings.llm_url)
    try:
        conn = http.client.HTTPConnection(url.hostname or "127.0.0.1", url.port or 8080, timeout=3)
        conn.request("GET", "/health")
        response = conn.getresponse()
        response.read()
        conn.close()
        if response.status == 200:
            return True, "ready"
        if response.status == 503:
            return False, "loading the model"
        return False, f"HTTP {response.status}"
    except OSError as exc:
        return False, f"not reachable ({exc.__class__.__name__})"


# --------------------------------------------------------------------------
# Parsing and validation. The model is never trusted to follow instructions.
# --------------------------------------------------------------------------

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SPACES = re.compile(r"[ \t\u00a0]+")


def clean_text(text: str, limit: int) -> str:
    """Normalise whitespace, drop control characters and cap the length at a sentence end."""
    text = _CONTROL.sub("", str(text)).replace("\r\n", "\n").replace("\r", "\n")
    text = _SPACES.sub(" ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) > limit:
        cut = text[:limit]
        end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "), cut.rfind(".\n"))
        text = cut[: end + 1] if end > limit * 0.5 else cut.rstrip() + "…"
    return text


def clean_line(text: str, limit: int) -> str:
    """Single-line version for titles, choices and prompts."""
    text = clean_text(text, limit * 2).replace("\n", " ")
    text = _SPACES.sub(" ", text).strip(" \"'")
    if len(text) > limit:
        text = text[:limit].rsplit(" ", 1)[0].rstrip(",;:- ")
    return text


def extract_string_field(raw: str, key: str) -> tuple[str | None, bool]:
    """Pull one string value out of JSON that may still be arriving or may be broken.

    Returns (text, complete). `complete` is True once the closing quote has been seen.
    """
    match = re.search(r'"' + re.escape(key) + r'"\s*:\s*"', raw)
    if not match:
        return None, False
    i = match.end()
    out: list[str] = []
    n = len(raw)
    while i < n:
        ch = raw[i]
        if ch == '"':
            return _decode_json_string("".join(out)), True
        if ch == "\\":
            if i + 1 >= n:
                break  # escape sequence not finished yet
            if raw[i + 1] == "u" and i + 6 > n:
                break
            width = 6 if raw[i + 1] == "u" else 2
            out.append(raw[i : i + width])
            i += width
            continue
        out.append(ch)
        i += 1
    return _decode_json_string("".join(out)), False


def _string_list(raw: str, key: str) -> list[str] | None:
    """A finished list of strings ("key": ["a", "b"]) from JSON that may still be arriving."""
    block = re.search(r'"' + re.escape(key) + r'"\s*:\s*\[(.*?)\]', raw, re.S)
    if not block:
        return None
    return [_decode_json_string(m) for m in re.findall(r'"((?:[^"\\]|\\.)*)"', block.group(1))]


def clean_details(value) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if isinstance(item, str):
            item = clean_line(item, 60).rstrip(".")
            if len(item) >= 3 and item.lower() not in (x.lower() for x in out):
                out.append(item)
    return out[:3]


def clean_action(value) -> str:
    value = value.strip().lower() if isinstance(value, str) else ""
    return value if value in HERO_ACTIONS else HERO_ACTIONS[0]


def extract_plan(raw: str) -> dict | None:
    """The page plan, once all of it has streamed in (it comes before the page text):
    {"setting", "visual_details", "action"}. None until then."""
    setting, setting_done = extract_string_field(raw, "setting")
    details = _string_list(raw, "visual_details")
    action, action_done = extract_string_field(raw, "action")
    if not (setting_done and setting and details is not None and action_done):
        return None
    return {
        "setting": clean_line(setting, 180),
        "visual_details": clean_details(details),
        "action": clean_action(action),
    }


def _decode_json_string(body: str) -> str:
    try:
        return json.loads('"' + body.replace("\n", "\\n").replace("\t", " ") + '"')
    except ValueError:
        return body.replace("\\n", "\n").replace('\\"', '"').replace("\\\\", "\\")


def _loads_lenient(raw: str) -> dict | None:
    """Parse JSON that may be wrapped in a code fence, have chatter around it, or be cut short."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    if start < 0:
        return None
    text = text[start:]
    end = text.rfind("}")
    candidates = [text[: end + 1]] if end > 0 else []
    # A reply cut off by the token limit: try to close what was left open.
    candidates += [text + suffix for suffix in ('"}', '"]}', "]}", "}")]
    for candidate in candidates:
        try:
            value = json.loads(candidate, strict=False)
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    return None


FALLBACK_CHOICES = (
    "Look more closely at what is nearby",
    "Call out and see who answers",
    "Take the bravest path forward",
)


def word_count(text: str) -> int:
    return len(text.split())


def minimum_words() -> int:
    """Shortest page we accept. Lenient on purpose: a slightly short page beats a retry."""
    return max(25, settings.segment_min_words // 2)


@dataclass
class Page:
    segment: str
    visual_prompt: str
    choices: list[str]
    summary: str
    title: str = ""
    character_look: str = ""
    world: str = ""
    setting: str = ""
    hero_name: str = ""
    visual_details: list[str] = field(default_factory=list)
    hero_action: str = HERO_ACTIONS[0]
    used_fallback_choices: bool = False
    repaired: bool = False


def parse_page(raw: str, first: bool, final: bool) -> Page:
    """Validate the model's reply. Raises WriterFailed when no usable page text exists."""
    data = _loads_lenient(raw)
    repaired = data is None
    if data is None:
        # Salvage field by field from broken JSON.
        data = {}
        for key in ("title", "setting", "action", "hero_name", "segment", "character_look", "world", "visual_prompt", "continuity_summary"):
            value, _complete = extract_string_field(raw, key)
            if value:
                data[key] = value
        details = _string_list(raw, "visual_details")
        if details:
            data["visual_details"] = details
        block = re.search(r'"choices"\s*:\s*\[(.*?)(\]|$)', raw, re.S)
        if block:
            data["choices"] = re.findall(r'"((?:[^"\\]|\\.)*)"', block.group(1))

    segment = data.get("segment") or data.get("story") or data.get("text") or ""
    if not isinstance(segment, str):
        segment = ""
    segment = clean_text(segment, 1800)
    if word_count(segment) < minimum_words():
        raise WriterFailed(f"page text too short ({word_count(segment)} words)")
    if segment.lstrip().startswith(("{", "[")) or '"visual_prompt"' in segment:
        raise WriterFailed("page text contains JSON")

    choices: list[str] = []
    raw_choices = data.get("choices")
    if isinstance(raw_choices, list) and not final:
        seen: set[str] = set()
        for item in raw_choices:
            if isinstance(item, dict):  # some models answer [{"text": "..."}]
                item = item.get("text") or item.get("choice") or ""
            if not isinstance(item, str):
                continue
            choice = clean_line(re.sub(r"^\s*(?:\d+[.)]|[-*•])\s*", "", item), 80).rstrip(".")
            key = choice.lower()
            if len(choice) >= 3 and key not in seen:
                seen.add(key)
                choices.append(choice)
            if len(choices) == 4:
                break
    used_fallback = False
    if not final and len(choices) < 2:
        used_fallback = True
        choices = (choices + [c for c in FALLBACK_CHOICES if c.lower() not in {x.lower() for x in choices}])[:3]

    def line(key: str, limit: int) -> str:
        value = data.get(key)
        return clean_line(value, limit) if isinstance(value, str) else ""

    visual = line("visual_prompt", 170)
    if not visual:
        visual = clean_line(segment.split(".")[0], 200)
    summary = data.get("continuity_summary") or data.get("summary") or ""
    summary = clean_line(summary, 520) if isinstance(summary, str) else ""

    return Page(
        segment=segment,
        visual_prompt=visual,
        choices=[] if final else choices,
        summary=summary,
        title=line("title", 70) if first else "",
        character_look=line("character_look", 180) if first else "",
        world=line("world", 180) if first else "",
        setting=line("setting", 180),
        hero_name=line("hero_name", 40) if first else "",
        visual_details=clean_details(data.get("visual_details")),
        hero_action=clean_action(data.get("action")),
        used_fallback_choices=used_fallback,
        repaired=repaired,
    )
