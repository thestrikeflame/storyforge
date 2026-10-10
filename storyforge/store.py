"""Story sessions: the structured state for each story, kept in memory and saved to disk.

Durability: every story is a small JSON file in data/sessions/. It survives a browser
refresh and a restart of the app. Stories untouched for 14 days are deleted at startup,
together with their pictures and narration files.
"""
from __future__ import annotations

import json
import os
import re
import secrets
import time
from pathlib import Path

from .config import settings
from .sysinfo import log

SESSION_ID = re.compile(r"^[a-f0-9]{16}$")
KEEP_DAYS = 14

# Keys that only make sense while the app is running and are never written to disk.
_RUNTIME_KEYS = ("partial", "text_complete")


def hero_path(story_id: str) -> Path:
    """Where the child's hero drawing for a story is kept (a transparent PNG, made on /draw)."""
    return settings.data_dir / "heroes" / f"{story_id}.png"


def new_image_state() -> dict:
    return {"status": "none", "progress": 0.0, "error": None, "file": None, "seconds": None, "attempts": 0}


def new_segment(index: int, action: str | None, kind: str) -> dict:
    return {
        "index": index,
        "status": "writing",  # writing | ready | failed
        "text": "",
        "partial": "",
        "text_complete": False,
        "visual_prompt": "",
        "setting": "",  # where this page happens; the background is painted from it
        "choices": [],
        "action": action,  # the reader's decision that led to this page
        "kind": kind,  # opening | choice | custom | continue | finish
        "final": kind == "finish",
        "error": None,
        "image": new_image_state(),
        "timings": {},
    }


def new_story(premise: str) -> dict:
    now = time.time()
    return {
        "id": secrets.token_hex(8),
        "created": now,
        "updated": now,
        "premise": premise,
        "title": "",
        "character_look": "",
        "world": "",
        "summary": "",  # compact continuity summary sent to the model instead of the full history
        "direction": "",  # the most recent decision: where the story is heading
        "decisions": [],  # every decision the reader has made, in order
        "segments": [],
        "ended": False,
        "abandoned": False,
        "seed": secrets.randbelow(2_000_000_000),  # same seed for every picture keeps the style steadier
        "hero": None,  # set (to a version number) once the child has drawn the hero on /draw
    }


class StoryStore:
    def __init__(self) -> None:
        self.dir: Path = settings.data_dir / "sessions"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.stories: dict[str, dict] = {}

    # ---- disk ----

    def _path(self, story_id: str) -> Path:
        return self.dir / f"{story_id}.json"

    def save(self, story: dict) -> None:
        story["updated"] = time.time()
        data = dict(story)
        data["segments"] = [
            {k: v for k, v in seg.items() if k not in _RUNTIME_KEYS} for seg in story["segments"]
        ]
        tmp = self._path(story["id"]).with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
            os.replace(tmp, self._path(story["id"]))  # atomic: never leaves a half-written file
        except OSError as exc:
            log.error("could not save story %s: %s", story["id"], exc)

    def get(self, story_id: str) -> dict | None:
        if not SESSION_ID.match(story_id):
            return None
        story = self.stories.get(story_id)
        if story is not None:
            return story
        try:
            story = json.loads(self._path(story_id).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(story, dict) or story.get("id") != story_id:
            return None
        self._repair_after_restart(story)
        self.stories[story_id] = story
        return story

    def add(self, story: dict) -> None:
        self.stories[story["id"]] = story
        self.save(story)

    @staticmethod
    def _repair_after_restart(story: dict) -> None:
        """Work that was in flight when the app stopped cannot finish; mark it retryable."""
        for seg in story.get("segments", []):
            seg.setdefault("partial", "")
            seg.setdefault("text_complete", seg.get("status") == "ready")
            seg.setdefault("image", new_image_state())
            if seg["status"] == "writing":
                seg["status"] = "failed"
                seg["error"] = {"code": "interrupted", "message": "The storyteller was restarted while writing this page."}
            if seg["image"]["status"] in ("queued", "generating"):
                seg["image"]["status"] = "failed"
                seg["image"]["error"] = "interrupted"

    def cleanup_old(self) -> None:
        cutoff = time.time() - KEEP_DAYS * 86400
        removed = 0
        for path in self.dir.glob("*.json"):
            try:
                if path.stat().st_mtime >= cutoff:
                    continue
                story_id = path.stem
                if not SESSION_ID.match(story_id):
                    continue
                # Only ever delete files this app created (they carry the sf_<story id> prefix).
                for folder in (settings.image_dir, settings.data_dir / "audio"):
                    for extra in folder.glob(f"sf_{story_id}_*"):
                        extra.unlink(missing_ok=True)
                hero_path(story_id).unlink(missing_ok=True)
                path.unlink(missing_ok=True)
                removed += 1
            except OSError:
                continue
        if removed:
            log.info("removed %d stories older than %d days", removed, KEEP_DAYS)


# ---- what the browser is allowed to see ----

def segment_view(story: dict, seg: dict, image_eta: float | None = None) -> dict:
    image = seg["image"]
    has_image = image["status"] == "completed" and image.get("file")
    return {
        "index": seg["index"],
        "status": seg["status"],
        "text": seg["text"] if seg["text"] else seg.get("partial", ""),
        "text_complete": bool(seg.get("text_complete")) or seg["status"] == "ready",
        "choices": seg["choices"] if seg["status"] == "ready" else [],
        "action": seg.get("action"),
        "final": bool(seg.get("final")),
        "error": seg.get("error"),
        "image": {
            "status": image["status"],
            "progress": round(float(image.get("progress") or 0), 2),
            "error": image.get("error"),
            "url": f"/api/stories/{story['id']}/segments/{seg['index']}/image?v={image.get('attempts', 0)}"
            if has_image
            else None,
            "alt": seg.get("setting") or seg.get("visual_prompt") or "",
            "eta_seconds": image_eta,
        },
    }


def story_view(story: dict, image_eta: float | None = None) -> dict:
    return {
        "id": story["id"],
        "title": story.get("title") or "",
        "premise": story["premise"],
        "ended": bool(story.get("ended")),
        # Always answers: the child's drawing, or the placeholder hero until one is drawn.
        "hero_url": f"/api/stories/{story['id']}/hero?v={story.get('hero') or 0}",
        "segments": [segment_view(story, seg, image_eta) for seg in story["segments"]],
    }
