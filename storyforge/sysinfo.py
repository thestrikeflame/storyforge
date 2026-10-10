"""Memory readings and the timing log (how long each page and picture took)."""
from __future__ import annotations

import json
import logging
import logging.handlers
import time
from collections import deque
from pathlib import Path

from .config import settings

log = logging.getLogger("storyforge")


def setup_logging() -> None:
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-7s %(message)s", "%Y-%m-%d %H:%M:%S")
    if not log.handlers:
        to_file = logging.handlers.RotatingFileHandler(
            settings.log_dir / "storyforge.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8"
        )
        to_file.setFormatter(fmt)
        to_screen = logging.StreamHandler()
        to_screen.setFormatter(fmt)
        log.addHandler(to_file)
        log.addHandler(to_screen)
        log.setLevel(logging.INFO)
        log.propagate = False


def memory_mb() -> dict[str, int]:
    """Total and available system memory in MB (on a Jetson the GPU shares this memory)."""
    values: dict[str, int] = {}
    try:
        with open("/proc/meminfo", encoding="ascii") as handle:
            for row in handle:
                name, _, rest = row.partition(":")
                if name in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree"):
                    values[name] = int(rest.split()[0]) // 1024
    except (OSError, ValueError, IndexError):
        pass
    return {
        "total": values.get("MemTotal", 0),
        "available": values.get("MemAvailable", 0),
        "swap_total": values.get("SwapTotal", 0),
        "swap_free": values.get("SwapFree", 0),
    }


class Timings:
    """Keeps recent measurements in memory and appends each one to logs/timings.jsonl."""

    def __init__(self) -> None:
        self.recent: deque[dict] = deque(maxlen=200)
        self._path: Path = settings.log_dir / "timings.jsonl"
        self._load()

    def _load(self) -> None:
        try:
            lines = self._path.read_text(encoding="utf-8").splitlines()[-200:]
        except OSError:
            return
        for row in lines:
            try:
                self.recent.append(json.loads(row))
            except ValueError:
                continue

    def record(self, kind: str, **fields) -> None:
        entry = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "kind": kind, **fields}
        self.recent.append(entry)
        try:
            settings.log_dir.mkdir(parents=True, exist_ok=True)
            with open(self._path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry) + "\n")
        except OSError:
            pass

    def average(self, kind: str, key: str, last: int = 8, **match) -> float | None:
        values = [
            e[key]
            for e in self.recent
            if e.get("kind") == kind
            and isinstance(e.get(key), (int, float))
            and all(e.get(k) == v for k, v in match.items())
        ][-last:]
        return round(sum(values) / len(values), 1) if values else None

    def summary(self) -> dict:
        return {
            "pages_measured": sum(1 for e in self.recent if e["kind"] == "page"),
            "images_measured": sum(1 for e in self.recent if e["kind"] == "image" and e.get("ok")),
            "avg_seconds_until_page_text_complete": self.average("page", "text_seconds"),
            "avg_seconds_until_first_words": self.average("page", "first_token_seconds"),
            "avg_page_total_seconds": self.average("page", "total_seconds"),
            "avg_tokens_per_second": self.average("page", "tokens_per_second"),
            "avg_image_seconds": self.average("image", "seconds", ok=True),
            "avg_narration_synthesis_seconds": self.average("narration", "seconds"),
            "recent": list(self.recent)[-25:],
        }


timings = Timings()
