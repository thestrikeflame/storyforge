"""The LED strip behind the book: an ESP32 running FastLED on USB serial (firmware/esp32_leds).

Protocol: 115200 baud, one line "C," + r,g,b for every LED + "\n"; the ESP32 answers "OK". Opening
the port resets the board, which prints "READY" once it has booted.

When the browser shows a page, its background is split top to bottom into one band per LED (the
centre 70% of the width), each band's dominant colour is found with a small k-means (large,
saturated clusters beat gray ones), the colours are made a little richer, smoothed only between
neighbours that are already alike (so a sky/grass edge stays sharp), gamma-corrected, kept dim, and
faded in from the previous colours. With no story open the strip glows gently; it is switched off
on shutdown. Without an ESP32 (or with LED_ENABLED=0) nothing happens.

Test: python -m storyforge.leds --image path.png
"""
from __future__ import annotations

import argparse
import errno
import glob
import logging
import logging.handlers
import math
import os
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np

from .config import settings

log = logging.getLogger("storyforge")
# Every raw line to and from the ESP32 (logs/leds.log; the main log only gets what matters).
wire = logging.getLogger("storyforge.leds.wire")


def setup_wire_log(to_screen: bool = False) -> None:
    if wire.handlers:
        return
    if to_screen:
        handler: logging.Handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("  %(message)s"))
    else:
        settings.log_dir.mkdir(parents=True, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(settings.log_dir / "leds.log", maxBytes=1_000_000,
                                                       backupCount=3, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s.%(msecs)03d  %(message)s", "%Y-%m-%d %H:%M:%S"))
    wire.addHandler(handler)
    wire.setLevel(logging.INFO)
    wire.propagate = False

Color = tuple[int, int, int]

BAUD = 115200
READY_SECONDS = 5.0      # the board resets when the port opens; READY follows within a couple of seconds
REPLY_SECONDS = 0.5      # each frame is answered with "OK"
FADE_SECONDS = 1.0
FADE_FRAMES = 10
IDLE_COLOR = (255, 150, 60)  # a warm glow while no story is open
IDLE_PERIOD = 6.0            # seconds per slow "breath"
IDLE_LEVEL = (0.04, 0.12)    # dimmest and brightest share of full brightness
RETRY_SECONDS = 15.0         # a missing or unplugged board is looked for again after this long
CENTRE = 0.70
SATURATION_BOOST = 1.20
GAMMA = 2.2
SMOOTH_CLOSE = 60.0          # neighbours closer than this (RGB distance) are blended


# ---------------------------------------------------------------- colours from a picture

def _saturation(rgb: np.ndarray) -> np.ndarray:
    high = rgb.max(axis=-1)
    low = rgb.min(axis=-1)
    return np.where(high > 0, (high - low) / np.maximum(high, 1e-6), 0.0)


def _kmeans(pixels: np.ndarray, k: int = 3, rounds: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """A small, deterministic k-means. Starts from the darkest, middle and brightest pixels.
    Returns (centres, how many pixels each one has)."""
    if len(pixels) <= k:
        return pixels.astype(float), np.ones(len(pixels))
    order = np.argsort(pixels.sum(axis=1))
    centres = pixels[order[np.linspace(0, len(pixels) - 1, k).astype(int)]].astype(float)
    labels = np.zeros(len(pixels), dtype=int)
    for _ in range(rounds):
        distance = ((pixels[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
        labels = distance.argmin(axis=1)
        for c in range(k):
            members = pixels[labels == c]
            if len(members):
                centres[c] = members.mean(axis=0)
    counts = np.bincount(labels, minlength=k).astype(float)
    return centres, counts


def dominant_color(pixels: np.ndarray) -> np.ndarray:
    """The band's colour: the k-means cluster scoring best on size, with saturated colours preferred
    over gray (a gray cluster needs to be much bigger to win)."""
    centres, counts = _kmeans(pixels.reshape(-1, 3).astype(float))
    share = counts / max(counts.sum(), 1.0)
    score = share * (0.25 + _saturation(centres / 255.0))
    return centres[int(score.argmax())]


def boost_saturation(rgb: np.ndarray, factor: float = SATURATION_BOOST) -> np.ndarray:
    """Push each colour away from its own gray by `factor` (about +20% saturation), keeping its brightness."""
    gray = rgb.mean(axis=-1, keepdims=True)
    return np.clip(gray + (rgb - gray) * factor, 0, 255)


def smooth(colors: np.ndarray, close: float = SMOOTH_CLOSE) -> np.ndarray:
    """Edge-aware smoothing: each LED is blended with a neighbour only when their colours are
    already close, so soft gradients get smoother and real edges (sky/grass) stay sharp."""
    out = colors.copy()
    for i in range(len(colors)):
        total, weight = colors[i] * 2.0, 2.0
        for j in (i - 1, i + 1):
            if 0 <= j < len(colors) and np.linalg.norm(colors[i] - colors[j]) < close:
                total += colors[j]
                weight += 1.0
        out[i] = total / weight
    return out


def to_strip(colors: np.ndarray, brightness: float) -> list[Color]:
    """Gamma 2.2 (LEDs are linear, eyes are not) and the brightness cap."""
    linear = (np.clip(colors, 0, 255) / 255.0) ** GAMMA
    return [tuple(int(round(v)) for v in c) for c in linear * 255.0 * brightness]


def band_colors(image, count: int) -> np.ndarray:
    """One dominant colour per horizontal band, top to bottom, from the centre 70% of the width
    (before any boosting, smoothing or gamma)."""
    from PIL import Image

    picture = image.convert("RGB")
    picture = picture.resize((96, max(count * 6, 6)), Image.BILINEAR)  # small is plenty, and fast
    pixels = np.asarray(picture, dtype=float)
    width = pixels.shape[1]
    margin = int(round(width * (1 - CENTRE) / 2))
    pixels = pixels[:, margin: width - margin]
    bands = np.array_split(pixels, count, axis=0)
    return np.array([dominant_color(band) for band in bands])


def colors_from_image(image, count: int | None = None, brightness: float | None = None,
                      reverse: bool | None = None) -> list[Color]:
    """The strip's colours for a picture (a PIL image or a path): LED 0 is the top band unless
    LED_REVERSE is set."""
    from PIL import Image

    count = count or settings.led_count
    brightness = settings.led_brightness if brightness is None else brightness
    reverse = settings.led_reverse if reverse is None else reverse
    if not hasattr(image, "convert"):
        with Image.open(image) as opened:
            image = opened.copy()
    colors = smooth(boost_saturation(band_colors(image, count)))
    strip = to_strip(colors, brightness)
    return strip[::-1] if reverse else strip


# ---------------------------------------------------------------- the serial port

def find_port() -> str | None:
    ports = sorted(glob.glob("/dev/ttyUSB*")) + sorted(glob.glob("/dev/ttyACM*"))
    return ports[0] if ports else None


def permission_help(port: str) -> str:
    return (f"No permission to use the LED strip on {port}. Fix it once with:  "
            f"sudo usermod -aG dialout $USER  (then log out and back in, and restart StoryForge). "
            f"For this session only:  sudo chmod a+rw {port}")


def _denied(exc: Exception) -> bool:
    """pyserial reports "permission denied" as a SerialException carrying errno 13."""
    return isinstance(exc, PermissionError) or getattr(exc, "errno", None) == errno.EACCES


def frame_line(colors: list[Color]) -> bytes:
    return ("C," + ",".join(str(max(0, min(255, int(v)))) for c in colors for v in c) + "\n").encode("ascii")


class Board:
    """One open connection to the ESP32. Raises an OSError (pyserial's SerialException is one) when
    it cannot be opened.

    ESP32 boards wire the USB chip's DTR to IO0 and RTS to EN (reset). Holding them the wrong way
    while the port opens leaves the board in its bootloader ("waiting for download") instead of
    our sketch, so both are released before opening, and the board is then reset cleanly into the
    sketch: RTS pulses EN low while DTR keeps IO0 high."""

    def __init__(self, port: str):
        import serial  # pyserial; imported here so StoryForge runs without it when there is no strip

        self.port = port
        self.received: list[str] = []  # every line while waiting for READY (for the log)
        self.serial = serial.Serial()  # no port yet, so nothing is opened with DTR/RTS held
        self.serial.baudrate = BAUD
        self.serial.timeout = 0.1
        self.serial.write_timeout = 1.0
        self.serial.exclusive = True
        self.serial.dtr = False
        self.serial.rts = False
        self.serial.port = port
        self.serial.open()
        try:
            self.reset()
            self.ready = self._wait_for("READY", READY_SECONDS, keep=self.received)
        except Exception:
            self.close()
            raise

    def reset(self) -> None:
        """A clean reset into the sketch: EN pulsed low, IO0 left high."""
        try:
            self.serial.dtr = False
            self.serial.rts = True
            time.sleep(0.1)
            self.serial.rts = False
        except OSError as exc:
            if exc.errno not in (errno.EINVAL, errno.ENOTTY):
                raise
            wire.info("%s  this port has no DTR/RTS lines; not reset", self.port)  # like pyserial's own open()
            return
        self.serial.reset_input_buffer()
        wire.info("%s  reset: RTS pulsed (EN low 0.1 s), DTR kept inactive (IO0 high)", self.port)

    def _wait_for(self, word: str, seconds: float, keep: list[str] | None = None) -> bool:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            raw = self.serial.readline()
            if not raw:
                continue
            line = raw.decode("ascii", "replace").strip()
            wire.info("%s  <- %r", self.port, raw)
            if keep is not None and line:
                keep.append(line)
            if line == word:
                return True
        return False

    def send(self, colors: list[Color]) -> bool:
        """Send one frame; True when the board answered OK."""
        line = frame_line(colors)
        self.serial.reset_input_buffer()
        wire.info("%s  -> %s", self.port, line.decode("ascii").strip())
        self.serial.write(line)
        self.serial.flush()
        return self._wait_for("OK", REPLY_SECONDS)

    def close(self) -> None:
        try:
            self.serial.close()
        except OSError:
            pass


def ready_problem(received: list[str]) -> str:
    """Why READY did not come, in plain words, with what the board said instead."""
    said = " | ".join(received[-8:]) if received else "nothing at all"
    text = " ".join(received).lower()
    if "waiting for download" in text or "download" in text:
        why = "it is in its bootloader (download mode), not running the LED sketch"
    elif "brownout" in text:
        why = "it is resetting because of a brownout: the strip probably needs its own 5 V power"
    elif received:
        why = "it is running something else, or a sketch that does not print READY"
    else:
        why = "it said nothing: check the cable, the baud rate (115200) and that the sketch is flashed"
    return f"no READY from the ESP32 within {READY_SECONDS:.0f} s; {why}. It sent: {said}"


class Strip:
    """Owns the board on a background thread, so pages never wait for the LEDs."""

    def __init__(self, count: int | None = None, enabled: bool | None = None, port: str | None = None):
        self.count = count or settings.led_count
        self.port = port or settings.led_port  # empty: found by itself
        self.enabled = settings.led_enabled if enabled is None else enabled
        self.board: Board | None = None
        self.state = "off" if not self.enabled else "looking"
        self.current: list[Color] = [(0, 0, 0)] * self.count
        self._jobs: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._last_try = -RETRY_SECONDS
        self._told: set[str] = set()  # each problem is logged once

    # ---- the app's side
    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        setup_wire_log()
        self._thread = threading.Thread(target=self._run, name="leds", daemon=True)
        self._thread.start()
        self.idle()

    def show(self, colors: list[Color]) -> None:
        if self._thread is not None:
            self._jobs.put(("show", colors))

    def idle(self) -> None:
        if self._thread is not None:
            self._jobs.put(("idle", None))

    def stop(self) -> None:
        """LEDs off and the port closed (on shutdown)."""
        if self._thread is None:
            return
        self._jobs.put(("off", None))
        self._thread.join(timeout=FADE_SECONDS + 3)
        self._thread = None

    def describe(self) -> dict:
        return {"enabled": self.enabled, "state": self.state, "port": self.board.port if self.board else None,
                "count": self.count}

    # ---- the thread
    def _connect(self) -> bool:
        if self.board is not None:
            return True
        if time.monotonic() - self._last_try < RETRY_SECONDS:
            return False
        self._last_try = time.monotonic()
        port = self.port or find_port()
        if not port:
            self.state = "no strip found"
            self._tell("none", logging.INFO, "no LED strip found (no /dev/ttyUSB* or /dev/ttyACM*); LEDs are off")
            return False
        try:
            self.board = Board(port)
        except Exception as exc:  # no permission, busy, unplugged, not a serial device...
            if _denied(exc):
                self.state = "no permission"
                self._tell("perm", logging.WARNING, permission_help(port))
                return False
            self.state = "cannot open"
            self._tell(f"open:{exc}", logging.WARNING, "LED strip on %s cannot be opened (%s)", port, exc)
            return False
        self.state = "ready" if self.board.ready else "no READY"
        self._told.clear()
        if self.board.ready:
            log.info("LED strip connected  port=%s  leds=%d  READY received", port, self.count)
        else:
            log.warning("LED strip on %s: %s (sending anyway; every line is in leds.log)", port,
                        ready_problem(self.board.received))
        return True

    def _tell(self, key: str, level: int, message: str, *args) -> None:
        if key not in self._told:
            self._told.add(key)
            log.log(level, message, *args)

    def _send(self, colors: list[Color]) -> None:
        if not self._connect():
            self.current = colors
            return
        try:
            ok = self.board.send(colors)
            self.current = colors
            if not ok:
                self._tell("noreply", logging.WARNING, "LED strip did not answer OK (still sending)")
        except Exception as exc:  # unplugged
            self._lost(str(exc))

    def _lost(self, why: str) -> None:
        log.warning("LED strip lost  port=%s  (%s); looking for it again", self.board.port, why)
        self.board.close()
        self.board = None
        self.state = "lost"
        self._last_try = -RETRY_SECONDS  # look again straight away (it may come back as another ttyUSB)

    def _fade_to(self, target: list[Color]) -> bool:
        """Fade from the current colours over about a second. False when a newer job arrived."""
        start = np.array(self.current, dtype=float)
        end = np.array(target, dtype=float)
        for frame in range(1, FADE_FRAMES + 1):
            if not self._jobs.empty():
                return False
            t = frame / FADE_FRAMES
            t = t * t * (3 - 2 * t)  # ease in and out
            self._send([tuple(int(round(v)) for v in c) for c in start + (end - start) * t])
            time.sleep(FADE_SECONDS / FADE_FRAMES)
        return True

    def _idle_frame(self, now: float) -> list[Color]:
        low, high = IDLE_LEVEL
        level = low + (high - low) * (0.5 - 0.5 * math.cos(2 * math.pi * now / IDLE_PERIOD))
        level *= settings.led_brightness / 0.4
        return [tuple(int(round(v * level)) for v in IDLE_COLOR)] * self.count

    def _run(self) -> None:
        mode = "idle"
        while True:
            try:
                kind, colors = self._jobs.get(timeout=0.15 if mode == "idle" else 1.0)
            except queue.Empty:
                if mode == "idle":
                    self._send(self._idle_frame(time.monotonic()))
                elif self.board is not None and not os.path.exists(self.board.port):
                    self._lost("the port is gone")  # unplugged while a page was showing
                elif self.board is None and self._connect():
                    self._send(self.current)  # the board came back: show what it missed
                continue
            if kind == "off":
                self._send([(0, 0, 0)] * self.count)
                if self.board is not None:
                    self.board.close()
                    self.board = None
                self.state = "off"
                return
            if kind == "idle":
                mode = "idle"
                self._fade_to(self._idle_frame(time.monotonic()))
            else:
                mode = "page"
                self._fade_to(colors)


# ---------------------------------------------------------------- the test command

def _block(color: Color) -> str:
    r, g, b = color
    return f"\x1b[48;2;{r};{g};{b}m      \x1b[0m"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m storyforge.leds",
                                     description="Show a picture's colours on the LED strip.")
    parser.add_argument("--image", type=Path, help="a picture (png or jpg)")
    parser.add_argument("--off", action="store_true", help="switch the LEDs off")
    args = parser.parse_args(argv)
    if not args.image and not args.off:
        parser.error("give --image path.png or --off")

    colors = [(0, 0, 0)] * settings.led_count
    if args.image:
        raw = smooth(boost_saturation(band_colors(_open(args.image), settings.led_count)))
        colors = colors_from_image(args.image)
        shown = [tuple(int(v) for v in c) for c in raw]
        print(f"{settings.led_count} LEDs, top to bottom{' (reversed on the strip)' if settings.led_reverse else ''}:")
        for i, (look, sent) in enumerate(zip(shown, colors[::-1] if settings.led_reverse else colors)):
            print(f"{i:2d} {_block(look)} {_block(tuple(min(255, v * 4) for v in sent))}  "
                  f"#{look[0]:02x}{look[1]:02x}{look[2]:02x}  sent {sent}")
        print("(left: the colour picked; right: what is sent, shown 4x brighter)")

    setup_wire_log(to_screen=True)  # every line sent and received
    port = settings.led_port or find_port()
    if not port:
        print("No LED strip found (no /dev/ttyUSB* or /dev/ttyACM*).")
        return 1
    try:
        board = Board(port)
    except Exception as exc:
        if _denied(exc):
            print(permission_help(port))
            return 1
        print(f"The LED strip on {port} cannot be opened: {exc}\n"
              "(Is StoryForge running with the strip? Only one program can use it at a time.)")
        return 1
    try:
        print(f"{port}: READY" if board.ready else f"{port}: {ready_problem(board.received)}")
        print("sent, the board answered OK" if board.send(colors) else "sent, but the board did not answer OK")
        if sys.stdin.isatty():
            input("Showing on the strip. Press Enter to finish. ")
    finally:
        board.close()
    return 0


def _open(path: Path):
    from PIL import Image

    with Image.open(path) as opened:
        return opened.copy()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sys.exit(main())
