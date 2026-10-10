"""Page backgrounds: painted by Hugging Face in the cloud, or locally by stable-diffusion.cpp.

Every picture is a BACKGROUND only (no characters): the child's hero drawing is layered on
top of it in the browser, so the same hero appears in every scene.

IMAGE_MODE=huggingface (default):
  * The background is requested as soon as the page's "setting" has streamed in, so it is
    painted in the cloud while Qwen is still writing. It uses no memory on the Jetson.
  * A request is given up after HF_TIMEOUT_SECONDS. On any failure the page falls back to the
    local queue below (HF_FALLBACK_LOCAL=1), but only once Qwen has finished the page.

When neither the cloud nor the Jetson can paint a page, it gets one of the preset backgrounds
in static/art/preset-*.jpg (the one whose place best matches the page's setting) instead of no
picture (except IMAGE_MODE=off).

IMAGE_MODE=local, and the fallback, use the queue in front of sd-cli:

Design choices for an 8 GB Jetson:
  * Exactly one picture is generated at a time. Two at once would exhaust shared memory.
  * sd-cli runs as a separate process, so the web server never freezes while it works,
    and a picture nobody is waiting for any more can be stopped safely by ending the process.
  * Free memory is checked before starting. If there is too little, the picture is skipped
    with a friendly message instead of risking an out-of-memory crash of the whole machine.
  * The output is written to a temporary name and renamed when complete, so a half-written
    picture can never be shown.

Fast mode (optional, see scripts/get_fast_models.sh):
  * LCM-LoRA lets the same SD 1.5 model finish in about 6 steps instead of 15 to 20, and with
    guidance 1 each step does half the work.
  * TAESD replaces the slow final "decode" stage with a tiny, fast decoder.
  Either is used only when its file exists and the installed sd-cli lists the option. If a
  fast-mode picture fails, the picture is repainted with the classic settings and fast mode
  is switched off until the next restart.
"""
from __future__ import annotations

import asyncio
import os
import re
import signal
import time
from collections import deque
from pathlib import Path

from . import hf_images
from .config import settings
from .sysinfo import log, memory_mb, timings

IMAGE_MODES = ("huggingface", "local", "off")
# Added to every background prompt; the hero is drawn by the child, never painted.
NO_CHARACTERS = "scenery only, empty landscape, no people, no animals, no characters"
# Added to the style. FLUX ignores "no ...", so the composition is asked for positively: without it a
# hero's feature in the prompt ("a cowboy hat") can come out as one giant object in the middle.
COMPOSITION = "wide establishing shot, open empty ground in the center foreground, no single large object in the middle"


# Ready-made backgrounds (static/art/preset-<place>.jpg) and the words that point to each.
PRESET_WORDS = {
    "castle": ("castle", "palace", "tower", "throne", "king", "queen", "knight", "dungeon", "hall"),
    "beach": ("beach", "sea", "shore", "sand", "coast", "island", "lagoon", "harbour", "harbor", "wave"),
    "village": ("village", "town", "street", "market", "square", "house", "home", "kitchen", "garden",
                "farm", "cottage", "bakery", "shop", "school", "park", "city"),
    "space": ("space", "star", "planet", "moon", "rocket", "galaxy", "comet", "orbit", "astronaut"),
    "underwater": ("underwater", "reef", "ocean floor", "seabed", "coral", "deep sea", "under the sea",
                   "submarine", "kelp"),
}


# The plan should never name characters, but a small model sometimes does ("Robot's kitchen",
# visual detail "robot"). These words are taken out of background prompts so none get painted.
CHARACTER_WORDS = frozenset("""
    person people man men woman women boy boys girl girls kid kids child children baby friend friends
    hero heroine family mom mum dad grandma grandpa granny teacher pirate pirates knight knights king
    queen prince princess wizard witch fairy fairies elf giant giants monster monsters alien aliens
    ghost robot robots dragon dragons unicorn unicorns cat cats kitten dog dogs puppy bear bears bunny
    rabbit rabbits fox foxes mouse mice owl owls bird birds fish lion lions tiger tigers hedgehog
    penguin penguins frog frogs duck ducks horse horses pony cow pig sheep monkey elephant squirrel
    turtle dolphin whale shark octopus crab seagull bee butterfly snail he she they his her their
    figure figures creature creatures character characters animal animals stickman
""".split())


def names_a_character(detail: str) -> bool:
    """A plan detail that is about a character: a character word, or a name ("Pogo, the ...",
    "a hat for Mia"): a capitalised word after the first, or a first word followed by a comma."""
    words = detail.split()
    if without_characters(detail) != detail.strip():
        return True
    if words and words[0].endswith(",") and words[0][:1].isupper():
        return True
    return any(w[:1].isupper() for w in words[1:])


def without_characters(text: str) -> str:
    """Drop character words (and their possessives) from a piece of the background prompt."""
    kept = [w for w in text.split() if re.sub(r"(?:'s|s')$", "", w.strip(",.;:!?").lower()) not in CHARACTER_WORDS]
    return " ".join(kept).strip(" ,")


def _word_key(word: str) -> str:
    """Lower case, no punctuation or possessive, singular: "Hats," -> "hat", "Bill's" -> "bill"."""
    w = re.sub(r"(?:'s|s')$", "", word.strip(",.;:!?\"'()").lower())
    if len(w) > 3 and w.endswith("es") and w[-3] in "sxz":
        return w[:-2]
    return w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w


# Things a character wears or holds. A plan detail with one of these is not scenery ("a cowboy hat"
# would be painted as a giant hat in a field), so it is dropped whole. Worn things are never shown
# as props either: the drawing already has them.
WORN_WORDS = frozenset(map(_word_key, """
    hat cap crown tiara helmet hood bonnet cape cloak scarf boot shoe sock glove mitten coat jacket
    dress shirt skirt trousers pants jeans belt vest apron tie bowtie necklace bracelet ring earring
    glasses sunglasses goggles mask costume outfit uniform clothes clothing backpack badge medal
    sneaker sandal slipper pocket ribbon bow headband bandana collar sweater jumper hoodie robe gown
    tutu wig
""".split()))
WORN_OR_HELD_WORDS = WORN_WORDS | frozenset(map(_word_key, """
    bag satchel purse wand sword shield lasso spear arrow umbrella toy doll teddy
""".split()))
# Words in a hero's description that say nothing about what the hero *is* (colours, sizes, filler),
# so they stay allowed in the background ("a blue sky" is fine with a blue hero).
_LOOK_FILLER = frozenset(map(_word_key, """
    a an the and or of with without in on at to for from by its it is are has have having wearing
    holding featuring that who which this one two three four five some very little small big large
    tall short long round tiny huge happy smiling friendly cute red orange yellow green blue purple
    pink brown black white grey gray gold golden silver dark light bright colorful colourful striped
    spotted my our your drawn drawing simple stick
""".split()))


def hero_words(story: dict) -> frozenset[str]:
    """Every meaningful word of the hero's kind, features and name ("Cowboy Bill", "a stick
    figure with a cowboy hat" -> cowboy, bill, figure, hat). None of these may reach the
    background, or the painter paints them as a giant object."""
    text = " ".join(str(story.get(k) or "") for k in ("hero_name", "hero_who", "hero_description", "character_look"))
    return frozenset(k for k in map(_word_key, text.split()) if len(k) > 1 and k not in _LOOK_FILLER)


def without_words(text: str, words: frozenset[str]) -> str:
    kept = [w for w in text.split() if _word_key(w) not in words]
    return " ".join(kept).strip(" ,")


def is_scenery(detail: str, hero: frozenset[str]) -> bool:
    """A plan detail fit for the background: sky, weather, buildings, plants, ground. Not a
    character, not anything worn or held, and nothing about the hero."""
    keys = {_word_key(w) for w in detail.split()}
    return not (names_a_character(detail) or keys & WORN_OR_HELD_WORDS or keys & hero)


def worn_or_about_the_hero(prop: str, hero: frozenset[str]) -> bool:
    """A prop that is really part of the hero (something it wears, or any word of its description
    or name): the drawing already shows it, so it is not shown again as an emoji."""
    keys = {_word_key(w) for w in prop.split()}
    return bool(keys & WORN_WORDS or keys & hero)


def preset_backgrounds() -> list[Path]:
    """The ready-made backgrounds used when no picture can be painted."""
    return sorted((settings.static_dir / "art").glob("preset-*.jpg"))


def pick_preset(presets: list[Path], place: str, fallback_index: int) -> Path:
    """The preset whose place best matches the page's setting; otherwise a steady choice per page."""
    text = f" {place.lower()} "
    best, score = None, 0
    for path in presets:
        words = PRESET_WORDS.get(path.stem.removeprefix("preset-"), ())
        hits = sum(1 for w in words if w in text)
        if hits > score:
            best, score = path, hits
    return best or presets[fallback_index % len(presets)]

_PROGRESS = re.compile(rb"(\d+)/(\d+)\s*-\s*[\d.]+\s*(?:s/it|it/s)")
# Lines in sd-cli's output that say how long each stage took; copied into the log.
_STAGE = re.compile(r"(load\w*|sampling|decode\w*|generat\w*|encod\w*|condition\w*)[^|]{0,60}?(?:taking|took|in)\s+([\d.]+)\s*(ms|s)\b", re.I)

MESSAGES = {
    "disabled": "Pictures are turned off.",
    "missing_program": "The picture maker is not installed where StoryForge expects it.",
    "missing_model": "The picture model file is missing.",
    "low_memory": "There was not enough free memory to paint this page.",
    "out_of_memory": "The picture maker ran out of memory.",
    "timeout": "The picture took too long and was stopped.",
    "failed": "The picture could not be painted this time.",
    "interrupted": "Painting was interrupted.",
    "cancelled": "Painting was cancelled.",
    "hf_no_token": "The cloud painter needs a Hugging Face key. Add HF_TOKEN to the .env file.",
    "hf_unauthorized": "The cloud painter did not accept the Hugging Face key.",
    "hf_busy": "The cloud painter is busy right now. Try again in a moment.",
    "hf_offline": "The cloud painter could not be reached. Check the internet connection.",
    "bad_mode": "IMAGE_MODE in settings.env should be huggingface, local or off.",
}


class ImageWorker:
    def __init__(self, on_change) -> None:
        self._queue: deque[tuple[dict, dict]] = deque()
        self._wake = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._current: tuple[dict, dict] | None = None
        self._process: asyncio.subprocess.Process | None = None
        self._cancel_current = False
        self._on_change = on_change  # called with the story whenever a job changes state
        self.supported_flags: set[str] = set()
        self.help_checked = False
        self.fast_disabled = False  # set when a fast-mode picture failed
        self._hf_tasks: dict[tuple[str, int], asyncio.Task] = {}  # cloud backgrounds being painted

    # ---- lifecycle ----

    async def start(self) -> None:
        settings.image_dir.mkdir(parents=True, exist_ok=True)
        await self._inspect_cli()
        self._task = asyncio.create_task(self._run(), name="image-worker")

    async def stop(self) -> None:
        for task in list(self._hf_tasks.values()):
            task.cancel()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
        await self._kill_process()

    async def _inspect_cli(self) -> None:
        """Ask the installed sd-cli which options it understands instead of assuming."""
        if settings.image_mode == "huggingface":
            log.info(
                "background mode: huggingface (%s via %s, %ds limit, key %s, local fallback %s)",
                settings.hf_image_model, settings.hf_provider, settings.hf_timeout,
                "set" if settings.hf_token else "MISSING", "on" if self.local_fallback_ok() else "off",
            )
        if not settings.sd_cli.is_file():
            return
        try:
            proc = await asyncio.create_subprocess_exec(
                str(settings.sd_cli), "--help",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            )
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=20)
        except (OSError, asyncio.TimeoutError) as exc:
            log.warning("could not read sd-cli --help (%s); using only the known-working options", exc)
            return
        text = out.decode("utf-8", "replace")
        for flag in ("--negative-prompt", "--seed", "--steps", "--sampling-method", "--cfg-scale",
                     "--taesd", "--lora-model-dir"):
            if flag in text:
                self.supported_flags.add(flag)
        self.help_checked = True
        unknown = [a for a in settings.sd_extra_args if a.startswith("--") and a not in text]
        if unknown:
            log.warning("sd-cli --help does not mention %s; check SD_EXTRA_ARGS in settings.env", ", ".join(unknown))
        log.info("sd-cli found; optional options available: %s", ", ".join(sorted(self.supported_flags)) or "none")
        log.info("picture mode: %s", self.describe_plan(self.plan()))

    def availability(self) -> tuple[bool, str]:
        mode = settings.image_mode
        if mode == "off":
            return False, "disabled"
        if mode not in IMAGE_MODES:
            return False, "bad_mode"
        if mode == "huggingface":
            if settings.hf_token or self.local_fallback_ok():
                return True, "ready"
            return False, "hf_no_token"
        return self.local_availability()

    def local_availability(self) -> tuple[bool, str]:
        if not settings.sd_cli.is_file() or not os.access(settings.sd_cli, os.X_OK):
            return False, "missing_program"
        if not settings.sd_model.is_file():
            return False, "missing_model"
        return True, "ready"

    # ---- which settings a picture is painted with ----

    def plan(self, classic: bool = False) -> dict:
        """Decide between fast mode and the classic, known-working settings."""
        fast_ok = settings.sd_fast and not self.fast_disabled and not classic
        lcm = (
            fast_ok
            and settings.sd_lcm_lora.is_file()
            and {"--lora-model-dir", "--cfg-scale", "--sampling-method"} <= self.supported_flags
        )
        taesd = fast_ok and settings.sd_taesd.is_file() and "--taesd" in self.supported_flags
        return {"lcm": lcm, "taesd": taesd, "steps": settings.sd_lcm_steps if lcm else settings.sd_steps}

    def local_fallback_ok(self) -> bool:
        return settings.hf_fallback_local and self.local_availability()[0]

    def describe_mode(self) -> str:
        if settings.image_mode == "huggingface":
            return f"huggingface ({settings.hf_image_model} via {settings.hf_provider})"
        if settings.image_mode == "local":
            return "local, " + self.describe_plan(self.plan())
        return settings.image_mode

    @staticmethod
    def describe_plan(plan: dict) -> str:
        parts = [name for name in ("lcm", "taesd") if plan[name]]
        return (f"fast ({'+'.join(parts)})" if parts else "classic") + f", {plan['steps']} steps"

    # ---- queue ----

    def enqueue(self, story: dict, seg: dict, retry: bool = False) -> bool:
        """Start or queue the background for this page. Returns False when one is already queued,
        running or finished, which is what prevents duplicate generations.

        Cloud backgrounds start at once (even while the page is being written). Local ones only
        start once the page is ready, so sd-cli never competes with Qwen for the GPU."""
        image = seg["image"]
        if retry and image["status"] == "completed" and image.get("source") == "preset":
            image.update(status="none", file=None, source=None)  # try for a real painting again
        if image["status"] in ("queued", "generating", "completed"):
            return False
        if retry:
            image.pop("hf_failed", None)  # a retry asks the cloud again first
        ok, reason = self.availability()
        if not ok:
            if reason == "disabled":
                image.update(status="skipped", error=reason, progress=0.0)
            else:
                self.use_preset(story, seg, reason)
            return False
        if settings.image_mode == "huggingface" and settings.hf_token and not image.get("hf_failed"):
            image.update(status="generating", progress=0.05, error=None)
            image["attempts"] = int(image.get("attempts", 0)) + 1
            key = (story["id"], seg["index"])
            self._hf_tasks[key] = asyncio.create_task(self._paint_in_cloud(story, seg), name=f"hf-{key}")
            return True
        if settings.image_mode == "huggingface" and not self.local_fallback_ok():
            self.use_preset(story, seg, image.get("error") or "failed")
            return False
        if seg["status"] != "ready":
            return False  # local painting waits for the page; write_segment calls enqueue again
        image.update(status="queued", progress=0.0, error=None)
        image["attempts"] = int(image.get("attempts", 0)) + 1
        self._queue.append((story, seg))
        self._wake.set()
        return True

    async def cancel_story(self, story_id: str) -> None:
        """Drop queued pictures of an abandoned story and stop the one being painted."""
        kept: deque[tuple[dict, dict]] = deque()
        for story, seg in self._queue:
            if story["id"] == story_id:
                seg["image"].update(status="failed", error="cancelled", progress=0.0)
            else:
                kept.append((story, seg))
        self._queue = kept
        if self._current and self._current[0]["id"] == story_id:
            self._cancel_current = True
            await self._kill_process()
        for key, task in list(self._hf_tasks.items()):
            if key[0] == story_id:
                task.cancel()

    def queue_length(self) -> int:
        return len(self._queue) + (1 if self._current else 0) + len(self._hf_tasks)

    def eta_seconds(self) -> float | None:
        if settings.image_mode == "huggingface" and settings.hf_token:
            return timings.average("image", "seconds", ok=True, mode="huggingface")
        mode = self.describe_plan(self.plan())
        return timings.average("image", "seconds", ok=True, mode=mode)

    # ---- cloud ----

    async def _paint_in_cloud(self, story: dict, seg: dict) -> None:
        image = seg["image"]
        page = seg["index"] + 1
        key = (story["id"], seg["index"])
        prompt = self.build_background_prompt(story, seg)
        started = time.monotonic()
        reason = ""
        try:
            self._on_change(story)
            data, kind = await asyncio.wait_for(
                asyncio.to_thread(
                    hf_images.generate, prompt, settings.sd_width, settings.sd_height, int(story.get("seed", 42))
                ),
                timeout=settings.hf_timeout + 2,  # backstop; generate() keeps its own deadline
            )
            seconds = round(time.monotonic() - started, 1)
            name = f"sf_{story['id']}_{seg['index']}.{kind}"
            tmp_path = settings.image_dir / f"{name}.part"
            tmp_path.write_bytes(data)
            os.replace(tmp_path, settings.image_dir / name)
            for other in ("png", "jpg"):
                if other != kind:
                    (settings.image_dir / f"sf_{story['id']}_{seg['index']}.{other}").unlink(missing_ok=True)
            image.update(status="completed", progress=1.0, error=None, file=name, seconds=seconds, source="huggingface")
            log.info(
                "background painted  story=%s page=%d  %.1fs  huggingface %s  %dx%d\n    prompt: %s",
                story["id"], page, seconds, settings.hf_image_model, settings.sd_width, settings.sd_height, prompt,
            )
            timings.record("image", ok=True, seconds=seconds, mode="huggingface",
                           size=f"{settings.sd_width}x{settings.sd_height}")
            return
        except asyncio.CancelledError:
            image.update(status="failed", error="cancelled", progress=0.0)
            raise
        except asyncio.TimeoutError:
            reason = "timeout"
        except hf_images.HFFailed as exc:
            reason = exc.reason
            if str(exc) != reason:
                log.warning("cloud background details  story=%s page=%d  %s", story["id"], page, exc)
        except OSError as exc:  # e.g. the picture folder could not be written
            log.error("could not save background story=%s page=%d: %s", story["id"], page, exc)
            reason = "failed"
        finally:
            if self._hf_tasks.get(key) is asyncio.current_task():
                self._hf_tasks.pop(key, None)
            self._on_change(story)

        seconds = round(time.monotonic() - started, 1)
        timings.record("image", ok=False, reason=reason, seconds=seconds, mode="huggingface")
        if story.get("abandoned"):
            image.update(status="failed", error="cancelled", progress=0.0)
            return
        if self.local_fallback_ok():
            log.warning(
                "cloud background failed  story=%s page=%d  reason=%s after %.1fs; painting it locally instead",
                story["id"], page, reason, seconds,
            )
            image["hf_failed"] = True
            image.update(status="none", progress=0.0, error=reason)
            if seg["status"] == "ready":
                self.enqueue(story, seg)  # otherwise write_segment queues it when the page is done
        else:
            log.warning("cloud background failed  story=%s page=%d  reason=%s after %.1fs",
                        story["id"], page, reason, seconds)
            self.use_preset(story, seg, reason)
        self._on_change(story)

    def use_preset(self, story: dict, seg: dict, reason: str) -> None:
        """Nothing could paint this page: show a ready-made forest instead of no picture.
        The same story and page always get the same one. Without presets the page stays unpainted."""
        image = seg["image"]
        presets = preset_backgrounds()
        if not presets:
            image.update(status="failed", error=reason, progress=0.0)
            return
        place = " ".join([seg.get("setting") or "", *(seg.get("visual_details") or []), story.get("world") or ""])
        chosen = pick_preset(presets, place, int(story.get("seed", 0)) + seg["index"])
        name = chosen.name
        image.update(status="completed", progress=1.0, error=None, file=name, source="preset", preset_reason=reason,
                     preset_alt=f"a storybook {chosen.stem.removeprefix('preset-').replace('underwater', 'underwater reef')} scene")
        log.info("preset background used  story=%s page=%d  %s  (no painting: %s)",
                 story["id"], seg["index"] + 1, name, reason)

    # ---- worker ----

    async def _run(self) -> None:
        while True:
            while not self._queue:
                self._wake.clear()
                await self._wake.wait()
            story, seg = self._queue.popleft()
            if story.get("abandoned"):
                seg["image"].update(status="failed", error="cancelled")
                continue
            self._current = (story, seg)
            self._cancel_current = False
            try:
                await self._generate(story, seg)
            except asyncio.CancelledError:
                seg["image"].update(status="failed", error="interrupted")
                raise
            except Exception:  # never let one bad job stop the worker
                log.exception("unexpected error while painting story=%s page=%s", story["id"], seg["index"] + 1)
                self.use_preset(story, seg, "failed")
            finally:
                self._current = None
                self._process = None
                self._on_change(story)

    @staticmethod
    def build_background_prompt(story: dict, seg: dict) -> str:
        """Art style and composition FIRST, then the place, then the no-characters rule. Never the
        hero, nor any word of its kind, features or name: the child's drawing is layered on top in
        the browser.

        SD 1.5 only reads about 75 tokens (roughly 50 words) and pays most attention to the
        beginning. The style goes first so it can never be cut off, and the place is capped."""
        # The hero's props are shown as emoji beside the drawing; painted, they'd be giant objects.
        hero = hero_words(story) | {_word_key(w) for prop in seg.get("prop_names") or [] for w in prop.split()
                                    if _word_key(w) not in _LOOK_FILLER}
        place = without_words(without_characters(seg.get("setting") or story.get("world") or ""), hero)
        place = place or "a storybook landscape"
        # A detail that names a character, the hero, a prop or something worn or held is dropped whole
        # ("a sleepy cat", "a cowboy hat" are not scenery).
        details = [d.strip().rstrip(".") for d in seg.get("visual_details") or []
                   if d.strip() and is_scenery(d, hero)]
        scene = ", ".join([place.rstrip("."), *details])
        scene = " ".join(scene.split()[: settings.sd_prompt_words])
        return f"{settings.sd_style}, {COMPOSITION}, {scene}, {NO_CHARACTERS}"

    def build_command(self, prompt: str, seed: int, out_path: Path, plan: dict | None = None) -> list[str]:
        # A list of arguments (never a shell string), so story text cannot inject commands.
        plan = plan or self.plan(classic=True)
        extra = list(settings.sd_extra_args)
        if plan["lcm"]:
            prompt = f"{prompt}<lora:{settings.sd_lcm_lora.stem}:1>"
        cmd = [
            str(settings.sd_cli),
            "-m", str(settings.sd_model),
            "-p", prompt,
            "-W", str(settings.sd_width),
            "-H", str(settings.sd_height),
            "--steps", str(plan["steps"]),
        ]
        if plan["taesd"]:
            # The tiny decoder needs little memory, so tiling (a memory saver for the big one) goes.
            extra = [a for a in extra if a != "--vae-tiling"]
            cmd += ["--taesd", str(settings.sd_taesd)]
        cmd += extra
        if plan["lcm"]:
            # Guidance 1 is what LCM expects; it also halves the work per step. The negative
            # prompt has no effect at guidance 1, so it is left out.
            cmd += ["--lora-model-dir", str(settings.sd_lcm_lora.parent), "--cfg-scale", "1", "--sampling-method", "lcm"]
        elif "--negative-prompt" in self.supported_flags and settings.sd_negative:
            cmd += ["--negative-prompt", f"{settings.sd_negative}, person, people, animal, creature, character"]
        if "--seed" in self.supported_flags:
            cmd += ["--seed", str(seed)]
        cmd += ["-o", str(out_path)]
        return cmd

    async def _generate(self, story: dict, seg: dict) -> None:
        image = seg["image"]
        page = seg["index"] + 1
        mem_before = memory_mb()
        if settings.sd_min_free_mb and 0 < mem_before["available"] < settings.sd_min_free_mb:
            log.warning(
                "picture skipped  story=%s page=%d  only %d MB free (need %d MB, SD_MIN_FREE_MB)",
                story["id"], page, mem_before["available"], settings.sd_min_free_mb,
            )
            timings.record("image", ok=False, reason="low_memory", free_mb=mem_before["available"])
            self.use_preset(story, seg, "low_memory")
            return

        final_path = settings.image_dir / f"sf_{story['id']}_{seg['index']}.png"
        tmp_path = settings.image_dir / f"sf_{story['id']}_{seg['index']}.part.png"
        prompt = self.build_background_prompt(story, seg)
        seed = int(story.get("seed", 42))

        first = self.plan()
        plans = [first] if not (first["lcm"] or first["taesd"]) else [first, self.plan(classic=True)]
        result: dict = {}
        mode = ""
        for attempt, plan in enumerate(plans):
            mode = self.describe_plan(plan)
            image.update(status="generating", progress=0.02, error=None)
            self._on_change(story)
            cmd = self.build_command(prompt, seed, tmp_path, plan)
            result = await self._run_cli(cmd, plan["steps"], image, tmp_path)
            if result["ok"] or result["reason"] == "cancelled" or attempt == len(plans) - 1:
                break
            # Fast mode did not work on this machine: fall back for this and all later pictures.
            self.fast_disabled = True
            log.warning(
                "fast picture mode failed (%s, %s); repainting with the classic settings and turning fast "
                "mode off until restart. Details: logs/sd-last.log\n    last output: %s",
                mode, result["reason"], "\n    ".join(result["tail"][-6:]) or "(none)",
            )

        seconds = result["seconds"]
        if result["ok"]:
            os.replace(tmp_path, final_path)
            (settings.image_dir / f"sf_{story['id']}_{seg['index']}.jpg").unlink(missing_ok=True)
            image.update(status="completed", progress=1.0, error=None, file=final_path.name, seconds=seconds, source="local")
            log.info(
                "picture painted  story=%s page=%d  %.1fs  mode=%s  %dx%d  free_mem before=%dMB lowest=%dMB%s\n"
                "    prompt: %s",
                story["id"], page, seconds, mode, settings.sd_width, settings.sd_height,
                mem_before["available"], result["lowest_free"],
                f"  stages: {result['stages']}" if result["stages"] else "", prompt,
            )
            timings.record(
                "image", ok=True, seconds=seconds, mode=mode,
                size=f"{settings.sd_width}x{settings.sd_height}",
                free_mb_before=mem_before["available"], free_mb_lowest=result["lowest_free"],
            )
            return

        tmp_path.unlink(missing_ok=True)
        reason = result["reason"]
        if reason == "cancelled":
            image.update(status="failed", error=reason, progress=0.0, seconds=seconds)
        else:
            self.use_preset(story, seg, reason)
        timings.record("image", ok=False, reason=reason, seconds=seconds, free_mb_lowest=result["lowest_free"])
        if reason != "cancelled":
            log.error(
                "picture failed  story=%s page=%d  reason=%s exit=%s after %.1fs  lowest free mem=%dMB\n    last output: %s",
                story["id"], page, reason, result["code"], seconds, result["lowest_free"],
                "\n    ".join(result["tail"][-8:]) or "(none)",
            )

    async def _run_cli(self, cmd: list[str], steps: int, image: dict, tmp_path: Path) -> dict:
        """Run sd-cli once. Returns ok, reason, exit code, seconds, memory low point and output."""
        started = time.monotonic()
        tail: deque[str] = deque(maxlen=400)
        lowest_free = memory_mb()["available"]
        tmp_path.unlink(missing_ok=True)

        def outcome(ok: bool, reason: str, code) -> dict:
            lines = list(tail)
            stages = "; ".join(
                f"{m.group(1).lower()} {m.group(2)}{m.group(3)}" for line in lines for m in [_STAGE.search(line)] if m
            )[:300]
            try:  # the complete output of the most recent run, for troubleshooting
                (settings.log_dir / "sd-last.log").write_text(
                    "command: " + " ".join(cmd) + "\n\n" + "\n".join(lines) + "\n", encoding="utf-8"
                )
            except OSError:
                pass
            return {
                "ok": ok, "reason": reason, "code": code, "tail": lines, "stages": stages,
                "seconds": round(time.monotonic() - started, 1), "lowest_free": lowest_free,
            }

        try:
            self._process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(settings.sd_cli.parent),
                start_new_session=True,  # its own process group, so it can be stopped cleanly
            )
        except OSError as exc:
            log.error("could not start sd-cli: %s", exc)
            return outcome(False, "missing_program", None)
        proc = self._process

        async def read_output() -> None:
            assert proc.stdout
            last_was_bar = [False]
            buffer = b""
            while True:
                chunk = await proc.stdout.read(512)
                if not chunk:
                    break
                buffer = (buffer + chunk)[-4096:]
                for match in _PROGRESS.finditer(buffer):
                    done, total = int(match.group(1)), int(match.group(2))
                    if total == steps and total > 0:
                        # Sampling is roughly the first 85%; decoding the picture is the rest.
                        image["progress"] = max(image["progress"], round(0.05 + 0.8 * done / total, 2))
                # Keep whole lines for the log, drop them from the scan buffer.
                pieces = re.split(rb"[\r\n]+", buffer)
                buffer = pieces[-1]
                for piece in pieces[:-1]:
                    text = piece.decode("utf-8", "replace").strip()
                    if not text:
                        continue
                    is_bar = bool(_PROGRESS.search(piece))
                    if is_bar and last_was_bar[0] and tail:
                        tail[-1] = text[:300]  # a progress bar redrawing itself: keep only the latest
                    else:
                        tail.append(text[:300])
                    last_was_bar[0] = is_bar

        async def watch_memory() -> None:
            nonlocal lowest_free
            while True:
                await asyncio.sleep(2)
                free = memory_mb()["available"]
                if free:
                    lowest_free = min(lowest_free, free) if lowest_free else free

        reader = asyncio.create_task(read_output())
        watcher = asyncio.create_task(watch_memory())
        timed_out = False
        try:
            await asyncio.wait_for(proc.wait(), timeout=settings.sd_timeout)
        except asyncio.TimeoutError:
            timed_out = True
            await self._kill_process()
        finally:
            watcher.cancel()
            try:
                await asyncio.wait_for(reader, timeout=5)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                reader.cancel()

        code = proc.returncode
        if code == 0 and tmp_path.is_file() and tmp_path.stat().st_size > 1000 and not self._cancel_current:
            return outcome(True, "", code)
        output = " | ".join(tail).lower()
        if self._cancel_current:
            reason = "cancelled"
        elif timed_out:
            reason = "timeout"
        elif code in (-signal.SIGKILL, 137) or "out of memory" in output or "bad_alloc" in output or "failed to allocate" in output:
            reason = "out_of_memory"
        else:
            reason = "failed"
        return outcome(False, reason, code)

    async def _kill_process(self) -> None:
        proc = self._process
        if proc is None or proc.returncode is not None:
            return
        for sig, wait in ((signal.SIGTERM, 4), (signal.SIGKILL, 4)):
            try:
                os.killpg(proc.pid, sig)
            except (ProcessLookupError, PermissionError):
                try:
                    proc.send_signal(sig)
                except ProcessLookupError:
                    return
            try:
                await asyncio.wait_for(proc.wait(), timeout=wait)
                return
            except asyncio.TimeoutError:
                continue
