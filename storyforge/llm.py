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
from difflib import SequenceMatcher
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
    "together with a reader. {reader} Write warmly, never frightening or violent. "
    "Keep it gentle: nobody is hurt, attacked or fought. "
    "Each page has at least {min_words} and at most {max_words} words, in one or two short "
    "paragraphs, told in the past tense. Keep names, looks and places consistent with what came before. "
    "Never wrap up the whole story unless you are told it is the last page. "
    "Reply with one JSON object only, no other text."
)

@dataclass(frozen=True)
class ReadingLevel:
    key: str
    label: str
    min_words: int  # words per page
    max_words: int
    style: str  # how the pages are written (in the system prompt)
    reader: str  # who the word definitions are for
    extra_tokens: int = 0  # longer pages need a little more room


READING_LEVELS = {
    "pre": ReadingLevel(
        "pre", "Pre-reader", 30, 45,
        "The reader is a pre-reader (ages 3 to 5) who listens along: use very short, simple sentences "
        "of about 4 to 8 words, everyday words a small child knows, and a little repetition.",
        "a pre-reader aged 3 to 5"),
    "early": ReadingLevel(
        "early", "Early reader", 45, 60,
        "The reader is an early reader (kindergarten to grade 2, ages 5 to 8): use short, simple "
        "sentences and familiar words, with only a few new words that the story makes clear.",
        "an early reader aged 5 to 8"),
    "confident": ReadingLevel(
        "confident", "Confident reader", 60, 80,
        "The reader is a confident reader (grades 3 to 5, ages 8 to 11): use varied sentences and "
        "rich, vivid, descriptive words.",
        "a confident reader aged 8 to 11", extra_tokens=60),
}
DEFAULT_LEVEL = "early"


def reading_level(story: dict) -> ReadingLevel:
    """The story's reading level; Early reader for stories made before levels existed."""
    return READING_LEVELS.get(story.get("reading_level") or "", READING_LEVELS[DEFAULT_LEVEL])


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
    "scenery only: sky, weather, time of day, buildings, plants or ground. Never characters. The page "
    "must happen in this place and mention every one of these details.\n"
    '"props": a list of 0 to 2 small objects the hero holds or uses on this page, each 1 or 2 words. '
    "The page must mention them. [] when there are none.\n"
    f'"action": what the hero is doing on this page, exactly one of: {", ".join(HERO_ACTIONS)}.\n'
)
_COMMON_KEYS = (
    '"visual_prompt": at most 20 words for the illustrator: ONE main character (say what it looks '
    "like, never its name), doing ONE thing, in ONE place. Simple and concrete, no feelings.\n"
)
_CHOICES_KEY = (
    '"choices": a list of 2 or 3 choices for what the hero does next. Each is a real fork in the '
    "quest: a different place to go, or a different decision about how to reach the goal, so each "
    "leads to a different next page. Never a small action like looking, waiting or listening. Each "
    "is one short, exciting sentence of at most 10 words that starts with a verb.\n"
)
_SUMMARY_KEY = (
    '"continuity_summary": at most 60 words holding everything worth remembering so far: '
    "names, goals, objects, promises and where everyone is."
)


# Weapon-like things a child might draw. Prompts never name these (naming them makes the model add
# them); only when the hero's description already contains one is a "toy or tool" line added.
_WEAPON = re.compile(
    r"\b(knife|knives|sword|swords|sabre|saber|dagger|daggers|blade|blades|axe|axes|hatchet|spear|spears|"
    r"lance|bow and arrows?|arrows?|crossbow|gun|guns|pistol|rifle|cannon|cannons|bomb|bombs|grenade|"
    r"club|mace|whip|trident|scythe|machete|katana|weapon|weapons)\b", re.I)


def weapon_rule(description: str) -> str:
    """One line for the storyteller, only when the drawn hero already has a weapon-like thing."""
    found = _WEAPON.search(description or "")
    if not found:
        return ""
    return (f"The {found.group(1).lower()} the hero was drawn with is only a toy or a tool here, for play, "
            "cooking or building; it is never used to hurt or scare anyone.\n")


# ---------------------------------------------------------------- the reader's choice

# Scary or violent choices are made gentle before the storyteller sees them (in Python, so the
# prompts never have to name violence). Order matters: longer phrases first.
_SOFTEN = [
    (r"\b(?:scare|chase|frighten)\s+(.+?)\s+away\b", r"make friends with \1"),
    (r"\b(?:beat up|fight with|fight against|fight|battle|attack|kill|slay|defeat|destroy|hurt|hit|punch|kick|"
     r"stab|shoot|bite|smash|crush|murder|capture|trap|chase away|scare away|scare)\b", "make friends with"),
    (r"\bsteal\b", "borrow"),
]
_WEAPON_PHRASE = re.compile(r"\s*\b(?:with|using)\s+(?:a|an|the|his|her|its|their|my)?\s*(?:\w+\s+)?" + _WEAPON.pattern[2:-2] + r"\b", re.I)


def soften_choice(choice: str) -> str:
    """"fight the dragon with a sword" -> "make friends with the dragon". Gentle choices pass unchanged."""
    text = _WEAPON_PHRASE.sub("", choice)
    for pattern, gentle in _SOFTEN:
        text = re.sub(pattern, gentle, text, flags=re.I)
    text = re.sub(r"\bmake friends with\s+(?:with|against)\s+", "make friends with ", text, flags=re.I)
    text = " ".join(text.split())
    if text and choice[:1].isupper():
        text = text[0].upper() + text[1:]
    return text or choice


_SMALL_WORDS = frozenset("""
    a an the and or but to of in on at by for from with into onto over under up down out off away about
    as is are was were be been it its it's he she they them their his her him we you your our my i me
    this that these those there here then now just very really so too all some any more much one two
    go goes going went get gets got let lets make makes made take takes try tries do does did can will
    see sees look looks find finds new little big
""".split())


def _stem(word: str) -> str:
    word = word.lower().strip("'")
    for suffix in ("ing", "ed", "es", "ly", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)]
            break
    return word[:5]


def choice_words(choice: str, known: str = "") -> set[str]:
    """The main words of a choice, stemmed. Words already in `known` (the previous page) prove
    nothing, so they are left out, unless that would leave nothing to check."""
    words = {_stem(w) for w in re.findall(r"[A-Za-z']+", choice) if w.lower() not in _SMALL_WORDS and len(w) > 2}
    seen = {_stem(w) for w in re.findall(r"[A-Za-z']+", known)}
    return (words - seen) or words


def choice_followed(choice: str, page: str, known: str = "") -> bool:
    """True when the page shares at least one main word with the reader's choice."""
    wanted = choice_words(choice, known)
    if not wanted:
        return True
    found = {_stem(w) for w in re.findall(r"[A-Za-z']+", page)}
    return bool(wanted & found)


def goal_reached(goal_done: str, page: str, hero: str | None = None) -> bool:
    """Stricter than choice_followed: the last page shows the quest's goal reached only when at
    least half of the goal's main words (leaving out the hero's name) are on it."""
    wanted = choice_words(goal_done) - {_stem(w) for w in re.findall(r"[A-Za-z']+", hero or "")}
    if not wanted:
        return True
    found = {_stem(w) for w in re.findall(r"[A-Za-z']+", page)}
    return len(wanted & found) * 2 >= len(wanted)


def bridge_sentence(choice: str, hero: str | None) -> str:
    """A short opening that carries out the choice, for a page that ignored it twice.
    Offered choices start with a verb ("Open the door" -> "So Mira decided to open the door.");
    anything else the reader said is told as it is ("the jars wake up" -> "And then, the jars wake up.")."""
    text = choice.strip().rstrip(".!?")
    first = text.split()[0].lower() if text else ""
    starts_with_subject = first in {"i", "we", "he", "she", "they", "it", "the", "a", "an", "my", "our", "his",
                                    "her", "their", "everyone", "everybody", "somebody", "someone"} or (
        text[:1].isupper() and hero and text.lower().startswith(hero.lower()))
    if starts_with_subject:
        return f"And then, {text[0].lower() + text[1:] if first != 'i' else text}."
    return f"So {hero or 'our hero'} decided to {text[0].lower() + text[1:]}."


def short_summary(summary: str) -> str:
    """At most two sentences of the running summary, so the reader's choice isn't drowned out."""
    sentences = re.split(r"(?<=[.!?])\s+", (summary or "").strip())
    return " ".join(sentences[:2]).strip()


# A sentence ends at . ! or ? (maybe followed by a closing quote) and then a space.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|(?<=[.!?][\"'\u201d\u2019])\s+")


def _sentence_words(sentence: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", sentence.lower().replace("\u2019", "'"))


def _repeats(words: list[str], earlier: list[list[str]]) -> bool:
    """Mostly the same as an earlier sentence: 75% alike overall, or 80% of it found inside a
    longer one. Very short sentences ("Mira smiled.") may come back, so they are kept."""
    if len(words) < 5:
        return False
    for other in earlier:
        match = SequenceMatcher(None, words, other, autojunk=False)
        if match.ratio() >= 0.75 or sum(b.size for b in match.get_matching_blocks()) >= 0.8 * len(words):
            return True
    return False


def drop_repeats(text: str, earlier_pages: list[str], floor: int = 0,
                 finished_only: bool = False) -> tuple[str, int]:
    """Drop every finished sentence that is mostly the same as one on an earlier page (the small
    model likes to start a page by retelling how the last one ended). Paragraphs are kept.
    An unfinished last sentence (still streaming) is left alone, or left out with `finished_only`,
    so that while a page streams in, only sentences already checked are shown and nothing appears
    and then vanishes. If dropping would leave fewer than `floor` words, nothing is dropped.
    Returns (text, sentences dropped)."""
    earlier = [_sentence_words(x) for page in earlier_pages for x in _SENTENCE_END.split(page or "")]
    earlier = [w for w in earlier if w]
    if not text or not (earlier or finished_only):
        return text, 0
    dropped = 0
    paragraphs = []
    blocks = text.split("\n\n")
    for b, paragraph in enumerate(blocks):
        pieces = _SENTENCE_END.split(paragraph.strip())
        kept = []
        for i, piece in enumerate(pieces):
            finished = (i < len(pieces) - 1 or b < len(blocks) - 1
                        or bool(re.search(r"[.!?][\"'\u201d\u2019]?$", piece)))
            if not finished and finished_only:
                continue
            if finished and earlier and _repeats(_sentence_words(piece), earlier):
                dropped += 1
            else:
                kept.append(piece)
        if any(k.strip() for k in kept):
            paragraphs.append(" ".join(kept))
    result = "\n\n".join(paragraphs)
    if dropped and word_count(result) < floor:
        return drop_repeats(text, [], finished_only=finished_only)[0], 0  # keep the original sentences
    return result, dropped


# Words that make a page not gentle. A sentence with one is dropped from the page (or, when that
# would leave the page too short, the words are softened). Weapons (_WEAPON) count too.
_ROUGH = re.compile(
    r"\b(?:fight\w*|fought|battl\w*|attack\w*|wars?|hurt\w*|injur\w*|kill\w*|scar(?:y|ier|iest)|"
    r"terrif\w*|horrif\w*|rag(?:e|es|ed|ing)|punch\w*|smash\w*|blood\w*|dead|death|die[sd]?|dying|"
    r"evil|monsters?)\b", re.I)
_SOFTER = {
    "fight": "game", "fights": "games", "fighting": "playing", "fought": "played", "battle": "contest",
    "battles": "contests", "attack": "rush", "attacked": "rushed", "attacks": "rushes", "war": "contest",
    "wars": "contests", "hurt": "bumped", "scary": "surprising", "raging": "going on", "rage": "fuss",
    "punch": "tap", "smash": "bump", "evil": "grumpy", "monster": "creature", "monsters": "creatures",
}


def not_gentle(text: str) -> bool:
    return bool(_ROUGH.search(text) or _WEAPON.search(text))


def soften(text: str) -> str:
    """Rough words swapped for gentle ones, keeping a capital letter."""
    def swap(match: re.Match) -> str:
        word = match.group(0)
        gentle = _SOFTER.get(word.lower(), "toy" if _WEAPON.fullmatch(word) else "trouble")
        return gentle[0].upper() + gentle[1:] if word[0].isupper() else gentle
    return _WEAPON.sub(swap, _ROUGH.sub(swap, text))


def tidy_page(text: str, max_words: int, floor: int = 0) -> tuple[str, dict]:
    """After the repeat filter, the last checks on a page's text, in this order:
    1. gentle: sentences with a rough word are dropped while at least `floor` words remain; any
       still left get their rough words softened.
    2. length: whole sentences are kept up to `max_words` (the first sentence always stays).
    Paragraphs are kept. Returns (text, {"gentle": sentences changed, "trimmed_from": words or 0})."""
    blocks = [_SENTENCE_END.split(p.strip()) for p in text.split("\n\n") if p.strip()]
    rough = sum(1 for b in blocks for x in b if not_gentle(x))
    total = word_count(text)
    changed = 0
    if rough:
        for b in blocks:
            for i, sentence in enumerate(b):
                if sentence and not_gentle(sentence):
                    if total - word_count(sentence) >= floor:
                        total -= word_count(sentence)
                        b[i] = ""
                    else:
                        b[i] = soften(sentence)
                    changed += 1
    kept_blocks, used, trimmed_from = [], 0, 0
    for b in blocks:
        kept = []
        for sentence in (x for x in b if x.strip()):
            words = word_count(sentence)
            if used and used + words > max_words:
                trimmed_from = total
                break
            kept.append(sentence)
            used += words
        if kept:
            kept_blocks.append(" ".join(kept))
        if trimmed_from:
            break
    return "\n\n".join(kept_blocks), {"gentle": changed, "trimmed_from": trimmed_from}


def goal_sentence(goal_done: str) -> str:
    """The closing sentence added when the last page forgot to show the quest's goal achieved."""
    line = clean_line(goal_done, 160).rstrip(".!? ")
    return f"{line[0].upper()}{line[1:]}." if line else ""


def hero_notes(description: str) -> tuple[str, list[str]]:
    """Split the vision description into the kind of character and a few short features, so the
    storyteller gets notes to use rather than a sentence to copy.
    "a small blue fish with a yellow crown, one black eye, and a red tail"
      -> ("a small blue fish", ["a yellow crown", "one black eye", "a red tail"])
    The kind ends at the first "with", "wearing", "holding", "has" and so on; the features are
    the pieces after it, split at commas and at "and a/an/two..." ("yellow and blue" stays together), with leading
    words like "featuring" or "has" dropped. At most three features."""
    text = description.strip().rstrip(".")
    # A keyword ends the kind; a comma only when there is none ("a round, yellow creature with ..."
    # keeps its adjectives together).
    split = re.search(r"\s(?:with|wearing|holding|having|has|featuring|that|who)\s", text) or re.search(",", text)
    if not split:
        return text, []
    kind, rest = text[:split.start()], text[split.start():]
    features = []
    # Split at commas, and at "and" before a new item ("... hat and a long tail"), but not inside
    # one ("a yellow and blue patch").
    for piece in re.split(r",\s*(?:and\s+)?|\s+and\s+(?=(?:a|an|the|one|two|three|four|some|its)\s)", rest):
        piece = re.sub(r"^\s*(?:and|with|featuring|having|has|that has|who has)\s+", "", piece.strip(" ,."), flags=re.I)
        if len(piece) > 2:
            features.append(piece)
    return kind.strip(" ,.") or text, features[:3]


# --------------------------------------------------------------------------
# The story outline and each page's beat (a choose-your-own-adventure plot)
# --------------------------------------------------------------------------
# One call when a story starts plans its quest: a goal, a problem in the way, one or two friends and
# the happy ending to steer toward. Every page then gets a beat from a fixed ladder, filled in from the
# outline, so the story moves forward page by page and ends on time (Pre-reader 4 pages, else 6).

OUTLINE_SYSTEM = (
    "You plan gentle picture-book adventures for young children. Nobody is ever hurt, attacked or "
    "fought, and nothing is scary. Reply with one JSON object only, no other text."
)
OUTLINE_MAX_TOKENS = 220


def planned_pages(story: dict) -> int:
    outline = story.get("outline") or {}
    return int(outline.get("pages") or (4 if reading_level(story).key == "pre" else 6))


def build_outline_messages(story: dict) -> list[dict]:
    hero = story.get("hero_description")
    who = story.get("hero_who")
    pages = 4 if reading_level(story).key == "pre" else 6
    hero_line = ""
    if who:
        hero_line = f'The reader calls the hero "{who}": use that name and kind of character.\n'
    elif hero:
        hero_line = f"The hero is a character the reader drew: {hero_notes(hero)[0]}.\n"
    user = (
        f"The reader's idea for a {pages}-page picture-book story:\n\"{story['premise']}\"\n"
        + hero_line +
        "Plan the story's quest. Use these JSON keys in this order:\n"
        '"hero_name": the hero\'s short, friendly name, 1 to 3 words.\n'
        '"hero_kind": the kind of animal, creature or person the hero is: a plain noun of 1 or 2 words, '
        'never a nickname, a description or a colour.\n'
        '"goal": the quest: one special thing the hero wants to find, reach, make or help with, in at most 15 words.\n'
        '"obstacle": the gentle problem in the way, in at most 15 words: bad weather, a tricky path, a '
        'puzzle, something lost or stuck, or someone who needs help. Never a fight, a chase or anything '
        'scary, and nobody is mean or hurt.\n'
        '"friends": a list of 1 or 2 friends the hero meets, each with "name" (1 or 2 words) and "kind" '
        '(the kind of animal, creature or person: a plain noun of 1 or 2 words).\n'
        '"goal_done": the moment the goal is reached, in at most 15 words, naming anyone who is found or helped.\n'
        '"ending": the happy ending, in at most 20 words: the quest is completed.'
    )
    return [{"role": "system", "content": OUTLINE_SYSTEM}, {"role": "user", "content": user}]


def outline_schema() -> dict:
    friend = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "kind": {"type": "string"}},
        "required": ["name", "kind"], "additionalProperties": False,
    }
    props = {
        "hero_name": {"type": "string"},
        "hero_kind": {"type": "string"},
        "goal": {"type": "string"},
        "obstacle": {"type": "string"},
        "friends": {"type": "array", "items": friend, "minItems": 1, "maxItems": 2},
        "goal_done": {"type": "string"},
        "ending": {"type": "string"},
    }
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def _short(value, words: int) -> str:
    text = clean_line(str(value or ""), 200).strip(" .")
    return " ".join(text.split()[:words])


# Kinds that say nothing about what a character is ("Barky (buddy)").
_VAGUE_KINDS = frozenset("""buddy friend helper pal hero companion sidekick guy boy girl kid character
    good bestie mate partner""".split())
# Gentle problems for when the model's obstacle is not gentle (one is picked per story).
GENTLE_OBSTACLES = (
    "a gusty wind has blown the way out of sight",
    "a rainy puddle has turned the path into a slippery muddle",
    "a tricky puzzle locks the way forward",
    "the bridge is washed out by a big rain",
    "something important is stuck up high",
)


def _vague(kind: str) -> bool:
    words = [w for w in kind.lower().split() if w not in ("a", "an", "the")]
    return not words or any(w in _VAGUE_KINDS for w in words)


def kind_from_description(description: str) -> str:
    """'a cartoon dog-like creature with ...' -> 'dog'; otherwise the last word of its kind."""
    found = re.search(r"\b([a-z]+)-like\b", description, re.I)
    if found:
        return found.group(1).lower()
    return hero_notes(description)[0].split()[-1].lower() if description.strip() else ""


def parse_outline(raw: str, story: dict) -> dict:
    """The model's outline, trimmed and checked. Raises WriterFailed when the quest is missing.
    Not gentle: a rough obstacle is swapped for a gentle one and other rough words softened.
    Vague kinds ("buddy", "good boy") give way to the drawing's kind, or to plain "friend"."""
    data = _loads_lenient(raw)
    if not isinstance(data, dict):
        raise WriterFailed("no outline in the reply")
    goal, obstacle, ending = _short(data.get("goal"), 18), _short(data.get("obstacle"), 18), _short(data.get("ending"), 24)
    if not goal or not ending:
        raise WriterFailed("the outline has no goal or no ending")
    hero_name = _short(data.get("hero_name"), 3).strip("\"'") or ""
    friends = []
    for item in data.get("friends") or []:
        if not isinstance(item, dict):
            continue
        name, kind = _short(item.get("name"), 2).strip("\"'"), _short(item.get("kind"), 3)
        if name and name.lower() != hero_name.lower() and name.lower() not in {f["name"].lower() for f in friends}:
            friends.append({"name": name, "kind": "friend" if _vague(kind) else soften(kind)})
    hero_kind = _short(data.get("hero_kind"), 3)
    if _vague(hero_kind or ""):
        hero_kind = kind_from_description(story.get("hero_description") or "")
    seed = sum(map(ord, story.get("id") or story.get("premise") or ""))
    gentle = {"obstacle": False}
    if not obstacle or not_gentle(obstacle):
        gentle["obstacle"] = bool(obstacle)
        obstacle = GENTLE_OBSTACLES[seed % len(GENTLE_OBSTACLES)]
    goal_done = _short(data.get("goal_done"), 18) or f"{hero_name or 'the hero'} reaches the goal: {goal}"
    return {
        "hero_name": hero_name,
        "hero_kind": soften(hero_kind or ""),
        "goal": soften(goal),
        "obstacle": obstacle,
        "friends": friends[:2],
        "goal_done": soften(goal_done),
        "ending": soften(ending),
        "pages": 4 if reading_level(story).key == "pre" else 6,
        "softened": gentle["obstacle"] or any(not_gentle(x) for x in (goal, goal_done, ending)),
    }


def fallback_outline(story: dict) -> dict:
    """When the outline call fails: a plain quest, so the pages still have beats to follow."""
    return {
        "hero_name": "", "hero_kind": kind_from_description(story.get("hero_description") or ""),
        "goal": "a special adventure from the reader's idea",
        "obstacle": "something tricky is in the way", "friends": [],
        "goal_done": "the special adventure is complete",
        "ending": "the adventure ends happily and everyone celebrates together",
        "pages": 4 if reading_level(story).key == "pre" else 6, "fallback": True,
    }


def hero_name_kind(story: dict) -> tuple[str, str]:
    """(name, kind) for every page after page 1: "Crabby", "a crab". Never any looks."""
    outline = story.get("outline") or {}
    name = story.get("hero_name") or outline.get("hero_name") or "the hero"
    kind = outline.get("hero_kind") or ""
    if not kind:
        look = story.get("hero_description") or story.get("character_look") or ""
        kind = hero_notes(look)[0].split()[-1] if look else ""
    kind = kind.strip(" .")
    if kind and not re.match(r"(?i)(a|an|the)\s", kind):
        kind = f"{'an' if kind[0].lower() in 'aeiou' else 'a'} {kind}"
    return name, kind


_LADDER = {
    4: ("begin", "meet_clue", "obstacle_solve", "ending"),
    6: ("begin", "meet", "clue", "obstacle", "solve", "ending"),
}


def page_beat(story: dict, page_number: int, final: bool) -> str:
    """What must happen on this page to move the quest forward (the ending on the last page)."""
    outline = story.get("outline") or fallback_outline(story)
    name, _ = hero_name_kind(story)
    friends = outline.get("friends") or []
    first = f"{friends[0]['name']} ({friends[0]['kind']})" if friends else "a new friend"
    helper = friends[-1]["name"] if friends else "a new friend"
    ladder = _LADDER[4 if planned_pages(story) <= 4 else 6]
    stage = "ending" if final else ladder[min(page_number - 1, len(ladder) - 2)]
    goal, obstacle, ending = outline["goal"], outline["obstacle"], outline["ending"]
    return {
        "begin": f"Show {name} at home and how the quest begins: {goal}. End with {name} ready to set off.",
        "meet": f"{name} sets off on the quest and meets {first}, who wants to help.",
        "clue": f"{name} and {friends[0]['name'] if friends else 'a friend'} discover a clue that shows the way to the goal.",
        "meet_clue": f"{name} meets {first}, and together they find a clue that shows the way.",
        "obstacle": f"The problem gets in the way: {obstacle}. {name} must find a way past it.",
        "obstacle_solve": f"The problem gets in the way: {obstacle}. {name} and {helper} solve it together, kindly.",
        "solve": f"{name} solves the problem in a clever, kind way, with help from {helper}.",
        "ending": f"The happy ending: {ending}. The quest is complete and everyone celebrates.",
    }[stage]


def _outline_lines(story: dict, scrub=lambda text: text) -> str:
    outline = story.get("outline") or {}
    if not outline:
        return ""
    friends = ", ".join(f"{f['name']} ({f['kind']})" for f in outline.get("friends") or [])
    return (
        f"The quest: {scrub(outline['goal'])}\n"
        f"The problem in the way: {scrub(outline['obstacle'])}\n"
        + (f"Friends in this story: {friends}\n" if friends else "")
        + f"The happy ending to steer toward: {scrub(outline['ending'])}\n"
    )


# ---- the hero's looks stay on page 1 ----
# Feature words are what the hero looks like (colours, clothes, body parts...), taken from the drawing's
# description, "Who is this?" and the page-1 look. Pages 2+ never get them in the prompt, and a page
# sentence that names the hero together with two of them is dropped. Sizes and small words are not
# features ("Crabby saw a big wave" is fine).
_LOOK_SIZES = frozenset("big small large little tiny long short tall round wide huge giant".split())
_NOT_FEATURES = frozenset("""with and has have having wearing holding like creature character cartoon
    drawing drawn word words featuring its his her their one two three four some very cute friendly
    simple hand oval shape shaped""".split())


def hero_feature_words(story: dict) -> frozenset[str]:
    """Stems of the words that describe how the hero looks (never its name or kind)."""
    text = " ".join(story.get(k) or "" for k in ("hero_description", "character_look", "hero_who"))
    name, kind = hero_name_kind(story)
    own = {_stem(w) for w in re.findall(r"[A-Za-z]+", f"{name} {kind}")}
    found = set()
    for word in re.findall(r"[A-Za-z]+", text.lower()):
        if len(word) < 3 or word in _SMALL_WORDS or word in _LOOK_SIZES or word in _NOT_FEATURES:
            continue
        stem = _stem(word)
        if stem not in own:
            found.add(stem)
    return frozenset(found)


def hero_stems(story: dict) -> frozenset[str]:
    """Stems that name the hero: its name and the noun of its kind ("Crabby", "crab")."""
    name, kind = hero_name_kind(story)
    words = [w for w in re.findall(r"[A-Za-z]+", name) if len(w) > 2 and w.lower() not in _SMALL_WORDS]
    if kind:
        words.append(kind.split()[-1])
    return frozenset(_stem(w) for w in words if w.lower() != "hero")


COLOUR_WORDS = frozenset("""red orange yellow green blue purple pink brown black white gray grey gold golden
    silver violet teal turquoise""".split())


def scrub_looks(text: str, features: frozenset[str], kind_stem: str = "") -> str:
    """The text without descriptions of the hero: phrases set off by commas ("Mira, a small purple
    dragon, found...") or starting with "with" ("a crab with big claws") that hold a feature word,
    and colours right before its kind or a feature ("red crab" -> "crab"). Other words stay, so a
    summary still reads ("found a shell" keeps its shell)."""
    if not text or not features:
        return text
    has = lambda part: any(_stem(w) in features for w in re.findall(r"[A-Za-z]+", part))
    text = re.sub(r",\s*(?:a|an|the)\s[^,.;!?]*,\s*", lambda m: " " if has(m.group(0)) else m.group(0), text, flags=re.I)
    def with_phrase(m: re.Match) -> str:
        # "with big claws and a tie lives..." -> drop through the last feature word only
        words = m.group(1).split()
        last = max((i for i, w in enumerate(words) if _stem(w.strip(",")) in features), default=-1)
        return "" if last < 0 else " " + " ".join(words[last + 1:]) if words[last + 1:] else ""
    text = re.sub(r"\s*\bwith\s+((?:[A-Za-z'-]+\s*){1,6})", lambda m: m.group(0) if not has(m.group(1)) else with_phrase(m) + " ",
                  text, flags=re.I)
    targets = set(features) | ({kind_stem} if kind_stem else set())
    text = re.sub(r"\b([A-Za-z]+)\s+(?=([A-Za-z]+))",
                  lambda m: "" if m.group(1).lower() in COLOUR_WORDS and _stem(m.group(2)) in targets else m.group(0), text)
    text = re.sub(r"\s+([,.;!?])", r"\1", re.sub(r"\s{2,}", " ", text))
    return text.strip(" ,")


def drop_hero_looks(text: str, hero: frozenset[str], features: frozenset[str], floor: int = 0,
                    finished_only: bool = False) -> tuple[str, int]:
    """Pages 2+: drop every finished sentence that names the hero (name or kind) together with two or
    more of its feature words ("Crabby, the red crab with big claws, smiled"). Paragraphs are kept;
    an unfinished last sentence is left alone (or left out with `finished_only`, while streaming).
    Never takes the page under `floor` words. Returns (text, sentences dropped)."""
    if not text or not hero or not features:
        return text, 0
    dropped = 0
    paragraphs = []
    blocks = text.split("\n\n")
    for b, paragraph in enumerate(blocks):
        pieces = _SENTENCE_END.split(paragraph.strip())
        kept = []
        for i, piece in enumerate(pieces):
            finished = (i < len(pieces) - 1 or b < len(blocks) - 1
                        or bool(re.search(r"[.!?][\"'\u201d\u2019]?$", piece)))
            if not finished and finished_only:
                continue
            stems = {_stem(w) for w in re.findall(r"[A-Za-z']+", piece)}
            if finished and stems & hero and len(stems & features) >= 2:
                dropped += 1
            else:
                kept.append(piece)
        if any(k.strip() for k in kept):
            paragraphs.append(" ".join(kept))
    result = "\n\n".join(paragraphs)
    if dropped and word_count(result) < floor:
        return text, 0  # keep the page whole rather than too short
    return result, dropped


def _choices_key(next_beat: str) -> str:
    return _CHOICES_KEY + f"  The next page will be about this, so each choice is a different way toward it: {next_beat}\n"


def must_show(story: dict, event: str | None, final: bool) -> str:
    """The prompt's last line: the reader's choice, and on the last page the quest's goal reached."""
    goal = (story.get("outline") or {}).get("goal_done") if final else ""
    if event and goal:
        return f"{event}, and then {goal}"
    return event or goal or ""


def build_messages(story: dict, action: str | None, final: bool, event: str | None = None) -> list[dict]:
    """Build the chat messages for the next page. Sends a compact summary plus only the
    most recent page, never the whole history, so the 2048-token context is not exceeded."""
    level = reading_level(story)
    system = SYSTEM_PROMPT.format(reader=level.style, min_words=level.min_words, max_words=level.max_words)
    ready = [s for s in story["segments"] if s["status"] == "ready"]
    page_number = len(ready) + 1
    pages = planned_pages(story)
    beat = page_beat(story, page_number, final)
    outline = story.get("outline") or {}

    if not ready:
        hero = story.get("hero_description")
        who = story.get("hero_who")
        hero_line = ""
        if hero:
            kind, features = hero_notes(hero)
            hero_line = (
                (f'The reader says their hero is "{who}". That comes first: use the name and the kind of '
                 "character they gave, even where the notes below differ.\n" if who else "")
                + f"The hero is a character the reader drew: {kind}."
                + (f" Notes: {'; '.join(features)}." if features else "")
                + " Make it the main character. "
                + (f'Its name is {outline["hero_name"]}: write that in "hero_name" ' if outline.get("hero_name")
                   else 'Choose its name in "hero_name" ')
                + "and begin the page with that name. Tell what the hero does, in fresh story words; never list its looks in one "
                "long sentence. Mention one of the notes naturally, later in the page. The hero only holds "
                "or uses things that are in these notes.\n"
                + weapon_rule(f"{hero} {who or ''}") + "\n"
            )
        next_beat = page_beat(story, 2, final=pages <= 2)
        name_line = (f"The hero's name: {outline['hero_name']}\n"
                     if outline.get("hero_name") and not hero else "")
        user = (
            f"The reader's idea for the story:\n\"{story['premise']}\"\n\n"
            + hero_line + name_line + _outline_lines(story) +
            f"The story has {pages} pages.\n\n"
            "Write page 1 only. Introduce the main character and the place, show the quest beginning, "
            "and end on a moment where the reader decides how to start. Do not resolve anything.\n"
            f"This page's beat: {beat}\n\n"
            "Use these JSON keys in this order:\n"
            '"beat": (already filled in: this page\'s beat.)\n'
            '"title": a title for the story, at most 6 words.\n'
            + _PLAN_KEYS
            + ('"hero_name": the hero\'s short, friendly name, 1 to 3 words.\n' if hero else "") +
            '"segment": the text of page 1, written from the plan above: '
            f'{level.min_words} to {level.max_words} words in one or two short paragraphs.\n'
            '"character_look": what the main character looks like in at most 20 words '
            "(kind of creature or person, colours, size, clothing).\n"
            '"world": the setting in at most 20 words.\n'
            + _COMMON_KEYS + _choices_key(next_beat) + _SUMMARY_KEY
        )
    else:
        # Only a short summary: the previous page's text would drown out the reader's choice (which a
        # small model then ignores), and its last sentence tempted it to retell how that page ended.
        name, kind = hero_name_kind(story)
        features = hero_feature_words(story)
        kind_stem = _stem(kind.split()[-1]) if kind else ""
        scrub = lambda text: scrub_looks(text, features, kind_stem)
        outline_ok = bool(outline) and not outline.get("fallback")
        if final:
            task = (
                f"Write page {page_number}, the LAST page. Wrap up the quest with a happy, satisfying "
                'ending that fits everything so far. "choices" must be an empty list [].'
            )
            choices_key = '"choices": [].\n'
        else:
            near_end = page_number >= pages - 1
            task = (
                f"Write page {page_number}{f' of {pages}' if page_number <= pages else ''} only. "
                "Show what happens because of the reader's "
                "decision and move the quest forward with this page's beat, then end on a moment where "
                "the reader decides what happens next. Do not end the story."
                + (" The story ends on the next page, so lead toward the happy ending." if near_end else "")
            )
            choices_key = _choices_key(page_beat(story, page_number + 1, final=page_number + 1 >= pages))
        decision = action or "Continue the adventure in the most natural way."
        user = (
            # Only the hero's name and kind after page 1, never its looks, which the model retold: no
            # "Who is this?" words, no idea (the outline holds the quest), and the title, summary and
            # outline without the hero's feature words.
            f"Story title: {scrub(story.get('title') or '') or 'Untitled'}\n"
            + ("" if outline_ok else f"The reader's original idea: \"{scrub(story['premise'])}\"\n")
            + f"The hero: {name}{', ' + kind if kind else ''}. Never describe what {name} looks like "
            "(no colours, size, body or clothes): the reader already knows.\n"
            + _outline_lines(story, scrub)
            + ("The hero only holds or uses things it already has, or things it finds in the story.\n"
               + weapon_rule(f"{story['hero_description']} {story.get('hero_who') or ''}")
               if story.get("hero_description") else "")
            +
            f"Setting: {story.get('world') or 'not described yet'}\n"
            f"Previous page's place: {ready[-1].get('setting') or 'not described'}"
            + (" (move somewhere new when the reader's choice goes somewhere)\n" if event
               else " (stay there or move somewhere new, as the story needs)\n")
            +
            f"Story so far: {scrub(short_summary(story.get('summary'))) or 'the story has just begun'}\n\n"
            f"The reader decided: \"{decision}\"\n\n"
            f"{task}\n"
            f"This page's beat: {beat}\n\n"
            "Use these JSON keys in this order:\n"
            + (f'"event": "{event}" (already filled in: the reader\'s choice. The setting, details, '
               "action and the whole page follow from it.)\n"
               f'The "setting" is where this event happens: if it goes somewhere new, that new place. '
               f'The "action" is what the hero does in this event.\n' if event else "")
            + '"beat": (already filled in: this page\'s beat. The page shows it happening.)\n'
            + _PLAN_KEYS +
            f'"segment": the text of page {page_number}, written from the plan above: '
            f'{level.min_words} to {level.max_words} words in one or two short paragraphs.\n'
            + _COMMON_KEYS + choices_key + _SUMMARY_KEY
            # Last, because a small model follows the last instruction best. On the last page the
            # quest's goal is reached there too ("the lost puppy is found and named").
            + (f"\n\nThis page must show: {must_show(story, event, final)}" if event or final else "")
        )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def response_schema(first: bool, final: bool, drawn_hero: bool = False, event: str | None = None,
                    beat: str | None = None) -> dict:
    """JSON schema handed to llama-server so its grammar forces well-formed output.
    `event` (the reader's choice) and `beat` (this page's step in the quest) are constant first
    properties: the grammar writes them into the plan before the model plans the setting and action,
    so the plan starts from the choice and the plot."""
    props: dict = {}
    if event:
        props["event"] = {"type": "string", "const": event}
    if beat:
        props["beat"] = {"type": "string", "const": beat}
    if first:
        props["title"] = {"type": "string"}
    props["setting"] = {"type": "string"}
    props["visual_details"] = {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 3}
    props["props"] = {"type": "array", "items": {"type": "string"}, "minItems": 0, "maxItems": 2}
    props["action"] = {"type": "string", "enum": list(HERO_ACTIONS)}
    if first and drawn_hero:
        props["hero_name"] = {"type": "string"}
    props["segment"] = {"type": "string"}
    if first:
        props["character_look"] = {"type": "string"}
        props["world"] = {"type": "string"}
    props["visual_prompt"] = {"type": "string"}
    props["choices"] = {"type": "array", "items": {"type": "string"},
                        "minItems": 0 if final else 2, "maxItems": 0 if final else 3}
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
    finish_reason: str | None = None  # "length" means the token limit cut the reply off
    max_tokens: int | None = None  # LLM_MAX_TOKENS unless given
    schema_name: str = "story_page"
    _conn: http.client.HTTPConnection | None = field(default=None, repr=False)
    _cancelled: threading.Event = field(default_factory=threading.Event, repr=False)

    @property
    def was_cancelled(self) -> bool:
        return self._cancelled.is_set()

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
            "max_tokens": self.max_tokens or settings.llm_max_tokens,
            "stream": True,
            "cache_prompt": True,
        }
        if style == "json_schema":
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": self.schema_name, "strict": True, "schema": self.schema},
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
                if choice.get("finish_reason"):
                    self.finish_reason = choice["finish_reason"]
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


# Small things the hero holds or uses, shown as a big emoji beside the drawing. Only gentle,
# everyday objects: a prop with no match here (or a weapon) is simply not shown.
PROP_EMOJI = {
    "strawberry": "🍓", "apple": "🍎", "banana": "🍌", "cherry": "🍒", "grape": "🍇", "orange": "🍊",
    "lemon": "🍋", "pear": "🍐", "peach": "🍑", "watermelon": "🍉", "carrot": "🥕", "mushroom": "🍄",
    "bread": "🍞", "cake": "🍰", "cupcake": "🧁", "cookie": "🍪", "biscuit": "🍪", "pie": "🥧", "honey": "🍯",
    "candy": "🍬", "sweet": "🍬", "lollipop": "🍭", "chocolate": "🍫", "ice cream": "🍦", "pizza": "🍕",
    "sandwich": "🥪", "cheese": "🧀", "egg": "🥚", "milk": "🥛", "tea": "🍵", "cup": "☕", "teacup": "☕",
    "acorn": "🌰", "nut": "🌰", "flower": "🌸", "rose": "🌹", "sunflower": "🌻", "tulip": "🌷", "daisy": "🌼",
    "leaf": "🍃", "clover": "🍀", "seashell": "🐚", "shell": "🐚", "feather": "🪶", "star": "⭐",
    "crystal": "💎", "gem": "💎", "jewel": "💎", "diamond": "💎", "pearl": "🦪", "rock": "🪨", "stone": "🪨",
    "pebble": "🪨", "key": "🔑", "map": "🗺️", "book": "📖", "storybook": "📖", "letter": "✉️", "note": "📝",
    "scroll": "📜", "pencil": "✏️", "crayon": "🖍️", "paintbrush": "🖌️", "brush": "🖌️", "paint": "🎨",
    "ball": "⚽", "balloon": "🎈", "kite": "🪁", "teddy": "🧸", "teddy bear": "🧸", "toy": "🧸", "gift": "🎁",
    "present": "🎁", "box": "📦", "basket": "🧺", "bucket": "🪣", "umbrella": "☂️", "lantern": "🏮",
    "lamp": "🪔", "candle": "🕯️", "torch": "🔦", "flashlight": "🔦", "magnifying glass": "🔍",
    "telescope": "🔭", "compass": "🧭", "bell": "🔔", "drum": "🥁", "guitar": "🎸", "violin": "🎻",
    "trumpet": "🎺", "flute": "🪈", "wand": "🪄", "magic wand": "🪄", "rope": "🪢", "fishing rod": "🎣", "shovel": "🪏",
    "spade": "🪏", "watering can": "🪣", "magnet": "🧲", "clock": "⏰", "camera": "📷", "phone": "📱",
    "jar": "🫙", "potion": "🧪", "trophy": "🏆", "coin": "🪙", "coins": "🪙",
    "snowball": "❄️", "snowflake": "❄️", "bubble": "🫧", "heart": "❤️", "rainbow": "🌈",
}


def _prop_key(word: str) -> str:
    """Lower case, no punctuation, singular: "Strawberries" -> "strawberry", "boxes" -> "box"."""
    w = re.sub(r"[^a-z]", "", word.lower())
    if w in PROP_EMOJI:
        return w
    for suffix, repl in (("ies", "y"), ("ves", "f"), ("es", ""), ("s", "")):
        if w.endswith(suffix) and w[: -len(suffix)] + repl in PROP_EMOJI:
            return w[: -len(suffix)] + repl
    return w


def prop_emoji(props: list[str]) -> list[dict]:
    """The emoji for each prop, matched on its last words first ("a ripe strawberry" -> 🍓, "ice
    cream cone" -> 🍦). Props with no match are skipped."""
    out = []
    for prop in props[:2]:
        words = [_prop_key(w) for w in prop.split()]
        emoji = None
        for i in range(len(words) - 1, -1, -1):
            pair = " ".join(words[i - 1 : i + 1]) if i else ""
            emoji = PROP_EMOJI.get(pair) or PROP_EMOJI.get(words[i])
            if emoji:
                break
        if emoji:
            out.append({"name": prop, "emoji": emoji})
    return out


# "...held the map", "unfolded her old map", "picked up a shiny shell": the hero holding or using
# something. The object is the first known prop word within the next few words.
_HOLDING = re.compile(
    r"\b(?:holds?|holding|held|carr(?:y|ies|ied|ying)|clutch(?:es|ed|ing)?|grip(?:s|ped|ping)?|"
    r"grab(?:s|bed|bing)?|hug(?:s|ged|ging)?|picks? up|picked up|picking up|took out|takes? out|"
    r"pulled out|pulls? out|unfold(?:s|ed|ing)?|unroll(?:s|ed|ing)?|open(?:s|ed|ing)?|reads?|reading|"
    r"use(?:s|d)?|using|wav(?:es|ed|ing)|lift(?:s|ed|ing)?|rais(?:es|ed|ing)|toss(?:es|ed|ing)?|"
    r"thr(?:ew|ows?|owing)|kick(?:s|ed|ing)?|ate|eats?|eating|munch(?:es|ed|ing)?|bl(?:ew|ows?|owing)|"
    r"play(?:s|ed|ing)|r(?:ang|ings?|inging)|shook|shak(?:es|ing)|point(?:s|ed|ing)|check(?:s|ed|ing)?|"
    r"studi(?:es|ed)|studying|look(?:s|ed|ing)? at|peer(?:s|ed|ing)? at|turn(?:s|ed|ing)?)"
    r"\b", re.I)


def held_in_text(text: str) -> list[str]:
    """The props the page says the hero holds or uses, in order, as their map words ("map")."""
    found: list[str] = []
    for match in _HOLDING.finditer(text or ""):
        after = re.match(r"[a-z' -]*", text[match.end():], re.I).group(0)  # up to the next punctuation
        words = [_prop_key(w) for w in after.split()[:5]]
        for i, word in enumerate(words):
            pair = f"{word} {words[i + 1]}" if i + 1 < len(words) else ""
            name = pair if pair in PROP_EMOJI else word if word in PROP_EMOJI else ""
            if name:
                if name not in found:
                    found.append(name)
                break
    return found


def choose_props(planned: list[str], text: str = "", skip=lambda prop: False) -> list[dict]:
    """The props to show beside the hero: first what the page text says the hero holds or uses,
    then the plan's props the text mentions, then the rest of the plan's. Props that `skip` (worn,
    or part of the hero's own description) are never shown. At most 2, each with an emoji."""
    in_text = {_prop_key(w) for w in (text or "").split()}
    mentioned = [p for p in planned if p.split() and _prop_key(p.split()[-1]) in in_text]
    ordered = held_in_text(text) + mentioned + [p for p in planned if p not in mentioned]
    out: list[dict] = []
    for prop in ordered:
        if skip(prop):
            continue
        for item in prop_emoji([prop]):
            if item["emoji"] not in {x["emoji"] for x in out}:
                out.append(item)
        if len(out) == 2:
            break
    return out


def clean_props(value) -> list[str]:
    """Up to 2 short props, never a weapon (those are left out, not swapped for something else)."""
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if isinstance(item, str):
            item = clean_line(item, 40).rstrip(".")
            if len(item) > 1 and not _WEAPON.search(item) and item.lower() not in {x.lower() for x in out}:
                out.append(item)
    return out[:2]


def clean_action(value) -> str:
    value = value.strip().lower() if isinstance(value, str) else ""
    return value if value in HERO_ACTIONS else HERO_ACTIONS[0]


def extract_plan(raw: str) -> dict | None:
    """The page plan, once all of it has streamed in (it comes before the page text):
    {"setting", "visual_details", "props", "action"}. None until then."""
    setting, setting_done = extract_string_field(raw, "setting")
    details = _string_list(raw, "visual_details")
    props = _string_list(raw, "props")
    action, action_done = extract_string_field(raw, "action")
    if not (setting_done and setting and details is not None and action_done):
        return None
    return {
        "setting": clean_line(setting, 180),
        "visual_details": clean_details(details),
        "props": clean_props(props or []),
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


SMALL_ACTIONS = frozenset("""
    look looks peek peer gaze stare watch listen wait stay rest sit smile think wonder pause
""".split())

FALLBACK_CHOICES = (
    "Look more closely at what is nearby",
    "Call out and see who answers",
    "Take the bravest path forward",
)


def word_count(text: str) -> int:
    return len(text.split())


UNUSABLE_WORDS = 15


def minimum_words() -> int:
    """Shortest page we keep (counted before the repeat filter, which never takes a page below it).
    Lenient on purpose: a 37-word page is a fine children's page, and much better than writing it
    again while the reader watches."""
    return UNUSABLE_WORDS


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
    props: list[str] = field(default_factory=list)
    hero_action: str = HERO_ACTIONS[0]
    used_fallback_choices: bool = False
    repaired: bool = False
    repeats_dropped: int = 0  # sentences left out because an earlier page already had them


def parse_page(raw: str, first: bool, final: bool, earlier_pages: list[str] | None = None) -> Page:
    """Validate the model's reply. Raises WriterFailed when no usable page text exists.
    Sentences repeated from `earlier_pages` are dropped before the length is checked."""
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
        props = _string_list(raw, "props")
        if props:
            data["props"] = props
        block = re.search(r'"choices"\s*:\s*\[(.*?)(\]|$)', raw, re.S)
        if block:
            data["choices"] = re.findall(r'"((?:[^"\\]|\\.)*)"', block.group(1))

    segment = data.get("segment") or data.get("story") or data.get("text") or ""
    if not isinstance(segment, str):
        segment = ""
    segment = clean_text(segment, 1800)
    if word_count(segment) < minimum_words():
        raise WriterFailed(f"page text too short ({word_count(segment)} words)")
    segment, dropped = drop_repeats(segment, earlier_pages or [], floor=minimum_words())
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
            choice = choice[:1].upper() + choice[1:]
            key = choice.lower()
            if len(choice) >= 3 and key not in seen:
                seen.add(key)
                choices.append(choice)
            if len(choices) == 4:
                break
    used_fallback = False
    # Choices are forks in the quest; small actions ("Look closer", "Wait and listen") are dropped
    # while at least two real ones remain.
    forks = [c for c in choices if c.split()[0].lower() not in SMALL_ACTIONS]
    if len(forks) >= 2:
        choices = forks
    choices = choices[:3]
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
        props=clean_props(data.get("props")),
        hero_action=clean_action(data.get("action")),
        used_fallback_choices=used_fallback,
        repaired=repaired,
        repeats_dropped=dropped,
    )


# --------------------------------------------------------------------------
# Tap-a-word definitions
# --------------------------------------------------------------------------
# Once a page is finished (and llama-server is idle), Qwen picks the 3-4 hardest words on it and
# defines each in one sentence for the story's reading level. Only words that really are on the
# page are kept; the browser underlines them.

DEFINE_SYSTEM = "You help children learn new words. Reply with one JSON object only, no other text."
DEFINE_MAX_TOKENS = 260


def build_definition_messages(text: str, level: ReadingLevel) -> list[dict]:
    # No example words or definitions: a small model copies examples.
    user = (
        f'Here is a page from a story for children:\n"""\n{text}\n"""\n\n'
        f"The reader is {level.reader}. Pick the 3 or 4 words on this page that this reader is "
        "least likely to know. Copy each word exactly as it is written on the page. Never pick a "
        "name or a very easy word.\n"
        f"For each word write a definition: one short sentence that {level.reader} understands, "
        "saying what the word means here.\n"
        'Use the JSON key "words": a list of objects with the keys "word" and "definition".'
    )
    return [{"role": "system", "content": DEFINE_SYSTEM}, {"role": "user", "content": user}]


def definition_schema() -> dict:
    item = {
        "type": "object",
        "properties": {"word": {"type": "string"}, "definition": {"type": "string"}},
        "required": ["word", "definition"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {"words": {"type": "array", "items": item, "minItems": 3, "maxItems": 4}},
        "required": ["words"],
        "additionalProperties": False,
    }


def _word_pattern(word: str) -> re.Pattern:
    return re.compile(r"(?<![\w'’-])" + re.escape(word) + r"(?![\w'’-])", re.I)


def _starts_sentence(text: str, at: int) -> bool:
    before = text[:at].rstrip(" \t\"'“‘(")
    return not before or before[-1] in ".!?\n…"


def parse_definitions(raw: str, text: str, names: frozenset[str] = frozenset()) -> list[dict]:
    """The model's words that are really on the page, each as the page writes it, with a one-sentence
    definition. Names (capitalised everywhere on the page, or one of `names`) and small words are
    left out. Raises WriterFailed when the reply is not usable at all."""
    data = _loads_lenient(raw)
    items = data.get("words") if isinstance(data, dict) else None
    if not isinstance(items, list):
        raise WriterFailed("no word list in the reply")
    found: list[dict] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        word = clean_line(str(item.get("word") or ""), 40).strip(" .,!?;:\"'“”‘’()")
        definition = clean_line(str(item.get("definition") or ""), 240)
        if not word or not definition or len(word.split()) > 3 or len(word) < 3 or not re.search(r"[A-Za-z]", word):
            continue
        key = word.lower()
        if key in seen or key in _SMALL_WORDS or key in names or definition.lower().strip(" .") == key:
            continue
        matches = list(_word_pattern(word).finditer(text))
        if not matches:
            continue  # not on the page (or a different form of the word)
        if all(m.group(0)[0].isupper() for m in matches) and (
                word[0].isupper() or any(not _starts_sentence(text, m.start()) for m in matches)):
            continue  # a name: capitalised everywhere on the page
        first = re.match(r"(.+?[.!?])(\s|$)", definition)
        definition = first.group(1) if first else definition.rstrip(",;:") + "."
        seen.add(key)
        found.append({"word": matches[0].group(0), "definition": definition[0].upper() + definition[1:]})
    return found[:4]
