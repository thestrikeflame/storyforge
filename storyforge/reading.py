"""Measurable reading levels: which words on a page get a tap-a-word definition (chosen here by how
rare they are, not by the model), how hard a page reads, and short sentences for pre-readers.

Word rarity is the Zipf frequency from wordfreq (about 7 for "the", 5 for "happy", 3 for
"shimmering"; 0 for a word it has never seen). A word is as easy as its most familiar form, so
"shimmering" is scored as its base form "shimmer" too, found by simple suffix stripping."""
from __future__ import annotations

import re
from dataclasses import dataclass

import textstat
from wordfreq import zipf_frequency


@dataclass(frozen=True)
class LevelTarget:
    max_words: int           # how many words get a definition
    zipf_min: float          # a word qualifies when min <= zipf < max
    zipf_max: float
    grade: str               # the Flesch-Kincaid grade the page aims for (in the prompt and the log)
    grade_range: tuple[float, float]  # what counts as on target in the log
    sentence_words: int      # longest sentence the prompt asks for


TARGETS = {
    "pre": LevelTarget(2, 3.5, 4.5, "about 1", (-99.0, 2.0), 8),  # any grade up to 2: easier is fine
    "early": LevelTarget(3, 0.0, 4.0, "about 2", (1.0, 3.0), 12),
    "confident": LevelTarget(4, 0.0, 3.3, "3 to 5", (3.0, 5.0), 18),
}
PRE_SPLIT_OVER = 10  # pre-reader sentences longer than this are split in Python

MIN_LETTERS = 4
STOP_WORDS = frozenset("""
    about above after again against also always another anyone anything around away back because
    been before behind being below beside between both came come comes could didn't does doesn't
    doing done don't down during each else even ever every everyone everything from gave give gives
    going gone good great have having he'd he'll here here's herself himself into isn't it'll it's
    itself just know knew like little long looked looking made make many maybe might more most much
    must myself never next nothing once only onto other ours over own really said same says shall
    she'd she'll should some something soon still such sure take than that that's their theirs them
    themselves then there there's these they they'd they'll they're thing things this those though
    through told took toward towards under until upon very want wanted wants wasn't well went were
    weren't what what's when where which while who whole whom whose will with within without won't
    would wouldn't yeah your yours yourself
""".split())

_WORD = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|(?<=[.!?][\"'”’])\s+")


def _zipf(word: str) -> float:
    return zipf_frequency(word, "en")


def base_forms(word: str) -> list[str]:
    """Possible base forms by simple suffix stripping: shimmering -> shimmer, bunnies -> bunny,
    running -> run, glided -> glide, happily -> happy."""
    w = word.lower()
    forms: list[str] = []

    def add(stem: str) -> None:
        if len(stem) >= 3 and stem not in forms and stem != w:
            forms.append(stem)

    for suffix in ("ing", "ed", "er", "est"):
        if w.endswith(suffix) and len(w) - len(suffix) >= 3:
            stem = w[: -len(suffix)]
            if stem.endswith("i"):
                add(stem[:-1] + "y")  # tried -> try, happier -> happy
            add(stem)
            add(stem + "e")  # glided -> glide
            if len(stem) >= 4 and stem[-1] == stem[-2] and stem[-1] not in "aeiouls":
                add(stem[:-1])  # running -> run, bigger -> big
    if w.endswith("ily") and len(w) > 5:
        add(w[:-3] + "y")  # happily -> happy
    if w.endswith("ly") and len(w) > 5:
        add(w[:-2])  # softly -> soft
    if w.endswith("ies") and len(w) > 5:
        add(w[:-3] + "y")
    if w.endswith("es") and len(w) > 4:
        add(w[:-2])
    if w.endswith("s") and not w.endswith("ss") and len(w) > 4:
        add(w[:-1])
    return forms


def word_zipf(word: str) -> float:
    """How common a word is, as its most familiar form (the word itself or a base form that wordfreq knows)."""
    w = word.lower().replace("’", "'")
    if w.endswith("'s"):
        w = w[:-2]
    return max([_zipf(w)] + [_zipf(f) for f in base_forms(w)])


def lemma(word: str) -> str:
    """The most common of the word and its base forms, used so "shimmer" and "shimmering" count as one word."""
    w = word.lower()
    best, score = w, _zipf(w)
    for form in base_forms(w):
        z = _zipf(form)
        if z > score:
            best, score = form, z
    return best


def syllables(word: str) -> int:
    return max(1, int(textstat.syllable_count(word)))


def _starts_sentence(text: str, at: int) -> bool:
    before = text[:at].rstrip(" \t\"'“‘(")
    return not before or before[-1] in ".!?\n…:"


def _names_on_page(text: str) -> set[str]:
    """Words written with a capital in the middle of a sentence (names), lowercased."""
    return {m.group(0).lower() for m in _WORD.finditer(text)
            if m.group(0)[0].isupper() and not _starts_sentence(text, m.start())}


def hardness(zipf: float, syl: int, letters: int) -> float:
    """Higher is harder: rarity first, then syllables, then length."""
    return (8.0 - zipf) + 0.35 * syl + 0.05 * letters


def pick_words(text: str, level: str, names: set[str] | frozenset[str] = frozenset()) -> list[dict]:
    """The hardest words on the page that qualify for this reading level, at most the level's number,
    each as first written on the page. Names (the hero's, or capitalised mid-sentence), stop words,
    words under 4 letters and words wordfreq has never seen are left out.
    Returns [{"word", "zipf", "syllables", "score"}] hardest first."""
    target = TARGETS.get(level, TARGETS["early"])
    names = {n.lower() for n in names} | _names_on_page(text)
    lower_seen = {m.group(0).lower() for m in _WORD.finditer(text) if m.group(0)[0].islower()}
    seen: set[str] = set()
    found: list[dict] = []
    for m in _WORD.finditer(text):
        word = m.group(0)
        key = word.lower().replace("’", "'")
        if key.endswith("'s"):
            key = key[:-2]
        if "'" in key or len(key) < MIN_LETTERS or key in STOP_WORDS or key in names:
            continue
        # Capitalised only at a sentence start and never written in lower case: a name we cannot rule out
        # unless wordfreq knows it as an ordinary word.
        base = lemma(key)
        if base in seen or key in seen:
            continue
        seen.update({key, base})
        zipf = word_zipf(key)
        if zipf <= 0:
            continue  # a made-up word or a name: nothing to define
        if word[0].isupper() and key not in lower_seen and _zipf(key) < 2.0:
            continue
        if not target.zipf_min <= zipf < target.zipf_max:
            continue
        syl = syllables(key)
        found.append({"word": word if not word.endswith(("'s", "’s")) else word[:-2],
                      "zipf": round(zipf, 2), "syllables": syl,
                      "score": round(hardness(zipf, syl, len(key)), 2)})
    found.sort(key=lambda w: -w["score"])
    return found[: target.max_words]


# ---------------------------------------------------------------- measuring a page

def sentences(text: str) -> list[str]:
    return [s for p in text.split("\n\n") for s in _SENTENCE_END.split(p.strip()) if s.strip()]


def _count(sentence: str) -> int:
    return len(_WORD.findall(sentence))


def measure(text: str, level: str) -> dict:
    """Flesch-Kincaid grade and sentence lengths, against the level's targets."""
    target = TARGETS.get(level, TARGETS["early"])
    lengths = [_count(s) for s in sentences(text)] or [0]
    grade = round(float(textstat.flesch_kincaid_grade(text)), 1) if text.strip() else 0.0
    low, high = target.grade_range
    return {
        "grade": grade,
        "grade_target": target.grade,
        "avg_sentence_words": round(sum(lengths) / len(lengths), 1),
        "longest_sentence_words": max(lengths),
        "sentence_limit": target.sentence_words,
        "sentences": len(lengths),
        "long_sentences": sum(1 for n in lengths if n > target.sentence_words),
        "on_target": low - 0.5 <= grade <= high + 0.5 and max(lengths) <= target.sentence_words,
    }


# ---------------------------------------------------------------- short sentences for pre-readers

# Where a long sentence may be split, best first. A plain "and" or a comma is split only before a
# pronoun ("..., and she ran"), so "Mia and Leo ran home" and lists ("a hat, a kite, and a ball")
# stay whole.
_PRONOUN = r"(?:I|he|she|they|it|we|you|then|soon|now|suddenly)\b"
_SUBJECT = r"(?:I|he|she|they|it|we|you|the|a|an|his|her|their|its|my|our|then|soon|now|[A-Z][a-z]+)\b"
_SPLITS = (
    (re.compile(r",\s+and\s+(?=" + _PRONOUN + ")", re.I), False),
    (re.compile(r",?\s+but\s+(?=" + _SUBJECT + ")"), False),
    (re.compile(r",\s+and\s+(?=" + _SUBJECT + ")"), True),  # only when the first half has no comma (a list)
    (re.compile(r"\s+and\s+(?=" + _PRONOUN + ")", re.I), False),
    (re.compile(r",\s+(?=" + _PRONOUN + ")", re.I), False),
)
# "In the morning, the sun rose" is not split at its comma: the first half would not be a sentence.
_OPENER = re.compile(r"(?:in|on|at|after|before|when|as|with|under|over|by|through|during|while|if|because|"
                     r"since|once|until|from|across|along|inside|outside|near|behind|beside|past)\b", re.I)


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:]


def _split_one(sentence: str, limit: int) -> list[str]:
    if _count(sentence) <= limit or re.search(r"[\"“”]", sentence):
        return [sentence]
    end = sentence[-1] if sentence[-1] in ".!?" else "."
    body = sentence.rstrip(".!? ")
    middle = len(body) / 2
    for pattern, no_list in _SPLITS:
        cuts = [m for m in pattern.finditer(body)
                if _count(body[: m.start()]) >= 3 and _count(body[m.end():]) >= 3
                and not (no_list and "," in body[: m.start()])
                and not (m.group(0).strip() == "," and _OPENER.match(body))]
        if not cuts:
            continue
        cut = min(cuts, key=lambda m: abs(m.start() - middle))
        joiner = cut.group(0).strip(" ,").lower()
        left = body[: cut.start()].rstrip(" ,;")
        right = body[cut.end():].lstrip()
        if joiner == "but":
            right = "But " + right  # "but" carries meaning; "and" does not
        first = left + "."
        second = _cap(right) + end
        return _split_one(first, limit) + _split_one(second, limit)
    return [sentence]


def split_long_sentences(text: str, limit: int = PRE_SPLIT_OVER) -> str:
    """Sentences over `limit` words split at ", and", " and ", " but " or a comma (before a new
    subject, so both halves are sentences). Paragraphs and short sentences are unchanged."""
    out = []
    for paragraph in text.split("\n\n"):
        if not paragraph.strip():
            continue
        pieces = _SENTENCE_END.split(paragraph.strip())
        out.append(" ".join(part for s in pieces for part in _split_one(s, limit)))
    return "\n\n".join(out)
