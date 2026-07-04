"""Normalization pass for pilot transmissions.

Turns spoken/ASR-transcribed aviation phraseology into a token stream where
number groups ("one zero one three") and NATO-alphabet letter groups
("victor tango alpha bravo charlie") are collapsed into single alnum tokens
("1013", "vtabc"), so the item grammar can work on clean values.

This table is intentionally small and versioned (NORMALIZER_VERSION) per the
project brief: grow it only from disputed golden-test cases, never guess.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

NORMALIZER_VERSION = "1.0.0"

NATO_ALPHABET = {
    "alpha": "a", "alfa": "a", "bravo": "b", "charlie": "c", "delta": "d",
    "echo": "e", "foxtrot": "f", "golf": "g", "hotel": "h", "india": "i",
    "juliett": "j", "juliet": "j", "kilo": "k", "lima": "l", "mike": "m",
    "november": "n", "oscar": "o", "papa": "p", "quebec": "q", "romeo": "r",
    "sierra": "s", "tango": "t", "uniform": "u", "victor": "v",
    "whiskey": "w", "xray": "x", "x-ray": "x", "yankee": "y", "zulu": "z",
}

NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    # ICAO-mandated pronunciations that a fixed-vocab ASR often emits verbatim
    "niner": "9", "tree": "3", "fife": "5",
}

DECIMAL_WORDS = {"decimal", "point"}

# filler/hesitation words are transparent to letter- and digit-run grouping:
# "victor, uh, tango, alpha bravo charlie" must still merge into one
# callsign token, not fracture into "v" + "tabc" (which then fails
# substring-based callsign matching downstream).
DISFLUENCY_WORDS = {"uh", "um", "er", "erm", "ah", "uhh", "umm"}

# Fixed, documented table of common ASR misrecognitions for ATC vocabulary.
# Grow this only from disputed golden-test cases (see tests/golden/).
SUBSTITUTIONS = {
    "decend": "descend",
    "decending": "descending",
    "hedding": "heading",
    "clim": "climb",
    "climbb": "climb",
    "readyback": "readback",
    "wilco": "wilco",
    "squack": "squawk",
    "squwak": "squawk",
}

_GROUPABLE = set(NATO_ALPHABET) | set(NUMBER_WORDS) | DECIMAL_WORDS
# apostrophes are preserved so contractions ("don't", "isn't") survive as one
# token instead of splitting into "don" + "t", which matches nothing.
_PUNCT_RE = re.compile(r"[^\w\s.\-']")
# text-mode agents sometimes type shorthand with no space ("FL80", "QNH1013")
# rather than speaking it word-by-word; split it before tokenizing.
_FUSED_SHORTHAND_RE = re.compile(r"\b(fl|qnh|hdg|rwy|squawk)(\d+)\b", re.IGNORECASE)


@dataclass
class NormalizedTransmission:
    raw: str
    text: str
    tokens: List[str]


def _tokenize(raw: str) -> List[str]:
    cleaned = _FUSED_SHORTHAND_RE.sub(r"\1 \2", raw.lower())
    # a real reply can render a decimal with stray whitespace after the
    # point ("switching to approach 127. 1." for "127.1") -- caught on a
    # live call. Collapse "digit . whitespace digit" back into a proper
    # decimal before the stricter adjacency rule below ever sees it.
    cleaned = re.sub(r"(\d)\.\s+(\d)", r"\1.\2", cleaned)
    # a period only survives as a decimal point when digits sit on both
    # sides (e.g. "119.7"); every other period -- sentence-final punctuation
    # on a real model's natural reply chief among them ("...QNH one zero two
    # one.") -- must go, or it fuses onto the last word and swallows it
    # ("one." != "one", so the trailing digit silently vanishes).
    cleaned = re.sub(r"\.(?!\d)", " ", cleaned)
    cleaned = re.sub(r"(?<!\d)\.", " ", cleaned)
    cleaned = _PUNCT_RE.sub(" ", cleaned)
    cleaned = cleaned.replace("-", " ")
    return [t for t in cleaned.split() if t]


def _apply_substitutions(tokens: List[str]) -> List[str]:
    return [SUBSTITUTIONS.get(t, t) for t in tokens]


def _group_chars(tokens: List[str]) -> List[str]:
    """Collapse runs of NATO-letter tokens into one letter token, and runs of
    number-word/decimal tokens into one digit token, e.g.
    ["flight","level","eight","zero"] -> ["flight","level","80"].

    Letter runs and digit runs are never merged into each other: a value
    (digits) is very often followed directly by a callsign (letters) with no
    separating word ("... qnh one zero one three victor tango alpha bravo
    charlie"), and the two must not collapse into one token. Callsign
    matching downstream works on the full concatenated string, so a
    letter-run/digit-run split within a mixed callsign like "tango tango one
    two three" is harmless.
    """
    out: List[str] = []
    buf: List[str] = []
    buf_kind: str = ""  # "num" or "nato"

    def flush():
        nonlocal buf_kind
        if buf:
            out.append("".join(buf).strip("."))
            buf.clear()
        buf_kind = ""

    for tok in tokens:
        if tok in DISFLUENCY_WORDS:
            continue  # never flush, never emit -- transparent to any in-progress run
        if tok in NATO_ALPHABET:
            if buf_kind == "num":
                flush()
            buf_kind = "nato"
            buf.append(NATO_ALPHABET[tok])
        elif tok in NUMBER_WORDS:
            if buf_kind == "nato":
                flush()
            buf_kind = "num"
            buf.append(NUMBER_WORDS[tok])
        elif tok in DECIMAL_WORDS:
            if buf_kind == "num":
                buf.append(".")
            # a decimal word outside a digit run (or with nothing preceding
            # it) carries no information; drop it.
        else:
            flush()
            out.append(tok)
    flush()
    return out


def normalize(raw: str) -> NormalizedTransmission:
    tokens = _tokenize(raw)
    tokens = _apply_substitutions(tokens)
    tokens = _group_chars(tokens)
    return NormalizedTransmission(raw=raw, text=" ".join(tokens), tokens=tokens)
