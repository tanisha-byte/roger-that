"""Callsign generation and ICAO-style spoken phrasing for controller lines.

The controller always *speaks* in digit-by-digit / NATO-alphabet phraseology
(the way real ATC does), regardless of transport mode -- text mode just
delivers that spoken-style string as text instead of synthesizing it. Ground
truth values used for grading are the underlying ints/floats, never the
spoken string.
"""
from __future__ import annotations

import random

_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_AMBIGUOUS = set("IO")  # avoid letters easily confused with digits 1/0

_NATO = {
    "A": "alpha", "B": "bravo", "C": "charlie", "D": "delta", "E": "echo",
    "F": "foxtrot", "G": "golf", "H": "hotel", "I": "india", "J": "juliett",
    "K": "kilo", "L": "lima", "M": "mike", "N": "november", "O": "oscar",
    "P": "papa", "Q": "quebec", "R": "romeo", "S": "sierra", "T": "tango",
    "U": "uniform", "V": "victor", "W": "whiskey", "X": "xray", "Y": "yankee",
    "Z": "zulu",
}

_DIGIT_WORD = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
               6: "six", 7: "seven", 8: "eight", 9: "niner"}


def make_callsign(rng: random.Random, prefix: str = "VT") -> dict:
    letters = "".join(rng.choice([c for c in _LETTERS if c not in _AMBIGUOUS]) for _ in range(3))
    compact = f"{prefix}{letters}"
    display = f"{prefix}-{letters}"
    return {"compact": compact.upper(), "display": display.upper(), "letters": letters.upper(), "prefix": prefix.upper()}


def similar_callsign(rng: random.Random, callsign: dict) -> dict:
    """A callsign that a rushed pilot could mistake for `callsign`: same
    prefix, same first/last letter, one letter in the middle changed."""
    letters = list(callsign["letters"])
    choices = [c for c in _LETTERS if c not in _AMBIGUOUS and c != letters[1]]
    letters[1] = rng.choice(choices)
    letters = "".join(letters)
    prefix = callsign["prefix"]
    return {"compact": f"{prefix}{letters}", "display": f"{prefix}-{letters}", "letters": letters, "prefix": prefix}


def spell_callsign(callsign: dict) -> str:
    """Spells the *full* registration (prefix + suffix), since that's what
    a controller actually reads out -- "victor tango lima echo november"
    for VT-LEN, not just the 3 suffix letters."""
    return " ".join(_NATO[c] for c in callsign["compact"])


def spell_letters(letters: str) -> str:
    return " ".join(_NATO.get(c, c.lower()) for c in letters if c.isalpha())


def spell_digits(n: int) -> str:
    return " ".join(_DIGIT_WORD[int(d)] for d in str(abs(int(n))))


def spell_freq(f: float) -> str:
    whole, frac = f"{f:.1f}".split(".")
    return f"{spell_digits(int(whole))} decimal {spell_digits(int(frac))}"


def spell_squawk(n: int) -> str:
    return spell_digits(n)


def spell_runway(rwy: str) -> str:
    suffix_words = {"L": "left", "R": "right", "C": "center"}
    has_suffix = rwy[-1] in suffix_words
    digits = rwy[:-1] if has_suffix else rwy
    spoken = spell_digits(int(digits))
    return f"{spoken} {suffix_words[rwy[-1]]}" if has_suffix else spoken
