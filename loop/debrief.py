"""Debrief protocol for the Hermes learning-loop sidecar.

The debrief message is observational only: what was missed, never why or
how to fix it. The `debrief_leakage` lint checks every debrief string
against a banned-phrase list before it is ever sent, so the harness cannot
accidentally teach the phraseology rules it's supposed to be testing for.
"""
from __future__ import annotations

from typing import List

BANNED_PHRASES = [
    "must read back",
    "must include",
    "icao requires",
    "icao doc",
    "regulation requires",
    "the rule is",
    "you should have",
    "you need to",
    "required items are",
    "mandatory items",
    "correct phraseology is",
    "the correct answer",
]


class DebriefLeakageError(Exception):
    pass


def lint_debrief(text: str) -> List[str]:
    lowered = text.lower()
    return [phrase for phrase in BANNED_PHRASES if phrase in lowered]


def build_debrief_text(scorecard: dict) -> str:
    """Purely descriptive: turn-by-turn outcomes and the aggregate numbers,
    with zero instruction. What Hermes does with this is up to Hermes."""
    lines = [f"Session score: {scorecard['score']:.2f}."]

    if scorecard.get("safety_violations"):
        for v in scorecard["safety_violations"]:
            lines.append(f"Flagged: {v}.")
    else:
        lines.append("No safety flags this session.")

    lines.append(f"Readback completeness: {scorecard['readback_completeness']:.2f}.")
    lines.append(f"Callsign discipline: {scorecard['callsign_discipline']:.2f}.")
    if scorecard.get("hearback_injected"):
        lines.append(f"Hearback checks: {scorecard['hearback_caught']}/{scorecard['hearback_injected']} caught.")
    lines.append(f"State consistency: {scorecard['state_consistency']:.2f}.")
    lines.append(f"Emergency completeness: {scorecard['emergency_completeness']:.2f}.")

    text = " ".join(lines)
    violations = lint_debrief(text)
    if violations:
        raise DebriefLeakageError(f"debrief text contains banned phrase(s): {violations}")
    return text


def send_debrief(agent, scorecard: dict) -> None:
    """Only ever call this for training-pool sessions -- held-out sessions
    must be scored silently (see orchestrator/campaign.py)."""
    text = build_debrief_text(scorecard)
    agent.debrief({**scorecard, "debrief_text": text})
