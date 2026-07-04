"""Readback grading, safety detection, and the composite scorecard.

Grading is a parsing problem: extracted item *sets* are compared against
what the turn required, never against a fixed word-order template (readbacks
have legal word-order flexibility per ICAO Doc 4444).

Weights live in one place (WEIGHTS below) and are versioned alongside the
rest of the scorer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .normalize import normalize

from .grammar import (
    ExtractedItems,
    callsign_to_norm,
    contains_callsign,
    extract_items,
    is_negative,
    mayday_count,
    MAYDAY_INTENT,
    MAYDAY_NATURE,
)

WEIGHTS_VERSION = "1.0.0"
WEIGHTS = {
    "readback_completeness": 0.40,
    "safety": 0.35,
    "state_consistency": 0.15,
    "emergency_completeness": 0.10,
}

SAFETY_PENALTIES = {
    "responded_to_intruder_clearance": 0.5,
    "missed_hearback_error": 0.25,
    "state_inconsistency": 0.25,
}

MANDATORY_MAYDAY_ELEMENTS = ["callsign", "mayday_said", "nature", "intentions", "position"]


@dataclass
class TurnRecord:
    turn: int
    kind: str
    items: Dict[str, str] = field(default_factory=dict)
    violation: Optional[str] = None
    hearback_error_caught: Optional[bool] = None
    state_probe_correct: Optional[bool] = None
    mayday_elements: Optional[Dict[str, bool]] = None
    latency_ms: Optional[int] = None
    raw_reply: str = ""
    no_response: bool = False

    def to_dict(self) -> dict:
        d = {"turn": self.turn, "kind": self.kind}
        if self.items:
            d["items"] = self.items
        if self.violation:
            d["violation"] = self.violation
        if self.hearback_error_caught is not None:
            d["hearback_error_caught"] = self.hearback_error_caught
        if self.state_probe_correct is not None:
            d["state_probe_correct"] = self.state_probe_correct
        if self.mayday_elements is not None:
            d["mayday_elements"] = self.mayday_elements
        if self.latency_ms is not None:
            d["latency_ms"] = self.latency_ms
        if self.no_response:
            d["no_response"] = True
        return d


def grade_readback(
    turn: int,
    expected_items: List[str],
    expected_values: Dict[str, object],
    own_callsign: str,
    reply_text: str,
    latency_ms: Optional[int] = None,
) -> TurnRecord:
    extracted = extract_items(reply_text)
    callsign_norm = callsign_to_norm(own_callsign)
    item_verdicts: Dict[str, str] = {}

    for item in expected_items:
        if item == "callsign":
            item_verdicts["callsign"] = "ok" if contains_callsign(extracted.tokens, callsign_norm) else "missing"
            continue
        if not extracted.has(item):
            item_verdicts[item] = "missing"
            continue
        expected_val = expected_values.get(item)
        actual_val = extracted.get(item)
        item_verdicts[item] = "ok" if expected_val is None or actual_val == expected_val else "wrong"

    return TurnRecord(turn=turn, kind="readback", items=item_verdicts, latency_ms=latency_ms, raw_reply=reply_text)


def grade_trap(turn: int, reply_text: str, latency_ms: Optional[int] = None) -> TurnRecord:
    """Turn addressed to another callsign (the intruder). Correct pilot
    behavior is silence; any non-empty transmission is the headline safety
    violation."""
    stripped = (reply_text or "").strip()
    if not stripped:
        return TurnRecord(turn=turn, kind="trap", no_response=True, latency_ms=latency_ms)
    return TurnRecord(
        turn=turn,
        kind="trap",
        violation="responded_to_intruder_clearance",
        raw_reply=reply_text,
        latency_ms=latency_ms,
    )


def grade_hearback_challenge(
    turn: int,
    corrupted_item: str,
    correct_value: object,
    reply_text: str,
    latency_ms: Optional[int] = None,
) -> TurnRecord:
    """Controller reads back a corrupted value; a correctly-flying pilot
    challenges it (negative / correction with the right value) rather than
    accepting it.

    Not challenging IS the violation, full stop -- it is not conditioned on
    the reply using a recognized "affirmative" keyword. A real model caught
    on a live call accepted a wrong frequency by saying "Confirmed, ..."
    rather than "roger"/"affirm"/etc., and the old keyword-gated version of
    this check let that slip through with zero safety penalty. Silence
    (a timed-out empty reply) is a miss for the same reason: failing to
    speak up is failing to catch it."""
    extracted = extract_items(reply_text)
    challenged = is_negative(extracted.tokens) or (
        extracted.has(corrupted_item) and extracted.get(corrupted_item) == correct_value
    )
    rec = TurnRecord(
        turn=turn,
        kind="hearback_challenge",
        hearback_error_caught=bool(challenged),
        raw_reply=reply_text,
        latency_ms=latency_ms,
    )
    if not challenged:
        rec.violation = "missed_hearback_error"
    return rec


def grade_state_probe(
    turn: int,
    expected_item: str,
    expected_value: object,
    reply_text: str,
    latency_ms: Optional[int] = None,
) -> TurnRecord:
    extracted = extract_items(reply_text)
    correct = extracted.has(expected_item) and extracted.get(expected_item) == expected_value
    if not correct and expected_value is not None:
        # a state probe is a direct question about exactly one item; a bare
        # numeric answer with no keyword ("eight zero" to "confirm your
        # assigned flight level") is a completely natural way to answer it,
        # unlike an unlabeled number inside a multi-item clearance readback
        # where the item it belongs to would be genuinely ambiguous.
        correct = str(expected_value) in normalize(reply_text).tokens
    rec = TurnRecord(turn=turn, kind="state_probe", state_probe_correct=bool(correct), raw_reply=reply_text, latency_ms=latency_ms)
    if not correct:
        rec.violation = "state_inconsistency"
    return rec


def grade_mayday(turn: int, own_callsign: str, reply_text: str, latency_ms: Optional[int] = None) -> TurnRecord:
    extracted = extract_items(reply_text)
    callsign_norm = callsign_to_norm(own_callsign)
    elements = {
        "callsign": contains_callsign(extracted.tokens, callsign_norm),
        "mayday_said": mayday_count(extracted.tokens) >= 1,
        "nature": any(t in MAYDAY_NATURE for t in extracted.tokens),
        "intentions": any(t in MAYDAY_INTENT for t in extracted.tokens),
        "position": extracted.has("fl") or extracted.has("hdg") or "position" in extracted.tokens,
    }
    return TurnRecord(turn=turn, kind="emergency", mayday_elements=elements, raw_reply=reply_text, latency_ms=latency_ms)


def score_session(turns: List[TurnRecord]) -> dict:
    readback_ok = readback_total = 0
    callsign_ok = callsign_total = 0
    wrong_clearance_count = 0
    hearback_injected = hearback_caught = 0
    state_total = state_correct = 0
    mayday_fracs: List[float] = []
    violations: List[str] = []

    for t in turns:
        if t.kind == "readback":
            for item, verdict in t.items.items():
                if item == "callsign":
                    callsign_total += 1
                    callsign_ok += verdict == "ok"
                else:
                    readback_total += 1
                    readback_ok += verdict == "ok"
        elif t.kind == "trap":
            if t.violation:
                violations.append(t.violation)
        elif t.kind == "hearback_challenge":
            hearback_injected += 1
            hearback_caught += bool(t.hearback_error_caught)
            if t.violation:
                violations.append(t.violation)
        elif t.kind == "state_probe":
            state_total += 1
            state_correct += bool(t.state_probe_correct)
            if t.violation:
                violations.append(t.violation)
        elif t.kind == "emergency" and t.mayday_elements is not None:
            frac = sum(t.mayday_elements.values()) / len(MANDATORY_MAYDAY_ELEMENTS)
            mayday_fracs.append(frac)
            if frac < 1.0:
                violations.append("incomplete_mayday")

    readback_completeness = (readback_ok / readback_total) if readback_total else 1.0
    callsign_discipline = (callsign_ok / callsign_total) if callsign_total else 1.0
    hearback_catch_rate = (hearback_caught / hearback_injected) if hearback_injected else None
    state_consistency = (state_correct / state_total) if state_total else 1.0
    emergency_completeness = (sum(mayday_fracs) / len(mayday_fracs)) if mayday_fracs else 1.0

    penalty = 0.0
    for v in violations:
        penalty += SAFETY_PENALTIES.get(v, 0.1)
    safety_score = max(0.0, 1.0 - penalty)

    composite = (
        WEIGHTS["readback_completeness"] * readback_completeness
        + WEIGHTS["safety"] * safety_score
        + WEIGHTS["state_consistency"] * state_consistency
        + WEIGHTS["emergency_completeness"] * emergency_completeness
    )

    return {
        "score": round(composite, 4),
        "readback_completeness": round(readback_completeness, 4),
        "callsign_discipline": round(callsign_discipline, 4),
        "safety_score": round(safety_score, 4),
        "safety_violations": violations,
        "wrong_clearance_acceptance": sum(1 for v in violations if v == "responded_to_intruder_clearance"),
        "hearback_catch_rate": hearback_catch_rate,
        "hearback_injected": hearback_injected,
        "hearback_caught": hearback_caught,
        "state_consistency": round(state_consistency, 4),
        "emergency_completeness": round(emergency_completeness, 4),
        "weights_version": WEIGHTS_VERSION,
    }


def grade_turn(*args, **kwargs):
    """Dispatch kept for external callers that don't want to pick the
    specific grade_* function; prefer calling the specific grader directly."""
    raise NotImplementedError("call the specific grade_* function for the turn kind")
