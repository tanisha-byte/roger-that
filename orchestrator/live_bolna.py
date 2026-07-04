"""Runs a Roger That scenario as a real phone call through Bolna, end to end,
with no manual transcript-wrangling: places the call, polls until it ends,
parses the real transcript, matches each spoken line back to the scenario
turn it corresponds to (robust to the controller reading lines out of
order -- see _match_transcript_to_turns), grades it with the exact same
scorer used everywhere else in this harness, and persists it into the same
SQLite DB the dashboard reads from.

Why this isn't a PilotAgent adapter: every other adapter in agents/ is a
synchronous "send one transmission, get one reply" protocol, because the
harness drives the conversation turn by turn. A Bolna phone call is the
opposite shape -- you place it, it runs to completion on its own, and you
only get to *observe* it afterward via the transcript. So this module
looks like campaign.py's counterpart for a fundamentally async, post-hoc
scoring path, not like session.py's turn-by-turn loop.

Current scope: the controller side is still a human reading scripted lines
into a live call (see the printed script sheet from render_scenario). What
this module automates is everything *after* the human starts talking --
call placement, waiting, transcript retrieval, turn matching, grading, and
storage. Fully automating the controller's *voice* too means a second Bolna
agent placing an agent-to-agent call, which needs a purchased/connected
phone number for inbound -- a real cost decision, deliberately left out of
this module. See README's "Live Bolna calls" section.

Turn-matching scope: robustly handles `clearance` and `trap`/`clutter` turns
(everything the current scenario catalog's callsign-trap/rapid-fire/simple-
descent scenarios use). `hearback_challenge`, `state_probe`, and
`emergency_trigger` turns match on "addressed to own callsign" only, with no
item-based disambiguation between multiple such turns in one scenario --
fine for the current catalog (none has more than one of each per scenario),
but worth strengthening before adding a scenario that does.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

from controller.phraseology import spell_callsign
from db.store import Store
from orchestrator.scenario_loader import RenderedScenario, RenderedTurn, render_scenario
from scorer.grader import (
    TurnRecord,
    grade_hearback_challenge,
    grade_mayday,
    grade_readback,
    grade_state_probe,
    grade_trap,
    score_session,
)
from scorer.grammar import callsign_to_norm, contains_callsign, extract_items

BOLNA_API_BASE = "https://api.bolna.ai"
TERMINAL_STATUSES = {"completed", "call-disconnected", "busy", "no-answer", "failed", "error"}


def _load_bolna_api_key(env_file: Optional[str] = None) -> str:
    key = os.environ.get("BOLNA_API_KEY")
    if key:
        return key
    candidates = [env_file] if env_file else []
    candidates.append(str(Path(__file__).parent.parent / ".env"))
    for path in candidates:
        if path and Path(path).exists():
            for line in Path(path).read_text().splitlines():
                if line.strip().startswith("BOLNA_API_KEY="):
                    return line.split("=", 1)[1].strip()
    raise RuntimeError("BOLNA_API_KEY not found in environment or .env -- set it before running a live call")


class NoTurnsMatchedError(RuntimeError):
    """Raised when zero spoken lines matched any scenario turn. Regression:
    score_session([]) returns a "perfect" 1.0 by design (every per-category
    default is 1.0 when that category had zero turns, which is correct when
    a scenario legitimately has no turns of some type -- e.g. no hearback
    turns). That default is actively dangerous at the whole-session level:
    a live call that matched nothing must never be silently reported as a
    flawless score. Caught on a real call where every line initially failed
    to match; this guard exists so a *future* total matching failure fails
    loudly instead of writing a fake perfect session into the database."""


@dataclass
class LiveCallResult:
    execution_id: str
    scorecard: dict
    turn_rows: List[Dict[str, Any]]
    unmatched_lines: List[str]
    transcript: str
    rendered: RenderedScenario


def place_call(agent_id: str, recipient_phone_number: str, user_data: dict, api_key: str, client: httpx.Client) -> str:
    resp = client.post(
        f"{BOLNA_API_BASE}/call",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"agent_id": agent_id, "recipient_phone_number": recipient_phone_number, "user_data": user_data},
    )
    resp.raise_for_status()
    return resp.json()["execution_id"]


def wait_for_execution(execution_id: str, api_key: str, client: httpx.Client, poll_interval_s: float = 5.0, timeout_s: float = 180.0) -> dict:
    deadline = time.monotonic() + timeout_s
    while True:
        resp = client.get(f"{BOLNA_API_BASE}/executions/{execution_id}", headers={"Authorization": f"Bearer {api_key}"})
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") in TERMINAL_STATUSES:
            return data
        if time.monotonic() > deadline:
            raise TimeoutError(f"execution {execution_id} did not reach a terminal status within {timeout_s}s (last status: {data.get('status')})")
        time.sleep(poll_interval_s)


_LINE_RE = re.compile(r"^(assistant|user):\s*(.*)$")


def _parse_transcript(transcript: str) -> List[Tuple[str, str]]:
    """Merges consecutive same-speaker lines, then returns [(speaker, text), ...]."""
    raw_lines = [_LINE_RE.match(line) for line in transcript.splitlines() if line.strip()]
    raw_lines = [(m.group(1), m.group(2).strip()) for m in raw_lines if m]

    merged: List[Tuple[str, str]] = []
    for speaker, text in raw_lines:
        if merged and merged[-1][0] == speaker:
            merged[-1] = (speaker, f"{merged[-1][1]} {text}".strip())
        else:
            merged.append((speaker, text))
    return merged


def _turn_addressed_to(turn: RenderedTurn, own_norm: str, intruder_norm: str) -> str:
    extracted = extract_items(turn.atc_text)
    if contains_callsign(extracted.tokens, own_norm):
        return "own"
    if intruder_norm and contains_callsign(extracted.tokens, intruder_norm):
        return "intruder"
    return "unknown"


def _line_addressed_to(text: str, own_norm: str, intruder_norm: str) -> str:
    extracted = extract_items(text)
    if contains_callsign(extracted.tokens, own_norm):
        return "own"
    if intruder_norm and contains_callsign(extracted.tokens, intruder_norm):
        return "intruder"
    return "unknown"


def _match_transcript_to_turns(
    user_lines: List[str], turns: List[RenderedTurn], own_norm: str, intruder_norm: str
) -> Tuple[List[Tuple[RenderedTurn, str]], List[str]]:
    """Matches each spoken (user) line to the scenario turn it most likely
    corresponds to, independent of the order they were actually spoken in --
    a human controller can (and did, in testing) read scenario lines out of
    order.

    Scoring is full-token overlap between the spoken line and each candidate
    turn's own canonical `atc_text` -- not just extracted-item overlap.
    Regression: item-overlap alone scores every non-clearance turn (state
    probes, hearback challenges) as a flat 0, so on a live call where two
    other turns' callsigns got mangled by ASR and also scored 0, a state-probe
    line that WAS transcribed perfectly still lost a 3-way tie to whichever
    turn happened to come first in scenario order. Token overlap against the
    full canonical line (e.g. "confirm your assigned flight level" sharing
    6 tokens with its own turn vs. 1-3 with the others) resolves this
    correctly regardless of turn type.

    The callsign-addressed-to check (own vs. intruder) is still a hard
    filter, but only when confidently known from the spoken line -- this is
    what keeps trap turns from ever being matched against an own-addressed
    clearance. When the callsign itself is unrecoverable (badly ASR-mangled,
    as also happened in testing), that hard filter is skipped and content
    overlap decides alone, rather than dropping an otherwise-recoverable
    line entirely.

    Consumes each candidate turn at most once."""
    remaining = list(turns)
    matched: List[Tuple[RenderedTurn, str]] = []
    unmatched: List[str] = []

    for line in user_lines:
        spoken_to = _line_addressed_to(line, own_norm, intruder_norm)
        spoken_tokens = set(extract_items(line).tokens)

        best, best_score = None, 0  # require at least one shared token -- never force-match pure noise
        for t in remaining:
            turn_to = _turn_addressed_to(t, own_norm, intruder_norm)
            if spoken_to != "unknown" and turn_to != spoken_to:
                continue
            turn_tokens = set(extract_items(t.atc_text).tokens)
            score = len(spoken_tokens & turn_tokens)
            if score > best_score:
                best, best_score = t, score

        if best is None:
            unmatched.append(line)
        else:
            matched.append((best, line))
            remaining.remove(best)

    return matched, unmatched


def _ground_truth_for_state_field(state_field: str, before_index: int, all_turns: List[RenderedTurn]) -> Optional[Any]:
    """A state_probe RenderedTurn's own `expected_values` is always empty --
    only `clearance` turns populate that dict (see scenario_loader.py). The
    real ground truth is whatever the most recent prior clearance turn
    assigned to this field (mirroring AircraftState's overwrite-on-amend
    semantics: a later re-clearance takes precedence over an earlier one)."""
    value = None
    for t in all_turns:
        if t.index >= before_index:
            break
        if t.type == "clearance" and state_field in t.expected_values:
            value = t.expected_values[state_field]
    return value


def _grade_matched_turn(turn: RenderedTurn, reply_text: str, callsign_display: str, all_turns: List[RenderedTurn]) -> TurnRecord:
    if turn.type == "clearance":
        items = [e["item"] for e in turn.expect_readback]
        return grade_readback(turn.index, items, turn.expected_values, callsign_display, reply_text)
    if turn.type in ("trap", "clutter"):
        return grade_trap(turn.index, reply_text)
    if turn.type == "hearback_challenge":
        return grade_hearback_challenge(turn.index, turn.item, turn.correct_value, reply_text)
    if turn.type == "state_probe":
        expected_value = _ground_truth_for_state_field(turn.state_field, turn.index, all_turns)
        return grade_state_probe(turn.index, turn.state_field, expected_value, reply_text)
    if turn.type == "emergency_trigger":
        return grade_mayday(turn.index, callsign_display, reply_text)
    raise ValueError(f"unhandled turn type: {turn.type}")


def score_transcript(transcript: str, rs: RenderedScenario) -> LiveCallResult:
    """The pure, network-free core: transcript in, graded+matched result
    out. Split out from run_live_bolna_session() so this logic (and its
    NoTurnsMatchedError guard) is directly unit-testable without mocking
    an HTTP call."""
    own_norm = callsign_to_norm(rs.callsign["display"])
    intruder = rs.ctx.get("intruder")
    intruder_norm = callsign_to_norm(intruder["display"]) if intruder else ""

    parsed = _parse_transcript(transcript)
    # first assistant line is always the welcome message ("<callsign>, standing by."),
    # never a reply to anything -- drop it before pairing user lines with replies
    if parsed and parsed[0][0] == "assistant":
        parsed = parsed[1:]

    user_lines = [text for speaker, text in parsed if speaker == "user"]
    matched, unmatched = _match_transcript_to_turns(user_lines, rs.turns, own_norm, intruder_norm)

    # pair each matched user line with the assistant line that followed it
    # in the original (unsorted) transcript order
    line_to_reply: Dict[str, str] = {}
    for i, (speaker, text) in enumerate(parsed):
        if speaker == "user":
            reply = next((t for s, t in parsed[i + 1:] if s == "assistant"), "")
            line_to_reply[text] = reply

    records: List[TurnRecord] = []
    turn_rows: List[Dict[str, Any]] = []
    for turn, user_line in matched:
        reply_text = line_to_reply.get(user_line, "")
        rec = _grade_matched_turn(turn, reply_text, rs.callsign["display"], rs.turns)
        records.append(rec)
        turn_rows.append({
            "turn_index": turn.index, "kind": turn.type, "atc_text": user_line,
            "reply_text": reply_text, "latency_ms": None, "record": rec.to_dict(),
        })

    if not records:
        raise NoTurnsMatchedError(
            f"0 of {len(user_lines)} spoken line(s) matched any scenario turn -- refusing to report or persist "
            f"a scorecard with no evidence behind it. Unmatched lines: {unmatched}"
        )

    scorecard = score_session(records)
    return LiveCallResult(
        execution_id="", scorecard=scorecard, turn_rows=turn_rows,
        unmatched_lines=unmatched, transcript=transcript, rendered=rs,
    )


def run_live_bolna_session(
    store: Store,
    scenario: dict,
    seed: int,
    agent_id: str,
    recipient_phone_number: str,
    agent_name: str = "bolna-live",
    campaign_id: str = "live-bolna",
    poll_interval_s: float = 5.0,
    poll_timeout_s: float = 180.0,
    api_key: Optional[str] = None,
) -> LiveCallResult:
    api_key = api_key or _load_bolna_api_key()
    rs = render_scenario(scenario, seed)

    with httpx.Client(timeout=30.0) as client:
        execution_id = place_call(agent_id, recipient_phone_number, {"callsign": rs.callsign["display"]}, api_key, client)
        execution = wait_for_execution(execution_id, api_key, client, poll_interval_s, poll_timeout_s)

    transcript = execution.get("transcript") or ""
    partial = score_transcript(transcript, rs)  # raises NoTurnsMatchedError before anything is persisted
    scorecard, turn_rows, unmatched = partial.scorecard, partial.turn_rows, partial.unmatched_lines

    session_id = store.insert_session({
        "campaign_id": campaign_id, "agent": agent_name, "scenario_id": scenario["id"], "seed": seed,
        "mode": "voice", "snr": None, "pool": "live", "session_index": 0,
        "score": scorecard["score"], "readback_completeness": scorecard["readback_completeness"],
        "callsign_discipline": scorecard["callsign_discipline"], "safety_score": scorecard["safety_score"],
        "safety_violations": scorecard["safety_violations"], "hearback_catch_rate": scorecard["hearback_catch_rate"],
        "state_consistency": scorecard["state_consistency"], "emergency_completeness": scorecard["emergency_completeness"],
        "weights_version": scorecard["weights_version"], "scorecard_json": __import__("json").dumps(scorecard),
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
    store.insert_turns(session_id, turn_rows)

    return LiveCallResult(
        execution_id=execution_id, scorecard=scorecard, turn_rows=turn_rows,
        unmatched_lines=unmatched, transcript=transcript, rendered=rs,
    )
