"""Runs one scenario session against one agent: the loop that ties the
controller, the agent-under-test, the aircraft state tracker, and the
scorer together.

Two transport modes, one scoring path (per the project brief): in text mode
`turn.atc_text` is sent straight to the agent as a string; in voice mode the
same text would be TTS'd and degraded first (see channel/) and the agent's
audio reply transcribed by the reference ASR before it ever reaches this
module. This module only ever sees text in and text out, so the scorer
truly cannot tell which mode produced it.
"""
from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from agents.base import PilotAgent, ROLE_PROMPT_TEMPLATE
from controller.state_machine import ScriptedController
from orchestrator.scenario_loader import RenderedScenario, RenderedTurn, render_scenario
from state.aircraft import AircraftState
from scorer.grader import (
    TurnRecord,
    grade_hearback_challenge,
    grade_mayday,
    grade_readback,
    grade_state_probe,
    grade_trap,
    score_session,
)

RESPONSE_TIMEOUT_S = 10.0
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)


@dataclass
class SessionResult:
    scorecard: dict
    turn_rows: List[Dict[str, Any]]
    rendered: RenderedScenario


def _call_agent(agent: PilotAgent, text: str, timeout_s: float = RESPONSE_TIMEOUT_S):
    future = _executor.submit(agent.on_transmission, text)
    try:
        return future.result(timeout=timeout_s)
    except concurrent.futures.TimeoutError:
        from agents.base import Reply

        return Reply(text="", latency_ms=int(timeout_s * 1000))


def _grade_clearance(turn: RenderedTurn, agent: PilotAgent, own_callsign: str, state: AircraftState) -> tuple[TurnRecord, List[Dict[str, Any]]]:
    items = [e["item"] for e in turn.expect_readback]
    reply = _call_agent(agent, turn.atc_text)
    rows = [{"turn_index": turn.index, "kind": "clearance", "atc_text": turn.atc_text, "reply_text": reply.text, "latency_ms": reply.latency_ms}]
    rec = grade_readback(turn.index, items, turn.expected_values, own_callsign, reply.text, reply.latency_ms)

    wrong_or_missing = [item for item, verdict in rec.items.items() if verdict != "ok" and item != "callsign"]
    if wrong_or_missing:
        correction = ScriptedController.correction_line(turn, wrong_or_missing)
        retry_reply = _call_agent(agent, correction)
        rows.append({"turn_index": turn.index, "kind": "clearance_retry", "atc_text": correction, "reply_text": retry_reply.text, "latency_ms": retry_reply.latency_ms})
        retry_rec = grade_readback(turn.index, items, turn.expected_values, own_callsign, retry_reply.text, retry_reply.latency_ms)
        for item, verdict in retry_rec.items.items():
            if rec.items.get(item) != "ok" and verdict == "ok":
                rec.items[item] = "ok"

    for item, verdict in rec.items.items():
        if verdict == "ok" and item != "callsign":
            state.apply_confirmed_item(item, turn.expected_values[item])

    rows[0]["record"] = rec.to_dict()
    if len(rows) > 1:
        rows[1]["record"] = rec.to_dict()
    return rec, rows


def run_session(
    agent: PilotAgent,
    scenario: dict,
    seed: int,
) -> SessionResult:
    rs = render_scenario(scenario, seed)
    own_callsign = rs.callsign["display"]
    state = AircraftState(callsign=rs.callsign["compact"])

    agent.start_session(ROLE_PROMPT_TEMPLATE.format(callsign=own_callsign))
    records: List[TurnRecord] = []
    turn_rows: List[Dict[str, Any]] = []

    try:
        for turn in rs.turns:
            if turn.type == "clearance":
                rec, rows = _grade_clearance(turn, agent, own_callsign, state)
                records.append(rec)
                turn_rows.extend(rows)

            elif turn.type in ("trap", "clutter"):
                reply = _call_agent(agent, turn.atc_text)
                rec = grade_trap(turn.index, reply.text, reply.latency_ms)
                records.append(rec)
                turn_rows.append({"turn_index": turn.index, "kind": turn.type, "atc_text": turn.atc_text, "reply_text": reply.text, "latency_ms": reply.latency_ms, "record": rec.to_dict()})

            elif turn.type == "hearback_challenge":
                reply = _call_agent(agent, turn.atc_text)
                rec = grade_hearback_challenge(turn.index, turn.item, turn.correct_value, reply.text, reply.latency_ms)
                records.append(rec)
                turn_rows.append({"turn_index": turn.index, "kind": turn.type, "atc_text": turn.atc_text, "reply_text": reply.text, "latency_ms": reply.latency_ms, "record": rec.to_dict()})

            elif turn.type == "state_probe":
                expected_value = state.get_item(turn.state_field)
                reply = _call_agent(agent, turn.atc_text)
                rec = grade_state_probe(turn.index, turn.state_field, expected_value, reply.text, reply.latency_ms)
                records.append(rec)
                turn_rows.append({"turn_index": turn.index, "kind": turn.type, "atc_text": turn.atc_text, "reply_text": reply.text, "latency_ms": reply.latency_ms, "record": rec.to_dict()})

            elif turn.type == "emergency_trigger":
                reply = _call_agent(agent, turn.atc_text)
                rec = grade_mayday(turn.index, own_callsign, reply.text, reply.latency_ms)
                records.append(rec)
                turn_rows.append({"turn_index": turn.index, "kind": turn.type, "atc_text": turn.atc_text, "reply_text": reply.text, "latency_ms": reply.latency_ms, "record": rec.to_dict()})

            else:
                raise ValueError(f"unhandled turn type: {turn.type}")
    finally:
        agent.end_session()

    scorecard = score_session(records)
    return SessionResult(scorecard=scorecard, turn_rows=turn_rows, rendered=rs)
