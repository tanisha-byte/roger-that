"""Atropos-compatible environment wrapper: exposes one scenario session as a
reset()/step() gym-style env, with the composite phraseology score as the
scalar reward, so an RL run can train directly against Roger That.

Honesty note: this follows the common `reset(seed) -> obs` /
`step(action) -> (obs, reward, done, info)` gym convention that
Nous's `atroposlib` environments are built on, but it has not been run
against the actual `atroposlib` package (not available in this workspace).
Treat this as the environment *logic* to slot into an `atroposlib.envs.BaseEnv`
subclass -- the reward/turn-advancement semantics are real and tested
(see tests/test_atropos_env.py); the wire-format glue to atroposlib itself
is not.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from agents.base import ROLE_PROMPT_TEMPLATE
from controller.state_machine import ScriptedController
from orchestrator.scenario_loader import RenderedTurn, render_scenario
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


@dataclass
class StepResult:
    observation: Optional[Dict[str, Any]]
    reward: float
    done: bool
    info: Dict[str, Any]


class RogerThatEnv:
    def __init__(self, scenario: dict):
        self.scenario = scenario
        self._rs = None
        self._state: Optional[AircraftState] = None
        self._turn_i = 0
        self._awaiting_retry: Optional[Tuple[RenderedTurn, TurnRecord]] = None
        self._records: List[TurnRecord] = []

    def reset(self, seed: int) -> Dict[str, Any]:
        self._rs = render_scenario(self.scenario, seed)
        self._state = AircraftState(callsign=self._rs.callsign["compact"])
        self._turn_i = 0
        self._awaiting_retry = None
        self._records = []
        return self._observation()

    def _observation(self) -> Optional[Dict[str, Any]]:
        if self._awaiting_retry is not None:
            turn, rec = self._awaiting_retry
            wrong_or_missing = [item for item, verdict in rec.items.items() if verdict != "ok" and item != "callsign"]
            text = ScriptedController.correction_line(turn, wrong_or_missing)
            return {"role_prompt": self._role_prompt(), "atc_text": text, "turn_index": turn.index, "turn_type": "clearance_retry"}
        if self._turn_i >= len(self._rs.turns):
            return None
        turn = self._rs.turns[self._turn_i]
        return {"role_prompt": self._role_prompt(), "atc_text": turn.atc_text, "turn_index": turn.index, "turn_type": turn.type}

    def _role_prompt(self) -> str:
        return ROLE_PROMPT_TEMPLATE.format(callsign=self._rs.callsign["display"])

    def step(self, action: str) -> StepResult:
        if self._rs is None:
            raise RuntimeError("call reset() before step()")

        if self._awaiting_retry is not None:
            turn, rec = self._awaiting_retry
            self._awaiting_retry = None
            retry_rec = grade_readback(turn.index, [e["item"] for e in turn.expect_readback], turn.expected_values, self._rs.callsign["display"], action)
            for item, verdict in retry_rec.items.items():
                if rec.items.get(item) != "ok" and verdict == "ok":
                    rec.items[item] = "ok"
            for item, verdict in rec.items.items():
                if verdict == "ok" and item != "callsign":
                    self._state.apply_confirmed_item(item, turn.expected_values[item])
            self._turn_i += 1
            return self._advance(reward=0.0)

        turn = self._rs.turns[self._turn_i]

        if turn.type == "clearance":
            items = [e["item"] for e in turn.expect_readback]
            rec = grade_readback(turn.index, items, turn.expected_values, self._rs.callsign["display"], action)
            wrong_or_missing = [i for i, v in rec.items.items() if v != "ok" and i != "callsign"]
            self._records.append(rec)
            if wrong_or_missing:
                self._awaiting_retry = (turn, rec)
                return StepResult(observation=self._observation(), reward=0.0, done=False, info={"turn_index": turn.index, "retry": True})
            for item, verdict in rec.items.items():
                if verdict == "ok" and item != "callsign":
                    self._state.apply_confirmed_item(item, turn.expected_values[item])
            self._turn_i += 1
            return self._advance(reward=0.0)

        if turn.type in ("trap", "clutter"):
            rec = grade_trap(turn.index, action)
        elif turn.type == "hearback_challenge":
            rec = grade_hearback_challenge(turn.index, turn.item, turn.correct_value, action)
        elif turn.type == "state_probe":
            expected_value = self._state.get_item(turn.state_field)
            rec = grade_state_probe(turn.index, turn.state_field, expected_value, action)
        elif turn.type == "emergency_trigger":
            rec = grade_mayday(turn.index, self._rs.callsign["display"], action)
        else:
            raise ValueError(f"unhandled turn type: {turn.type}")

        self._records.append(rec)
        self._turn_i += 1
        return self._advance(reward=0.0)

    def _advance(self, reward: float) -> StepResult:
        obs = self._observation()
        if obs is None:
            scorecard = score_session(self._records)
            return StepResult(observation=None, reward=scorecard["score"], done=True, info={"scorecard": scorecard})
        return StepResult(observation=obs, reward=reward, done=False, info={})
