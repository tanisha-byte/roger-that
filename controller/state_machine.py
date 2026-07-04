"""Scripted controller: a deterministic state machine, not an LLM.

Determinism in the controller is what keeps the eval reproducible. It just
walks the RenderedScenario's turn list in order; it does not itself decide
what "correct" looks like (that is scorer.grader's job) beyond the one
piece of real ATC behavior it must reproduce -- issuing a "readback
incorrect, I say again" correction when the pilot muffs an item, and giving
the pilot exactly one retry.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from orchestrator.scenario_loader import RenderedTurn


@dataclass
class ControllerLine:
    turn_index: int
    turn_type: str
    text: str
    is_retry: bool = False


class ScriptedController:
    def __init__(self, turns: List['RenderedTurn']):
        self._turns = turns
        self._pos = 0

    def __iter__(self):
        return self

    def __next__(self) -> 'RenderedTurn':
        if self._pos >= len(self._turns):
            raise StopIteration
        turn = self._turns[self._pos]
        self._pos += 1
        return turn

    @property
    def remaining(self) -> int:
        return len(self._turns) - self._pos

    @staticmethod
    def correction_line(turn: 'RenderedTurn', missing_or_wrong: List[str]) -> str:
        """The standard real-world ATC branch: pilot's readback missed or
        botched an item, so the controller repeats the clearance instead of
        silently letting it go. This exchange is itself scored (see
        session.py) since a correct pilot should nail the retry."""
        items = ", ".join(missing_or_wrong)
        return f"Readback incorrect, I say again: {turn.atc_text} -- confirm {items}"
