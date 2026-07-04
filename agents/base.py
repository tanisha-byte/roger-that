"""The adapter interface every pilot agent implements.

Kept deliberately minimal: start/respond/end, plus an optional learning-loop
extension for agents like Hermes that can be debriefed between sessions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

ROLE_PROMPT_TEMPLATE = (
    "You are the pilot of {callsign} on this frequency. Respond to ATC."
)


@dataclass
class Reply:
    text: str
    latency_ms: int
    audio: Optional[bytes] = None


@runtime_checkable
class PilotAgent(Protocol):
    def start_session(self, role_prompt: str) -> None: ...
    def on_transmission(self, transmission: str) -> Reply: ...
    def end_session(self) -> None: ...


@runtime_checkable
class LearningAgent(PilotAgent, Protocol):
    """Extension for agents whose learning loop the harness can observe
    (e.g. Hermes). The harness only ever calls `debrief` for training-pool
    sessions and only ever sends observational scorecards -- see
    loop/debrief.py and the leakage lint."""

    def debrief(self, scorecard: dict) -> None: ...
    def skill_dir(self) -> Optional[str]: ...
