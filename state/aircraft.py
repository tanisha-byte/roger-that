"""Aircraft state tracker: a per-session dict, not a flight model.

Updated only when the scorer confirms a clearance item was correctly read
back. This is what lets later "confirm assigned level" turns be graded
against ground truth instead of a judge.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Optional


@dataclass
class AircraftState:
    callsign: str
    altitude_fl: Optional[int] = None
    heading: Optional[int] = None
    speed: Optional[int] = None
    qnh: Optional[int] = None
    frequency: Optional[float] = None
    runway: Optional[str] = None
    squawk: Optional[int] = None
    # arbitrary open clearances not covered by the typed fields above,
    # e.g. {"after_passing_fl": 100, "then_speed": 220}
    open_clearances: dict = field(default_factory=dict)
    history: list = field(default_factory=list)

    FIELD_BY_ITEM = {
        "fl": "altitude_fl",
        "hdg": "heading",
        "spd": "speed",
        "qnh": "qnh",
        "freq": "frequency",
        "rwy": "runway",
        "squawk": "squawk",
    }

    def apply_confirmed_item(self, item_type: str, value: Any) -> None:
        field_name = self.FIELD_BY_ITEM.get(item_type)
        if field_name is None:
            return
        self.history.append((field_name, getattr(self, field_name), value))
        setattr(self, field_name, value)

    def get_item(self, item_type: str) -> Any:
        field_name = self.FIELD_BY_ITEM.get(item_type)
        if field_name is None:
            return self.open_clearances.get(item_type)
        return getattr(self, field_name)

    def snapshot(self) -> dict:
        return {
            "callsign": self.callsign,
            "altitude_fl": self.altitude_fl,
            "heading": self.heading,
            "speed": self.speed,
            "qnh": self.qnh,
            "frequency": self.frequency,
            "runway": self.runway,
            "squawk": self.squawk,
            "open_clearances": dict(self.open_clearances),
        }

    def clone(self) -> "AircraftState":
        return replace(self, open_clearances=dict(self.open_clearances), history=list(self.history))
