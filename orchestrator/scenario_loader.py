"""Scenario YAML loader + seeded rendering.

Scenarios are parameterized: callsigns, flight levels, headings, QNH,
frequencies are regenerated from a seeded RNG every session, so nothing is
answerable by memorization -- only the procedure transfers.

YAML schema (see scenarios/*.yaml for real examples):

    id: descent-with-confusion-01
    difficulty: 2
    airport: VOBL
    title: Descent with a similar-callsign trap
    vars:
      callsign: {type: callsign, prefix: VT}
      intruder: {type: similar_callsign, based_on: callsign}
      fl: {type: int_range, min: 60, max: 150, step: 10}
      qnh: {type: int_range, min: 990, max: 1030, step: 1}
    script:
      - type: clearance
        atc: "{callsign_spoken}, descend flight level {fl_spoken}, QNH {qnh_spoken}"
        expect_readback: [fl, qnh, callsign]
      - type: trap
        atc: "{intruder_spoken}, turn left heading {hdg2_spoken}"

This deviates from the brief's illustrative YAML sketch in one deliberate
way: fault turns (hearback errors, traps) are written directly into `script`
as their own turn `type` rather than a separate `faults:` block that
mutates the script at runtime. Same effect, far simpler loader -- there is
no dynamic script rewriting to get wrong, which matters for a scorer that
has to be trusted more than the agents it grades.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from controller.phraseology import make_callsign, similar_callsign, spell_callsign, spell_digits, spell_freq, spell_runway

SCENARIOS_DIR = Path(__file__).parent.parent / "scenarios"

_ITEM_KEYS = {"fl", "qnh", "hdg", "freq", "spd", "rwy", "squawk"}

# how far a corrupted hearback value drifts from the truth, per item type --
# large enough that a correct pilot must notice, small enough to be
# plausible ASR/controller slip rather than an obviously different number
_CORRUPTION_OFFSETS = {
    "freq": 0.2,
    "qnh": 6,
    "fl": 10,
    "hdg": 20,
    "spd": 20,
    "squawk": 1100,
}


@dataclass
class RenderedTurn:
    index: int
    type: str
    atc_text: str
    expect_readback: List[Dict[str, str]] = field(default_factory=list)
    expected_values: Dict[str, Any] = field(default_factory=dict)
    item: Optional[str] = None
    correct_value: Optional[Any] = None
    wrong_value: Optional[Any] = None
    state_field: Optional[str] = None


@dataclass
class RenderedScenario:
    id: str
    difficulty: int
    airport: str
    title: str
    callsign: dict
    ctx: Dict[str, Any]
    turns: List[RenderedTurn]
    seed: int


def load_scenario_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def list_scenarios(directory: Path = SCENARIOS_DIR) -> List[dict]:
    return [load_scenario_yaml(p) for p in sorted(directory.glob("*.yaml"))]


def _gen_var(rng: random.Random, name: str, spec: dict, ctx: Dict[str, Any]) -> None:
    kind = spec["type"]
    if kind == "callsign":
        val = make_callsign(rng, prefix=spec.get("prefix", "VT"))
        ctx[name] = val
        ctx[f"{name}_spoken"] = spell_callsign(val)
        ctx[f"{name}_display"] = val["display"]
        ctx[f"{name}_compact"] = val["compact"]
    elif kind == "similar_callsign":
        base = ctx[spec["based_on"]]
        val = similar_callsign(rng, base)
        ctx[name] = val
        ctx[f"{name}_spoken"] = spell_callsign(val)
        ctx[f"{name}_display"] = val["display"]
        ctx[f"{name}_compact"] = val["compact"]
    elif kind == "int_range":
        step = spec.get("step", 1)
        lo, hi = spec["min"], spec["max"]
        val = rng.randrange(lo, hi + 1, step)
        ctx[name] = val
        ctx[f"{name}_spoken"] = spell_digits(val)
    elif kind == "freq_range":
        lo, hi = spec["min"], spec["max"]
        val = round(rng.uniform(lo, hi), 1)
        ctx[name] = val
        ctx[f"{name}_spoken"] = spell_freq(val)
    elif kind == "choice":
        val = rng.choice(spec["values"])
        ctx[name] = val
        ctx[f"{name}_spoken"] = spell_runway(val) if _looks_like_runway(val) else str(val)
    elif kind == "squawk":
        val = rng.randint(1000, 7777)
        ctx[name] = val
        ctx[f"{name}_spoken"] = spell_digits(val)
    else:
        raise ValueError(f"unknown var type: {kind}")


def _looks_like_runway(v: str) -> bool:
    return isinstance(v, str) and v[:2].isdigit()


def _normalize_readback_spec(entries: List[Any]) -> List[Dict[str, str]]:
    out = []
    for e in entries:
        if isinstance(e, str):
            out.append({"item": e, "var": e})
        else:
            out.append({"item": e["item"], "var": e.get("var", e["item"])})
    return out


def _apply_bounds(item: str, value: int) -> int:
    """Keeps a corrupted int value inside its real-world domain. Without
    this, a heading near 0 (e.g. 10) offset downward went negative (-10),
    and spell_digits() takes abs() before speaking it -- so the "corrupted"
    hearback challenge was spoken identically to the truth ("one zero" for
    both -10 and 10), silently defeating the fault: a pilot that caught
    nothing would score a false "caught it"."""
    if item == "hdg":
        value = value % 360
        return 360 if value == 0 else value  # ATC headings are 001-360, never 000
    if item in ("qnh", "fl", "spd"):
        return max(1, value)
    if item == "squawk":
        return max(0, min(7777, value))
    return value


def _corrupt(item: str, correct: Any, rng: random.Random) -> Any:
    offset = _CORRUPTION_OFFSETS.get(item, 1)
    sign = rng.choice([-1, 1])

    def compute(s: int):
        if isinstance(correct, float):
            return round(correct + s * offset, 1)
        return _apply_bounds(item, int(correct) + s * int(offset))

    wrong = compute(sign)
    if wrong == correct:
        # bounds/wraparound collapsed the fault back onto the truth -- flip
        # direction rather than silently emitting a fault-free "fault"
        wrong = compute(-sign)
    return wrong


def render_scenario(scenario: dict, seed: int) -> RenderedScenario:
    rng = random.Random(seed)
    ctx: Dict[str, Any] = {}
    for name, spec in scenario.get("vars", {}).items():
        _gen_var(rng, name, spec, ctx)

    turns: List[RenderedTurn] = []
    for i, raw_turn in enumerate(scenario["script"], start=1):
        ttype = raw_turn["type"]

        if ttype in ("clearance", "trap", "clutter", "emergency_trigger"):
            atc_text = raw_turn["atc"].format(**ctx)
            rt = RenderedTurn(index=i, type=ttype, atc_text=atc_text)
            if ttype == "clearance":
                spec = _normalize_readback_spec(raw_turn["expect_readback"])
                rt.expect_readback = spec
                rt.expected_values = {
                    e["item"]: (ctx[e["var"]]["compact"] if e["item"] == "callsign" else ctx[e["var"]])
                    for e in spec
                }
            turns.append(rt)

        elif ttype == "hearback_challenge":
            item = raw_turn["item"]
            var = raw_turn.get("var", item)
            correct = ctx[var]
            wrong = _corrupt(item, correct, rng)
            wrong_spoken = spell_digits(wrong) if item != "freq" else spell_freq(wrong)
            local_ctx = dict(ctx)
            local_ctx[f"{item}_wrong_spoken"] = wrong_spoken
            atc_text = raw_turn["atc"].format(**local_ctx)
            turns.append(RenderedTurn(index=i, type=ttype, atc_text=atc_text, item=item, correct_value=correct, wrong_value=wrong))

        elif ttype == "state_probe":
            atc_text = raw_turn["atc"].format(**ctx)
            turns.append(RenderedTurn(index=i, type=ttype, atc_text=atc_text, state_field=raw_turn["item"]))

        else:
            raise ValueError(f"unknown turn type: {ttype}")

    return RenderedScenario(
        id=scenario["id"],
        difficulty=scenario.get("difficulty", 1),
        airport=scenario.get("airport", ""),
        title=scenario.get("title", scenario["id"]),
        callsign=ctx["callsign"],
        ctx=ctx,
        turns=turns,
        seed=seed,
    )
