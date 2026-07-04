"""Tests for the live-call transcript scoring pipeline (orchestrator/live_bolna.py).

Every case here is a regression against something a real live Bolna call
actually produced -- these are not hypothetical inputs.
"""
import pytest

from orchestrator.live_bolna import NoTurnsMatchedError, score_transcript
from orchestrator.scenario_loader import load_scenario_yaml, render_scenario, SCENARIOS_DIR


def test_state_probe_turn_matched_and_graded_correctly():
    # regression: this exact real transcript used to mismatch the
    # state-probe line to turn 1 (a 3-way tie under item-only scoring,
    # broken by list order) and then, even once matched to the right turn,
    # graded it against an empty expected_values dict (always False)
    # regardless of what the pilot actually said.
    transcript = (
        "assistant: VT-FVC, standing by.\n"
        "user:  three t f v c descent flight level six zero qnh niner niner five\n"
        "assistant:  Descent flight level six zero, QNH niner niner five, VT-FVC.\n"
        "user:  vt\n"
        "user:  contact approach one three zero decimal seven\n"
        "assistant:  Contact approach one three zero decimal seven, VT-FVC.\n"
        "user:  victor tango foxtrot victor charlie confirm your assigned flight level\n"
        "assistant:  Flight level six zero, VT-FVC.\n"
        "user:  affirm\n"
        "assistant:   Affirm, flight level six zero, VT-FVC.\n"
    )
    scenario = load_scenario_yaml(SCENARIOS_DIR / "state-probe-01.yaml")
    rs = render_scenario(scenario, seed=20260703100)

    result = score_transcript(transcript, rs)

    kinds = {r["kind"] for r in [t["record"] for t in result.turn_rows]}
    assert kinds == {"readback", "state_probe"}
    state_probe_record = next(t["record"] for t in result.turn_rows if t["record"]["kind"] == "state_probe")
    assert state_probe_record.get("state_probe_correct") is True
    assert result.scorecard["score"] == 1.0


def test_decimal_with_stray_whitespace_still_extracts_frequency():
    # regression: a real reply rendered "switching to approach 127. 1."
    # (space between the decimal point and the fractional digit), which
    # broke the strict digit-adjacent decimal rule and lost the value entirely.
    transcript = (
        "assistant: VT-LEN, standing by.\n"
        "user:  victor tango lima echo november contact approach one two seven decimal one\n"
        "assistant: VT-LEN, roger, switching to approach 127. 1.\n"
    )
    scenario = load_scenario_yaml(SCENARIOS_DIR / "hearback-error-01.yaml")
    rs = render_scenario(scenario, seed=7)

    result = score_transcript(transcript, rs)
    rec = result.turn_rows[0]["record"]
    assert rec["items"]["freq"] == "ok"


def test_no_turns_matched_raises_instead_of_reporting_false_perfect_score():
    # regression: score_session([]) returns a "perfect" 1.0 by design (every
    # empty category defaults to 1.0). At the whole-session level that is a
    # false positive, not a legitimate default -- a session where literally
    # nothing matched must fail loudly, not silently report a flawless score.
    transcript = (
        "assistant: VT-ABC, standing by.\n"
        "user: completely unrelated static noise with no callsign or items\n"
        "assistant: could not parse that\n"
    )
    scenario = load_scenario_yaml(SCENARIOS_DIR / "simple-descent-01.yaml")
    rs = render_scenario(scenario, seed=1)

    with pytest.raises(NoTurnsMatchedError):
        score_transcript(transcript, rs)


def test_garbled_callsign_falls_back_to_content_matching():
    # regression: when the controller's spoken callsign is badly ASR-mangled
    # ("three t f v c" for "victor tango foxtrot victor charlie"), the
    # addressed-to signal is unrecoverable -- matching must fall back to
    # content overlap against the turn's own canonical line instead of
    # dropping an otherwise-recoverable turn.
    transcript = (
        "assistant: VT-FVC, standing by.\n"
        "user:  three t f v c descent flight level six zero qnh niner niner five\n"
        "assistant:  Descent flight level six zero, QNH niner niner five, VT-FVC.\n"
    )
    scenario = load_scenario_yaml(SCENARIOS_DIR / "state-probe-01.yaml")
    rs = render_scenario(scenario, seed=20260703100)

    result = score_transcript(transcript, rs)
    assert len(result.turn_rows) == 1
    assert result.turn_rows[0]["record"]["items"]["fl"] == "ok"
    assert result.turn_rows[0]["record"]["items"]["qnh"] == "ok"
