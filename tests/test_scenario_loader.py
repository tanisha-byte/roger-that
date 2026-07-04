"""Tests for scenario rendering and fault injection.

The hearback_challenge corruption math lives here, not in the scorer --
this is the fault-injection ground truth itself. A bug here is worse than a
scorer bug: it means the *test* is broken, not just the grading of it.
"""
import random

from controller.phraseology import spell_digits, spell_freq
from orchestrator.scenario_loader import _apply_bounds, _corrupt, load_scenario_yaml, render_scenario, SCENARIOS_DIR


def test_heading_corruption_never_collides_with_truth_after_wraparound():
    # regression: heading=10 offset by -20 used to go to -10, and
    # spell_digits() takes abs() before speaking it, so the "corrupted"
    # value was spoken identically to the truth ("one zero" for both).
    # A pilot that caught nothing would have scored a false "caught it".
    wrong = _apply_bounds("hdg", 10 - 20)
    assert wrong != 10
    assert spell_digits(wrong) != spell_digits(10)


def test_heading_wraps_into_valid_atc_range():
    for raw in [-30, -1, 0, 361, 400]:
        wrapped = _apply_bounds("hdg", raw)
        assert 1 <= wrapped <= 360


def test_corrupt_never_collides_with_truth_across_full_domain():
    items_and_ranges = {
        "hdg": range(0, 361, 10),
        "qnh": range(984, 1042),
        "fl": range(50, 161, 10),
        "spd": range(160, 291, 10),
        "squawk": range(0, 7778, 137),
        "freq": [round(118.0 + i * 0.3, 1) for i in range(50)],
    }
    for item, values in items_and_ranges.items():
        for correct in values:
            for seed in range(10):
                wrong = _corrupt(item, correct, random.Random(seed))
                assert wrong != correct, f"{item}={correct} seed={seed} produced a colliding corruption"


def test_corrupted_frequency_spoken_form_differs_from_truth_across_seeds():
    scenario = load_scenario_yaml(SCENARIOS_DIR / "hearback-error-01.yaml")
    for seed in range(200):
        rs = render_scenario(scenario, seed=seed)
        for t in rs.turns:
            if t.type == "hearback_challenge":
                assert t.wrong_value != t.correct_value
                assert spell_freq(t.wrong_value) != spell_freq(t.correct_value)
