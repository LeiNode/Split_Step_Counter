"""Linking split steps to the shots that follow them."""

from split_step import features
from split_step.timing import link_split_steps


def _ev(*ts):
    return [{"timestamp_ms": t} for t in ts]


def test_lead_time_and_during_swing():
    events = _ev(1000, 2850, 5000)          # 2850 is a hop during the 3000 swing
    swings = _ev(2000, 3000, 9000)
    s = link_split_steps(events, swings)
    assert [e["during_swing"] for e in events] == [False, True, False]
    # 2000: split step 1000 ms before; 3000: 1000 is 2000 ms before (2850 ignored); 9000: nothing in range
    assert [w["split_step_lead_ms"] for w in swings] == [1000, 2000, None]
    assert s["shots_with_split_step"] == 2 and s["split_steps_during_swings"] == 1
    assert s["fraction_with_split_step"] == round(2 / 3, 3)


def test_too_close_or_too_early_doesnt_count():
    swings = _ev(5000)
    link_split_steps(_ev(4700), swings)                 # 300 ms: during the swing / too late
    assert swings[0]["split_step_lead_ms"] is None
    swings = _ev(5000)
    link_split_steps(_ev(1000), swings)                 # 4 s: belongs to an earlier shot
    assert swings[0]["split_step_lead_ms"] is None


def test_tips():
    swings = _ev(2000, 4000, 6000, 8000)
    assert "Only 0 of 4" in link_split_steps([], swings)["tips"][0]
    swings = _ev(2000, 4000, 6000)
    assert "Good habit" in link_split_steps(_ev(1000, 3000, 5000), swings)["tips"][0]
    assert link_split_steps([], _ev(2000))["tips"] == []   # too few shots to judge


def test_timing_needs_both_parents(monkeypatch):
    monkeypatch.setenv("SPLIT_STEP_FEATURES", "stroke_training=off")
    assert features.all_features()["split_step_timing"] is False
    monkeypatch.setenv("SPLIT_STEP_FEATURES", "split_step_counting=off")
    assert features.all_features()["split_step_timing"] is False
