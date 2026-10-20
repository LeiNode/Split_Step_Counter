"""Ready stance detection and recovery timing on synthetic poses."""

import numpy as np

from split_step.detector import LEFT_WRIST, RIGHT_WRIST, LEFT_KNEE, RIGHT_KNEE
from split_step.ready_position import is_ready, recovery
from tests.test_strokes import _body, _track


def _ready(w, frames):
    """Bend the knees, spread the feet and bring both hands together in front at waist height."""
    w[frames, LEFT_KNEE, 2] = w[frames, RIGHT_KNEE, 2] = 0.2
    w[frames, LEFT_WRIST, :3] = (-0.05, -0.25, -0.3)
    w[frames, RIGHT_WRIST, :3] = (0.05, -0.25, -0.3)


def test_is_ready():
    w = _body(10)
    w[:, 27, 0], w[:, 28, 0] = -0.3, 0.3          # feet apart
    _ready(w, slice(5, 10))
    t = _track(w)
    assert not is_ready(t, 0)                     # straight legs, arms at sides
    assert is_ready(t, 7)


def test_follow_through_is_not_ready():
    w = _body(3)
    w[:, 27, 0], w[:, 28, 0] = -0.3, 0.3
    _ready(w, slice(0, 3))
    w[:, LEFT_WRIST, 1] = w[:, RIGHT_WRIST, 1] = -0.45   # hands together up by the shoulder
    assert not is_ready(_track(w), 1)


def test_recovery_time():
    w = _body(150)                                # 5 s clip
    w[:, 27, 0], w[:, 28, 0] = -0.3, 0.3
    _ready(w, slice(60, 150))                     # ready from 2.0 s
    t = _track(w)
    swings = [{"timestamp_ms": 1000.0}, {"timestamp_ms": 3500.0}]
    s = recovery(t, swings)
    assert swings[0]["recovery_ms"] == 1000.0     # ready 1 s after the first swing
    assert swings[1]["recovery_ms"] == 400.0      # already ready once the follow-through window ends
    assert s["recovered"] == 2 and s["median_recovery_ms"] == 700.0
