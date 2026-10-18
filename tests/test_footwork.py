"""Step counting on synthetic ankle motion."""

import numpy as np

from split_step.detector import LEFT_ANKLE, RIGHT_ANKLE
from split_step.footwork import footwork_report, step_times
from tests.test_strokes import _body, _track


def _lift(w, foot, start, frames=6, height=0.08):
    w[start:start + frames, foot, 1] -= height * np.sin(np.linspace(0, np.pi, frames))


def test_alternating_steps_counted():
    w = _body(90)
    for i, f in enumerate(range(10, 80, 10)):                 # 7 alternating lifts, 1 every 333 ms
        _lift(w, LEFT_ANKLE if i % 2 == 0 else RIGHT_ANKLE, f)
    assert len(step_times(_track(w))) == 7


def test_split_step_and_stance_offset_are_not_steps():
    w = _body(90)
    _lift(w, LEFT_ANKLE, 30); _lift(w, RIGHT_ANKLE, 30)       # both feet together = split step
    w[50:, LEFT_ANKLE, 1] -= 0.08                             # staggered stance held, not a lift
    assert len(step_times(_track(w))) <= 1                    # the stance change may register once


def test_steps_between_shots_same_rally_only():
    w = _body(300)
    for i, f in enumerate(range(45, 85, 10)):                 # 4 steps at 1.5-2.5 s, between shots
        _lift(w, LEFT_ANKLE if i % 2 == 0 else RIGHT_ANKLE, f)
    swings = [{"timestamp_ms": 1000.0}, {"timestamp_ms": 3000.0}, {"timestamp_ms": 9500.0}]
    r = footwork_report(_track(w), swings)
    assert [s["steps_before"] for s in swings] == [None, 4, None]   # 6.5 s gap = new rally
    assert r["steps"] == 4 and r["median_steps_between_shots"] == 4.0
