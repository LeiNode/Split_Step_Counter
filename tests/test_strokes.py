"""Stroke detection on synthetic swings (a right-handed player, 30 fps)."""

import numpy as np
import pytest

from split_step.detector import (
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST,
    LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE,
)
from split_step.pose_track import PoseTrack
from split_step.strokes import NOSE, detect_hand, stroke_report

FPS = 30.0


def _body(n):
    """Standing player, x = player's right, y down (metres, hip-relative)."""
    w = np.zeros((n, 33, 4), np.float32)
    w[..., 3] = 1.0
    pts = {NOSE: (0, -0.65, 0), LEFT_SHOULDER: (-0.2, -0.5, 0), RIGHT_SHOULDER: (0.2, -0.5, 0),
           LEFT_ELBOW: (-0.3, -0.25, 0), RIGHT_ELBOW: (0.3, -0.25, 0),
           LEFT_WRIST: (-0.3, -0.05, 0), RIGHT_WRIST: (0.3, -0.05, 0),
           LEFT_HIP: (-0.1, 0, 0), RIGHT_HIP: (0.1, 0, 0), LEFT_KNEE: (-0.12, 0.45, 0),
           RIGHT_KNEE: (0.12, 0.45, 0), LEFT_ANKLE: (-0.14, 0.9, 0), RIGHT_ANKLE: (0.14, 0.9, 0)}
    for j, p in pts.items():
        w[:, j, :3] = p
    return w


def _track(w):
    n = len(w)
    img = np.full((n, 33, 3), 0.5, np.float32)
    return PoseTrack(ts_ms=np.arange(n) * 1000 / FPS, world=w, image=img, aspect=1.0)


def _groundstroke(w, at, start_x, end_x, frames=6):
    """Swing the right wrist from start_x to end_x (player-lateral) centred on frame `at`."""
    w[: at - frames // 2, RIGHT_WRIST, 0] = start_x
    xs = np.linspace(start_x, end_x, frames)
    w[at - frames // 2: at + frames // 2, RIGHT_WRIST, 0] = xs
    w[at + frames // 2:, RIGHT_WRIST, 0] = end_x
    # chest-height contact and a high finish
    w[at - frames // 2: at + frames // 2, RIGHT_WRIST, 1] = np.linspace(-0.1, -0.6, frames)
    w[at + frames // 2:, RIGHT_WRIST, 1] = -0.6


def test_forehand_detected_and_classified():
    w = _body(90)
    _groundstroke(w, 45, start_x=0.6, end_x=-0.5)       # right side -> across to the left
    r = stroke_report(_track(w), "auto", "right")
    assert [s["stroke"] for s in r["swings"]] == ["forehand"]
    s = r["swings"][0]
    assert s["peak_wrist_speed"] > 6.5
    assert s["finish_above_shoulder_m"] > 0


def test_backhand_classified():
    w = _body(90)
    _groundstroke(w, 45, start_x=-0.5, end_x=0.6)       # from the left side across to the right
    assert [s["stroke"] for s in stroke_report(_track(w), "auto", "right")["swings"]] == ["backhand"]


def test_serve_from_toss_arm():
    w = _body(90)
    w[20:45, LEFT_WRIST, 1] = -1.2                       # toss arm well above the head
    _groundstroke(w, 45, start_x=0.6, end_x=-0.5)
    r = stroke_report(_track(w), "auto", "right")
    assert [s["stroke"] for s in r["swings"]] == ["serve"]
    assert r["swings"][0]["toss_height_m"] > 0.3


def test_drill_overrides_classification():
    w = _body(90)
    _groundstroke(w, 45, start_x=0.6, end_x=-0.5)
    r = stroke_report(_track(w), "backhand", "right")
    assert r["swings"][0]["stroke"] == "backhand" and "backhand" in r["summary"]


def test_no_swings_when_still():
    r = stroke_report(_track(_body(60)), "auto", "right")
    assert r["swings"] == [] and "No swings" in r["tips"][0]


def test_detect_hand():
    w = _body(90)
    w[30:40, LEFT_WRIST, 0] = np.linspace(-0.5, 0.6, 10)   # only the left arm moves fast
    assert detect_hand(_track(w)) == "left"


def test_bad_args():
    with pytest.raises(ValueError):
        stroke_report(_track(_body(10)), "volley")


def test_live_counter_matches_offline():
    from types import SimpleNamespace as NS
    from split_step.strokes import LiveSwingCounter, detect_swings
    w = _body(150)
    _groundstroke(w, 45, start_x=0.6, end_x=-0.5)
    w[100:, RIGHT_WRIST, :3] = w[0, RIGHT_WRIST, :3]          # back to the start...
    _groundstroke(w[60:], 60, start_x=0.6, end_x=-0.5)       # ...second swing at frame 120
    t = _track(w)
    now = [0.0]
    c = LiveSwingCounter("right", clock=lambda: now[0])
    for k in range(len(t)):
        now[0] = t.ts_ms[k] / 1000
        c.process([NS(x=a, y=b, z=z, visibility=v) for a, b, z, v in t.world[k]])
    assert c.count == len(detect_swings(t, "right")) == 2
    c.reset()
    assert c.count == 0 and c.hand == "right"


def test_consistency_score():
    from split_step.strokes import consistency
    same = [{"peak_wrist_speed": 10.0, "contact_height": 0.5, "shoulder_turn_deg": 120.0,
             "finish_above_shoulder_m": 0.2} for _ in range(4)]
    assert consistency("forehand", same) == 100
    wild = [{"peak_wrist_speed": v, "contact_height": c, "shoulder_turn_deg": t, "finish_above_shoulder_m": f}
            for v, c, t, f in [(5, 0.0, 40, -0.3), (15, 1.2, 170, 0.5), (6, 0.1, 60, -0.2), (16, 1.1, 160, 0.4)]]
    assert consistency("forehand", wild) < 20
    assert consistency("forehand", same[:2]) is None          # too few swings


def test_shoulder_turn_ignores_left_right_swaps():
    """A 180° flip of the shoulder line (MediaPipe swapping L/R) is not rotation."""
    from split_step.strokes import swing_metrics
    w = _body(60)
    w[30:32, LEFT_SHOULDER, 0], w[30:32, RIGHT_SHOULDER, 0] = 0.2, -0.2   # 2-frame label swap
    m = swing_metrics(_track(w), 30, "forehand", "right", 10.0)
    assert m["shoulder_turn_deg"] < 1


def test_serve_toss_lateral_and_tip():
    from split_step.strokes import stroke_tips
    w = _body(90)
    w[20:45, LEFT_WRIST, 1] = -1.2
    w[20:45, LEFT_WRIST, 0] = -0.4                        # toss out on the non-hitting side
    _groundstroke(w, 45, start_x=0.6, end_x=-0.5)
    s = stroke_report(_track(w), "serve", "right")["swings"][0]
    assert s["toss_lateral_m"] < -0.3
    wobbly = [{"peak_wrist_speed": 10.0, "toss_lateral_m": x} for x in (-0.3, 0.0, 0.3, -0.25)]
    assert any("toss release point moves around" in t for t in stroke_tips("serve", wobbly))


def test_head_movement_at_contact():
    from split_step.strokes import CONTACT_LEAD_MS, stroke_tips, swing_metrics
    w = _body(90)
    assert swing_metrics(_track(w), 45, "forehand", "right", 10.0)["head_movement_m"] < 0.01
    contact = 45 - round(CONTACT_LEAD_MS * FPS / 1000)
    w[contact - 3:contact + 4, NOSE, 0] += np.linspace(0, 0.4, 7)       # head jerks 40 cm sideways
    assert swing_metrics(_track(w), 45, "forehand", "right", 10.0)["head_movement_m"] > 0.15
    still = [{"peak_wrist_speed": 10.0, "head_movement_m": 0.05}] * 3
    moving = [{"peak_wrist_speed": 10.0, "head_movement_m": 0.25}] * 3
    assert not any("head still" in t for t in stroke_tips("forehand", still))
    assert any("head still" in t for t in stroke_tips("backhand", moving))
    assert not any("head still" in t for t in stroke_tips("backhand", moving[:2]))   # too few swings
