"""Unit tests for tennis_metrics on synthetic pose tracks (no video needed)."""

import numpy as np

from split_step.detector import (
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP,
    LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE,
)
from split_step.pose_track import PoseTrack
from split_step.tennis_metrics import (
    coaching_tips, knee_angle, session_summary, split_step_quality, stance_ratio,
)


def _standing_track(n=90, fps=30.0, hop_at=None, knee_bend=0.0, feet_apart=0.5):
    """A player standing (shoulder width 0.4 m). Optionally hops at frame
    hop_at (hip image y rises by 0.05) and lands with bent knees."""
    world = np.zeros((n, 33, 4), np.float32)
    image = np.zeros((n, 33, 3), np.float32)
    world[..., 3] = image[..., 2] = 1.0  # all visible
    for f in range(n):
        w = world[f]
        w[LEFT_SHOULDER, :3]  = (-0.2, -0.5, 0)
        w[RIGHT_SHOULDER, :3] = ( 0.2, -0.5, 0)
        w[LEFT_HIP, :3]       = (-0.1,  0.0, 0)
        w[RIGHT_HIP, :3]      = ( 0.1,  0.0, 0)
        knee_x = (0.1 + feet_apart / 2) / 2  # midway between hip and ankle => straight leg
        w[LEFT_KNEE, :3]      = (-knee_x, 0.45, knee_bend)
        w[RIGHT_KNEE, :3]     = ( knee_x, 0.45, knee_bend)
        w[LEFT_ANKLE, :3]     = (-feet_apart / 2, 0.9, 0)
        w[RIGHT_ANKLE, :3]    = ( feet_apart / 2, 0.9, 0)
        hip_y = 0.6
        if hop_at is not None and hop_at - 3 <= f <= hop_at + 3:
            hip_y -= 0.05 * (1 - abs(f - hop_at) / 4)
        if hop_at is not None and hop_at + 4 <= f <= hop_at + 8:
            hip_y += 0.02  # landing load
        image[f, LEFT_SHOULDER, :2]  = (0.45, hip_y - 0.25)
        image[f, RIGHT_SHOULDER, :2] = (0.55, hip_y - 0.25)
        image[f, LEFT_HIP, :2]       = (0.47, hip_y)
        image[f, RIGHT_HIP, :2]      = (0.53, hip_y)
    ts = np.arange(n) * 1000 / fps
    return PoseTrack(ts_ms=ts, world=world, image=image, aspect=1.0)


def test_straight_vs_bent_knees():
    assert knee_angle(_standing_track(), 0) > 175
    assert knee_angle(_standing_track(knee_bend=0.25), 0) < 135


def test_stance_ratio():
    assert abs(stance_ratio(_standing_track(feet_apart=0.4), 0) - 1.0) < 0.01
    assert abs(stance_ratio(_standing_track(feet_apart=0.8), 0) - 2.0) < 0.01


def test_split_step_quality_measures_hop():
    t = _standing_track(hop_at=45)
    q = split_step_quality(t, t.ts_ms[45])
    assert 0.15 < q["hop_height"] < 0.25        # 0.05 rise / 0.25 torso = 0.2
    assert q["landing_ms"] > t.ts_ms[45]
    assert q["landing_knee_angle"] is not None


def test_missing_pose_gives_none():
    t = _standing_track()
    t.world[:] = np.nan
    t.image[:] = np.nan
    q = split_step_quality(t, t.ts_ms[45])
    assert all(v is None for v in q.values())


def test_session_summary_and_tips():
    t = _standing_track(n=300)
    events = [{"timestamp_ms": ms, "hop_height": 0.01, "landing_knee_angle": 170.0,
               "landing_stance_ratio": 0.8} for ms in (1000, 2000, 9000)]
    s = session_summary(t, events)
    assert s["split_steps_per_minute"] > 0
    assert s["longest_gap_s"] == 7.0
    tips = " ".join(s["tips"])
    assert "Bend your knees" in tips and "wider" in tips and "very small" in tips and "stretch" in tips


def test_no_events_tip():
    t = _standing_track()
    assert "No split steps" in session_summary(t, [])["tips"][0]


def test_export_skeleton():
    from split_step.pose_track import SKELETON_EDGES, SKELETON_LANDMARKS
    t = _standing_track(n=5)
    t.world[2] = np.nan                       # frame without a pose
    t.image[3, LEFT_ANKLE, 2] = 0.1           # invisible landmark
    sk = t.export_skeleton()
    assert len(sk["xy"]) == len(sk["ts_ms"]) == 5
    assert sk["xy"][2] is None
    assert len(sk["xy"][0]) == 2 * len(SKELETON_LANDMARKS)
    j = SKELETON_LANDMARKS.index(LEFT_ANKLE)
    assert sk["xy"][3][2 * j:2 * j + 2] == [-1, -1]
    assert all(max(e) < len(SKELETON_LANDMARKS) for e in SKELETON_EDGES)


def test_export_skeleton_is_capped():
    t = _standing_track(n=100)
    sk = t.export_skeleton(max_frames=30)
    assert len(sk["xy"]) == len(sk["ts_ms"]) <= 30
    assert sk["ts_ms"] == sorted(sk["ts_ms"])
