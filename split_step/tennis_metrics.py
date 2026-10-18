"""
tennis_metrics.py — coaching metrics computed from a PoseTrack.

split_step_quality()  per split step: hop height, landing knee bend, stance width
session_summary()     rate, rhythm and averages over the whole video, plus tips

All thresholds behind the coaching tips are coaching heuristics, not values
fitted to labeled data. Treat them as starting points to tune.

Units: hop height is a fraction of the player's own torso length (so it
doesn't depend on camera distance); stance width is ankle spread divided by
shoulder width; angles are degrees (180 = straight leg).
"""

import numpy as np

from .detector import (
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP,
    LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE,
)
from .pose_track import PoseTrack

_TORSO = (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP)
_LEFT_LEG  = (LEFT_HIP, LEFT_KNEE, LEFT_ANKLE)
_RIGHT_LEG = (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE)
_STANCE = (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ANKLE, RIGHT_ANKLE)

# Windows around a counted split step, relative to its timestamp (ms)
_BASELINE_MS = (-700, -250)   # standing height before the hop
_PEAK_MS     = (-300, 250)    # the hop's highest point lies in here
_LOAD_MS     = 450            # landing / deepest knee bend within this after the peak

# Coaching heuristics
TIP_KNEE_ANGLE   = 155   # median landing knee angle above this -> legs too straight
TIP_STANCE_RATIO = 1.1   # median landing stance below this -> base too narrow
TIP_HOP_HEIGHT   = 0.03  # median hop below this fraction of torso -> barely leaving the ground
TIP_GAP_S        = 6.0   # a gap this long between split steps -> probably skipped one


def _hip_y(track: PoseTrack, idx) -> np.ndarray:
    """Hip-centre image y (fraction of frame height; larger = lower)."""
    return (track.image[idx, LEFT_HIP, 1] + track.image[idx, RIGHT_HIP, 1]) / 2


def _torso_len(track: PoseTrack, idx: int) -> float:
    sh = (track.image_xy(idx, LEFT_SHOULDER) + track.image_xy(idx, RIGHT_SHOULDER)) / 2
    hp = (track.image_xy(idx, LEFT_HIP) + track.image_xy(idx, RIGHT_HIP)) / 2
    return float(np.linalg.norm(sh - hp))


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle ABC in degrees."""
    v1, v2 = a - b, c - b
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def knee_angle(track: PoseTrack, idx: int) -> float | None:
    """Mean hip-knee-ankle angle of the visible legs (world coords), or None."""
    angles = [_angle(*(track.world[idx, j, :3] for j in leg))
              for leg in (_LEFT_LEG, _RIGHT_LEG) if track.visible(idx, leg)]
    return float(np.mean(angles)) if angles else None


def stance_ratio(track: PoseTrack, idx: int) -> float | None:
    """Ankle spread / shoulder width, measured on the ground plane (world x, z)."""
    if not track.visible(idx, _STANCE):
        return None
    w = track.world[idx]
    feet = np.linalg.norm(w[LEFT_ANKLE, [0, 2]] - w[RIGHT_ANKLE, [0, 2]])
    shoulders = np.linalg.norm(w[LEFT_SHOULDER, [0, 2]] - w[RIGHT_SHOULDER, [0, 2]])
    return float(feet / shoulders) if shoulders > 1e-3 else None


def _visible_frames(track: PoseTrack, idx: np.ndarray, landmarks) -> np.ndarray:
    return np.array([i for i in idx if track.visible(i, landmarks)], dtype=int)


def split_step_quality(track: PoseTrack, event_ms: float) -> dict:
    """Metrics for one split step counted at event_ms. Missing values are None
    (e.g. when the hips or legs weren't visible at the key moments)."""
    out = {"hop_height": None, "landing_knee_angle": None, "landing_stance_ratio": None,
           "landing_ms": None}

    base = _visible_frames(track, track.window(event_ms + _BASELINE_MS[0], event_ms + _BASELINE_MS[1]), _TORSO)
    peak = _visible_frames(track, track.window(event_ms + _PEAK_MS[0], event_ms + _PEAK_MS[1]), _TORSO)
    if len(peak) == 0:
        return out

    peak_i = int(peak[np.argmin(_hip_y(track, peak))])  # smallest y = highest point
    if len(base):
        torso = _torso_len(track, peak_i)
        if torso > 1e-3:
            rise = float(np.median(_hip_y(track, base)) - _hip_y(track, peak_i))
            out["hop_height"] = round(max(rise, 0.0) / torso, 3)

    # Landing = deepest point (largest hip y) shortly after the peak
    land = _visible_frames(track, track.window(track.ts_ms[peak_i], track.ts_ms[peak_i] + _LOAD_MS), _TORSO)
    if len(land):
        land_i = int(land[np.argmax(_hip_y(track, land))])
        out["landing_ms"] = round(float(track.ts_ms[land_i]), 1)
        k = knee_angle(track, land_i)
        s = stance_ratio(track, land_i)
        out["landing_knee_angle"] = None if k is None else round(k, 1)
        out["landing_stance_ratio"] = None if s is None else round(s, 2)
    return out


def _stats(values) -> dict | None:
    v = np.array([x for x in values if x is not None], dtype=float)
    if len(v) == 0:
        return None
    return {"mean": round(float(v.mean()), 3), "median": round(float(np.median(v)), 3),
            "min": round(float(v.min()), 3), "max": round(float(v.max()), 3)}


def session_summary(track: PoseTrack, events: list[dict]) -> dict:
    """Whole-video summary. events are the per-split-step dicts (timestamp_ms
    plus split_step_quality() fields)."""
    duration_s = float(track.ts_ms[-1] - track.ts_ms[0]) / 1000 if len(track) > 1 else 0.0
    present = track.present
    times_s = np.array([e["timestamp_ms"] for e in events]) / 1000
    gaps = np.diff(times_s) if len(times_s) > 1 else np.array([])

    summary = {
        "duration_s": round(duration_s, 2),
        "pose_coverage": round(float(present.mean()), 3) if len(track) else 0.0,
        "split_steps_per_minute": round(len(events) / (duration_s / 60), 2) if duration_s > 0 else 0.0,
        "interval_s": _stats(gaps.tolist()),
        "longest_gap_s": round(float(gaps.max()), 2) if len(gaps) else None,
        "hop_height": _stats([e.get("hop_height") for e in events]),
        "landing_knee_angle": _stats([e.get("landing_knee_angle") for e in events]),
        "landing_stance_ratio": _stats([e.get("landing_stance_ratio") for e in events]),
    }
    summary["tips"] = coaching_tips(summary, len(events))
    return summary


def coaching_tips(summary: dict, n_events: int) -> list[str]:
    """Plain-language suggestions from the session summary (heuristic)."""
    tips = []
    if summary["pose_coverage"] < 0.6:
        tips.append("The player was out of view for much of the video. Keep the whole "
                    "body, feet included, in frame for reliable results.")
    if n_events == 0:
        tips.append("No split steps detected. Try a small two-footed hop just as "
                    "your opponent strikes the ball.")
        return tips
    k, s, h = summary["landing_knee_angle"], summary["landing_stance_ratio"], summary["hop_height"]
    if k and k["median"] > TIP_KNEE_ANGLE:
        tips.append(f"Bend your knees more as you land (median {k['median']:.0f}°). "
                    "A loaded stance lets you push off faster.")
    if s and s["median"] < TIP_STANCE_RATIO:
        tips.append(f"Land with your feet a bit wider than your shoulders "
                    f"(currently {s['median']:.2f}× shoulder width).")
    if h and h["median"] < TIP_HOP_HEIGHT:
        tips.append("Your split steps are very small. A light, visible hop helps "
                    "with timing and balance.")
    if summary["longest_gap_s"] and summary["longest_gap_s"] > TIP_GAP_S:
        tips.append(f"There was a {summary['longest_gap_s']:.0f} s stretch without a split step. "
                    "Aim to split step before every opponent shot.")
    if not tips:
        tips.append("Solid split steps: good knee bend, base width and rhythm.")
    return tips
