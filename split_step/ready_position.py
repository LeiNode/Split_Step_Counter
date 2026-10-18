"""
ready_position.py — recovery to a ready stance between shots.

A frame is a "ready stance" when:
  - knees are flexed (mean knee angle < READY_MAX_KNEE)
  - feet are at least shoulder-width-ish apart (stance ratio >= READY_MIN_STANCE)
  - both hands are together in front of the body (wrists within
    READY_MAX_WRIST_GAP of each other, at least READY_MIN_HAND_DROP below the
    shoulders and not far below the hips): racket held with both hands, the
    classic ready position. The drop rule matters: a forehand finish also
    brings the hands together, but up by the shoulder (~0.07 m below it, vs
    0.25-0.37 m in a real ready stance on the calibration footage).

After each swing we look for the first run of READY_MIN_FRAMES consecutive
ready frames, starting RECOVERY_START_MS after the swing (the follow-through
itself is never a ready stance). The time to get there is the recovery time;
if it doesn't happen before the next swing, that shot wasn't prepared from a
ready stance.

Thresholds are coaching heuristics checked by eye on pro practice footage.
"""

import numpy as np

from .detector import (
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_WRIST, RIGHT_WRIST, LEFT_HIP, RIGHT_HIP,
)
from .pose_track import PoseTrack
from .tennis_metrics import knee_angle, stance_ratio

READY_MAX_KNEE      = 165    # degrees
READY_MIN_STANCE    = 0.9    # ankle spread / shoulder width
READY_MAX_WRIST_GAP = 0.35   # metres between the wrists
READY_MIN_HAND_DROP = 0.15   # metres the hands must sit below the shoulder line
READY_MIN_FRAMES    = 4      # ~130 ms at 30 fps, so a single noisy frame doesn't count
RECOVERY_START_MS   = 400    # skip the follow-through
RECOVERY_MAX_MS     = 4000   # stop looking after this long
TIP_SLOW_RECOVERY_S = 1.5
TIP_READY_FRACTION  = 0.5

_HANDS = (LEFT_WRIST, RIGHT_WRIST, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP)


def is_ready(track: PoseTrack, k: int) -> bool:
    """Whether frame k shows a ready stance (False when not measurable)."""
    if not track.visible(k, _HANDS):
        return False
    knee = knee_angle(track, k)
    stance = stance_ratio(track, k)
    if knee is None or stance is None or knee >= READY_MAX_KNEE or stance < READY_MIN_STANCE:
        return False
    w = track.world[k]
    if np.linalg.norm(w[LEFT_WRIST, :3] - w[RIGHT_WRIST, :3]) > READY_MAX_WRIST_GAP:
        return False
    hip_y = (w[LEFT_HIP, 1] + w[RIGHT_HIP, 1]) / 2
    sh_y = (w[LEFT_SHOULDER, 1] + w[RIGHT_SHOULDER, 1]) / 2
    wrist_y = (w[LEFT_WRIST, 1] + w[RIGHT_WRIST, 1]) / 2
    return bool(sh_y + READY_MIN_HAND_DROP <= wrist_y <= hip_y + 0.1)   # y points down


def recovery(track: PoseTrack, swings: list[dict]) -> dict:
    """Annotates swings with recovery_ms (None = no ready stance before the next
    shot) in place; returns a summary with tips."""
    times = [s["timestamp_ms"] for s in swings]
    rec = []
    for n, s in enumerate(swings):
        start = s["timestamp_ms"] + RECOVERY_START_MS
        end = min(s["timestamp_ms"] + RECOVERY_MAX_MS, times[n + 1] if n + 1 < len(times) else np.inf)
        run, found = 0, None
        for k in track.window(start, end):
            run = run + 1 if is_ready(track, k) else 0
            if run == READY_MIN_FRAMES:
                found = track.ts_ms[k - READY_MIN_FRAMES + 1] - s["timestamp_ms"]
                break
        s["recovery_ms"] = None if found is None else round(float(found), 1)
        # The last swing has no "next shot", so it only counts when it did recover
        if n + 1 < len(swings) or found is not None:
            rec.append(found)

    got = [r for r in rec if r is not None]
    summary = {
        "shots_measured": len(rec),
        "recovered": len(got),
        "fraction_recovered": round(len(got) / len(rec), 3) if rec else None,
        "median_recovery_ms": round(float(np.median(got)), 1) if got else None,
        "tips": [],
    }
    if len(rec) >= 3:
        if summary["fraction_recovered"] < TIP_READY_FRACTION:
            summary["tips"].append("After hitting, get back to a ready position (knees bent, racket in "
                                   "front with both hands, feet apart) before the next ball.")
        elif summary["median_recovery_ms"] / 1000 > TIP_SLOW_RECOVERY_S:
            summary["tips"].append(f"Recover faster: you take about {summary['median_recovery_ms'] / 1000:.1f} s "
                                   "to get back to a ready position after each shot.")
    return summary
