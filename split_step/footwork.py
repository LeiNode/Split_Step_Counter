"""
footwork.py — counts steps: "keep your feet moving".

A step is one foot lifting clearly higher than the other and coming back
down. The signal is the left-minus-right ankle height (world coords). A
staggered stance holds a steady offset between the feet for a second or
more, so a rolling median over STEP_BASELINE_MS is subtracted first; what
is left are the short pulses of actual lifts. Each pulse past STEP_ON_M
(and back under STEP_OFF_M) counts as one step for the lifted foot.

A split step lifts both feet together, so it barely moves the difference
and isn't counted as a step.

Calibrated by eye (2026-10-17) on pro practice footage: ~1 step per
250-300 ms during adjustment footwork between shots.
"""

import numpy as np

from .detector import LEFT_ANKLE, RIGHT_ANKLE
from .pose_track import PoseTrack
from .rallies import RALLY_GAP_MS

STEP_BASELINE_MS = 900    # rolling-median window that removes stance offsets
STEP_ON_M        = 0.030  # foot this much higher than its baseline = lifted
STEP_OFF_M       = 0.012  # back under this = landed (hysteresis)
ADJUST_START_MS  = 400    # after a swing (skip the follow-through)
ADJUST_END_MS    = 150    # before the next swing's peak (≈ contact)
TIP_FEW_STEPS    = 2      # median adjustment steps between shots below this -> tip
TIP_MIN_SHOTS    = 3


def _rolling_median(x: np.ndarray, ts: np.ndarray, window_ms: float) -> np.ndarray:
    out = np.full_like(x, np.nan)
    half = window_ms / 2
    lo = hi = 0
    for i, t in enumerate(ts):
        while ts[lo] < t - half:
            lo += 1
        while hi < len(ts) and ts[hi] <= t + half:
            hi += 1
        seg = x[lo:hi]
        seg = seg[np.isfinite(seg)]
        if len(seg):
            out[i] = np.median(seg)
    return out


def step_times(track: PoseTrack) -> list[float]:
    """Timestamps (ms) at which a step's lift began."""
    if len(track) == 0:
        return []
    ok = (track.world[:, LEFT_ANKLE, 3] >= 0.5) & (track.world[:, RIGHT_ANKLE, 3] >= 0.5)
    # y points down: > 0 means the LEFT ankle is higher (left foot lifted)
    diff = np.where(ok, track.world[:, RIGHT_ANKLE, 1] - track.world[:, LEFT_ANKLE, 1], np.nan)
    pulse = diff - _rolling_median(diff, track.ts_ms, STEP_BASELINE_MS)

    steps, lifted = [], 0          # lifted: 0 none, +1 left foot, -1 right foot
    for k, v in enumerate(pulse):
        if not np.isfinite(v):
            lifted = 0             # tracking gap: don't bridge a step across it
            continue
        if lifted == 0 and abs(v) >= STEP_ON_M:
            lifted = 1 if v > 0 else -1
            steps.append(float(track.ts_ms[k]))
        elif lifted and (lifted * v) <= STEP_OFF_M:
            lifted = 0
            if abs(v) >= STEP_ON_M:            # swung straight to the other foot
                lifted = 1 if v > 0 else -1
                steps.append(float(track.ts_ms[k]))
    return steps


def footwork_report(track: PoseTrack, swings: list[dict] | None = None) -> dict:
    """Steps per minute over the clip; with swings, adjustment steps between
    consecutive shots of the same rally (swings[].steps_before; None for a
    rally's first shot, where the steps would include walking between
    points) and a tip."""
    steps = np.array(step_times(track))
    duration_s = float(track.ts_ms[-1] - track.ts_ms[0]) / 1000 if len(track) > 1 else 0.0
    out = {
        "steps": int(len(steps)),
        "steps_per_minute": round(len(steps) / (duration_s / 60), 1) if duration_s > 0 else 0.0,
        "median_steps_between_shots": None,
        "tips": [],
    }
    if swings:
        between = []
        times = [s["timestamp_ms"] for s in swings]
        for n, s in enumerate(swings):
            if n == 0 or s["timestamp_ms"] - times[n - 1] > RALLY_GAP_MS:
                s["steps_before"] = None
                continue
            lo, hi = times[n - 1] + ADJUST_START_MS, s["timestamp_ms"] - ADJUST_END_MS
            s["steps_before"] = int(np.sum((steps >= lo) & (steps <= hi))) if hi > lo else 0
            between.append(s["steps_before"])
        if between:
            out["median_steps_between_shots"] = float(np.median(between))
            if len(between) >= TIP_MIN_SHOTS and np.median(between) < TIP_FEW_STEPS:
                out["tips"].append("Keep your feet moving between shots: small adjustment steps help "
                                   "you set up at the right distance from the ball.")
    return out
