"""
camera_view.py — which way the camera sees the player, and whether that
hurts accuracy.

The split-step LSTM was trained mostly on back-view footage (see TODO.md,
"Camera-angle imbalance"); it is measurably weaker from the side and the
front. This classifies every frame so users can be told when their video
is one of the weaker cases.

Per frame (shoulders visible):
  side   — the shoulder line points mostly toward the camera: its sideways
           share |dx| / |(dx, dz)| (world coords) is below SIDE_MAX_SQUARENESS
  front  — shoulders square to the camera and the player's LEFT shoulder
           appears on the RIGHT of the image (they're facing the camera)
  back   — shoulders square, left shoulder on the image's left

Calibrated by eye on six pro clips (2026-10-17): side-view practice 0.49
squareness; back and front views 0.63-0.83, separated by the image-side cue
(7-20% vs 68-76% of frames).
"""

import numpy as np

from .detector import LEFT_SHOULDER, RIGHT_SHOULDER
from .pose_track import PoseTrack

VIEWS = ("back", "front", "side")
SIDE_MAX_SQUARENESS = 0.55
TIP_WEAK_VIEW_FRACTION = 0.5


def frame_views(track: PoseTrack) -> np.ndarray:
    """(N,) array of "back" / "front" / "side", or "" where the shoulders aren't visible."""
    out = np.full(len(track), "", dtype=object)
    for k in range(len(track)):
        if not track.visible(k, (LEFT_SHOULDER, RIGHT_SHOULDER)):
            continue
        d = track.world[k, RIGHT_SHOULDER, :3] - track.world[k, LEFT_SHOULDER, :3]
        squareness = abs(d[0]) / (np.hypot(d[0], d[2]) + 1e-9)
        if squareness < SIDE_MAX_SQUARENESS:
            out[k] = "side"
        elif track.image[k, LEFT_SHOULDER, 0] > track.image[k, RIGHT_SHOULDER, 0]:
            out[k] = "front"
        else:
            out[k] = "back"
    return out


def camera_report(track: PoseTrack) -> dict:
    """{view, fractions, tips}: the dominant view, each view's share of frames
    with visible shoulders, and a warning when weak views dominate."""
    views = frame_views(track)
    seen = views[views != ""]
    if len(seen) == 0:
        return {"view": None, "fractions": {}, "tips": []}
    fractions = {v: round(float(np.mean(seen == v)), 3) for v in VIEWS}
    view = max(fractions, key=fractions.get)
    tips = []
    weak = fractions["side"] + fractions["front"]
    if weak > TIP_WEAK_VIEW_FRACTION:
        tips.append(f"Most of this video shows the player from the {view}. The split-step model was "
                    "trained mainly on footage filmed from behind the player, so split-step counts "
                    "may be less reliable here. For best results, film from behind the baseline.")
    return {"view": view, "fractions": fractions, "tips": tips}
