"""
strokes.py — swing detection and forehand / backhand / serve training feedback.

Works on a PoseTrack (see pose_track.py). Pipeline:

1. detect_swings(): peaks in the hitting wrist's speed. World landmarks are
   hip-relative, so running around the court doesn't register as a swing.
   A peak lands just *after* contact (the arm is still accelerating through
   the ball), so "contact" metrics look slightly before the peak.
2. Each swing gets a stroke type. In a drill ("forehand", "backhand",
   "serve") the player has said what they're hitting, so every swing is that
   stroke. In "auto" it is classified from body-relative geometry:
     - serve:     toss arm or hitting arm clearly above the head
     - forehand:  hitting wrist travels across the body from its own side
     - backhand:  hitting wrist travels across from the other side
   Forehand/backhand auto-classification is EXPERIMENTAL: it was calibrated
   by eye on a handful of pro clips and is unreliable from behind the player
   (MediaPipe's depth estimate is weakest there). Prefer drills.
3. swing_metrics(): per-stroke coaching metrics; stroke_report() summarises
   them and turns them into tips.

Calibration notes (2026-10-17, pro practice/match clips, 25-30 fps): real
swings peak at 7-18 m/s wrist speed against a 2-3 m/s noise floor; serves
put the toss arm 0.4-0.8 m or the hitting arm 0.4+ m above the nose, while
high groundstroke finishes stay under ~0.35 m. Super-slow-motion footage
breaks the speed thresholds and isn't supported.
"""

import numpy as np

from .detector import (
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST,
    LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE,
)
from .pose_track import PoseTrack
from .tennis_metrics import knee_angle

NOSE = 0
STROKES = ("forehand", "backhand", "serve")
DRILLS  = ("auto",) + STROKES
HANDS   = ("auto", "right", "left")

# ── Detection thresholds (see calibration notes above) ────────────────────────
SWING_MIN_SPEED   = 6.5    # m/s, hitting-wrist peak speed
SWING_MAX_SPEED   = 30.0   # m/s, faster is a tracking glitch, not a swing
SWING_MIN_GAP_MS  = 900    # two swings can't peak closer than this
SWING_VIS_WINDOW  = (-500, 200)
SWING_MIN_VISIBLE = 0.6    # fraction of frames in that window with shoulder + elbow visible
SERVE_TOSS_ABOVE  = 0.30   # m above nose for the non-hitting (toss) wrist
SERVE_HIT_ABOVE   = 0.40   # m above nose for the hitting wrist
FH_MAX_DELTA      = -0.10  # lateral travel (m, + = toward hitting side) for a forehand
BH_MIN_DELTA      = 0.15   # ... and for a backhand
CONTACT_LEAD_MS   = 130    # contact happens this long before the wrist-speed peak

# ── Coaching heuristics ───────────────────────────────────────────────────────
TIP_CONTACT_LOW     = 0.10  # contact height below this (0 = hips, 1 = shoulders)
TIP_CONTACT_HIGH    = 1.10
TIP_SHOULDER_TURN   = 35    # degrees of shoulder rotation through the stroke
TIP_GROUND_KNEE     = 160   # deepest knee angle while loading
TIP_SPEED_CV        = 0.30  # swing-speed coefficient of variation
TIP_SERVE_ELBOW     = 150   # elbow angle at contact
TIP_TROPHY_KNEE     = 155   # deepest knee bend before a serve
TIP_TOSS_HEIGHT     = 0.25  # m above nose


def _arm(hand: str):
    """(shoulder, elbow, wrist) landmark ids of the hitting arm and the other wrist."""
    if hand == "left":
        return LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST, RIGHT_WRIST
    return RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST, LEFT_WRIST


def wrist_speed(track: PoseTrack, wrist: int) -> np.ndarray:
    """(N,) wrist speed in m/s (world coords), 3-frame median filtered; NaN where unknown."""
    p = track.world[:, wrist, :3].astype(float)
    p[~(track.world[:, wrist, 3] >= 0.5)] = np.nan
    raw = np.full(len(track), np.nan)
    if len(track) > 1:
        dt = np.maximum(np.diff(track.ts_ms) / 1000.0, 1e-3)
        raw[1:] = np.linalg.norm(np.diff(p, axis=0), axis=1) / dt
    out = np.full_like(raw, np.nan)
    for i in range(1, len(raw) - 1):
        win = raw[i - 1:i + 2]
        if np.isfinite(win).sum() >= 2:
            out[i] = np.nanmedian(win)
    return out


def detect_hand(track: PoseTrack) -> str:
    """Guess the hitting hand: the wrist with the faster top-end speeds."""
    fast = []
    for wrist in (RIGHT_WRIST, LEFT_WRIST):
        v = wrist_speed(track, wrist)
        v = v[np.isfinite(v)]
        fast.append(np.percentile(v, 98) if len(v) else 0.0)
    return "left" if fast[1] > fast[0] * 1.1 else "right"


def detect_swings(track: PoseTrack, hand: str) -> list[int]:
    """Frame indices of swing peaks for the given hitting hand."""
    shoulder, elbow, wrist, _ = _arm(hand)
    v = wrist_speed(track, wrist)
    cand = [i for i in range(1, len(v) - 1)
            if np.isfinite(v[i]) and SWING_MIN_SPEED <= v[i] <= SWING_MAX_SPEED
            and v[i] >= np.nanmax(v[max(0, i - 3):i + 4])]
    chosen: list[int] = []
    for i in sorted(cand, key=lambda i: -v[i]):           # strongest first
        if any(abs(track.ts_ms[i] - track.ts_ms[j]) < SWING_MIN_GAP_MS for j in chosen):
            continue
        win = track.window(track.ts_ms[i] + SWING_VIS_WINDOW[0], track.ts_ms[i] + SWING_VIS_WINDOW[1])
        # shoulder + elbow, not wrist: the wrist often blurs out of tracking mid-swing
        vis = np.mean([track.visible(k, (shoulder, elbow)) for k in win]) if len(win) else 0
        if vis >= SWING_MIN_VISIBLE:                       # rejects players entering frame / cuts
            chosen.append(i)
    return sorted(chosen)


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _frames(track, t, lo, hi):
    return track.window(t + lo, t + hi)


def _height_above_nose(track, k, j):
    if not track.visible(k, (NOSE, j)):
        return np.nan
    return float(track.world[k, NOSE, 1] - track.world[k, j, 1])  # y points down


def _lateral(track, k, j, hand):
    """Sideways position of landmark j relative to the shoulder centre,
    + toward the hitting-hand side."""
    if not track.visible(k, (LEFT_SHOULDER, RIGHT_SHOULDER, j)):
        return np.nan
    ls, rs = track.world[k, LEFT_SHOULDER, :3], track.world[k, RIGHT_SHOULDER, :3]
    axis = (rs - ls) / (np.linalg.norm(rs - ls) + 1e-9)
    if hand == "left":
        axis = -axis
    return float(np.dot(track.world[k, j, :3] - (ls + rs) / 2, axis))


def _nanmax(vals):
    vals = [v for v in vals if np.isfinite(v)]
    return max(vals) if vals else np.nan


def _nanmedian(vals):
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.median(vals)) if vals else np.nan


def _angle(a, b, c) -> float:
    v1, v2 = a - b, c - b
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def classify_swing(track: PoseTrack, i: int, hand: str) -> tuple[str, float]:
    """(stroke, confidence 0-1) for the swing peaking at frame i. Stroke may be
    "unknown" when the geometry is ambiguous."""
    t = track.ts_ms[i]
    _, _, wrist, other = _arm(hand)
    pre = _frames(track, t, -900, 150)
    toss = _nanmax([_height_above_nose(track, k, other) for k in pre])
    hit = _nanmax([_height_above_nose(track, k, wrist) for k in pre])
    if (np.isfinite(toss) and toss > SERVE_TOSS_ABOVE) or (np.isfinite(hit) and hit > SERVE_HIT_ABOVE):
        margin = max(toss - SERVE_TOSS_ABOVE if np.isfinite(toss) else -1,
                     hit - SERVE_HIT_ABOVE if np.isfinite(hit) else -1)
        return "serve", round(min(1.0, 0.6 + margin), 2)

    before = _nanmedian([_lateral(track, k, wrist, hand) for k in _frames(track, t, -450, -150)])
    after = _nanmedian([_lateral(track, k, wrist, hand) for k in _frames(track, t, 100, 350)])
    if not (np.isfinite(before) and np.isfinite(after)):
        return "unknown", 0.0
    delta = after - before
    if delta <= FH_MAX_DELTA:
        return "forehand", round(min(1.0, 0.5 + (FH_MAX_DELTA - delta)), 2)
    if delta >= BH_MIN_DELTA:
        return "backhand", round(min(1.0, 0.5 + (delta - BH_MIN_DELTA)), 2)
    return "unknown", 0.0


def swing_metrics(track: PoseTrack, i: int, stroke: str, hand: str, speed: float) -> dict:
    """Coaching metrics for one swing. Values are None when not measurable."""
    t = track.ts_ms[i]
    shoulder, elbow, wrist, other = _arm(hand)
    # The wrist keeps accelerating upward through the ball, so the speed peak
    # trails contact by roughly 4 frames at 30 fps.
    contact = int(np.argmin(np.abs(track.ts_ms - (t - CONTACT_LEAD_MS))))
    m = {"peak_wrist_speed": round(float(speed), 1)}

    # Deepest knee bend while loading
    knees = [knee_angle(track, k) for k in _frames(track, t, -1200 if stroke == "serve" else -600, 0)]
    knees = [k for k in knees if k is not None]
    load_knee = round(min(knees), 1) if knees else None

    if stroke == "serve":
        reach = _nanmax([_height_above_nose(track, k, wrist) for k in _frames(track, t, -300, 100)])
        toss = _nanmax([_height_above_nose(track, k, other) for k in _frames(track, t, -1500, 0)])
        elbows = [_angle(*(track.world[k, j, :3] for j in (shoulder, elbow, wrist)))
                  for k in _frames(track, t, -300, 100) if track.visible(k, (shoulder, elbow, wrist))]
        m.update({
            "contact_reach_m": None if not np.isfinite(reach) else round(reach, 2),
            "toss_height_m": None if not np.isfinite(toss) else round(toss, 2),
            "elbow_extension": round(max(elbows), 1) if elbows else None,
            "trophy_knee_angle": load_knee,
        })
        return m

    # Groundstrokes: contact height 0 = hips, 1 = shoulders
    contact_h = None
    if track.visible(contact, (LEFT_HIP, RIGHT_HIP, LEFT_SHOULDER, RIGHT_SHOULDER, wrist)):
        w = track.world[contact]
        hip_y = (w[LEFT_HIP, 1] + w[RIGHT_HIP, 1]) / 2
        sh_y = (w[LEFT_SHOULDER, 1] + w[RIGHT_SHOULDER, 1]) / 2
        if hip_y - sh_y > 1e-3:
            contact_h = round(float((hip_y - w[wrist, 1]) / (hip_y - sh_y)), 2)

    # Shoulder rotation through the stroke (heading of the shoulder line on the ground plane)
    headings = []
    for k in _frames(track, t, -800, 300):
        if track.visible(k, (LEFT_SHOULDER, RIGHT_SHOULDER)):
            d = track.world[k, RIGHT_SHOULDER, [0, 2]] - track.world[k, LEFT_SHOULDER, [0, 2]]
            headings.append(np.arctan2(d[1], d[0]))
    turn = None
    if len(headings) >= 3:
        h = np.degrees(np.unwrap(headings))
        turn = round(float(h.max() - h.min()), 1)

    # Follow-through: how far above the shoulders the hitting wrist finishes
    finish = _nanmax([float(track.world[k, shoulder, 1] - track.world[k, wrist, 1])
                      for k in _frames(track, t, 0, 600) if track.visible(k, (shoulder, wrist))])
    m.update({
        "contact_height": contact_h,
        "shoulder_turn_deg": turn,
        "finish_above_shoulder_m": None if not np.isfinite(finish) else round(finish, 2),
        "load_knee_angle": load_knee,
    })
    return m


def _median(swings, key):
    vals = [s[key] for s in swings if s.get(key) is not None]
    return float(np.median(vals)) if vals else None


def _summary(swings) -> dict:
    keys = sorted({k for s in swings for k, v in s.items()
                   if isinstance(v, (int, float)) and k not in ("timestamp_ms", "confidence")})
    out = {"count": len(swings)}
    for k in keys:
        vals = np.array([s[k] for s in swings if s.get(k) is not None], float)
        if len(vals):
            out[k] = {"mean": round(float(vals.mean()), 2), "median": round(float(np.median(vals)), 2),
                      "min": round(float(vals.min()), 2), "max": round(float(vals.max()), 2)}
    return out


def stroke_tips(stroke: str, swings: list[dict]) -> list[str]:
    """Heuristic coaching tips for one stroke type."""
    if not swings:
        return []
    name = stroke.capitalize()
    tips = []
    speeds = np.array([s["peak_wrist_speed"] for s in swings], float)
    if len(speeds) >= 4 and speeds.std() / speeds.mean() > TIP_SPEED_CV:
        tips.append(f"{name}: swing speed varies a lot between shots. Aim for a repeatable rhythm.")
    if stroke == "serve":
        e, k, t = (_median(swings, x) for x in ("elbow_extension", "trophy_knee_angle", "toss_height_m"))
        if e is not None and e < TIP_SERVE_ELBOW:
            tips.append(f"Serve: reach up and fully extend your hitting arm at contact (elbow ~{e:.0f}°).")
        if k is not None and k > TIP_TROPHY_KNEE:
            tips.append("Serve: bend your knees more in the trophy position to drive up into the ball.")
        if t is not None and t < TIP_TOSS_HEIGHT:
            tips.append("Serve: toss a little higher so you can hit at full extension.")
    else:
        c, turn, f, k = (_median(swings, x) for x in
                         ("contact_height", "shoulder_turn_deg", "finish_above_shoulder_m", "load_knee_angle"))
        if c is not None and c < TIP_CONTACT_LOW:
            tips.append(f"{name}: you're making contact low. Bend your knees and get down to the ball.")
        if c is not None and c > TIP_CONTACT_HIGH:
            tips.append(f"{name}: contact is above shoulder height. Step back or take it earlier.")
        if turn is not None and turn < TIP_SHOULDER_TURN:
            tips.append(f"{name}: turn your shoulders more in the preparation (≈{turn:.0f}° now).")
        if f is not None and f < 0:
            tips.append(f"{name}: finish your follow-through higher, over the shoulder.")
        if k is not None and k > TIP_GROUND_KNEE:
            tips.append(f"{name}: load your legs. Your knees stay almost straight.")
    if not tips:
        tips.append(f"{name}: technique metrics look solid. Keep it up!")
    return tips


def stroke_report(track: PoseTrack, drill: str = "auto", hand: str = "auto") -> dict:
    """Detect, label and score every swing in the track.

    drill — "forehand" / "backhand" / "serve": every swing is that stroke.
            "auto": classify each swing (forehand/backhand experimental).
    hand  — "right" / "left", or "auto" to guess from wrist speeds.
    """
    if drill not in DRILLS:
        raise ValueError(f"drill must be one of {DRILLS}, got {drill!r}")
    if hand not in HANDS:
        raise ValueError(f"hand must be one of {HANDS}, got {hand!r}")
    if len(track) < 3:
        return {"drill": drill, "hand": hand, "swings": [], "summary": {}, "tips": []}
    if hand == "auto":
        hand = detect_hand(track)

    _, _, wrist, _ = _arm(hand)
    speed = wrist_speed(track, wrist)
    swings = []
    for i in detect_swings(track, hand):
        if drill == "auto":
            stroke, conf = classify_swing(track, i, hand)
        else:
            stroke, conf = drill, 1.0
        s = {"timestamp_ms": round(float(track.ts_ms[i]), 1), "stroke": stroke, "confidence": conf}
        s.update(swing_metrics(track, i, stroke if stroke != "unknown" else "forehand", hand, speed[i]))
        swings.append(s)

    by_type = {st: [s for s in swings if s["stroke"] == st] for st in STROKES + ("unknown",)}
    summary = {st: _summary(v) for st, v in by_type.items() if v}
    tips = [tip for st in STROKES for tip in stroke_tips(st, by_type[st])]
    if not swings:
        tips.append("No swings detected. Make sure the hitting arm stays in frame.")
    return {"drill": drill, "hand": hand, "swings": swings, "summary": summary, "tips": tips}


class LiveSwingCounter:
    """
    Online swing counter for live drills (desktop GUI / webcam).

    Same detection rule as detect_swings(), but streaming: a speed peak is
    confirmed once LOOKAHEAD later frames show it was the local maximum, so
    a swing registers ~100 ms after it happens. Every swing counts as the
    drill's stroke (no classification live).

    clock() -> seconds, as for the split-step counters.
    """

    LOOKAHEAD = 3

    def __init__(self, hand: str = "right", clock=None):
        import time
        self._clock = clock or time.time
        self.hand = hand
        self._shoulder, self._elbow, self._wrist, _ = _arm(hand)
        self.count = 0
        self._prev = None          # (t, wrist xyz) of the previous visible frame
        self._raw: list[float] = []
        self._speeds: list[tuple[float, float, bool]] = []   # (t, filtered speed, arm visible)
        self._last_swing_t = -1e9

    def process(self, world_landmarks) -> bool:
        """Feed one frame (pose_world_landmarks). True when a swing was just confirmed."""
        t = self._clock()
        sh, el, wr = (world_landmarks[j] for j in (self._shoulder, self._elbow, self._wrist))
        arm_visible = sh.visibility >= 0.5 and el.visibility >= 0.5
        speed = np.nan
        if wr.visibility >= 0.5:
            p = np.array([wr.x, wr.y, wr.z])
            if self._prev is not None and t > self._prev[0]:
                speed = float(np.linalg.norm(p - self._prev[1]) / (t - self._prev[0]))
            self._prev = (t, p)
        else:
            self._prev = None
        self._raw = (self._raw + [speed])[-3:]
        finite = [v for v in self._raw if np.isfinite(v)]
        filtered = float(np.median(finite)) if len(finite) >= 2 else np.nan
        self._speeds = (self._speeds + [(t, filtered, arm_visible)])[-(2 * self.LOOKAHEAD + 1 + 15):]

        # Candidate = the frame LOOKAHEAD back; compare with its neighbours on both sides
        if len(self._speeds) < 2 * self.LOOKAHEAD + 1:
            return False
        ct, cv, _ = self._speeds[-1 - self.LOOKAHEAD]
        around = [v for _, v, _ in self._speeds[-(2 * self.LOOKAHEAD + 1):]]
        if not (np.isfinite(cv) and SWING_MIN_SPEED <= cv <= SWING_MAX_SPEED
                and cv >= np.nanmax(around) and (ct - self._last_swing_t) * 1000 >= SWING_MIN_GAP_MS):
            return False
        vis = np.mean([v for _, _, v in self._speeds])
        if vis < SWING_MIN_VISIBLE:
            return False
        self._last_swing_t = ct
        self.count += 1
        return True

    def reset(self):
        self.__init__(self.hand, self._clock)
