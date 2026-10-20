"""
annotate.py — writes a copy of the video with the analysis drawn on it.

Burned in, so the clip can be shared anywhere (messages, social media) with
no viewer needed:
  - the skeleton (from the result's "pose", i.e. include_pose=True)
  - a "SPLIT STEP" banner for BANNER_MS after each split step
  - the stroke and wrist speed for BANNER_MS after each swing
  - running counts in the corner

    python -m split_step.analyze clip.mp4 --drill auto --annotate clip_annotated.mp4
"""

import bisect

import cv2

BANNER_MS = 800
_GREEN = (136, 196, 92)        # BGR, matches the web overlay
_STROKE_BGR = {"forehand": (176, 108, 43), "backhand": (191, 79, 138), "serve": (43, 130, 217), "unknown": (143, 148, 138)}


def _nearest(ts: list[float], ms: float, max_gap: float = 100) -> int | None:
    k = bisect.bisect_left(ts, ms)
    best = min((i for i in (k - 1, k) if 0 <= i < len(ts)), key=lambda i: abs(ts[i] - ms), default=None)
    return best if best is not None and abs(ts[best] - ms) <= max_gap else None


def _label(img, text, org, color, scale):
    thick = max(1, round(2 * scale))
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = org
    cv2.rectangle(img, (x - 6, y - h - 8), (x + w + 6, y + base + 4), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def annotate_video(video_path: str, result: dict, out_path: str) -> int:
    """Draw the result onto the video; returns the number of frames written.
    Skeleton only when result has "pose"; banners need events / strokes."""
    pose = result.get("pose")
    pose_ts = pose["ts_ms"] if pose else []
    splits = sorted(e["timestamp_ms"] for e in result.get("events", []) if not e.get("during_swing"))
    swings = sorted((s["timestamp_ms"], s["stroke"], s.get("peak_wrist_speed")) for s in
                    (result.get("strokes") or {}).get("swings", []))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    scale = max(0.5, min(w, h) / 720)   # shorter side, so portrait phone video fits too
    written = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            ms = cap.get(cv2.CAP_PROP_POS_MSEC)

            k = _nearest(pose_ts, ms) if pose else None
            if k is not None and pose["xy"][k]:
                xy = pose["xy"][k]
                pt = lambda i: None if xy[2 * i] < 0 else (int(xy[2 * i] * w), int(xy[2 * i + 1] * h))
                for a, b in pose["edges"]:
                    if pt(a) and pt(b):
                        cv2.line(frame, pt(a), pt(b), _GREEN, max(2, round(3 * scale)), cv2.LINE_AA)
                for i in range(len(pose["landmarks"])):
                    if pt(i):
                        cv2.circle(frame, pt(i), max(2, round(4 * scale)), (255, 255, 255), -1, cv2.LINE_AA)

            n_split = bisect.bisect_right(splits, ms)
            n_swing = bisect.bisect_right([s[0] for s in swings], ms)
            corner = []
            if "count" in result:
                corner.append(f"Split steps: {n_split}")
            if swings or result.get("strokes"):
                corner.append(f"Swings: {n_swing}")
            for row, text in enumerate(corner):
                _label(frame, text, (int(16 * scale), int((40 + 36 * row) * scale)), (255, 255, 255), 0.8 * scale)

            banners = []
            if n_split and ms - splits[n_split - 1] <= BANNER_MS:
                banners.append(("SPLIT STEP", _GREEN))
            if n_swing and ms - swings[n_swing - 1][0] <= BANNER_MS:
                _, stroke, speed = swings[n_swing - 1]
                name = "SWING" if stroke == "unknown" else stroke.upper()
                banners.append((name + (f"  {speed:.1f} m/s" if speed else ""), _STROKE_BGR.get(stroke, _GREEN)))
            for row, (text, color) in enumerate(banners):
                _label(frame, text, (int(16 * scale), h - int((24 + 48 * row) * scale)), color, 1.2 * scale)

            out.write(frame)
            written += 1
    finally:
        cap.release()
        out.release()
    return written
