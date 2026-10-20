"""
features.py — feature switches for the major features.

A disabled feature disappears everywhere: the API stops returning its data
and rejects requests for it, the web UI hides its controls (it reads
GET /v1/features), and the desktop GUI hides its widgets.

Each switch is resolved in this order (later wins):
  1. DEFAULTS below
  2. a JSON file {"feature_name": true/false, ...}: $SPLIT_STEP_FEATURES_FILE,
     or features.json in the repo root if it exists
  3. $SPLIT_STEP_FEATURES, e.g. "serve_training=off,live_view=on"
     (on/off, true/false, 1/0, yes/no)

Flags are re-read on every call, so a changed env var or file takes effect
on the next request without restarting anything long-lived.
"""

import json
import os

# name -> (default, description)
DEFAULTS: dict[str, tuple[bool, str]] = {
    "split_step_counting": (True,  "Count split steps (the core feature)"),
    "coach_summary":       (True,  "Prioritised summary: top things to work on + strengths"),
    "practice_plan":       (True,  "Drills for the next session, one per focus point of the coach's summary"),
    "footwork":            (True,  "Step counting: steps per minute, adjustment steps between shots"),
    "fatigue_trend":       (True,  "Compare the start and end of long sessions: does the player fade?"),
    "camera_view_check":   (True,  "Detect the camera angle and warn when it's one the model handles poorly"),
    "split_step_metrics":  (True,  "Per-split-step hop height, knee bend, stance width + session summary"),
    "stroke_training":     (True,  "Swing detection and stroke feedback (master switch for the three below)"),
    "forehand_training":   (True,  "Forehand drill"),
    "backhand_training":   (True,  "Backhand drill"),
    "serve_training":      (True,  "Serve drill"),
    "auto_stroke_detection": (True, "'auto' drill: classify each swing (forehand/backhand is experimental)"),
    "ready_position":      (True,  "Recovery to a ready stance after each shot"),
    "swing_chart":         (True,  "Web app: chart of each swing's wrist speed through the clip"),
    "side_by_side":        (True,  "Web app: play your most typical and odd-one-out swing side by side"),
    "swing_highlights":    (True,  "Point out the most typical, fastest and odd-one-out swing of each stroke"),
    "rally_stats":         (True,  "Group swings into rallies: shots per rally, longest rally"),
    "split_step_timing":   (True,  "Link split steps to the following shots (needs counting + stroke training)"),
    "lstm_model":          (True,  "LSTM split-step model"),
    "rules_model":         (True,  "Rule-based split-step model"),
    "clip_range":          (True,  "Analyse only part of a video (start / end time)"),
    "direct_upload":       (True,  "API: POST /v1/count with the video in the request"),
    "browser_uploads":     (True,  "API: presigned S3 uploads (/v1/upload-url, /v1/count/s3)"),
    "live_view":           (True,  "Desktop GUI: webcam live view"),
    "rep_target":          (True,  "Desktop GUI: swings per set in live drills, with a set-complete signal"),
    "audio_cues":          (True,  "Desktop GUI: a beep for each split step and swing"),
    "pose_overlay":        (True,  "Web app: draw the detected skeleton over the video (API: include_pose)"),
    "playback_controls":   (True,  "Web app: slow motion, previous/next shot and frame-step controls"),
    "markdown_report":     (True,  "API POST /v1/report + web app download: the result as a Markdown report"),
    "report_download":     (True,  "Web app: download results as JSON / CSV"),
    "drill_goals":         (True,  "Web app: set targets and see which ones each session met"),
    "session_history":     (True,  "Web app: keep past results in the browser and show progress"),
    "history_transfer":    (True,  "Web app: export history to a file and import it in another browser"),
    "session_notes":       (True,  "Web app: add a note to each saved session (shown in history, chart and compare)"),
    "session_compare":     (True,  "Web app: compare two past sessions side by side"),
}

# Child features are only on when all their parents are on too.
PARENTS = {
    "forehand_training": ("stroke_training",),
    "backhand_training": ("stroke_training",),
    "serve_training": ("stroke_training",),
    "auto_stroke_detection": ("stroke_training",),
    "split_step_metrics": ("split_step_counting",),
    "split_step_timing": ("split_step_counting", "stroke_training"),
    "rally_stats": ("stroke_training",),
    "swing_highlights": ("stroke_training",),
    "swing_chart": ("stroke_training",),
    "side_by_side": ("swing_highlights",),
    "session_compare": ("session_history",),
    "history_transfer": ("session_history",),
    "session_notes": ("session_history",),
    "rep_target": ("stroke_training",),
    "practice_plan": ("coach_summary",),
    "ready_position": ("stroke_training",),
}

_TRUE  = {"1", "true", "on", "yes"}
_FALSE = {"0", "false", "off", "no"}
_REPO_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "features.json")


class UnknownFeature(KeyError):
    pass


def _parse_bool(name: str, value) -> bool:
    if isinstance(value, bool):
        return value
    v = str(value).strip().lower()
    if v in _TRUE:
        return True
    if v in _FALSE:
        return False
    raise ValueError(f"feature {name!r}: can't read {value!r} as on/off")


def _check(name: str) -> None:
    if name not in DEFAULTS:
        raise UnknownFeature(f"unknown feature {name!r}; known: {sorted(DEFAULTS)}")


def all_features() -> dict[str, bool]:
    """Every feature's effective on/off state (parents applied)."""
    flags = {name: default for name, (default, _) in DEFAULTS.items()}

    path = os.environ.get("SPLIT_STEP_FEATURES_FILE") or (_REPO_FILE if os.path.exists(_REPO_FILE) else None)
    if path:
        with open(path, encoding="utf-8") as f:
            for name, value in json.load(f).items():
                _check(name)
                flags[name] = _parse_bool(name, value)

    for item in os.environ.get("SPLIT_STEP_FEATURES", "").split(","):
        if item.strip():
            name, _, value = item.partition("=")
            name = name.strip()
            _check(name)
            flags[name] = _parse_bool(name, value)

    for child, parents in PARENTS.items():
        flags[child] = flags[child] and all(flags[p] for p in parents)
    return flags


def enabled(name: str) -> bool:
    _check(name)
    return all_features()[name]


def describe() -> list[dict]:
    """[{name, enabled, description}] for UIs and docs."""
    flags = all_features()
    return [{"name": n, "enabled": flags[n], "description": d} for n, (_, d) in DEFAULTS.items()]


def drill_feature(drill: str) -> str:
    """Feature switch that controls a stroke drill ('auto' -> auto_stroke_detection)."""
    return "auto_stroke_detection" if drill == "auto" else f"{drill}_training"


def model_feature(model: str) -> str:
    return f"{model}_model"
