"""
analyze.py — headless, offline analysis of one video file.

This is what the HTTP API (api/main.py) calls. Unlike the GUI workers it does
no drawing and no playback pacing: frames are processed as fast as the CPU
allows, so the detectors are driven by the video's own timestamps instead of
the wall clock. Otherwise cooldowns and the LSTM's velocity features would
depend on how fast the server happens to be.

One pass over the video records a PoseTrack; split-step counting runs live
during the pass, and everything else (split-step quality, stroke training)
is computed from the track afterwards.

Usage
-----
    python -m split_step.analyze VIDEO [--model lstm|rules] [--drill auto|forehand|backhand|serve]

    from split_step.analyze import analyze_video
    result = analyze_video("clip.mp4", model="rules", drill="forehand")
"""

import json
import time

import cv2

from .detector import download_model, make_pose_landmarker, detect_pose, SplitStepDetector
from .pose_track import PoseTrackBuilder
from .strokes import DRILLS, HANDS, stroke_report
from .tennis_metrics import session_summary, split_step_quality
from .camera_view import camera_report
from .coach import coach_summary
from .fatigue import fatigue_report
from .footwork import footwork_report
from .rallies import find_rallies
from .ready_position import recovery
from .timing import link_split_steps

MODELS = ("lstm", "rules")

# Frames closer together than this fraction of 1/max_fps are skipped, so a
# 60 fps phone video is analysed at ~30 fps (jittery timestamps still pass).
_FRAME_SKIP_SLACK = 0.75


class VideoError(ValueError):
    """The input can't be opened or decoded as a video."""


class VideoTooLong(VideoError):
    """The video is longer than the max_seconds the caller allows."""


def _make_detector(model: str, clock, require_full_pose: bool = True):
    if model == "lstm":
        from .lstm_counter import LSTMCounter  # lazy: only import torch when needed
        return LSTMCounter(clock=clock, require_full_pose=require_full_pose)
    if model == "rules":
        return SplitStepDetector(clock=clock)
    raise ValueError(f"model must be one of {MODELS}, got {model!r}")


def analyze_video(video_path: str, model: str = "lstm", require_full_pose: bool = True,
                  drill: str | None = None, hand: str = "auto",
                  count_split_steps: bool = True, split_step_metrics: bool = True,
                  split_step_timing: bool = True, rally_stats: bool = True,
                  include_pose: bool = False, ready_position: bool = True,
                  camera_view_check: bool = True, footwork: bool = True,
                  coach: bool = True, fatigue: bool = True, max_fps: float | None = 30.0,
                  max_seconds: float | None = None) -> dict:
    """Analyse a video file.

    model              — split-step counter: "lstm" (GUI default) or "rules"
    require_full_pose  — skip the LSTM on frames where the hips/ankles/heels/toes
                         aren't all visible (the rules detector always does this)
    drill              — None: no stroke analysis. "forehand" / "backhand" /
                         "serve": every swing is that stroke. "auto": classify.
    hand               — hitting hand for stroke analysis: "right", "left", "auto"
    count_split_steps  — False skips split-step counting entirely
    split_step_metrics — False skips per-split-step quality metrics + session summary
    split_step_timing  — with counting and a drill: link split steps to the shots
                         that follow (see timing.py)
    rally_stats        — with a drill: group swings into rallies (see rallies.py)
    include_pose       — also return the per-frame skeleton (PoseTrack.export_skeleton)
    ready_position     — with a drill: recovery to a ready stance after each shot
    camera_view_check  — report the camera angle (camera_view.py)
    footwork           — count steps (footwork.py)
    fatigue            — compare the first and last third of the session (fatigue.py)
    coach              — prioritised summary of all tips (coach.py)
    max_fps            — analyse at most this many frames per second (None = all).
                         The LSTM was trained on ~30 fps footage and the rules
                         detector's thresholds are per frame, so higher-fps video
                         is thinned to match; it also halves the cost of 60 fps.
                         On a 60 fps match clip, full-rate analysis found 5 extra
                         "swings", 3 of them clearly false (running, ball bouncing).
    max_seconds        — refuse videos longer than this (VideoTooLong). Checked up
                         front from the container's metadata and again while
                         decoding, since some formats (e.g. webm) report no length.

    Returns a JSON-serialisable dict. Always present:
        frames, frames_with_pose, duration_ms, fps, processing_s,
        frames_analysed (fewer than frames when max_fps thinned the video)
    With count_split_steps:
        count, model, frames_tracked,
        events  — one dict per split step: timestamp_ms (+ quality metrics)
        session — whole-video stats and coaching tips (with split_step_metrics)
    With a drill:
        strokes — stroke_report(): swings, per-stroke summary, tips
    With counting, a drill and split_step_timing:
        timing  — link_split_steps(): share of shots preceded by a split step;
                  also adds events[].during_swing and swings[].split_step_lead_ms
    With a drill and rally_stats:
        rallies — find_rallies(): rallies list + summary; adds swings[].rally
    With a drill and ready_position:
        recovery — recovery(): share of shots recovered from + median time;
                   adds swings[].recovery_ms
    With footwork:
        footwork — steps, steps per minute; with a drill also adjustment steps
                   between shots (adds swings[].steps_before)
    With fatigue:
        fatigue — early vs late medians of hop height, knee bend, swing speed,
                  recovery; only measured on sessions of 2+ minutes
    With coach:
        coach   — {focus: top tips to work on, strengths, more}
    With camera_view_check:
        camera  — dominant view (back/front/side), per-view shares, warning tip
    With include_pose:
        pose    — per-frame image-space skeleton for drawing overlays
    Raises VideoError if the file can't be opened or has no frames.
    """
    if model not in MODELS:
        raise ValueError(f"model must be one of {MODELS}, got {model!r}")
    if drill is not None and drill not in DRILLS:
        raise ValueError(f"drill must be one of {DRILLS}, got {drill!r}")
    if hand not in HANDS:
        raise ValueError(f"hand must be one of {HANDS}, got {hand!r}")
    download_model()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise VideoError("Could not open the file as a video.")

    if max_seconds:
        n, f = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS)
        if n > 0 and f > 0 and n / f > max_seconds:
            cap.release()
            raise VideoTooLong(f"Video is {n / f / 60:.1f} min long; the limit is {max_seconds / 60:g} min.")

    started  = time.perf_counter()
    video_ts = [0.0]  # seconds; the detectors read this as their clock
    detector = (_make_detector(model, clock=lambda: video_ts[0], require_full_pose=require_full_pose)
                if count_split_steps else None)

    track = PoseTrackBuilder(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                             int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    events, frames, with_pose, tracked, last_ts_ms = [], 0, 0, 0, 0.0
    min_gap_ms = 1000.0 / max_fps * _FRAME_SKIP_SLACK if max_fps else 0.0
    analysed, last_used_ms = 0, None
    try:
        with make_pose_landmarker() as landmarker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                ts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                if frames and ts_ms <= last_ts_ms:
                    # MediaPipe VIDEO mode rejects non-increasing timestamps
                    # (some containers report duplicates); nudge forward 1 ms.
                    ts_ms = last_ts_ms + 1
                if max_seconds and ts_ms > max_seconds * 1000:
                    raise VideoTooLong(f"Video is longer than the {max_seconds / 60:g} min limit.")
                frames    += 1
                last_ts_ms = ts_ms
                if last_used_ms is not None and ts_ms - last_used_ms < min_gap_ms:
                    continue
                last_used_ms = ts_ms
                analysed += 1
                video_ts[0] = ts_ms / 1000.0

                world_lms, img_lms = detect_pose(landmarker, frame, ts_ms)
                track.add(ts_ms, world_lms, img_lms)
                if world_lms is None:
                    continue
                with_pose += 1
                if detector is not None:
                    hop = detector.process(world_lms, img_lms)
                    tracked += detector.tracking
                    if hop:
                        events.append({"timestamp_ms": round(ts_ms, 1)})
        fps = cap.get(cv2.CAP_PROP_FPS)
    finally:
        cap.release()

    if frames == 0:
        raise VideoError("The video contains no decodable frames.")

    pose_track = track.build()
    result = {}
    if detector is not None:
        result.update({"count": detector.count, "model": model, "events": events})
        if split_step_metrics:
            for e in events:
                e.update(split_step_quality(pose_track, e["timestamp_ms"]))
            result["session"] = session_summary(pose_track, events)
        result["frames_tracked"] = tracked
    if drill is not None:
        result["strokes"] = stroke_report(pose_track, drill, hand)
        if detector is not None and split_step_timing:
            result["timing"] = link_split_steps(events, result["strokes"]["swings"])
        if rally_stats:
            result["rallies"] = find_rallies(result["strokes"]["swings"])
        if ready_position:
            result["recovery"] = recovery(pose_track, result["strokes"]["swings"])
    if include_pose:
        result["pose"] = pose_track.export_skeleton()
    if camera_view_check:
        result["camera"] = camera_report(pose_track)
    if footwork:
        result["footwork"] = footwork_report(pose_track, result["strokes"]["swings"] if drill else None)
    if fatigue:
        result["fatigue"] = fatigue_report(result.get("events"), result["strokes"]["swings"] if drill else None,
                                           last_ts_ms)
    if coach:
        result["coach"] = coach_summary(result)

    result.update({
        "frames":           frames,
        "frames_analysed":  analysed,
        "frames_with_pose": with_pose,
        "duration_ms":      round(last_ts_ms, 1),
        "fps":              round(fps, 3) if fps and fps > 0 else 0,
        "processing_s":     round(time.perf_counter() - started, 3),
    })
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Count split steps and analyse strokes in a video file.")
    parser.add_argument("video", help="Path to the video file")
    parser.add_argument("--model", choices=MODELS, default="lstm")
    parser.add_argument("--allow-partial-pose", action="store_true",
                        help="Run the LSTM even when the lower body isn't fully visible")
    parser.add_argument("--drill", choices=DRILLS, help="Also analyse strokes (forehand/backhand/serve/auto)")
    parser.add_argument("--hand", choices=HANDS, default="auto", help="Hitting hand for stroke analysis")
    parser.add_argument("--max-fps", type=float, default=30.0,
                        help="Analyse at most this many frames per second; 0 = every frame (default 30)")
    args = parser.parse_args()
    print(json.dumps(analyze_video(args.video, args.model,
                                   require_full_pose=not args.allow_partial_pose,
                                   drill=args.drill, hand=args.hand,
                                   max_fps=args.max_fps or None), indent=2))
