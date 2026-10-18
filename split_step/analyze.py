"""
analyze.py — headless, offline split-step counting for one video file.

This is what the HTTP API (api/main.py) calls. Unlike the GUI workers it does
no drawing and no playback pacing: frames are processed as fast as the CPU
allows, so the detectors are driven by the video's own timestamps instead of
the wall clock. Otherwise cooldowns and the LSTM's velocity features would
depend on how fast the server happens to be.

Usage
-----
    python -m split_step.analyze VIDEO [--model lstm|rules]

    from split_step.analyze import analyze_video
    result = analyze_video("clip.mp4", model="rules")
    result["count"]
"""

import json
import time

import cv2

from .detector import download_model, make_pose_landmarker, detect_pose, SplitStepDetector

MODELS = ("lstm", "rules")


class VideoError(ValueError):
    """The input can't be opened or decoded as a video."""


def _make_detector(model: str, clock):
    if model == "lstm":
        from .lstm_counter import LSTMCounter  # lazy: only import torch when needed
        return LSTMCounter(clock=clock)
    if model == "rules":
        return SplitStepDetector(clock=clock)
    raise ValueError(f"model must be one of {MODELS}, got {model!r}")


def analyze_video(video_path: str, model: str = "lstm") -> dict:
    """Count split steps in a video file.

    model — "lstm" (stateful LSTM, the GUI default) or "rules" (SplitStepDetector).

    Returns a JSON-serialisable dict:
        count            — number of split steps
        model            — the model used
        events           — [{"timestamp_ms": ...}] one per counted split step
        frames           — frames decoded
        frames_with_pose — frames where a person was detected
        duration_ms      — timestamp of the last decoded frame
        fps              — the file's reported frame rate (0 if unknown)
        processing_s     — wall time spent
    Raises VideoError if the file can't be opened or has no frames.
    """
    if model not in MODELS:
        raise ValueError(f"model must be one of {MODELS}, got {model!r}")
    download_model()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise VideoError("Could not open the file as a video.")

    started  = time.perf_counter()
    video_ts = [0.0]  # seconds; the detectors read this as their clock
    detector = _make_detector(model, clock=lambda: video_ts[0])

    events, frames, with_pose, last_ts_ms = [], 0, 0, 0.0
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
                frames    += 1
                last_ts_ms = ts_ms
                video_ts[0] = ts_ms / 1000.0

                world_lms, img_lms = detect_pose(landmarker, frame, ts_ms)
                if world_lms is None:
                    continue
                with_pose += 1
                if detector.process(world_lms, img_lms):
                    events.append({"timestamp_ms": round(ts_ms, 1)})
        fps = cap.get(cv2.CAP_PROP_FPS)
    finally:
        cap.release()

    if frames == 0:
        raise VideoError("The video contains no decodable frames.")

    return {
        "count":            detector.count,
        "model":            model,
        "events":           events,
        "frames":           frames,
        "frames_with_pose": with_pose,
        "duration_ms":      round(last_ts_ms, 1),
        "fps":              round(fps, 3) if fps and fps > 0 else 0,
        "processing_s":     round(time.perf_counter() - started, 3),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Count split steps in a video file.")
    parser.add_argument("video", help="Path to the video file")
    parser.add_argument("--model", choices=MODELS, default="lstm")
    args = parser.parse_args()
    print(json.dumps(analyze_video(args.video, args.model), indent=2))
