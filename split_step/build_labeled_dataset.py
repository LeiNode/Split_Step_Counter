"""
build_labeled_dataset.py — labels the original training videos using
split-step timestamp ranges parsed from a folder of pre-cut clip filenames
(e.g. "video_2 -00.00.06.102-00.00.06.835-seg2 (splitstep).mp4"), then runs
pose detection on the ORIGINAL full-length videos (not the cut clips, which
may be corrupted by short-segment export failures) and writes one combined
labeled CSV.

Usage
-----
    python -m split_step.build_labeled_dataset \\
        "assets/split_step_videos/Labeled Training Set 1" \\
        "assets/split_step_videos/Training Set 1" \\
        assets/csv_files/labeled_training_set_1.csv
"""

import argparse
import math
import os
import re

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

from .detector import download_model, MODEL_PATH, KEY_LANDMARKS, MIN_VISIBILITY
from .pose_csv_recorder import PoseCSVRecorder


def _key_landmarks_visible(world_lms) -> bool:
    """True when every KEY_LANDMARKS point (hips, ankles, heels, foot-index) is
    visible enough to trust — mirrors SplitStepDetector's own gate, so a frame
    where the ankles are cropped out of the shot doesn't feed noisy leg
    coordinates into training."""
    if world_lms is None:
        return False
    return all(world_lms[i].visibility >= MIN_VISIBILITY for i in KEY_LANDMARKS)


# Scene-cut (hard edit) detection for compilation-style source videos.
#
# Pose landmarks turned out to be the wrong signal for this: MediaPipe's
# world landmarks are defined relative to the person's own hip, so a totally
# different shot can still show a similar hip-relative body pose (e.g. two
# different players both in a ready stance) with no detectable "jump" at all.
# A hard cut is a property of the video's pixels, not the estimated pose, so
# it's detected directly on the decoded frame: a real cut causes color
# content to change abruptly, while continuous footage stays highly
# correlated frame-to-frame (median correlation ~1.0 in practice).
_HIST_CORR_THRESHOLD = 0.7
# Cooldown after a detected cut, in frames (~1s at 30fps). Multi-frame
# transition effects (fades/wipes) dip the correlation for several frames in
# a row; without a cooldown that reads as many cuts instead of one.
_CUT_COOLDOWN_FRAMES = 30


def _frame_histogram(frame_bgr):
    small = cv2.resize(frame_bgr, (64, 36))
    hist = cv2.calcHist([small], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist


class _CutDetector:
    """Tracks whether consecutive frames belong to the same continuous shot."""

    def __init__(self):
        self._prev_hist = None
        self._frames_since_cut = _CUT_COOLDOWN_FRAMES  # allow an immediate cut if needed
        self.sub_idx = 0
        self.cuts = 0

    def update(self, frame_bgr) -> None:
        hist = _frame_histogram(frame_bgr)
        self._frames_since_cut += 1
        if self._prev_hist is not None:
            corr = cv2.compareHist(self._prev_hist, hist, cv2.HISTCMP_CORREL)
            if corr < _HIST_CORR_THRESHOLD and self._frames_since_cut >= _CUT_COOLDOWN_FRAMES:
                self.sub_idx += 1
                self.cuts += 1
                self._frames_since_cut = 0
        self._prev_hist = hist

_VIDEO_EXTS = (".mp4", ".webm")

# Matches both filename conventions seen across labeled-clip batches:
#   "video_2 -00.00.06.102-00.00.06.835-seg2 (splitstep).mp4"   (seg number after timestamps)
#   "video22-seg01-00.00.00.000-00.00.02.900(not_splitstep).webm" (seg number before timestamps)
_RANGE_PATTERN = re.compile(
    r"^(video_?\d+)(?:-seg\d+)?\s*-(\d{2}\.\d{2}\.\d{2}\.\d{3})-(\d{2}\.\d{2}\.\d{2}\.\d{3})",
    re.IGNORECASE,
)
_LABEL_TAG_PATTERN = re.compile(r"\((?:not[_ ]?)?split[_ ]?step\)", re.IGNORECASE)


def _parse_timestamp(ts: str) -> float:
    """'HH.MM.SS.mmm' -> milliseconds."""
    hh, mm, ss, mmm = (int(p) for p in ts.split("."))
    return (((hh * 60) + mm) * 60 + ss) * 1000 + mmm


def _parse_ranges(clips_dir: str) -> dict:
    """Return {video_prefix: [(start_ms, end_ms), ...]} for every clip in a folder,
    regardless of any label tag in the filename."""
    ranges_by_video: dict = {}
    for fname in os.listdir(clips_dir):
        if not fname.lower().endswith(_VIDEO_EXTS):
            continue
        m = _RANGE_PATTERN.match(fname)
        if not m:
            continue
        video, start_s, end_s = m.groups()
        ranges_by_video.setdefault(video, []).append(
            (_parse_timestamp(start_s), _parse_timestamp(end_s))
        )
    return ranges_by_video


def _parse_segments(clips_dir: str) -> dict:
    """Return {video_prefix: [(start_ms, end_ms), ...]} of split-step ranges only."""
    ranges_by_video: dict = {}
    for fname in os.listdir(clips_dir):
        if not fname.lower().endswith(_VIDEO_EXTS):
            continue
        m = _RANGE_PATTERN.match(fname)
        tag_m = _LABEL_TAG_PATTERN.search(fname)
        if not m or not tag_m or "not" in tag_m.group(0).lower():
            continue
        video, start_s, end_s = m.groups()
        ranges_by_video.setdefault(video, []).append(
            (_parse_timestamp(start_s), _parse_timestamp(end_s))
        )
    return ranges_by_video


def _in_any_range(timestamp_ms: float, ranges: list) -> int:
    for start, end in ranges:
        if math.floor(start) <= timestamp_ms <= math.floor(end):
            return 1
    return 0


def _load_video_mapping(clips_dir: str) -> dict:
    """Read an optional video_mapping.txt from clips_dir, mapping generic clip
    prefixes (e.g. 'video22') to their actual source filename in source_dir —
    for batches where clip names don't match the source video's own filename.
    Lines are '<prefix>=<source filename>'; '#' starts a comment."""
    path = os.path.join(clips_dir, "video_mapping.txt")
    mapping: dict = {}
    if not os.path.exists(path):
        return mapping
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, _, value = line.partition("=")
            mapping[key.strip()] = value.strip()
    return mapping


def _find_source_video(source_dir: str, video_prefix: str, mapping: dict) -> str | None:
    """Returns the source video path, or None if the mapping marks this
    video_prefix as excluded (e.g. a known-bad recording)."""
    mapped = mapping.get(video_prefix)
    if mapped is not None:
        if mapped.upper().startswith("EXCLUDED"):
            return None
        path = os.path.join(source_dir, mapped)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Mapped source not found: {path}")
        return path

    for fname in os.listdir(source_dir):
        if fname.lower().endswith(_VIDEO_EXTS) and fname.startswith(video_prefix):
            return os.path.join(source_dir, fname)
    raise FileNotFoundError(f"No source video found for {video_prefix} in {source_dir}")


def build_dataset(clips_dir: str, source_dir: str, out_path: str,
                   exclude_dir: str | None = None) -> None:
    download_model()

    all_by_video = _parse_ranges(clips_dir)
    if not all_by_video:
        raise ValueError(f"No clips parsed from filenames in {clips_dir}")
    ranges_by_video = _parse_segments(clips_dir)  # positive (splitstep) ranges only

    video_mapping = _load_video_mapping(clips_dir)
    exclude_by_video = _parse_ranges(exclude_dir) if exclude_dir else {}

    out_dir, out_name = os.path.split(out_path)
    recorder = PoseCSVRecorder(out_dir or ".", out_name, extra_columns=["label", "source_video"])

    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    total = 0
    try:
        for video_prefix in sorted(all_by_video):
            ranges = ranges_by_video.get(video_prefix, [])  # [] = every clip is not_splitstep
            exclude_ranges = exclude_by_video.get(video_prefix, [])
            source_path = _find_source_video(source_dir, video_prefix, video_mapping)
            if source_path is None:
                print(f"{video_prefix}: excluded via video_mapping.txt, skipping", flush=True)
                continue
            print(f"{video_prefix}: {len(ranges)} split-step range(s), "
                  f"{len(exclude_ranges)} excluded (test) range(s) -> {source_path}", flush=True)

            cap = cv2.VideoCapture(source_path)
            if not cap.isOpened():
                print(f"  WARNING: could not open {source_path}, skipping", flush=True)
                continue

            n = 0
            positive = 0
            excluded = 0
            low_vis = 0
            cutter = _CutDetector()
            with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    ts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                    cutter.update(frame)  # runs on every decoded frame, independent of pose

                    if _in_any_range(ts_ms, exclude_ranges):
                        excluded += 1
                        continue

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    result = landmarker.detect_for_video(mp_img, int(ts_ms))

                    world_lms = result.pose_world_landmarks[0] if result.pose_world_landmarks else None
                    img_lms   = result.pose_landmarks[0]       if result.pose_landmarks       else None

                    if not _key_landmarks_visible(world_lms):
                        low_vis += 1
                        continue

                    label = _in_any_range(ts_ms, ranges)
                    positive += label
                    recorder.record(world_lms, img_lms, 0, int(ts_ms),
                                    extra={"label": label, "source_video": f"{video_prefix}#{cutter.sub_idx}"})
                    n += 1
            cap.release()
            total += n
            print(f"  {n} frames, {positive} positive, {excluded} excluded (test set), "
                  f"{low_vis} skipped (low ankle/hip visibility), {cutter.cuts} scene cut(s) detected "
                  f"({cutter.sub_idx + 1} sub-clip(s))", flush=True)
    finally:
        recorder.close()

    print(f"\nTotal frames: {total}")


def build_unlabeled_range(source_video: str, start_ms: float, end_ms: float,
                           out_path: str, source_video_name: str | None = None) -> None:
    """Extract raw pose landmarks (no label column) for one timestamp range of a
    video — for a held-out test region that hasn't been manually labeled yet."""
    download_model()

    label = source_video_name or os.path.splitext(os.path.basename(source_video))[0]
    out_dir, out_name = os.path.split(out_path)
    recorder = PoseCSVRecorder(out_dir or ".", out_name, extra_columns=["source_video"])

    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(source_video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open {source_video}")

    n = 0
    low_vis = 0
    cutter = _CutDetector()
    try:
        with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                ts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                if ts_ms < start_ms:
                    continue
                if ts_ms > end_ms:
                    break

                cutter.update(frame)  # runs on every decoded frame, independent of pose

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect_for_video(mp_img, int(ts_ms))

                world_lms = result.pose_world_landmarks[0] if result.pose_world_landmarks else None
                img_lms   = result.pose_landmarks[0]       if result.pose_landmarks       else None

                if not _key_landmarks_visible(world_lms):
                    low_vis += 1
                    continue

                recorder.record(world_lms, img_lms, 0, int(ts_ms),
                                extra={"source_video": f"{label}#{cutter.sub_idx}"})
                n += 1
    finally:
        cap.release()
        recorder.close()

    print(f"{n} frames extracted from {source_video} [{start_ms}-{end_ms}ms] -> {out_path} "
          f"({low_vis} skipped for low ankle/hip visibility, {cutter.cuts} scene cut(s) detected, "
          f"{cutter.sub_idx + 1} sub-clip(s))")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Label original training videos using split-step ranges parsed from cut-clip filenames."
    )
    parser.add_argument("clips_dir",  help="Folder of labeled clips, e.g. 'Labeled Training Set 1'")
    parser.add_argument("source_dir", help="Folder with the original full-length videos, e.g. 'Training Set 1'")
    parser.add_argument("out_csv",    help="Output CSV path, e.g. assets/csv_files/labeled_training_set_1.csv")
    parser.add_argument("--exclude",  help="Folder of held-out test clips whose timestamp ranges "
                                            "should be skipped entirely, e.g. 'Test_Set'")
    args = parser.parse_args()
    build_dataset(args.clips_dir, args.source_dir, args.out_csv, exclude_dir=args.exclude)
