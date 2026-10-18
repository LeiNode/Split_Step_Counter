"""
PoseCSVRecorder — streams one CSV row per frame of pose landmarks.

Each row contains:
  - timestamp_ms and split-step count
  - world landmark coordinates (x, y, z metres, hip-relative) + visibility
  - image landmark coordinates (x, y normalised 0-1) for all 33 MediaPipe landmarks
  - any extra_columns passed to the constructor (e.g. label, source_video)
Missing landmarks are written as empty cells.

Usage
-----
    python -m split_step.pose_csv_recorder VIDEO [--out DIR]   # -> DIR/sideview1.csv

    recorder = PoseCSVRecorder("assets/csv_files", "pose1.csv")
    recorder.record(world_lms, img_lms, detector.count, ts_ms)
    recorder.close()                      # or use as a context manager
"""

import csv
import os
import time

ASSETS_DIR = "assets"

# MediaPipe BlazePose landmark names in index order (0-32)
_LANDMARK_NAMES = [
    "nose",
    "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear",
    "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_pinky", "right_pinky",
    "left_index", "right_index",
    "left_thumb", "right_thumb",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]

def _build_header() -> list:
    cols = ["timestamp_ms", "split_step_count"]
    for name in _LANDMARK_NAMES:
        cols += [
            f"{name}_world_x", f"{name}_world_y",
            f"{name}_world_z", f"{name}_world_vis",
            f"{name}_img_x",   f"{name}_img_y",
        ]
    return cols


class PoseCSVRecorder:
    """Streams pose landmark data to output_dir/output_name, overwriting any existing file."""

    def __init__(self, output_dir: str = ASSETS_DIR, output_name: str = "sideview1.csv",
                 extra_columns: list | None = None):
        os.makedirs(output_dir, exist_ok=True)
        self.path          = os.path.join(output_dir, output_name)
        self._file         = open(self.path, "w", newline="", encoding="utf-8")
        self._writer       = csv.writer(self._file)
        self._frame        = 0
        self._extra_columns = extra_columns or []
        self._writer.writerow(_build_header() + self._extra_columns)
        print(f"Recording -> {self.path}")

    # ── Public API ────────────────────────────────────────────────────────────

    def record(
        self,
        world_landmarks,
        image_landmarks,
        count:        int,
        timestamp_ms: int | None = None,
        extra:        dict | None = None,
    ) -> None:
        """
        Append one row for the current frame.

        world_landmarks — pose_world_landmarks[0]  (or None if unavailable)
        image_landmarks — pose_landmarks[0]         (or None if unavailable)
        count           — detector.count int
        timestamp_ms    — video position or wall-clock ms (auto-filled if None)
        extra           — values for the extra_columns passed to __init__
        """
        if timestamp_ms is None:
            timestamp_ms = int(time.time() * 1000)

        row = [timestamp_ms, count]

        for i in range(len(_LANDMARK_NAMES)):
            if world_landmarks is not None and i < len(world_landmarks):
                wlm = world_landmarks[i]
                row += [
                    round(wlm.x, 6), round(wlm.y, 6),
                    round(wlm.z, 6), round(wlm.visibility, 4),
                ]
            else:
                row += ["", "", "", ""]

            if image_landmarks is not None and i < len(image_landmarks):
                ilm = image_landmarks[i]
                row += [round(ilm.x, 6), round(ilm.y, 6)]
            else:
                row += ["", ""]

        for col in self._extra_columns:
            row.append((extra or {}).get(col, ""))

        self._writer.writerow(row)
        self._frame += 1

    def close(self) -> None:
        """Flush and close the file."""
        self._file.flush()
        self._file.close()
        print(f"Saved {self._frame} frames -> {self.path}")

    # ── Context manager support ───────────────────────────────────────────────

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import cv2
    from .detector import download_model, make_pose_landmarker, detect_pose, SplitStepDetector

    parser = argparse.ArgumentParser(description="Record pose landmarks from a video to CSV.")
    parser.add_argument("video", help="Path to the input video file")
    parser.add_argument("--out", default=ASSETS_DIR, help="Output directory (default: assets)")
    args = parser.parse_args()

    download_model()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {args.video}")

    SAMPLE_INTERVAL_MS = 100.0 / 3          # 33.333… ms  →  30 samples per second

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing {total} frames from {args.video} "
          f"(sampling every {SAMPLE_INTERVAL_MS:.2f} ms) …")

    detector      = SplitStepDetector()
    next_target   = 0.0                     # next timestamp we want to record

    with PoseCSVRecorder(args.out) as recorder:
        with make_pose_landmarker() as landmarker:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                ts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

                if ts_ms < next_target:     # not yet at the next sample point
                    continue

                next_target += SAMPLE_INTERVAL_MS

                world_lms, img_lms = detect_pose(landmarker, frame, ts_ms)

                if world_lms:
                    detector.process(world_lms, img_lms)

                recorder.record(world_lms, img_lms, detector.count, int(ts_ms))

                if recorder._frame % 100 == 0:
                    print(f"  {recorder._frame} samples recorded …")

    cap.release()
