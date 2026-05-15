"""
Split Step Counter
Detects and counts tennis split steps using webcam + MediaPipe Pose Landmarker.

A split step is a small hop where both feet briefly leave the ground,
used by tennis players to prepare for movement before the opponent hits.

Detection logic:
  - Track the vertical position of both ankles over time
  - A split step is counted when both ankles rise (hop up) above a rolling baseline
  - A cooldown prevents double-counting a single hop

Requires: pip install opencv-python mediapipe numpy
The pose model (~6 MB) is downloaded automatically on first run.
"""

import os
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

# ── Model ──────────────────────────────────────────────────────────────────────
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
MODEL_PATH = "pose_landmarker_full.task"

# Landmark indices (same as the old PoseLandmark enum)
LEFT_HIP    = 23
RIGHT_HIP   = 24
LEFT_ANKLE  = 27
RIGHT_ANKLE = 28

# Connections to draw (pairs of landmark indices)
POSE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
    (9,10),(11,12),(11,13),(13,15),(15,17),(15,19),(15,21),(17,19),
    (12,14),(14,16),(16,18),(16,20),(16,22),(18,20),
    (11,23),(12,24),(23,24),(23,25),(24,26),(25,27),(26,28),(27,29),
    (28,30),(29,31),(30,32),(27,31),(28,32),
]

# ── Tuning parameters ──────────────────────────────────────────────────────────
# How much BOTH ankles must rise above the grounded baseline, expressed as a
# fraction of hip-to-ankle leg length. Using leg length makes the threshold
# scale-invariant: moving closer/farther from the camera doesn't change it.
# Raise this if you get false positives; lower it if real hops are being missed.
HOP_THRESHOLD = 0.04

# How close the ankles must return to baseline (fraction of leg length) before
# we consider the player landed and increment the count.
LAND_THRESHOLD = 0.02

# Minimum frames between two detected split steps (prevents double-counting a
# single landing bounce).
COOLDOWN_FRAMES = 20

# Smoothing factor for the exponential moving average applied to ankle positions.
# Range (0, 1): lower = smoother but laggier, higher = more responsive but noisier.
EMA_ALPHA = 0.25

# Number of frames at startup before detection begins (lets the baseline settle).
WARMUP_FRAMES = 30
# ──────────────────────────────────────────────────────────────────────────────


# Detection states
GROUNDED = "grounded"
IN_HOP   = "in_hop"


def download_model() -> None:
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading pose model to {MODEL_PATH} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download complete.")


def draw_landmarks(frame: np.ndarray, landmarks: list) -> None:
    """Draw pose skeleton onto frame using pixel coordinates."""
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    for a, b in POSE_CONNECTIONS:
        if a < len(pts) and b < len(pts):
            cv2.line(frame, pts[a], pts[b], (0, 180, 0), 2)

    for x, y in pts:
        cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)


def main() -> None:
    download_model()

    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check that a camera is connected.")

    split_step_count = 0
    cooldown        = 0
    frame_idx       = 0
    state           = GROUNDED

    # EMA-smoothed ankle and hip positions (initialised on first detection)
    ema_left: float | None      = None
    ema_right: float | None     = None
    ema_left_hip: float | None  = None
    ema_right_hip: float | None = None

    # Grounded baseline: updated only while the player is on the ground
    baseline_left: float | None  = None
    baseline_right: float | None = None

    with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            h, w = frame.shape[:2]
            hop_detected = False
            frame_idx += 1

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]

                raw_left      = landmarks[LEFT_ANKLE].y
                raw_right     = landmarks[RIGHT_ANKLE].y
                raw_left_hip  = landmarks[LEFT_HIP].y
                raw_right_hip = landmarks[RIGHT_HIP].y

                # Exponential moving average to reduce jitter
                if ema_left is None:
                    ema_left, ema_right         = raw_left, raw_right
                    ema_left_hip, ema_right_hip = raw_left_hip, raw_right_hip
                else:
                    ema_left  = EMA_ALPHA * raw_left  + (1 - EMA_ALPHA) * ema_left
                    ema_right = EMA_ALPHA * raw_right + (1 - EMA_ALPHA) * ema_right
                    ema_left_hip  = EMA_ALPHA * raw_left_hip  + (1 - EMA_ALPHA) * ema_left_hip
                    ema_right_hip = EMA_ALPHA * raw_right_hip + (1 - EMA_ALPHA) * ema_right_hip

                # Leg length (ankle_y - hip_y, positive since ankle is lower in frame).
                # Used to normalise thresholds so they don't change with camera distance.
                leg_scale = max(
                    ((ema_left - ema_left_hip) + (ema_right - ema_right_hip)) / 2,
                    0.01,  # guard against zero if legs aren't visible
                )

                # Warm up: establish baseline before detecting anything
                if frame_idx <= WARMUP_FRAMES:
                    baseline_left  = ema_left
                    baseline_right = ema_right
                else:
                    # ── State machine ────────────────────────────────────────
                    # y increases downward, so a hop = ankle y DECREASES.
                    # Divide by leg_scale → thresholds are now a fraction of leg
                    # length rather than frame height, so distance to camera doesn't
                    # affect sensitivity.
                    rise_left  = (baseline_left  - ema_left)  / leg_scale
                    rise_right = (baseline_right - ema_right) / leg_scale

                    if state == GROUNDED:
                        if rise_left > HOP_THRESHOLD and rise_right > HOP_THRESHOLD:
                            state = IN_HOP
                        else:
                            # Slowly track baseline while standing still
                            baseline_left  = EMA_ALPHA * ema_left  + (1 - EMA_ALPHA) * baseline_left
                            baseline_right = EMA_ALPHA * ema_right + (1 - EMA_ALPHA) * baseline_right

                    elif state == IN_HOP:
                        if rise_left < LAND_THRESHOLD and rise_right < LAND_THRESHOLD:
                            # Both feet back down — count the split step
                            if cooldown == 0:
                                split_step_count += 1
                                hop_detected = True
                                cooldown = COOLDOWN_FRAMES
                            state = GROUNDED

                draw_landmarks(frame, landmarks)

            if cooldown > 0:
                cooldown -= 1

            # ── HUD ───────────────────────────────────────────────────────────
            state_color = (0, 140, 255) if state == IN_HOP else (0, 255, 0)
            cv2.rectangle(frame, (0, 0), (300, 80), (0, 0, 0), -1)
            cv2.putText(
                frame,
                f"Split Steps: {split_step_count}",
                (10, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"State: {state}",
                (10, 68),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                state_color,
                2,
            )

            if hop_detected:
                cv2.putText(
                    frame,
                    "HOP!",
                    (w // 2 - 40, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2.0,
                    (0, 0, 255),
                    4,
                )

            cv2.imshow("Split Step Counter  |  press Q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nSession complete. Total split steps detected: {split_step_count}")


if __name__ == "__main__":
    main()
