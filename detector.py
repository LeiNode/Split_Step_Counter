"""
Core detection logic: constants, utilities, SplitStepDetector, and the
_AnalysisWorker base class that LiveViewAnalysis and VideoAnalysis extend.
"""

import os
import queue
import threading
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

# ── Model ──────────────────────────────────────────────────────────────────────
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
MODEL_PATH = "pose_landmarker_full.task"

# ── Landmark indices ───────────────────────────────────────────────────────────
LEFT_HIP    = 23
RIGHT_HIP   = 24
LEFT_ANKLE  = 27
RIGHT_ANKLE = 28

POSE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
    (9,10),(11,12),(11,13),(13,15),(15,17),(15,19),(15,21),(17,19),
    (12,14),(14,16),(16,18),(16,20),(16,22),(18,20),
    (11,23),(12,24),(23,24),(23,25),(24,26),(25,27),(26,28),(27,29),
    (28,30),(29,31),(30,32),(27,31),(28,32),
]

# ── Tuning ─────────────────────────────────────────────────────────────────────
MIN_VISIBILITY   = 0.5   # landmarks below this confidence are treated as out-of-frame
LIFT_THRESHOLD   = 0.06  # metres shortened from max extension to consider foot in air
HOP_SYNC_WINDOW  = 0.20  # seconds between lifts to count as a simultaneous hop

KEY_LANDMARKS = [LEFT_HIP, RIGHT_HIP, LEFT_ANKLE, RIGHT_ANKLE]

GROUNDED      = "grounded"
AIRBORNE      = "airborne"
LEFT_LEG_UP   = "left_leg_up"
RIGHT_LEG_UP  = "right_leg_up"

CANVAS_W = 854
CANVAS_H = 480


# ── Utilities ──────────────────────────────────────────────────────────────────

def download_model() -> None:
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading pose model to {MODEL_PATH} …")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download complete.")


def draw_landmarks(frame: np.ndarray, landmarks: list) -> None:
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in POSE_CONNECTIONS:
        if a < len(pts) and b < len(pts):
            cv2.line(frame, pts[a], pts[b], (0, 180, 0), 2)
    for x, y in pts:
        cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)


def fit_frame(frame: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Letterbox-resize frame to fit inside (target_w, target_h)."""
    h, w = frame.shape[:2]
    scale = min(target_w / w, target_h / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (nw, nh))
    out = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y0 = (target_h - nh) // 2
    x0 = (target_w - nw) // 2
    out[y0:y0+nh, x0:x0+nw] = resized
    return out


_STATE_COLORS = {
    GROUNDED:     (0, 220, 80),    # green
    AIRBORNE:     (0, 140, 255),   # orange
    LEFT_LEG_UP:  (255, 200, 0),   # cyan-yellow
    RIGHT_LEG_UP: (200, 0, 255),   # purple
}

def annotate_frame(frame: np.ndarray, det: "SplitStepDetector", hop: bool) -> None:
    h, w = frame.shape[:2]
    state_color = _STATE_COLORS.get(det.state, (255, 255, 255))
    cv2.rectangle(frame, (0, 0), (320, 80), (0, 0, 0), -1)
    cv2.putText(frame, f"Split Steps: {det.count}",
                (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 220, 220), 2)
    cv2.putText(frame, f"State: {det.state}",
                (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2)
    if hop:
        cv2.putText(frame, "HOP!", (w // 2 - 70, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 5)


# ── Foot-state tracker ────────────────────────────────────────────────────────

class FootStateTracker:
    def __init__(self):
        self.max_left_len   = 0.0
        self.max_right_len  = 0.0
        self.left_lift_time  = 0.0
        self.right_lift_time = 0.0
        self.left_is_up  = False
        self.right_is_up = False

    def reset(self):
        self.__init__()

    def process_frame(self, world_landmarks) -> str:
        current_time = time.time()

        l_hip   = np.array([world_landmarks[LEFT_HIP].x,   world_landmarks[LEFT_HIP].y,   world_landmarks[LEFT_HIP].z])
        r_hip   = np.array([world_landmarks[RIGHT_HIP].x,  world_landmarks[RIGHT_HIP].y,  world_landmarks[RIGHT_HIP].z])
        l_ankle = np.array([world_landmarks[LEFT_ANKLE].x,  world_landmarks[LEFT_ANKLE].y,  world_landmarks[LEFT_ANKLE].z])
        r_ankle = np.array([world_landmarks[RIGHT_ANKLE].x, world_landmarks[RIGHT_ANKLE].y, world_landmarks[RIGHT_ANKLE].z])

        dist_left  = np.linalg.norm(l_hip - l_ankle)
        dist_right = np.linalg.norm(r_hip - r_ankle)

        self.max_left_len  = max(self.max_left_len,  dist_left)
        self.max_right_len = max(self.max_right_len, dist_right)

        if self.max_left_len == 0 or self.max_right_len == 0:
            return "Calibrating..."

        left_lifted_now  = (self.max_left_len  - dist_left)  > LIFT_THRESHOLD
        right_lifted_now = (self.max_right_len - dist_right) > LIFT_THRESHOLD

        if left_lifted_now and not self.left_is_up:
            self.left_lift_time = current_time
        if right_lifted_now and not self.right_is_up:
            self.right_lift_time = current_time

        self.left_is_up  = left_lifted_now
        self.right_is_up = right_lifted_now

        if self.left_is_up and self.right_is_up:
            if abs(self.left_lift_time - self.right_lift_time) <= HOP_SYNC_WINDOW:
                return "HOP DETECTED"
            return "Sequential step"
        if self.left_is_up:
            return "Left Foot in Air"
        if self.right_is_up:
            return "Right Foot in Air"
        return "Both Feet Grounded"


# ── Detector state machine ─────────────────────────────────────────────────────

class SplitStepDetector:
    def __init__(self):
        self.reset()

    def reset(self):
        self.count   = 0
        self.state   = GROUNDED
        self._tracker = FootStateTracker()

    def process(self, landmarks) -> bool:
        """Process one frame's world landmarks. Returns True if a hop was just counted."""
        if not all(landmarks[i].visibility >= MIN_VISIBILITY for i in KEY_LANDMARKS):
            return False

        result = self._tracker.process_frame(landmarks)

        if result == "Calibrating...":
            return False

        prev_state = self.state

        if result == "HOP DETECTED":
            self.state = AIRBORNE
        elif result == "Sequential step":
            self.state = AIRBORNE
        elif result == "Left Foot in Air":
            self.state = LEFT_LEG_UP
        elif result == "Right Foot in Air":
            self.state = RIGHT_LEG_UP
        else:
            self.state = GROUNDED

        hop = self.state == AIRBORNE and prev_state == GROUNDED
        if hop:
            self.count += 1

        return hop


# ── Analysis worker base ───────────────────────────────────────────────────────

class _AnalysisWorker:
    """
    Base class for analysis workers. Subclasses supply the video source and
    implement _timestamp_ms() to return a monotonically increasing timestamp
    in milliseconds for each frame.

    Pushes to frame_queue:
      ("frame", rgb_array, count, state)  — each processed frame
      ("done",  final_count)              — when source is exhausted or stopped
      ("error", message)                  — if the source cannot be opened
    """

    def __init__(self, frame_queue: queue.Queue, stop_event: threading.Event):
        self._queue = frame_queue
        self._stop  = stop_event

    def _source(self):
        """Return the cv2.VideoCapture source (int or str)."""
        raise NotImplementedError

    def _timestamp_ms(self, cap: cv2.VideoCapture) -> int:
        """Return the timestamp in ms to pass to MediaPipe for this frame."""
        raise NotImplementedError

    def run(self) -> None:
        options = mp_vision.PoseLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        detector = SplitStepDetector()
        cap = cv2.VideoCapture(self._source())

        if not cap.isOpened():
            self._queue.put(("error", "Could not open video source."))
            return

        try:
            with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
                while not self._stop.is_set():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    ts     = self._timestamp_ms(cap)
                    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    result = landmarker.detect_for_video(mp_img, ts)

                    hop = False
                    if result.pose_landmarks:
                        draw_landmarks(frame, result.pose_landmarks[0])
                    if result.pose_world_landmarks:
                        hop = detector.process(result.pose_world_landmarks[0])

                    annotate_frame(frame, detector, hop)
                    display = fit_frame(frame, CANVAS_W, CANVAS_H)
                    rgb_out = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)

                    try:
                        self._queue.put_nowait(
                            ("frame", rgb_out, detector.count, detector.state)
                        )
                    except queue.Full:
                        pass
        finally:
            cap.release()
            self._queue.put(("done", detector.count))
