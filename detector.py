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
LEFT_SHOULDER    = 11
RIGHT_SHOULDER   = 12
LEFT_ELBOW       = 13
RIGHT_ELBOW      = 14
LEFT_WRIST       = 15
RIGHT_WRIST      = 16
LEFT_HIP         = 23
RIGHT_HIP        = 24
LEFT_ANKLE       = 27
RIGHT_ANKLE      = 28
LEFT_HEEL        = 29
RIGHT_HEEL       = 30
LEFT_FOOT_INDEX  = 31
RIGHT_FOOT_INDEX = 32

POSE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
    (9,10),(11,12),(11,13),(13,15),(15,17),(15,19),(15,21),(17,19),
    (12,14),(14,16),(16,18),(16,20),(16,22),(18,20),
    (11,23),(12,24),(23,24),(23,25),(24,26),(25,27),(26,28),(27,29),
    (28,30),(29,31),(30,32),(27,31),(28,32),
]

# ── Tuning ─────────────────────────────────────────────────────────────────────
MIN_VISIBILITY      = 0.5   # landmarks below this confidence are treated as out-of-frame
HOP_SYNC_WINDOW     = 0.30  # seconds between lifts to count as a simultaneous hop
MIN_GROUNDED_FRAMES  = 3    # consecutive grounded frames required before a hop counts
MIN_AIRBORNE_FRAMES  = 2    # both-feet-airborne frames required to confirm a real hop
VEL_EMA_ALPHA        = 0.4    # smoothing for per-foot velocity (0=laggy, 1=no smoothing)
VEL_LIFT_MIN         = 0.001  # m/frame minimum upward EMA velocity to count a landmark as rising
VEL_SIMILARITY_RATIO = 0.3    # slower foot must reach at least this fraction of the faster foot's upward speed
TORSO_BOX_MARGIN     = 0.20 # metres of extra padding added to each side of the torso box
WRIST_PROXIMITY      = 0.55 # max 3-D distance (metres) between wrists to count as close together
MAX_ELBOW_ANGLE      = 155  # degrees; above this an arm is too extended for a ready stance
MAX_FOOT_HEIGHT_DIFF = 0.20 # max y-gap (metres) between ankles at takeoff; running cross-steps exceed this

KEY_LANDMARKS = [LEFT_HIP, RIGHT_HIP, LEFT_ANKLE, RIGHT_ANKLE,
                 LEFT_HEEL, RIGHT_HEEL, LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX]

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
    _LEFT_LANDMARKS  = (LEFT_ANKLE,  LEFT_HEEL,  LEFT_FOOT_INDEX)
    _RIGHT_LANDMARKS = (RIGHT_ANKLE, RIGHT_HEEL, RIGHT_FOOT_INDEX)

    def __init__(self):
        self.left_lift_time  = 0.0
        self.right_lift_time = 0.0
        self.left_is_up  = False
        self.right_is_up = False
        all_lms = self._LEFT_LANDMARKS + self._RIGHT_LANDMARKS
        self._prev_y = {i: None for i in all_lms}
        self._vel    = {i: 0.0  for i in all_lms}

    def reset(self):
        self.__init__()

    @property
    def left_vel(self) -> float:
        return sum(self._vel[i] for i in self._LEFT_LANDMARKS) / 3

    @property
    def right_vel(self) -> float:
        return sum(self._vel[i] for i in self._RIGHT_LANDMARKS) / 3

    def _update_velocities(self, lms) -> None:
        a = VEL_EMA_ALPHA
        for idx in self._prev_y:
            y = lms[idx].y
            if self._prev_y[idx] is not None:
                raw = y - self._prev_y[idx]
                self._vel[idx] = a * raw + (1 - a) * self._vel[idx]
            self._prev_y[idx] = y

    def _all_rising(self, indices) -> bool:
        """True when every landmark in the group has upward EMA velocity above the noise floor."""
        return all(self._vel[i] > VEL_LIFT_MIN for i in indices)

    def process_frame(self, world_landmarks) -> str:
        current_time = time.time()
        self._update_velocities(world_landmarks)

        left_rising  = self._all_rising(self._LEFT_LANDMARKS)
        right_rising = self._all_rising(self._RIGHT_LANDMARKS)

        if left_rising and not self.left_is_up:
            self.left_lift_time = current_time
        if right_rising and not self.right_is_up:
            self.right_lift_time = current_time

        self.left_is_up  = left_rising
        self.right_is_up = right_rising

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
        self.count             = 0
        self.state             = GROUNDED
        self._tracker          = FootStateTracker()
        self._grounded_streak  = 0
        self._last_grnd_streak = 0
        self._airborne_frames  = 0
        self._hop_was_valid    = False

    @staticmethod
    def _velocities_similar(tracker: "FootStateTracker") -> bool:
        """Both feet must be rising above the noise floor and at similar upward speeds."""
        vl, vr = tracker.left_vel, tracker.right_vel
        if vl <= VEL_LIFT_MIN or vr <= VEL_LIFT_MIN:
            return False
        return min(vl, vr) / max(vl, vr) >= VEL_SIMILARITY_RATIO

    @staticmethod
    def _hands_in_ready_position(landmarks) -> bool:
        """Both wrists must lie within the torso bounding box (shoulders → hips),
        expanded by TORSO_BOX_MARGIN on every side. Falls back to True when any
        reference landmark is not visible."""
        lw = landmarks[LEFT_WRIST]
        rw = landmarks[RIGHT_WRIST]
        ls = landmarks[LEFT_SHOULDER]
        rs = landmarks[RIGHT_SHOULDER]
        lh = landmarks[LEFT_HIP]
        rh = landmarks[RIGHT_HIP]

        if any(lm.visibility < MIN_VISIBILITY for lm in (lw, rw, ls, rs, lh, rh)):
            return True

        m = TORSO_BOX_MARGIN
        box_x_min = min(ls.x, lh.x) - m
        box_x_max = max(rs.x, rh.x) + m
        box_y_min = min(lh.y, rh.y) - m
        box_y_max = max(ls.y, rs.y) + m

        def in_box(w) -> bool:
            return box_x_min <= w.x <= box_x_max and box_y_min <= w.y <= box_y_max

        return in_box(lw) and in_box(rw)

    @staticmethod
    def _wrists_close_together(landmarks) -> bool:
        """Both wrists must be within WRIST_PROXIMITY metres of each other (3-D).
        Falls back to True when wrists are not visible."""
        lw = landmarks[LEFT_WRIST]
        rw = landmarks[RIGHT_WRIST]
        if lw.visibility < MIN_VISIBILITY or rw.visibility < MIN_VISIBILITY:
            return True
        dist = np.linalg.norm([lw.x - rw.x, lw.y - rw.y, lw.z - rw.z])
        return dist <= WRIST_PROXIMITY

    @staticmethod
    def _elbows_bent(landmarks) -> bool:
        """Both elbows must be bent below MAX_ELBOW_ANGLE degrees.
        A straightened arm indicates a stroke follow-through, not a ready stance.
        Falls back to True for any arm whose landmarks are not visible."""
        for s_idx, e_idx, w_idx in ((LEFT_SHOULDER,  LEFT_ELBOW,  LEFT_WRIST),
                                     (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST)):
            s = landmarks[s_idx]
            e = landmarks[e_idx]
            w = landmarks[w_idx]
            if any(lm.visibility < MIN_VISIBILITY for lm in (s, e, w)):
                continue  # can't assess this arm — don't penalise
            v1 = np.array([s.x - e.x, s.y - e.y, s.z - e.z])
            v2 = np.array([w.x - e.x, w.y - e.y, w.z - e.z])
            cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
            angle = np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))
            if angle > MAX_ELBOW_ANGLE:
                return False  # this arm is too straight — block the count
        return True

    @staticmethod
    def _feet_at_similar_height(landmarks) -> bool:
        """Both ankles must be within MAX_FOOT_HEIGHT_DIFF metres of each other vertically.
        During a running cross-step the swing ankle is much higher than the stance ankle."""
        la = landmarks[LEFT_ANKLE]
        ra = landmarks[RIGHT_ANKLE]
        if la.visibility < MIN_VISIBILITY or ra.visibility < MIN_VISIBILITY:
            return True
        return abs(la.y - ra.y) <= MAX_FOOT_HEIGHT_DIFF

    def process(self, landmarks) -> bool:
        """Process one frame's world landmarks. Returns True if a hop was just confirmed."""
        if not all(landmarks[i].visibility >= MIN_VISIBILITY for i in KEY_LANDMARKS):
            return False

        result = self._tracker.process_frame(landmarks)

        prev_state = self.state

        if result in ("HOP DETECTED", "Sequential step"):
            self.state = AIRBORNE
        elif result == "Left Foot in Air":
            self.state = LEFT_LEG_UP
        elif result == "Right Foot in Air":
            self.state = RIGHT_LEG_UP
        else:
            self.state = GROUNDED

        # ── Grounded streak ──────────────────────────────────────────────────
        if self.state == GROUNDED:
            self._grounded_streak += 1
        else:
            if prev_state == GROUNDED:
                self._last_grnd_streak = self._grounded_streak
            if self.state not in (AIRBORNE,):
                self._grounded_streak = 0  # entering a leg-up state

        ready = (self._grounded_streak  >= MIN_GROUNDED_FRAMES or
                 self._last_grnd_streak >= MIN_GROUNDED_FRAMES)

        # ── Airborne duration + landing confirmation ─────────────────────────
        hop = False
        if self.state == AIRBORNE:
            if prev_state != AIRBORNE:
                # Just took off: record whether the ready condition was met.
                self._hop_was_valid   = (ready
                                         and self._velocities_similar(self._tracker)
                                         and self._feet_at_similar_height(landmarks)
                                         and self._hands_in_ready_position(landmarks)
                                         and self._wrists_close_together(landmarks)
                                         and self._elbows_bent(landmarks))
                self._grounded_streak = 0
                self._last_grnd_streak = 0
            self._airborne_frames += 1
        elif prev_state == AIRBORNE:
            # Just landed: confirm the hop only if it was valid AND long enough
            # to distinguish a real hop from a brief lateral weight transfer.
            if self._hop_was_valid and self._airborne_frames >= MIN_AIRBORNE_FRAMES:
                self.count += 1
                hop = True
            self._airborne_frames = 0
            self._hop_was_valid   = False

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

    def _frame_delay_s(self, cap: cv2.VideoCapture) -> float:
        """Seconds to wait after processing each frame. Override to pace playback."""
        return 0.0

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
                    frame_start = time.perf_counter()
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

                    delay = self._frame_delay_s(cap)
                    if delay > 0:
                        elapsed = time.perf_counter() - frame_start
                        remaining = delay - elapsed
                        if remaining > 0:
                            time.sleep(remaining)
        finally:
            cap.release()
            self._queue.put(("done", detector.count))
