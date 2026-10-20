"""
LSTMCounter — counts split steps using a stateful, per-frame LSTM trained in
model_experimentation/lstm_model.ipynb.

Unlike the old fixed-window classifier, this model carries its LSTM hidden
state across frames and emits one probability per frame directly — no
rolling buffer, no re-encoding a window from scratch every call. This also
removes the old ~333ms mandatory warm-up before any prediction could be made.

Per-frame features: 16 landmarks x (x, y, z) world positions, plus their
velocities (units/second) against the previous frame = 96 values, normalised
with the training-set mean/std.

Requires feat_mean.npy and feat_std.npy saved from the notebook:
    np.save("models/feat_mean.npy", feat_mean)
    np.save("models/feat_std.npy",  feat_std)

SplitStepLSTM is defined here (not duplicated in the notebook) so training
and live inference always agree on the model's architecture.
"""

import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .detector import (
    GROUNDED, AIRBORNE, key_landmarks_visible,
    LEFT_ANKLE, LEFT_HEEL, LEFT_FOOT_INDEX,
    RIGHT_ANKLE, RIGHT_HEEL, RIGHT_FOOT_INDEX,
    LEFT_KNEE, RIGHT_KNEE, LEFT_HIP, RIGHT_HIP,
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW,
    LEFT_WRIST, RIGHT_WRIST,
)
from .paths import SPLIT_STEP_LSTM_PATH, FEAT_MEAN_PATH, FEAT_STD_PATH

# Landmarks in the order expected by the model's feature vector. Must match
# FEATURE_LANDMARK_NAMES in the notebook and data_windowing/window_dataset.py —
# reordering silently breaks the trained model.
LANDMARK_INDICES = [
    LEFT_ANKLE,  LEFT_HEEL,  LEFT_FOOT_INDEX,
    RIGHT_ANKLE, RIGHT_HEEL, RIGHT_FOOT_INDEX,
    LEFT_KNEE,     RIGHT_KNEE,
    LEFT_HIP,      RIGHT_HIP,
    LEFT_SHOULDER, RIGHT_SHOULDER,
    LEFT_ELBOW,    RIGHT_ELBOW,
    LEFT_WRIST,    RIGHT_WRIST,
]

N_POS  = len(LANDMARK_INDICES) * 3   # 48  (x, y, z per landmark)
N_FEAT = N_POS * 2                   # 96  (position + velocity)

# Model size + threshold for the currently-deployed checkpoint ("Run 8" in
# model_experimentation/refinement_log.md). Both must be updated together if
# the checkpoint is ever replaced — the hidden size has to match whatever
# split_step_lstm.pt was actually trained with, and the threshold has to come
# from sweeping on a validation set (NEVER on test_set_labeled.csv's labels —
# see the log's "CORRECTION" section for what happened the one time that rule
# was broken: a 0.02 threshold picked by sweeping on test gave a fake 0.961
# recall that evaporated to 0.062 under honest evaluation).
#
# Full search history (9 training runs: regularization, capacity, LR,
# pos_weight, chunk length) found hidden_size to be the only lever that
# mattered, with a non-monotonic sweet spot — this is NOT "smaller is always
# better", recall peaked at 32 and partially reversed at 16:
#   hidden_size  128   64    32    16
#   recall      0.128 0.241 0.291 0.266
#   precision   0.745 0.748 0.579 0.850
# Run 8 (hidden_size=32) is deployed here for its higher recall, matching the
# project's stated priority of catching hops over avoiding false positives.
# Even so, honest held-out recall (0.291) is still slightly below the old
# pre-redesign windowed model's (0.300) — this redesign has not yet
# surpassed it. See refinement_log.md "FINAL SUMMARY" for the full picture
# and why more/diverse training data, not further tuning, is the likely next
# lever (TODO.md).
_HIDDEN_SIZE    = 32
_THRESHOLD      = 0.4    # val-selected for hidden_size=32 (Run 8) — re-sweep
                         # on validation (never on test) if the checkpoint changes.
_COOLDOWN       = 0.5    # seconds between counted events
_GAP_THRESHOLD_S = 0.15  # a tracking gap longer than this isn't bridged with a velocity


def extract_position(world_landmarks) -> np.ndarray:
    """Return the (N_POS,) float32 position vector for one frame."""
    return np.array(
        [c for idx in LANDMARK_INDICES
           for c in (world_landmarks[idx].x, world_landmarks[idx].y, world_landmarks[idx].z)],
        dtype=np.float32,
    )


class SplitStepLSTM(nn.Module):
    """
    Shared model definition used by both training (model_experimentation/
    lstm_model.ipynb) and live inference (LSTMCounter below).

    Call with a whole chunk (B, T, N_FEAT) for training (every timestep gets
    its own logit), or one frame (1, 1, N_FEAT) plus the hidden state carried
    from the previous call for streaming inference.
    """

    def __init__(self, input_size=N_FEAT, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x, hidden=None):
        """
        x      — (B, T, N_FEAT)
        hidden — optional (h, c) to continue a previous call; None starts fresh.
        Returns (logits, hidden) where logits is (B, T) — one logit per timestep.
        """
        out, hidden = self.lstm(x, hidden)
        logits = self.classifier(out).squeeze(-1)
        return logits, hidden


class LSTMCounter:
    """
    Drop-in replacement for SplitStepDetector that uses the stateful LSTM.
    Exposes the same .count, .state, .process() and .reset() interface.

    State is only ever GROUNDED or AIRBORNE (model probability >= threshold).
    A count is registered on each rising edge, at most once per _COOLDOWN.

    clock() -> seconds; defaults to wall time. Pass a video-time clock when
    feeding frames faster than real time, since velocities and the cooldown
    are computed from it.

    require_full_pose (default True) skips the model entirely on frames where
    any KEY_LANDMARKS point (hips, ankles, heels, toes) isn't visible: the
    model was only ever trained on such frames (build_labeled_dataset drops
    the rest), so its output on a cropped-feet frame is meaningless. A skipped
    stretch longer than _GAP_THRESHOLD_S restarts the hidden state, exactly
    like a gap in the training data. .tracking says whether the last frame
    was used.
    """

    def __init__(
        self,
        model_path: str = SPLIT_STEP_LSTM_PATH,
        mean_path:  str = FEAT_MEAN_PATH,
        std_path:   str = FEAT_STD_PATH,
        device:     Optional[str] = None,
        clock=None,
        require_full_pose: bool = True,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self._clock = clock or time.time
        self.require_full_pose = require_full_pose

        self._mean = np.load(mean_path)
        self._std  = np.load(std_path)

        self._model = SplitStepLSTM(hidden_size=_HIDDEN_SIZE).to(device)
        self._model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True)
        )
        self._model.eval()

        self.reset()

    def process(self, world_landmarks, image_lms=None) -> bool:
        """Feed one frame. Returns True when a new split step is counted.
        image_lms is accepted for interface parity with SplitStepDetector and ignored."""
        if self.require_full_pose and not key_landmarks_visible(world_landmarks):
            self.tracking = False
            return False
        self.tracking = True
        now = self._clock()
        dt  = (now - self._last_call_t) if self._last_call_t is not None else None
        self._last_call_t = now

        pos = extract_position(world_landmarks)

        if dt is None or dt > _GAP_THRESHOLD_S:
            # First frame ever, or a tracking gap too large to treat as
            # continuous motion — start a fresh sub-sequence rather than
            # computing a velocity across the gap or carrying stale state.
            vel = np.zeros_like(pos)
            self._hidden = None
        else:
            vel = (pos - self._prev_pos) / dt
        self._prev_pos = pos

        feat = (np.concatenate([pos, vel]) - self._mean) / self._std
        x = torch.from_numpy(feat).view(1, 1, -1).float().to(self.device)

        with torch.no_grad():
            logit, self._hidden = self._model(x, self._hidden)
            prob = torch.sigmoid(logit).item()
        self.last_prob = prob

        is_on      = prob >= _THRESHOLD
        self.state = AIRBORNE if is_on else GROUNDED

        hop = False
        if is_on and not self._active:
            now_count = self._clock()
            if (now_count - self._last_t) >= _COOLDOWN:
                self.count  += 1
                self._last_t = now_count
                hop          = True
        self._active = is_on
        return hop

    def reset(self):
        """Zero the count and drop all carried state (hidden state, last position)."""
        self.count    = 0
        self.state    = GROUNDED
        self.tracking = False
        self.last_prob = 0.0       # model probability on the last frame it ran on
        self._active  = False
        self._last_t = 0.0

        self._prev_pos    = None
        self._hidden      = None
        self._last_call_t = None
