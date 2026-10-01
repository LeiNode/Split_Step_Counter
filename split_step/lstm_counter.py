"""
LSTMCounter — counts split steps using a stateful, per-frame LSTM trained in
model_experimentation/lstm_model.ipynb.

Unlike the old fixed-window classifier, this model carries its LSTM hidden
state across frames and emits one probability per frame directly — no
rolling buffer, no re-encoding a window from scratch every call. This also
removes the old ~333ms mandatory warm-up before any prediction could be made.

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

from .detector import GROUNDED, AIRBORNE
from .paths import SPLIT_STEP_LSTM_PATH, FEAT_MEAN_PATH, FEAT_STD_PATH

# MediaPipe landmark indices, in the order expected by the model's feature vector
LANDMARK_INDICES = [
    27,  # left_ankle
    29,  # left_heel
    31,  # left_foot_index
    28,  # right_ankle
    30,  # right_heel
    32,  # right_foot_index
    25,  # left_knee
    26,  # right_knee
    23,  # left_hip
    24,  # right_hip
    11,  # left_shoulder
    12,  # right_shoulder
    13,  # left_elbow
    14,  # right_elbow
    15,  # left_wrist
    16,  # right_wrist
]

N_POS  = len(LANDMARK_INDICES) * 3   # 48  (x, y, z per landmark)
N_FEAT = N_POS * 2                   # 96  (position + velocity)

_THRESHOLD      = 0.8
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
    Exposes the same .count, .state, and .process() interface.
    """

    def __init__(
        self,
        model_path: str = SPLIT_STEP_LSTM_PATH,
        mean_path:  str = FEAT_MEAN_PATH,
        std_path:   str = FEAT_STD_PATH,
        device:     Optional[str] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self._mean = np.load(mean_path)
        self._std  = np.load(std_path)

        self._model = SplitStepLSTM().to(device)
        self._model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True)
        )
        self._model.eval()

        self.count   = 0
        self.state   = GROUNDED
        self._active = False
        self._last_t = 0.0

        self._prev_pos      = None
        self._hidden        = None
        self._last_call_t   = None

    def process(self, world_landmarks, image_lms=None) -> bool:
        """Feed one frame. Returns True when a new split step is counted."""
        now = time.time()
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

        is_on      = prob >= _THRESHOLD
        self.state = AIRBORNE if is_on else GROUNDED

        hop = False
        if is_on and not self._active:
            now_count = time.time()
            if (now_count - self._last_t) >= _COOLDOWN:
                self.count  += 1
                self._last_t = now_count
                hop          = True
        self._active = is_on
        return hop

    def reset(self):
        self.count   = 0
        self.state   = GROUNDED
        self._active = False
        self._last_t = 0.0

        self._prev_pos    = None
        self._hidden      = None
        self._last_call_t = None
