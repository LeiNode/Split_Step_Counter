"""
LSTMCounter — counts split steps using the binary LSTM trained in
model_experimentation/lstm_model.ipynb.

Requires feat_mean.npy and feat_std.npy saved from the notebook:
    np.save("models/feat_mean.npy", feat_mean)
    np.save("models/feat_std.npy",  feat_std)
"""

import time
from collections import deque
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .detector import GROUNDED, AIRBORNE

# MediaPipe landmark indices matching the notebook's FEATURE_LANDMARK_NAMES order
_LANDMARK_INDICES = [
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

_N_POS    = len(_LANDMARK_INDICES) * 3   # 48  (x, y, z per landmark)
_N_FEAT   = _N_POS * 2                   # 96  (positions + velocities)
_SEQ_LEN  = 10
_THRESHOLD = 0.8
_COOLDOWN  = 0.5   # seconds between counted events


class _BinaryLSTM(nn.Module):
    def __init__(self, input_size=_N_FEAT, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.classifier(out[:, -1, :]).squeeze(1)


class LSTMCounter:
    """
    Drop-in replacement for SplitStepDetector that uses the binary LSTM model.
    Exposes the same .count, .state, and .process() interface.
    """

    def __init__(
        self,
        model_path: str = "models/split_step_lstm.pt",
        mean_path:  str = "models/feat_mean.npy",
        std_path:   str = "models/feat_std.npy",
        device:     Optional[str] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self._mean = np.load(mean_path)
        self._std  = np.load(std_path)

        self._model = _BinaryLSTM().to(device)
        self._model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True)
        )
        self._model.eval()

        self.count   = 0
        self.state   = GROUNDED
        self._buf    = deque(maxlen=_SEQ_LEN)
        self._prev   = None
        self._active = False
        self._last_t = 0.0

    def process(self, world_landmarks, image_lms=None) -> bool:
        """Feed one frame. Returns True when a new split step is counted."""
        pos = np.array(
            [c for idx in _LANDMARK_INDICES
               for c in (world_landmarks[idx].x,
                         world_landmarks[idx].y,
                         world_landmarks[idx].z)],
            dtype=np.float32,
        )
        vel = pos - self._prev if self._prev is not None else np.zeros_like(pos)
        self._prev = pos

        frame = (np.concatenate([pos, vel]) - self._mean) / self._std
        self._buf.append(frame)

        if len(self._buf) < _SEQ_LEN:
            return False

        x = torch.from_numpy(np.stack(self._buf)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            prob = torch.sigmoid(self._model(x)).item()

        is_on      = prob >= _THRESHOLD
        self.state = AIRBORNE if is_on else GROUNDED

        hop = False
        if is_on and not self._active:
            now = time.time()
            if (now - self._last_t) >= _COOLDOWN:
                self.count  += 1
                self._last_t = now
                hop          = True
        self._active = is_on
        return hop

    def reset(self):
        self.count   = 0
        self.state   = GROUNDED
        self._buf.clear()
        self._prev   = None
        self._active = False
        self._last_t = 0.0
