"""
LSTM-based split-step classifier.

Pipeline
--------
1. Record labelled frames while running the detector using FrameRecorder.
2. Persist / reload recordings with save_dataset / load_dataset.
3. Train the model with train().
4. Run real-time inference on a rolling frame window with Predictor.

Install PyTorch before use:
    pip install torch
"""

import json
import pathlib
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from detector import (
    LEFT_ANKLE, LEFT_HEEL, LEFT_FOOT_INDEX,
    RIGHT_ANKLE, RIGHT_HEEL, RIGHT_FOOT_INDEX,
    LEFT_HIP, RIGHT_HIP,
    LEFT_SHOULDER, RIGHT_SHOULDER,
    LEFT_ELBOW, RIGHT_ELBOW,
    LEFT_WRIST, RIGHT_WRIST,
    GROUNDED, AIRBORNE, LEFT_LEG_UP, RIGHT_LEG_UP,
)

# ── Configuration ──────────────────────────────────────────────────────────────

FEATURE_LANDMARKS = [
    LEFT_ANKLE,  LEFT_HEEL,  LEFT_FOOT_INDEX,
    RIGHT_ANKLE, RIGHT_HEEL, RIGHT_FOOT_INDEX,
    LEFT_HIP,    RIGHT_HIP,
    LEFT_SHOULDER, RIGHT_SHOULDER,
    LEFT_ELBOW,  RIGHT_ELBOW,
    LEFT_WRIST,  RIGHT_WRIST,
]

LABEL_MAP   = {GROUNDED: 0, LEFT_LEG_UP: 1, RIGHT_LEG_UP: 2, AIRBORNE: 3}
LABEL_NAMES = [GROUNDED, LEFT_LEG_UP, RIGHT_LEG_UP, AIRBORNE]

N_FEATURES = len(FEATURE_LANDMARKS) * 3   # x, y, z per landmark
N_CLASSES  = len(LABEL_MAP)
SEQ_LEN    = 30                            # frames per window (~1 s at 30 fps)

# ── Feature extraction ─────────────────────────────────────────────────────────

def extract_features(world_landmarks) -> np.ndarray:
    """Return a (N_FEATURES,) float32 vector from one frame of world landmarks."""
    vec = []
    for idx in FEATURE_LANDMARKS:
        lm = world_landmarks[idx]
        vec.extend([lm.x, lm.y, lm.z])
    return np.array(vec, dtype=np.float32)

# ── Dataset ────────────────────────────────────────────────────────────────────

def build_windows(
    feature_frames: List[np.ndarray],
    label_frames:   List[int],
    seq_len:        int = SEQ_LEN,
) -> Tuple[List[np.ndarray], List[int]]:
    """Slice a continuous recording into overlapping sliding windows."""
    seqs, labels = [], []
    for i in range(seq_len - 1, len(feature_frames)):
        window = np.stack(feature_frames[i - seq_len + 1 : i + 1])
        seqs.append(window)
        labels.append(label_frames[i])
    return seqs, labels


class SplitStepDataset(Dataset):
    """
    Each sample is (seq, label) where:
      seq   — float32 tensor of shape (SEQ_LEN, N_FEATURES)
      label — int64 scalar: class of the last frame in the window
    """

    def __init__(self, seqs: List[np.ndarray], labels: List[int]):
        self.X = [torch.from_numpy(s) for s in seqs]
        self.y = [torch.tensor(l, dtype=torch.long) for l in labels]

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]

# ── Persistence ────────────────────────────────────────────────────────────────

def save_dataset(
    feature_frames: List[np.ndarray],
    label_frames:   List[int],
    path:           str,
) -> None:
    payload = {
        "features": [f.tolist() for f in feature_frames],
        "labels":   label_frames,
    }
    pathlib.Path(path).write_text(json.dumps(payload))


def load_dataset(path: str) -> Tuple[List[np.ndarray], List[int]]:
    payload = json.loads(pathlib.Path(path).read_text())
    features = [np.array(f, dtype=np.float32) for f in payload["features"]]
    return features, payload["labels"]

# ── Model ──────────────────────────────────────────────────────────────────────

class SplitStepLSTM(nn.Module):
    def __init__(
        self,
        input_size:  int   = N_FEATURES,
        hidden_size: int   = 128,
        num_layers:  int   = 2,
        num_classes: int   = N_CLASSES,
        dropout:     float = 0.3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)        # (batch, seq_len, hidden_size)
        last   = out[:, -1, :]       # use the last time-step's hidden state
        return self.classifier(last) # (batch, num_classes)

# ── Training ───────────────────────────────────────────────────────────────────

def train(
    feature_frames: List[np.ndarray],
    label_frames:   List[int],
    model_path:     str   = "split_step_lstm.pt",
    seq_len:        int   = SEQ_LEN,
    epochs:         int   = 40,
    batch_size:     int   = 32,
    lr:             float = 1e-3,
    val_split:      float = 0.15,
    device:         Optional[str] = None,
) -> SplitStepLSTM:
    """Train the LSTM on recorded frames and save weights to model_path."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}")

    seqs, labels = build_windows(feature_frames, label_frames, seq_len)
    if not seqs:
        raise ValueError("Not enough frames to build windows — record more data.")

    split    = int(len(seqs) * (1 - val_split))
    train_ds = SplitStepDataset(seqs[:split],  labels[:split])
    val_ds   = SplitStepDataset(seqs[split:],  labels[split:])
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    model     = SplitStepLSTM().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

    best_val_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_dl:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_dl:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds    = model(X_batch).argmax(dim=1)
                correct += (preds == y_batch).sum().item()
                total   += y_batch.size(0)

        val_acc = correct / total if total else 0.0
        scheduler.step(1 - val_acc)
        print(f"Epoch {epoch:3d}/{epochs}  "
              f"loss={running_loss / len(train_dl):.4f}  "
              f"val_acc={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_path)

    print(f"Best val_acc={best_val_acc:.3f} — model saved → {model_path}")
    return model

# ── Inference ──────────────────────────────────────────────────────────────────

class Predictor:
    """
    Maintains a rolling frame buffer and classifies each incoming frame.

    Usage
    -----
        predictor = Predictor("split_step_lstm.pt")
        label, confidence = predictor.update(world_landmarks)
    """

    def __init__(
        self,
        model_path: str = "split_step_lstm.pt",
        seq_len:    int = SEQ_LEN,
        device:     Optional[str] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device  = device
        self.seq_len = seq_len
        self.buffer: List[np.ndarray] = []

        self.model = SplitStepLSTM().to(device)
        self.model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True)
        )
        self.model.eval()

    def update(self, world_landmarks) -> Tuple[str, float]:
        """
        Feed one frame of world landmarks.
        Returns (label_name, confidence) once the buffer is full;
        returns (GROUNDED, 0.0) while still warming up.
        """
        self.buffer.append(extract_features(world_landmarks))
        if len(self.buffer) > self.seq_len:
            self.buffer.pop(0)
        if len(self.buffer) < self.seq_len:
            return GROUNDED, 0.0

        x = torch.from_numpy(np.stack(self.buffer)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(x), dim=1)[0]
            cls   = probs.argmax().item()
        return LABEL_NAMES[cls], float(probs[cls])

# ── Data collection ────────────────────────────────────────────────────────────

class FrameRecorder:
    """
    Accumulates labelled frames during a live or video session.

    Usage
    -----
        recorder = FrameRecorder()
        recorder.collect(world_landmarks, detector.state)   # call each frame
        recorder.save("my_session.json")
    """

    def __init__(self):
        self.feature_frames: List[np.ndarray] = []
        self.label_frames:   List[int]        = []

    def collect(self, world_landmarks, label: str) -> None:
        self.feature_frames.append(extract_features(world_landmarks))
        self.label_frames.append(LABEL_MAP[label])

    def save(self, path: str) -> None:
        save_dataset(self.feature_frames, self.label_frames, path)
        print(f"Saved {len(self.feature_frames)} frames → {path}")

    def clear(self) -> None:
        self.feature_frames.clear()
        self.label_frames.clear()

    def __len__(self) -> int:
        return len(self.feature_frames)
