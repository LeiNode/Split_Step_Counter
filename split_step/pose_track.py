"""
PoseTrack — the whole video's pose landmarks as dense numpy arrays.

analyze.py records one of these while it runs the counters, so every
post-hoc tennis metric (split-step quality, movement, swings, ...) can work
on plain arrays instead of re-running pose detection.

Frames without a detected pose are kept, with NaN landmarks, so indices line
up 1:1 with decoded frames and timestamps.
"""

from dataclasses import dataclass

import numpy as np

N_LANDMARKS = 33


@dataclass
class PoseTrack:
    ts_ms:  np.ndarray   # (N,)        frame timestamps, strictly increasing
    world:  np.ndarray   # (N, 33, 4)  x, y, z (metres, hip-relative), visibility
    image:  np.ndarray   # (N, 33, 3)  x, y (normalised 0-1), visibility
    aspect: float        # frame width / height, to compare image x and y distances

    def __len__(self) -> int:
        return len(self.ts_ms)

    @property
    def present(self) -> np.ndarray:
        """(N,) bool — a pose was detected in this frame."""
        return ~np.isnan(self.world[:, 0, 0])

    def window(self, start_ms: float, end_ms: float) -> np.ndarray:
        """Indices of frames with start_ms <= ts <= end_ms."""
        return np.flatnonzero((self.ts_ms >= start_ms) & (self.ts_ms <= end_ms))

    def visible(self, idx: int, landmarks, min_vis: float = 0.5) -> bool:
        """True if every landmark in `landmarks` is detected and visible in frame idx."""
        v = self.world[idx, list(landmarks), 3]
        return bool(np.all(v >= min_vis))  # NaN compares False -> not visible

    def image_xy(self, idx, landmark: int) -> np.ndarray:
        """Image position(s) with x rescaled by the aspect ratio, so x and y
        distances are in the same unit (fractions of frame height)."""
        xy = self.image[idx, landmark, :2].copy()
        xy[..., 0] *= self.aspect
        return xy


class PoseTrackBuilder:
    """Accumulates per-frame landmarks; call build() once at the end."""

    def __init__(self, frame_w: int, frame_h: int):
        self._aspect = (frame_w / frame_h) if frame_h else 1.0
        self._ts, self._world, self._image = [], [], []

    def add(self, ts_ms: float, world_lms, img_lms) -> None:
        w = np.full((N_LANDMARKS, 4), np.nan, dtype=np.float32)
        i = np.full((N_LANDMARKS, 3), np.nan, dtype=np.float32)
        if world_lms is not None:
            w[:] = [(lm.x, lm.y, lm.z, lm.visibility) for lm in world_lms[:N_LANDMARKS]]
        if img_lms is not None:
            i[:] = [(lm.x, lm.y, lm.visibility) for lm in img_lms[:N_LANDMARKS]]
        self._ts.append(ts_ms)
        self._world.append(w)
        self._image.append(i)

    def build(self) -> PoseTrack:
        n = len(self._ts)
        return PoseTrack(
            ts_ms=np.asarray(self._ts, dtype=np.float64),
            world=np.stack(self._world) if n else np.zeros((0, N_LANDMARKS, 4), np.float32),
            image=np.stack(self._image) if n else np.zeros((0, N_LANDMARKS, 3), np.float32),
            aspect=self._aspect,
        )
