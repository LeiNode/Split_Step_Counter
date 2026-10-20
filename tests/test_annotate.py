"""Annotated video export on a generated clip."""

import cv2
import numpy as np

from split_step.annotate import annotate_video
from tests.test_analyze import _clip


def _frame(path, index):
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, f = cap.read()
    cap.release()
    assert ok
    return f.astype(int)


def test_banners_and_skeleton_drawn(tmp_path):
    src, dst = tmp_path / "in.mp4", tmp_path / "out.mp4"
    _clip(src, 30)                                            # 2 s, flat grey frames
    xy = [0.5, 0.2, 0.5, 0.8] + [-1, -1] * 15                 # nose + one shoulder visible, 17 landmarks
    result = {"count": 1, "events": [{"timestamp_ms": 1000.0}, {"timestamp_ms": 300.0, "during_swing": True}],
              "strokes": {"swings": [{"timestamp_ms": 1000.0, "stroke": "forehand", "peak_wrist_speed": 9.1}]},
              "pose": {"landmarks": list(range(17)), "edges": [[0, 1]], "ts_ms": [0.0, 1000.0, 1500.0],
                       "xy": [None, xy, xy]}}
    assert annotate_video(str(src), result, str(dst)) == 60
    plain, before, during = _frame(src, 33), _frame(dst, 15), _frame(dst, 33)
    h = plain.shape[0]
    bottom = (slice(h - 40, h), slice(0, 120))
    assert np.abs(during[bottom] - plain[bottom]).max() > 50      # banner after the split step + swing
    assert np.abs(before[bottom] - _frame(src, 15)[bottom]).max() < 30   # nothing yet at 0.5 s
    assert np.abs(during[60, 75:85] - plain[60, 75:85]).max() > 50       # skeleton line at x = 0.5
