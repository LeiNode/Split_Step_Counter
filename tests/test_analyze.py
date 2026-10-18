"""analyze_video end to end on generated clips (no assets needed)."""

import cv2
import numpy as np

from split_step.analyze import analyze_video


def _clip(path, fps, seconds=2):
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (160, 120))
    for i in range(int(fps * seconds)):
        out.write(np.full((120, 160, 3), i % 255, np.uint8))
    out.release()


def test_high_fps_is_thinned_to_30(tmp_path):
    clip = tmp_path / "clip60.mp4"
    _clip(clip, 60)
    r = analyze_video(str(clip), "rules")
    assert r["frames"] == 120
    assert 55 <= r["frames_analysed"] <= 65          # ~30 fps worth
    full = analyze_video(str(clip), "rules", max_fps=None)
    assert full["frames_analysed"] == 120


def test_30fps_untouched(tmp_path):
    clip = tmp_path / "clip30.mp4"
    _clip(clip, 30)
    r = analyze_video(str(clip), "rules")
    assert r["frames_analysed"] == r["frames"] == 60
