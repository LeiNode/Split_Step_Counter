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


def test_analyse_part_of_a_video(tmp_path):
    import pytest
    from split_step.analyze import VideoError, VideoTooLong, parse_time
    clip = tmp_path / "clip30.mp4"
    _clip(clip, 30, seconds=4)
    r = analyze_video(str(clip), "rules", start_s=1.0, end_s=2.5)
    assert 40 <= r["frames"] <= 50                       # ~1.5 s at 30 fps
    assert 1000 <= r["range_ms"][0] < 1100 and 2400 < r["range_ms"][1] <= 2500
    assert "range_ms" not in analyze_video(str(clip), "rules")
    with pytest.raises(VideoError):
        analyze_video(str(clip), "rules", start_s=10)
    with pytest.raises(ValueError):
        analyze_video(str(clip), "rules", start_s=2, end_s=1)
    with pytest.raises(VideoTooLong):                    # the limit applies to the chosen part
        analyze_video(str(clip), "rules", start_s=0.5, max_seconds=3)
    assert analyze_video(str(clip), "rules", start_s=1.5, max_seconds=3)["frames"] > 0
    assert parse_time("1:15") == 75 and parse_time("1:01:15.5") == 3675.5 and parse_time("90") == 90
