"""Rate limiting and the video length cap."""

import cv2
import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import api.main
from api.limits import RateLimiter
from split_step.analyze import VideoTooLong, analyze_video

client = TestClient(api.main.app)


def test_sliding_window():
    now = [0.0]
    rl = RateLimiter(clock=lambda: now[0])
    for _ in range(3):
        rl.check("1.2.3.4", 3)
    with pytest.raises(HTTPException) as e:
        rl.check("1.2.3.4", 3)
    assert e.value.status_code == 429 and int(e.value.headers["Retry-After"]) > 0
    rl.check("5.6.7.8", 3)                      # other clients unaffected
    now[0] = 3601
    rl.check("1.2.3.4", 3)                      # window slid


def test_endpoint_rate_limited(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_HOUR", "2")
    monkeypatch.setattr("api.limits._limiter", RateLimiter())
    codes = [client.post("/v1/count", files={"file": ("a.mp4", b"x", "video/mp4")}).status_code for _ in range(3)]
    assert codes == [422, 422, 429]


def _clip(path, seconds, fps=10):
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
    for _ in range(int(seconds * fps)):
        out.write(np.zeros((48, 64, 3), np.uint8))
    out.release()


def test_too_long_video(tmp_path, monkeypatch):
    clip = tmp_path / "long.mp4"
    _clip(clip, 5)
    with pytest.raises(VideoTooLong):
        analyze_video(str(clip), "rules", max_seconds=2)
    assert analyze_video(str(clip), "rules", max_seconds=10)["frames"] == 50
    monkeypatch.setattr(api.main, "MAX_VIDEO_SECONDS", 2)
    with open(clip, "rb") as f:
        r = client.post("/v1/count", files={"file": ("long.mp4", f, "video/mp4")})
    assert r.status_code == 413 and "limit" in r.json()["detail"]
