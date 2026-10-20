"""
API tests. Run from the repo root:  python -m pytest tests -q

Uses the real models and a training video from assets/ (skipped if the
local-only video isn't present, e.g. on a fresh clone).
"""

import os
import pathlib

import pytest
from fastapi.testclient import TestClient

from api.main import app

REPO   = pathlib.Path(__file__).resolve().parent.parent
VIDEO  = REPO / "assets" / "split_step_videos" / "Training Set 1" / "video_1.mp4"
needs_video = pytest.mark.skipif(not VIDEO.exists(), reason="local training video not available")

client = TestClient(app)


def _upload(model="lstm", path=VIDEO, headers=None):
    with open(path, "rb") as f:
        return client.post(f"/v1/count?model={model}", headers=headers or {},
                           files={"file": (path.name, f, "video/mp4")})


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


@needs_video
@pytest.mark.parametrize("model", ["lstm", "rules"])
def test_count_upload(model):
    r = _upload(model)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model"] == model
    assert body["count"] == len(body["events"]) >= 0
    assert body["frames"] > 300 and 0 < body["frames_with_pose"] <= body["frames"]
    assert body["duration_ms"] > 10_000
    s = body["session"]
    assert 0 < s["pose_coverage"] <= 1 and s["duration_s"] > 10 and s["tips"]
    for e in body["events"]:
        assert {"hop_height", "landing_knee_angle", "landing_stance_ratio"} <= e.keys()
        if e["landing_knee_angle"] is not None:
            assert 30 < e["landing_knee_angle"] <= 180


@needs_video
def test_count_is_deterministic():
    # Video-time clock => processing speed must not change the result.
    assert _upload("rules").json()["events"] == _upload("rules").json()["events"]


def test_rejects_non_video(tmp_path):
    bad = tmp_path / "notes.mp4"
    bad.write_bytes(b"this is not a video")
    r = _upload(path=bad)
    assert r.status_code == 422


def test_rejects_empty(tmp_path):
    empty = tmp_path / "empty.mp4"
    empty.write_bytes(b"")
    assert _upload(path=empty).status_code == 400


def test_bad_model_name():
    r = client.post("/v1/count?model=gpt", files={"file": ("a.mp4", b"x", "video/mp4")})
    assert r.status_code == 422


def test_upload_limit(tmp_path, monkeypatch):
    import api.main
    monkeypatch.setattr(api.main, "MAX_UPLOAD_BYTES", 10)
    big = tmp_path / "big.mp4"
    big.write_bytes(b"0" * 100)
    assert _upload(path=big).status_code == 413


def test_api_key(monkeypatch, tmp_path):
    monkeypatch.setenv("API_KEY", "secret")
    f = tmp_path / "x.mp4"
    f.write_bytes(b"x")
    assert _upload(path=f).status_code == 401
    assert _upload(path=f, headers={"X-API-Key": "wrong"}).status_code == 401
    assert _upload(path=f, headers={"X-API-Key": "secret"}).status_code == 422  # authed; bad video
    assert client.get("/health").status_code == 200  # health stays open


def test_lambda_handler_imports():
    from api.lambda_handler import handler
    assert callable(handler)


# ── Public-website features ───────────────────────────────────────────────────

@pytest.fixture
def fake_aws(monkeypatch):
    # Presigning is pure local crypto; fake credentials are enough.
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAFAKE")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "fake")
    monkeypatch.setenv("AWS_REGION", "us-west-2")
    monkeypatch.setenv("UPLOAD_BUCKET", "my-uploads")


def test_cors_preflight(monkeypatch):
    import importlib
    import api.main
    monkeypatch.setenv("ALLOWED_ORIGINS", "https://www.example.com")
    mod = importlib.reload(api.main)  # CORS is configured at import time
    try:
        c = TestClient(mod.app)
        ok = c.options("/v1/count", headers={"Origin": "https://www.example.com",
                                             "Access-Control-Request-Method": "POST"})
        assert ok.headers.get("access-control-allow-origin") == "https://www.example.com"
        bad = c.options("/v1/count", headers={"Origin": "https://evil.example",
                                              "Access-Control-Request-Method": "POST"})
        assert "access-control-allow-origin" not in bad.headers
    finally:
        monkeypatch.delenv("ALLOWED_ORIGINS")
        importlib.reload(api.main)


def test_upload_url_requires_bucket(monkeypatch):
    monkeypatch.delenv("UPLOAD_BUCKET", raising=False)
    assert client.post("/v1/upload-url", json={"filename": "a.mp4"}).status_code == 501


def test_upload_url_issues_presigned_post(fake_aws):
    r = client.post("/v1/upload-url", json={"filename": "Match Day.MOV"})
    assert r.status_code == 200, r.text
    t = r.json()
    assert t["bucket"] == "my-uploads"
    assert t["key"].startswith("uploads/") and t["key"].endswith(".mov")
    assert t["url"] == "https://my-uploads.s3.us-west-2.amazonaws.com/"
    assert t["fields"]["key"] == t["key"] and "policy" in t["fields"]


@pytest.mark.parametrize("bucket,key", [
    ("other-bucket", "uploads/" + "a" * 32 + ".mp4"),   # wrong bucket
    ("my-uploads",   "private/secret.mp4"),            # outside the upload prefix
    ("my-uploads",   "uploads/../private/x.mp4"),      # not an issued key
])
def test_count_s3_locked_to_issued_keys(fake_aws, bucket, key):
    r = client.post("/v1/count/s3", json={"bucket": bucket, "key": key})
    assert r.status_code == 403


def test_web_app_served():
    r = client.get("/app/")
    assert r.status_code == 200 and "Tennis Trainer" in r.text


def test_markdown_report():
    import json
    result = json.loads((REPO / "tests" / "web" / "analysis_fixture.json").read_text(encoding="utf-8"))
    r = client.post("/v1/report", json={"result": result, "title": "match.mp4"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/markdown")
    assert r.text.startswith("# match.mp4\n") and "## Strokes" in r.text
    assert client.post("/v1/report", json={"result": {"count": 3}}).status_code == 422


def test_markdown_report_switch(monkeypatch):
    monkeypatch.setenv("SPLIT_STEP_FEATURES", "markdown_report=off")
    assert client.post("/v1/report", json={"result": {}}).status_code == 404
