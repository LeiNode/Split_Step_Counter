"""Feature switches: resolution order, parent/child rules, and API enforcement."""

import json

import pytest
from fastapi.testclient import TestClient

from api.main import app
from split_step import features

client = TestClient(app)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    monkeypatch.delenv("SPLIT_STEP_FEATURES", raising=False)
    # point at a non-existent file so a developer's local features.json can't leak in
    monkeypatch.setenv("SPLIT_STEP_FEATURES_FILE", str(tmp_path / "none.json"))
    monkeypatch.setattr(features, "_REPO_FILE", str(tmp_path / "none.json"))
    monkeypatch.delenv("SPLIT_STEP_FEATURES_FILE")


def test_defaults_all_on():
    assert all(features.all_features().values())


def test_env_override(monkeypatch):
    monkeypatch.setenv("SPLIT_STEP_FEATURES", "serve_training=off, live_view=0")
    f = features.all_features()
    assert f["serve_training"] is False and f["live_view"] is False and f["forehand_training"] is True


def test_file_then_env(monkeypatch, tmp_path):
    path = tmp_path / "flags.json"
    path.write_text(json.dumps({"backhand_training": False, "rules_model": False}))
    monkeypatch.setenv("SPLIT_STEP_FEATURES_FILE", str(path))
    monkeypatch.setenv("SPLIT_STEP_FEATURES", "rules_model=on")   # env beats file
    f = features.all_features()
    assert f["backhand_training"] is False and f["rules_model"] is True


def test_parent_switch_disables_children(monkeypatch):
    monkeypatch.setenv("SPLIT_STEP_FEATURES", "stroke_training=off")
    f = features.all_features()
    assert not any(f[n] for n in ("forehand_training", "backhand_training", "serve_training",
                                  "auto_stroke_detection"))


def test_unknown_or_bad_values(monkeypatch):
    monkeypatch.setenv("SPLIT_STEP_FEATURES", "tennis_elbow=on")
    with pytest.raises(features.UnknownFeature):
        features.all_features()
    monkeypatch.setenv("SPLIT_STEP_FEATURES", "serve_training=maybe")
    with pytest.raises(ValueError):
        features.all_features()


def test_features_endpoint(monkeypatch):
    monkeypatch.setenv("SPLIT_STEP_FEATURES", "serve_training=off")
    body = client.get("/v1/features").json()
    assert body["features"]["serve_training"] is False
    assert "serve_training" in body["descriptions"]


@pytest.mark.parametrize("flags,url,status", [
    ("serve_training=off",   "/v1/count?drill=serve", 404),
    ("stroke_training=off",  "/v1/count?drill=forehand", 404),
    ("auto_stroke_detection=off", "/v1/count?drill=auto", 404),
    ("rules_model=off",      "/v1/count?model=rules", 404),
    ("direct_upload=off",    "/v1/count", 404),
    ("split_step_counting=off", "/v1/count", 400),        # nothing left to do without a drill
])
def test_disabled_features_rejected(monkeypatch, flags, url, status):
    monkeypatch.setenv("SPLIT_STEP_FEATURES", flags)
    r = client.post(url, files={"file": ("a.mp4", b"x", "video/mp4")})
    assert r.status_code == status, r.text


def test_disabled_browser_uploads(monkeypatch):
    monkeypatch.setenv("SPLIT_STEP_FEATURES", "browser_uploads=off")
    assert client.post("/v1/upload-url", json={}).status_code == 404
    assert client.post("/v1/count/s3", json={"bucket": "b", "key": "k"}).status_code == 404


def test_enabled_drill_passes_flag_check(monkeypatch):
    # Flags allow it, so the request gets as far as video decoding (422 for junk bytes).
    r = client.post("/v1/count?drill=serve", files={"file": ("a.mp4", b"x", "video/mp4")})
    assert r.status_code == 422
