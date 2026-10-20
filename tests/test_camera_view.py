"""Camera-angle classification on synthetic poses."""

import numpy as np

from split_step.camera_view import camera_report, frame_views
from split_step.detector import LEFT_SHOULDER, RIGHT_SHOULDER
from tests.test_strokes import _body, _track


def _view_track(n, lat, depth, left_on_image_right):
    w = _body(n)
    w[:, LEFT_SHOULDER, :3] = (-lat, -0.5, -depth)
    w[:, RIGHT_SHOULDER, :3] = (lat, -0.5, depth)
    t = _track(w)
    t.image[:, LEFT_SHOULDER, 0] = 0.6 if left_on_image_right else 0.4
    t.image[:, RIGHT_SHOULDER, 0] = 0.4 if left_on_image_right else 0.6
    return t


def test_views():
    assert set(frame_views(_view_track(3, 0.2, 0.02, False))) == {"back"}
    assert set(frame_views(_view_track(3, 0.2, 0.02, True))) == {"front"}
    assert set(frame_views(_view_track(3, 0.05, 0.2, False))) == {"side"}


def test_report_warns_on_weak_views():
    back = camera_report(_view_track(10, 0.2, 0.02, False))
    assert back["view"] == "back" and back["fractions"]["back"] == 1.0 and back["tips"] == []
    side = camera_report(_view_track(10, 0.05, 0.2, False))
    assert side["view"] == "side" and "film from behind" in side["tips"][0]


def test_no_shoulders():
    t = _view_track(5, 0.2, 0.02, False)
    t.world[:] = np.nan
    assert camera_report(t) == {"view": None, "fractions": {}, "tips": []}
