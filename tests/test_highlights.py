"""Typical / fastest / odd-one-out swing per stroke."""

from split_step.highlights import stroke_highlights


def _fh(t, speed, contact, turn=100.0):
    return {"timestamp_ms": t, "stroke": "forehand", "peak_wrist_speed": speed,
            "contact_height": contact, "shoulder_turn_deg": turn, "finish_above_shoulder_m": 0.2}


def test_typical_fastest_outlier():
    swings = [_fh(1000, 10.0, 0.40), _fh(3000, 10.2, 0.45), _fh(5000, 9.9, 0.42),
              _fh(7000, 13.0, 0.43), _fh(9000, 10.1, 1.10),            # contact way up
              {"timestamp_ms": 9500, "stroke": "serve", "peak_wrist_speed": 20.0}]
    h = stroke_highlights(swings)
    assert set(h) == {"forehand"}                                      # 1 serve: too few
    fh = h["forehand"]
    assert fh["fastest"]["timestamp_ms"] == 7000
    assert fh["typical"]["timestamp_ms"] in (3000, 5000)
    assert fh["outlier"]["timestamp_ms"] == 9000 and fh["outlier"]["metric"] == "contact_height"
    assert fh["outlier"]["label"] == "contact height 1.10, usually 0.43"


def test_no_outlier_when_all_alike_and_missing_values():
    swings = [_fh(t, 10.0, 0.4) for t in (0, 2000, 4000, 6000, 8000)]
    swings[0]["contact_height"] = None                                 # not measured on one swing
    fh = stroke_highlights(swings)["forehand"]
    assert fh["outlier"] is None
    assert fh["typical"]["timestamp_ms"] != 0                          # prefers fully measured swings
