"""Early-vs-late session comparison."""

from split_step.fatigue import FATIGUE_MIN_MS, fatigue_report

DUR = 300_000  # 5 min


def _swings(speeds_early, speeds_late):
    early = [{"timestamp_ms": 5_000 + 10_000 * i, "peak_wrist_speed": v} for i, v in enumerate(speeds_early)]
    late = [{"timestamp_ms": 210_000 + 10_000 * i, "peak_wrist_speed": v} for i, v in enumerate(speeds_late)]
    return early + late


def test_swing_speed_drop_flagged():
    r = fatigue_report(None, _swings([12, 12, 13, 12], [10, 10, 9.5, 10]), DUR)
    m = r["metrics"]["peak_wrist_speed"]
    assert r["measured"] and m["dropped"] and m["change"] < -0.1
    assert len(r["tips"]) == 1 and "swing speed (-17%)" in r["tips"][0]


def test_steady_session_is_a_strength():
    events = [{"timestamp_ms": t, "hop_height": 0.2, "landing_knee_angle": 130}
              for t in [*range(1_000, 90_000, 20_000), *range(205_000, 299_000, 20_000)]]
    events.append({"timestamp_ms": 250_000, "hop_height": 0.01, "during_swing": True})  # ignored
    r = fatigue_report(events, _swings([12] * 4, [12] * 4), DUR)
    assert set(r["metrics"]) == {"hop_height", "landing_knee_angle", "peak_wrist_speed"}
    assert not any(m["dropped"] for m in r["metrics"].values())
    assert r["tips"][0].startswith("Good habit")


def test_short_or_sparse_sessions_not_measured():
    assert fatigue_report(None, _swings([12] * 4, [9] * 4), FATIGUE_MIN_MS - 1) == {"measured": False, "metrics": {}, "tips": []}
    r = fatigue_report(None, _swings([12] * 3, [9] * 4), DUR)        # 3 early swings: too few
    assert not r["measured"] and not r["tips"]


def test_thirds_measured_from_the_start_of_the_analysed_part():
    shifted = [{**s, "timestamp_ms": s["timestamp_ms"] + 600_000} for s in _swings([12] * 4, [9] * 4)]
    r = fatigue_report(None, shifted, DUR + 600_000, start_ms=600_000)
    assert r["metrics"]["peak_wrist_speed"]["dropped"]
