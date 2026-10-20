"""Grouping swings into rallies."""

from split_step.rallies import RALLY_GAP_MS, find_rallies


def _sw(*items):
    return [{"timestamp_ms": t, "stroke": k} for t, k in items]


def test_groups_by_gap():
    swings = _sw((1000, "serve"), (2500, "forehand"), (4000, "backhand"),
                 (4000 + RALLY_GAP_MS + 1, "serve"), (20000, "forehand"))
    out = find_rallies(swings)
    assert [r["shots"] for r in out["rallies"]] == [3, 1, 1]
    assert out["rallies"][0]["strokes"] == {"serve": 1, "forehand": 1, "backhand": 1}
    assert [s["rally"] for s in swings] == [1, 1, 1, 2, 3]
    assert out["summary"]["longest"] == 3 and out["summary"]["mean_shots"] == round(5 / 3, 2)


def test_short_rallies_tip_and_empty():
    swings = _sw((0, "forehand"), (10000, "forehand"), (20000, "forehand"))
    assert find_rallies(swings)["summary"]["tips"]
    empty = find_rallies([])
    assert empty["rallies"] == [] and empty["summary"]["mean_shots"] is None
