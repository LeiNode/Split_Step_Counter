"""Prioritising tips into the coach's summary."""

from split_step.coach import coach_summary


def test_priority_and_strengths():
    result = {
        "strokes": {"tips": ["Forehand: turn your shoulders more.", "Serve: technique metrics look solid. Keep it up!"]},
        "camera": {"tips": ["Most of this video shows the player from the side..."]},
        "timing": {"tips": ["Only 1 of 6 shots had a split step before them."]},
        "rallies": {"summary": {"tips": ["Most rallies end after one shot."]}},
        "footwork": {"tips": ["Keep your feet moving between shots."]},
    }
    c = coach_summary(result)
    assert c["focus"] == ["Most of this video shows the player from the side...",
                          "Only 1 of 6 shots had a split step before them.",
                          "Keep your feet moving between shots."]
    assert c["more"] == 2
    assert c["strengths"] == ["Serve: technique metrics look solid. Keep it up!"]


def test_empty():
    assert coach_summary({}) == {"focus": [], "strengths": [], "more": 0}
