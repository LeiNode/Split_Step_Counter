"""Markdown report from a real analysis (the web-app fixture) and from a bare result."""

import json
import pathlib

from split_step.report import to_markdown

FIXTURE = pathlib.Path(__file__).resolve().parent / "web" / "analysis_fixture.json"


def test_full_report_sections():
    result = json.loads(FIXTURE.read_text(encoding="utf-8"))
    result["coach"]["drills"] = [{"name": "Split on the feed", "dose": "3 sets of 10 balls", "how": "Hop as they hit."}]
    md = to_markdown(result, "practice.mp4")
    assert md.startswith("# practice.mp4\n")
    for heading in ("## Coach's summary", "### Practice plan", "## Split steps", "## Strokes", "## Habits and movement"):
        assert heading in md, heading
    assert "**Split on the feed** (3 sets of 10 balls)" in md
    assert f"**{result['count']}** split steps" in md
    swing_rows = [l for l in md.splitlines() if l.startswith("| ") and " m/s |" in l]
    assert len(swing_rows) == len(result["strokes"]["swings"])
    assert md.count("film from behind") == 1                  # side-view clip: warned once, in the summary
    del result["coach"]
    assert "> **Camera angle:**" in to_markdown(result)        # ...or as a callout without one


def test_minimal_result():
    md = to_markdown({"duration_ms": 65400, "frames": 10, "frames_with_pose": 0})
    assert md == "# Tennis session report\n\n1:05.4 of video · 0 of 10 frames with the player tracked\n"
