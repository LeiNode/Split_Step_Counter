"""Batch analysis of a folder (generated clips, no assets needed)."""

import csv
import os

from split_step.batch import COLUMNS, find_videos, run_batch, summary_row
from tests.test_analyze import _clip


def test_folder_to_csv_with_a_broken_clip(tmp_path):
    _clip(tmp_path / "a.mp4", 30)
    (tmp_path / "b.mov").write_bytes(b"not a video")
    (tmp_path / "notes.txt").write_text("ignored")
    (tmp_path / "sub").mkdir()
    _clip(tmp_path / "sub" / "c.mp4", 30)

    assert [os.path.basename(p) for p in find_videos(str(tmp_path))] == ["a.mp4", "b.mov"]
    assert len(find_videos(str(tmp_path), recursive=True)) == 3

    out = tmp_path / "summary.csv"
    rows = run_batch(find_videos(str(tmp_path)), str(out), str(tmp_path / "json"), model="rules")
    assert [r["file"] for r in rows] == ["a.mp4", "b.mov"]
    assert rows[0]["error"] is None and rows[0]["split_steps"] == 0 and rows[0]["duration_s"] > 1.5
    assert rows[1]["error"]
    with open(out, newline="", encoding="utf-8") as f:
        written = list(csv.DictReader(f))
    assert list(written[0]) == COLUMNS and len(written) == 2
    assert (tmp_path / "json" / "a.json").exists() and not (tmp_path / "json" / "b.json").exists()


def test_summary_row_flattens_sections():
    result = {"duration_ms": 61000, "count": 12, "session": {"split_steps_per_minute": 11.8,
                                                             "landing_knee_angle": {"median": 128.0}},
              "strokes": {"swings": [{}] * 9, "summary": {"forehand": {"count": 9, "consistency": 64}}},
              "timing": {"fraction_with_split_step": 0.556}, "recovery": {"median_recovery_ms": 1234.0},
              "rallies": {"summary": {"longest": 5}}, "coach": {"focus": ["Bend your knees."]}}
    r = summary_row("x.mp4", result)
    assert r["landing_knee_angle"] == 128.0 and r["forehand_consistency"] == 64 and r["serve_consistency"] is None
    assert r["median_recovery_s"] == 1.23 and r["longest_rally"] == 5 and r["top_tip"] == "Bend your knees."
    assert set(r) == set(COLUMNS)


def test_index_overview():
    from split_step.batch import index_markdown
    tip_a = "Only 1 of 6 shots had a split step before them. Make it a habit."
    tip_b = "Only 3 of 9 shots had a split step before them. Make it a habit."
    rows = [{**dict.fromkeys(COLUMNS), "file": "day 1.mp4", "duration_s": 61.0, "swings": 6, "forehand_consistency": 55,
             "shots_with_split_step": 0.167, "error": None},
            {**dict.fromkeys(COLUMNS), "file": "day2.mp4", "duration_s": 90.0, "swings": 9, "error": None},
            {**dict.fromkeys(COLUMNS), "file": "bad.mov", "error": "Could not open the file as a video."}]
    md = index_markdown(rows, {"day 1.mp4": [tip_a, "Forehand: keep your head still."], "day2.mp4": [tip_b]})
    assert "2 clips analysed, 1 could not be read." in md
    assert "| [day 1.mp4](day%201.md) | 61 s | – | 6 | 55/100 | 17% | – |" in md
    assert "- **2 of 2 clips**: Only 1 of 6 shots" in md and "(drill: Split on the feed)" in md   # grouped by drill
    assert "- bad.mov: Could not open" in md
