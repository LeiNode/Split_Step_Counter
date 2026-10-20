"""Watch-folder mode on generated clips."""

import os

from split_step.watch import SETTLE_POLLS, pending, watch
from tests.test_analyze import _clip


def test_waits_for_file_to_settle_and_skips_reported(tmp_path):
    _clip(tmp_path / "a.mp4", 30)
    sizes = {}
    for _ in range(SETTLE_POLLS):
        assert pending(str(tmp_path), sizes) == []                 # size not yet stable
    assert [os.path.basename(p) for p in pending(str(tmp_path), sizes)] == ["a.mp4"]
    (tmp_path / "a.report.md").write_text("done")
    assert pending(str(tmp_path), {}) == []                         # already has a report


def test_once_writes_reports_and_errors(tmp_path):
    _clip(tmp_path / "a.mp4", 30)
    (tmp_path / "b.mov").write_bytes(b"not a video")
    logs = []
    n = watch(str(tmp_path), every_s=0, once=True, log=logs.append, model="rules")
    assert n == 1
    assert (tmp_path / "a.report.md").read_text(encoding="utf-8").startswith("# a.mp4")
    assert (tmp_path / "a.report.json").exists()
    assert (tmp_path / "b.report.error.txt").exists() and not (tmp_path / "b.report.md").exists()
    assert any("could not read b.mov" in line for line in logs)
    assert watch(str(tmp_path), every_s=0, once=True, log=logs.append, model="rules") == 0   # restart-safe
