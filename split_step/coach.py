"""
coach.py — the short version: what to work on first, and what's going well.

Every analysis section emits its own tips; this collects them into one
prioritised list for the top of a report. Order of importance:

  1. camera  — a bad camera angle makes every other number less reliable
  2. habits  — split step before shots, recovery, footwork, fatigue, split-step quality
  3. technique — per-stroke tips, then rally consistency

Within a section the original order is kept. Tips that report something
good ("solid", "good habit", "looks solid") are strengths, not work items.
"""

TOP_N = 3

# (result key, how to pull its tips), in priority order
_SOURCES = [
    ("camera",   lambda r: r.get("tips", [])),
    ("timing",   lambda r: r.get("tips", [])),
    ("recovery", lambda r: r.get("tips", [])),
    ("footwork", lambda r: r.get("tips", [])),
    ("fatigue",  lambda r: r.get("tips", [])),
    ("session",  lambda r: r.get("tips", [])),
    ("strokes",  lambda r: r.get("tips", [])),
    ("rallies",  lambda r: r.get("summary", {}).get("tips", [])),
]
_POSITIVE = ("solid", "good habit", "keep it up")


def coach_summary(result: dict) -> dict:
    """{"focus": up to TOP_N tips to work on, "strengths": positive notes, "more": count of other tips}."""
    work, strengths = [], []
    for key, get in _SOURCES:
        section = result.get(key)
        if not section:
            continue
        for tip in get(section):
            (strengths if any(p in tip.lower() for p in _POSITIVE) else work).append(tip)
    return {"focus": work[:TOP_N], "strengths": strengths, "more": max(0, len(work) - TOP_N)}
