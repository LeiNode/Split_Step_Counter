"""
fatigue.py — does the player fade as the session goes on?

Tired players hop lower into their split steps, land with straighter knees,
swing slower and take longer to get back to a ready stance. This compares the
first third of the session with the last third, metric by metric (medians,
so one odd shot doesn't decide it).

A metric needs FATIGUE_MIN_SAMPLES values in both thirds, and the session
must last FATIGUE_MIN_MS: on a 20 s clip "early" and "late" are seconds
apart and any difference is noise.

Thresholds are coaching heuristics, not fitted to data: a drop smaller than
the per-shot noise on pro practice footage (~10% wrist speed) isn't flagged.
"""

import numpy as np

FATIGUE_MIN_MS      = 120_000   # 2 min
FATIGUE_MIN_SAMPLES = 4         # values per third

# metric -> (source, label, better: "higher" | "lower", relative change that counts as a drop)
METRICS = {
    "hop_height":         ("events", "split-step hop height", "higher", 0.20),
    "landing_knee_angle": ("events", "split-step knee bend",  "lower",  0.05),
    "peak_wrist_speed":   ("swings", "swing speed",           "higher", 0.10),
    "recovery_ms":        ("swings", "recovery time",         "lower",  0.25),
}


def _third(items: list[dict], key: str, lo: float, hi: float) -> list[float]:
    return [x[key] for x in items if x.get(key) is not None and lo <= x["timestamp_ms"] <= hi]


def fatigue_report(events: list[dict] | None, swings: list[dict] | None, duration_ms: float) -> dict:
    """{"measured": bool, "metrics": {name: {early, late, change, dropped}}, "tips": [...]}.

    change is relative (late / early - 1, rounded to 3 decimals); dropped is
    True when the metric got worse by more than its threshold. Split steps
    flagged during_swing are left out, as in timing.py.
    """
    out = {"measured": False, "metrics": {}, "tips": []}
    if duration_ms < FATIGUE_MIN_MS:
        return out
    sources = {"events": [e for e in (events or []) if not e.get("during_swing")], "swings": swings or []}
    cut1, cut2 = duration_ms / 3, 2 * duration_ms / 3
    dropped = []
    for key, (src, label, better, limit) in METRICS.items():
        early = _third(sources[src], key, 0, cut1)
        late = _third(sources[src], key, cut2, duration_ms)
        if len(early) < FATIGUE_MIN_SAMPLES or len(late) < FATIGUE_MIN_SAMPLES:
            continue
        e, l = float(np.median(early)), float(np.median(late))
        if e == 0:
            continue
        change = l / e - 1
        worse = -change if better == "higher" else change
        out["metrics"][key] = {"early": round(e, 3), "late": round(l, 3), "change": round(change, 3),
                               "dropped": bool(worse > limit)}
        if worse > limit:
            dropped.append(f"{label} ({'-' if change < 0 else '+'}{abs(change) * 100:.0f}%)")
    out["measured"] = bool(out["metrics"])
    if dropped:
        out["tips"].append("You fade late in the session: " + ", ".join(dropped) + " in the last third "
                           "compared with the first. Take short breaks, and keep the effort on footwork "
                           "when you get tired.")
    elif out["measured"]:
        out["tips"].append("Good habit: no drop-off late in the session; your intensity held up.")
    return out
