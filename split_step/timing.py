"""
timing.py — links split steps to the shots that follow them.

The habit coaches drill: split step as the opponent strikes, then move and
hit. For each swing we look back for the most recent split step 0.4-2.5 s
earlier (the time to read the ball, move and prepare).

A split-step detection within ±300 ms of a swing peak is the small hop many
players make while hitting, not a split step; such events are flagged
`during_swing` and ignored here (the raw count is left unchanged).
"""

import numpy as np

LEAD_MIN_MS        = 400    # a split step this close before contact is too late to help
LEAD_MAX_MS        = 2500   # ... and one this long before belongs to an earlier shot
SWING_OVERLAP_MS   = 300    # detection this close to a swing peak = hop during the stroke
TIP_MIN_SHOTS      = 3      # don't judge habits from fewer shots than this
TIP_GOOD_FRACTION  = 0.6    # below this share of shots with a split step -> tip


def link_split_steps(events: list[dict], swings: list[dict]) -> dict:
    """Annotates events (during_swing) and swings (split_step_lead_ms) in place;
    returns a summary with tips."""
    swing_ts = np.array([s["timestamp_ms"] for s in swings], float)
    for e in events:
        e["during_swing"] = bool(len(swing_ts) and np.min(np.abs(swing_ts - e["timestamp_ms"])) <= SWING_OVERLAP_MS)
    split_ts = sorted(e["timestamp_ms"] for e in events if not e["during_swing"])

    leads = []
    for s in swings:
        prior = [s["timestamp_ms"] - t for t in split_ts if LEAD_MIN_MS <= s["timestamp_ms"] - t <= LEAD_MAX_MS]
        s["split_step_lead_ms"] = round(min(prior), 1) if prior else None
        if prior:
            leads.append(min(prior))

    n = len(swings)
    summary = {
        "shots": n,
        "shots_with_split_step": len(leads),
        "fraction_with_split_step": round(len(leads) / n, 3) if n else None,
        "median_lead_ms": round(float(np.median(leads)), 1) if leads else None,
        "split_steps_during_swings": sum(e["during_swing"] for e in events),
        "tips": [],
    }
    if n >= TIP_MIN_SHOTS:
        frac = summary["fraction_with_split_step"]
        if frac < TIP_GOOD_FRACTION:
            summary["tips"].append(
                f"Only {len(leads)} of {n} shots had a split step before them. Make it a habit: "
                "a small hop as your opponent hits, then move.")
        else:
            summary["tips"].append(f"Good habit: {len(leads)} of {n} shots had a split step before them.")
    return summary
