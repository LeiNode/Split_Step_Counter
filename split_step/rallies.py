"""
rallies.py — groups swings into rallies.

Consecutive swings belong to the same rally while the gap between them is at
most RALLY_GAP_MS; a longer pause (ball out, walking back, picking up balls)
starts a new rally. In a fed drill each "rally" is a basket sequence.
Only the filmed player's shots are visible, so a rally of N shots means N
shots by this player.
"""

RALLY_GAP_MS = 4000   # pro rallies have ~1.2-2 s between one player's shots; 4 s is a clear break


def find_rallies(swings: list[dict]) -> dict:
    """Annotates swings with rally (1-based) in place; returns {rallies, summary}."""
    rallies: list[dict] = []
    prev_t = None
    for s in sorted(swings, key=lambda s: s["timestamp_ms"]):
        t = s["timestamp_ms"]
        if prev_t is None or t - prev_t > RALLY_GAP_MS:
            rallies.append({"start_ms": t, "end_ms": t, "shots": 0, "strokes": {}})
        r = rallies[-1]
        r["end_ms"] = t
        r["shots"] += 1
        r["strokes"][s["stroke"]] = r["strokes"].get(s["stroke"], 0) + 1
        s["rally"] = len(rallies)
        prev_t = t

    shots = [r["shots"] for r in rallies]
    summary = {
        "rallies": len(rallies),
        "mean_shots": round(sum(shots) / len(shots), 2) if shots else None,
        "longest": max(shots) if shots else None,
        "tips": [],
    }
    if len(rallies) >= 3 and summary["mean_shots"] < 2:
        summary["tips"].append("Most rallies end after one shot. Work on consistency: "
                               "aim for height and depth over the net before adding pace.")
    return {"rallies": rallies, "summary": summary}
