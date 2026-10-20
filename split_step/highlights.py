"""
highlights.py — which swings are worth watching back.

Per stroke type:
  typical — the swing closest to your own median on every consistency
            metric: your "normal" stroke, to compare others against
  fastest — the highest peak wrist speed
  outlier — the swing furthest from your median, and the metric that set it
            apart most ("contact height 0.95, usually 0.42"), when it is at
            least OUTLIER_MIN_DISTANCE tolerances away

Distances use the consistency tolerances from strokes.py, so a metric's
spread counts the same here as it does in the consistency score: each
metric's deviation is divided by its tolerance (wrist speed relative to the
median, like there), and a swing's distance is the RMS over its metrics.
"""

import numpy as np

from .strokes import CONSISTENCY_TOLERANCE, STROKES

MIN_SWINGS = 4               # fewer and "typical" / "odd one out" mean nothing
OUTLIER_MIN_DISTANCE = 0.75  # in tolerances (1 = as far off as a zero-consistency spread)

LABELS = {
    "peak_wrist_speed": ("wrist speed", "{:.1f} m/s"),
    "contact_height": ("contact height", "{:.2f}"),
    "shoulder_turn_deg": ("shoulder turn", "{:.0f}°"),
    "finish_above_shoulder_m": ("finish height", "{:+.2f} m"),
    "contact_reach_m": ("reach", "{:.2f} m"),
    "toss_height_m": ("toss height", "{:.2f} m"),
    "toss_lateral_m": ("toss release (side)", "{:+.2f} m"),
    "elbow_extension": ("elbow extension", "{:.0f}°"),
}


def _deviations(swings: list[dict], tol: dict) -> tuple[np.ndarray, list[str], dict]:
    """(n_swings, n_metrics) deviations in tolerances (NaN = not measured), metric names, medians."""
    keys, medians, cols = [], {}, []
    for key, limit in tol.items():
        vals = np.array([np.nan if s.get(key) is None else s[key] for s in swings], float)
        if np.isfinite(vals).sum() < MIN_SWINGS:
            continue
        med = float(np.nanmedian(vals))
        if key == "peak_wrist_speed":
            if med <= 0:
                continue
            cols.append((vals / med - 1) / limit)
        else:
            cols.append((vals - med) / limit)
        keys.append(key)
        medians[key] = med
    dev = np.column_stack(cols) if cols else np.zeros((len(swings), 0))
    return dev, keys, medians


def _ref(s: dict) -> dict:
    return {"timestamp_ms": s["timestamp_ms"]}


def stroke_highlights(swings: list[dict]) -> dict:
    """{stroke: {typical, fastest, outlier | None}} for strokes with MIN_SWINGS+ swings.
    Each entry has timestamp_ms; outlier also has metric, value, usual, label."""
    out = {}
    for stroke in STROKES:
        mine = [s for s in swings if s.get("stroke") == stroke]
        if len(mine) < MIN_SWINGS:
            continue
        tol = CONSISTENCY_TOLERANCE["serve" if stroke == "serve" else "groundstroke"]
        dev, keys, medians = _deviations(mine, tol)
        entry = {"typical": None, "fastest": None, "outlier": None}
        speeds = [s.get("peak_wrist_speed") or 0 for s in mine]
        entry["fastest"] = _ref(mine[int(np.argmax(speeds))])
        if keys:
            measured = np.isfinite(dev).sum(axis=1)
            sq_sum = np.nansum(dev ** 2, axis=1)
            dist = np.full(len(mine), np.nan)
            dist[measured > 0] = np.sqrt(sq_sum[measured > 0] / measured[measured > 0])
            # Typical: among swings measured on every metric when there are any
            full = measured == len(keys)
            pool = np.where(full if full.any() else measured > 0, dist, np.inf)
            entry["typical"] = _ref(mine[int(np.argmin(pool))])
            far = int(np.nanargmax(dist))
            if dist[far] >= OUTLIER_MIN_DISTANCE:
                j = int(np.nanargmax(np.abs(dev[far])))
                key = keys[j]
                name, fmt = LABELS.get(key, (key, "{:.2f}"))
                value, usual = mine[far][key], medians[key]
                entry["outlier"] = {**_ref(mine[far]), "metric": key, "value": value, "usual": round(usual, 2),
                                    "label": f"{name} {fmt.format(value)}, usually {fmt.format(usual)}"}
        out[stroke] = entry
    return out
