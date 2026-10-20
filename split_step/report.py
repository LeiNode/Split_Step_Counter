"""
report.py — an analyze_video() result as a readable Markdown report.

For sharing with a coach or keeping in notes: the coach's summary and
practice plan first, then one section per analysis that ran. Sections that
weren't computed (feature off, no drill) are left out.

    python -m split_step.analyze clip.mp4 --drill auto --report session.md
"""

_STROKE_COLUMNS = {
    "serve": [("Reach", "contact_reach_m", "{:.2f} m"), ("Toss", "toss_height_m", "{:.2f} m"),
              ("Elbow", "elbow_extension", "{:.0f}°"), ("Trophy knee", "trophy_knee_angle", "{:.0f}°")],
    "groundstroke": [("Contact height", "contact_height", "{:.2f}"), ("Shoulder turn", "shoulder_turn_deg", "{:.0f}°"),
                     ("Turned before contact", "prep_lead_ms", "{:.0f} ms"),
                     ("Head movement", "head_movement_m", "{:.2f} m")],
}


def _t(ms: float) -> str:
    s = ms / 1000
    return f"{int(s // 60)}:{s % 60:04.1f}"


def _v(value, fmt: str) -> str:
    return "–" if value is None else fmt.format(value)


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    return ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers),
            *("| " + " | ".join(r) + " |" for r in rows), ""]


def _bullets(items) -> list[str]:
    return [f"- {x}" for x in items] + ([""] if items else [])


def to_markdown(result: dict, title: str = "Tennis session report") -> str:
    out = [f"# {title}", "",
           f"{_t(result['duration_ms'])} of video · {result['frames_with_pose']} of {result['frames']} frames "
           "with the player tracked", ""]

    coach = result.get("coach")
    if coach and (coach["focus"] or coach["strengths"]):
        out += ["## Coach's summary", ""]
        if coach["focus"]:
            out += ["**Work on these first:**", "", *(f"{n}. {t}" for n, t in enumerate(coach["focus"], 1)), ""]
        if coach["strengths"]:
            out += ["**Going well:**", "", *_bullets(coach["strengths"])]
        if coach.get("drills"):
            out += ["### Practice plan", ""]
            for d in coach["drills"]:
                out += [f"- **{d['name']}** ({d['dose']}): {d['how']}"]
            out += [""]

    camera_tips = [t for t in (result.get("camera") or {}).get("tips", []) if t not in (coach or {}).get("focus", [])]
    if camera_tips:
        out += ["> **Camera angle:** " + " ".join(camera_tips), ""]

    if "count" in result:
        s = result.get("session") or {}
        out += ["## Split steps", "", f"**{result['count']}** split steps"
                + (f", {s['split_steps_per_minute']:.1f} per minute" if s.get("split_steps_per_minute") is not None else "")
                + ".", ""]
        if result["events"] and "hop_height" in result["events"][0]:
            out += _table(["Time", "Hop height", "Knee angle", "Stance (× shoulders)"],
                          [[_t(e["timestamp_ms"]) + (" (during a swing)" if e.get("during_swing") else ""),
                            _v(e.get("hop_height"), "{:.2f}"), _v(e.get("landing_knee_angle"), "{:.0f}°"),
                            _v(e.get("landing_stance_ratio"), "{:.2f}")] for e in result["events"]])
        out += _bullets(s.get("tips", []))

    strokes = result.get("strokes")
    if strokes:
        out += ["## Strokes", "", f"{len(strokes['swings'])} swings, {strokes['hand']}-handed.", ""]
        scores = [f"{k} {v['consistency']}/100" for k, v in strokes["summary"].items() if v.get("consistency") is not None]
        if scores:
            out += ["Consistency: " + ", ".join(scores) + ".", ""]
        for kind in ("serve", "groundstroke"):
            swings = [s for s in strokes["swings"] if (s["stroke"] == "serve") == (kind == "serve")]
            if not swings:
                continue
            cols = _STROKE_COLUMNS[kind]
            out += _table(["Time", "Stroke", "Wrist speed", *(c[0] for c in cols)],
                          [[_t(s["timestamp_ms"]), s["stroke"], _v(s.get("peak_wrist_speed"), "{:.1f} m/s"),
                            *(_v(s.get(key), fmt) for _, key, fmt in cols)] for s in swings])
        highlights = result.get("highlights") or {}
        lines = []
        for stroke, h in highlights.items():
            parts = [f"most typical at {_t(h['typical']['timestamp_ms'])}"] if h.get("typical") else []
            parts.append(f"fastest at {_t(h['fastest']['timestamp_ms'])}")
            if h.get("outlier"):
                parts.append(f"odd one out at {_t(h['outlier']['timestamp_ms'])} ({h['outlier']['label']})")
            lines.append(f"{stroke.capitalize()}: " + ", ".join(parts))
        if lines:
            out += ["**Worth watching back:**", "", *_bullets(lines)]
        out += _bullets(strokes["tips"])

    habits = []
    if result.get("timing") and result["timing"]["shots"]:
        t = result["timing"]
        habits.append(f"Split step before the shot: {t['shots_with_split_step']} of {t['shots']} shots")
    if result.get("recovery") and result["recovery"].get("median_recovery_ms") is not None:
        habits.append(f"Back to a ready position: {result['recovery']['median_recovery_ms'] / 1000:.1f} s after a shot "
                      f"(median; {result['recovery']['recovered']} of {result['recovery']['shots_measured']} shots)")
    if result.get("rallies") and result["rallies"]["summary"]["rallies"]:
        r = result["rallies"]["summary"]
        habits.append(f"Rallies: {r['rallies']}, longest {r['longest']} shots, {r['mean_shots']:.1f} on average")
    if result.get("footwork"):
        f = result["footwork"]
        habits.append(f"Footwork: {f['steps_per_minute']:.0f} steps per minute"
                      + (f", {f['median_steps_between_shots']:g} adjustment steps between shots"
                         if f.get("median_steps_between_shots") is not None else ""))
    fatigue = result.get("fatigue") or {}
    for key, m in fatigue.get("metrics", {}).items():
        habits.append(f"Late vs early session, {key.replace('_', ' ')}: {m['change'] * 100:+.0f}%"
                      + (" (dropped off)" if m["dropped"] else ""))
    if habits:
        out += ["## Habits and movement", "", *_bullets(habits)]
        tips = [*(result.get("timing") or {}).get("tips", []), *(result.get("recovery") or {}).get("tips", []),
                *(result.get("rallies") or {}).get("summary", {}).get("tips", []),
                *(result.get("footwork") or {}).get("tips", []), *fatigue.get("tips", [])]
        out += _bullets(tips)

    return "\n".join(out).rstrip() + "\n"
