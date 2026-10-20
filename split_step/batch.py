"""
batch.py — analyse every video in a folder; one summary row per clip.

For a coach with a season of clips, or to compare sessions side by side in a
spreadsheet. Each clip goes through analyze_video() with the same options;
a clip that can't be read gets a row with its error instead of stopping the
run.

Usage
-----
    python -m split_step.batch FOLDER [--out summary.csv] [--json-dir reports/]
                               [--model lstm|rules] [--drill auto|forehand|backhand|serve]
                               [--hand auto|right|left] [--recursive]
"""

import csv
import json
import os
import sys

from .analyze import MODELS, VideoError, analyze_video
from .strokes import DRILLS, HANDS, STROKES

VIDEO_EXTS = (".mp4", ".mov", ".m4v", ".webm", ".avi", ".mkv")

COLUMNS = [
    "file", "duration_s", "split_steps", "split_steps_per_minute", "landing_knee_angle",
    "swings", *(f"{s}_consistency" for s in STROKES), "shots_with_split_step",
    "median_recovery_s", "longest_rally", "steps_per_minute", "camera_view", "top_tip", "error",
]


def find_videos(folder: str, recursive: bool = False) -> list[str]:
    """Video files in folder (sorted), optionally including subfolders."""
    if recursive:
        found = [os.path.join(d, f) for d, _, files in os.walk(folder) for f in files]
    else:
        found = [os.path.join(folder, f) for f in os.listdir(folder)]
    return sorted(p for p in found if os.path.isfile(p) and p.lower().endswith(VIDEO_EXTS))


def summary_row(name: str, result: dict) -> dict:
    """Flatten one analyze_video() result into a COLUMNS row (missing values = None)."""
    session = result.get("session") or {}
    strokes = result.get("strokes") or {}
    timing, recovery = result.get("timing") or {}, result.get("recovery") or {}
    knee = session.get("landing_knee_angle")
    row = {
        "file": name,
        "duration_s": round(result["duration_ms"] / 1000, 1),
        "split_steps": result.get("count"),
        "split_steps_per_minute": session.get("split_steps_per_minute"),
        "landing_knee_angle": knee.get("median") if isinstance(knee, dict) else None,
        "swings": len(strokes["swings"]) if strokes else None,
        "shots_with_split_step": timing.get("fraction_with_split_step"),
        "median_recovery_s": (round(recovery["median_recovery_ms"] / 1000, 2)
                              if recovery.get("median_recovery_ms") is not None else None),
        "longest_rally": (result.get("rallies") or {}).get("summary", {}).get("longest"),
        "steps_per_minute": (result.get("footwork") or {}).get("steps_per_minute"),
        "camera_view": (result.get("camera") or {}).get("view"),
        "top_tip": ((result.get("coach") or {}).get("focus") or [None])[0],
        "error": None,
    }
    for s in STROKES:
        row[f"{s}_consistency"] = (strokes.get("summary") or {}).get(s, {}).get("consistency")
    return row


def run_batch(videos: list[str], out_csv: str | None = None, json_dir: str | None = None,
              progress=None, **analyze_kwargs) -> list[dict]:
    """Analyse each video; returns the rows and writes them to out_csv if given.
    With json_dir, each clip's full result is also saved as <name>.json there."""
    rows = []
    if json_dir:
        os.makedirs(json_dir, exist_ok=True)
    for n, path in enumerate(videos, 1):
        name = os.path.basename(path)
        if progress:
            progress(f"[{n}/{len(videos)}] {name}")
        try:
            result = analyze_video(path, **analyze_kwargs)
        except VideoError as e:
            rows.append({**dict.fromkeys(COLUMNS), "file": name, "error": str(e)})
            continue
        rows.append(summary_row(name, result))
        if json_dir:
            with open(os.path.join(json_dir, os.path.splitext(name)[0] + ".json"), "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
    if out_csv:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            w.writeheader()
            w.writerows(rows)
    return rows


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyse every video in a folder into one summary CSV.")
    parser.add_argument("folder")
    parser.add_argument("--out", default="summary.csv", help="Summary CSV path (default summary.csv)")
    parser.add_argument("--json-dir", help="Also save each clip's full report as JSON here")
    parser.add_argument("--recursive", action="store_true", help="Include videos in subfolders")
    parser.add_argument("--model", choices=MODELS, default="lstm")
    parser.add_argument("--drill", choices=DRILLS, help="Also analyse strokes (forehand/backhand/serve/auto)")
    parser.add_argument("--hand", choices=HANDS, default="auto")
    args = parser.parse_args()

    videos = find_videos(args.folder, args.recursive)
    if not videos:
        sys.exit(f"No videos ({', '.join(VIDEO_EXTS)}) found in {args.folder}")
    rows = run_batch(videos, args.out, args.json_dir, progress=lambda m: print(m, file=sys.stderr),
                     model=args.model, drill=args.drill, hand=args.hand)
    failed = sum(r["error"] is not None for r in rows)
    print(f"Wrote {args.out}: {len(rows) - failed} analysed, {failed} failed", file=sys.stderr)
