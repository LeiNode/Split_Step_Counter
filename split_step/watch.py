"""
watch.py — analyse new clips as they arrive in a folder.

Point it at a phone's synced camera folder (OneDrive, Dropbox, Google Photos
backup, ...) and every new video gets a report next to it:

    clip.mp4  ->  clip.report.md  +  clip.report.json

A clip counts as "arrived" once its size has stopped changing for
SETTLE_POLLS polls in a row, so half-synced files aren't read. Clips that
already have a report are skipped, which makes restarting safe; one that
can't be read gets clip.report.error.txt instead (delete it to retry).

    python -m split_step.watch FOLDER [--drill auto] [--model rules] [--every 10] [--once]
"""

import json
import os
import sys
import time

from .analyze import MODELS, VideoError, analyze_video
from .batch import find_videos
from .report import to_markdown
from .strokes import DRILLS, HANDS

SETTLE_POLLS = 2


def _outputs(video: str) -> tuple[str, str, str]:
    stem = os.path.splitext(video)[0]
    return stem + ".report.md", stem + ".report.json", stem + ".report.error.txt"


def pending(folder: str, sizes: dict, recursive: bool = False) -> list[str]:
    """Videos without a report whose size has been stable for SETTLE_POLLS polls.
    sizes ({path: (size, stable_polls)}) carries state between calls."""
    ready = []
    for v in find_videos(folder, recursive):
        if any(os.path.exists(p) for p in _outputs(v)):
            continue
        size = os.path.getsize(v)
        prev, polls = sizes.get(v, (None, 0))
        polls = polls + 1 if size == prev and size > 0 else 0
        sizes[v] = (size, polls)
        if polls >= SETTLE_POLLS:
            ready.append(v)
    return ready


def process(video: str, log=print, **analyze_kwargs) -> bool:
    """Analyse one clip and write its report files; False if it failed."""
    md, js, err = _outputs(video)
    name = os.path.basename(video)
    log(f"analysing {name} ...")
    try:
        result = analyze_video(video, **analyze_kwargs)
    except VideoError as e:
        with open(err, "w", encoding="utf-8") as f:
            f.write(f"{e}\n")
        log(f"  could not read {name}: {e}")
        return False
    with open(js, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    with open(md, "w", encoding="utf-8") as f:
        f.write(to_markdown(result, name))
    focus = (result.get("coach") or {}).get("focus") or []
    log(f"  done -> {os.path.basename(md)}" + (f"  (top tip: {focus[0]})" if focus else ""))
    return True


def watch(folder: str, every_s: float = 10.0, once: bool = False, recursive: bool = False,
          log=print, **analyze_kwargs) -> int:
    """Poll the folder until interrupted (or, with once, until nothing is left
    that's ready). Returns the number of clips analysed."""
    sizes, done = {}, 0
    while True:
        for v in pending(folder, sizes, recursive):
            done += process(v, log, **analyze_kwargs)
            sizes.pop(v, None)
        if once and not any(n < SETTLE_POLLS for _, n in sizes.values()):
            return done
        time.sleep(every_s)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyse new clips as they arrive in a folder.")
    parser.add_argument("folder")
    parser.add_argument("--every", type=float, default=10.0, help="Seconds between checks (default 10)")
    parser.add_argument("--once", action="store_true", help="Process what's there, then exit")
    parser.add_argument("--recursive", action="store_true", help="Also watch subfolders")
    parser.add_argument("--model", choices=MODELS, default="lstm")
    parser.add_argument("--drill", choices=DRILLS, help="Also analyse strokes (forehand/backhand/serve/auto)")
    parser.add_argument("--hand", choices=HANDS, default="auto")
    args = parser.parse_args()
    if not os.path.isdir(args.folder):
        sys.exit(f"Not a folder: {args.folder}")
    print(f"Watching {args.folder} (Ctrl+C to stop)")
    try:
        n = watch(args.folder, args.every, args.once, args.recursive,
                  model=args.model, drill=args.drill, hand=args.hand)
        print(f"{n} clip(s) analysed")
    except KeyboardInterrupt:
        print("stopped")
