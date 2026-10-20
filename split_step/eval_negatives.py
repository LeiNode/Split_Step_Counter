"""
eval_negatives.py — how often does the deployed LSTM fire where there's no split step?

TODO.md "Hard-negative verification": the negatives in the labelled CSVs
(serves, forehand drills, walking, ...) were never checked to be *hard*
negatives. This replays each source video's labelled frames through
LSTMCounter exactly as at inference (same threshold, cooldown and gap
handling, timestamps as the clock) and reports per video:

  neg_fp_rate        share of label-0 frames the model calls "in a hop"
  false_per_min      counted split steps that start on a label-0 frame, per
                     minute of label-0 footage (only with MIN_NEG_S of it:
                     one false start in a 3-frame segment isn't "600 a minute")
  pos_recall         share of label-1 frames the model calls "in a hop"

Frames on the training CSVs were seen in training, so their numbers are
optimistic; test_set_labeled.csv is the honest held-out check.

Usage
-----
    python -m split_step.eval_negatives assets/csv_files/labeled_training_set_2.csv [more.csv ...]
"""

import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

from .lstm_counter import _THRESHOLD, LSTMCounter

LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]
MIN_NEG_S = 5
_COLS = [f"{n}_world_{c}" for n in LANDMARK_NAMES for c in ("x", "y", "z", "vis")]


def _frames(df: pd.DataFrame):
    """Per-row landmark lists shaped like MediaPipe's (x, y, z, visibility objects)."""
    arr = df[_COLS].to_numpy(np.float32).reshape(len(df), len(LANDMARK_NAMES), 4)
    for row in arr:
        yield [SimpleNamespace(x=x, y=y, z=z, visibility=v) for x, y, z, v in row]


def evaluate_video(df: pd.DataFrame, counter: LSTMCounter) -> dict:
    """Replay one video's rows (sorted by time) and score them against "label"."""
    df = df.sort_values("timestamp_ms")
    now = [0.0]
    counter._clock = lambda: now[0]
    counter.reset()
    on, starts = [], []
    for t, lms in zip(df["timestamp_ms"].to_numpy(float), _frames(df)):
        now[0] = t / 1000.0
        starts.append(counter.process(lms))
        on.append(counter.tracking and counter.last_prob >= _THRESHOLD)
    on, starts, label = np.array(on), np.array(starts), df["label"].to_numpy() == 1
    ts = df["timestamp_ms"].to_numpy(float)
    dt = np.diff(ts, append=ts[-1] + (np.median(np.diff(ts)) if len(ts) > 1 else 33))
    neg_minutes = float(np.sum(np.minimum(dt[~label], 150)) / 60000)   # gaps between segments don't count
    return _scores({"frames": int(len(df)), "neg_frames": int((~label).sum()), "pos_frames": int(label.sum()),
                    "neg_on": int(on[~label].sum()), "pos_on": int(on[label].sum()),
                    "false_starts": int(starts[~label].sum()), "neg_minutes": round(neg_minutes, 4)})


def _scores(c: dict) -> dict:
    """Add the rates to a dict of raw counts."""
    return {**c,
            "neg_fp_rate": round(c["neg_on"] / c["neg_frames"], 4) if c["neg_frames"] else None,
            "false_per_min": (round(c["false_starts"] / c["neg_minutes"], 2)
                              if c["neg_minutes"] * 60 >= MIN_NEG_S else None),
            "pos_recall": round(c["pos_on"] / c["pos_frames"], 4) if c["pos_frames"] else None}


def evaluate_csv(path: str, counter: LSTMCounter | None = None) -> dict:
    """{source_video: evaluate_video(...)} plus "ALL" (rates over the summed counts)."""
    counter = counter or LSTMCounter()
    usecols = ["timestamp_ms", "label", "source_video", *_COLS]
    df = pd.read_csv(path, usecols=lambda c: c in usecols)
    if "source_video" not in df:
        df["source_video"] = "all"
    out = {str(v): evaluate_video(g, counter) for v, g in df.groupby("source_video", sort=True)}
    keys = ("frames", "neg_frames", "pos_frames", "neg_on", "pos_on", "false_starts", "neg_minutes")
    out["ALL"] = _scores({k: sum(r[k] for r in out.values()) for k in keys})
    return out


def _print(path: str, results: dict) -> None:
    print(f"\n{path}")
    print(f"  {'video':40s} {'frames':>7s} {'neg':>6s} {'pos':>5s} {'neg FP rate':>12s} {'false/min':>10s} {'pos recall':>11s}")
    fmt = lambda v, p: "-" if v is None else (f"{v:.1%}" if p else f"{v:.2f}")
    for name, r in results.items():
        print(f"  {name[:40]:40s} {r['frames']:7d} {r['neg_frames']:6d} {r['pos_frames']:5d} "
              f"{fmt(r['neg_fp_rate'], True):>12s} {fmt(r['false_per_min'], False):>10s} {fmt(r['pos_recall'], True):>11s}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    counter = LSTMCounter()
    for p in sys.argv[1:]:
        _print(p, evaluate_csv(p, counter))
