"""
window_dataset.py — prune a labeled pose CSV and slice it into ~300 ms windows
for the split-step LSTM. See TASK.md in this folder for the full spec.

Pipeline
--------
1. Load one or more labeled CSVs (must have `timestamp_ms`, `label`, and the
   16 feature landmarks' world x/y/z; `source_video` is optional — without it
   the whole file is one video).
2. Drop empty pose rows (any feature NaN, or every feature exactly 0). Rows are
   dropped outright — nothing is forward-filled.
3. Split each video into contiguous segments: a new segment starts wherever the
   timestamp gap exceeds GAP_FACTOR x the video's median frame interval (i.e.
   where frames were dropped). Windows never cross a video or segment boundary.
4. For every frame, build a causal window of the SEQ_LEN frames ending at it
   (label = that last frame's label, matching the notebook and LSTMCounter).
   Near a segment start, where fewer than SEQ_LEN frames exist:
     - >= min_frames real frames -> left-pad with zeros to SEQ_LEN
     - <  min_frames real frames -> drop the window
5. Features per frame = positions (48) + in-window velocities (48) = 96.
   Velocity is 0 on the first real frame and on every padded frame.

Usage
-----
    python data_windowing/window_dataset.py \\
        assets/csv_files/labeled_training_set_1.csv \\
        --out data_windowing/output/training_set_1_windows.npz
"""

import argparse
import json
import pathlib
from typing import List

import numpy as np
import pandas as pd

# Same order as split_step/lstm_counter.py LANDMARK_INDICES and the notebook's
# FEATURE_LANDMARK_NAMES — changing it silently breaks the trained model.
FEATURE_LANDMARK_NAMES = [
    "left_ankle",    "left_heel",     "left_foot_index",
    "right_ankle",   "right_heel",    "right_foot_index",
    "left_knee",     "right_knee",
    "left_hip",      "right_hip",
    "left_shoulder", "right_shoulder",
    "left_elbow",    "right_elbow",
    "left_wrist",    "right_wrist",
]
FEATURE_COLS = [f"{n}_world_{ax}" for n in FEATURE_LANDMARK_NAMES for ax in ("x", "y", "z")]

WINDOW_MS  = 300.0
GAP_FACTOR = 1.5    # gap > 1.5x median frame interval => dropped frame(s) => new segment


def load_and_prune(csv_path: str) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(csv_path)
    missing = [c for c in FEATURE_COLS + ["timestamp_ms", "label"] if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path}: missing columns {missing}")

    stem = pathlib.Path(csv_path).stem
    videos = df["source_video"].astype(str) if "source_video" in df.columns else "all"
    df = df[["timestamp_ms", "label"] + FEATURE_COLS].assign(video=stem + ":" + videos)

    feats = df[FEATURE_COLS]
    empty = feats.isna().any(axis=1) | (feats.fillna(0).abs().sum(axis=1) == 0)
    stats = {"rows_in": len(df), "rows_dropped_empty": int(empty.sum())}
    df = df.loc[~empty].sort_values(["video", "timestamp_ms"], kind="stable")
    df = df.drop_duplicates(subset=["video", "timestamp_ms"], keep="first")
    stats["rows_kept"] = len(df)
    return df[["video", "timestamp_ms", "label"] + FEATURE_COLS].reset_index(drop=True), stats


def assign_segments(df: pd.DataFrame) -> pd.DataFrame:
    """Add `frame_ms` (per-video median interval) and `segment` (contiguous run id)."""
    parts = []
    for video, g in df.groupby("video", sort=False):
        dt = g["timestamp_ms"].diff()
        frame_ms = float(dt.median()) if len(g) > 1 else 1000.0 / 30
        new_seg = dt.isna() | (dt > GAP_FACTOR * frame_ms)
        parts.append(g.assign(frame_ms=frame_ms,
                              segment=video + "#" + new_seg.cumsum().astype(str)))
    return pd.concat(parts, ignore_index=True)


def make_windows(positions: np.ndarray, seq_len: int, min_frames: int):
    """Yield (end_index, window[seq_len, 96], n_real) for one contiguous segment."""
    n_feat = positions.shape[1]
    for end in range(len(positions)):
        n_real = min(end + 1, seq_len)
        if n_real < min_frames:
            continue
        real = positions[end + 1 - n_real : end + 1]
        vel = np.zeros_like(real)
        vel[1:] = real[1:] - real[:-1]
        win = np.zeros((seq_len, n_feat * 2), dtype=np.float32)
        win[seq_len - n_real:] = np.concatenate([real, vel], axis=1)
        yield end, win, n_real


def build(csv_paths: List[str], window_ms: float, min_frames: int | None, seq_len: int | None):
    frames, stats = [], {}
    for p in csv_paths:
        df, s = load_and_prune(p)
        frames.append(df)
        stats[p] = s
    df = assign_segments(pd.concat(frames, ignore_index=True))

    if seq_len is None:
        seq_len = max(1, round(window_ms / df["frame_ms"].median()))
    if min_frames is None:
        min_frames = seq_len          # default: no padding, drop short windows
    min_frames = max(1, min(min_frames, seq_len))

    X, y, mask, end_ts, vids, segs = [], [], [], [], [], []
    seg_dropped = 0
    for seg, g in df.groupby("segment", sort=False):
        if len(g) < min_frames:
            seg_dropped += 1
        pos = g[FEATURE_COLS].to_numpy(np.float32)
        labels = g["label"].to_numpy(int)
        ts = g["timestamp_ms"].to_numpy(float)
        for end, win, n_real in make_windows(pos, seq_len, min_frames):
            m = np.zeros(seq_len, dtype=bool)
            m[seq_len - n_real:] = True
            X.append(win); y.append(labels[end]); mask.append(m)
            end_ts.append(ts[end]); vids.append(g["video"].iat[0]); segs.append(seg)

    out = {
        "X": np.stack(X) if X else np.zeros((0, seq_len, len(FEATURE_COLS) * 2), np.float32),
        "y": np.array(y, dtype=np.int64),
        "mask": np.array(mask, dtype=bool).reshape(-1, seq_len),
        "end_timestamp_ms": np.array(end_ts),
        "video": np.array(vids),
        "segment": np.array(segs),
    }
    summary = {
        "inputs": stats,
        "window_ms": window_ms,
        "seq_len": seq_len,
        "min_frames": min_frames,
        "median_frame_ms": float(df["frame_ms"].median()),
        "segments": int(df["segment"].nunique()),
        "segments_too_short": seg_dropped,
        "windows": int(len(out["y"])),
        "positive_windows": int(out["y"].sum()),
        "padded_windows": int((~out["mask"]).any(axis=1).sum()),
        "feature_landmarks": FEATURE_LANDMARK_NAMES,
        "per_video": {
            v: {"windows": int((out["video"] == v).sum()),
                "positive": int(out["y"][out["video"] == v].sum())}
            for v in pd.unique(out["video"])
        },
    }
    return out, df, summary


def main():
    ap = argparse.ArgumentParser(description="Prune labeled pose CSV(s) and build ~300 ms windows.")
    ap.add_argument("csv", nargs="+", help="Labeled CSV(s), e.g. assets/csv_files/labeled_training_set_1.csv")
    ap.add_argument("--out", required=True, help="Output .npz path")
    ap.add_argument("--window-ms", type=float, default=WINDOW_MS, help="Target window length (default 300)")
    ap.add_argument("--seq-len", type=int, help="Override frames per window (default: round(window_ms / frame_ms))")
    ap.add_argument("--min-frames", type=int,
                    help="Min real frames to keep a window; shorter ones are zero-padded up to seq_len. "
                         "Default = seq_len (no padding, short windows dropped).")
    ap.add_argument("--pruned-csv", help="Optional: also write the pruned per-frame CSV with segment ids")
    args = ap.parse_args()

    out, df, summary = build(args.csv, args.window_ms, args.min_frames, args.seq_len)

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **out)
    out_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    if args.pruned_csv:
        pathlib.Path(args.pruned_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.pruned_csv, index=False)

    print(json.dumps({k: v for k, v in summary.items() if k not in ("feature_landmarks",)}, indent=2))
    print(f"X shape {out['X'].shape} -> {out_path}")


if __name__ == "__main__":
    main()
