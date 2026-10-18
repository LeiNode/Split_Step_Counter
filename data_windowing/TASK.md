# Task: prune labeled pose data into ~300 ms windows

Hand-off spec for whoever (human or agent) picks this up next.

## Goal
Turn a labeled pose CSV (e.g. `assets/csv_files/labeled_training_set_1.csv`) into
fixed-length ~300 ms windows ready to feed the split-step LSTM.

## Requirements
1. **Window length ≈ 300 ms.** Frames per window = `round(300 / median_frame_ms)`.
   The current data runs at ~30 fps (33 ms/frame), which gives **9 frames**.
2. **One video per window.** A window never mixes frames from two
   `source_video` values. It also never crosses a timestamp gap, meaning a jump
   of more than 1.5× the median frame interval, which marks dropped frames.
3. **Short windows get padded or dropped.** Near the start of a segment there
   are fewer than `seq_len` frames available:
   - if at least `--min-frames` real frames exist, the window is left-padded
     with zeros up to `seq_len`;
   - otherwise the window is dropped.

   The default is `min_frames = seq_len`, which means no padding.
4. **Empty poses are dropped, never filled.** A row is dropped if any feature
   is NaN or every feature is 0. No forward-fill or interpolation is done.
   Because a dropped row leaves a timestamp gap, it also splits the segment
   (rule 2).

## Input format
The CSV must have:
- the columns `timestamp_ms` and `label` (0 or 1);
- `<landmark>_world_{x,y,z}` for each of the 16 feature landmarks;
- optionally `source_video`. Without it, the whole file is treated as one video.

The landmark order is fixed. It must match `FEATURE_LANDMARK_NAMES` in
`model_experimentation/lstm_model.ipynb` and `LANDMARK_INDICES` in
`split_step/lstm_counter.py`.

## Output (`window_dataset.py --out X.npz`)
| key | shape | meaning |
|---|---|---|
| `X` | (N, seq_len, 96) | 48 positions plus 48 in-window velocities. Padded frames are all zeros, and the first real frame has velocity 0. |
| `y` | (N,) | label of the **last** frame (causal, the same as in the notebook and `LSTMCounter`) |
| `mask` | (N, seq_len) | True for real frames, False for padding |
| `end_timestamp_ms` | (N,) | timestamp of the labeled (last) frame |
| `video`, `segment` | (N,) | where each window came from, for grouped train/val splits |

The script also writes `X.summary.json` with row, drop, window and positive
counts per video. The optional `--pruned-csv` writes the cleaned per-frame
table with segment ids.

Features are **not normalized** here. Fit the mean and std on the training
windows only (use `mask` to skip padded frames), the same way the notebook does.

## Commands
```bash
python data_windowing/window_dataset.py assets/csv_files/labeled_training_set_1.csv \
    --out data_windowing/output/training_set_1_windows.npz --min-frames 5 \
    --pruned-csv data_windowing/output/training_set_1_pruned.csv
python data_windowing/window_dataset.py assets/csv_files/test_set_labeled.csv \
    --out data_windowing/output/test_set_windows.npz --min-frames 5
```

## Results so far (2026-09-24, `--min-frames 5`)
- Training set: 4181 rows in, 52 empty rows dropped, 4129 kept, 15 contiguous
  segments across 7 videos. That produced **4069 windows of shape (9, 96)**:
  540 positive and 57 zero-padded.
- Test set: 1350 windows.
- Checks passed: padded frames are exactly 0 and there are no NaNs.

## Open items for the next agent
- **Not currently wired into training.** Since 2026-10-01 the deployed model
  is a *stateful* per-frame LSTM (`split_step/lstm_counter.py`): the notebook
  trains it on continuous sequences, and inference carries hidden state frame
  to frame, with no fixed window. These ~300 ms windows only apply if a
  windowed classifier is revisited. In that case, also make `LSTMCounter`
  zero-pad short windows the same way, and regenerate `models/feat_mean.npy`
  and `models/feat_std.npy`.
- **Consider a grouped validation split.** Splitting by `video` or `segment`
  instead of a random shuffle would avoid near-duplicate overlapping windows
  leaking into validation.
- **Scope.** Side-view-only training data is intentional. Don't expand it
  unprompted.
