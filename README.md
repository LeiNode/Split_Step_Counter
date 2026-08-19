# Split_Step_Counter
A computer vision project that takes in a video and counts the number of times a split step has been completed.

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```
python main.py
```

Launches the GUI — pick a video file or start a live webcam feed. The pose
model (~9 MB) downloads automatically on first run into `models/`.

## Project layout

```
split_step/              detection + GUI package
  detector.py             SplitStepDetector state machine, landmark utilities
  app.py                  tkinter GUI
  video_analysis.py       analysis worker for video files
  live_view_analysis.py   analysis worker for webcam input
  lstm_counter.py         LSTM-based drop-in replacement for the detector
  pose_csv_recorder.py    records pose landmarks from a video to CSV
  csv_labeler.py          labels a recorded pose CSV with split-step ranges
model_experimentation/   LSTM training notebook + script (not part of the app)
models/                  pose_landmarker_full.task (downloaded), split_step_lstm.pt
assets/                  recorded videos / CSVs / result plots (large media gitignored)
main.py                  entry point
```
