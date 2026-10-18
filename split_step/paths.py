"""
Shared filesystem paths for model artifacts.

Kept in one place so detector.py, lstm_counter.py, and model_experimentation
agree on where things live without redefining the strings.

Resolves to <repo>/models regardless of the current working directory, so the
GUI, CLI scripts and the API all find the same files. Set SPLIT_STEP_MODELS_DIR
to point somewhere else (e.g. a mounted volume in a container).
"""

import os

MODELS_DIR = os.environ.get(
    "SPLIT_STEP_MODELS_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"),
)

POSE_LANDMARKER_PATH = os.path.join(MODELS_DIR, "pose_landmarker_full.task")
SPLIT_STEP_LSTM_PATH = os.path.join(MODELS_DIR, "split_step_lstm.pt")
FEAT_MEAN_PATH       = os.path.join(MODELS_DIR, "feat_mean.npy")
FEAT_STD_PATH        = os.path.join(MODELS_DIR, "feat_std.npy")
