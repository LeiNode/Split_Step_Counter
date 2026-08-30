"""
Shared filesystem paths for model artifacts.

Kept in one place so detector.py, lstm_counter.py, and model_experimentation
agree on where things live without redefining the strings.
"""

MODELS_DIR = "models"

POSE_LANDMARKER_PATH = f"{MODELS_DIR}/pose_landmarker_full.task"
SPLIT_STEP_LSTM_PATH = f"{MODELS_DIR}/split_step_lstm.pt"
FEAT_MEAN_PATH        = f"{MODELS_DIR}/feat_mean.npy"
FEAT_STD_PATH         = f"{MODELS_DIR}/feat_std.npy"
