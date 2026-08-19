"""
VideoAnalysis — reads from a video file and runs split-step detection.
Uses the file's built-in timestamp (CAP_PROP_POS_MSEC) for MediaPipe so
detections are aligned with the actual playback position.
"""

import queue
import threading

import cv2

from .detector import _AnalysisWorker


class VideoAnalysis(_AnalysisWorker):
    def __init__(self, video_path: str, frame_queue: queue.Queue,
                 stop_event: threading.Event, use_lstm: bool = False):
        super().__init__(frame_queue, stop_event)
        self._video_path = video_path
        self._use_lstm   = use_lstm

    def _make_detector(self):
        if self._use_lstm:
            from .lstm_counter import LSTMCounter
            return LSTMCounter()
        return super()._make_detector()

    def _source(self):
        return self._video_path

    def _timestamp_ms(self, cap: cv2.VideoCapture) -> int:
        return int(cap.get(cv2.CAP_PROP_POS_MSEC))

    def _frame_delay_s(self, cap: cv2.VideoCapture) -> float:
        fps = cap.get(cv2.CAP_PROP_FPS)
        return 1.0 / fps if fps > 0 else 0.0
