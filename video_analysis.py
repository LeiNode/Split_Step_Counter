"""
VideoAnalysis — reads from a video file and runs split-step detection.
Uses the file's built-in timestamp (CAP_PROP_POS_MSEC) for MediaPipe so
detections are aligned with the actual playback position.
"""

import queue
import threading

import cv2

from detector import _AnalysisWorker


class VideoAnalysis(_AnalysisWorker):
    def __init__(self, video_path: str, frame_queue: queue.Queue,
                 stop_event: threading.Event):
        super().__init__(frame_queue, stop_event)
        self._video_path = video_path

    def _source(self):
        return self._video_path

    def _timestamp_ms(self, cap: cv2.VideoCapture) -> int:
        return int(cap.get(cv2.CAP_PROP_POS_MSEC))
