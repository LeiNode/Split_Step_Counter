"""
LiveViewAnalysis — reads from a webcam and runs split-step detection.
Uses wall-clock time for MediaPipe timestamps so the feed never stalls
even when cv2.CAP_PROP_POS_MSEC returns 0 on a live camera.
"""

import queue
import threading
import time

import cv2

from detector import _AnalysisWorker


class LiveViewAnalysis(_AnalysisWorker):
    def __init__(self, frame_queue: queue.Queue, stop_event: threading.Event,
                 camera_index: int = 0):
        super().__init__(frame_queue, stop_event)
        self._camera_index  = camera_index
        self._session_start = 0.0

    def _source(self):
        return self._camera_index

    def _timestamp_ms(self, cap: cv2.VideoCapture) -> int:
        return int((time.perf_counter() - self._session_start) * 1000)

    def run(self) -> None:
        self._session_start = time.perf_counter()
        super().run()
