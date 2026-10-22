"""Threaded video input supporting files, RTSP sources, and webcams."""

from __future__ import annotations

import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np


class VideoSource:
    """Capture frames on a daemon thread and expose the latest frame safely."""

    def __init__(self, source: str, loop: bool = True, reconnect_delay: float = 2.0,
                 latest_only: bool = True) -> None:
        """Create a source; numeric strings are interpreted as webcam indices."""
        self.source = source
        self.loop = loop
        self.reconnect_delay = reconnect_delay
        self.latest_only = latest_only
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._frame: np.ndarray | None = None
        self._sequence = 0
        self._frames: deque[tuple[int, np.ndarray, float]] = deque(maxlen=1 if latest_only else 8)
        self._stopped = threading.Event()
        self._thread: threading.Thread | None = None
        self._capture: cv2.VideoCapture | None = None
        self._fps = 25.0

    def start(self) -> "VideoSource":
        """Open the source and start its capture worker."""
        if self._thread is not None:
            return self
        self._capture = self._open()
        if not self._capture.isOpened():
            raise RuntimeError(f"Unable to open video source: {self.source}")
        self._fps = self._capture.get(cv2.CAP_PROP_FPS) or 25.0
        self._thread = threading.Thread(target=self._capture_loop, name="video-capture", daemon=True)
        self._thread.start()
        return self

    def _open(self) -> cv2.VideoCapture:
        """Create a VideoCapture for the configured source."""
        if self.source.isdigit():
            return cv2.VideoCapture(int(self.source))
        if not self.source.lower().startswith(("rtsp://", "http://", "https://")):
            path = Path(self.source)
            return cv2.VideoCapture(str(path))
        return cv2.VideoCapture(self.source)

    def _capture_loop(self) -> None:
        """Publish captured frames and reconnect when a live source drops."""
        while not self._stopped.is_set():
            capture = self._capture
            if capture is None:
                break
            ok, frame = capture.read()
            if ok and frame is not None:
                with self._condition:
                    while not self.latest_only and len(self._frames) == self._frames.maxlen and not self._stopped.is_set():
                        self._condition.wait()
                    if self._stopped.is_set():
                        break
                    self._frame = frame
                    self._sequence += 1
                    self._frames.append((self._sequence, frame, time.time()))
                    self._condition.notify_all()
                continue
            if self.loop and not self.source.lower().startswith(("rtsp://", "http://", "https://")):
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            if self.source.lower().startswith(("rtsp://", "http://", "https://")):
                capture.release()
                if self._stopped.wait(self.reconnect_delay):
                    break
                self._capture = self._open()
                continue
            break
        with self._condition:
            self._condition.notify_all()

    def read(self, after_sequence: int = 0, timeout: float = 5.0) -> tuple[np.ndarray | None, int, float]:
        """Wait for a frame newer than after_sequence and return its capture time."""
        deadline = time.monotonic() + timeout
        with self._condition:
            while (not self._frames or self._frames[-1][0] <= after_sequence) and not self._stopped.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0 or (self._thread and not self._thread.is_alive()):
                    break
                self._condition.wait(remaining)
            available = next((item for item in self._frames if item[0] > after_sequence), None)
            if available is None:
                return None, self._sequence, time.time()
            sequence, frame, captured_at = available
            if not self.latest_only:
                self._frames.popleft()
                self._condition.notify_all()
            return frame.copy(), sequence, captured_at

    @property
    def fps(self) -> float:
        """Return the source-reported frame rate when available."""
        return self._fps

    def stop(self) -> None:
        """Stop capture and release the underlying device."""
        self._stopped.set()
        with self._condition:
            self._condition.notify_all()
        if self._capture is not None:
            self._capture.release()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def __enter__(self) -> "VideoSource":
        """Start capture on entry to a with block."""
        return self.start()

    def __exit__(self, *_: object) -> None:
        """Stop capture on exit from a with block."""
        self.stop()
