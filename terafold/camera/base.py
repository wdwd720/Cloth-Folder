"""Camera abstraction.

All cameras (mock, OpenCV/webcam) return frames as ``(H, W, 3)`` uint8 RGB so
the rest of the stack never has to care about BGR vs RGB or backend specifics.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class CameraFrame:
    """A single captured frame with a timestamp (seconds, monotonic-ish)."""

    image: np.ndarray  # (H, W, 3) uint8 RGB
    timestamp: float
    frame_index: int = 0


class BaseCamera(abc.ABC):
    """Abstract camera interface."""

    def __init__(self) -> None:
        self._open = False
        self._frame_index = 0

    @abc.abstractmethod
    def connect(self) -> None: ...

    @abc.abstractmethod
    def disconnect(self) -> None: ...

    @abc.abstractmethod
    def read(self) -> CameraFrame:
        """Capture and return the latest frame as RGB uint8."""

    @property
    def is_open(self) -> bool:
        return self._open

    @abc.abstractmethod
    def resolution(self) -> Tuple[int, int]:
        """Return ``(width, height)``."""

    def __enter__(self) -> "BaseCamera":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.disconnect()


__all__ = ["CameraFrame", "BaseCamera"]
