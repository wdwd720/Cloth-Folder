"""OpenCV / webcam camera backend (lazy cv2).

cv2 is imported lazily inside the methods that need it so that
``import terafold.camera.opencv_camera`` always succeeds even when OpenCV is not
installed. If cv2 is missing, a clear, actionable error is raised the moment a
real capture is attempted.
"""

from __future__ import annotations

import time
from typing import Optional, Tuple

import numpy as np

from terafold.camera.base import BaseCamera, CameraFrame
from terafold.config.schema import CameraConfig

__all__ = ["OpenCVCamera", "have_cv2"]

_CV2_INSTALL_HINT = (
    "OpenCV (cv2) is required for the OpenCVCamera backend but is not installed. "
    "Install it with `pip install opencv-python`, or use a MockCamera "
    "(CameraConfig(type='mock')) for the dry-run pipeline."
)


def have_cv2() -> bool:
    """Return True if cv2 can be imported."""
    try:
        import cv2  # noqa: F401
    except Exception:
        return False
    return True


class OpenCVCamera(BaseCamera):
    """A webcam / USB camera backed by ``cv2.VideoCapture``.

    Frames are converted from OpenCV's native BGR to RGB so the rest of the
    stack only ever sees RGB uint8.
    """

    def __init__(self, config: CameraConfig) -> None:
        super().__init__()
        self.config = config
        self._cap = None
        self._t0 = time.monotonic()

    # -- lifecycle ---------------------------------------------------------
    def connect(self) -> None:
        try:
            import cv2
        except Exception as exc:  # pragma: no cover - exercised only w/o cv2
            raise ImportError(_CV2_INSTALL_HINT) from exc

        cap = cv2.VideoCapture(self.config.index)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(
                f"Could not open camera index {self.config.index}. "
                "Check the device is connected and not in use by another app."
            )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.config.width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.config.height))
        cap.set(cv2.CAP_PROP_FPS, float(self.config.fps))
        self._cap = cap
        self._open = True
        self._frame_index = 0

    def disconnect(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._open = False

    # -- capture -----------------------------------------------------------
    def read(self) -> CameraFrame:
        try:
            import cv2
        except Exception as exc:  # pragma: no cover - exercised only w/o cv2
            raise ImportError(_CV2_INSTALL_HINT) from exc

        if self._cap is None or not self._open:
            raise RuntimeError("OpenCVCamera.read() called before connect().")

        ok, bgr = self._cap.read()
        if not ok or bgr is None:
            raise RuntimeError("Failed to read frame from camera.")

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if self.config.flip_horizontal:
            rgb = np.ascontiguousarray(rgb[:, ::-1, :])
        if self.config.flip_vertical:
            rgb = np.ascontiguousarray(rgb[::-1, :, :])

        frame = CameraFrame(
            image=np.ascontiguousarray(rgb, dtype=np.uint8),
            timestamp=time.monotonic() - self._t0,
            frame_index=self._frame_index,
        )
        self._frame_index += 1
        return frame

    def resolution(self) -> Tuple[int, int]:
        if self._cap is not None:
            try:
                import cv2

                w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if w > 0 and h > 0:
                    return (w, h)
            except Exception:
                pass
        return (int(self.config.width), int(self.config.height))
