"""Camera <-> table calibration helpers.

The non-interactive core (:func:`run_calibration`) is numpy-only and is what the
rest of the stack and tests rely on. :func:`capture_calibration_image` grabs a
frame from any :class:`BaseCamera` and writes it via the pure image I/O.
:func:`interactive_calibrate` offers a cv2-based point clicker when OpenCV is
available, and otherwise degrades to clear textual guidance.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from terafold.camera.base import BaseCamera
from terafold.camera.homography import (
    homography_from_point_correspondences,
    save_frames,
)
from terafold.math.frames import FrameTransforms
from terafold.vision.imageio import imread, imwrite

__all__ = [
    "run_calibration",
    "capture_calibration_image",
    "interactive_calibrate",
]


def run_calibration(
    image_pts: Sequence[Sequence[float]],
    table_pts: Sequence[Sequence[float]],
    out_path: str,
) -> FrameTransforms:
    """Estimate ``H_IT`` from point correspondences and save it.

    Parameters
    ----------
    image_pts, table_pts:
        Matched ``(N, 2)`` point sets (``N >= 4``); pixels and metric table xy.
    out_path:
        Destination JSON file for the :class:`FrameTransforms` bundle.

    Returns
    -------
    FrameTransforms
        Bundle with ``image_calibrated=True``.
    """
    image_arr = np.asarray(image_pts, dtype=np.float64)
    table_arr = np.asarray(table_pts, dtype=np.float64)
    if image_arr.shape != table_arr.shape or image_arr.ndim != 2 or image_arr.shape[1] != 2:
        raise ValueError(
            "image_pts and table_pts must both be (N, 2) and the same shape, "
            f"got {image_arr.shape} and {table_arr.shape}"
        )
    if image_arr.shape[0] < 4:
        raise ValueError("Need at least 4 point correspondences for a homography.")

    H_IT = homography_from_point_correspondences(image_arr, table_arr)
    frames = FrameTransforms(H_IT=H_IT, image_calibrated=True)
    save_frames(out_path, frames)
    return frames


def capture_calibration_image(camera: BaseCamera, out_path: str) -> str:
    """Capture a single frame from ``camera`` and write it to ``out_path``.

    Returns the path written. Connects the camera if it is not already open.
    """
    opened_here = False
    if not camera.is_open:
        camera.connect()
        opened_here = True
    try:
        frame = camera.read()
        imwrite(out_path, frame.image)
    finally:
        if opened_here:
            camera.disconnect()
    return out_path


def interactive_calibrate(
    image_path: str,
    out_path: str,
    table_points: Optional[Sequence[Sequence[float]]] = None,
) -> FrameTransforms:
    """Pick image points for known table points and save the calibration.

    If OpenCV is available, opens a window so the operator can click the image
    points corresponding to ``table_points`` (defaulting to a unit-square's four
    corners). Without cv2, raises a clear error explaining how to supply the
    correspondences programmatically via :func:`run_calibration`.

    Parameters
    ----------
    image_path:
        Path to a previously captured calibration image.
    out_path:
        Destination JSON for the :class:`FrameTransforms` bundle.
    table_points:
        ``(N, 2)`` metric table coordinates the operator will click in order.
        Defaults to a ``0.30 x 0.30`` m square's corners (TL, TR, BR, BL).
    """
    if table_points is None:
        table_points = [[0.0, 0.0], [0.30, 0.0], [0.30, 0.30], [0.0, 0.30]]
    table_arr = np.asarray(table_points, dtype=np.float64)
    n = table_arr.shape[0]

    image = imread(image_path)

    try:
        import cv2
    except Exception as exc:
        raise ImportError(
            "interactive_calibrate needs OpenCV (cv2) for the point clicker. "
            "Install it with `pip install opencv-python`, or call "
            "terafold.camera.calibration_cli.run_calibration(image_pts, table_pts, out_path) "
            "directly with manually measured pixel correspondences."
        ) from exc

    clicked: List[List[float]] = []
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    window = "calibrate: click %d table points in order" % n

    def _on_mouse(event, x, y, flags, param):  # pragma: no cover - GUI
        if event == cv2.EVENT_LBUTTONDOWN and len(clicked) < n:
            clicked.append([float(x), float(y)])
            cv2.circle(bgr, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow(window, bgr)

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, _on_mouse)
    cv2.imshow(window, bgr)
    while len(clicked) < n:  # pragma: no cover - GUI loop
        if cv2.waitKey(20) & 0xFF == 27:  # ESC to abort
            break
    cv2.destroyWindow(window)

    if len(clicked) < n:
        raise RuntimeError(
            f"Calibration aborted: clicked {len(clicked)}/{n} required points."
        )

    return run_calibration(clicked, table_arr, out_path)
