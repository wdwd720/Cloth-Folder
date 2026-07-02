"""Camera intrinsics calibration (OpenCV chessboard, gracefully optional).

This produces the pinhole intrinsics ``K`` and lens distortion ``dist`` from a
set of chessboard images. It is the one calibration step that genuinely needs
OpenCV, so ``cv2`` is **lazy-imported inside the function** and its absence (or
the absence of usable images) is reported as a clean ``status="unavailable"``
result rather than an import-time crash. This keeps the core importable and the
test-suite runnable with only numpy + pyyaml installed.

The written artifact carries ``valid_for_hover`` (a sub-pixel RMS gate) so the
central capability probe in :mod:`terafold.robot.capabilities` can honor it.
"""

from __future__ import annotations

import glob as _glob
from typing import Any, Dict, Tuple

from terafold.calibration.table_homography import _dump_yaml

__all__ = ["calibrate_camera"]

_INSTALL_HINT = "pip install -e '.[vision]'"


def _unavailable(reason: str) -> Dict[str, Any]:
    return {"status": "unavailable", "reason": reason, "install": _INSTALL_HINT}


def calibrate_camera(
    image_glob: str,
    out: str,
    board: Tuple[int, int] = (9, 6),
    square_m: float = 0.025,
) -> Dict[str, Any]:
    """Calibrate camera intrinsics from chessboard images (OpenCV optional).

    Parameters
    ----------
    image_glob:
        Glob pattern matching the calibration images (e.g.
        ``runs/calibration/cam/*.png``).
    out:
        Destination YAML path (conventionally
        ``runs/calibration/<robot>_camera_intrinsics.yaml``).
    board:
        Inner-corner grid of the chessboard as ``(cols, rows)``.
    square_m:
        Physical size of one chessboard square in meters.

    Returns
    -------
    dict
        On success: ``{"status": "ok", "K", "dist", "rms_px", "image_count",
        "valid_for_hover", ..., "out"}``. If OpenCV is missing OR no usable
        images are found, returns ``{"status": "unavailable", "reason", "install"}``
        WITHOUT raising.
    """
    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover - exercised only when cv2 absent
        return _unavailable(f"OpenCV (cv2) is not importable: {exc}")

    import numpy as np

    paths = sorted(_glob.glob(image_glob)) if image_glob else []
    if not paths:
        return _unavailable(f"no images matched {image_glob!r}")

    cols, rows = int(board[0]), int(board[1])

    # One canonical board of metric object points (z = 0 plane).
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= float(square_m)

    objpoints = []
    imgpoints = []
    image_size = None
    used = 0
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]  # (w, h)
        found, corners = cv2.findChessboardCorners(gray, (cols, rows), None)
        if not found:
            continue
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners)
        used += 1

    if used == 0 or image_size is None:
        return _unavailable(
            f"chessboard {cols}x{rows} not detected in any of {len(paths)} image(s)"
        )

    rms_px, K, dist, _rvecs, _tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )

    record: Dict[str, Any] = {
        "type": "camera_intrinsics",
        "status": "ok",
        "K": [[float(v) for v in row] for row in np.asarray(K, dtype=np.float64)],
        "dist": [float(v) for v in np.asarray(dist, dtype=np.float64).reshape(-1)],
        "rms_px": float(rms_px),
        "image_count": int(used),
        "image_width": int(image_size[0]),
        "image_height": int(image_size[1]),
        "board": [cols, rows],
        "square_m": float(square_m),
        "valid_for_hover": bool(float(rms_px) <= 1.0),
    }
    _dump_yaml(out, record)
    result = dict(record)
    result["out"] = out
    return result
