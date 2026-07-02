"""Held-out validation of a saved table homography.

Calibration that fits its *own* points perfectly can still be wrong; the honest
check is reprojection error on points that were NOT used to fit. This module
loads a homography artifact written by
:func:`terafold.calibration.table_homography.calibrate_table`, applies it to a
held-out set of image points, and re-derives ``valid_for_hover`` /
``valid_for_contact`` against the thresholds stored in the artifact. A large
held-out error therefore marks the calibration invalid even if the file claimed
otherwise.

Pure numpy + pyyaml.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import yaml

from terafold.calibration.table_homography import apply_homography

__all__ = ["validate_table_calibration"]


def validate_table_calibration(
    calibration_path: str,
    image_points: Any,
    table_points: Any,
) -> Dict[str, Any]:
    """Validate a saved homography against held-out correspondences.

    Parameters
    ----------
    calibration_path:
        Path to a homography YAML written by :func:`calibrate_table` (must carry
        a ``3x3`` ``H`` and, ideally, ``valid_threshold_m``).
    image_points, table_points:
        Held-out matched correspondences ``(N, 2)`` (not used during the fit).

    Returns
    -------
    dict
        ``{"rms_error_m", "max_error_m", "valid_for_hover", "valid_for_contact",
        "valid_threshold_m", "num_points", "calibration_path"}``. ``valid_for_*``
        use the thresholds stored in the artifact (defaulting to ``0.01`` m).
    """
    with open(calibration_path) as f:
        cal = yaml.safe_load(f) or {}

    H = np.asarray(cal.get("H"), dtype=np.float64)
    if H.shape != (3, 3):
        raise ValueError(
            f"calibration {calibration_path!r} has no valid 3x3 homography 'H'"
        )

    thr = float(cal.get("valid_threshold_m", 0.01))
    img = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    tab = np.asarray(table_points, dtype=np.float64).reshape(-1, 2)
    if img.shape[0] != tab.shape[0]:
        raise ValueError(
            f"image_points ({img.shape[0]}) and table_points ({tab.shape[0]}) "
            "must have the same length"
        )
    if img.shape[0] < 1:
        raise ValueError("need >= 1 held-out point to validate")

    pred = apply_homography(H, img)
    err = np.sqrt(((pred - tab) ** 2).sum(axis=1))
    rms = float(np.sqrt((err ** 2).mean()))
    max_err = float(err.max())

    return {
        "calibration_path": calibration_path,
        "num_points": int(img.shape[0]),
        "rms_error_m": rms,
        "max_error_m": max_err,
        "valid_threshold_m": thr,
        "valid_for_hover": bool(rms <= thr),
        "valid_for_contact": bool(rms <= thr / 2.0),
    }
