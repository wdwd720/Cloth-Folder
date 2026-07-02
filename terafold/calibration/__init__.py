"""TeraFold calibration scaffolding.

Three artifacts gate physical motion above Level 4 (calibrated hover) and are
required before any contact:

* ``table_homography`` — image pixels -> metric table coordinates (pure numpy).
* ``camera_intrinsics`` — pinhole ``K`` + distortion (OpenCV, gracefully
  optional; reports ``status="unavailable"`` when ``cv2`` or images are absent).
* ``robot_table_transform`` — rigid table -> robot-base alignment (Umeyama).

Each calibrator writes a YAML artifact carrying ``valid_for_hover`` /
``valid_for_contact`` flags so the central capability probe in
:mod:`terafold.robot.capabilities` can honor them without re-deriving anything.
Calibration math is pure numpy/pyyaml; OpenCV is lazy-imported and optional.
"""

from __future__ import annotations

from terafold.calibration.camera_intrinsics import calibrate_camera
from terafold.calibration.robot_table_transform import (
    fit_robot_table_transform,
    robot_touch_calibration,
    umeyama,
)
from terafold.calibration.table_homography import (
    apply_homography,
    calibrate_table,
    reprojection_rms,
    solve_homography,
)
from terafold.calibration.validation import validate_table_calibration

__all__ = [
    "solve_homography",
    "apply_homography",
    "reprojection_rms",
    "calibrate_table",
    "calibrate_camera",
    "umeyama",
    "robot_touch_calibration",
    "fit_robot_table_transform",
    "validate_table_calibration",
]
