"""Homography / frame-transform persistence and table calibration helpers.

This module builds the image->table homography ``H_IT`` (and the optional
table->robot transform ``T_BT``) used throughout TeraFold, and persists the
:class:`~terafold.math.frames.FrameTransforms` bundle to/from JSON.

Two ways to build ``H_IT`` are offered:

* :func:`homography_from_point_correspondences` -- general ``>=4`` matched
  (image pixel, table metric) pairs.
* :func:`homography_from_table_rectangle` -- the common top-down case where the
  four detected image corners of a known table rectangle map to a metric
  rectangle ``width_m x height_m`` with the table origin at the first corner.

Pure numpy; persistence uses the JSON helpers in
:mod:`terafold.data.episode_schema`.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from terafold.data.episode_schema import read_json, write_json
from terafold.math.frames import FrameTransforms
from terafold.math.transforms import compute_homography

ArrayLike = np.ndarray

__all__ = [
    "save_homography",
    "load_homography",
    "save_frames",
    "load_frames",
    "homography_from_point_correspondences",
    "homography_from_table_rectangle",
]


def homography_from_point_correspondences(
    image_pts: ArrayLike, table_pts: ArrayLike
) -> np.ndarray:
    """Estimate ``H_IT`` mapping image pixels to metric table coordinates.

    Parameters
    ----------
    image_pts, table_pts:
        Matched point sets, each ``(N, 2)`` with ``N >= 4``. ``image_pts`` are
        pixel ``(u, v)``; ``table_pts`` are metric table ``(x, y)``.

    Returns
    -------
    np.ndarray
        ``3x3`` homography mapping image pixels -> table metric coordinates.
    """
    image_arr = np.asarray(image_pts, dtype=np.float64)
    table_arr = np.asarray(table_pts, dtype=np.float64)
    return compute_homography(image_arr, table_arr)


def homography_from_table_rectangle(
    image_corners: ArrayLike, width_m: float, height_m: float
) -> np.ndarray:
    """Build ``H_IT`` from the 4 image corners of a known table rectangle.

    The four detected image corners (ordered TL, TR, BR, BL -- i.e. clockwise
    starting top-left) map to a metric rectangle with the table origin at the
    first corner::

        TL -> (0, 0)
        TR -> (width_m, 0)
        BR -> (width_m, height_m)
        BL -> (0, height_m)

    Parameters
    ----------
    image_corners:
        ``(4, 2)`` pixel coordinates of the rectangle corners.
    width_m, height_m:
        Known metric dimensions of the rectangle (table) in meters.

    Returns
    -------
    np.ndarray
        ``3x3`` homography mapping image pixels -> table metric coordinates.
    """
    image_arr = np.asarray(image_corners, dtype=np.float64)
    if image_arr.shape != (4, 2):
        raise ValueError(
            f"image_corners must be (4, 2) [TL, TR, BR, BL], got {image_arr.shape}"
        )
    w = float(width_m)
    h = float(height_m)
    table_corners = np.array(
        [
            [0.0, 0.0],
            [w, 0.0],
            [w, h],
            [0.0, h],
        ],
        dtype=np.float64,
    )
    return compute_homography(image_arr, table_corners)


def save_homography(
    path: str,
    H_IT: ArrayLike,
    T_BT: Optional[ArrayLike] = None,
    z_table_in_robot: float = 0.0,
    robot_calibrated: bool = False,
    meta: Optional[dict] = None,
) -> None:
    """Persist a homography (and optional table->robot transform) to JSON.

    Parameters
    ----------
    path:
        Destination JSON file.
    H_IT:
        ``3x3`` image->table homography.
    T_BT:
        Optional ``3x3`` table->robot transform; identity if ``None``.
    z_table_in_robot:
        Robot-frame z of the table surface.
    robot_calibrated:
        Whether ``T_BT`` is trustworthy.
    meta:
        Optional extra metadata stored alongside the transforms.
    """
    frames = FrameTransforms(
        H_IT=np.asarray(H_IT, dtype=np.float64),
        T_BT=np.eye(3) if T_BT is None else np.asarray(T_BT, dtype=np.float64),
        z_table_in_robot=float(z_table_in_robot),
        robot_calibrated=bool(robot_calibrated),
    )
    d = frames.to_dict()
    if meta is not None:
        d["meta"] = meta
    write_json(path, d)


def load_homography(path: str) -> FrameTransforms:
    """Load a :class:`FrameTransforms` bundle previously saved as JSON."""
    d = read_json(path)
    return FrameTransforms.from_dict(d)


def save_frames(path: str, frames: FrameTransforms) -> None:
    """Persist a :class:`FrameTransforms` bundle to JSON."""
    write_json(path, frames.to_dict())


def load_frames(path: str) -> FrameTransforms:
    """Load a :class:`FrameTransforms` bundle from JSON."""
    d = read_json(path)
    return FrameTransforms.from_dict(d)
