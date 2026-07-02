"""Image->table homography calibration (pure numpy normalized DLT).

The table homography ``H`` maps detected image pixels ``(u, v)`` to metric table
coordinates ``(x, y)`` in meters. It is the first calibration artifact required
before any hover/contact motion: with it the perception keypoints become table
positions the planner can reason about.

This module is intentionally pure numpy + pyyaml — OpenCV is **not** used. The
solver is the normalized Direct Linear Transform (DLT) with an SVD solve, shared
with :func:`terafold.math.transforms.compute_homography`.

The artifact written by :func:`calibrate_table` carries its own validity flags
(``valid_for_hover`` / ``valid_for_contact``) so the central capability probe in
:mod:`terafold.robot.capabilities` can honor them without re-deriving anything.
"""

from __future__ import annotations

import os
from typing import Any, Dict

import numpy as np
import yaml

from terafold.math.transforms import apply_homography as _apply_homography
from terafold.math.transforms import compute_homography

__all__ = [
    "solve_homography",
    "apply_homography",
    "reprojection_rms",
    "calibrate_table",
]

ArrayLike = Any


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------


def _dump_yaml(path: str, obj: Dict[str, Any]) -> None:
    """Write a plain-Python dict to YAML (creates the parent directory)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def _as_nx2(pts: ArrayLike) -> np.ndarray:
    return np.asarray(pts, dtype=np.float64).reshape(-1, 2)


def _image_meta(image: Any) -> Dict[str, Any]:
    """Record a light reference to the calibration image (never the pixels)."""
    if isinstance(image, str):
        return {"path": image}
    shape = getattr(image, "shape", None)
    if shape is not None:
        return {"shape": [int(s) for s in shape]}
    return {"provided": True}


def _residuals(H: ArrayLike, image_pts: ArrayLike, table_pts: ArrayLike) -> np.ndarray:
    """Per-point Euclidean reprojection distances (meters), shape ``(N,)``."""
    pred = apply_homography(H, image_pts)
    tgt = _as_nx2(table_pts)
    if pred.shape[0] != tgt.shape[0]:
        raise ValueError("image_points and table_points must have the same length")
    return np.sqrt(((pred - tgt) ** 2).sum(axis=1))


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------


def solve_homography(image_pts: ArrayLike, table_pts: ArrayLike) -> np.ndarray:
    """Solve the ``3x3`` image->table homography via normalized DLT (pure numpy).

    Parameters
    ----------
    image_pts, table_pts:
        Matched point sets, each ``(N, 2)`` with ``N >= 4``. ``image_pts`` are
        pixels ``(u, v)``; ``table_pts`` are metric table ``(x, y)`` in meters.

    Returns
    -------
    np.ndarray
        ``3x3`` homography mapping image pixels -> metric table coordinates,
        normalized so ``H[2, 2] == 1``.
    """
    img = _as_nx2(image_pts)
    tab = _as_nx2(table_pts)
    if img.shape[0] != tab.shape[0]:
        raise ValueError(
            f"image_pts ({img.shape[0]}) and table_pts ({tab.shape[0]}) "
            "must have the same number of points"
        )
    if img.shape[0] < 4:
        raise ValueError(f"homography needs >= 4 correspondences, got {img.shape[0]}")
    return compute_homography(img, tab)


def apply_homography(H: ArrayLike, pts: ArrayLike) -> np.ndarray:
    """Apply a ``3x3`` homography to 2D points, always returning ``(N, 2)``."""
    H = np.asarray(H, dtype=np.float64)
    pts = _as_nx2(pts)
    return np.atleast_2d(_apply_homography(H, pts))


def reprojection_rms(H: ArrayLike, image_pts: ArrayLike, table_pts: ArrayLike) -> float:
    """Root-mean-square reprojection error of ``H`` over correspondences (meters)."""
    d = _residuals(H, image_pts, table_pts)
    return float(np.sqrt((d ** 2).mean()))


def calibrate_table(
    image_points: ArrayLike,
    table_points: ArrayLike,
    out: str,
    image: Any = None,
    operator: str = "",
    valid_threshold_m: float = 0.01,
    timestamp: float = 0.0,
) -> Dict[str, Any]:
    """Solve and persist the image->table homography with validity flags.

    Parameters
    ----------
    image_points, table_points:
        Matched correspondences ``(N, 2)`` with ``N >= 4`` (raises otherwise).
    out:
        Destination YAML path. Conventionally
        ``runs/calibration/<robot>_table_homography.yaml`` so the capability
        probe discovers it; the caller passes the resolved path.
    image:
        Optional calibration image (array or path) — only a light reference is
        stored, never the pixels.
    operator:
        Free-form operator id for provenance.
    valid_threshold_m:
        Hover validity gate. ``valid_for_hover`` requires ``rms <= threshold``;
        ``valid_for_contact`` requires the stricter ``rms <= threshold / 2``.
    timestamp:
        Caller-supplied time (kept a pure arg; the CLI passes ``time.time()``).

    Returns
    -------
    dict
        The persisted record plus an ``"out"`` key with the written path.
    """
    img = _as_nx2(image_points)
    tab = _as_nx2(table_points)
    if img.shape[0] != tab.shape[0]:
        raise ValueError(
            f"image_points ({img.shape[0]}) and table_points ({tab.shape[0]}) "
            "must have the same length"
        )
    if img.shape[0] < 4:
        raise ValueError(f"need >= 4 correspondences, got {img.shape[0]}")

    H = solve_homography(img, tab)
    d = _residuals(H, img, tab)
    rms = float(np.sqrt((d ** 2).mean()))
    max_err = float(d.max())
    thr = float(valid_threshold_m)

    record: Dict[str, Any] = {
        "type": "table_homography",
        "source_points": img.tolist(),
        "target_points": tab.tolist(),
        "H": [[float(v) for v in row] for row in np.asarray(H, dtype=np.float64)],
        "rms_error_m": rms,
        "max_error_m": max_err,
        "num_points": int(img.shape[0]),
        "timestamp": float(timestamp),
        "operator": str(operator),
        "valid_threshold_m": thr,
        "valid_for_hover": bool(rms <= thr),
        "valid_for_contact": bool(rms <= thr / 2.0),
    }
    if image is not None:
        record["image"] = _image_meta(image)

    _dump_yaml(out, record)
    result = dict(record)
    result["out"] = out
    return result
