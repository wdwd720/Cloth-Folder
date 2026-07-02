"""Perception accuracy metrics (numpy-only).

These score the *perception* stage of a fold: how well predicted cloth corners,
grasp/place points, and the fold line match a ground-truth annotation, plus a
keypoint-confidence gate. Everything here is geometry on small point arrays, so
the module imports numpy alone and never touches torch/cv2.

All errors are returned in whatever units the inputs are in (pixels in image
space, meters in table space) — the caller decides. Distances are plain
Euclidean.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

__all__ = [
    "corner_error",
    "point_error",
    "grasp_error",
    "place_error",
    "fold_line_error",
    "confidence_pass",
]


def _as_points(arr: Any) -> np.ndarray:
    """Coerce to an ``(N, 2)`` float array."""
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 1:
        a = a.reshape(-1, 2)
    if a.ndim != 2 or a.shape[1] != 2:
        raise ValueError(f"expected an (N, 2) point array, got shape {a.shape}")
    return a


def _as_point(xy: Any) -> np.ndarray:
    """Coerce to a length-2 float vector."""
    a = np.asarray(xy, dtype=np.float64).reshape(-1)
    if a.size != 2:
        raise ValueError(f"expected a length-2 point, got size {a.size}")
    return a


def corner_error(pred: Any, gt: Any) -> Dict[str, Any]:
    """Per-corner Euclidean error between predicted and ground-truth corners.

    Parameters
    ----------
    pred, gt:
        ``(N, 2)`` arrays of corresponding corner coordinates (same ordering).

    Returns
    -------
    dict with ``mean`` (float), ``max`` (float) and ``per_corner`` (list of
    floats). Identical inputs give all-zero error.
    """
    p = _as_points(pred)
    g = _as_points(gt)
    n = min(len(p), len(g))
    if n == 0:
        return {"mean": 0.0, "max": 0.0, "per_corner": []}
    d = np.linalg.norm(p[:n] - g[:n], axis=1)
    return {
        "mean": float(d.mean()),
        "max": float(d.max()),
        "per_corner": [float(x) for x in d],
    }


def point_error(pred_xy: Any, gt_xy: Any) -> float:
    """Euclidean distance between two points (zero when equal)."""
    return float(np.linalg.norm(_as_point(pred_xy) - _as_point(gt_xy)))


def grasp_error(pred_xy: Any, gt_xy: Any) -> float:
    """Grasp-point localization error (wrapper over :func:`point_error`)."""
    return point_error(pred_xy, gt_xy)


def place_error(pred_xy: Any, gt_xy: Any) -> float:
    """Place-point localization error (wrapper over :func:`point_error`)."""
    return point_error(pred_xy, gt_xy)


def _line_point_dir(line: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(midpoint, unit_direction, sample_point)`` for a fold line.

    Accepts two endpoints (shape ``(2, 2)`` or a flat ``(4,)``), or a mapping
    with ``p1``/``p2`` (endpoints) or ``point``/``direction``.
    """
    if isinstance(line, dict):
        if "direction" in line:
            p0 = _as_point(line.get("point", [0.0, 0.0]))
            d = _as_point(line["direction"])
            return p0, _unit(d), p0
        pts = np.array([_as_point(line["p1"]), _as_point(line["p2"])], dtype=np.float64)
    else:
        a = np.asarray(line, dtype=np.float64).reshape(-1)
        if a.size != 4:
            raise ValueError("fold line must be two endpoints (shape (2,2) or (4,))")
        pts = a.reshape(2, 2)
    mid = pts.mean(axis=0)
    return mid, _unit(pts[1] - pts[0]), pts[0]


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.array([1.0, 0.0])
    return v / n


def fold_line_error(pred_line: Any, gt_line: Any) -> Dict[str, float]:
    """Angle + perpendicular-offset error between a predicted and GT fold line.

    A fold line is undirected, so the angle error is folded into ``[0, 90]``
    degrees. The offset is the perpendicular distance from the predicted line's
    midpoint to the (infinite) ground-truth line, in input units.

    Returns ``{"angle_deg": float, "offset": float}`` — both zero for identical
    lines. (Returns a dict rather than a bare float because the two components
    are independently useful.)
    """
    p_mid, p_dir, _ = _line_point_dir(pred_line)
    g_mid, g_dir, g_p0 = _line_point_dir(gt_line)

    ang_p = np.degrees(np.arctan2(p_dir[1], p_dir[0]))
    ang_g = np.degrees(np.arctan2(g_dir[1], g_dir[0]))
    diff = abs((ang_p - ang_g) % 180.0)
    angle_deg = float(min(diff, 180.0 - diff))

    # Perpendicular distance of the predicted midpoint to the GT line.
    rel = p_mid - g_p0
    offset = float(abs(g_dir[0] * rel[1] - g_dir[1] * rel[0]))
    return {"angle_deg": angle_deg, "offset": offset}


def confidence_pass(confidence: Any, threshold: float = 0.7) -> bool:
    """True iff every confidence value is at or above ``threshold``.

    Accepts a scalar or an array-like of per-keypoint confidences. An empty set
    fails (nothing to be confident about).
    """
    arr = np.asarray(confidence, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return False
    return bool(np.all(arr >= float(threshold)))
