"""Calibration accuracy metrics (numpy-only, self-contained).

Quantifies how well a calibration maps between spaces:

* a table **homography** (image pixels -> table meters): reprojection RMS and a
  leave-one-out held-out error;
* a robot **touch** check (commanded vs measured table points): RMS;
* boolean gates that decide whether a calibration is good enough to *hover* or
  to make *contact*, given RMS thresholds.

To stay decoupled from the calibration package this module re-implements the
tiny apply-homography and a DLT fit with numpy only — nothing from
``terafold.calibration`` is imported. Pure numpy; no cv2/torch.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "apply_homography",
    "fit_homography",
    "homography_rms",
    "held_out_error",
    "robot_touch_rms",
    "valid_for_hover",
    "valid_for_contact",
]


def _as_points(arr: Any) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 1:
        a = a.reshape(-1, 2)
    if a.ndim != 2 or a.shape[1] != 2:
        raise ValueError(f"expected an (N, 2) point array, got shape {a.shape}")
    return a


def apply_homography(H: Any, pts: Any) -> np.ndarray:
    """Apply a 3x3 homography to ``(N, 2)`` points, returning ``(N, 2)``.

    Points are lifted to homogeneous coordinates, transformed, and projected
    back. A degenerate (near-zero) homogeneous ``w`` is clamped to avoid NaNs.
    """
    Hm = np.asarray(H, dtype=np.float64).reshape(3, 3)
    p = _as_points(pts)
    homog = np.concatenate([p, np.ones((len(p), 1))], axis=1)  # (N, 3)
    out = homog @ Hm.T  # (N, 3)
    w = out[:, 2:3]
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)
    return out[:, :2] / w


def fit_homography(image_pts: Any, table_pts: Any) -> np.ndarray:
    """Fit a 3x3 homography mapping ``image_pts -> table_pts`` via DLT (SVD).

    Needs at least four non-degenerate correspondences. The result is scaled so
    ``H[2, 2] == 1`` when possible.
    """
    src = _as_points(image_pts)
    dst = _as_points(table_pts)
    if len(src) != len(dst) or len(src) < 4:
        raise ValueError("need >= 4 matched correspondences to fit a homography")
    rows = []
    for (x, y), (u, v) in zip(src, dst):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    A = np.asarray(rows, dtype=np.float64)
    _, _, vh = np.linalg.svd(A)
    h = vh[-1].reshape(3, 3)
    if abs(h[2, 2]) > 1e-12:
        h = h / h[2, 2]
    return h


def homography_rms(H: Any, image_pts: Any, table_pts: Any) -> float:
    """Root-mean-square reprojection error of ``H`` over correspondences.

    Maps ``image_pts`` through ``H`` and compares to ``table_pts``; returns the
    RMS of the per-point Euclidean residuals (~0 for a perfect homography).
    """
    pred = apply_homography(H, image_pts)
    gt = _as_points(table_pts)
    n = min(len(pred), len(gt))
    if n == 0:
        return 0.0
    res = np.linalg.norm(pred[:n] - gt[:n], axis=1)
    return float(np.sqrt(np.mean(res ** 2)))


def held_out_error(image_pts: Any, table_pts: Any) -> float:
    """Leave-one-out held-out reprojection RMS.

    For each correspondence: fit a homography on all *other* points and measure
    the reprojection error on the held-out one; return the RMS of those held-out
    residuals. This estimates generalization rather than fit quality. Requires
    at least five correspondences (DLT needs four for the training split).
    """
    src = _as_points(image_pts)
    dst = _as_points(table_pts)
    n = min(len(src), len(dst))
    if n < 5:
        raise ValueError("need >= 5 correspondences for leave-one-out held-out error")
    residuals = []
    idx = np.arange(n)
    for i in range(n):
        train = idx != i
        H = fit_homography(src[train], dst[train])
        pred = apply_homography(H, src[i:i + 1])[0]
        residuals.append(float(np.linalg.norm(pred - dst[i])))
    res = np.asarray(residuals, dtype=np.float64)
    return float(np.sqrt(np.mean(res ** 2)))


def robot_touch_rms(pred: Any, gt: Any) -> float:
    """RMS Euclidean error between predicted and measured touch points.

    ``pred``/``gt`` are ``(N, 2)`` or ``(N, 3)`` table-space points (commanded vs
    where the robot actually touched). ~0 when they coincide.
    """
    p = np.asarray(pred, dtype=np.float64)
    g = np.asarray(gt, dtype=np.float64)
    if p.ndim == 1:
        p = p.reshape(1, -1)
    if g.ndim == 1:
        g = g.reshape(1, -1)
    n = min(len(p), len(g))
    d = min(p.shape[1], g.shape[1])
    if n == 0:
        return 0.0
    res = np.linalg.norm(p[:n, :d] - g[:n, :d], axis=1)
    return float(np.sqrt(np.mean(res ** 2)))


def valid_for_hover(rms: float, threshold: float) -> bool:
    """True iff the calibration RMS is within the (looser) hover threshold."""
    r = float(rms)
    return bool(np.isfinite(r) and r <= float(threshold))


def valid_for_contact(rms: float, threshold: float) -> bool:
    """True iff the calibration RMS is within the (tighter) contact threshold."""
    r = float(rms)
    return bool(np.isfinite(r) and r <= float(threshold))
