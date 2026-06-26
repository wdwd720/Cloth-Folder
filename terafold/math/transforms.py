"""Projective and rigid transforms used across TeraFold.

This module is pure numpy and has no TeraFold dependencies. It provides the
planar homography solver (DLT + SVD) used to map between the image plane and
the metric table plane, plus helpers to apply/invert homographies and build
2D rigid/similarity transforms (used for the table <-> robot mapping).

Conventions
-----------
* Points are passed as ``(2,)`` for a single point or ``(N, 2)`` for a batch.
* Homographies are ``3x3`` matrices mapping homogeneous source points to
  homogeneous destination points: ``q ~ H @ p`` with ``p = [u, v, 1]``.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

ArrayLike = np.ndarray

__all__ = [
    "to_homogeneous",
    "from_homogeneous",
    "apply_homography",
    "invert_homography",
    "compute_homography",
    "rigid_transform_2d",
    "similarity_transform_2d",
    "apply_affine_2d",
    "invert_affine_2d",
    "homography_reprojection_error",
]


def _as_points(pts: ArrayLike) -> Tuple[np.ndarray, bool]:
    """Return ``(N, 2)`` float array and a flag marking single-point input."""
    arr = np.asarray(pts, dtype=np.float64)
    single = arr.ndim == 1
    if single:
        arr = arr[None, :]
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"expected points of shape (2,) or (N, 2), got {arr.shape}")
    return arr, single


def to_homogeneous(pts: ArrayLike) -> np.ndarray:
    """Append a column of ones: ``(N, 2) -> (N, 3)``."""
    arr, _ = _as_points(pts)
    return np.concatenate([arr, np.ones((arr.shape[0], 1))], axis=1)


def from_homogeneous(pts_h: ArrayLike) -> np.ndarray:
    """Divide by the last coordinate: ``(N, 3) -> (N, 2)``."""
    arr = np.asarray(pts_h, dtype=np.float64)
    w = arr[..., -1:]
    # Guard against division by zero (points at infinity) -> tiny epsilon.
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)
    return arr[..., :-1] / w


def apply_homography(H: ArrayLike, pts: ArrayLike) -> np.ndarray:
    """Apply a ``3x3`` homography to 2D points.

    Returns the same shape family as the input (``(2,)`` in -> ``(2,)`` out).
    """
    H = np.asarray(H, dtype=np.float64)
    if H.shape != (3, 3):
        raise ValueError(f"homography must be 3x3, got {H.shape}")
    arr, single = _as_points(pts)
    ph = to_homogeneous(arr)  # (N, 3)
    qh = ph @ H.T  # (N, 3)
    out = from_homogeneous(qh)
    return out[0] if single else out


def invert_homography(H: ArrayLike) -> np.ndarray:
    """Return the inverse homography (maps destination back to source)."""
    H = np.asarray(H, dtype=np.float64)
    if H.shape != (3, 3):
        raise ValueError(f"homography must be 3x3, got {H.shape}")
    Hinv = np.linalg.inv(H)
    # Normalize so H[2,2] == 1 when possible (purely cosmetic / numerical).
    if abs(Hinv[2, 2]) > 1e-12:
        Hinv = Hinv / Hinv[2, 2]
    return Hinv


def _normalization_matrix(pts: np.ndarray) -> np.ndarray:
    """Hartley isotropic normalization: centroid to origin, mean dist sqrt(2)."""
    centroid = pts.mean(axis=0)
    shifted = pts - centroid
    mean_dist = np.sqrt((shifted**2).sum(axis=1)).mean()
    if mean_dist < 1e-12:
        scale = 1.0
    else:
        scale = np.sqrt(2.0) / mean_dist
    T = np.array(
        [
            [scale, 0.0, -scale * centroid[0]],
            [0.0, scale, -scale * centroid[1]],
            [0.0, 0.0, 1.0],
        ]
    )
    return T


def compute_homography(src: ArrayLike, dst: ArrayLike) -> np.ndarray:
    """Estimate the homography mapping ``src`` points to ``dst`` points.

    Uses the normalized Direct Linear Transform (DLT) with an SVD solve, which
    handles 4 exact correspondences and least-squares for >4 (over-determined)
    correspondences. This is the standard top-down camera calibration math for
    V0: pick >=4 known table points visible in the image and solve ``H_IT``.

    Parameters
    ----------
    src, dst:
        Matched point sets, each ``(N, 2)`` with ``N >= 4``.

    Returns
    -------
    np.ndarray
        ``3x3`` homography with ``H[2, 2]`` normalized to 1.
    """
    src_arr, _ = _as_points(src)
    dst_arr, _ = _as_points(dst)
    if src_arr.shape[0] != dst_arr.shape[0]:
        raise ValueError("src and dst must have the same number of points")
    n = src_arr.shape[0]
    if n < 4:
        raise ValueError(f"homography needs >= 4 correspondences, got {n}")

    # Normalize both point sets for numerical conditioning.
    Ts = _normalization_matrix(src_arr)
    Td = _normalization_matrix(dst_arr)
    src_n = from_homogeneous(to_homogeneous(src_arr) @ Ts.T)
    dst_n = from_homogeneous(to_homogeneous(dst_arr) @ Td.T)

    # Build the 2N x 9 DLT matrix.
    A = np.zeros((2 * n, 9))
    for i in range(n):
        x, y = src_n[i]
        u, v = dst_n[i]
        A[2 * i] = [-x, -y, -1, 0, 0, 0, u * x, u * y, u]
        A[2 * i + 1] = [0, 0, 0, -x, -y, -1, v * x, v * y, v]

    # Solution is the right singular vector with smallest singular value.
    _, _, Vt = np.linalg.svd(A)
    H_n = Vt[-1].reshape(3, 3)

    # Denormalize: H = Td^-1 @ H_n @ Ts
    H = np.linalg.inv(Td) @ H_n @ Ts
    if abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]
    return H


def homography_reprojection_error(H: ArrayLike, src: ArrayLike, dst: ArrayLike) -> float:
    """Mean Euclidean reprojection error of ``H`` over correspondences."""
    pred = apply_homography(H, src)
    pred, _ = _as_points(pred)
    dst_arr, _ = _as_points(dst)
    return float(np.sqrt(((pred - dst_arr) ** 2).sum(axis=1)).mean())


def rigid_transform_2d(theta: float, tx: float, ty: float) -> np.ndarray:
    """Build a ``3x3`` homogeneous 2D rigid transform (rotation + translation)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array(
        [
            [c, -s, tx],
            [s, c, ty],
            [0.0, 0.0, 1.0],
        ]
    )


def similarity_transform_2d(theta: float, scale: float, tx: float, ty: float) -> np.ndarray:
    """Build a ``3x3`` homogeneous 2D similarity transform (rot + uniform scale + trans)."""
    c, s = np.cos(theta) * scale, np.sin(theta) * scale
    return np.array(
        [
            [c, -s, tx],
            [s, c, ty],
            [0.0, 0.0, 1.0],
        ]
    )


def apply_affine_2d(M: ArrayLike, pts: ArrayLike) -> np.ndarray:
    """Apply a ``3x3`` homogeneous 2D affine/rigid/similarity transform to points."""
    M = np.asarray(M, dtype=np.float64)
    if M.shape != (3, 3):
        raise ValueError(f"affine transform must be 3x3, got {M.shape}")
    arr, single = _as_points(pts)
    out = from_homogeneous(to_homogeneous(arr) @ M.T)
    return out[0] if single else out


def invert_affine_2d(M: ArrayLike) -> np.ndarray:
    """Invert a ``3x3`` homogeneous 2D affine transform."""
    return np.linalg.inv(np.asarray(M, dtype=np.float64))
