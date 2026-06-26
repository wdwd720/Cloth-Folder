"""Geometric primitives for fold planning.

The centerpiece is :func:`reflect_point_across_line`, which implements the
"fold over a crease" approximation: a half-fold maps a grasp point to its
mirror image across the crease line. Everything here is pure numpy with no
TeraFold dependencies, so it can be unit-tested in isolation and reused by the
physics, planning, and vision layers.

Lines are represented as ``(point_on_line, direction)`` pairs. Directions need
not be unit length — functions normalize internally.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

ArrayLike = np.ndarray

__all__ = [
    "as_vector",
    "normalize",
    "reflect_point_across_line",
    "project_point_onto_line",
    "point_line_distance",
    "line_from_two_points",
    "line_angle",
    "midpoint",
    "fold_line_from_corners",
    "polygon_area",
    "polygon_centroid",
    "polygon_perimeter",
    "order_corners_clockwise",
    "line_intersection",
]


def as_vector(p: ArrayLike) -> np.ndarray:
    """Coerce to a 1D float vector."""
    return np.asarray(p, dtype=np.float64).reshape(-1)


def normalize(v: ArrayLike, eps: float = 1e-12) -> np.ndarray:
    """Return the unit vector in the direction of ``v`` (safe for near-zero)."""
    v = as_vector(v)
    n = float(np.linalg.norm(v))
    if n < eps:
        raise ValueError("cannot normalize a near-zero-length vector")
    return v / n


def reflect_point_across_line(
    point: ArrayLike, line_point: ArrayLike, line_direction: ArrayLike
) -> np.ndarray:
    """Reflect ``point`` across the line through ``line_point`` along ``line_direction``.

    Implements the exact formula from the TeraFold spec:

        d = line_direction / ||line_direction||
        projection  = c + d * dot(point - c, d)
        perpendicular = point - projection
        reflected   = point - 2 * perpendicular

    Works for 2D or 3D points. For a half-fold over a vertical center crease,
    the place point is the reflection of the grasp point. Reflecting twice
    returns the original point; a point already on the line is unchanged.
    """
    p = as_vector(point)
    c = as_vector(line_point)
    d = normalize(line_direction)
    if p.shape != c.shape or p.shape != d.shape:
        raise ValueError(
            f"dimension mismatch: point{p.shape}, line_point{c.shape}, dir{d.shape}"
        )
    projection = c + d * float(np.dot(p - c, d))
    perpendicular = p - projection
    reflected = p - 2.0 * perpendicular
    return reflected


def project_point_onto_line(
    point: ArrayLike, line_point: ArrayLike, line_direction: ArrayLike
) -> np.ndarray:
    """Orthogonal projection of ``point`` onto the line."""
    p = as_vector(point)
    c = as_vector(line_point)
    d = normalize(line_direction)
    return c + d * float(np.dot(p - c, d))


def point_line_distance(
    point: ArrayLike, line_point: ArrayLike, line_direction: ArrayLike
) -> float:
    """Perpendicular distance from ``point`` to the line."""
    p = as_vector(point)
    proj = project_point_onto_line(p, line_point, line_direction)
    return float(np.linalg.norm(p - proj))


def line_from_two_points(p1: ArrayLike, p2: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(point_on_line, unit_direction)`` for the line through p1, p2."""
    a = as_vector(p1)
    b = as_vector(p2)
    return a, normalize(b - a)


def line_angle(line_direction: ArrayLike) -> float:
    """Angle (radians) of a 2D line direction relative to +x axis, in (-pi, pi]."""
    d = as_vector(line_direction)
    if d.shape[0] < 2:
        raise ValueError("line_angle requires a 2D direction")
    return float(np.arctan2(d[1], d[0]))


def midpoint(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """Midpoint of two points."""
    return 0.5 * (as_vector(a) + as_vector(b))


def fold_line_from_corners(
    top_left: ArrayLike,
    top_right: ArrayLike,
    bottom_left: ArrayLike,
    bottom_right: ArrayLike,
    direction: str = "right_to_left",
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the crease line from four cloth corners.

    For ``right_to_left`` / ``left_to_right`` the crease is the *vertical*
    center line: it passes through the midpoint of the cloth and runs parallel
    to the average of the left and right edges. This is translation- and
    rotation-stable because it is derived from edge midpoints, not absolute
    pixel positions.

    For ``top_to_bottom`` / ``bottom_to_top`` the crease is the *horizontal*
    center line.

    Returns ``(point_on_crease, unit_direction)``.
    """
    tl = as_vector(top_left)
    tr = as_vector(top_right)
    bl = as_vector(bottom_left)
    br = as_vector(bottom_right)
    center = 0.25 * (tl + tr + bl + br)

    direction = direction.lower()
    if direction in ("right_to_left", "left_to_right", "vertical_center", "vertical"):
        # Vertical crease: direction is the average of the two vertical edges
        # (left edge tl->bl and right edge tr->br).
        left_edge = bl - tl
        right_edge = br - tr
        crease_dir = left_edge + right_edge
    elif direction in ("top_to_bottom", "bottom_to_top", "horizontal_center", "horizontal"):
        # Horizontal crease: average of the two horizontal edges.
        top_edge = tr - tl
        bottom_edge = br - bl
        crease_dir = top_edge + bottom_edge
    else:
        raise ValueError(f"unknown fold direction: {direction!r}")

    if np.linalg.norm(crease_dir) < 1e-9:
        raise ValueError("degenerate corners: cannot determine crease direction")
    return center, normalize(crease_dir)


def polygon_area(points: ArrayLike) -> float:
    """Shoelace area of a 2D polygon (absolute value, vertex order independent)."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 3:
        raise ValueError("polygon_area expects (N>=3, 2) points")
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def polygon_perimeter(points: ArrayLike, closed: bool = True) -> float:
    """Perimeter (or path length) of a sequence of 2D points."""
    pts = np.asarray(points, dtype=np.float64)
    diffs = np.diff(pts, axis=0)
    total = float(np.sqrt((diffs**2).sum(axis=1)).sum())
    if closed and len(pts) > 1:
        total += float(np.linalg.norm(pts[0] - pts[-1]))
    return total


def polygon_centroid(points: ArrayLike) -> np.ndarray:
    """Area-weighted centroid of a simple polygon (falls back to vertex mean)."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] < 3:
        return pts.mean(axis=0)
    x = pts[:, 0]
    y = pts[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    area = 0.5 * float(cross.sum())
    if abs(area) < 1e-12:
        return pts.mean(axis=0)
    cx = float(((x + np.roll(x, -1)) * cross).sum()) / (6.0 * area)
    cy = float(((y + np.roll(y, -1)) * cross).sum()) / (6.0 * area)
    return np.array([cx, cy])


def order_corners_clockwise(corners: ArrayLike) -> np.ndarray:
    """Order 4 corners as [top_left, top_right, bottom_right, bottom_left].

    Uses angle about the centroid in image coordinates (y down). Useful for
    normalizing detector output before computing edges / creases.
    """
    pts = np.asarray(corners, dtype=np.float64)
    if pts.shape != (4, 2):
        raise ValueError("order_corners_clockwise expects (4, 2)")
    center = pts.mean(axis=0)
    # Sort by angle (image y-down -> clockwise visually).
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    order = np.argsort(angles)
    pts_sorted = pts[order]
    # Rotate so the first point is the top-left-most (min x+y).
    start = int(np.argmin(pts_sorted.sum(axis=1)))
    pts_sorted = np.roll(pts_sorted, -start, axis=0)
    return pts_sorted


def line_intersection(
    p1: ArrayLike, d1: ArrayLike, p2: ArrayLike, d2: ArrayLike
) -> np.ndarray:
    """Intersection of two 2D lines given as point+direction. Raises if parallel."""
    p1 = as_vector(p1)
    p2 = as_vector(p2)
    d1 = as_vector(d1)
    d2 = as_vector(d2)
    A = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]])
    det = float(np.linalg.det(A))
    if abs(det) < 1e-12:
        raise ValueError("lines are parallel; no unique intersection")
    t = np.linalg.solve(A, p2 - p1)
    return p1 + d1 * t[0]
