"""Interpolation, smoothing, and time-parameterization for trajectories.

Pure numpy. These helpers are used by :mod:`terafold.planning.trajectory` to
build smooth, velocity/acceleration-limited fold motions. Keeping them here
(separate from the planner) means they can be tested in isolation and reused.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

ArrayLike = np.ndarray

__all__ = [
    "lerp",
    "linear_waypoints",
    "catmull_rom",
    "quadratic_bezier",
    "resample_by_arclength",
    "moving_average_smooth",
    "time_parameterize",
    "finite_difference_velocity",
    "finite_difference_acceleration",
    "enforce_velocity_limit",
]


def lerp(a: ArrayLike, b: ArrayLike, t: float) -> np.ndarray:
    """Linear interpolation between ``a`` and ``b`` at fraction ``t`` in [0, 1]."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return (1.0 - t) * a + t * b


def linear_waypoints(start: ArrayLike, end: ArrayLike, num: int) -> np.ndarray:
    """``num`` points evenly spaced from ``start`` to ``end`` (inclusive)."""
    if num < 2:
        raise ValueError("num must be >= 2")
    ts = np.linspace(0.0, 1.0, num)
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    return np.array([lerp(start, end, float(t)) for t in ts])


def quadratic_bezier(p0: ArrayLike, p1: ArrayLike, p2: ArrayLike, num: int) -> np.ndarray:
    """Sample a quadratic Bezier with control points p0 (start), p1 (mid), p2 (end).

    This is the workhorse for the fold *arc*: ``p1`` is lifted above the table
    so the cloth swings over the crease rather than dragging across it.
    """
    if num < 2:
        raise ValueError("num must be >= 2")
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    ts = np.linspace(0.0, 1.0, num)[:, None]
    return (1 - ts) ** 2 * p0 + 2 * (1 - ts) * ts * p1 + ts**2 * p2


def catmull_rom(points: ArrayLike, samples_per_segment: int = 10) -> np.ndarray:
    """Catmull-Rom spline through ``points`` (C1-continuous, passes through pts)."""
    pts = np.asarray(points, dtype=np.float64)
    n = pts.shape[0]
    if n < 2:
        raise ValueError("need at least 2 control points")
    if n == 2:
        return linear_waypoints(pts[0], pts[1], samples_per_segment)
    # Pad endpoints so the spline reaches the first/last point.
    padded = np.vstack([pts[0], pts, pts[-1]])
    out = []
    ts = np.linspace(0.0, 1.0, samples_per_segment, endpoint=False)
    for i in range(1, len(padded) - 2):
        p0, p1, p2, p3 = padded[i - 1], padded[i], padded[i + 1], padded[i + 2]
        for t in ts:
            t2 = t * t
            t3 = t2 * t
            out.append(
                0.5
                * (
                    (2 * p1)
                    + (-p0 + p2) * t
                    + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                    + (-p0 + 3 * p1 - 3 * p2 + p3) * t3
                )
            )
    out.append(pts[-1])
    return np.asarray(out)


def resample_by_arclength(points: ArrayLike, num: int) -> np.ndarray:
    """Resample a polyline to ``num`` points evenly spaced by arc length."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] < 2:
        return pts
    seg = np.sqrt((np.diff(pts, axis=0) ** 2).sum(axis=1))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    if total < 1e-12:
        return np.repeat(pts[:1], num, axis=0)
    target = np.linspace(0.0, total, num)
    out = np.empty((num, pts.shape[1]))
    for dim in range(pts.shape[1]):
        out[:, dim] = np.interp(target, s, pts[:, dim])
    return out


def moving_average_smooth(points: ArrayLike, window: int = 3) -> np.ndarray:
    """Smooth a path with a centered moving average, preserving endpoints."""
    pts = np.asarray(points, dtype=np.float64)
    if window < 2 or pts.shape[0] <= window:
        return pts.copy()
    if window % 2 == 0:
        window += 1
    half = window // 2
    out = pts.copy()
    for i in range(half, pts.shape[0] - half):
        out[i] = pts[i - half : i + half + 1].mean(axis=0)
    return out


def time_parameterize(
    waypoints: ArrayLike, max_speed: float, max_accel: float
) -> np.ndarray:
    """Assign timestamps to waypoints under speed/accel limits (trapezoidal-ish).

    Returns a 1D array of cumulative times (seconds), monotonically increasing,
    starting at 0. The scheme caps cruise speed at ``max_speed`` and ramps the
    *effective* speed between consecutive segments so the implied acceleration
    stays under ``max_accel``. It is intentionally conservative (slow + safe).
    """
    pts = np.asarray(waypoints, dtype=np.float64)
    n = pts.shape[0]
    if n < 2:
        return np.zeros(n)
    if max_speed <= 0:
        raise ValueError("max_speed must be > 0")
    if max_accel <= 0:
        raise ValueError("max_accel must be > 0")

    seg_len = np.sqrt((np.diff(pts, axis=0) ** 2).sum(axis=1))
    seg_len = np.maximum(seg_len, 1e-9)

    # Forward pass: speed limited by acceleration from a standstill start.
    v = np.zeros(n)
    for i in range(1, n):
        v[i] = min(max_speed, np.sqrt(v[i - 1] ** 2 + 2 * max_accel * seg_len[i - 1]))
    # Backward pass: ensure we can decelerate to a stop at the end.
    v[-1] = 0.0
    for i in range(n - 2, -1, -1):
        v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2 * max_accel * seg_len[i]))

    times = np.zeros(n)
    for i in range(1, n):
        v_avg = max(0.5 * (v[i - 1] + v[i]), 1e-6)
        times[i] = times[i - 1] + seg_len[i - 1] / v_avg
    return times


def finite_difference_velocity(positions: ArrayLike, times: ArrayLike) -> np.ndarray:
    """Central finite-difference velocity for a position/time sequence."""
    pos = np.asarray(positions, dtype=np.float64)
    t = np.asarray(times, dtype=np.float64)
    return np.gradient(pos, t, axis=0)


def finite_difference_acceleration(positions: ArrayLike, times: ArrayLike) -> np.ndarray:
    """Central finite-difference acceleration for a position/time sequence."""
    vel = finite_difference_velocity(positions, times)
    t = np.asarray(times, dtype=np.float64)
    return np.gradient(vel, t, axis=0)


def enforce_velocity_limit(
    positions: ArrayLike, times: ArrayLike, max_speed: float
) -> Tuple[np.ndarray, bool]:
    """Stretch ``times`` (uniformly per-segment) so no segment exceeds ``max_speed``.

    Returns ``(new_times, modified)``. Positions are unchanged; only the time
    parameterization is slowed where needed.
    """
    pos = np.asarray(positions, dtype=np.float64)
    t = np.asarray(times, dtype=np.float64).copy()
    modified = False
    for i in range(1, len(t)):
        dist = float(np.linalg.norm(pos[i] - pos[i - 1]))
        dt = t[i] - t[i - 1]
        if dt <= 0:
            continue
        speed = dist / dt
        if speed > max_speed:
            needed = dist / max_speed
            shift = needed - dt
            t[i:] += shift
            modified = True
    return t, modified
