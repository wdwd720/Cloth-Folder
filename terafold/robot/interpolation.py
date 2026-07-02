"""Joint-space interpolation in RAW servo units (no kinematics required).

These helpers turn a start/goal pose (per servo, raw 0..4095 counts) into a
sequence of intermediate targets so a move is split into small, smooth steps
rather than a single jump. They are intentionally tiny, pure-stdlib, and
side-effect free: they NEVER touch a backend or move hardware — a caller feeds
the resulting targets through :mod:`terafold.robot.safety_gate` and
:mod:`terafold.robot.motion_core`.

Inclusivity convention (consistent across every function here)
--------------------------------------------------------------
All three functions return a list of length ``steps + 1`` that **includes both
the start and the goal**. ``steps`` is the number of *segments*; with
``steps=1`` you get exactly ``[start, goal]``. The first and last samples are
forced to the exact endpoints (no rounding drift).

Separate from :mod:`terafold.math.interpolation`, which interpolates Cartesian
``(x, y, z)`` waypoints; this module works on per-servo integer counts.
"""

from __future__ import annotations

from typing import Dict, List

__all__ = [
    "interpolate_raw_joint_targets",
    "minimum_jerk_interpolation",
    "trapezoidal_profile",
]


def _check_steps(steps: int) -> int:
    steps = int(steps)
    if steps < 1:
        raise ValueError(f"steps must be >= 1 (got {steps}); a move needs at least one segment.")
    return steps


def interpolate_raw_joint_targets(
    start: Dict[int, int], goal: Dict[int, int], steps: int
) -> List[Dict[int, int]]:
    """Linearly interpolate raw servo targets, per servo id, in integer counts.

    Each returned frame maps ``servo_id -> raw_units`` (rounded to the nearest
    integer). The result has ``steps + 1`` frames and is inclusive of both
    ``start`` and ``goal`` (the endpoints are exact, not rounded).

    Servo ids are the union of both dicts; an id missing from one side is held
    at the value it has on the other side (so it never moves spuriously).
    """
    steps = _check_steps(steps)
    s = {int(k): int(v) for k, v in start.items()}
    g = {int(k): int(v) for k, v in goal.items()}
    ids = sorted(set(s) | set(g))

    def _start(sid: int) -> int:
        return s[sid] if sid in s else g[sid]

    def _goal(sid: int) -> int:
        return g[sid] if sid in g else s[sid]

    frames: List[Dict[int, int]] = []
    for i in range(steps + 1):
        t = i / steps
        frames.append({sid: int(round(_start(sid) + (_goal(sid) - _start(sid)) * t)) for sid in ids})
    # Force exact endpoints (avoid any banker's-rounding drift).
    frames[0] = {sid: _start(sid) for sid in ids}
    frames[-1] = {sid: _goal(sid) for sid in ids}
    return frames


def minimum_jerk_interpolation(start: float, goal: float, steps: int) -> List[float]:
    """Minimum-jerk scalar profile using ``s(t) = 10t^3 - 15t^4 + 6t^5``.

    Returns ``steps + 1`` samples inclusive of both endpoints. The profile is
    monotonic (``s'(t) = 30 t^2 (1 - t)^2 >= 0``) so for ``goal > start`` the
    output is monotonically non-decreasing, with zero velocity at both ends.
    """
    steps = _check_steps(steps)
    start = float(start)
    goal = float(goal)
    delta = goal - start
    out: List[float] = []
    for i in range(steps + 1):
        t = i / steps
        s = 10.0 * t**3 - 15.0 * t**4 + 6.0 * t**5
        out.append(start + delta * s)
    out[0] = start
    out[-1] = goal
    return out


def _trapezoid_unit(t: float, a: float) -> float:
    """Normalized trapezoidal *position* in [0, 1] for time ``t`` in [0, 1].

    ``a`` is the fraction of the move spent accelerating (and, symmetrically,
    decelerating). ``a == 0`` degenerates to constant-velocity (linear);
    ``a == 0.5`` is a triangular profile.
    """
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0
    if a <= 0.0:
        return t  # no accel phase -> constant velocity
    vmax = 1.0 / (1.0 - a)  # area under the trapezoid must equal 1
    if t < a:
        return vmax * t * t / (2.0 * a)
    if t <= 1.0 - a:
        return vmax * (t - a / 2.0)
    return 1.0 - vmax * (1.0 - t) ** 2 / (2.0 * a)


def trapezoidal_profile(
    start: float, goal: float, steps: int, accel_frac: float = 0.25
) -> List[float]:
    """Trapezoidal-velocity scalar profile (ramp up, cruise, ramp down).

    Returns ``steps + 1`` samples inclusive of both endpoints. ``accel_frac`` is
    the fraction of the move spent accelerating; it is clamped to ``[0, 0.5]``
    (``0`` -> linear, ``0.5`` -> triangular). The implied velocity starts and
    ends at zero, and the position is monotonic for ``goal > start``.
    """
    steps = _check_steps(steps)
    a = max(0.0, min(float(accel_frac), 0.5))
    start = float(start)
    goal = float(goal)
    delta = goal - start
    out = [start + delta * _trapezoid_unit(i / steps, a) for i in range(steps + 1)]
    out[0] = start
    out[-1] = goal
    return out
