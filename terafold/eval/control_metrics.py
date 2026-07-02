"""Low-level servo control metrics from readback traces (numpy-only).

A *readback trace* is the time series a backend records while (or after) it
commands a servo: a sequence of ``(t, position, speed)`` samples in raw servo
units. Given a trace and a commanded ``target`` these helpers quantify how well
the servo tracked it — settling error, overshoot, time-to-settle, whether it
ever reached the target — plus characterization quantities (readback noise,
deadband, backlash) used by the servo-mapping/characterization stage.

Everything is unit-agnostic: positions and the target are in the same raw units,
times in seconds. Pure numpy; no hardware, no optional deps.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "settle_error",
    "overshoot",
    "time_to_settle",
    "failure_to_reach",
    "readback_noise",
    "deadband_estimate",
    "backlash",
    "unpack_trace",
]


def unpack_trace(trace: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize a readback trace into ``(t, pos, speed)`` arrays.

    Each sample may be a ``(t, pos, speed)`` / ``(t, pos)`` tuple, or a mapping
    with keys among ``{t, time, timestamp}``, ``{pos, position, present_pos}``
    and ``{speed, present_speed, vel}``. Missing times are filled with the sample
    index; missing speeds with NaN.
    """
    ts: List[float] = []
    ps: List[float] = []
    sp: List[float] = []
    for i, s in enumerate(trace or []):
        if isinstance(s, dict):
            t = s.get("t", s.get("time", s.get("timestamp")))
            pos = s.get("pos", s.get("position", s.get("present_pos")))
            spd = s.get("speed", s.get("present_speed", s.get("vel")))
        else:
            seq = list(s)
            if len(seq) >= 3:
                t, pos, spd = seq[0], seq[1], seq[2]
            elif len(seq) == 2:
                t, pos, spd = seq[0], seq[1], None
            else:
                t, pos, spd = None, seq[0], None
        ts.append(float(i) if t is None else float(t))
        ps.append(np.nan if pos is None else float(pos))
        sp.append(np.nan if spd is None else float(spd))
    return (np.asarray(ts, dtype=np.float64),
            np.asarray(ps, dtype=np.float64),
            np.asarray(sp, dtype=np.float64))


def _final_pos(t: np.ndarray, pos: np.ndarray, speed: np.ndarray,
               speed_tol: float = 5.0) -> float:
    """The settled final position: mean over the trailing at-rest samples.

    Falls back to the last valid position when no speed information is present.
    """
    valid = pos[np.isfinite(pos)]
    if valid.size == 0:
        return float("nan")
    if np.any(np.isfinite(speed)):
        rest = pos[np.isfinite(speed) & (np.abs(speed) <= float(speed_tol)) & np.isfinite(pos)]
        # Use the rest samples only if they include the tail of the motion.
        if rest.size:
            return float(rest[-min(rest.size, 3):].mean())
    return float(valid[-1])


def settle_error(trace: Any, target: float, speed_tol: float = 5.0) -> float:
    """Absolute error between the settled final position and ``target``."""
    t, pos, speed = unpack_trace(trace)
    return float(abs(_final_pos(t, pos, speed, speed_tol) - float(target)))


def overshoot(trace: Any, target: float, start: Optional[float] = None) -> float:
    """Peak excursion past ``target`` in the direction of travel (>= 0).

    The travel direction is inferred from the first sample (or ``start`` if
    given) toward ``target``. Overshoot is how far the position went beyond the
    target on that side; zero if it never crossed.
    """
    _, pos, _ = unpack_trace(trace)
    pos = pos[np.isfinite(pos)]
    if pos.size == 0:
        return 0.0
    start_pos = float(pos[0]) if start is None else float(start)
    tgt = float(target)
    if tgt > start_pos:
        return float(max(0.0, pos.max() - tgt))
    if tgt < start_pos:
        return float(max(0.0, tgt - pos.min()))
    return float(np.abs(pos - tgt).max())


def time_to_settle(trace: Any, target: float, tol: float) -> float:
    """Elapsed time until the position stays within ``tol`` of ``target``.

    Returns the time from the first sample to the earliest sample after which
    *every* subsequent position is within ``tol``. ``inf`` if it never settles.
    """
    t, pos, _ = unpack_trace(trace)
    mask = np.isfinite(pos)
    t, pos = t[mask], pos[mask]
    if pos.size == 0:
        return float("inf")
    within = np.abs(pos - float(target)) <= float(tol)
    settled_from = None
    for i in range(len(pos)):
        if np.all(within[i:]):
            settled_from = i
            break
    if settled_from is None:
        return float("inf")
    return float(t[settled_from] - t[0])


def failure_to_reach(trace: Any, target: float, tol: float) -> bool:
    """True iff the servo never settled within ``tol`` of ``target``."""
    return bool(settle_error(trace, target) > float(tol))


def readback_noise(trace: Any, speed_tol: float = 5.0) -> float:
    """Standard deviation of position while the servo is at rest.

    Uses samples whose speed is within ``speed_tol`` of zero; if no speed info is
    available, uses the trailing samples. Returns 0.0 for a constant readback.
    """
    _, pos, speed = unpack_trace(trace)
    pos = pos.copy()
    if np.any(np.isfinite(speed)):
        rest = pos[np.isfinite(speed) & (np.abs(speed) <= float(speed_tol)) & np.isfinite(pos)]
    else:
        rest = pos[np.isfinite(pos)]
        rest = rest[-min(rest.size, 5):]
    if rest.size < 2:
        return 0.0
    return float(np.std(rest))


def deadband_estimate(deltas: Sequence[float], moved: Sequence[Any]) -> float:
    """Estimate the command deadband from probe deltas and move outcomes.

    ``deltas`` are commanded magnitudes; ``moved`` is a parallel sequence of
    booleans (did the servo actually move?). The deadband estimate is the
    largest commanded delta that produced *no* motion. Returns 0.0 if every
    delta moved the servo.
    """
    d = np.abs(np.asarray(deltas, dtype=np.float64).reshape(-1))
    m = np.asarray([bool(x) for x in moved], dtype=bool).reshape(-1)
    n = min(d.size, m.size)
    if n == 0:
        return 0.0
    d, m = d[:n], m[:n]
    no_move = d[~m]
    if no_move.size == 0:
        return 0.0
    return float(no_move.max())


def backlash(approach_pos: Any, approach_neg: Any) -> float:
    """Backlash: |final position approaching a target from each direction|.

    Each argument is either a scalar final position or a full readback trace
    (the settled final position is extracted from it).
    """
    return float(abs(_coerce_pos(approach_pos) - _coerce_pos(approach_neg)))


def _coerce_pos(x: Any) -> float:
    """A scalar position, or the settled final position of a trace."""
    try:
        return float(x)
    except (TypeError, ValueError):
        t, pos, speed = unpack_trace(x)
        return _final_pos(t, pos, speed)
