"""Pure safety clamps and shadow-mode gates for the YAM/MolmoAct2 stack.

Everything here is a pure function over plain Python numbers/sequences: no I/O,
no robot SDK, no numpy. The module encodes two invariants for this sprint:

1. **Shadow mode only.** :func:`assert_shadow_mode` refuses any config with
   ``autonomous_execution_enabled`` set, and :func:`assert_no_hardware_execution`
   refuses any *attempt* to command hardware.
2. **Future execution stays double-gated.** :data:`HARDWARE_EXECUTION_IMPLEMENTED`
   is a module constant (not config!) that is ``False`` this sprint. Even when a
   future sprint flips it, :func:`hardware_execution_allowed` still requires BOTH
   dangerous flags (``--enable-yam-motion`` and
   ``--i-understand-this-moves-the-yam``), mirroring the old cheap-arm double
   gate — which this module deliberately does not touch.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

__all__ = [
    "ShadowModeViolation",
    "HardwareExecutionForbidden",
    "HARDWARE_EXECUTION_IMPLEMENTED",
    "clamp_joint_delta",
    "clamp_gripper_delta",
    "validate_workspace_bounds",
    "assert_shadow_mode",
    "assert_no_hardware_execution",
    "hardware_execution_allowed",
]

#: Real YAM execution is NOT implemented this sprint. A future sprint may flip
#: this constant in code (never via config); the double dangerous-flag gate in
#: :func:`hardware_execution_allowed` still applies after that.
HARDWARE_EXECUTION_IMPLEMENTED = False


class ShadowModeViolation(RuntimeError):
    """Raised when something tries to leave shadow mode."""


class HardwareExecutionForbidden(ShadowModeViolation):
    """Raised when something tries to command YAM hardware."""


def clamp_joint_delta(
    current: Sequence[float], target: Sequence[float], max_delta_rad: float
) -> List[float]:
    """Clamp each target joint so it moves at most ``max_delta_rad`` from current.

    ``current`` and ``target`` must have the same length. Returns the clamped
    target list; inputs are never mutated.
    """
    if max_delta_rad < 0:
        raise ValueError(f"max_delta_rad must be >= 0, got {max_delta_rad}")
    cur = [float(v) for v in current]
    tgt = [float(v) for v in target]
    if len(cur) != len(tgt):
        raise ValueError(f"current ({len(cur)}) and target ({len(tgt)}) length mismatch")
    out: List[float] = []
    for c, t in zip(cur, tgt):
        delta = t - c
        if delta > max_delta_rad:
            delta = max_delta_rad
        elif delta < -max_delta_rad:
            delta = -max_delta_rad
        out.append(c + delta)
    return out


def clamp_gripper_delta(current: float, target: float, max_delta: float) -> float:
    """Clamp a scalar gripper target to at most ``max_delta`` away from current."""
    if max_delta < 0:
        raise ValueError(f"max_delta must be >= 0, got {max_delta}")
    c, t = float(current), float(target)
    delta = max(-max_delta, min(max_delta, t - c))
    return c + delta


def validate_workspace_bounds(
    point_xyz: Sequence[float], bounds: Dict[str, Sequence[float]]
) -> List[str]:
    """Check an ``(x, y, z)`` point against ``{x: [lo, hi], y: ..., z: ...}`` bounds.

    Returns a list of human-readable violations (empty means inside bounds).
    Axes missing from ``bounds`` are not checked.
    """
    if len(point_xyz) != 3:
        return [f"point must be (x, y, z), got {len(point_xyz)} values"]
    violations: List[str] = []
    for axis, value in zip("xyz", point_xyz):
        rng = bounds.get(axis)
        if rng is None:
            continue
        lo, hi = float(rng[0]), float(rng[1])
        v = float(value)
        if not (lo <= v <= hi):
            violations.append(f"{axis}={v:.4f} outside workspace bounds [{lo}, {hi}]")
    return violations


def _autonomous_flag(config: Any) -> bool:
    if isinstance(config, dict):
        return bool(config.get("autonomous_execution_enabled", False))
    return bool(getattr(config, "autonomous_execution_enabled", False))


def assert_shadow_mode(config: Any) -> None:
    """Raise :class:`ShadowModeViolation` unless the config keeps autonomy off.

    Accepts a :class:`~terafold.yam.config.YamDualConfig` or a plain dict.
    """
    if _autonomous_flag(config):
        raise ShadowModeViolation(
            "autonomous_execution_enabled must be false: this sprint is shadow "
            "mode only (no autonomous YAM motion, no hardware commands)."
        )


def hardware_execution_allowed(
    enable_yam_motion: bool = False, i_understand_this_moves_the_yam: bool = False
) -> bool:
    """Whether commanding YAM hardware would be permitted.

    Always ``False`` this sprint (:data:`HARDWARE_EXECUTION_IMPLEMENTED` is
    ``False``). Even in a future execution sprint, BOTH dangerous flags are
    required.
    """
    return bool(
        HARDWARE_EXECUTION_IMPLEMENTED
        and enable_yam_motion
        and i_understand_this_moves_the_yam
    )


def assert_no_hardware_execution(
    enable_yam_motion: bool = False,
    i_understand_this_moves_the_yam: bool = False,
    context: str = "",
) -> None:
    """Refuse any attempt to command YAM hardware.

    Call this at the top of every function that could ever grow an execution
    path. With no flags set it is a no-op (pure shadow work is fine). If a
    caller asks for execution it raises :class:`HardwareExecutionForbidden`
    unless :func:`hardware_execution_allowed` grants it — which it never does
    this sprint.
    """
    wants_execution = bool(enable_yam_motion or i_understand_this_moves_the_yam)
    if not wants_execution:
        return
    if hardware_execution_allowed(enable_yam_motion, i_understand_this_moves_the_yam):
        return  # future sprint, both flags — permitted there, unreachable now
    where = f" ({context})" if context else ""
    if not HARDWARE_EXECUTION_IMPLEMENTED:
        raise HardwareExecutionForbidden(
            f"YAM hardware execution is not implemented in this sprint{where}; "
            "shadow mode only — no commands are ever sent to the arms."
        )
    raise HardwareExecutionForbidden(
        f"YAM hardware execution requires BOTH --enable-yam-motion and "
        f"--i-understand-this-moves-the-yam{where}."
    )
