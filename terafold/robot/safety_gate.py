"""The single safety gate every joint-space motion request must pass through.

This centralizes the checks that were previously scattered across the real-motion
helpers and the CLI: dry-run vs real, the two-flag human gate, raw-unit limits,
the active/disabled servo set, and the calibration/joint-map/contact preconditions
(consulted against a :class:`~terafold.robot.capabilities.RobotCapabilities`
snapshot when one is supplied).

The gate is **pure and side-effect free** — it only *decides*; it never opens a
port or writes a servo. A ``True`` result means "this request is permitted as
specified"; for a dry-run that simply means the simulated/preview run may proceed
(``checks["real_motion"]`` stays ``False``, so no write is ever implied).

Pure stdlib. No numpy/serial/torch import at module load.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

__all__ = [
    "MotionRequest",
    "MotionGate",
    "validate_targets_against_limits",
]


@dataclass
class MotionRequest:
    """A request to move one or more servos to raw-unit targets.

    Dry-run is the default. Real motion additionally requires BOTH
    ``enable_motion`` and ``acknowledge`` (the ``--enable-motion`` and
    ``--i-understand-this-moves-hardware`` flags). ``contact`` is for any request
    that would touch the cloth/table; it is gated extremely hard (see
    :class:`MotionGate`).
    """

    robot: str
    servo_targets: Dict[int, int]
    speed: int
    acc: int
    dry_run: bool = True
    enable_motion: bool = False
    acknowledge: bool = False
    requires_joint_map: bool = False
    requires_calibration: bool = False
    contact: bool = False
    safe_units: Tuple[int, int] = (400, 3700)
    active_ids: Sequence[int] = field(default_factory=tuple)


def validate_targets_against_limits(
    targets: Dict[int, int], safe_units: Tuple[int, int]
) -> List[str]:
    """Return a list of human-readable violations (empty if every target is safe).

    A target is a violation when it falls outside the inclusive ``[lo, hi]`` raw
    range. Non-integer / unparseable targets are reported as violations too.
    """
    lo, hi = int(safe_units[0]), int(safe_units[1])
    violations: List[str] = []
    for sid, val in targets.items():
        try:
            v = int(val)
        except (TypeError, ValueError):
            violations.append(f"servo {sid} target {val!r} is not an integer raw count")
            continue
        if not (lo <= v <= hi):
            violations.append(f"servo {sid} target {v} outside safe raw range [{lo}, {hi}]")
    return violations


class MotionGate:
    """Evaluate a :class:`MotionRequest` and report whether it is allowed and why not.

    The checks run in a fixed order of importance:

    1. dry-run vs real (dry-run is allowed and flagged non-real);
    2. real motion requires BOTH ``enable_motion`` and ``acknowledge``;
    3. every target is within the safe raw limits;
    4. only active servo ids are commanded (disabled ids are never moved);
    5. if a joint map is required ``capabilities.joint_map_present`` must hold;
    6. if calibration is required ``capabilities.calibration_valid_for_hover``
       must hold;
    7. contact is **always refused** unless ``capabilities.contact_allowed``
       (which is essentially never granted here — contact folding stays LOCKED).

    When ``capabilities`` is ``None`` only the flag/limit/id checks (1–4) and the
    contact refusal run; a note records that protocol/joint-map/calibration were
    not verified.
    """

    def check(self, request: MotionRequest, capabilities: Any = None) -> Dict[str, Any]:
        refusals: List[str] = []
        checks: Dict[str, Any] = {}

        # 1. dry-run vs real -------------------------------------------------
        is_real = not bool(request.dry_run)
        checks["dry_run"] = bool(request.dry_run)
        checks["real_motion"] = is_real

        # 2. two-flag human gate (real motion only) --------------------------
        if is_real:
            both = bool(request.enable_motion and request.acknowledge)
            checks["both_flags"] = both
            if not both:
                refusals.append(
                    "real motion requires BOTH enable_motion=True and acknowledge=True "
                    "(--enable-motion and --i-understand-this-moves-hardware); defaulting to dry-run."
                )
        else:
            checks["both_flags"] = None  # not required while dry-running

        # 3. raw-unit limits -------------------------------------------------
        limit_violations = validate_targets_against_limits(request.servo_targets, request.safe_units)
        checks["targets_within_limits"] = not limit_violations
        refusals.extend(limit_violations)

        # 4. only active ids (disabled ids never move) -----------------------
        active = {int(i) for i in request.active_ids}
        if active:
            bad_ids = sorted(int(sid) for sid in request.servo_targets if int(sid) not in active)
            checks["only_active_ids"] = not bad_ids
            for sid in bad_ids:
                refusals.append(
                    f"servo {sid} is not in the active set {sorted(active)} "
                    "(disabled/unconfirmed ids are never moved)"
                )
        else:
            checks["only_active_ids"] = None  # no active set supplied to restrict against

        # 5/6. capability-dependent preconditions ----------------------------
        if capabilities is None:
            checks["capabilities_verified"] = False
        else:
            checks["capabilities_verified"] = True
            if request.requires_joint_map:
                jm = bool(getattr(capabilities, "joint_map_present", False))
                checks["joint_map_present"] = jm
                if not jm:
                    refusals.append(
                        "a joint map is required but capabilities.joint_map_present is False "
                        "(run `map-servo-joints` first)"
                    )
            if request.requires_calibration:
                cal = bool(getattr(capabilities, "calibration_valid_for_hover", False))
                checks["calibration_valid_for_hover"] = cal
                if not cal:
                    refusals.append(
                        "valid hover calibration is required but "
                        "capabilities.calibration_valid_for_hover is False"
                    )

        # 7. contact ALWAYS refused unless explicitly unlocked ---------------
        if request.contact:
            contact_ok = bool(getattr(capabilities, "contact_allowed", False)) if capabilities else False
            checks["contact_allowed"] = contact_ok
            if not contact_ok:
                refusals.append(
                    "contact folding is LOCKED (capabilities.contact_allowed is False); "
                    "contact is refused"
                )

        result: Dict[str, Any] = {
            "allowed": not refusals,
            "refusals": refusals,
            "checks": checks,
        }
        if capabilities is None:
            result["note"] = (
                "capabilities not supplied: protocol/joint-map/calibration were NOT verified "
                "(only the flag, raw-limit, active-id, and contact checks ran)."
            )
        return result
