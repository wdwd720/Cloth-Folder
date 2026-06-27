"""Real-hardware safety helpers: joint-space teach/replay, ghost fold, gates.

Everything here is conservative and dry-run friendly. No serial protocol is
invented; the actual sending of motion lives behind the adapter's
``protocol_confirmed`` gate and the two-flag :func:`require_motion_enabled` check.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

from terafold.robot.safety import SafetyError

__all__ = [
    "JointDemo", "save_joint_demo", "load_joint_demo",
    "GHOST_FOLD_POSES", "record_teach_demo",
    "ghost_fold_trajectory", "ghost_fold_core", "joint_space_ghost_fold",
    "image_real_motion_gate",
    "IMAGE_CALIBRATION_REFUSAL", "estop_instructions", "print_estop_instructions",
    "countdown", "check_nudge_delta", "motion_logger", "SPEED_PRESETS",
]

IMAGE_CALIBRATION_REFUSAL = (
    "Image-based real motion requires table calibration. Use real-ghost-fold "
    "above the table or calibrate-table first."
)

SPEED_PRESETS = {"very_slow": 0.02, "slow": 0.04}  # m/s caps (informational)

# The teachable joint-space ghost-fold waypoints (no kinematics required).
GHOST_FOLD_POSES = [
    "home",
    "hover_grasp",
    "lower_near_grasp",   # NEAR the grasp but NOT touching the cloth
    "lift",
    "arc_over_crease",
    "hover_place",
    "release",
    "home_return",
]


# ----------------------------------------------------------------------
# Joint-space teach / replay demo
# ----------------------------------------------------------------------


@dataclass
class JointDemo:
    robot: str
    dof: int
    joint_names: List[str] = field(default_factory=list)
    waypoints: List[Dict[str, Any]] = field(default_factory=list)
    protocol_confirmed: bool = False
    note: str = ""
    created: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "JointDemo":
        known = {f for f in cls.__dataclass_fields__}  # noqa: F821
        return cls(**{k: v for k, v in d.items() if k in known})

    def has_positions(self) -> bool:
        return any(w.get("servo_positions") for w in self.waypoints)


def save_joint_demo(path: str, demo: JointDemo) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(demo.to_dict(), f, indent=2)
    return path


def load_joint_demo(path: str) -> JointDemo:
    with open(path) as f:
        return JointDemo.from_dict(json.load(f))


def record_teach_demo(
    adapter,
    robot: str,
    dof: int,
    joint_names: List[str],
    poses: Optional[List[str]] = None,
    input_fn: Callable[[str], str] = input,
    log: Callable[[str], None] = print,
    now: float = 0.0,
) -> JointDemo:
    """Walk the user through posing the arm and record servo positions per pose.

    The arm is moved BY HAND (torque off / loose); for each pose the user presses
    Enter and the current positions are read. If the protocol is unconfirmed the
    positions cannot be read, so they are stored as ``None`` and the demo is
    flagged — it cannot be replayed until a confirmed protocol is wired.
    """
    poses = poses or GHOST_FOLD_POSES
    waypoints: List[Dict[str, Any]] = []
    can_read = adapter.protocol_confirmed
    if not can_read:
        log("[warn] Servo positions cannot be read (protocol not confirmed); "
            "recording a SKELETON demo with null positions. It cannot be replayed "
            "until a verified protocol/backend is wired.")
    for name in poses:
        input_fn(f"  Pose the arm by hand at '{name}', then press Enter to record... ")
        positions = None
        if can_read:
            r = adapter.read_positions()
            positions = r.get("positions") if r.get("supported") else None
        waypoints.append({"name": name, "servo_positions": positions, "gripper": None})
        log(f"   recorded '{name}': "
            f"{'positions captured' if positions else 'no positions (protocol unconfirmed)'}")
    return JointDemo(
        robot=robot, dof=dof, joint_names=list(joint_names), waypoints=waypoints,
        protocol_confirmed=can_read,
        note=("teachable joint-space ghost fold (above the table; do not touch the cloth)"),
        created=float(now),
    )


# ----------------------------------------------------------------------
# Ghost fold trajectory (above the table, never touching the cloth)
# ----------------------------------------------------------------------


def _load_plan(path: str) -> Dict[str, Any]:
    with open(path) as f:
        d = json.load(f)
    if isinstance(d, dict):
        if "trajectory" in d:
            return d
        for k in ("plan", "fold_plan"):
            if isinstance(d.get(k), dict):
                return d[k]
    return d


def ghost_fold_core(wp, grip, phases, height_clearance_m=0.10, table_z=0.0,
                    max_gripper_close=0.5) -> Dict[str, Any]:
    """Lift a trajectory into the AIR above the table; gripper never closes hard."""
    floor = table_z + float(height_clearance_m)
    out = []
    min_z = float("inf")
    grip = grip or [1.0] * len(wp)
    phases = phases or [""] * len(wp)
    for i, p in enumerate(wp):
        x, y, z = float(p[0]), float(p[1]), float(p[2])
        lz = max(z, floor)
        min_z = min(min_z, lz)
        g = max(float(grip[i]) if i < len(grip) else 1.0, float(max_gripper_close))
        out.append({"phase": phases[i] if i < len(phases) else "",
                    "xyz": [round(x, 4), round(y, 4), round(lz, 4)],
                    "gripper": round(g, 3)})
    return {
        "table_z": table_z,
        "height_clearance_m": float(height_clearance_m),
        "min_z": (round(min_z, 4) if out else None),
        "touches_table": bool(out and min_z < floor - 1e-9),
        "num_waypoints": len(out),
        "waypoints": out,
        "note": ("AIR-ONLY ghost fold: lifted >= clearance above the table; gripper "
                 "kept open (no hard close). Cartesian preview — needs kinematics or "
                 "a taught/mapped joint demo to execute."),
    }


def ghost_fold_trajectory(plan_json: str, height_clearance_m: float = 0.10,
                          table_z: float = 0.0, max_gripper_close: float = 0.5) -> Dict[str, Any]:
    """Cartesian air-only ghost fold loaded from a plan JSON file."""
    plan = _load_plan(plan_json)
    traj = plan.get("trajectory") or {}
    return ghost_fold_core(traj.get("waypoints") or [], traj.get("gripper"),
                           traj.get("phases"), height_clearance_m, table_z, max_gripper_close)


# Fold direction -> base-yaw sweep sign (the moving half travels toward the crease).
_DIR_SWEEP = {"right_to_left": -1, "left_to_right": +1, "top_to_bottom": -1, "bottom_to_top": +1}


def joint_space_ghost_fold(fold_direction, joint_map, current_positions,
                           max_sweep_units: int = 80, safe_units=(400, 3700),
                           speed: int = 300, acc: int = 20) -> Dict[str, Any]:
    """A CONSERVATIVE joint-space ghost fold DERIVED from the plan + joint map.

    Without kinematics we cannot do Cartesian IK, so the ghost is a small, bounded
    **base-yaw sweep** that traces the fold direction horizontally (no vertical
    motion → stays at the arm's current height). Targets are relative to the read
    current positions (never hardcoded), clamped to the safe range. Only the
    base_yaw servo is commanded; the rest hold position.
    """
    base_id = joint_map.id_for_joint("base_yaw")
    if base_id is None:
        return {"ok": False, "needed_joint": "base_yaw",
                "refusal": "joint 'base_yaw' is not mapped; run `map-servo-joints` first."}
    cur = current_positions.get(base_id)
    if cur is None:
        return {"ok": False, "refusal": f"cannot read base_yaw (servo {base_id}) current position."}
    sign = int(joint_map.sign_for_id(base_id)) * _DIR_SWEEP.get(fold_direction, -1)
    sweep = int(round(max_sweep_units)) * sign
    lo, hi = int(safe_units[0]), int(safe_units[1])
    steps = [("hover_start", int(cur)),
             ("sweep_toward_place", int(cur) + sweep),
             ("return_home", int(cur))]
    bad = {k: v for k, v in steps if not (lo <= v <= hi)}
    if bad:
        return {"ok": False, "refusal": f"sweep targets {bad} outside safe range [{lo}, {hi}]."}
    waypoints = [{"label": k, "servo_id": int(base_id), "target_units": int(v),
                  "speed": int(speed), "acc": int(acc)} for k, v in steps]
    return {"ok": True, "base_id": int(base_id), "sweep_units": int(sweep),
            "waypoints": waypoints,
            "note": ("horizontal base-yaw sweep tracing the fold direction; no vertical "
                     "motion (stays at the current height). Only base_yaw is commanded.")}


# ----------------------------------------------------------------------
# Gates / safety helpers
# ----------------------------------------------------------------------


def image_real_motion_gate(
    *,
    homography_calibration: Optional[str] = None,
    table_bounds: Any = None,
    robot_base_transform: Any = None,
    dry_run_done: bool = False,
    ghost_fold_done: bool = False,
    estop_confirmed: bool = False,
    conservative_speed: bool = False,
    enable_motion: bool = False,
    acknowledge: bool = False,
) -> Dict[str, Any]:
    """Gate image/table-based REAL motion. Returns ``{allowed, missing, message}``.

    Refuses (with :data:`IMAGE_CALIBRATION_REFUSAL`) until the full calibration +
    safety set is present. The image-to-table mapping is never assumed correct.
    """
    missing: List[str] = []
    if not (homography_calibration and os.path.exists(str(homography_calibration))):
        missing.append("a homography calibration file (calibrate-table-from-image)")
    if table_bounds is None:
        missing.append("table bounds")
    if robot_base_transform is None:
        missing.append("a robot base transform (table->robot)")
    if not dry_run_done:
        missing.append("a successful dry-run")
    if not ghost_fold_done:
        missing.append("a successful ghost-fold above the table")
    if not estop_confirmed:
        missing.append("emergency stop confirmed working")
    if not conservative_speed:
        missing.append("conservative speed")
    if not (enable_motion and acknowledge):
        missing.append("both --enable-motion and --i-understand-this-moves-hardware")
    return {
        "allowed": len(missing) == 0,
        "missing": missing,
        "message": IMAGE_CALIBRATION_REFUSAL,
    }


def check_nudge_delta(delta_deg: float, hard_limit_deg: float = 5.0,
                      allow_larger: bool = False) -> None:
    """Refuse a single-servo nudge larger than the hard limit unless overridden."""
    if abs(float(delta_deg)) > float(hard_limit_deg) and not allow_larger:
        raise SafetyError(
            f"nudge delta {delta_deg:.1f} deg exceeds the hard limit "
            f"{hard_limit_deg:.1f} deg. Pass --dangerous-allow-larger-motion to override "
            "(NOT recommended for first tests)."
        )


def estop_instructions() -> List[str]:
    return [
        "EMERGENCY STOP:",
        "  1. CUT POWER: switch off / unplug the DC 9-12.6V supply immediately.",
        "  2. Ctrl+C in this terminal (disables torque if a verified backend exists).",
        "  3. Unplug the USB-C serial cable.",
        "  4. Keep hands clear of the joints until power is removed.",
    ]


def print_estop_instructions(log: Callable[[str], None] = print) -> None:
    for line in estop_instructions():
        log(line)


def countdown(n: int = 5, log: Callable[[str], None] = print,
              sleep_fn: Callable[[float], None] = time.sleep) -> None:
    nums = " ".join(f"{i}…" for i in range(n, 0, -1))
    log(f"Moving real hardware in {nums}")
    for i in range(n, 0, -1):
        log(f"  {i}…")
        sleep_fn(1.0)


def motion_logger(command: str, base_dir: str = "runs/real_motion_logs",
                  stamp: Optional[str] = None):
    """A :class:`CommandLogger` writing to runs/real_motion_logs/<command>_<ts>.jsonl."""
    from terafold.robot.command_logger import CommandLogger

    os.makedirs(base_dir, exist_ok=True)
    stamp = stamp or time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(base_dir, f"{command}_{stamp}.jsonl")
    return CommandLogger(path=path, echo=False)
