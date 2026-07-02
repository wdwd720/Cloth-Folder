"""A one-glance report of the kinematic-model state for a robot.

Answers: *do we have enough of a model to do anything beyond joint-space moves?*
For the custom 7-DOF arm the honest answer is "no" — there is no joint map yet,
no confirmed raw->angle scale, no validated FK/DH chain, and so IK and contact
folding are refused. This module assembles that state (purely from on-disk config
+ joint map + calibration artifacts, no serial) and renders it as markdown.

Pure stdlib + TeraFold loaders (no numpy / serial at import).
"""

from __future__ import annotations

from typing import Any, Dict, List

from terafold.kinematics.custom_arm_model import CustomArmModel
from terafold.robot.arm_config import load_arm_config
from terafold.robot.capabilities import discover_artifacts
from terafold.robot.joint_map import load_joint_map

__all__ = ["robot_model_status", "report_markdown"]

# Joints that have a confirmed raw range constitute "raw limits known".
_RAW_LIMIT_KEYS = ("raw_min", "raw_max")


def _expected_ids(cfg: Any) -> List[int]:
    """The servo IDs we expect (explicit list, else 1..dof)."""
    if isinstance(cfg.servo_ids, list) and cfg.servo_ids:
        return sorted(int(i) for i in cfg.servo_ids)
    return list(range(1, int(cfg.dof) + 1))


def robot_model_status(robot: str) -> Dict[str, Any]:
    """Assemble the kinematic-model status for ``robot`` as a plain dict."""
    cfg = load_arm_config(robot)
    jm = load_joint_map(cfg.robot_name)
    model = CustomArmModel.from_robot(cfg.robot_name)

    active_ids = [int(i) for i in cfg.active_servo_ids]
    mapped_ids = jm.mapped_ids() if jm is not None else []
    # IDs we expect or run but have NOT mapped to a joint.
    expected = _expected_ids(cfg)
    missing_ids = sorted(set(active_ids + expected) - set(mapped_ids))

    known_joint_names = [j["joint"] for j in model.joints if j.get("joint")]
    known_signs = {int(j["servo_id"]): int(j["sign"]) for j in model.joints}
    raw_limits_known = any(
        all(j.get(k) is not None for k in _RAW_LIMIT_KEYS) for j in model.joints
    )
    scale_known = any(j.get("scale") for j in model.joints)

    raw_to_angle_status = (
        "known (per-joint scale present)"
        if scale_known
        else "UNKNOWN: no confirmed raw->angle scale (only a 360/4096 single-turn "
        "guess exists, known=False) — raw counts are NOT treated as angles"
    )
    fk_status = (
        "no validated FK/DH model for this custom arm; a toy DH chain exists for "
        "testing only"
    )
    ik_status = (
        "REFUSED: custom-arm IK is unvalidated (validated=False) — Cartesian IK and "
        "contact folding are LOCKED until a model is measured and validated"
    )

    arts = discover_artifacts(cfg.robot_name)
    cal_bits = []
    for name in ("kinematics", "camera_intrinsics", "table_homography",
                 "robot_table_transform"):
        info = arts.get(name, {})
        if not info.get("present"):
            cal_bits.append(f"{name}: absent")
        else:
            valid = info.get("valid_for_hover") or info.get("validated")
            cal_bits.append(f"{name}: present ({'valid' if valid else 'unvalidated'})")
    calibration_status = "; ".join(cal_bits) + " (nothing validated for IK/contact)"

    allowed_capabilities = _allowed_capabilities(
        joint_map_present=jm is not None,
        known_joint_names=known_joint_names,
        raw_limits_known=raw_limits_known,
        validated=model.validated,
    )

    return {
        "robot": cfg.robot_name,
        "joint_map_present": jm is not None,
        "active_ids": active_ids,
        "missing_ids": missing_ids,
        "known_joint_names": known_joint_names,
        "known_signs": known_signs,
        "raw_limits_known": raw_limits_known,
        "raw_to_angle_status": raw_to_angle_status,
        "fk_status": fk_status,
        "ik_status": ik_status,
        "calibration_status": calibration_status,
        "allowed_capabilities": allowed_capabilities,
        "validated": model.validated,
        "allowed_for_contact": model.allowed_for_contact,
    }


def _allowed_capabilities(
    *,
    joint_map_present: bool,
    known_joint_names: List[str],
    raw_limits_known: bool,
    validated: bool,
) -> List[str]:
    """Conservative list of what the *kinematic* state permits (never IK if unvalidated)."""
    allowed = ["sim-fold", "plan-fold", "robot-status", "robot-model-status"]
    if joint_map_present:
        allowed.append("map-servo-joints")
        if "base_yaw" in known_joint_names and raw_limits_known:
            allowed.append("real-image-ghost-fold (bounded base-yaw sweep, no contact)")
    if validated:
        allowed.append("cartesian-ik")
    return allowed


def report_markdown(status: Dict[str, Any]) -> str:
    """Render a :func:`robot_model_status` dict as a human-readable markdown report."""
    caps = status.get("allowed_capabilities") or []
    lines = [
        f"# Kinematic model status - {status.get('robot')}",
        "",
        f"- validated model: **{status.get('validated')}**   "
        f"contact allowed: **{status.get('allowed_for_contact')}**",
        f"- joint map present: {status.get('joint_map_present')}",
        f"- active servo IDs: {status.get('active_ids')}",
        f"- unmapped / missing IDs: {status.get('missing_ids')}",
        f"- known joint names: {status.get('known_joint_names') or 'none'}",
        f"- known signs: {status.get('known_signs') or 'none'}",
        f"- raw limits known: {status.get('raw_limits_known')}",
        f"- raw->angle: {status.get('raw_to_angle_status')}",
        f"- FK status: {status.get('fk_status')}",
        f"- IK status: {status.get('ik_status')}",
        f"- calibration: {status.get('calibration_status')}",
        "",
        "## Allowed capabilities",
    ]
    lines.extend(f"- {c}" for c in caps)
    lines += [
        "",
        "## Locked (needs a validated kinematic model)",
        "- Cartesian IK",
        "- calibrated hover to image points",
        "- any contact / fold primitive",
    ]
    return "\n".join(lines)
