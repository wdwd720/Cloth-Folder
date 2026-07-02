"""A representable-but-UNVALIDATED kinematic model for the custom 7-DOF arm.

The custom bus-servo arm has no confirmed link lengths, no confirmed raw->angle
scale, and no validated DH chain. This module gives us a place to *hold* such a
model once it is measured, while making the safety posture explicit: until the
model is marked ``validated`` (and additionally ``allowed_for_contact``), every
request for Cartesian / contact IK is REFUSED with a clear, actionable reason.

It loads whatever structure we *do* know (the servo->joint map and the arm
config) without ever pretending that gives us kinematics.

Pure stdlib + TeraFold loaders (no numpy needed at import; no serial).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from terafold.robot.arm_config import load_arm_config
from terafold.robot.joint_map import load_joint_map

__all__ = ["CustomArmModel"]


@dataclass
class CustomArmModel:
    """The (currently unvalidated) kinematic model for a custom arm.

    Fields
    ------
    robot:
        Robot config name.
    joints:
        Per-joint structure we actually know, each a dict with ``servo_id``,
        ``joint``, ``sign``, ``raw_home``, ``raw_min``, ``raw_max``, ``scale``.
        Empty when no joint map exists.
    validated:
        ``True`` only when a kinematic model has been measured AND verified.
        Always ``False`` for the custom arm today.
    allowed_for_contact:
        Operator-gated flag; contact IK additionally requires this.
    link_lengths_m:
        Measured link lengths once known (``None`` until then).
    home_pose:
        A reference home pose once known (``None`` until then).
    """

    robot: str
    joints: List[Dict[str, Any]] = field(default_factory=list)
    validated: bool = False
    allowed_for_contact: bool = False
    link_lengths_m: Optional[List[float]] = None
    home_pose: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def from_robot(cls, robot: str) -> "CustomArmModel":
        """Build the model from the arm config + joint map (no kinematics implied).

        If no joint map exists, ``joints`` is empty and ``validated`` is False.
        A model loaded this way is NEVER validated automatically — validation is
        an explicit, deliberate step that does not exist for the custom arm yet.
        """
        cfg = load_arm_config(robot)
        jm = load_joint_map(cfg.robot_name)
        joints: List[Dict[str, Any]] = []
        if jm is not None:
            for sid, info in sorted(jm.servo_joint_map.items()):
                info = info if isinstance(info, dict) else {}
                joints.append(
                    {
                        "servo_id": int(sid),
                        "joint": info.get("joint"),
                        "sign": int(info.get("sign", 1)),
                        "raw_home": info.get("raw_home"),
                        "raw_min": info.get("raw_min"),
                        "raw_max": info.get("raw_max"),
                        "scale": info.get("scale"),
                    }
                )
        return cls(robot=cfg.robot_name, joints=joints, validated=False,
                   allowed_for_contact=False)

    # ------------------------------------------------------------------
    # Capability gates
    # ------------------------------------------------------------------
    def can_do_contact_ik(self) -> bool:
        """Contact IK is allowed only when the model is validated AND unlocked."""
        return bool(self.validated and self.allowed_for_contact)

    def ik_refused(self) -> Dict[str, Any]:
        """The structured refusal explaining why custom-arm IK is not available."""
        return {
            "ok": False,
            "refused": True,
            "robot": self.robot,
            "validated": self.validated,
            "allowed_for_contact": self.allowed_for_contact,
            "reason": (
                f"Custom-arm IK is REFUSED for {self.robot!r}: the kinematic model "
                "is NOT validated (validated=False). No confirmed link lengths, "
                "raw->angle scale, or DH chain exist, so Cartesian IK and contact "
                "folding stay LOCKED."
            ),
            "remedy": [
                "measure link lengths and the raw->angle scale per joint",
                "populate and VALIDATE a kinematic model (set validated=True)",
                "for contact, additionally set allowed_for_contact via an explicit "
                "operator unlock",
            ],
        }

    def ik(self, target: Any) -> Dict[str, Any]:
        """Attempt Cartesian IK. Returns the refusal dict while unvalidated.

        Even once ``validated`` is set, no solver is wired for the custom arm yet,
        so this raises rather than silently returning a bogus pose.
        """
        if not self.can_do_contact_ik():
            return self.ik_refused()
        raise NotImplementedError(
            "A validated custom-arm IK solver is not implemented yet; "
            "use the toy planar DLS solver only for testing."
        )
