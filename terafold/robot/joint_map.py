"""Servo-ID -> joint-name mapping for a physical arm (built by `map-servo-joints`).

This maps confirmed servo IDs to TeraFold joint names so a fold plan can be turned
into a *conservative* joint-space ghost fold without hardcoding servo numbers. It
does NOT provide kinematics — autonomous Cartesian IK is still refused.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml

from terafold.robot.arm_config import configs_robots_dir

__all__ = ["JointMap", "joint_map_path", "save_joint_map", "load_joint_map"]


def joint_map_path(robot: str) -> str:
    return os.path.join(configs_robots_dir(), f"{robot}_joint_map.yaml")


# The fields a fully-populated per-servo joint-map entry should carry. Older,
# minimal maps ({"joint", "sign"}) remain valid — these are optional enrichments.
JOINT_ENTRY_FIELDS = (
    "joint", "role", "sign", "raw_home", "raw_min", "raw_max",
    "last_observed", "motion_notes", "confidence", "operator_notes",
)


@dataclass
class JointMap:
    robot: str
    servo_joint_map: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    created: float = 0.0
    note: str = ""
    disabled_ids: List[int] = field(default_factory=list)
    operator_notes: str = ""

    def joint_to_id(self) -> Dict[str, int]:
        return {v["joint"]: int(k) for k, v in self.servo_joint_map.items()
                if v.get("joint")}

    def id_for_joint(self, joint: str) -> Optional[int]:
        return self.joint_to_id().get(joint)

    def sign_for_id(self, servo_id: int) -> int:
        return int(self.servo_joint_map.get(int(servo_id), {}).get("sign", 1))

    def mapped_ids(self) -> List[int]:
        return [int(k) for k in self.servo_joint_map]

    def raw_limits_for_id(self, servo_id: int) -> Optional[tuple]:
        v = self.servo_joint_map.get(int(servo_id), {})
        if "raw_min" in v and "raw_max" in v:
            return (int(v["raw_min"]), int(v["raw_max"]))
        return None

    def has_raw_limits(self) -> bool:
        return any("raw_min" in v and "raw_max" in v
                   for v in self.servo_joint_map.values() if isinstance(v, dict))

    def validate(self) -> List[str]:
        """Return a list of schema problems (empty ⇒ valid)."""
        problems: List[str] = []
        if not self.robot:
            problems.append("missing robot name")
        for sid, v in self.servo_joint_map.items():
            if not isinstance(v, dict):
                problems.append(f"servo {sid}: entry is not a mapping")
                continue
            if not v.get("joint"):
                problems.append(f"servo {sid}: missing joint name")
            if "raw_min" in v and "raw_max" in v and v["raw_min"] > v["raw_max"]:
                problems.append(f"servo {sid}: raw_min > raw_max")
            sign = v.get("sign", 1)
            if sign not in (-1, 1):
                problems.append(f"servo {sid}: sign must be ±1 (got {sign})")
        return problems

    def to_dict(self) -> Dict[str, Any]:
        return {"robot": self.robot,
                "servo_joint_map": {int(k): v for k, v in self.servo_joint_map.items()},
                "created": self.created, "note": self.note,
                "disabled_ids": [int(i) for i in self.disabled_ids],
                "operator_notes": self.operator_notes}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "JointMap":
        raw = d.get("servo_joint_map", {}) or {}
        smap = {int(k): v for k, v in raw.items()}
        return cls(robot=d.get("robot", ""), servo_joint_map=smap,
                   created=float(d.get("created", 0.0)), note=d.get("note", ""),
                   disabled_ids=[int(i) for i in (d.get("disabled_ids") or [])],
                   operator_notes=d.get("operator_notes", ""))


def save_joint_map(jm: JointMap, path: Optional[str] = None) -> str:
    path = path or joint_map_path(jm.robot)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(jm.to_dict(), f, sort_keys=True)
    return path


def load_joint_map(robot_or_path: str) -> Optional[JointMap]:
    """Load a joint map by robot name or path; ``None`` if it does not exist."""
    path = robot_or_path
    if not (robot_or_path.endswith((".yaml", ".yml")) and os.path.exists(robot_or_path)):
        path = joint_map_path(robot_or_path)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return JointMap.from_dict(yaml.safe_load(f) or {})
