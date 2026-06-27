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


@dataclass
class JointMap:
    robot: str
    servo_joint_map: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    created: float = 0.0
    note: str = ""

    def joint_to_id(self) -> Dict[str, int]:
        return {v["joint"]: int(k) for k, v in self.servo_joint_map.items()
                if v.get("joint")}

    def id_for_joint(self, joint: str) -> Optional[int]:
        return self.joint_to_id().get(joint)

    def sign_for_id(self, servo_id: int) -> int:
        return int(self.servo_joint_map.get(int(servo_id), {}).get("sign", 1))

    def mapped_ids(self) -> List[int]:
        return [int(k) for k in self.servo_joint_map]

    def to_dict(self) -> Dict[str, Any]:
        return {"robot": self.robot,
                "servo_joint_map": {int(k): v for k, v in self.servo_joint_map.items()},
                "created": self.created, "note": self.note}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "JointMap":
        raw = d.get("servo_joint_map", {}) or {}
        smap = {int(k): v for k, v in raw.items()}
        return cls(robot=d.get("robot", ""), servo_joint_map=smap,
                   created=float(d.get("created", 0.0)), note=d.get("note", ""))


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
