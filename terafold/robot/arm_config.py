"""Loader for a physical bus-servo arm config (configs/robots/*.yaml).

A light dataclass — conservative, safety-first. The limits it carries are
temporary placeholders for safe first tests, NOT an IK envelope.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml

__all__ = ["PhysicalArmConfig", "load_arm_config", "configs_robots_dir"]


def configs_robots_dir() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", "..", "configs", "robots"))


@dataclass
class PhysicalArmConfig:
    robot_name: str = "physical_arm"
    dof: int = 7
    adapter: str = "waveshare_bus_servo"
    port: str = "auto"
    baudrate_candidates: List[int] = field(default_factory=lambda: [1000000, 115200])
    servo_ids: Any = "unknown"
    joint_names: List[str] = field(default_factory=list)
    joint_limits_deg: Dict[str, List[float]] = field(default_factory=dict)
    safe_speed: str = "very_slow"
    max_delta_per_test_deg: float = 3.0
    hard_delta_limit_deg: float = 5.0
    table_clearance_m: float = 0.10
    motion_default: str = "disabled"
    hardware: Dict[str, Any] = field(default_factory=dict)
    source_path: Optional[str] = None

    @property
    def motion_enabled_by_default(self) -> bool:
        return str(self.motion_default).lower() in ("enabled", "on", "true")

    @property
    def baudrate(self) -> int:
        return int(self.baudrate_candidates[0]) if self.baudrate_candidates else 1000000

    @classmethod
    def from_dict(cls, d: Dict[str, Any], source_path: Optional[str] = None) -> "PhysicalArmConfig":
        known = {f for f in cls.__dataclass_fields__}  # noqa: F821
        kwargs = {k: v for k, v in d.items() if k in known}
        return cls(source_path=source_path, **kwargs)

    @classmethod
    def from_yaml(cls, path: str) -> "PhysicalArmConfig":
        with open(path) as f:
            d = yaml.safe_load(f) or {}
        return cls.from_dict(d, source_path=os.path.abspath(path))


def load_arm_config(name_or_path: str) -> PhysicalArmConfig:
    """Resolve a robot name or YAML path to a :class:`PhysicalArmConfig`.

    ``"physical_7dof_waveshare"`` -> ``configs/robots/physical_7dof_waveshare.yaml``.
    A path ending in ``.yaml``/``.yml`` is loaded directly.
    """
    if name_or_path.endswith((".yaml", ".yml")) and os.path.exists(name_or_path):
        return PhysicalArmConfig.from_yaml(name_or_path)
    candidate = os.path.join(configs_robots_dir(), f"{name_or_path}.yaml")
    if os.path.exists(candidate):
        return PhysicalArmConfig.from_yaml(candidate)
    raise FileNotFoundError(
        f"robot config {name_or_path!r} not found (looked for {candidate}). "
        "Pass a known name or a path to a robots/*.yaml file."
    )
