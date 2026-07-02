"""Load and validate the dual-YAM / MolmoAct2 reference config.

Thin YAML layer (pyyaml only) that turns ``configs/yam_dual_reference.yaml``
into a :class:`YamDualConfig` dataclass with the derived shapes the rest of the
stack needs (action dim, state layout). Loading *enforces shadow mode*: a
config with ``autonomous_execution_enabled: true`` is refused outright.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from terafold.yam.safety import assert_shadow_mode

__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_NORM_TAG",
    "DEFAULT_CAMERA_ORDER",
    "DEFAULT_TASK",
    "REFERENCE_CONFIG",
    "YamDualConfig",
    "load_yam_config",
]

PathLike = Union[str, Path]

DEFAULT_MODEL = "allenai/MolmoAct2-BimanualYAM"
DEFAULT_NORM_TAG = "yam_dual_molmoact2"
DEFAULT_CAMERA_ORDER = ("top", "left", "right")
DEFAULT_TASK = "fold the towel in half neatly"

#: The reference config shipped with the repo.
REFERENCE_CONFIG: Path = (
    Path(__file__).resolve().parents[2] / "configs" / "yam_dual_reference.yaml"
)


@dataclass
class YamDualConfig:
    """Validated view of ``yam_dual_reference.yaml`` (raw dict kept in ``raw``)."""

    robot: str = "dual_yam_standard"
    arms: List[str] = field(default_factory=lambda: ["left_yam", "right_yam"])
    camera_order: List[str] = field(default_factory=lambda: list(DEFAULT_CAMERA_ORDER))
    norm_tag: str = DEFAULT_NORM_TAG
    action_mode: str = "continuous"
    control_mode: str = "absolute_joint_pose"
    autonomous_execution_enabled: bool = False
    safe_speed_scale: float = 0.10
    max_joint_delta_rad: Optional[float] = None
    max_gripper_delta: Optional[float] = None
    workspace_bounds: Dict[str, Any] = field(default_factory=dict)
    table: Dict[str, Any] = field(default_factory=dict)
    joints_per_arm: int = 6
    grippers_per_arm: int = 1
    cameras: Dict[str, Any] = field(default_factory=dict)
    paths: Dict[str, Any] = field(default_factory=dict)
    model_name: str = DEFAULT_MODEL
    model_dtype: str = "bfloat16"
    notes: str = ""
    source: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def per_arm_dim(self) -> int:
        return int(self.joints_per_arm) + int(self.grippers_per_arm)

    @property
    def action_dim(self) -> int:
        """arms * (joints + grippers); action vector layout = :meth:`state_layout`."""
        return len(self.arms) * self.per_arm_dim

    @property
    def state_dim(self) -> int:
        return self.action_dim

    def state_layout(self) -> List[str]:
        """Ordered slot names, e.g. ``left_yam.joint0 .. left_yam.gripper, right_yam...``."""
        names: List[str] = []
        for arm in self.arms:
            names.extend(f"{arm}.joint{i}" for i in range(int(self.joints_per_arm)))
            names.extend(f"{arm}.gripper{i}" if int(self.grippers_per_arm) > 1 else f"{arm}.gripper"
                         for i in range(int(self.grippers_per_arm)))
        return names


def load_yam_config(path: Optional[PathLike] = None) -> YamDualConfig:
    """Load + validate a YAM config YAML (defaults to the reference config).

    Raises ``FileNotFoundError`` for a missing file, ``ValueError`` for a
    malformed one, and :class:`~terafold.yam.safety.ShadowModeViolation` if the
    config tries to enable autonomous execution.
    """
    p = Path(path) if path is not None else REFERENCE_CONFIG
    if not p.exists():
        raise FileNotFoundError(f"YAM config not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        try:
            data = yaml.safe_load(fh) or {}
        except yaml.YAMLError as e:
            raise ValueError(f"Malformed YAML in {p}: {e}") from e
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping at top level of {p}, got {type(data).__name__}")

    model = data.get("model") or {}
    if not isinstance(model, dict):
        raise ValueError(f"'model' must be a mapping in {p}")

    cfg = YamDualConfig(
        robot=str(data.get("robot", "dual_yam_standard")),
        arms=[str(a) for a in (data.get("arms") or ["left_yam", "right_yam"])],
        camera_order=[str(c) for c in (data.get("camera_order") or DEFAULT_CAMERA_ORDER)],
        norm_tag=str(data.get("norm_tag", DEFAULT_NORM_TAG)),
        action_mode=str(data.get("action_mode", "continuous")),
        control_mode=str(data.get("control_mode", "absolute_joint_pose")),
        autonomous_execution_enabled=bool(data.get("autonomous_execution_enabled", False)),
        safe_speed_scale=float(data.get("safe_speed_scale", 0.10)),
        max_joint_delta_rad=(None if data.get("max_joint_delta_rad") is None
                             else float(data["max_joint_delta_rad"])),
        max_gripper_delta=(None if data.get("max_gripper_delta") is None
                           else float(data["max_gripper_delta"])),
        workspace_bounds=dict(data.get("workspace_bounds") or {}),
        table=dict(data.get("table") or {}),
        joints_per_arm=int(data.get("joints_per_arm", 6)),
        grippers_per_arm=int(data.get("grippers_per_arm", 1)),
        cameras=dict(data.get("cameras") or {}),
        paths=dict(data.get("paths") or {}),
        model_name=str(model.get("name", DEFAULT_MODEL)),
        model_dtype=str(model.get("dtype", "bfloat16")),
        notes=str(data.get("notes", "")),
        source=str(p),
        raw=data,
    )

    if not cfg.camera_order:
        raise ValueError(f"camera_order must be a non-empty list in {p}")
    if not cfg.arms:
        raise ValueError(f"arms must be a non-empty list in {p}")
    if cfg.joints_per_arm <= 0 or cfg.grippers_per_arm < 0:
        raise ValueError(f"joints_per_arm/grippers_per_arm invalid in {p}")

    # Shadow mode is enforced at load time: a config that flips autonomy on is refused.
    assert_shadow_mode(cfg)
    return cfg
