"""Load TeraFold YAML configs into validated pydantic models.

Thin, numpy-free layer over :mod:`terafold.config.schema`. Reads YAML with
``pyyaml`` and constructs the strict (``extra='forbid'``) schema models so a
malformed key fails loudly at load time rather than deep in the pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union

import yaml

from terafold.config.schema import CameraConfig, RobotConfig, TaskConfig

__all__ = [
    "CONFIG_DIR",
    "load_yaml",
    "load_task_config",
    "load_robot_config",
    "load_camera_config",
    "default_task_config",
    "default_robot_config",
    "default_camera_config",
]

PathLike = Union[str, Path]

#: Absolute path to the repo's ``configs/`` directory.
CONFIG_DIR: Path = (Path(__file__).resolve().parents[2] / "configs").resolve()


def load_yaml(path: PathLike) -> Dict[str, Any]:
    """Read a YAML file into a plain dict (empty file -> ``{}``)."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping at top level of {p}, got {type(data).__name__}")
    return data


def load_task_config(path: PathLike) -> TaskConfig:
    """Load and validate a task config YAML."""
    return TaskConfig(**load_yaml(path))


def load_robot_config(path: PathLike) -> RobotConfig:
    """Load and validate a robot config YAML."""
    return RobotConfig(**load_yaml(path))


def load_camera_config(path: PathLike) -> CameraConfig:
    """Load and validate a camera config YAML."""
    return CameraConfig(**load_yaml(path))


def default_task_config() -> Path:
    """Path to the default task config shipped in ``configs/``."""
    return CONFIG_DIR / "task_fold_towel_half.yaml"


def default_robot_config() -> Path:
    """Path to the default (mock) robot config shipped in ``configs/``."""
    return CONFIG_DIR / "robot_mock.yaml"


def default_camera_config() -> Path:
    """Path to the default (mock) camera config shipped in ``configs/``."""
    return CONFIG_DIR / "camera_mock.yaml"
