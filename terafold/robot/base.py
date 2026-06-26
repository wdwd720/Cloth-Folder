"""Robot abstraction: observation/action schemas and the ``BaseRobot`` interface.

Everything physical in TeraFold talks to robots through this interface. The
mock robot, the LeArm shell, the SO-101 adapter, and the future bimanual robot
all implement :class:`BaseRobot`. The observation/action dataclasses are the
canonical wire format that the recorder and the LeRobot exporter consume, so
they expose flat numeric vectors (``observation.state`` / ``action``).

Pure-python + numpy: no torch / serial / opencv imports here.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


def _arr(x) -> Optional[np.ndarray]:
    return None if x is None else np.asarray(x, dtype=np.float64).reshape(-1)


def _list(x) -> Optional[list]:
    return None if x is None else np.asarray(x, dtype=np.float64).reshape(-1).tolist()


@dataclass
class RobotObservation:
    """A single robot observation.

    ``ee_pose`` is ``[x, y, z, roll, pitch, yaw]`` (meters / radians) in the
    robot base frame when available. ``gripper_state`` is a normalized opening
    in ``[0, 1]`` (0 = closed, 1 = open).
    """

    timestamp: float
    joint_positions: Optional[np.ndarray] = None
    joint_velocities: Optional[np.ndarray] = None
    gripper_state: float = 1.0
    ee_pose: Optional[np.ndarray] = None
    raw_vendor_state: Optional[Dict] = None

    def __post_init__(self) -> None:
        self.joint_positions = _arr(self.joint_positions)
        self.joint_velocities = _arr(self.joint_velocities)
        self.ee_pose = _arr(self.ee_pose)

    def to_state_vector(self) -> np.ndarray:
        """Flat numeric state for imitation learning (``observation.state``).

        Layout: ``[joint_positions..., gripper_state, ee_pose...]`` using
        whichever components are present. This is the vector exported to LeRobot.
        """
        parts: List[np.ndarray] = []
        if self.joint_positions is not None:
            parts.append(self.joint_positions)
        parts.append(np.array([self.gripper_state]))
        if self.ee_pose is not None:
            parts.append(self.ee_pose)
        return np.concatenate(parts) if parts else np.zeros(0)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "joint_positions": _list(self.joint_positions),
            "joint_velocities": _list(self.joint_velocities),
            "gripper_state": self.gripper_state,
            "ee_pose": _list(self.ee_pose),
            "raw_vendor_state": self.raw_vendor_state,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RobotObservation":
        return cls(
            timestamp=float(d["timestamp"]),
            joint_positions=d.get("joint_positions"),
            joint_velocities=d.get("joint_velocities"),
            gripper_state=float(d.get("gripper_state", 1.0)),
            ee_pose=d.get("ee_pose"),
            raw_vendor_state=d.get("raw_vendor_state"),
        )


@dataclass
class RobotAction:
    """A commanded action.

    Either ``target_joint_positions`` or ``target_ee_pose`` (or both) may be
    set. ``gripper`` is a normalized target opening in ``[0, 1]``. ``speed`` is
    a fraction in ``[0, 1]`` of the configured max speed. ``duration`` is the
    intended execution time in seconds (used for time-parameterized motion).
    """

    target_joint_positions: Optional[np.ndarray] = None
    target_ee_pose: Optional[np.ndarray] = None
    gripper: Optional[float] = None
    speed: float = 0.5
    duration: float = 1.0
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.target_joint_positions = _arr(self.target_joint_positions)
        self.target_ee_pose = _arr(self.target_ee_pose)

    def to_action_vector(self) -> np.ndarray:
        """Flat numeric action for imitation learning (the ``action`` field).

        Layout: ``[target_joint_positions..., gripper]`` if joints are set,
        else ``[target_ee_pose..., gripper]``. The gripper defaults to NaN-free
        by substituting the last commanded value (1.0 open) when unset.
        """
        g = 1.0 if self.gripper is None else float(self.gripper)
        if self.target_joint_positions is not None:
            return np.concatenate([self.target_joint_positions, [g]])
        if self.target_ee_pose is not None:
            return np.concatenate([self.target_ee_pose, [g]])
        return np.array([g])

    def to_dict(self) -> dict:
        return {
            "target_joint_positions": _list(self.target_joint_positions),
            "target_ee_pose": _list(self.target_ee_pose),
            "gripper": self.gripper,
            "speed": self.speed,
            "duration": self.duration,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RobotAction":
        return cls(
            target_joint_positions=d.get("target_joint_positions"),
            target_ee_pose=d.get("target_ee_pose"),
            gripper=d.get("gripper"),
            speed=float(d.get("speed", 0.5)),
            duration=float(d.get("duration", 1.0)),
            metadata=d.get("metadata", {}),
        )


class BaseRobot(abc.ABC):
    """Abstract robot interface.

    Concrete robots must implement the abstract methods. Convenience methods
    (:meth:`execute_trajectory`, context-manager support) are provided on top.

    Safety contract: a concrete adapter MUST NOT produce physical motion unless
    it is constructed with ``dry_run=False`` *and* its own hardware enablement
    checks pass. The default everywhere is dry-run.
    """

    def __init__(self, dry_run: bool = True) -> None:
        self.dry_run = dry_run
        self._connected = False

    # -- lifecycle ------------------------------------------------------
    @abc.abstractmethod
    def connect(self) -> None: ...

    @abc.abstractmethod
    def disconnect(self) -> None: ...

    @property
    def is_connected(self) -> bool:
        return self._connected

    # -- io -------------------------------------------------------------
    @abc.abstractmethod
    def get_observation(self) -> RobotObservation: ...

    @abc.abstractmethod
    def send_action(self, action: RobotAction) -> RobotObservation: ...

    @abc.abstractmethod
    def move_ee(self, pose: np.ndarray, speed: float = 0.5) -> RobotObservation: ...

    @abc.abstractmethod
    def open_gripper(self) -> None: ...

    @abc.abstractmethod
    def close_gripper(self) -> None: ...

    # -- safety ---------------------------------------------------------
    @abc.abstractmethod
    def stop(self) -> None:
        """Halt current motion (soft stop)."""

    @abc.abstractmethod
    def emergency_stop(self) -> None:
        """Immediately cut motion (hard stop). Must be safe to call any time."""

    # -- helpers --------------------------------------------------------
    def execute_trajectory(self, actions: List[RobotAction]) -> List[RobotObservation]:
        """Execute a sequence of actions, returning the observation after each."""
        observations = []
        for action in actions:
            observations.append(self.send_action(action))
        return observations

    def __enter__(self) -> "BaseRobot":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # On any error inside the block, stop motion before disconnecting.
        if exc_type is not None:
            try:
                self.emergency_stop()
            except Exception:
                pass
        self.disconnect()


__all__ = ["RobotObservation", "RobotAction", "BaseRobot"]
