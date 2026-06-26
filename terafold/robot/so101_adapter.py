"""SO-101 adapter — a LeRobot-backed arm shell.

The SO-101 is supported through the `lerobot` stack. That dependency is heavy and
optional, so it is **lazily imported inside the methods that need it**. Importing
this module never requires lerobot; if it is missing, the methods raise a clear
``ImportError`` with install guidance. Dry-run is the default and never touches
hardware.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from terafold.config.schema import RobotConfig
from terafold.robot.base import BaseRobot, RobotAction, RobotObservation
from terafold.robot.command_logger import CommandLogger
from terafold.robot.safety import SafetyChecker, SafetyConfig

__all__ = ["check_lerobot_available", "SO101Adapter"]

_INSTALL_GUIDANCE = (
    "lerobot is not installed. Install it with: pip install -e '.[lerobot]' "
    "(or `pip install lerobot`). The SO-101 adapter requires lerobot for real "
    "hardware control; until then use the mock robot (dry-run)."
)


def check_lerobot_available() -> bool:
    """Return True if the ``lerobot`` package can be imported (no hard import)."""
    import importlib.util

    return importlib.util.find_spec("lerobot") is not None


class SO101Adapter(BaseRobot):
    """SO-101 adapter that defers all hardware work to lerobot (lazily)."""

    def __init__(self, config: RobotConfig, dry_run: bool = True,
                 logger: Optional[CommandLogger] = None) -> None:
        super().__init__(dry_run=dry_run)
        self.config = config
        self.logger = logger
        self.safety = SafetyChecker(SafetyConfig.from_robot_config(config))
        self._gripper_state = float(config.gripper_open_pos)
        self._ee_pose = np.array(
            config.home_pose if config.home_pose is not None else [0.0, 0.0, 0.0, np.pi, 0.0, 0.0],
            dtype=np.float64,
        )
        self._lerobot = None

    def _require_lerobot(self):
        """Lazily import lerobot, raising actionable guidance if missing."""
        if self._lerobot is not None:
            return self._lerobot
        try:
            import lerobot  # noqa: F401  (heavy, optional dep — lazy on purpose)
        except ImportError as e:
            raise ImportError(_INSTALL_GUIDANCE) from e
        self._lerobot = lerobot
        return lerobot

    # -- lifecycle ------------------------------------------------------
    def connect(self) -> None:
        if self.dry_run:
            self._connected = True
            if self.logger:
                self.logger.log("connect", {"dry_run": True, "type": "so101"})
            return
        self._require_lerobot()
        # Real bring-up would construct a lerobot robot here from self.config.
        raise NotImplementedError(
            "Real SO-101 bring-up via lerobot is not wired in this shell; "
            "construct the lerobot robot from your config and connect it here."
        )

    def disconnect(self) -> None:
        self._connected = False
        if self.logger:
            self.logger.log("disconnect", {"type": "so101"})

    # -- io -------------------------------------------------------------
    def get_observation(self) -> RobotObservation:
        if self.dry_run:
            return RobotObservation(
                timestamp=time.time(),
                gripper_state=float(self._gripper_state),
                ee_pose=self._ee_pose.copy(),
                raw_vendor_state={"adapter": "so101", "dry_run": True, "placeholder": True},
            )
        self._require_lerobot()
        raise NotImplementedError(
            "Real SO-101 observation requires a connected lerobot robot."
        )

    def send_action(self, action: RobotAction) -> RobotObservation:
        if self.dry_run:
            self.safety.check_action(action, current_pose=self._ee_pose)
            if action.target_ee_pose is not None:
                target = np.asarray(action.target_ee_pose, dtype=np.float64).reshape(-1)
                n = min(target.shape[0], self._ee_pose.shape[0])
                self._ee_pose[:n] = target[:n]
            if action.gripper is not None:
                self._gripper_state = float(action.gripper)
            obs = self.get_observation()
            if self.logger:
                self.logger.log_action(action, obs, extra={"dry_run": True, "type": "so101"})
            return obs
        self._require_lerobot()
        raise NotImplementedError(
            "Real SO-101 motion requires a connected lerobot robot."
        )

    def move_ee(self, pose: np.ndarray, speed: float = 0.5) -> RobotObservation:
        return self.send_action(RobotAction(target_ee_pose=pose, speed=speed))

    def open_gripper(self) -> None:
        self._gripper_state = float(self.config.gripper_open_pos)
        if self.logger:
            self.logger.log("open_gripper", {"dry_run": self.dry_run})

    def close_gripper(self) -> None:
        self._gripper_state = float(self.config.gripper_close_pos)
        if self.logger:
            self.logger.log("close_gripper", {"dry_run": self.dry_run})

    # -- safety ---------------------------------------------------------
    def stop(self) -> None:
        if self.logger:
            self.logger.log("stop", {"type": "so101"})

    def emergency_stop(self) -> None:
        if self.logger:
            self.logger.log("emergency_stop", {"type": "so101"})
