"""LeArm adapter — a SAFE SHELL only.

The LeArm is a hobby serial servo arm. TeraFold ships a *shell* adapter that can
plan and dry-run against it, but **deliberately does not invent a serial
protocol**: real motion requires a user-supplied protocol file. Until that file
exists and its encoder is implemented, every real-motion path raises a clear,
actionable error instead of guessing servo byte sequences (which could damage
hardware or people).

Numpy only at import time; no serial library is imported here.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from terafold.config.schema import RobotConfig
from terafold.robot.base import BaseRobot, RobotAction, RobotObservation
from terafold.robot.command_logger import CommandLogger
from terafold.robot.safety import SafetyChecker, SafetyConfig, SafetyError

__all__ = ["LeArmAdapter"]

_PROTOCOL_GUIDANCE = (
    "Provide a LeArm protocol file; see configs/robot_learm_template.yaml. "
    "The serial command encoding for your specific LeArm variant must be "
    "supplied by you — TeraFold will not invent servo byte sequences."
)


class LeArmAdapter(BaseRobot):
    """Safe shell adapter for the LeArm serial servo arm.

    In dry-run (the default) this logs the commands it *would* send and returns
    placeholder observations. Real motion is blocked unless ``dry_run=False``
    **and** ``config.protocol_file`` points to an existing, implemented protocol.
    """

    def __init__(
        self,
        config: RobotConfig,
        dry_run: bool = True,
        logger: Optional[CommandLogger] = None,
    ) -> None:
        super().__init__(dry_run=dry_run)
        self.config = config
        self.logger = logger
        self.safety = SafetyChecker(SafetyConfig.from_robot_config(config))
        self._gripper_state = float(config.gripper_open_pos)
        self._ee_pose = np.array(
            config.home_pose if config.home_pose is not None else [0.0, 0.0, 0.0, np.pi, 0.0, 0.0],
            dtype=np.float64,
        )

    # -- lifecycle ------------------------------------------------------
    def connect(self) -> None:
        if self.dry_run:
            self._connected = True
            if self.logger:
                self.logger.log("connect", {"dry_run": True, "type": "learm"})
            return
        # Real connection requires a user-provided, implemented protocol file.
        self._require_protocol()
        # Even with a protocol file present, the encoder/serial transport is not
        # implemented in this shell — block rather than open a port blindly.
        raise NotImplementedError(_PROTOCOL_GUIDANCE)

    def disconnect(self) -> None:
        self._connected = False
        if self.logger:
            self.logger.log("disconnect", {"type": "learm"})

    # -- io -------------------------------------------------------------
    def get_observation(self) -> RobotObservation:
        """Return a placeholder observation (no real encoder feedback)."""
        return RobotObservation(
            timestamp=time.time(),
            gripper_state=float(self._gripper_state),
            ee_pose=self._ee_pose.copy(),
            raw_vendor_state={"adapter": "learm", "dry_run": self.dry_run, "placeholder": True},
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
                self.logger.log_action(action, obs, extra={"dry_run": True, "type": "learm"})
            return obs
        # Real motion path is explicitly blocked.
        self._send_serial(b"")  # raises NotImplementedError
        raise NotImplementedError(_PROTOCOL_GUIDANCE)  # unreachable, defensive

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
            self.logger.log("stop", {"type": "learm"})

    def emergency_stop(self) -> None:
        if self.logger:
            self.logger.log("emergency_stop", {"type": "learm"})

    # -- protocol (intentionally unimplemented) -------------------------
    def _require_protocol(self) -> None:
        import os

        pf = self.config.protocol_file
        if not pf or not os.path.exists(pf):
            raise SafetyError(
                "Real LeArm motion requires an existing protocol_file. " + _PROTOCOL_GUIDANCE
            )

    def encode_joint_command(self, *args, **kwargs) -> bytes:
        """Encode joints into serial bytes — NOT implemented (no invented protocol)."""
        raise NotImplementedError(_PROTOCOL_GUIDANCE)

    def _send_serial(self, *args, **kwargs) -> None:
        """Transmit bytes over serial — NOT implemented (no invented protocol)."""
        raise NotImplementedError(_PROTOCOL_GUIDANCE)
