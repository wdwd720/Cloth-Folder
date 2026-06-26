"""GenericArmAdapter — a safe shell for an unidentified robot arm.

We don't know the exact arm yet, so this adapter:

* defaults to DRY-RUN only,
* logs the intended end-effector trajectory,
* can export that trajectory to JSON/CSV for manual / vendor-software testing,
* and REFUSES real motion unless a *verified command backend* is supplied
  (a callable that actually drives the hardware via a confirmed protocol).

There is deliberately no invented serial protocol here — providing a fake one
would be unsafe. Wire a real backend only after identifying the arm with
``terafold robot-scan`` + ``robot-info-template`` and testing the vendor software.
"""

from __future__ import annotations

import time
from typing import Callable, List, Optional

import numpy as np

from terafold.robot.base import BaseRobot, RobotAction, RobotObservation
from terafold.robot.safety import SafetyChecker, SafetyConfig, SafetyError

__all__ = ["GenericArmAdapter"]


class GenericArmAdapter(BaseRobot):
    """Dry-run-only shell for an unknown arm, with trajectory logging + export."""

    def __init__(
        self,
        config: object = None,
        dry_run: bool = True,
        logger: object = None,
        command_backend: Optional[Callable[[RobotAction], RobotObservation]] = None,
        start_pose: Optional[List[float]] = None,
    ) -> None:
        super().__init__(dry_run=dry_run)
        self.config = config
        self.logger = logger
        # A verified backend is the ONLY way real motion is allowed.
        self._command_backend = command_backend
        self._ee_pose = np.array(
            start_pose if start_pose is not None else [0.0, 0.0, 0.0, np.pi, 0.0, 0.0],
            dtype=np.float64,
        )
        self._gripper_state = 1.0
        ws = getattr(config, "workspace", None)
        self.safety = SafetyChecker(
            SafetyConfig.from_robot_config(config) if config is not None and ws is not None
            else SafetyConfig()
        )
        self._log: List[dict] = []  # intended-command log (for export)

    # -- lifecycle ------------------------------------------------------
    def connect(self) -> None:
        if self.dry_run:
            self._connected = True
            self._note("connect", {"dry_run": True, "type": "generic"})
            return
        if self._command_backend is None:
            raise SafetyError(
                "GenericArmAdapter: real motion requested but no verified command "
                "backend is wired. Identify the arm (terafold robot-scan / "
                "robot-info-template), test the vendor software, then provide a "
                "command_backend. Until then run in --dry-run."
            )
        self._connected = True
        self._note("connect", {"dry_run": False, "type": "generic", "backend": "provided"})

    def disconnect(self) -> None:
        self._connected = False
        self._note("disconnect", {"type": "generic"})

    # -- io -------------------------------------------------------------
    def get_observation(self) -> RobotObservation:
        return RobotObservation(
            timestamp=time.time(),
            gripper_state=float(self._gripper_state),
            ee_pose=self._ee_pose.copy(),
            raw_vendor_state={"adapter": "generic", "dry_run": self.dry_run},
        )

    def send_action(self, action: RobotAction) -> RobotObservation:
        self.safety.check_action(action)
        if action.target_ee_pose is not None:
            target = np.asarray(action.target_ee_pose, dtype=np.float64).reshape(-1)
            n = min(target.shape[0], self._ee_pose.shape[0])
            self._ee_pose[:n] = target[:n]
        if action.gripper is not None:
            self._gripper_state = float(action.gripper)

        self._log.append(
            {
                "phase": (action.metadata or {}).get("phase", ""),
                "ee_pose": self._ee_pose.tolist(),
                "gripper": float(self._gripper_state),
                "speed": float(action.speed),
                "duration": float(action.duration),
            }
        )
        obs = self.get_observation()
        if self.dry_run or self._command_backend is None:
            self._note("send_action(dry_run)", {"ee_pose": self._ee_pose.tolist()})
            return obs
        # Real motion path: delegate to the VERIFIED backend (never fabricated).
        return self._command_backend(action)

    def move_ee(self, pose: np.ndarray, speed: float = 0.5) -> RobotObservation:
        return self.send_action(RobotAction(target_ee_pose=pose, speed=speed))

    def open_gripper(self) -> None:
        self._gripper_state = 1.0

    def close_gripper(self) -> None:
        self._gripper_state = 0.0

    def stop(self) -> None:
        self._note("stop", {})

    def emergency_stop(self) -> None:
        self._note("emergency_stop", {})

    # -- export ---------------------------------------------------------
    def export_trajectory(self, out: str, fmt: str = "csv") -> str:
        """Export the logged intended trajectory to CSV/JSON for external testing."""
        from terafold.data.trajectory_export import write_rows

        return write_rows(self._log, out, fmt=fmt)

    # -- internal -------------------------------------------------------
    def _note(self, event: str, data: dict) -> None:
        if self.logger is not None:
            try:
                self.logger.log(event, data)
            except Exception:
                pass
