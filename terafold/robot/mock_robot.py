"""A fully-simulated robot for Stage-0 development, testing, and dry-runs.

:class:`MockRobot` implements the :class:`~terafold.robot.base.BaseRobot`
interface against a trivial kinematic model: it tracks an end-effector pose
``[x, y, z, roll, pitch, yaw]`` and a normalized gripper opening, applying each
commanded action directly (perfect tracking). It logs every command and
**enforces the workspace** — sending an out-of-bounds target raises
:class:`~terafold.robot.safety.SafetyError`. No hardware, no torch, numpy only.
"""

from __future__ import annotations

import time
from typing import List, Optional, Sequence, Union

import numpy as np

from terafold.config.schema import WorkspaceConfig
from terafold.robot.base import BaseRobot, RobotAction, RobotObservation
from terafold.robot.command_logger import CommandLogger
from terafold.robot.safety import SafetyChecker, SafetyConfig, SafetyError

__all__ = ["MockRobot"]

# Tool-down resting orientation [roll, pitch, yaw].
_DEFAULT_ORIENTATION = (np.pi, 0.0, 0.0)


class MockRobot(BaseRobot):
    """Simulated end-effector robot with workspace enforcement."""

    def __init__(
        self,
        num_joints: int = 6,
        workspace: Optional[WorkspaceConfig] = None,
        safety: Optional[SafetyChecker] = None,
        logger: Optional[CommandLogger] = None,
        dry_run: bool = True,
        start_pose: Optional[Sequence[float]] = None,
        enforce_workspace: bool = True,
    ) -> None:
        super().__init__(dry_run=dry_run)
        self.num_joints = int(num_joints)
        self.workspace = workspace or WorkspaceConfig()
        self.enforce_workspace = enforce_workspace
        self.logger = logger
        if safety is not None:
            self.safety = safety
        else:
            self.safety = SafetyChecker(SafetyConfig(workspace=self.workspace,
                                                     min_z_m=self.workspace.z_min_m,
                                                     max_z_m=self.workspace.z_max_m))

        if start_pose is not None:
            self._ee_pose = np.asarray(start_pose, dtype=np.float64).reshape(-1)
        else:
            center = self.workspace
            self._ee_pose = np.array(
                [
                    0.5 * (center.x_min_m + center.x_max_m),
                    0.5 * (center.y_min_m + center.y_max_m),
                    center.z_max_m,
                    *_DEFAULT_ORIENTATION,
                ],
                dtype=np.float64,
            )
        self._joint_positions = np.zeros(self.num_joints, dtype=np.float64)
        self._gripper_state = 1.0  # open

    # -- lifecycle ------------------------------------------------------
    def connect(self) -> None:
        self._connected = True
        if self.logger:
            self.logger.log("connect", {"dry_run": self.dry_run, "type": "mock"})

    def disconnect(self) -> None:
        self._connected = False
        if self.logger:
            self.logger.log("disconnect", {"type": "mock"})

    # -- io -------------------------------------------------------------
    def get_observation(self) -> RobotObservation:
        return RobotObservation(
            timestamp=time.time(),
            joint_positions=self._joint_positions.copy(),
            gripper_state=float(self._gripper_state),
            ee_pose=self._ee_pose.copy(),
        )

    def _enforce(self, target_xyz: np.ndarray) -> None:
        """Raise :class:`SafetyError` if the target is out of bounds."""
        if not self.enforce_workspace:
            return
        x, y, z = float(target_xyz[0]), float(target_xyz[1]), float(target_xyz[2])
        if not self.workspace.contains_xyz(x, y, z):
            raise SafetyError(
                f"MockRobot target ({x:.3f}, {y:.3f}, {z:.3f}) outside workspace "
                f"x[{self.workspace.x_min_m}, {self.workspace.x_max_m}] "
                f"y[{self.workspace.y_min_m}, {self.workspace.y_max_m}] "
                f"z[{self.workspace.z_min_m}, {self.workspace.z_max_m}]"
            )

    def send_action(self, action: RobotAction) -> RobotObservation:
        """Apply an action to the simulated state and return the new observation."""
        if action.target_ee_pose is not None:
            target = np.asarray(action.target_ee_pose, dtype=np.float64).reshape(-1)
            self._enforce(target[:3])
            # Perfect tracking; keep resting orientation if none supplied.
            pose = self._ee_pose.copy()
            n = min(target.shape[0], pose.shape[0])
            pose[:n] = target[:n]
            self._ee_pose = pose
        elif action.target_joint_positions is not None:
            joints = np.asarray(action.target_joint_positions, dtype=np.float64).reshape(-1)
            n = min(joints.shape[0], self._joint_positions.shape[0])
            self._joint_positions[:n] = joints[:n]

        if action.gripper is not None:
            self._gripper_state = float(np.clip(action.gripper, 0.0, 1.0))

        obs = self.get_observation()
        if self.logger:
            self.logger.log_action(action, obs)
        return obs

    def move_ee(self, pose: np.ndarray, speed: float = 0.5) -> RobotObservation:
        """Move the end-effector to ``pose`` (validated against the workspace)."""
        return self.send_action(RobotAction(target_ee_pose=pose, speed=speed))

    def open_gripper(self) -> None:
        self._gripper_state = 1.0
        if self.logger:
            self.logger.log("open_gripper")

    def close_gripper(self) -> None:
        self._gripper_state = 0.0
        if self.logger:
            self.logger.log("close_gripper")

    # -- safety ---------------------------------------------------------
    def stop(self) -> None:
        if self.logger:
            self.logger.log("stop")

    def emergency_stop(self) -> None:
        if self.logger:
            self.logger.log("emergency_stop")

    # -- helpers --------------------------------------------------------
    def replay(self, trajectory_or_actions: Union[object, List[RobotAction]]) -> List[RobotObservation]:
        """Replay a :class:`FoldTrajectory` or a list of actions, returning obs."""
        actions = self._coerce_actions(trajectory_or_actions)
        return self.execute_trajectory(actions)

    @staticmethod
    def _coerce_actions(traj_or_actions) -> List[RobotAction]:
        if hasattr(traj_or_actions, "waypoints"):
            from terafold.planning.trajectory import trajectory_to_actions

            return trajectory_to_actions(traj_or_actions)
        return list(traj_or_actions)

    @property
    def position(self) -> np.ndarray:
        """Current end-effector xyz."""
        return self._ee_pose[:3].copy()
