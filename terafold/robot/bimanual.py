"""Bimanual robot — a coordinator over two :class:`BaseRobot` arms.

Two single-arm adapters are composed into one logical robot. Observations from
both arms are merged; actions are dispatched per-arm. True synchronized dual-arm
folding (e.g. grabbing two corners and folding together) is a V4 roadmap item
and is intentionally left as a guarded ``NotImplementedError``. Numpy only.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from terafold.config.schema import WorkspaceConfig
from terafold.robot.base import BaseRobot, RobotAction, RobotObservation
from terafold.robot.safety import SafetyChecker, SafetyConfig

__all__ = ["BimanualRobot"]

ActionPair = Union[Dict[str, RobotAction], Tuple[Optional[RobotAction], Optional[RobotAction]]]


class BimanualRobot(BaseRobot):
    """Coordinate a ``left`` and ``right`` arm as a single robot."""

    def __init__(
        self,
        left: BaseRobot,
        right: BaseRobot,
        shared_workspace: Optional[WorkspaceConfig] = None,
        safety: Optional[SafetyChecker] = None,
    ) -> None:
        # Bimanual is dry-run unless BOTH arms are live.
        super().__init__(dry_run=bool(left.dry_run or right.dry_run))
        self.left = left
        self.right = right
        self.shared_workspace = shared_workspace
        if safety is not None:
            self.safety = safety
        elif shared_workspace is not None:
            self.safety = SafetyChecker(
                SafetyConfig(
                    workspace=shared_workspace,
                    min_z_m=shared_workspace.z_min_m,
                    max_z_m=shared_workspace.z_max_m,
                )
            )
        else:
            self.safety = None

    # -- lifecycle ------------------------------------------------------
    def connect(self) -> None:
        self.left.connect()
        self.right.connect()
        self._connected = self.left.is_connected and self.right.is_connected

    def disconnect(self) -> None:
        self.left.disconnect()
        self.right.disconnect()
        self._connected = False

    # -- io -------------------------------------------------------------
    def get_observation(self) -> RobotObservation:
        """Merge both arms' observations into one (joints/ee concatenated)."""
        lo = self.left.get_observation()
        ro = self.right.get_observation()

        def _cat(a, b):
            if a is None and b is None:
                return None
            a = np.zeros(0) if a is None else np.asarray(a, dtype=np.float64).reshape(-1)
            b = np.zeros(0) if b is None else np.asarray(b, dtype=np.float64).reshape(-1)
            return np.concatenate([a, b])

        return RobotObservation(
            timestamp=time.time(),
            joint_positions=_cat(lo.joint_positions, ro.joint_positions),
            joint_velocities=_cat(lo.joint_velocities, ro.joint_velocities),
            gripper_state=0.5 * (float(lo.gripper_state) + float(ro.gripper_state)),
            ee_pose=_cat(lo.ee_pose, ro.ee_pose),
            raw_vendor_state={"left": lo.to_dict(), "right": ro.to_dict()},
        )

    @staticmethod
    def _split(action: ActionPair) -> Tuple[Optional[RobotAction], Optional[RobotAction]]:
        if isinstance(action, dict):
            return action.get("left"), action.get("right")
        if isinstance(action, (tuple, list)) and len(action) == 2:
            return action[0], action[1]
        raise TypeError(
            "BimanualRobot.send_action expects a dict {'left':..,'right':..} "
            "or a (left, right) tuple of RobotAction"
        )

    def send_action(self, action: ActionPair) -> RobotObservation:
        """Dispatch a (left, right) action pair and return merged observation."""
        la, ra = self._split(action)
        if la is not None:
            self.left.send_action(la)
        if ra is not None:
            self.right.send_action(ra)
        return self.get_observation()

    def execute_pair(
        self,
        left_actions: List[RobotAction],
        right_actions: List[RobotAction],
    ) -> List[RobotObservation]:
        """Execute two action sequences synchronized step-by-step."""
        observations: List[RobotObservation] = []
        n = max(len(left_actions), len(right_actions))
        for i in range(n):
            if i < len(left_actions):
                self.left.send_action(left_actions[i])
            if i < len(right_actions):
                self.right.send_action(right_actions[i])
            observations.append(self.get_observation())
        return observations

    def move_ee(self, pose: np.ndarray, speed: float = 0.5) -> RobotObservation:
        """Move both arms to the same pose (rarely useful; provided for the API)."""
        self.left.move_ee(pose, speed=speed)
        self.right.move_ee(pose, speed=speed)
        return self.get_observation()

    def open_gripper(self) -> None:
        self.left.open_gripper()
        self.right.open_gripper()

    def close_gripper(self) -> None:
        self.left.close_gripper()
        self.right.close_gripper()

    # -- safety ---------------------------------------------------------
    def stop(self) -> None:
        self.left.stop()
        self.right.stop()

    def emergency_stop(self) -> None:
        self.left.emergency_stop()
        self.right.emergency_stop()

    # -- coordinated folding (future) -----------------------------------
    def dual_arm_fold(self, *args, **kwargs) -> None:
        """Synchronized two-corner fold — placeholder.

        Coordinated dual-arm cloth manipulation is a V4 roadmap feature.
        """
        raise NotImplementedError("dual_arm_fold is a V4 roadmap feature")
