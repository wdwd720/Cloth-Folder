"""Fold task abstraction — a thin, behavior-rich wrapper over a validated config.

Separating the *task* (what to do, success criteria, cloth/fold priors) from the
*planner* (how to compute motion) and the *robot* (how to execute) keeps each
swappable. A :class:`FoldTask` also exposes a small ``task_embedding`` so the
residual and policy models can condition on the task.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from terafold.config.schema import (
    ClothConfig,
    FoldConfig,
    LearningConfig,
    RobotLimitsConfig,
    SuccessConfig,
    TaskConfig,
    WorkspaceConfig,
)

__all__ = ["FoldTask", "FOLD_DIRECTIONS"]

FOLD_DIRECTIONS = ("right_to_left", "left_to_right", "top_to_bottom", "bottom_to_top")


@dataclass
class FoldTask:
    """Runtime view of a fold task."""

    name: str
    instruction: str
    cloth: ClothConfig
    fold: FoldConfig
    workspace: WorkspaceConfig
    success: SuccessConfig
    learning: LearningConfig
    robot_limits: RobotLimitsConfig

    @classmethod
    def from_config(cls, cfg: TaskConfig) -> "FoldTask":
        return cls(
            name=cfg.task_name,
            instruction=cfg.instruction,
            cloth=cfg.cloth,
            fold=cfg.fold,
            workspace=cfg.workspace,
            success=cfg.success,
            learning=cfg.learning,
            robot_limits=cfg.robot,
        )

    @property
    def direction(self) -> str:
        return self.fold.direction

    @property
    def expected_size_m(self) -> Tuple[float, float]:
        return (self.cloth.expected_width_m, self.cloth.expected_height_m)

    @property
    def workspace_center(self) -> np.ndarray:
        return np.array(
            [
                0.5 * (self.workspace.x_min_m + self.workspace.x_max_m),
                0.5 * (self.workspace.y_min_m + self.workspace.y_max_m),
            ]
        )

    def task_embedding(self) -> np.ndarray:
        """A small fixed-length task descriptor for conditioning learned models.

        Layout: [dir_onehot(4), expected_w, expected_h, thickness, allow_rot_deg/180].
        """
        onehot = np.zeros(len(FOLD_DIRECTIONS))
        if self.direction in FOLD_DIRECTIONS:
            onehot[FOLD_DIRECTIONS.index(self.direction)] = 1.0
        return np.concatenate(
            [
                onehot,
                [
                    self.cloth.expected_width_m,
                    self.cloth.expected_height_m,
                    self.cloth.thickness_m,
                    self.cloth.allow_rotation_deg / 180.0,
                ],
            ]
        )
