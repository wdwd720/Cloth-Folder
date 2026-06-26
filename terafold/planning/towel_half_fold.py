"""V0 task planner: ``fold_towel_half_right_to_left``.

This is the canonical entry point used by the ``plan-fold`` / ``dry-run-fold``
CLI commands. It composes perception (a keypoint predictor) with the geometric
+ physics planner (:func:`terafold.planning.fold_plan.plan_fold`) and the
optional learned residual.

Perception is *injected* (the predictor is passed in) so this module never hard-
imports torch/opencv. If no predictor is supplied, it lazily asks
:func:`terafold.vision.infer_keypoints.get_keypoint_predictor` for the best
available one (learned checkpoint if present, classical fallback otherwise).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from terafold.physics.cloth_state import FoldState
from terafold.planning.fold_plan import FoldPlan, plan_fold
from terafold.planning.fold_task import FoldTask

__all__ = ["TowelHalfFoldPlanner", "plan_towel_half_fold"]


class TowelHalfFoldPlanner:
    """Plan a right-to-left half fold from either an image or a perceived state."""

    def __init__(
        self,
        task: FoldTask,
        frames=None,
        keypoint_predictor=None,
        residual_model=None,
    ) -> None:
        self.task = task
        self.frames = frames
        self._predictor = keypoint_predictor
        self.residual_model = residual_model
        if task.direction not in ("right_to_left", "left_to_right"):
            # Not fatal — the generic planner supports all directions — but the
            # V0 task is specifically the horizontal half fold.
            self.task.fold.direction = "right_to_left"

    @property
    def predictor(self):
        if self._predictor is None:
            from terafold.vision.infer_keypoints import get_keypoint_predictor

            ckpt = self.task.learning.keypoint_checkpoint
            self._predictor = get_keypoint_predictor(
                checkpoint=ckpt if self.task.learning.use_keypoint_model else None
            )
        return self._predictor

    def perceive(self, image: np.ndarray) -> FoldState:
        """Run the keypoint predictor to get an image-frame cloth state."""
        return self.predictor.predict(image)

    def plan_from_state(self, image_fold_state: FoldState) -> FoldPlan:
        return plan_fold(
            image_fold_state, self.task, frames=self.frames,
            residual_model=self.residual_model,
        )

    def plan_from_image(self, image: np.ndarray) -> FoldPlan:
        return self.plan_from_state(self.perceive(image))


def plan_towel_half_fold(
    image_fold_state: FoldState,
    task: FoldTask,
    frames=None,
    residual_model: Optional[object] = None,
) -> FoldPlan:
    """Functional shortcut for the V0 half-fold plan from a perceived state."""
    return plan_fold(image_fold_state, task, frames=frames, residual_model=residual_model)
