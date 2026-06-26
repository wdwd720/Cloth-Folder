"""Closed-loop rollout of a folding policy behind the safety gate.

:class:`GeometryPolicy` wraps any planner exposing ``plan_from_image`` so the
classical geometric planner can act as a policy (the same interface a learned
ACT / SmolVLA policy would expose). :func:`safe_rollout` drives that policy on a
camera + robot, enforcing the full safety chain before any motion:

1. dry-run by default;
2. the two-flag human gate (:func:`require_motion_enabled`) for real motion;
3. the STOP file checked every step;
4. a workspace/trajectory validation pass before each fold is executed.

Pure numpy at import time; nothing here moves hardware on its own.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from terafold.planning.trajectory import trajectory_to_actions
from terafold.robot.base import BaseRobot
from terafold.robot.safety import (
    SafetyChecker,
    SafetyConfig,
    SafetyError,
    require_motion_enabled,
)

__all__ = ["GeometryPolicy", "safe_rollout"]


class GeometryPolicy:
    """Adapt a planner to a policy: ``act(image) -> FoldPlan``."""

    def __init__(self, planner: Any) -> None:
        self.planner = planner

    def act(self, image: np.ndarray):
        """Perceive + plan a fold for the given image, returning a ``FoldPlan``."""
        return self.planner.plan_from_image(image)


def safe_rollout(
    policy: Any,
    robot: BaseRobot,
    camera: Any,
    frames: Any = None,
    safety: Optional[SafetyChecker] = None,
    task: Any = None,
    dry_run: bool = True,
    enable_motion: bool = False,
    acknowledge: bool = False,
    max_steps: int = 1,
    recorder: Any = None,
) -> Dict[str, Any]:
    """Roll out ``policy`` on ``camera`` + ``robot`` under the safety gate.

    Returns a summary dict::

        {
          "dry_run": bool, "steps": int, "executed_actions": int,
          "stopped": bool, "stop_reason": str|None,
          "violations": [ ... per-step trajectory violations ... ],
          "plans": [ plan.summary() strings ],
        }

    Safety: when ``dry_run`` is False, :func:`require_motion_enabled` must pass
    (both ``enable_motion`` and ``acknowledge`` True) or a :class:`SafetyError`
    is raised before anything moves. A STOP file aborts at the next check.
    """
    if not dry_run:
        require_motion_enabled(enable_motion, acknowledge)

    if safety is None:
        cfg = SafetyConfig.from_task(task) if task is not None else SafetyConfig()
        safety = SafetyChecker(cfg)

    # Bring camera + robot up if needed (idempotent on the mock implementations).
    if not getattr(camera, "is_open", False):
        camera.connect()
    if not robot.is_connected:
        robot.connect()

    summary: Dict[str, Any] = {
        "dry_run": bool(dry_run),
        "steps": 0,
        "executed_actions": 0,
        "stopped": False,
        "stop_reason": None,
        "violations": [],
        "plans": [],
    }

    for step in range(int(max_steps)):
        if safety.is_stop_requested():
            summary["stopped"] = True
            summary["stop_reason"] = "STOP file present"
            break

        frame = camera.read()
        plan = policy.act(frame.image)
        summary["steps"] += 1
        try:
            summary["plans"].append(plan.summary())
        except Exception:
            summary["plans"].append(str(getattr(plan, "task_name", "plan")))

        # Validate the full trajectory in the workspace before executing it.
        violations = safety.check_trajectory(plan.trajectory)
        if violations:
            summary["violations"].extend(violations)
            summary["stopped"] = True
            summary["stop_reason"] = "trajectory workspace violation"
            break

        actions = trajectory_to_actions(plan.trajectory, frames=frames)
        aborted = False
        for action in actions:
            if safety.is_stop_requested():
                summary["stopped"] = True
                summary["stop_reason"] = "STOP file present"
                aborted = True
                break
            try:
                # Pose-in-workspace + STOP check. The per-step jump limit between
                # consecutive waypoints is already enforced by check_trajectory;
                # we deliberately don't pass current_pose here so the legitimate
                # home -> pregrasp approach isn't flagged as an over-large jump.
                safety.check_action(action)
                obs = robot.send_action(action)
            except SafetyError as exc:
                summary["violations"].append({"step": step, "reason": str(exc)})
                summary["stopped"] = True
                summary["stop_reason"] = str(exc)
                aborted = True
                break
            summary["executed_actions"] += 1
            if recorder is not None:
                try:
                    recorder.record_step(frame.image, obs, action=action)
                except Exception:
                    pass

        if aborted:
            break

    return summary
