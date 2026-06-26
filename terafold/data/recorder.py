"""Episode recorder + the end-to-end ``record_demo`` driver.

:class:`EpisodeRecorder` writes the on-disk format defined in
:mod:`terafold.data.episode_schema` (metadata, frames, observations, actions,
keypoints, plan, safety, result). :func:`record_demo` runs a *complete* mock
folding episode — camera -> plan -> execute on a robot -> record -> score — so a
demonstration can be captured TODAY with no hardware, entirely on numpy.

Image I/O goes through :mod:`terafold.vision.imageio` (never cv2/PIL directly).
Scoring is best-effort: it lazily tries :func:`terafold.eval.score_fold.score_fold`
and degrades to the plan's predicted success if scoring is unavailable.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

from terafold.data.episode_schema import (
    EpisodeMetadata,
    EpisodePaths,
    EpisodeResult,
    FailureMode,
    append_jsonl,
    next_episode_dir,
    write_json,
)
from terafold.robot.base import BaseRobot, RobotAction, RobotObservation
from terafold.vision.imageio import imwrite

__all__ = ["EpisodeRecorder", "record_demo"]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class EpisodeRecorder:
    """Incrementally writes one episode directory to disk.

    Usage::

        rec = EpisodeRecorder(task_root, metadata)
        rec.start()
        rec.set_plan(plan)
        idx = rec.record_step(image, observation, action, fold_state)
        rec.finish(EpisodeResult(success=True))
    """

    def __init__(self, task_root: str, metadata: EpisodeMetadata) -> None:
        self.task_root = task_root
        self.metadata = metadata
        self.paths: Optional[EpisodePaths] = None
        self._frame_index = 0
        self._notes: List[str] = []
        self._safety: Optional[dict] = None

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> EpisodePaths:
        """Allocate the next ``episode_NNNNNN`` dir and write initial metadata."""
        self.paths = next_episode_dir(self.task_root)
        self.paths.ensure_dirs()
        try:
            self.metadata.episode_index = int(
                os.path.basename(self.paths.root).split("_")[1]
            )
        except (IndexError, ValueError):
            self.metadata.episode_index = None
        if not self.metadata.start_time:
            self.metadata.start_time = _now_iso()
        self._write_metadata()
        return self.paths

    def _require_started(self) -> EpisodePaths:
        if self.paths is None:
            raise RuntimeError("EpisodeRecorder.start() must be called first.")
        return self.paths

    def _write_metadata(self) -> None:
        paths = self._require_started()
        self.metadata.num_frames = self._frame_index
        write_json(paths.metadata, self.metadata.to_dict())

    # -- per-step ----------------------------------------------------------
    def record_step(
        self,
        image: np.ndarray,
        observation: RobotObservation,
        action: Optional[RobotAction] = None,
        fold_state: Optional[object] = None,
    ) -> int:
        """Record one frame + observation (+ optional action / perception).

        Returns the integer frame index. The three jsonl streams stay aligned:
        every call appends exactly one line to each (``{}`` when absent).
        """
        paths = self._require_started()
        i = self._frame_index
        imwrite(paths.frame_path(i), np.asarray(image))
        append_jsonl(paths.observations, observation.to_dict())
        append_jsonl(paths.actions, action.to_dict() if action is not None else {})
        append_jsonl(
            paths.keypoints,
            fold_state.to_dict() if fold_state is not None else {},
        )
        self._frame_index += 1
        self.metadata.num_frames = self._frame_index
        return i

    # -- side artifacts ----------------------------------------------------
    def set_plan(self, plan: object) -> None:
        """Persist the :class:`FoldPlan` (or any ``to_dict``-able plan)."""
        paths = self._require_started()
        if hasattr(plan, "save"):
            plan.save(paths.plan)
        elif hasattr(plan, "to_dict"):
            write_json(paths.plan, plan.to_dict())
        else:
            write_json(paths.plan, dict(plan))  # type: ignore[arg-type]

    def set_safety(self, safety: Dict[str, Any]) -> None:
        paths = self._require_started()
        self._safety = dict(safety)
        write_json(paths.safety, self._safety)

    def add_note(self, note: str) -> None:
        self._notes.append(str(note))

    def finish(self, result: EpisodeResult) -> EpisodePaths:
        """Write result + notes, finalize metadata, return the episode paths."""
        paths = self._require_started()
        self.metadata.end_time = _now_iso()
        self.metadata.success = result.success
        self.metadata.failure_mode = result.failure_mode
        self._write_metadata()
        write_json(paths.result, result.to_dict())
        if self._notes:
            os.makedirs(paths.root, exist_ok=True)
            with open(paths.notes, "w") as f:
                f.write("\n".join(self._notes) + "\n")
        return paths


# --------------------------------------------------------------------------
# End-to-end mock demo recorder.
# --------------------------------------------------------------------------


def _robot_type(robot: BaseRobot) -> str:
    cfg = getattr(robot, "config", None)
    return getattr(cfg, "type", None) or type(robot).__name__


def _camera_type(camera) -> str:
    cfg = getattr(camera, "config", None)
    return getattr(cfg, "type", None) or type(camera).__name__


def _score_episode(
    plan: object,
    before_image: np.ndarray,
    after_image: np.ndarray,
    task,
    score: bool,
) -> EpisodeResult:
    """Build an :class:`EpisodeResult`, scoring via ``eval.score_fold`` if possible."""
    metrics: Dict[str, Any] = {}
    scorer: Dict[str, Any] = {}
    success: Optional[bool] = None

    if score:
        try:
            from terafold.eval.score_fold import score_fold

            res = score_fold(
                before_image,
                after_image,
                success_cfg=task.success,
                direction=task.direction,
            )
            metrics = {k: v for k, v in res.items()}
            success = bool(res.get("success")) if res.get("success") is not None else None
            scorer = {"method": "eval.score_fold"}
        except Exception as exc:  # eval.score_fold absent / failed -> degrade.
            scorer = {"method": "unavailable", "error": f"{type(exc).__name__}: {exc}"}

    if not metrics:
        metrics = {
            "predicted_success": float(getattr(plan, "predicted_success", 0.0)),
            "confidence": float(getattr(plan, "confidence", 0.0)),
        }
        scorer.setdefault("method", "plan.predicted_success")

    return EpisodeResult(
        success=success,
        failure_mode=FailureMode.NONE.value,
        metrics=metrics,
        scorer=scorer,
        notes="",
    )


def record_demo(
    task,
    robot: BaseRobot,
    camera,
    planner,
    frames=None,
    task_root: Optional[str] = None,
    operator: str = "",
    dry_run: bool = True,
    max_steps: Optional[int] = None,
    safety=None,
    score: bool = True,
) -> str:
    """Run and record a full folding episode end-to-end.

    Flow: connect camera + robot -> capture a frame -> perceive + plan with
    ``planner.plan_from_image`` -> convert the plan trajectory to robot actions
    -> execute each action, recording the frame/observation/action per step ->
    score the result -> finish. Designed to run on the mock camera + mock robot
    with no hardware. Returns the episode directory path.
    """
    from terafold.planning.trajectory import trajectory_to_actions

    if task_root is None:
        task_root = os.path.join("data", "episodes", task.name)

    metadata = EpisodeMetadata(
        task_name=task.name,
        task_instruction=task.instruction,
        robot_type=_robot_type(robot),
        camera_type=_camera_type(camera),
        start_time=_now_iso(),
        operator=operator or "unknown",
        cloth_type=getattr(task.cloth, "type", "small_towel"),
        dry_run=dry_run,
        terafold_version=_terafold_version(),
    )
    recorder = EpisodeRecorder(task_root, metadata)
    recorder.start()
    recorder.add_note(
        f"Mock demo recorded {_now_iso()} (dry_run={dry_run}, operator="
        f"{operator or 'unknown'})."
    )

    camera.connect()
    robot.connect()
    try:
        # 1. Perceive + plan from the first frame.
        first_frame = camera.read()
        before_image = np.asarray(first_frame.image)
        plan = planner.plan_from_image(before_image)
        recorder.set_plan(plan)

        # 2. Record the initial perception step (no action yet).
        obs0 = robot.get_observation()
        recorder.record_step(
            before_image, obs0, action=None, fold_state=plan.image_fold_state
        )

        # 3. Convert plan trajectory to robot actions.
        actions = trajectory_to_actions(plan.trajectory, frames=frames)
        if max_steps is not None:
            actions = actions[: int(max_steps)]

        # 4. Pre-flight safety check on the trajectory.
        violations: List[dict] = []
        if safety is not None:
            try:
                violations = safety.check_trajectory(plan.trajectory) or []
            except Exception as exc:
                recorder.add_note(f"Safety check error: {exc}")

        # 5. Execute, recording each step.
        after_image = before_image
        for action in actions:
            obs = robot.send_action(action)
            frame = camera.read()
            after_image = np.asarray(frame.image)
            recorder.record_step(after_image, obs, action=action, fold_state=None)

        recorder.set_safety(
            {
                "num_violations": len(violations),
                "violations": violations,
                "dry_run": dry_run,
                "checked": safety is not None,
            }
        )

        # 6. Score + finish.
        result = _score_episode(plan, before_image, after_image, task, score)
        recorder.add_note(
            f"Executed {len(actions)} actions over {recorder._frame_index} frames."
        )
        recorder.finish(result)
    except Exception:
        try:
            robot.emergency_stop()
        except Exception:
            pass
        recorder.finish(
            EpisodeResult(
                success=False,
                failure_mode=FailureMode.HUMAN_INTERRUPTED.value,
                notes="Episode aborted by exception.",
            )
        )
        raise
    finally:
        try:
            robot.disconnect()
        finally:
            camera.disconnect()

    assert recorder.paths is not None
    return recorder.paths.root


def _terafold_version() -> str:
    try:
        from terafold import __version__

        return __version__
    except Exception:
        return "0.0.1"
