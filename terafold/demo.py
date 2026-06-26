"""``demo-today`` — the end-to-end real-world towel half-fold demo pipeline.

One command ties the whole stack together for a *today* demo with one
SO-101/LeRobot arm, one top-down webcam, and a small towel with 4 colored corner
markers:

    camera -> marker (or keypoint) perception -> fold plan -> overlay ->
    safety-gated dry-run (default) / slow real execution -> episode recording.

Safety is layered and conservative by default:

* dry-run unless ``--robot so101 --enable-motion --i-understand-this-moves-hardware``,
* slow speed + workspace bounds (from the task config),
* a ``STOP_TERAFOLD`` stop-file and Ctrl+C emergency stop via
  :class:`~terafold.robot.safety.MotionWatchdog`,
* the planned trajectory is validated against the workspace before any motion,
* the run is recorded as an episode and an overlay image is saved.

This module is importable with numpy only; OpenCV is needed solely for a real
webcam (it degrades to the marker-rendering mock camera when unavailable).
"""

from __future__ import annotations

import math
import os
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from terafold.config.schema import CameraConfig
from terafold.data.episode_schema import EpisodeMetadata, EpisodeResult, FailureMode
from terafold.planning.fold_task import FoldTask
from terafold.robot.base import RobotAction, RobotObservation
from terafold.robot.safety import (
    MotionWatchdog,
    SafetyChecker,
    SafetyConfig,
    SafetyError,
    require_motion_enabled,
)

__all__ = ["run_demo_today", "REAL_ROBOTS"]

REAL_ROBOTS = ("so101", "learm")
_ROBOT_CONFIGS = {
    "so101": "configs/robot_so101_template.yaml",
    "learm": "configs/robot_learm_template.yaml",
}


def _noop(_msg: str) -> None:  # pragma: no cover
    pass


def _build_camera(camera: str, camera_index: int, mode: str, log: Callable[[str], None]):
    """Build a camera; fall back to the marker-rendering mock if a webcam fails."""
    marker_mode = mode in ("markers", "auto")
    if camera == "mock":
        from terafold.camera.mock_camera import MockCamera

        return MockCamera(CameraConfig(type="mock"), image_size=256, marker_mode=marker_mode), "mock"
    # Real webcam path.
    try:
        from terafold.camera.opencv_camera import OpenCVCamera

        cam = OpenCVCamera(CameraConfig(type="opencv", index=camera_index))
        cam.connect()
        return cam, "opencv"
    except Exception as exc:
        log(
            f"[warn] could not open webcam {camera_index} ({exc}); falling back to the "
            "marker-rendering mock camera. Install terafold[vision] + plug in a camera "
            "for the real demo."
        )
        from terafold.camera.mock_camera import MockCamera

        return MockCamera(CameraConfig(type="mock"), image_size=256, marker_mode=marker_mode), "mock(fallback)"


def _build_predictor(mode: str, checkpoint: Optional[str]):
    from terafold.vision.infer_keypoints import (
        MarkerKeypointPredictor,
        get_keypoint_predictor,
    )

    if mode == "keypoints":
        return get_keypoint_predictor(checkpoint)
    # markers / auto -> marker detector with classical fallback.
    return MarkerKeypointPredictor(fallback=True)


def _build_robot(robot: str, task: FoldTask, dry: bool, log: Callable[[str], None]):
    if robot == "mock":
        from terafold.robot.mock_robot import MockRobot

        return MockRobot(workspace=task.workspace, dry_run=dry)
    from terafold.config.load import load_robot_config

    cfg_path = _ROBOT_CONFIGS.get(robot)
    if cfg_path is None or not os.path.exists(cfg_path):
        raise ValueError(f"unknown robot {robot!r}")
    cfg = load_robot_config(cfg_path)
    if robot == "so101":
        from terafold.robot.so101_adapter import SO101Adapter

        return SO101Adapter(cfg, dry_run=dry)
    from terafold.robot.learm_adapter import LeArmAdapter

    return LeArmAdapter(cfg, dry_run=dry)


def _approach_actions(
    start_xyz: np.ndarray, target_pose: np.ndarray, max_step: float, speed_frac: float
) -> List[RobotAction]:
    """Slow, sub-``max_step`` interpolated approach from home to the first waypoint.

    Splitting the home->pregrasp move into small steps keeps every command under
    the per-step distance limit (which adapters like SO-101 enforce from their
    internal pose), and is the physically safe way to start.
    """
    start = np.asarray(start_xyz, dtype=np.float64).reshape(-1)[:3]
    tgt = np.asarray(target_pose, dtype=np.float64).reshape(-1)
    target_xyz = tgt[:3]
    orient = tgt[3:6] if tgt.shape[0] >= 6 else np.array([math.pi, 0.0, 0.0])
    dist = float(np.linalg.norm(target_xyz - start))
    n = max(1, int(math.ceil(dist / max(max_step * 0.8, 1e-3))))
    actions: List[RobotAction] = []
    for i in range(1, n + 1):
        p = start + (target_xyz - start) * (i / n)
        actions.append(
            RobotAction(
                target_ee_pose=np.concatenate([p, orient]),
                gripper=1.0,  # open while approaching
                speed=speed_frac,
                duration=0.5,
                metadata={"phase": "approach"},
            )
        )
    return actions


def run_demo_today(
    task_path: str = "configs/task_fold_towel_half.yaml",
    camera: str = "opencv",
    camera_index: int = 0,
    mode: str = "markers",
    robot: str = "mock",
    checkpoint: Optional[str] = None,
    calibration: Optional[str] = None,
    out: Optional[str] = None,
    overlay_out: Optional[str] = None,
    dry_run: bool = False,
    enable_motion: bool = False,
    acknowledge: bool = False,
    operator: str = "demo",
    on_log: Callable[[str], None] = _noop,
) -> Dict[str, Any]:
    """Run the towel half-fold demo end-to-end. Returns a structured summary.

    Real motion happens ONLY when ``robot`` is a real arm AND ``dry_run`` is
    False AND both ``enable_motion`` and ``acknowledge`` are True. Everything
    else is a safe dry-run.
    """
    log = on_log or _noop
    from terafold.config.load import load_task_config
    from terafold.config.validate import validate_task_config

    cfg = load_task_config(task_path)
    for w in validate_task_config(cfg):
        log(f"[config warning] {w}")
    task = FoldTask.from_config(cfg)

    # Resolve dry-run vs. real motion (safe by default).
    real_motion = (not dry_run) and (robot in REAL_ROBOTS) and enable_motion and acknowledge
    dry = not real_motion
    log(
        f"== demo-today: {task.name} ==\n"
        f"   mode={mode} robot={robot} camera={camera}[{camera_index}] "
        f"{'REAL MOTION' if real_motion else 'DRY-RUN'}"
    )

    # Frames (optional homography for metric planning).
    frames = None
    if calibration:
        from terafold.camera.homography import load_homography

        frames = load_homography(calibration)

    # Perception backend + camera.
    predictor = _build_predictor(mode, checkpoint)
    cam, camera_used = _build_camera(camera, camera_index, mode, log)

    summary: Dict[str, Any] = {
        "status": "ok",
        "task": task.name,
        "dry_run": dry,
        "real_motion": real_motion,
        "camera": camera_used,
        "robot": robot,
        "mode": mode,
    }

    try:
        if not getattr(cam, "is_open", False):
            cam.connect()
        frame0 = cam.read()
    except Exception as exc:
        log(f"[error] camera capture failed: {exc}")
        summary.update(status="error", error=f"camera: {exc}")
        return summary

    # Perceive + plan.
    fold_state = predictor.predict(frame0.image)
    perception = (fold_state.metadata or {}).get("perception", getattr(predictor, "name", "unknown"))
    from terafold.planning.towel_half_fold import TowelHalfFoldPlanner

    planner = TowelHalfFoldPlanner(task, frames=frames, keypoint_predictor=predictor)
    plan = planner.plan_from_state(fold_state)

    g, p = plan.grasp_place.grasp, plan.grasp_place.place
    summary.update(
        perception=perception,
        calibrated=plan.calibrated,
        grasp=[float(g[0]), float(g[1])],
        place=[float(p[0]), float(p[1])],
        num_waypoints=plan.trajectory.num_waypoints,
        duration_s=round(plan.trajectory.duration, 2),
        confidence=round(plan.confidence, 3),
    )
    log("\n" + plan.summary())
    log(
        f"   perception={perception}  grasp(m)=({g[0]:+.3f},{g[1]:+.3f})  "
        f"place(m)=({p[0]:+.3f},{p[1]:+.3f})  waypoints={plan.trajectory.num_waypoints}"
    )

    # Save overlay image.
    overlay_path = overlay_out or os.path.join("runs", "demo_today", task.name, "overlay.png")
    try:
        from terafold.vision.visualization import (
            draw_fold_plan,
            draw_keypoints,
            save_visualization,
        )

        vis = draw_fold_plan(frame0.image.copy(), plan, frames)
        vis = draw_keypoints(vis, fold_state.keypoints)
        save_visualization(overlay_path, vis)
        summary["overlay_path"] = overlay_path
        log(f"   overlay -> {overlay_path}")
    except Exception as exc:  # overlay is best-effort
        log(f"[warn] could not save overlay: {exc}")

    # Safety envelope (slow + workspace + stop-file + watchdog).
    safety = SafetyChecker(SafetyConfig.from_task(task))
    if real_motion:
        require_motion_enabled(enable_motion, acknowledge)  # raises unless both True

    violations = safety.check_trajectory(plan.trajectory)
    if violations:
        log(f"[abort] trajectory leaves the workspace: {violations[:2]} ...")
        summary.update(status="aborted", stop_reason="trajectory workspace violation",
                       violations=violations)
        # still record the (un-executed) plan below with no actions.

    # Episode recorder.
    from terafold.data.recorder import EpisodeRecorder

    task_root = out or os.path.join("data", "episodes", f"{task.name}_demo")
    meta = EpisodeMetadata(
        task_name=task.name,
        task_instruction=task.instruction,
        robot_type=f"{robot}{'' if real_motion else ' (dry-run)'}",
        camera_type=camera_used,
        start_time=datetime.now(timezone.utc).isoformat(),
        operator=operator,
        cloth_type=task.cloth.type,
        variation_tags=["demo-today", mode, perception],
        dry_run=dry,
    )
    recorder = EpisodeRecorder(task_root, meta)
    recorder.start()
    recorder.set_plan(plan)
    recorder.set_safety(
        {
            "dry_run": dry,
            "real_motion": real_motion,
            "max_speed_mps": safety.config.max_speed_mps,
            "workspace": [
                task.workspace.x_min_m, task.workspace.x_max_m,
                task.workspace.y_min_m, task.workspace.y_max_m,
                task.workspace.z_min_m, task.workspace.z_max_m,
            ],
            "stop_file": safety.config.stop_file,
            "trajectory_violations": violations,
        }
    )

    executed = 0
    stopped = bool(violations)
    stop_reason = "trajectory workspace violation" if violations else None
    failure_mode = FailureMode.COLLISION_RISK.value if violations else FailureMode.NONE.value

    robot_obj = _build_robot(robot, task, dry, log)
    connected = False
    if not violations:
        try:
            robot_obj.connect()
            connected = True
        except Exception as exc:
            # Real bring-up not wired (e.g. SO-101 via lerobot): keep the dry plan
            # + overlay + episode, and report honestly. No motion happened.
            log(
                f"[note] {robot} real bring-up unavailable ({exc}). The plan and "
                "overlay are saved; complete the adapter to execute on hardware."
            )
            summary.update(status="real_robot_unavailable", robot_error=str(exc))
            stopped = True
            stop_reason = f"{robot} bring-up unavailable"

    if connected:
        # Build approach (home -> first waypoint) + the fold trajectory actions.
        from terafold.planning.trajectory import trajectory_to_actions

        fold_actions = trajectory_to_actions(
            plan.trajectory, frames=frames, max_speed=task.robot_limits.max_speed_mps
        )
        try:
            home = robot_obj.get_observation().ee_pose
            home_xyz = home[:3] if home is not None else np.zeros(3)
        except Exception:
            home_xyz = np.zeros(3)
        speed_frac = 0.1 if safety.config.slow_mode else 0.3  # slow by default
        approach = (
            _approach_actions(home_xyz, fold_actions[0].target_ee_pose,
                              safety.config.max_step_m, speed_frac)
            if fold_actions and fold_actions[0].target_ee_pose is not None
            else []
        )
        all_actions = approach + fold_actions
        log(f"   executing {len(approach)} approach + {len(fold_actions)} fold "
            f"waypoints {'(DRY-RUN)' if dry else '(REAL, slow)'} ...")

        try:
            obs0 = robot_obj.get_observation()
            recorder.record_step(frame0.image, obs0, action=None, fold_state=fold_state)
        except Exception:
            recorder.record_step(
                frame0.image, RobotObservation(timestamp=0.0), action=None, fold_state=fold_state
            )

        with MotionWatchdog(timeout_s=safety.config.timeout_s, on_stop=robot_obj.emergency_stop):
            for action in all_actions:
                if safety.is_stop_requested():
                    stopped = True
                    stop_reason = "STOP file present"
                    failure_mode = FailureMode.HUMAN_INTERRUPTED.value
                    break
                try:
                    safety.check_action(action)  # pose-in-workspace (no home-jump flag)
                    obs = robot_obj.send_action(action)
                except SafetyError as exc:
                    stopped = True
                    stop_reason = str(exc)
                    failure_mode = FailureMode.COLLISION_RISK.value
                    log(f"[stop] safety: {exc}")
                    break
                # Recapture the scene for a real run; reuse frame0 in dry-run.
                step_img = cam.read().image if (real_motion and camera_used == "opencv") else frame0.image
                recorder.record_step(step_img, obs, action=action)
                executed += 1

    # Outcome + optional scoring.
    metrics: Dict[str, Any] = {"predicted_success": float(plan.predicted_success)}
    success: Optional[bool] = None
    if real_motion and connected and not stopped:
        try:
            from terafold.eval.score_fold import score_fold

            after = cam.read().image
            metrics = score_fold(frame0.image, after, predictor=predictor,
                                 success_cfg=task.success, frames=frames)
            success = bool(metrics.get("success"))
        except Exception as exc:  # scoring is best-effort
            metrics["score_error"] = str(exc)
    else:
        metrics["note"] = "dry-run: no physical fold; metrics reflect the plan only."

    notes = (
        f"demo-today {mode}/{perception}; executed={executed}; "
        f"{'real' if real_motion else 'dry-run'}; stopped={stopped} ({stop_reason})"
    )
    recorder.add_note(notes)
    ep_dir = recorder.finish(
        EpisodeResult(success=success, failure_mode=failure_mode, metrics=metrics, notes=notes)
    ).root

    try:
        robot_obj.disconnect()
    except Exception:
        pass
    try:
        cam.disconnect()
    except Exception:
        pass

    summary.update(
        episode_dir=ep_dir,
        executed_actions=executed,
        stopped=stopped,
        stop_reason=stop_reason,
        failure_mode=failure_mode,
        metrics=metrics,
        success=success,
    )
    log(
        f"\n   episode -> {ep_dir}\n"
        f"   executed={executed} actions, stopped={stopped}"
        + (f" ({stop_reason})" if stop_reason else "")
    )
    if dry:
        log("   DRY-RUN complete. For real motion: --robot so101 --enable-motion "
            "--i-understand-this-moves-hardware")
    return summary
