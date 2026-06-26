"""Image + live demo pipelines for the towel half-fold.

Two entry points share one perception path (:mod:`terafold.perception`):

* :func:`run_demo_image` — perceive a still photo, plan, save overlay + JSON.
  Ideal when a phone-as-webcam is fiddly: just take a top-down photo.
* :func:`run_demo_today` — capture from a (phone/USB) camera, perceive, plan,
  dry-run on a robot, and record the episode.

Both run a perception ``mode`` of ``markers`` | ``model`` | ``claude`` and report
honestly what happened (markers detected vs. fallback, confidence, warnings).

Real physical motion is REFUSED unless every requirement in
:func:`evaluate_real_motion_gate` is met (trusted perception above threshold,
calibrated table mapping, a known adapter with a verified backend, workspace-
valid trajectory, a completed dry-run episode, and BOTH motion flags). When any
is missing, the overlay / JSON / plan are still saved.
"""

from __future__ import annotations

import math
import os
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from terafold.config.schema import CameraConfig
from terafold.data.episode_schema import EpisodeMetadata, EpisodeResult, FailureMode, write_json
from terafold.perception import TRUSTED_PERCEPTIONS, perceive
from terafold.planning.fold_plan import plan_fold
from terafold.planning.fold_task import FoldTask
from terafold.robot.base import RobotAction, RobotObservation
from terafold.robot.safety import (
    MotionWatchdog,
    SafetyChecker,
    SafetyConfig,
    SafetyError,
    require_motion_enabled,
)

__all__ = ["run_demo_today", "run_demo_image", "evaluate_real_motion_gate", "REAL_ROBOTS"]

REAL_ROBOTS = ("so101", "learm")
KNOWN_REAL_ROBOTS = ("so101", "learm")  # adapters that could drive hardware once a backend is wired
_ROBOT_CONFIGS = {
    "so101": "configs/robot_so101_template.yaml",
    "learm": "configs/robot_learm_template.yaml",
}


def _noop(_msg: str) -> None:  # pragma: no cover
    pass


def _mock_cam(mode: str, label: str):
    from terafold.camera.mock_camera import MockCamera

    return MockCamera(CameraConfig(type="mock"), image_size=256,
                      marker_mode=mode in ("markers", "auto")), label


def _build_camera(camera: str, camera_index: int, mode: str, log: Callable[[str], None]):
    """Build a camera; fall back to the marker-rendering mock if a webcam fails.

    The webcam index is first probed in an ISOLATED subprocess so a denied-camera
    permission abort (which Python cannot catch in-process) can't crash the demo.
    """
    if camera == "mock":
        return _mock_cam(mode, "mock")

    from terafold.camera.discovery import _probe_isolated

    probe = _probe_isolated(camera_index, None, warmup=2, min_blur=0.0, timeout=8.0)
    if not probe.opened:
        log(
            f"[warn] camera {camera_index} unavailable ({probe.error}); using the "
            "marker-rendering mock camera. Run `terafold list-cameras` to find a "
            "working index, and `pip install -e \".[vision]\"` for a real webcam."
        )
        return _mock_cam(mode, "mock(fallback)")
    try:
        from terafold.camera.opencv_camera import OpenCVCamera

        cam = OpenCVCamera(CameraConfig(type="opencv", index=camera_index))
        cam.connect()
        return cam, "opencv"
    except Exception as exc:
        log(f"[warn] could not open webcam {camera_index} ({exc}); using mock.")
        return _mock_cam(mode, "mock(fallback)")


def _build_robot(robot: str, task: FoldTask, dry: bool, log: Callable[[str], None]):
    if robot == "mock":
        from terafold.robot.mock_robot import MockRobot

        return MockRobot(workspace=task.workspace, dry_run=dry)
    if robot == "generic":
        from terafold.robot.generic_adapter import GenericArmAdapter

        return GenericArmAdapter(config=None, dry_run=dry)
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


def _approach_actions(start_xyz, target_pose, max_step, speed_frac) -> List[RobotAction]:
    """Slow, sub-``max_step`` interpolated approach from home to the first waypoint."""
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
            RobotAction(target_ee_pose=np.concatenate([p, orient]), gripper=1.0,
                        speed=speed_frac, duration=0.5, metadata={"phase": "approach"})
        )
    return actions


# --------------------------------------------------------------------------
# Real-motion gate (Part 9)
# --------------------------------------------------------------------------


def evaluate_real_motion_gate(
    *,
    perception,
    frames,
    plan,
    task,
    robot: str,
    confidence_threshold: float,
    enable_motion: bool,
    acknowledge: bool,
    dry_run_episode_done: bool,
    has_verified_backend: bool,
) -> Dict[str, Any]:
    """Return ``{allowed, missing}`` — real motion is allowed only when ALL pass."""
    missing: List[str] = []
    if not (enable_motion and acknowledge):
        missing.append("both --enable-motion and --i-understand-this-moves-hardware")
    if perception.fallback_used or perception.perception not in TRUSTED_PERCEPTIONS:
        missing.append(f"trusted perception (got '{perception.perception}', not a fallback)")
    if perception.confidence < confidence_threshold:
        missing.append(
            f"perception confidence >= {confidence_threshold:.2f} (got {perception.confidence:.2f})"
        )
    if frames is None or not getattr(frames, "image_calibrated", False):
        missing.append("a calibrated table mapping (run `terafold calibrate-table-from-image`)")
    if robot not in KNOWN_REAL_ROBOTS or not has_verified_backend:
        missing.append("a known robot adapter with a VERIFIED command backend (see robot-scan)")
    if SafetyChecker(SafetyConfig.from_task(task)).check_trajectory(plan.trajectory):
        missing.append("the planned trajectory fully inside the workspace")
    if not dry_run_episode_done:
        missing.append("a completed dry-run episode for this task")
    return {"allowed": len(missing) == 0, "missing": missing}


# --------------------------------------------------------------------------
# Shared result + overlay
# --------------------------------------------------------------------------


def _build_result(task, pr, plan, gate, frames) -> Dict[str, Any]:
    g, p = plan.grasp_place.grasp, plan.grasp_place.place
    return {
        "task": task.name,
        "instruction": task.instruction,
        "perception": pr.to_dict(),
        "calibrated": plan.calibrated,
        "frame": plan.frame,
        "dry_run_only": (not plan.calibrated) or pr.fallback_used,
        "keypoints": pr.fold_state.keypoints.to_dict(),
        "fold_line": plan.fold_line.to_dict(),
        "grasp": [float(g[0]), float(g[1])],
        "place": [float(p[0]), float(p[1])],
        "confidence": pr.confidence,
        "warnings": pr.warnings,
        "plan_summary": plan.summary(),
        "real_motion": gate,
        "plan": plan.to_dict(),
    }


def _save_overlay(image, plan, pr, frames, path, log=_noop) -> Optional[str]:
    try:
        from terafold.vision.visualization import draw_fold_plan, draw_keypoints, save_visualization

        vis = draw_fold_plan(np.asarray(image).copy(), plan, frames)
        vis = draw_keypoints(vis, pr.fold_state.keypoints)
        save_visualization(path, vis)
        return path
    except Exception as exc:  # overlay is best-effort
        log(f"[warn] could not save overlay: {exc}")
        return None


def _claude_labeler(mode: str, claude_responder):
    if mode != "claude":
        return None
    from terafold.vision.claude_labeler import ClaudeKeypointLabeler

    return ClaudeKeypointLabeler(responder=claude_responder) if claude_responder else None


# --------------------------------------------------------------------------
# demo-image (Part 2)
# --------------------------------------------------------------------------


def run_demo_image(
    image_path: str,
    mode: str = "markers",
    task_path: str = "configs/task_fold_towel_half.yaml",
    checkpoint: Optional[str] = None,
    calibration: Optional[str] = None,
    marker_map: Optional[str] = None,
    debug_markers: bool = False,
    overlay_out: Optional[str] = None,
    save_json: Optional[str] = None,
    confidence_threshold: float = 0.70,
    claude_responder=None,
    on_log: Callable[[str], None] = _noop,
) -> Dict[str, Any]:
    """Perceive a still image, plan the fold, and save an overlay + JSON result."""
    log = on_log or _noop
    from terafold.config.load import load_task_config
    from terafold.config.validate import validate_task_config
    from terafold.vision.imageio import imread
    from terafold.vision.marker_detector import parse_marker_map

    cfg = load_task_config(task_path)
    for w in validate_task_config(cfg):
        log(f"[config warning] {w}")
    task = FoldTask.from_config(cfg)

    frames = None
    if calibration:
        from terafold.camera.homography import load_homography

        frames = load_homography(calibration)

    image = imread(image_path)
    overlay_path = overlay_out or os.path.join("runs", "demo_image", "overlay.png")
    json_path = save_json or os.path.join("runs", "demo_image", "result.json")
    debug_dir = (
        os.path.join(os.path.dirname(overlay_path) or "runs/demo_image", "debug_markers")
        if debug_markers else None
    )

    mm = parse_marker_map(marker_map) if mode == "markers" else None
    pr = perceive(image, mode=mode, direction=task.direction, checkpoint=checkpoint,
                  marker_map=mm, debug_markers_dir=debug_dir,
                  claude_labeler=_claude_labeler(mode, claude_responder))
    plan = plan_fold(pr.fold_state, task, frames=frames)

    gate = evaluate_real_motion_gate(
        perception=pr, frames=frames, plan=plan, task=task, robot="generic",
        confidence_threshold=confidence_threshold, enable_motion=False, acknowledge=False,
        dry_run_episode_done=False, has_verified_backend=False,
    )

    _save_overlay(image, plan, pr, frames, overlay_path, log)
    result = _build_result(task, pr, plan, gate, frames)
    write_json(json_path, result)

    g, p = result["grasp"], result["place"]
    log(f"perception     : {pr.perception}")
    log(f"confidence     : {pr.confidence:.2f}")
    log(f"detected corners: {[[round(c[0],1), round(c[1],1)] for c in pr.detected_corners]}")
    log(f"grasp / place  : ({g[0]:.3f},{g[1]:.3f}) / ({p[0]:.3f},{p[1]:.3f})")
    log(f"calibrated     : {plan.calibrated}")
    log(f"DRY-RUN ONLY   : {result['dry_run_only']}")
    for w in pr.warnings:
        log(f"[warn] {w}")
    log(f"overlay -> {overlay_path}")
    log(f"json    -> {json_path}")
    log("next (dry-run execution):")
    log(f"  terafold dry-run-fold --task {task_path} --robot mock --camera mock")
    log(f"  terafold export-trajectory --plan-json {json_path} --out runs/demo_image/trajectory.csv")

    return {
        "status": "ok",
        "perception": pr.perception,
        "confidence": pr.confidence,
        "calibrated": plan.calibrated,
        "dry_run_only": result["dry_run_only"],
        "grasp": g, "place": p,
        "num_waypoints": plan.trajectory.num_waypoints,
        "overlay_path": overlay_path,
        "json_path": json_path,
        "real_motion": gate,
        "warnings": pr.warnings,
    }


# --------------------------------------------------------------------------
# demo-today (Part 3)
# --------------------------------------------------------------------------


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
    save_frame: Optional[str] = None,
    save_json: Optional[str] = None,
    dry_run: bool = False,
    enable_motion: bool = False,
    acknowledge: bool = False,
    confidence_threshold: float = 0.70,
    marker_map: Optional[str] = None,
    debug_markers: bool = False,
    operator: str = "demo",
    claude_responder=None,
    on_log: Callable[[str], None] = _noop,
) -> Dict[str, Any]:
    """Capture -> perceive -> plan -> (safe) dry-run / refuse real motion -> record."""
    log = on_log or _noop
    from terafold.config.load import load_task_config
    from terafold.config.validate import validate_task_config
    from terafold.data.episode_schema import list_episode_dirs
    from terafold.vision.marker_detector import parse_marker_map

    cfg = load_task_config(task_path)
    for w in validate_task_config(cfg):
        log(f"[config warning] {w}")
    task = FoldTask.from_config(cfg)

    requested_motion = (not dry_run) and (robot in REAL_ROBOTS) and enable_motion and acknowledge

    frames = None
    if calibration:
        from terafold.camera.homography import load_homography

        frames = load_homography(calibration)

    cam, camera_used = _build_camera(camera, camera_index, mode, log)
    summary: Dict[str, Any] = {
        "status": "ok", "task": task.name, "camera": camera_used, "robot": robot,
        "mode": mode, "requested_motion": requested_motion,
    }
    log(f"== demo-today: {task.name} ==  mode={mode} robot={robot} camera={camera_used}")

    try:
        if not getattr(cam, "is_open", False):
            cam.connect()
        frame0 = cam.read()
    except Exception as exc:
        log(f"[error] camera capture failed: {exc}")
        summary.update(status="error", error=f"camera: {exc}")
        return summary

    base_dir = os.path.join("runs", "demo_today", task.name)
    if save_frame:
        from terafold.vision.imageio import imwrite

        imwrite(save_frame, frame0.image)
        summary["frame_path"] = save_frame

    # Perception (explicit about fallbacks).
    mm = parse_marker_map(marker_map) if mode == "markers" else None
    debug_dir = os.path.join(base_dir, "debug_markers") if debug_markers else None
    pr = perceive(frame0.image, mode=mode, direction=task.direction, checkpoint=checkpoint,
                  marker_map=mm, debug_markers_dir=debug_dir,
                  claude_labeler=_claude_labeler(mode, claude_responder))
    log(f"   perception={pr.perception}  confidence={pr.confidence:.2f}  "
        f"missing_colors={pr.missing_colors}")
    for w in pr.warnings:
        log(f"[warn] {w}")

    plan = plan_fold(pr.fold_state, task, frames=frames)
    g, p = plan.grasp_place.grasp, plan.grasp_place.place
    log("\n" + plan.summary())

    overlay_path = overlay_out or os.path.join(base_dir, "overlay.png")
    if _save_overlay(frame0.image, plan, pr, frames, overlay_path, log):
        summary["overlay_path"] = overlay_path

    # Safety + real-motion gate.
    safety = SafetyChecker(SafetyConfig.from_task(task))
    violations = safety.check_trajectory(plan.trajectory)
    task_root = out or os.path.join("data", "episodes", f"{task.name}_demo")
    prior_eps = len(list_episode_dirs(task_root)) > 0
    gate = evaluate_real_motion_gate(
        perception=pr, frames=frames, plan=plan, task=task, robot=robot,
        confidence_threshold=confidence_threshold, enable_motion=enable_motion,
        acknowledge=acknowledge, dry_run_episode_done=prior_eps, has_verified_backend=False,
    )
    real_motion = requested_motion and gate["allowed"] and not violations
    refused = requested_motion and not real_motion
    dry = not real_motion
    execute = ((not requested_motion) or gate["allowed"]) and not violations

    if real_motion:
        require_motion_enabled(enable_motion, acknowledge)
    if refused:
        log("[refuse] Real motion requested but REFUSED. Missing requirements:")
        for m in gate["missing"]:
            log(f"   - {m}")
        log("   (overlay / JSON / plan are still saved.)")
    if violations:
        log(f"[abort] trajectory leaves the workspace: {violations[:2]} ...")

    # Result JSON.
    result = _build_result(task, pr, plan, gate, frames)
    result.update(requested_motion=requested_motion, real_motion=real_motion, camera=camera_used)
    json_path = save_json or os.path.join(base_dir, "result.json")
    write_json(json_path, result)
    summary["json_path"] = json_path

    # Episode recorder.
    from terafold.data.recorder import EpisodeRecorder

    meta = EpisodeMetadata(
        task_name=task.name, task_instruction=task.instruction,
        robot_type=f"{robot}{'' if real_motion else ' (dry-run)'}", camera_type=camera_used,
        start_time=datetime.now(timezone.utc).isoformat(), operator=operator,
        cloth_type=task.cloth.type, variation_tags=["demo-today", mode, pr.perception], dry_run=dry,
    )
    recorder = EpisodeRecorder(task_root, meta)
    recorder.start()
    recorder.set_plan(plan)
    recorder.set_safety({
        "dry_run": dry, "real_motion": real_motion, "requested_motion": requested_motion,
        "max_speed_mps": safety.config.max_speed_mps, "stop_file": safety.config.stop_file,
        "trajectory_violations": violations, "perception": pr.to_dict(),
        "real_motion_gate": gate,
    })

    executed = 0
    stopped = bool(violations or refused)
    stop_reason = (
        "trajectory workspace violation" if violations
        else ("real motion refused" if refused else None)
    )
    failure_mode = FailureMode.COLLISION_RISK.value if violations else FailureMode.NONE.value

    robot_obj = _build_robot(robot, task, dry, log)
    connected = False
    try:
        obs0 = robot_obj.get_observation() if hasattr(robot_obj, "get_observation") else None
    except Exception:
        obs0 = None
    recorder.record_step(frame0.image, obs0 or RobotObservation(timestamp=0.0),
                         action=None, fold_state=pr.fold_state)

    if execute:
        try:
            robot_obj.connect()
            connected = True
        except Exception as exc:
            log(f"[note] {robot} bring-up unavailable ({exc}). Plan + overlay + episode saved.")
            summary["robot_error"] = str(exc)
            stopped = True
            stop_reason = f"{robot} bring-up unavailable"

    if connected:
        from terafold.planning.trajectory import trajectory_to_actions

        fold_actions = trajectory_to_actions(
            plan.trajectory, frames=frames, max_speed=task.robot_limits.max_speed_mps
        )
        try:
            home = robot_obj.get_observation().ee_pose
            home_xyz = home[:3] if home is not None else np.zeros(3)
        except Exception:
            home_xyz = np.zeros(3)
        speed_frac = 0.1 if safety.config.slow_mode else 0.3
        approach = (
            _approach_actions(home_xyz, fold_actions[0].target_ee_pose, safety.config.max_step_m, speed_frac)
            if fold_actions and fold_actions[0].target_ee_pose is not None else []
        )
        all_actions = approach + fold_actions
        log(f"   executing {len(approach)} approach + {len(fold_actions)} fold waypoints "
            f"{'(REAL, slow)' if real_motion else '(DRY-RUN)'} ...")
        with MotionWatchdog(timeout_s=safety.config.timeout_s, on_stop=robot_obj.emergency_stop):
            for action in all_actions:
                if safety.is_stop_requested():
                    stopped, stop_reason = True, "STOP file present"
                    failure_mode = FailureMode.HUMAN_INTERRUPTED.value
                    break
                try:
                    safety.check_action(action)
                    obs = robot_obj.send_action(action)
                except SafetyError as exc:
                    stopped, stop_reason = True, str(exc)
                    failure_mode = FailureMode.COLLISION_RISK.value
                    log(f"[stop] safety: {exc}")
                    break
                step_img = cam.read().image if (real_motion and camera_used == "opencv") else frame0.image
                recorder.record_step(step_img, obs, action=action)
                executed += 1

    metrics: Dict[str, Any] = {"predicted_success": float(plan.predicted_success)}
    success: Optional[bool] = None
    if real_motion and connected and not stopped:
        try:
            from terafold.eval.score_fold import score_fold

            metrics = score_fold(frame0.image, cam.read().image, success_cfg=task.success, frames=frames)
            success = bool(metrics.get("success"))
        except Exception as exc:
            metrics["score_error"] = str(exc)
    else:
        metrics["note"] = "dry-run / not executed; metrics reflect the plan only."

    notes = f"demo-today {mode}/{pr.perception}; executed={executed}; dry={dry}; stopped={stopped}"
    recorder.add_note(notes)
    ep_dir = recorder.finish(
        EpisodeResult(success=success, failure_mode=failure_mode, metrics=metrics, notes=notes)
    ).root

    for closer in (robot_obj.disconnect, cam.disconnect):
        try:
            closer()
        except Exception:
            pass

    if violations:
        summary["status"] = "aborted"
    elif refused:
        summary["status"] = "real_motion_refused"
        summary["missing"] = gate["missing"]
    elif execute and not connected:
        summary["status"] = "real_robot_unavailable"
    else:
        summary["status"] = "ok"

    summary.update(
        perception=pr.perception, confidence=round(pr.confidence, 3), calibrated=plan.calibrated,
        dry_run=dry, real_motion=real_motion, grasp=[float(g[0]), float(g[1])],
        place=[float(p[0]), float(p[1])], num_waypoints=plan.trajectory.num_waypoints,
        episode_dir=ep_dir, executed_actions=executed, stopped=stopped, stop_reason=stop_reason,
        real_motion_gate=gate,
    )
    log(f"\n   episode -> {ep_dir}\n   executed={executed}, status={summary['status']}")
    if dry:
        log("   DRY-RUN. Real motion needs: " + "; ".join(gate["missing"]) if gate["missing"]
            else "   DRY-RUN complete.")
    return summary
