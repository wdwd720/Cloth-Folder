"""Cross-field / physical-plausibility checks on top of schema validation.

The schema models (``terafold.config.schema``) already enforce field types and
basic bounds. These functions return *advisory* warning strings about
configurations that are valid but likely wrong or unsafe (e.g. a tiny
workspace that cannot contain the cloth, speeds above the safe Stage-0 cap).
An empty list means "no concerns".
"""

from __future__ import annotations

from typing import List

from terafold.config.schema import CameraConfig, RobotConfig, TaskConfig

__all__ = [
    "validate_task_config",
    "validate_robot_config",
    "validate_camera_config",
]

# Conservative Stage-0 safety cap (matches RobotLimitsConfig defaults).
_SAFE_MAX_SPEED_MPS = 0.10
_SAFE_MAX_ACCEL_MPS2 = 0.50


def validate_task_config(cfg: TaskConfig) -> List[str]:
    """Return advisory warnings about a task config (empty == OK)."""
    warnings: List[str] = []

    ws = cfg.workspace
    cloth = cfg.cloth
    span_x = ws.x_max_m - ws.x_min_m
    span_y = ws.y_max_m - ws.y_min_m
    if cloth.expected_width_m > span_x or cloth.expected_height_m > span_y:
        warnings.append(
            f"cloth ({cloth.expected_width_m}x{cloth.expected_height_m} m) is larger than "
            f"workspace span ({span_x:.3f}x{span_y:.3f} m); folds may exit the workspace"
        )

    fold = cfg.fold
    if fold.arc_height_m > ws.z_max_m:
        warnings.append(
            f"fold.arc_height_m ({fold.arc_height_m}) exceeds workspace z_max_m ({ws.z_max_m})"
        )
    if fold.release_height_m > fold.lift_height_m:
        warnings.append(
            "fold.release_height_m is greater than lift_height_m (cloth released before it is lowered)"
        )

    robot = cfg.robot
    if robot.max_speed_mps > _SAFE_MAX_SPEED_MPS:
        warnings.append(
            f"robot.max_speed_mps ({robot.max_speed_mps}) exceeds the safe Stage-0 cap "
            f"({_SAFE_MAX_SPEED_MPS})"
        )
    if robot.max_accel_mps2 > _SAFE_MAX_ACCEL_MPS2:
        warnings.append(
            f"robot.max_accel_mps2 ({robot.max_accel_mps2}) exceeds the safe Stage-0 cap "
            f"({_SAFE_MAX_ACCEL_MPS2})"
        )
    if not robot.dry_run_default:
        warnings.append("robot.dry_run_default is False; real motion is enabled by default")

    learning = cfg.learning
    if learning.use_residual_model and not learning.residual_checkpoint:
        warnings.append("learning.use_residual_model is True but residual_checkpoint is null")
    if learning.use_keypoint_model and not learning.keypoint_checkpoint:
        warnings.append(
            "learning.use_keypoint_model is True but keypoint_checkpoint is null "
            "(planner will fall back to the classical predictor)"
        )

    return warnings


def validate_robot_config(cfg: RobotConfig) -> List[str]:
    """Return advisory warnings about a robot config (empty == OK)."""
    warnings: List[str] = []

    if cfg.max_z_m <= cfg.min_z_m:
        warnings.append("robot.max_z_m must be greater than min_z_m")
    if not (cfg.workspace.z_min_m <= cfg.min_z_m <= cfg.max_z_m <= cfg.workspace.z_max_m):
        warnings.append("robot [min_z_m, max_z_m] is not contained within the workspace z bounds")

    if cfg.max_speed_mps > _SAFE_MAX_SPEED_MPS:
        warnings.append(
            f"robot.max_speed_mps ({cfg.max_speed_mps}) exceeds the safe Stage-0 cap "
            f"({_SAFE_MAX_SPEED_MPS})"
        )
    if not cfg.dry_run_default:
        warnings.append("robot.dry_run_default is False; real motion is enabled by default")

    if cfg.type == "learm":
        if not cfg.protocol_file:
            warnings.append(
                "robot.type is 'learm' but protocol_file is null; real motion is blocked "
                "until a protocol file is provided"
            )
        if cfg.port is None and not cfg.dry_run_default:
            warnings.append("learm robot has no port set but dry_run_default is False")
    if cfg.type == "so101" and cfg.port is None and not cfg.dry_run_default:
        warnings.append("so101 robot has no port set but dry_run_default is False")

    if cfg.home_pose is not None and len(cfg.home_pose) not in (cfg.dof, 6):
        warnings.append(
            f"robot.home_pose length ({len(cfg.home_pose)}) does not match dof ({cfg.dof}) or a 6-DoF pose"
        )

    return warnings


def validate_camera_config(cfg: CameraConfig) -> List[str]:
    """Return advisory warnings about a camera config (empty == OK)."""
    warnings: List[str] = []

    if cfg.type in ("opencv", "webcam") and cfg.index < 0:
        warnings.append(f"camera.index ({cfg.index}) is negative for a real camera")
    if cfg.calibration_file is None:
        warnings.append(
            "camera.calibration_file is null; table/robot frames are uncalibrated "
            "(planning falls back to image-frame coordinates)"
        )
    if cfg.fps > 60:
        warnings.append(f"camera.fps ({cfg.fps}) is unusually high; many webcams cap at 30-60")

    return warnings
