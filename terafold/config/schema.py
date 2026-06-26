"""Typed, validated configuration schemas (pydantic v2).

These models are the single source of truth for what a valid TeraFold config
looks like. ``terafold.config.load`` reads YAML into these models and
``terafold.config.validate`` runs extra cross-field / physical-plausibility
checks on top of the field validators here.

Keeping validation strict but with sensible defaults means a minimal YAML still
produces a usable, *safe* config (dry-run on, conservative speeds).
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

Hint = Literal["very_low", "low", "medium", "high", "very_high"]
FoldDirection = Literal["right_to_left", "left_to_right", "top_to_bottom", "bottom_to_top"]


class WorkspaceConfig(BaseModel):
    """Metric workspace bounds in the table frame. Hard safety scaffolding."""

    frame: str = "table"
    x_min_m: float = 0.0
    x_max_m: float = 0.70
    y_min_m: float = 0.0
    y_max_m: float = 0.50
    z_min_m: float = 0.0
    z_max_m: float = 0.25

    @model_validator(mode="after")
    def _check_bounds(self) -> "WorkspaceConfig":
        if self.x_max_m <= self.x_min_m:
            raise ValueError("x_max_m must be > x_min_m")
        if self.y_max_m <= self.y_min_m:
            raise ValueError("y_max_m must be > y_min_m")
        if self.z_max_m <= self.z_min_m:
            raise ValueError("z_max_m must be > z_min_m")
        return self

    def contains_xy(self, x: float, y: float) -> bool:
        return self.x_min_m <= x <= self.x_max_m and self.y_min_m <= y <= self.y_max_m

    def contains_xyz(self, x: float, y: float, z: float) -> bool:
        return self.contains_xy(x, y) and self.z_min_m <= z <= self.z_max_m


class ClothConfig(BaseModel):
    type: str = "small_towel"
    expected_width_m: float = Field(0.30, gt=0)
    expected_height_m: float = Field(0.30, gt=0)
    thickness_m: float = Field(0.003, gt=0)
    stiffness_hint: Hint = "medium"
    friction_hint: Hint = "medium"
    allow_rotation_deg: float = Field(25.0, ge=0, le=180)


class FoldConfig(BaseModel):
    direction: FoldDirection = "right_to_left"
    crease: str = "vertical_center"
    grasp_strategy: str = "right_edge_midpoint"
    place_strategy: str = "reflect_across_crease"
    lift_height_m: float = Field(0.045, gt=0)
    arc_height_m: float = Field(0.080, gt=0)
    release_height_m: float = Field(0.012, ge=0)
    num_waypoints: int = Field(12, ge=4, le=200)
    slip_compensation_m: float = Field(0.010, ge=0)

    @model_validator(mode="after")
    def _check_heights(self) -> "FoldConfig":
        if self.arc_height_m < self.lift_height_m:
            raise ValueError("arc_height_m should be >= lift_height_m (arc clears the lift)")
        return self


class RobotLimitsConfig(BaseModel):
    """The robot limits embedded in a task config (kinematic + safety defaults)."""

    max_speed_mps: float = Field(0.04, gt=0, le=2.0)
    max_accel_mps2: float = Field(0.10, gt=0, le=10.0)
    dry_run_default: bool = True


class SuccessConfig(BaseModel):
    max_corner_error_m: float = Field(0.05, gt=0)
    min_overlap_ratio: float = Field(0.70, ge=0, le=1)
    max_wrinkle_score: float = Field(0.35, ge=0, le=1)
    min_visible_area_ratio: float = Field(0.45, ge=0, le=1)


class LearningConfig(BaseModel):
    use_keypoint_model: bool = True
    use_residual_model: bool = False
    residual_checkpoint: Optional[str] = None
    keypoint_checkpoint: Optional[str] = None
    success_checkpoint: Optional[str] = None


class TaskConfig(BaseModel):
    """Top-level task description (e.g. configs/task_fold_towel_half.yaml)."""

    task_name: str = "fold_towel_half_right_to_left"
    instruction: str = "fold the small towel in half from right to left"
    cloth: ClothConfig = Field(default_factory=ClothConfig)
    fold: FoldConfig = Field(default_factory=FoldConfig)
    workspace: WorkspaceConfig = Field(default_factory=WorkspaceConfig)
    robot: RobotLimitsConfig = Field(default_factory=RobotLimitsConfig)
    success: SuccessConfig = Field(default_factory=SuccessConfig)
    learning: LearningConfig = Field(default_factory=LearningConfig)

    model_config = {"extra": "forbid"}


# --------------------------------------------------------------------------
# Hardware-facing configs (robot + camera). Kept separate from the task so a
# task can run against any robot/camera combination.
# --------------------------------------------------------------------------


class RobotConfig(BaseModel):
    """Robot adapter configuration (configs/robot_*.yaml)."""

    type: Literal["mock", "learm", "so101", "bimanual"] = "mock"
    name: str = "mock_robot"
    dof: int = Field(6, ge=1, le=14)

    # Connection (hardware adapters). Left as placeholders for mock.
    port: Optional[str] = None
    baudrate: int = 115200
    protocol_file: Optional[str] = None  # required before LeArm can move for real

    # Kinematic / safety limits (may override the task's RobotLimitsConfig).
    max_speed_mps: float = Field(0.04, gt=0)
    max_accel_mps2: float = Field(0.10, gt=0)
    gripper_open_pos: float = 1.0
    gripper_close_pos: float = 0.0
    home_pose: Optional[List[float]] = None

    # Safety
    dry_run_default: bool = True
    workspace: WorkspaceConfig = Field(default_factory=WorkspaceConfig)
    forbidden_zones: List[List[float]] = Field(default_factory=list)  # [x0,y0,x1,y1]
    max_z_m: float = 0.25
    min_z_m: float = 0.0

    model_config = {"extra": "forbid"}


class CameraConfig(BaseModel):
    """Camera configuration (configs/camera_*.yaml)."""

    type: Literal["mock", "opencv", "webcam"] = "mock"
    index: int = 0
    width: int = Field(640, gt=0)
    height: int = Field(480, gt=0)
    fps: int = Field(30, gt=0)
    calibration_file: Optional[str] = None  # path to homography.json
    # For the mock camera: optionally render a synthetic towel scene.
    mock_render_cloth: bool = True
    flip_horizontal: bool = False
    flip_vertical: bool = False

    model_config = {"extra": "forbid"}


__all__ = [
    "Hint",
    "FoldDirection",
    "WorkspaceConfig",
    "ClothConfig",
    "FoldConfig",
    "RobotLimitsConfig",
    "SuccessConfig",
    "LearningConfig",
    "TaskConfig",
    "RobotConfig",
    "CameraConfig",
]
