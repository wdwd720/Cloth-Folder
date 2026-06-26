"""The fold planner — turns perceived cloth state into an executable plan.

Pipeline (Stages 1-3 of the TeraFold design):

1. Map perceived **image-frame** keypoints into the **table frame**:
   * Calibrated: use the homography ``H_IT``.
   * Uncalibrated: fall back to a similarity scaling derived from the cloth's
     *expected* metric size and place it at the workspace center. This is
     clearly flagged (``calibrated=False``, frame ``"table_uncalibrated"``) and
     is for dry-run / development only — never trust it for real motion.
2. Compute the crease (geometry), the grasp (strategy or learned keypoint), and
   the place (reflection of grasp across the crease).
3. Compute physics features (lift, slip, grasp stability) and build the
   pick-lift-arc-place trajectory.
4. Optionally apply a learned **residual correction** on top of the geometry.

The planner never assumes the fold succeeds: it emits a ``predicted_success``
and a ``confidence``, and surfaces every feature so the residual learner and the
success scorer can do their jobs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from terafold.math.geometry import normalize
from terafold.physics.cloth_state import (
    ClothKeypoints,
    FoldLine,
    FoldState,
    GraspPlacePair,
)
from terafold.physics.fold_geometry import (
    compute_fold_line,
    grasp_place_from_keypoints,
)
from terafold.physics.quasi_static import (
    estimate_grasp_stability,
    estimate_required_lift_height,
    estimate_arc_height,
    estimate_slip_compensation,
)
from terafold.physics.friction import estimate_table_friction, estimate_stiffness
from terafold.planning.fold_task import FoldTask
from terafold.planning.trajectory import FoldTrajectory, make_fold_arc_trajectory
from terafold.data.episode_schema import read_json, write_json

__all__ = ["FoldPlan", "plan_fold"]


@dataclass
class FoldPlan:
    """A complete, serializable fold plan."""

    task_name: str
    instruction: str
    frame: str  # "table" or "table_uncalibrated"
    calibrated: bool
    fold_state: FoldState  # table-frame perceived state
    grasp_place: GraspPlacePair
    fold_line: FoldLine
    trajectory: FoldTrajectory
    features: Dict[str, float] = field(default_factory=dict)
    residual_applied: bool = False
    confidence: float = 0.0
    predicted_success: float = 0.0
    image_fold_state: Optional[FoldState] = None  # original image-frame state
    metadata: Dict = field(default_factory=dict)

    def summary(self) -> str:
        g = self.grasp_place.grasp
        p = self.grasp_place.place
        lines = [
            f"FoldPlan[{self.task_name}] ({self.frame}, calibrated={self.calibrated})",
            f"  instruction : {self.instruction}",
            f"  grasp (m)   : ({g[0]:+.3f}, {g[1]:+.3f})",
            f"  place (m)   : ({p[0]:+.3f}, {p[1]:+.3f})",
            f"  lift / arc  : {self.features.get('lift_height_m', 0):.3f} / "
            f"{self.features.get('arc_height_m', 0):.3f} m",
            f"  grasp stab. : {self.features.get('grasp_stability', 0):.2f}",
            f"  waypoints   : {self.trajectory.num_waypoints} "
            f"(~{self.trajectory.duration:.1f}s)",
            f"  residual    : {'applied' if self.residual_applied else 'none'}",
            f"  confidence  : {self.confidence:.2f}   "
            f"predicted success: {self.predicted_success:.2f}",
        ]
        if not self.calibrated:
            lines.append("  WARNING     : uncalibrated table mapping — DRY-RUN ONLY.")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "task_name": self.task_name,
            "instruction": self.instruction,
            "frame": self.frame,
            "calibrated": self.calibrated,
            "fold_state": self.fold_state.to_dict(),
            "grasp_place": self.grasp_place.to_dict(),
            "fold_line": self.fold_line.to_dict(),
            "trajectory": self.trajectory.to_dict(),
            "features": self.features,
            "residual_applied": self.residual_applied,
            "confidence": self.confidence,
            "predicted_success": self.predicted_success,
            "image_fold_state": self.image_fold_state.to_dict()
            if self.image_fold_state
            else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FoldPlan":
        return cls(
            task_name=d["task_name"],
            instruction=d["instruction"],
            frame=d["frame"],
            calibrated=d["calibrated"],
            fold_state=FoldState.from_dict(d["fold_state"]),
            grasp_place=GraspPlacePair.from_dict(d["grasp_place"]),
            fold_line=FoldLine.from_dict(d["fold_line"]),
            trajectory=FoldTrajectory.from_dict(d["trajectory"]),
            features=d.get("features", {}),
            residual_applied=d.get("residual_applied", False),
            confidence=d.get("confidence", 0.0),
            predicted_success=d.get("predicted_success", 0.0),
            image_fold_state=FoldState.from_dict(d["image_fold_state"])
            if d.get("image_fold_state")
            else None,
            metadata=d.get("metadata", {}),
        )

    def save(self, path: str) -> None:
        write_json(path, self.to_dict())

    @classmethod
    def load(cls, path: str) -> "FoldPlan":
        return cls.from_dict(read_json(path))


def _build_pixel_to_table(
    kp_img: ClothKeypoints, task: FoldTask, frames
) -> Tuple[Callable[[np.ndarray], np.ndarray], str, bool]:
    """Return ``(to_table_fn, frame_label, calibrated)``.

    Calibrated path uses the homography; the uncalibrated fallback maps pixels to
    meters with a uniform scale from the cloth's expected size and centers the
    cloth in the workspace.
    """
    if frames is not None and getattr(frames, "image_calibrated", False):
        return (lambda uv: np.asarray(frames.pixel_to_table_xy(uv), dtype=np.float64),
                "table", True)

    # Uncalibrated similarity fallback.
    px_w = max(kp_img.width(), 1e-6)
    px_h = max(kp_img.height(), 1e-6)
    exp_w, exp_h = task.expected_size_m
    s = float(0.5 * (exp_w / px_w + exp_h / px_h))  # uniform scale (m/px)
    center_px = kp_img.center
    center_m = task.workspace_center

    def to_table(uv):
        uv = np.asarray(uv, dtype=np.float64).reshape(-1)[:2]
        return s * (uv - center_px) + center_m

    return to_table, "table_uncalibrated", False


def plan_fold(
    image_fold_state: FoldState,
    task: FoldTask,
    frames=None,
    residual_model: Optional[object] = None,
) -> FoldPlan:
    """Compute a fold plan from an image-frame perceived cloth state.

    Parameters
    ----------
    image_fold_state:
        Perceived cloth state in the image frame (from the keypoint detector or
        the classical fallback). Must have keypoints; mask is optional.
    task:
        The :class:`FoldTask`.
    frames:
        Optional calibrated :class:`~terafold.math.frames.FrameTransforms`.
    residual_model:
        Optional learned residual model (anything with a ``predict(features)``
        returning a residual correction, or a path handled by the caller). When
        ``None`` the plan is pure geometry + physics prior.
    """
    kp_img = image_fold_state.keypoints
    to_table, frame_label, calibrated = _build_pixel_to_table(kp_img, task, frames)

    # 1. Map corners (and any learned grasp/place) into the table frame.
    table_kp = ClothKeypoints(
        top_left=to_table(kp_img.top_left),
        top_right=to_table(kp_img.top_right),
        bottom_right=to_table(kp_img.bottom_right),
        bottom_left=to_table(kp_img.bottom_left),
        grasp=to_table(kp_img.grasp) if kp_img.grasp is not None else None,
        place=to_table(kp_img.place) if kp_img.place is not None else None,
    )

    direction = task.direction
    fold_line = compute_fold_line(table_kp, direction)

    # 2. Grasp / place. Prefer a learned grasp keypoint if the detector gave one.
    grasp_override = table_kp.grasp if table_kp.grasp is not None else None
    gp = grasp_place_from_keypoints(
        table_kp,
        direction=direction,
        grasp_strategy=task.fold.grasp_strategy,
        grasp_override=grasp_override,
    )
    grasp = gp.grasp.copy()
    place = gp.place.copy()

    # 3. Physics features + priors.
    cloth_size = (table_kp.width(), table_kp.height())
    mu = estimate_table_friction(task.cloth.friction_hint)
    stiff = estimate_stiffness(task.cloth.stiffness_hint)
    phys_lift = estimate_required_lift_height(
        cloth_size, task.cloth.stiffness_hint, task.cloth.friction_hint
    )
    phys_arc = estimate_arc_height(phys_lift, cloth_size)
    grasp_stability = estimate_grasp_stability(table_kp, grasp)

    # Slip compensation: direction from physics, magnitude from the task prior.
    slip_dir_vec = estimate_slip_compensation(grasp, place, None, mu)
    if np.linalg.norm(slip_dir_vec) > 1e-9:
        slip_comp = normalize(slip_dir_vec) * task.fold.slip_compensation_m
    else:
        slip_comp = np.zeros(2)

    # Geometric trajectory priors come from the task config (the tunable prior).
    lift_height = task.fold.lift_height_m
    arc_height = max(task.fold.arc_height_m, lift_height + 0.005)
    release_height = task.fold.release_height_m
    num_waypoints = task.fold.num_waypoints

    # 4. Optional learned residual correction (Stage 3).
    residual_applied = False
    confidence = float(np.clip(0.5 * table_kp.rectangularity() + 0.5 * grasp_stability, 0, 1))
    predicted_success = confidence

    features = {
        "cloth_width_m": cloth_size[0],
        "cloth_height_m": cloth_size[1],
        "cloth_diag_m": float(np.hypot(*cloth_size)),
        "rectangularity": float(table_kp.rectangularity()),
        "friction": mu,
        "stiffness": stiff,
        "physics_required_lift_m": phys_lift,
        "physics_arc_height_m": phys_arc,
        "grasp_stability": grasp_stability,
        "slip_comp_dx": float(slip_comp[0]),
        "slip_comp_dy": float(slip_comp[1]),
        "travel_distance_m": float(np.linalg.norm(place - grasp)),
        "lift_height_m": lift_height,
        "arc_height_m": arc_height,
        "release_height_m": release_height,
        "grasp_x": float(grasp[0]),
        "grasp_y": float(grasp[1]),
        "place_x": float(place[0]),
        "place_y": float(place[1]),
    }

    if residual_model is not None:
        # Lazy import so a broken/absent residual module never blocks geometry-only plans.
        from terafold.planning.residual_model import apply_residual_correction

        grasp, place, lift_height, arc_height, release_height, residual_info = (
            apply_residual_correction(
                residual_model, features, task, grasp, place,
                lift_height, arc_height, release_height,
            )
        )
        residual_applied = True
        confidence = float(residual_info.get("confidence", confidence))
        predicted_success = float(residual_info.get("predicted_success", predicted_success))
        features["residual"] = residual_info
        # Recompute slip + arc floor after correction.
        arc_height = max(arc_height, lift_height + 0.005)
        features.update(
            {"lift_height_m": lift_height, "arc_height_m": arc_height,
             "release_height_m": release_height,
             "grasp_x": float(grasp[0]), "grasp_y": float(grasp[1]),
             "place_x": float(place[0]), "place_y": float(place[1])}
        )

    # 5. Build the trajectory.
    trajectory = make_fold_arc_trajectory(
        grasp=grasp,
        place=place,
        crease_line=fold_line,
        lift_height=lift_height,
        arc_height=arc_height,
        num_waypoints=num_waypoints,
        release_height=release_height,
        slip_compensation=slip_comp,
        max_speed=task.robot_limits.max_speed_mps,
        max_accel=task.robot_limits.max_accel_mps2,
        workspace=task.workspace,
    )

    final_gp = GraspPlacePair(
        grasp=grasp, place=place, grasp_z=0.0, place_z=release_height,
        confidence=confidence,
    )
    table_state = FoldState(
        keypoints=table_kp,
        fold_line=fold_line,
        mask=image_fold_state.mask,
        grasp_place=final_gp,
        frame=frame_label,
        image_shape=image_fold_state.image_shape,
        metadata={"direction": direction},
    )

    return FoldPlan(
        task_name=task.name,
        instruction=task.instruction,
        frame=frame_label,
        calibrated=calibrated,
        fold_state=table_state,
        grasp_place=final_gp,
        fold_line=fold_line,
        trajectory=trajectory,
        features=features,
        residual_applied=residual_applied,
        confidence=confidence,
        predicted_success=predicted_success,
        image_fold_state=image_fold_state,
        metadata={"direction": direction, "slip_compensation": slip_comp.tolist()},
    )
