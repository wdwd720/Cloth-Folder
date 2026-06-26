"""Fold trajectory primitive: a physically-plausible pick-lift-arc-place motion.

A naive fold drags the cloth flat across the table, which bunches it. TeraFold
instead lifts the grasped flap to reduce table friction, swings it over the
crease along a smooth arc, and places it slightly short of the geometric target
to compensate for release slip. The motion is broken into labeled *phases* so
the recorder, the residual learner, and the safety layer can reason about each
segment.

All coordinates are in the **table frame** (meters, ``z = 0`` at the table
surface). :func:`trajectory_to_actions` converts a trajectory into robot
:class:`~terafold.robot.base.RobotAction` commands, mapping into the robot base
frame when a calibrated :class:`~terafold.math.frames.FrameTransforms` is given.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from terafold.math.geometry import as_vector, midpoint
from terafold.math.interpolation import linear_waypoints, quadratic_bezier, time_parameterize
from terafold.physics.cloth_state import FoldLine
from terafold.robot.base import RobotAction

__all__ = [
    "FoldTrajectory",
    "make_fold_arc_trajectory",
    "trajectory_to_actions",
    "clamp_point_to_workspace",
    "trajectory_workspace_violations",
    "FOLD_PHASES",
]

# Phase order of the fold primitive.
FOLD_PHASES = (
    "pregrasp",
    "descend",
    "close",
    "tension_lift",
    "lift",
    "arc",
    "place",
    "release",
    "retreat",
    "inspect",
)

GRIP_OPEN = 1.0
GRIP_CLOSED = 0.0


@dataclass
class FoldTrajectory:
    """A time-parameterized fold motion in the table frame."""

    waypoints: np.ndarray  # (N, 3) xyz
    gripper: np.ndarray  # (N,) opening in [0, 1]
    times: np.ndarray  # (N,) seconds, monotonically increasing from 0
    phases: List[str]  # length N
    frame: str = "table"
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.waypoints = np.asarray(self.waypoints, dtype=np.float64)
        self.gripper = np.asarray(self.gripper, dtype=np.float64)
        self.times = np.asarray(self.times, dtype=np.float64)
        n = self.waypoints.shape[0]
        if self.waypoints.shape[1] != 3:
            raise ValueError("waypoints must be (N, 3)")
        if not (len(self.gripper) == len(self.times) == len(self.phases) == n):
            raise ValueError("waypoints/gripper/times/phases length mismatch")

    @property
    def num_waypoints(self) -> int:
        return self.waypoints.shape[0]

    @property
    def duration(self) -> float:
        return float(self.times[-1] - self.times[0]) if self.num_waypoints else 0.0

    def max_z(self) -> float:
        return float(self.waypoints[:, 2].max())

    def min_z(self) -> float:
        return float(self.waypoints[:, 2].min())

    def segment_speeds(self) -> np.ndarray:
        """Per-segment Cartesian speed (m/s)."""
        d = np.linalg.norm(np.diff(self.waypoints, axis=0), axis=1)
        dt = np.diff(self.times)
        dt = np.where(dt <= 0, 1e-9, dt)
        return d / dt

    def phase_indices(self, phase: str) -> List[int]:
        return [i for i, p in enumerate(self.phases) if p == phase]

    def to_dict(self) -> dict:
        return {
            "frame": self.frame,
            "waypoints": self.waypoints.tolist(),
            "gripper": self.gripper.tolist(),
            "times": self.times.tolist(),
            "phases": list(self.phases),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FoldTrajectory":
        return cls(
            waypoints=np.asarray(d["waypoints"], dtype=np.float64),
            gripper=np.asarray(d["gripper"], dtype=np.float64),
            times=np.asarray(d["times"], dtype=np.float64),
            phases=list(d["phases"]),
            frame=d.get("frame", "table"),
            metadata=d.get("metadata", {}),
        )


def clamp_point_to_workspace(xyz: np.ndarray, workspace) -> Tuple[np.ndarray, bool]:
    """Clamp a table-frame xyz into workspace bounds. Returns ``(xyz, clamped)``."""
    xyz = np.asarray(xyz, dtype=np.float64).copy()
    clamped = False
    new_x = float(np.clip(xyz[0], workspace.x_min_m, workspace.x_max_m))
    new_y = float(np.clip(xyz[1], workspace.y_min_m, workspace.y_max_m))
    new_z = float(np.clip(xyz[2], workspace.z_min_m, workspace.z_max_m))
    if (new_x, new_y, new_z) != (xyz[0], xyz[1], xyz[2]):
        clamped = True
    return np.array([new_x, new_y, new_z]), clamped


def trajectory_workspace_violations(traj: FoldTrajectory, workspace) -> List[dict]:
    """List waypoints that fall outside the workspace (for safety reporting)."""
    violations = []
    for i, p in enumerate(traj.waypoints):
        if not workspace.contains_xyz(float(p[0]), float(p[1]), float(p[2])):
            violations.append({"index": i, "phase": traj.phases[i], "xyz": p.tolist()})
    return violations


def _seg(start, end, n, grip, phase, drop_first=True):
    """Build a linear sub-path of n points; optionally drop the duplicated first."""
    pts = linear_waypoints(start, end, n)
    grips = [grip] * len(pts)
    phases = [phase] * len(pts)
    if drop_first:
        return pts[1:], grips[1:], phases[1:]
    return pts, grips, phases


def make_fold_arc_trajectory(
    grasp: np.ndarray,
    place: np.ndarray,
    crease_line: Optional[FoldLine] = None,
    lift_height: float = 0.045,
    arc_height: float = 0.080,
    num_waypoints: int = 12,
    release_height: float = 0.012,
    slip_compensation: Optional[np.ndarray] = None,
    grasp_z: float = 0.0,
    pregrasp_height: Optional[float] = None,
    retreat_height: Optional[float] = None,
    max_speed: float = 0.04,
    max_accel: float = 0.10,
    workspace=None,
    grip_dwell_s: float = 0.6,
) -> FoldTrajectory:
    """Build the full pick-lift-arc-place fold trajectory in the table frame.

    Parameters
    ----------
    grasp, place:
        2D table-frame points (the place is the geometric reflection target).
    crease_line:
        The fold crease (used to place the arc apex over the crease). Optional;
        if absent the apex is taken above the grasp/place midpoint.
    lift_height:
        Height to raise the flap before the arc (reduces table friction).
    arc_height:
        Apex height of the swing arc (>= lift_height).
    num_waypoints:
        Number of samples along the arc (the dominant resolution knob).
    release_height:
        Height at which the cloth is released onto its target (slightly above
        the table to account for thickness / soft landing).
    slip_compensation:
        2D offset added to the place point (typically nudges *short* of target).

    The motion: pregrasp hover -> descend -> close gripper -> tension-check lift
    -> lift -> arc over crease -> place -> release -> retreat -> inspect.
    """
    grasp = as_vector(grasp)[:2]
    place = as_vector(place)[:2]
    if slip_compensation is not None:
        place = place + as_vector(slip_compensation)[:2]

    arc_height = max(arc_height, lift_height + 0.005)
    if pregrasp_height is None:
        pregrasp_height = max(lift_height + 0.04, 0.05)
    if retreat_height is None:
        retreat_height = max(lift_height + 0.04, arc_height)

    g3 = np.array([grasp[0], grasp[1], grasp_z])
    p3 = np.array([place[0], place[1], release_height])

    waypoints: List[np.ndarray] = []
    grips: List[float] = []
    phases: List[str] = []

    def add(pts, gr, ph):
        for p, g, h in zip(pts, gr, ph):
            waypoints.append(np.asarray(p, dtype=np.float64))
            grips.append(float(g))
            phases.append(h)

    # 1. pregrasp hover above grasp.
    add([[grasp[0], grasp[1], pregrasp_height]], [GRIP_OPEN], ["pregrasp"])
    # 2. descend to the cloth.
    add(*_seg([grasp[0], grasp[1], pregrasp_height], g3, 4, GRIP_OPEN, "descend"))
    # 3. close the gripper (dwell at grasp).
    add([g3, g3], [GRIP_CLOSED, GRIP_CLOSED], ["close", "close"])
    # 4. tension-check lift (small lift, watch for the cloth coming with us).
    tension_z = grasp_z + 0.012
    add(*_seg(g3, [grasp[0], grasp[1], tension_z], 3, GRIP_CLOSED, "tension_lift"))
    # 5. lift to the working height.
    add(*_seg([grasp[0], grasp[1], tension_z], [grasp[0], grasp[1], lift_height], 3,
              GRIP_CLOSED, "lift"))
    # 6. arc over the crease (quadratic Bezier in 3D; apex above the crease).
    if crease_line is not None:
        # Apex xy: project the grasp/place midpoint onto the crease for a clean swing.
        from terafold.math.geometry import project_point_onto_line

        apex_xy = project_point_onto_line(midpoint(grasp, place), crease_line.point,
                                          crease_line.direction)
    else:
        apex_xy = midpoint(grasp, place)
    p0 = np.array([grasp[0], grasp[1], lift_height])
    p1 = np.array([apex_xy[0], apex_xy[1], arc_height])
    p2 = p3
    arc = quadratic_bezier(p0, p1, p2, max(num_waypoints, 4))
    add(arc[1:], [GRIP_CLOSED] * (len(arc) - 1), ["arc"] * (len(arc) - 1))
    # 7. place dwell (settle before release).
    add([p3], [GRIP_CLOSED], ["place"])
    # 8. release (open the gripper).
    add([p3], [GRIP_OPEN], ["release"])
    # 9. retreat straight up.
    add(*_seg(p3, [place[0], place[1], retreat_height], 3, GRIP_OPEN, "retreat"))
    # 10. move to an inspection hover over the fold center.
    center = midpoint(grasp, place)
    add(*_seg([place[0], place[1], retreat_height],
              [center[0], center[1], retreat_height], 3, GRIP_OPEN, "inspect"))

    wp = np.array(waypoints)
    gr = np.array(grips)

    clamped_any = False
    if workspace is not None:
        for i in range(len(wp)):
            wp[i], c = clamp_point_to_workspace(wp[i], workspace)
            clamped_any = clamped_any or c

    # Time parameterization under speed/accel limits, then add gripper dwells.
    times = time_parameterize(wp, max_speed=max_speed, max_accel=max_accel)
    times = _add_gripper_dwells(times, phases, grip_dwell_s)

    meta = {
        "lift_height": lift_height,
        "arc_height": arc_height,
        "release_height": release_height,
        "num_arc_waypoints": int(max(num_waypoints, 4)),
        "slip_compensation": None if slip_compensation is None
        else as_vector(slip_compensation)[:2].tolist(),
        "max_speed": max_speed,
        "max_accel": max_accel,
        "clamped_to_workspace": clamped_any,
    }
    return FoldTrajectory(waypoints=wp, gripper=gr, times=times, phases=phases, metadata=meta)


def _add_gripper_dwells(times: np.ndarray, phases: List[str], dwell: float) -> np.ndarray:
    """Insert a fixed dwell after each gripper-change phase (close / release)."""
    times = times.copy()
    for i, ph in enumerate(phases):
        if ph in ("close", "release"):
            times[i:] += dwell
    return times


def trajectory_to_actions(
    traj: FoldTrajectory,
    frames=None,
    tool_orientation: Tuple[float, float, float] = (np.pi, 0.0, 0.0),
    max_speed: float = 0.04,
) -> List[RobotAction]:
    """Convert a table-frame trajectory into robot end-effector actions.

    If ``frames`` (a calibrated :class:`FrameTransforms`) is provided, table
    coordinates are mapped into the robot base frame; otherwise the table
    coordinates are passed through unchanged (valid for mock / dry-run only).
    The tool is held pointing down by default.
    """
    actions: List[RobotAction] = []
    roll, pitch, yaw = tool_orientation
    n = traj.num_waypoints
    for i in range(n):
        x, y, z = traj.waypoints[i]
        if frames is not None and getattr(frames, "robot_calibrated", False):
            rx, ry = frames.table_to_robot([x, y])
            rz = frames.table_z_to_robot_z(z)
        else:
            rx, ry, rz = x, y, z
        pose = np.array([rx, ry, rz, roll, pitch, yaw])
        dt = float(traj.times[i] - traj.times[i - 1]) if i > 0 else 0.5
        seg_speed = 0.0
        if i > 0 and dt > 0:
            seg_speed = float(np.linalg.norm(traj.waypoints[i] - traj.waypoints[i - 1]) / dt)
        speed_frac = float(np.clip(seg_speed / max(max_speed, 1e-6), 0.0, 1.0))
        actions.append(
            RobotAction(
                target_ee_pose=pose,
                gripper=float(traj.gripper[i]),
                speed=max(speed_frac, 0.05),
                duration=max(dt, 1e-3),
                metadata={"phase": traj.phases[i], "frame": "robot" if frames else "table"},
            )
        )
    return actions
