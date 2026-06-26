"""Safety scaffolding for all physical (and dry-run) robot motion.

This module is the single chokepoint every adapter and rollout must pass through
before commanding motion. It enforces, in order of importance:

1. **Dry-run by default** — nothing here moves hardware; it only *validates*.
2. **Two-flag human gate** — :func:`require_motion_enabled` demands BOTH an
   ``--enable-motion`` style flag and an explicit "I understand this moves
   hardware" acknowledgement before any real motion is permitted.
3. **A STOP file** — dropping a ``STOP_TERAFOLD`` file in the working directory
   aborts motion at the next check (a dead-simple panic button).
4. **Geometric envelopes** — workspace bounds, z-height limits, per-step jump
   size, and forbidden zones, all checked before a command is accepted.
5. **A watchdog** — a timeout / Ctrl+C context manager that trips a stop.

Pure python + numpy. No torch / serial / opencv.
"""

from __future__ import annotations

import os
import signal
import threading
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Union

import numpy as np

from terafold.config.schema import RobotConfig, WorkspaceConfig
from terafold.robot.base import RobotAction

__all__ = [
    "STOP_FILE",
    "SafetyError",
    "SafetyConfig",
    "SafetyChecker",
    "require_motion_enabled",
    "MotionWatchdog",
]

STOP_FILE = "STOP_TERAFOLD"


class SafetyError(Exception):
    """Raised whenever a command would violate a safety constraint."""


@dataclass
class SafetyConfig:
    """Tunable safety envelope. Conservative defaults (slow, small steps)."""

    workspace: WorkspaceConfig = field(default_factory=WorkspaceConfig)
    max_speed_mps: float = 0.04
    max_accel_mps2: float = 0.10
    min_z_m: float = 0.0
    max_z_m: float = 0.25
    forbidden_zones: List[List[float]] = field(default_factory=list)  # [x0,y0,x1,y1]
    max_step_m: float = 0.15
    stop_file: str = STOP_FILE
    timeout_s: float = 120.0
    slow_mode: bool = True

    @classmethod
    def from_task(cls, task) -> "SafetyConfig":
        """Build a config from a :class:`~terafold.planning.fold_task.FoldTask`."""
        ws = task.workspace
        return cls(
            workspace=ws,
            max_speed_mps=task.robot_limits.max_speed_mps,
            max_accel_mps2=task.robot_limits.max_accel_mps2,
            min_z_m=ws.z_min_m,
            max_z_m=ws.z_max_m,
        )

    @classmethod
    def from_robot_config(cls, robot_cfg: RobotConfig) -> "SafetyConfig":
        """Build a config from a :class:`~terafold.config.schema.RobotConfig`."""
        return cls(
            workspace=robot_cfg.workspace,
            max_speed_mps=robot_cfg.max_speed_mps,
            max_accel_mps2=robot_cfg.max_accel_mps2,
            min_z_m=robot_cfg.min_z_m,
            max_z_m=robot_cfg.max_z_m,
            forbidden_zones=[list(z) for z in robot_cfg.forbidden_zones],
        )


class SafetyChecker:
    """Validates poses, actions, and trajectories against a :class:`SafetyConfig`.

    All ``check_*`` methods raise :class:`SafetyError` on violation and return
    ``None`` when the command is safe (except :meth:`check_trajectory`, which is
    non-raising and returns a list of violations for reporting/planning).
    """

    def __init__(self, config: Optional[SafetyConfig] = None) -> None:
        self.config = config or SafetyConfig()

    # -- point / pose checks -------------------------------------------
    def check_xyz(self, xyz: Sequence[float]) -> None:
        """Validate a single table/robot-frame xyz point."""
        p = np.asarray(xyz, dtype=np.float64).reshape(-1)
        if p.shape[0] < 3:
            raise SafetyError(f"xyz must have 3 components, got {p.shape[0]}")
        x, y, z = float(p[0]), float(p[1]), float(p[2])
        ws = self.config.workspace
        if not ws.contains_xy(x, y):
            raise SafetyError(
                f"point ({x:.3f}, {y:.3f}) outside workspace "
                f"x[{ws.x_min_m}, {ws.x_max_m}] y[{ws.y_min_m}, {ws.y_max_m}]"
            )
        if not (self.config.min_z_m <= z <= self.config.max_z_m):
            raise SafetyError(
                f"z={z:.3f} outside z-limits "
                f"[{self.config.min_z_m}, {self.config.max_z_m}]"
            )
        for zone in self.config.forbidden_zones:
            x0, y0, x1, y1 = zone
            if min(x0, x1) <= x <= max(x0, x1) and min(y0, y1) <= y <= max(y0, y1):
                raise SafetyError(
                    f"point ({x:.3f}, {y:.3f}) inside forbidden zone {zone}"
                )

    def check_pose(self, pose6: Sequence[float]) -> None:
        """Validate a 6-DoF pose ``[x, y, z, roll, pitch, yaw]`` (xyz portion)."""
        p = np.asarray(pose6, dtype=np.float64).reshape(-1)
        if p.shape[0] < 3:
            raise SafetyError(f"pose must have >=3 components, got {p.shape[0]}")
        self.check_xyz(p[:3])

    def check_action(
        self, action: RobotAction, current_pose: Optional[Sequence[float]] = None
    ) -> None:
        """Validate a commanded action.

        Checks the target end-effector pose against the geometric envelope and,
        if ``current_pose`` is given, enforces the per-step jump limit.
        """
        if self.is_stop_requested():
            raise SafetyError(
                f"STOP requested (found '{self.config.stop_file}'); refusing to move"
            )
        if action.target_ee_pose is not None:
            target = np.asarray(action.target_ee_pose, dtype=np.float64).reshape(-1)
            self.check_pose(target)
            if current_pose is not None:
                cur = np.asarray(current_pose, dtype=np.float64).reshape(-1)
                step = float(np.linalg.norm(target[:3] - cur[:3]))
                if step > self.config.max_step_m:
                    raise SafetyError(
                        f"step {step:.3f} m exceeds max_step_m "
                        f"{self.config.max_step_m} m (jump too large)"
                    )

    def check_trajectory(
        self, traj_or_waypoints: Union[object, Sequence[Sequence[float]]]
    ) -> List[dict]:
        """Return a list of violations for a trajectory (empty if fully safe).

        Accepts a :class:`~terafold.planning.trajectory.FoldTrajectory` (anything
        exposing a ``waypoints`` (N,3) attribute) or a raw (N,3) array.
        """
        waypoints = getattr(traj_or_waypoints, "waypoints", traj_or_waypoints)
        wp = np.asarray(waypoints, dtype=np.float64)
        if wp.ndim != 2 or wp.shape[1] < 3:
            raise SafetyError("trajectory waypoints must be an (N, 3) array")
        phases = getattr(traj_or_waypoints, "phases", None)
        violations: List[dict] = []
        prev: Optional[np.ndarray] = None
        for i in range(wp.shape[0]):
            p = wp[i, :3]
            reasons: List[str] = []
            try:
                self.check_xyz(p)
            except SafetyError as e:
                reasons.append(str(e))
            if prev is not None:
                step = float(np.linalg.norm(p - prev))
                if step > self.config.max_step_m:
                    reasons.append(f"step {step:.3f} m > max_step_m {self.config.max_step_m}")
            if reasons:
                violations.append(
                    {
                        "index": i,
                        "phase": phases[i] if phases is not None else None,
                        "xyz": p.tolist(),
                        "reasons": reasons,
                    }
                )
            prev = p
        return violations

    # -- stop file ------------------------------------------------------
    def is_stop_requested(self, cwd: Optional[str] = None) -> bool:
        """True if a STOP file is present in ``cwd`` (defaults to current dir)."""
        root = cwd if cwd is not None else os.getcwd()
        return os.path.exists(os.path.join(root, self.config.stop_file))

    def assert_ok(self) -> None:
        """Raise if a stop has been requested. Cheap to call in a loop."""
        if self.is_stop_requested():
            raise SafetyError(
                f"STOP requested (found '{self.config.stop_file}'); aborting"
            )


def require_motion_enabled(enable_motion: bool, acknowledge: bool) -> None:
    """Hard gate for real hardware motion.

    Both flags must be ``True``. This enforces the two CLI flags
    ``--enable-motion`` AND ``--i-understand-this-moves-hardware``. Raises
    :class:`SafetyError` (with actionable guidance) unless both are set.
    """
    if not enable_motion or not acknowledge:
        raise SafetyError(
            "Real motion is blocked. To move hardware you must pass BOTH "
            "--enable-motion and --i-understand-this-moves-hardware "
            "(equivalently enable_motion=True AND acknowledge=True). "
            "Defaulting to dry-run."
        )


class MotionWatchdog:
    """Context manager that trips a stop on timeout or Ctrl+C.

    On entering, installs a SIGINT handler and a timeout timer. If either fires,
    ``on_stop`` is invoked (if provided), otherwise a STOP file is written. The
    previous SIGINT handler is restored on exit.

    Example
    -------
    >>> with MotionWatchdog(timeout_s=30, on_stop=robot.emergency_stop):
    ...     robot.execute_trajectory(actions)
    """

    def __init__(
        self,
        timeout_s: float,
        on_stop: Optional[Callable[[], None]] = None,
        stop_file: str = STOP_FILE,
    ) -> None:
        self.timeout_s = float(timeout_s)
        self.on_stop = on_stop
        self.stop_file = stop_file
        self.tripped = False
        self._timer: Optional[threading.Timer] = None
        self._prev_handler = None

    def _trip(self, reason: str) -> None:
        if self.tripped:
            return
        self.tripped = True
        if self.on_stop is not None:
            try:
                self.on_stop()
            except Exception:
                pass
        else:
            try:
                with open(self.stop_file, "w") as f:
                    f.write(f"STOP: {reason}\n")
            except Exception:
                pass

    def _handle_sigint(self, signum, frame) -> None:
        self._trip("SIGINT (Ctrl+C)")
        # Restore and re-raise so the user's Ctrl+C still propagates.
        if self._prev_handler not in (None, signal.SIG_DFL, signal.SIG_IGN):
            try:
                self._prev_handler(signum, frame)
                return
            except Exception:
                pass
        raise KeyboardInterrupt

    def __enter__(self) -> "MotionWatchdog":
        self.tripped = False
        if self.timeout_s > 0:
            self._timer = threading.Timer(self.timeout_s, self._trip, args=("timeout",))
            self._timer.daemon = True
            self._timer.start()
        try:
            self._prev_handler = signal.signal(signal.SIGINT, self._handle_sigint)
        except (ValueError, OSError):
            # Not on the main thread; signal handling unavailable. Timer still works.
            self._prev_handler = None
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        if self._prev_handler is not None:
            try:
                signal.signal(signal.SIGINT, self._prev_handler)
            except (ValueError, OSError):
                pass
            self._prev_handler = None
