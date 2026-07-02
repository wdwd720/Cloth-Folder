"""Structured episode schema: the bridge from a raw motion log to LeRobot.

Every real / dry-run interaction with the arm produces a *motion log* (a JSONL of
:class:`terafold.robot.command_logger.CommandLogger` events). That log is the
ground truth, but it is event-shaped, not learning-shaped. This module defines a
small, explicit, **learning-shaped** contract:

* :class:`EpisodeFrame` — one timestep: what was observed (joint positions,
  perception, images), what was commanded (joint targets / action vector), and the
  safety posture (dry-run, contact-enabled, success).
* :class:`Episode` — ``{meta, frames}`` with a :meth:`Episode.validate` sanity
  check.
* :func:`action_vector` — the **single source of truth** for action ordering:
  joint targets serialized into a vector ordered by *ascending servo id*. This must
  be stable so an exported dataset and a live policy agree on which slot is which
  servo.

Pure stdlib (dataclasses / typing). No numpy, no optional deps — so the recorder,
the exporter and the summary tool all share one definition.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

__all__ = [
    "EpisodeFrame",
    "Episode",
    "action_vector",
    "state_vector",
    "servo_ids_in_frame",
]


def _sorted_servo_ids(raw: Optional[Dict[Any, Any]]) -> List[int]:
    """Return the servo ids of a ``{servo_id: value}`` dict, ascending.

    Tolerates both ``int`` keys (in-memory) and ``str`` keys (after a JSON round
    trip). Keys that are not int-like are dropped (they cannot be a servo id).
    """
    if not raw:
        return []
    ids: List[int] = []
    for k in raw:
        try:
            ids.append(int(k))
        except (TypeError, ValueError):
            continue
    return sorted(ids)


def _get_by_id(raw: Optional[Dict[Any, Any]], sid: int, default: Any = None) -> Any:
    """Look up a servo id in a raw dict, trying both int and str keys."""
    if not raw:
        return default
    if sid in raw:
        return raw[sid]
    s = str(sid)
    if s in raw:
        return raw[s]
    return default


@dataclass
class EpisodeFrame:
    """One recorded timestep of a (real or dry-run) cloth-folding episode.

    Only :attr:`timestamp` is required; everything else is optional so a frame can
    be built from a partial event (e.g. a perception-only step has no action, a
    servo write has no image). ``*_raw`` dicts are keyed by **servo id**.
    """

    timestamp: float = 0.0

    episode_id: Optional[str] = None
    task: Optional[str] = None
    robot_type: Optional[str] = None
    robot_config: Optional[str] = None
    port: Optional[str] = None
    protocol: Optional[str] = None
    baudrate: Optional[int] = None

    servo_ids: List[int] = field(default_factory=list)
    joint_positions_raw: Dict[int, Any] = field(default_factory=dict)
    joint_targets_raw: Dict[int, Any] = field(default_factory=dict)
    joint_speeds_raw: Dict[int, Any] = field(default_factory=dict)

    action_raw: List[Any] = field(default_factory=list)
    action_normalized: Optional[List[Any]] = None

    image_paths: List[str] = field(default_factory=list)
    camera_intrinsics_id: Optional[str] = None
    table_homography_id: Optional[str] = None
    robot_table_transform_id: Optional[str] = None

    perception: Dict[str, Any] = field(default_factory=dict)
    towel_corners: Optional[Any] = None
    fold_line: Optional[Any] = None
    grasp_point: Optional[Any] = None
    place_point: Optional[Any] = None
    fold_direction: Optional[str] = None

    planned_trajectory: Optional[Any] = None
    ghost_motion_plan: Optional[Any] = None
    executed_command: Optional[Any] = None
    readbacks: Optional[Any] = None

    safety_flags: Dict[str, Any] = field(default_factory=dict)
    dry_run: bool = True
    contact_enabled: bool = False

    success: Optional[bool] = None
    failure_reason: Optional[str] = None
    operator_notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EpisodeFrame":
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in (d or {}).items() if k in known})


@dataclass
class Episode:
    """A full episode: descriptive ``meta`` plus an ordered list of frames."""

    meta: Dict[str, Any] = field(default_factory=dict)
    frames: List[EpisodeFrame] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"meta": dict(self.meta), "frames": [f.to_dict() for f in self.frames]}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Episode":
        d = d or {}
        frames = [EpisodeFrame.from_dict(x) for x in (d.get("frames") or [])]
        return cls(meta=dict(d.get("meta") or {}), frames=frames)

    def validate(self) -> List[str]:
        """Return a list of human-readable problems ([] means OK).

        Checks: the episode has at least one frame, and every frame that carries
        BOTH an explicit ``action_raw`` and ``joint_targets_raw`` has its action
        vector ordered by ascending servo id (the canonical ordering). Per-frame
        action dimensions may legitimately differ — a real fold commands different
        servo subsets at different waypoints — so that is NOT treated as an error
        (the exporter pads to the union of servo ids).
        """
        problems: List[str] = []
        if not self.frames:
            problems.append("episode has no frames")
            return problems

        for i, f in enumerate(self.frames):
            # If a frame carries BOTH an explicit action vector and joint targets,
            # the explicit vector must match the canonical servo-id ordering.
            if f.action_raw and f.joint_targets_raw:
                if list(f.action_raw) != action_vector(f):
                    problems.append(
                        f"frame {i}: action_raw is not ordered by ascending servo id"
                    )
        return problems


def servo_ids_in_frame(frame: EpisodeFrame) -> List[int]:
    """All servo ids referenced by a frame (targets ∪ positions), ascending."""
    ids = set(_sorted_servo_ids(frame.joint_targets_raw))
    ids |= set(_sorted_servo_ids(frame.joint_positions_raw))
    ids |= {int(s) for s in frame.servo_ids if str(s).lstrip("-").isdigit()}
    return sorted(ids)


def action_vector(frame: EpisodeFrame) -> List[Any]:
    """The action of a frame as a vector ordered by **ascending servo id**.

    This is the canonical, stable ordering used everywhere (export + policy). If
    the frame has ``joint_targets_raw`` they define the vector (sorted by id);
    otherwise the already-built ``action_raw`` is returned as-is.
    """
    jt = frame.joint_targets_raw or {}
    if jt:
        return [_get_by_id(jt, sid) for sid in _sorted_servo_ids(jt)]
    return list(frame.action_raw or [])


def state_vector(frame: EpisodeFrame) -> List[Any]:
    """The observed state of a frame ordered by ascending servo id.

    Mirrors :func:`action_vector` but for ``joint_positions_raw`` (the readbacks).
    """
    jp = frame.joint_positions_raw or {}
    return [_get_by_id(jp, sid) for sid in _sorted_servo_ids(jp)]
