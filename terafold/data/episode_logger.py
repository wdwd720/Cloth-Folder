"""Turn a raw motion log (CommandLogger JSONL) into a structured :class:`Episode`.

The robot adapters log every command they *intend* to send via
:class:`terafold.robot.command_logger.CommandLogger` /
:func:`terafold.robot.real_motion.motion_logger`. Those logs are the auditable
ground truth, but they are event-shaped. :func:`episode_from_motion_log` replays
the events and reconstructs learning-shaped frames:

* ``image_ghost_request`` → a perception/plan frame (perception, fold direction,
  dry-run posture derived from ``enable_motion``/``acknowledge``).
* ``protocol_confirm``     → updates protocol/baud + captures the position readback.
* ``write_position`` / ``moved`` → servo targets; *consecutive* writes to distinct
  servos are merged into one timestep so the action vector spans the arm.
* ``refused``              → a frame recording the refusal reason (``success=False``).

Everything is tolerant of missing fields. Pure stdlib — reuses the JSON(L) helpers
in :mod:`terafold.data.episode_schema`; no numpy / optional deps.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from terafold.data.episode_schema import (
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)
from terafold.data.schema import Episode, EpisodeFrame, action_vector

__all__ = [
    "EpisodeLogger",
    "episode_from_motion_log",
    "write_episode",
    "read_episode",
]

# Events that carry a single servo position command.
_WRITE_EVENTS = {"write_position", "set_position", "moved"}

# Canonical file names inside a structured-episode directory.
META_JSON = "meta.json"
FRAMES_JSONL = "frames.jsonl"
EPISODE_JSON = "episode.json"


def _as_perception(val: Any) -> Dict[str, Any]:
    """Coerce a logged ``perception`` value into a dict."""
    if isinstance(val, dict):
        return dict(val)
    if val is None:
        return {}
    return {"summary": val}


def _servo_id(data: Dict[str, Any]) -> Optional[int]:
    raw = data.get("id", data.get("servo_id"))
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


class _Context:
    """Running episode context accumulated while scanning the log."""

    def __init__(self, episode_id: Optional[str]) -> None:
        self.episode_id = episode_id
        self.task: Optional[str] = None
        self.robot_type: Optional[str] = None
        self.robot_config: Optional[str] = None
        self.port: Optional[str] = None
        self.protocol: Optional[str] = None
        self.baudrate: Optional[int] = None
        self.perception: Dict[str, Any] = {}
        self.fold_direction: Optional[str] = None
        self.dry_run: bool = True
        self.contact_enabled: bool = False
        self.safety_flags: Dict[str, Any] = {}
        self.image_paths: List[str] = []
        self.camera_intrinsics_id: Optional[str] = None
        self.table_homography_id: Optional[str] = None
        self.robot_table_transform_id: Optional[str] = None

    def new_frame(self, timestamp: Any, **overrides: Any) -> EpisodeFrame:
        base: Dict[str, Any] = dict(
            timestamp=float(timestamp) if timestamp is not None else 0.0,
            episode_id=self.episode_id,
            task=self.task,
            robot_type=self.robot_type,
            robot_config=self.robot_config,
            port=self.port,
            protocol=self.protocol,
            baudrate=self.baudrate,
            perception=dict(self.perception),
            fold_direction=self.fold_direction,
            dry_run=self.dry_run,
            contact_enabled=self.contact_enabled,
            safety_flags=dict(self.safety_flags),
            image_paths=list(self.image_paths),
            camera_intrinsics_id=self.camera_intrinsics_id,
            table_homography_id=self.table_homography_id,
            robot_table_transform_id=self.robot_table_transform_id,
        )
        base.update(overrides)
        return EpisodeFrame(**base)


def episode_from_motion_log(
    jsonl_path: str, meta: Optional[Dict[str, Any]] = None
) -> Episode:
    """Build an :class:`Episode` from a CommandLogger / motion-log JSONL file."""
    episode_id = (meta or {}).get("episode_id") or os.path.splitext(
        os.path.basename(jsonl_path)
    )[0]
    ctx = _Context(episode_id=episode_id)
    records = list(read_jsonl(jsonl_path))

    frames: List[EpisodeFrame] = []
    # A pending group of consecutive servo writes that share one timestep.
    pending: Dict[int, Any] = {}
    pending_speeds: Dict[int, Any] = {}
    pending_cmds: List[Dict[str, Any]] = []
    pending_ts: Any = None

    def flush_writes() -> None:
        nonlocal pending, pending_speeds, pending_cmds, pending_ts
        if not pending:
            return
        targets = dict(pending)
        order = sorted(targets, key=int)
        frame = ctx.new_frame(
            pending_ts,
            joint_targets_raw=targets,
            joint_speeds_raw=dict(pending_speeds),
            servo_ids=order,
            action_raw=[targets[k] for k in order],
            executed_command={"writes": list(pending_cmds)},
        )
        frames.append(frame)
        pending = {}
        pending_speeds = {}
        pending_cmds = []
        pending_ts = None

    for rec in records:
        if not isinstance(rec, dict):
            continue
        event = rec.get("event")
        data = rec.get("data") if isinstance(rec.get("data"), dict) else {}
        ts = rec.get("timestamp")

        if event in _WRITE_EVENTS:
            sid = _servo_id(data)
            if sid is None:
                continue
            # A repeated servo id means a new timestep for that joint.
            if sid in pending:
                flush_writes()
            pending[sid] = data.get("target")
            if data.get("speed") is not None:
                pending_speeds[sid] = data.get("speed")
            pending_cmds.append(
                {
                    "event": event,
                    "id": sid,
                    "target": data.get("target"),
                    "speed": data.get("speed"),
                    "acc": data.get("acc"),
                    "ok": data.get("ok"),
                    "label": data.get("label"),
                }
            )
            if pending_ts is None:
                pending_ts = ts
            continue

        # Any non-write event closes the current write timestep first.
        flush_writes()

        if event == "image_ghost_request":
            ctx.robot_config = data.get("robot", ctx.robot_config)
            ctx.robot_type = data.get("robot_type", data.get("robot", ctx.robot_type))
            ctx.port = data.get("port", ctx.port)
            ctx.perception = _as_perception(data.get("perception"))
            if data.get("confidence") is not None:
                ctx.perception["confidence"] = data.get("confidence")
            ctx.fold_direction = data.get("direction", ctx.fold_direction)
            enable = bool(data.get("enable_motion"))
            ack = bool(data.get("acknowledge"))
            ctx.dry_run = not (enable and ack)
            ctx.safety_flags.update(
                {"enable_motion": enable, "acknowledge": ack, "dry_run": ctx.dry_run}
            )
            frames.append(
                ctx.new_frame(
                    ts,
                    ghost_motion_plan={
                        "clearance_m": data.get("clearance_m"),
                        "kind": "ghost_fold",
                    },
                )
            )

        elif event == "protocol_confirm":
            if data.get("baudrate") is not None:
                ctx.baudrate = data.get("baudrate")
            confirmed = bool(data.get("confirmed"))
            ctx.safety_flags["protocol_confirmed"] = confirmed
            readbacks = data.get("read_example")
            positions: Dict[int, Any] = {}
            if isinstance(readbacks, dict):
                for k, v in readbacks.items():
                    try:
                        positions[int(k)] = v
                    except (TypeError, ValueError):
                        continue
            frames.append(
                ctx.new_frame(
                    ts,
                    joint_positions_raw=positions,
                    servo_ids=sorted(positions),
                    readbacks=readbacks,
                )
            )

        elif event == "refused":
            reason = data.get("reason") or data.get("refusal")
            ctx.safety_flags["refused"] = reason
            frames.append(
                ctx.new_frame(ts, success=False, failure_reason=reason)
            )

        elif event == "action":
            # CommandLogger.log_action style record (observation + action present).
            obs = rec.get("observation") or {}
            positions = {}
            if isinstance(obs, dict):
                jp = obs.get("joint_positions") or obs.get("positions")
                if isinstance(jp, dict):
                    for k, v in jp.items():
                        try:
                            positions[int(k)] = v
                        except (TypeError, ValueError):
                            continue
            frames.append(
                ctx.new_frame(
                    ts,
                    joint_positions_raw=positions,
                    servo_ids=sorted(positions),
                    executed_command=rec.get("action"),
                )
            )
        # All other lifecycle events (connect/close/...) carry no frame data.

    flush_writes()

    out_meta: Dict[str, Any] = {
        "episode_id": ctx.episode_id,
        "task": ctx.task,
        "robot_type": ctx.robot_type,
        "robot_config": ctx.robot_config,
        "port": ctx.port,
        "protocol": ctx.protocol,
        "baudrate": ctx.baudrate,
        "source_log": os.path.abspath(jsonl_path),
        "n_events": len(records),
        "dry_run": ctx.dry_run,
        "contact_enabled": ctx.contact_enabled,
        "fold_direction": ctx.fold_direction,
        "calibration": {
            "present": any(
                (
                    ctx.camera_intrinsics_id,
                    ctx.table_homography_id,
                    ctx.robot_table_transform_id,
                )
            ),
            "camera_intrinsics": ctx.camera_intrinsics_id,
            "table_homography": ctx.table_homography_id,
            "robot_table_transform": ctx.robot_table_transform_id,
        },
    }
    if meta:
        out_meta.update(meta)
    return Episode(meta=out_meta, frames=frames)


def write_episode(episode: Episode, out_dir: str) -> str:
    """Write an :class:`Episode` to ``out_dir`` (meta.json + frames.jsonl + combined).

    Returns the directory path. Reuses the shared JSON(L) helpers.
    """
    os.makedirs(out_dir, exist_ok=True)
    write_json(os.path.join(out_dir, META_JSON), episode.meta)
    write_jsonl(
        os.path.join(out_dir, FRAMES_JSONL), [f.to_dict() for f in episode.frames]
    )
    write_json(os.path.join(out_dir, EPISODE_JSON), episode.to_dict())
    return out_dir


def read_episode(path: str) -> Episode:
    """Read a structured episode written by :func:`write_episode`.

    ``path`` is the episode directory. Prefers the combined ``episode.json`` and
    falls back to ``meta.json`` + ``frames.jsonl``.
    """
    combined = os.path.join(path, EPISODE_JSON)
    if os.path.exists(combined):
        return Episode.from_dict(read_json(combined))
    meta_path = os.path.join(path, META_JSON)
    meta = read_json(meta_path) if os.path.exists(meta_path) else {}
    frames = [
        EpisodeFrame.from_dict(r)
        for r in read_jsonl(os.path.join(path, FRAMES_JSONL))
    ]
    return Episode(meta=meta if isinstance(meta, dict) else {}, frames=frames)


class EpisodeLogger:
    """Thin facade over the module functions (instantiable for ergonomic use)."""

    #: re-exported for callers that prefer ``EpisodeLogger.action_vector``.
    action_vector = staticmethod(action_vector)

    def episode_from_motion_log(
        self, jsonl_path: str, meta: Optional[Dict[str, Any]] = None
    ) -> Episode:
        return episode_from_motion_log(jsonl_path, meta=meta)

    def write_episode(self, episode: Episode, out_dir: str) -> str:
        return write_episode(episode, out_dir)

    def read_episode(self, path: str) -> Episode:
        return read_episode(path)
