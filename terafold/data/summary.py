"""Human-readable summary of an episode, whatever shape it is stored in.

A single entry point that accepts any of the three on-disk forms used in this
stack and answers the operator's questions: how many frames, what task, which
robot, was it a dry-run, did it touch the cloth (contact), how many servo writes,
how confident was perception, and did it succeed.

Accepted inputs (auto-detected):

* a raw **motion-log JSONL** (CommandLogger events),
* a structured **episode directory** (``meta.json`` + ``frames.jsonl``),
* a structured **episode JSON** (``{meta, frames}``) or a ``frames.jsonl``.

Pure stdlib. The ``episode-summary`` CLI is a thin printer over
:func:`summary_markdown`.
"""

from __future__ import annotations

import os
from typing import Any, Dict

from terafold.data.episode_schema import read_json, read_jsonl
from terafold.data.episode_logger import episode_from_motion_log, read_episode
from terafold.data.schema import Episode, EpisodeFrame

__all__ = ["episode_summary", "summary_markdown", "load_episode_any"]


def load_episode_any(path: str) -> Episode:
    """Load an :class:`Episode` from a dir, an episode JSON, or a motion-log JSONL."""
    if os.path.isdir(path):
        return read_episode(path)

    if path.endswith(".jsonl"):
        records = list(read_jsonl(path))
        if records and isinstance(records[0], dict) and "event" in records[0]:
            return episode_from_motion_log(path)
        # otherwise it is a frames.jsonl
        frames = [EpisodeFrame.from_dict(r) for r in records]
        return Episode(meta={}, frames=frames)

    # A .json file: either a combined episode or (fallback) a motion log.
    try:
        obj = read_json(path)
    except (ValueError, OSError):
        return episode_from_motion_log(path)
    if isinstance(obj, dict) and "frames" in obj:
        return Episode.from_dict(obj)
    return episode_from_motion_log(path)


def episode_summary(path: str) -> Dict[str, Any]:
    """Return a compact summary dict for the episode at ``path``."""
    ep = load_episode_any(path)
    frames = ep.frames
    meta = ep.meta or {}

    n_frames = len(frames)

    dry_run = meta.get("dry_run")
    if dry_run is None:
        dry_run = all(bool(f.dry_run) for f in frames) if frames else True

    contact = meta.get("contact_enabled")
    if contact is None:
        contact = any(bool(f.contact_enabled) for f in frames)

    n_writes = sum(len(f.joint_targets_raw or {}) for f in frames)

    perception_confidence = None
    for f in frames:
        p = f.perception or {}
        if isinstance(p, dict) and p.get("confidence") is not None:
            perception_confidence = p.get("confidence")

    success = meta.get("success")
    if success is None:
        for f in frames:
            if f.success is not None:
                success = f.success

    calibration = meta.get("calibration") or {}

    return {
        "source": os.path.abspath(path),
        "n_frames": n_frames,
        "task": meta.get("task"),
        "robot": meta.get("robot_type") or meta.get("robot_config"),
        "dry_run": bool(dry_run),
        "contact_enabled": bool(contact),
        "n_writes": int(n_writes),
        "perception_confidence": perception_confidence,
        "calibration_present": bool(calibration.get("present")),
        "success": success,
    }


def summary_markdown(path: str) -> str:
    """Render :func:`episode_summary` as a short markdown report."""
    s = episode_summary(path)
    success = s["success"]
    success_str = "unknown" if success is None else ("yes" if success else "no")
    conf = s["perception_confidence"]
    conf_str = "n/a" if conf is None else str(conf)
    lines = [
        f"# Episode summary — {os.path.basename(s['source'].rstrip(os.sep))}",
        "",
        f"- source: `{s['source']}`",
        f"- task: {s['task'] or 'unknown'}",
        f"- robot: {s['robot'] or 'unknown'}",
        f"- frames: {s['n_frames']}",
        f"- servo writes: {s['n_writes']}",
        f"- dry-run: {s['dry_run']}",
        f"- contact enabled: {s['contact_enabled']}",
        f"- calibration present: {s['calibration_present']}",
        f"- perception confidence: {conf_str}",
        f"- success: {success_str}",
    ]
    if not s["dry_run"] and s["contact_enabled"]:
        lines.append("")
        lines.append("> NOTE: this episode reports REAL contact motion.")
    return "\n".join(lines)
