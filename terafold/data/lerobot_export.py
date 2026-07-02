"""Export structured episodes to a LeRobot-compatible dataset scaffold.

This is the second half of the data foundation: :mod:`terafold.data.episode_logger`
turns a raw motion log into an :class:`Episode`; this module turns a batch of
episodes into the on-disk layout LeRobot expects::

    <out>/meta/info.json        # fps + robot_type + feature schema
    <out>/meta/episodes.jsonl   # one row per episode (length, task, calibration)
    <out>/data/episode_000000.parquet   # if pandas+pyarrow available
    <out>/data/episode_000000.jsonl     # otherwise (pure-stdlib fallback)
    <out>/README.md             # how to finish the conversion / missing deps

Design guarantees:

* **No hard dependency on ``lerobot``, ``pandas`` or ``pyarrow``.** Those are
  lazy-imported; if absent we fall back to JSONL data files and say so in the
  README + the returned summary.
* **Action ordering is by ascending servo id** — the union of servo ids across all
  frames defines the slots, and ``observation.state`` / ``action`` features list
  ``servo_<id>`` names in that order. A policy and this dataset therefore agree.
* **Calibration presence is recorded explicitly** (per-episode and aggregate) so a
  consumer never silently trains on uncalibrated, dry-run-only data.
"""

from __future__ import annotations

import importlib.util
import os
from typing import Any, Dict, List, Union

from terafold.data.episode_schema import read_jsonl, write_json, write_jsonl
from terafold.data.episode_logger import (
    EPISODE_JSON,
    FRAMES_JSONL,
    META_JSON,
    episode_from_motion_log,
    read_episode,
)
from terafold.data.schema import Episode, EpisodeFrame, _get_by_id

__all__ = ["export_episodes_to_lerobot"]

IMAGE_KEY = "observation.images.top"
STATE_KEY = "observation.state"
ACTION_KEY = "action"


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _looks_like_motion_log(path: str) -> bool:
    """A motion-log JSONL has records with an ``event`` key."""
    for rec in read_jsonl(path):
        return isinstance(rec, dict) and "event" in rec
    return False


def _episodes_from_dir(root: str) -> List[Episode]:
    """Discover episodes under a directory: episode dirs and/or motion-log JSONL."""
    episodes: List[Episode] = []
    for name in sorted(os.listdir(root)):
        full = os.path.join(root, name)
        if os.path.isdir(full):
            if any(
                os.path.exists(os.path.join(full, f))
                for f in (EPISODE_JSON, META_JSON, FRAMES_JSONL)
            ):
                episodes.append(read_episode(full))
        elif name.endswith(".jsonl") and _looks_like_motion_log(full):
            episodes.append(episode_from_motion_log(full))
    return episodes


def _coerce_episodes(
    episodes: Union[List[Any], str],
) -> List[Episode]:
    """Normalize the flexible ``episodes`` argument into ``list[Episode]``."""
    if isinstance(episodes, str):
        return _episodes_from_dir(episodes)
    out: List[Episode] = []
    for e in episodes:
        if isinstance(e, Episode):
            out.append(e)
        elif isinstance(e, dict):
            out.append(Episode.from_dict(e))
        elif isinstance(e, str):
            if os.path.isdir(e):
                out.append(read_episode(e))
            elif _looks_like_motion_log(e):
                out.append(episode_from_motion_log(e))
        # silently skip anything we cannot interpret (tolerant by design).
    return out


def _union_servo_ids(episodes: List[Episode]) -> List[int]:
    """The sorted union of servo ids referenced by targets/positions everywhere."""
    ids: set = set()
    for ep in episodes:
        for f in ep.frames:
            for raw in (f.joint_targets_raw, f.joint_positions_raw):
                for k in raw or {}:
                    try:
                        ids.add(int(k))
                    except (TypeError, ValueError):
                        continue
    return sorted(ids)


def _episode_calibration(ep: Episode) -> Dict[str, Any]:
    """Resolve a calibration record for one episode (meta first, then frames)."""
    cal = dict((ep.meta or {}).get("calibration") or {})
    cam = cal.get("camera_intrinsics")
    homo = cal.get("table_homography")
    rtt = cal.get("robot_table_transform")
    for f in ep.frames:
        cam = cam or f.camera_intrinsics_id
        homo = homo or f.table_homography_id
        rtt = rtt or f.robot_table_transform_id
    return {
        "present": bool(cam or homo or rtt),
        "camera_intrinsics": cam,
        "table_homography": homo,
        "robot_table_transform": rtt,
    }


def _frame_row(
    f: EpisodeFrame,
    servo_ids: List[int],
    *,
    frame_index: int,
    episode_index: int,
    global_index: int,
    task: str,
    fps: int,
) -> Dict[str, Any]:
    state = [float(_get_by_id(f.joint_positions_raw, sid, 0.0) or 0.0) for sid in servo_ids]
    action = [float(_get_by_id(f.joint_targets_raw, sid, 0.0) or 0.0) for sid in servo_ids]
    image = f.image_paths[0] if f.image_paths else ""
    return {
        IMAGE_KEY: image,
        STATE_KEY: state,
        ACTION_KEY: action,
        "timestamp": round(frame_index / float(max(int(fps), 1)), 6),
        "frame_index": frame_index,
        "episode_index": episode_index,
        "index": global_index,
        "task_index": 0,
        "task": task,
        "dry_run": bool(f.dry_run),
        "contact_enabled": bool(f.contact_enabled),
    }


def _write_data_file(no_ext: str, rows: List[Dict[str, Any]], use_parquet: bool) -> str:
    if use_parquet:
        import pandas as pd  # lazy

        path = no_ext + ".parquet"
        pd.DataFrame(rows).to_parquet(path)
        return path
    path = no_ext + ".jsonl"
    write_jsonl(path, rows)
    return path


def _readme(out_dir: str, fmt: str, lerobot_installed: bool, calibration_present: bool) -> str:
    lines = [
        "# TeraFold → LeRobot export",
        "",
        f"- data format: **{fmt}**",
        f"- lerobot installed: **{lerobot_installed}**",
        f"- calibration present in source episodes: **{calibration_present}**",
        "",
        "`meta/info.json` describes the features. `observation.state` and `action`",
        "are joint-space vectors **ordered by ascending servo id** (`names` →",
        "`servo_<id>`), so slot *i* is always the same physical servo.",
        "",
    ]
    if fmt == "jsonl":
        lines += [
            "## Finishing the conversion (pandas/pyarrow not installed)",
            "",
            "Data was written as JSONL because `pandas`/`pyarrow` are unavailable.",
            "Install them to get parquet shards:",
            "",
            "    pip install pandas pyarrow",
            "",
            "then re-run the export.",
            "",
        ]
    if not lerobot_installed:
        lines += [
            "## LeRobot",
            "",
            "`lerobot` is not installed. The scaffold is still valid; install LeRobot",
            "to convert/load it into a `LeRobotDataset`:",
            "",
            "    pip install -e '.[lerobot]'",
            "",
        ]
    if not calibration_present:
        lines += [
            "## ⚠ Calibration missing",
            "",
            "No camera/table/robot calibration was present in these episodes (e.g.",
            "dry-run or ghost-only data). Joint-space states/actions are still valid,",
            "but image↔table↔robot grounding is NOT available. This is recorded in",
            "`meta/info.json` (`calibration.present = false`).",
            "",
        ]
    return "\n".join(lines)


def export_episodes_to_lerobot(
    episodes: Union[List[Any], str],
    out_dir: str,
    fps: int = 10,
    robot_type: str = "custom_7dof_sms_sts",
) -> Dict[str, Any]:
    """Export ``episodes`` to a LeRobot-compatible scaffold under ``out_dir``.

    Parameters
    ----------
    episodes:
        Either a ``list`` of :class:`Episode` (or dicts / paths), or a directory
        string containing motion-log JSONL files and/or structured-episode dirs.
    out_dir:
        Destination directory (created if missing).
    fps:
        Frames-per-second written into ``info.json`` and used for timestamps.
    robot_type:
        Recorded in ``info.json`` (the SMS/STS 7-DOF arm by default).

    Returns
    -------
    dict
        ``{episodes, frames, out, format, lerobot_installed, note, ...}``.
    """
    eps = _coerce_episodes(episodes)
    servo_ids = _union_servo_ids(eps)

    meta_dir = os.path.join(out_dir, "meta")
    data_dir = os.path.join(out_dir, "data")
    os.makedirs(meta_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    use_parquet = _module_available("pandas") and _module_available("pyarrow")
    fmt = "parquet" if use_parquet else "jsonl"
    lerobot_installed = _module_available("lerobot")

    total_frames = 0
    global_index = 0
    calibration_present_any = False
    episodes_meta: List[Dict[str, Any]] = []

    for ep_i, ep in enumerate(eps):
        task = (ep.meta or {}).get("task") or "fold"
        rows: List[Dict[str, Any]] = []
        for fi, f in enumerate(ep.frames):
            rows.append(
                _frame_row(
                    f,
                    servo_ids,
                    frame_index=fi,
                    episode_index=ep_i,
                    global_index=global_index,
                    task=task,
                    fps=fps,
                )
            )
            global_index += 1
        no_ext = os.path.join(data_dir, f"episode_{ep_i:06d}")
        _write_data_file(no_ext, rows, use_parquet)
        total_frames += len(rows)

        cal = _episode_calibration(ep)
        calibration_present_any = calibration_present_any or cal["present"]
        episodes_meta.append(
            {
                "episode_index": ep_i,
                "length": len(rows),
                "task": task,
                "dry_run": bool((ep.meta or {}).get("dry_run", True)),
                "contact_enabled": bool((ep.meta or {}).get("contact_enabled", False)),
                "calibration": cal,
                "source_log": (ep.meta or {}).get("source_log"),
            }
        )

    state_names = [f"servo_{sid}" for sid in servo_ids]
    dim = len(servo_ids)
    info = {
        "codebase_version": "v2.0",
        "robot_type": robot_type,
        "fps": int(fps),
        "total_episodes": len(eps),
        "total_frames": total_frames,
        "data_path": "data/episode_{episode_index:06d}." + fmt,
        "data_format": fmt,
        "calibration": {
            "present": bool(calibration_present_any),
            "note": (
                "joint-space only; no image↔table↔robot grounding"
                if not calibration_present_any
                else "at least one episode carried calibration ids"
            ),
        },
        "features": {
            IMAGE_KEY: {
                "dtype": "image",
                "shape": [0, 0, 3],
                "names": ["height", "width", "channel"],
            },
            STATE_KEY: {
                "dtype": "float32",
                "shape": [dim],
                "names": state_names,
            },
            ACTION_KEY: {
                "dtype": "float32",
                "shape": [dim],
                "names": state_names,
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    write_json(os.path.join(meta_dir, "info.json"), info)
    write_jsonl(os.path.join(meta_dir, "episodes.jsonl"), episodes_meta)

    readme = _readme(out_dir, fmt, lerobot_installed, calibration_present_any)
    with open(os.path.join(out_dir, "README.md"), "w") as fh:
        fh.write(readme)

    note_bits = [
        f"{len(eps)} episode(s), {total_frames} frame(s) → {fmt}.",
        "action/state ordered by ascending servo id.",
    ]
    if not lerobot_installed:
        note_bits.append("lerobot not installed (scaffold only).")
    if not calibration_present_any:
        note_bits.append("calibration absent (recorded in info.json).")

    return {
        "episodes": len(eps),
        "frames": total_frames,
        "out": os.path.abspath(out_dir),
        "format": fmt,
        "lerobot_installed": bool(lerobot_installed),
        "servo_ids": servo_ids,
        "calibration_present": bool(calibration_present_any),
        "info_json": os.path.join(meta_dir, "info.json"),
        "note": " ".join(note_bits),
    }
