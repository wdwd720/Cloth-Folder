"""Export recorded episodes to a LeRobot-compatible intermediate dataset.

The on-disk layout uses LeRobot's exact field names so the result can be loaded
by, or converted into, a real ``LeRobotDataset`` later — but it is written with
pure stdlib + numpy (no pandas/torch/lerobot required)::

    <out>/meta/info.json        # fps + feature schema (shapes/dtypes)
    <out>/meta/episodes.jsonl   # one row per episode (index, length, tasks)
    <out>/meta/tasks.jsonl      # one row per distinct task/instruction
    <out>/data/episode_000000.jsonl   # one row per frame
    <out>/images/episode_000000/frame_000000.png  # copied frames

Each data row carries: ``observation.images.top`` (relative image path),
``observation.state`` (list), ``action`` (list), ``timestamp``, ``frame_index``,
``episode_index``, ``index``, ``task_index`` and ``task``.

If ``lerobot`` happens to be installed, a real ``LeRobotDataset`` export is also
*attempted* (guarded — it never hard-fails the intermediate export).
"""

from __future__ import annotations

import os
import shutil
from typing import Any, Dict, List, Optional

from terafold.data.episode_schema import (
    list_episode_dirs,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)
from terafold.robot.base import RobotAction, RobotObservation

__all__ = ["export_to_lerobot", "print_lerobot_training_commands"]

IMAGE_KEY = "observation.images.top"
STATE_KEY = "observation.state"
ACTION_KEY = "action"


def _frame_files(frames_dir: str) -> List[str]:
    if not os.path.isdir(frames_dir):
        return []
    return sorted(n for n in os.listdir(frames_dir) if n.endswith(".png"))


def _image_shape(path: str) -> List[int]:
    from terafold.vision.imageio import imread

    img = imread(path)
    h, w = int(img.shape[0]), int(img.shape[1])
    c = int(img.shape[2]) if img.ndim == 3 else 1
    return [h, w, c]


def export_to_lerobot(
    episodes_dir: str,
    out_dir: str,
    fps: int = 10,
    task: Optional[object] = None,
) -> Dict[str, Any]:
    """Export every episode under ``episodes_dir`` into ``out_dir``.

    Returns a summary dict (counts, dims, output paths). ``task`` is optional and
    only used as a fallback source for the instruction string.
    """
    meta_dir = os.path.join(out_dir, "meta")
    data_dir = os.path.join(out_dir, "data")
    images_root = os.path.join(out_dir, "images")
    os.makedirs(meta_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(images_root, exist_ok=True)

    episode_paths = [
        p for p in list_episode_dirs(episodes_dir) if os.path.exists(p.metadata)
    ]

    tasks_index: Dict[str, int] = {}
    built_episodes: List[Dict[str, Any]] = []
    state_dim = 0
    action_dim = 0
    img_shape: Optional[List[int]] = None
    global_index = 0

    for ep_i, paths in enumerate(episode_paths):
        meta = read_json(paths.metadata)
        instruction = (
            meta.get("task_instruction")
            or (getattr(task, "instruction", None) if task else None)
            or meta.get("task_name", "fold")
        )
        if instruction not in tasks_index:
            tasks_index[instruction] = len(tasks_index)
        task_index = tasks_index[instruction]

        obs_rows = list(read_jsonl(paths.observations))
        act_rows = list(read_jsonl(paths.actions))
        frame_names = _frame_files(paths.frames_dir)
        n = min(len(obs_rows), len(frame_names)) if frame_names else len(obs_rows)

        ep_img_dir = os.path.join(images_root, f"episode_{ep_i:06d}")
        os.makedirs(ep_img_dir, exist_ok=True)

        rows: List[Dict[str, Any]] = []
        prev_action: Optional[List[float]] = None
        for i in range(n):
            obs = RobotObservation.from_dict(obs_rows[i])
            state = obs.to_state_vector().astype(float).tolist()
            state_dim = max(state_dim, len(state))

            act_d = act_rows[i] if i < len(act_rows) else {}
            if act_d:
                action = RobotAction.from_dict(act_d).to_action_vector().astype(float).tolist()
                prev_action = action
                action_dim = max(action_dim, len(action))
            else:
                action = prev_action  # may be None for the initial perception row

            # Copy the frame into the dataset image tree.
            src = os.path.join(paths.frames_dir, frame_names[i]) if i < len(frame_names) else None
            rel_img = ""
            if src and os.path.exists(src):
                if img_shape is None:
                    img_shape = _image_shape(src)
                dst_name = f"frame_{i:06d}.png"
                shutil.copyfile(src, os.path.join(ep_img_dir, dst_name))
                rel_img = os.path.join("images", f"episode_{ep_i:06d}", dst_name)

            rows.append(
                {
                    IMAGE_KEY: rel_img,
                    STATE_KEY: state,
                    ACTION_KEY: action,
                    "timestamp": round(i / float(max(fps, 1)), 6),
                    "frame_index": i,
                    "episode_index": ep_i,
                    "index": global_index,
                    "task_index": task_index,
                    "task": instruction,
                }
            )
            global_index += 1

        built_episodes.append(
            {
                "episode_index": ep_i,
                "length": len(rows),
                "tasks": [instruction],
                "source": paths.root,
                "rows": rows,
            }
        )

    # Backfill any leading ``None`` actions (initial perception steps) with zeros.
    action_dim = max(action_dim, 1)
    state_dim = max(state_dim, 1)
    for ep in built_episodes:
        for row in ep["rows"]:
            if row[ACTION_KEY] is None:
                row[ACTION_KEY] = [0.0] * action_dim
            elif len(row[ACTION_KEY]) < action_dim:
                row[ACTION_KEY] = row[ACTION_KEY] + [0.0] * (action_dim - len(row[ACTION_KEY]))
            if len(row[STATE_KEY]) < state_dim:
                row[STATE_KEY] = row[STATE_KEY] + [0.0] * (state_dim - len(row[STATE_KEY]))

    # Write per-episode data files + episodes.jsonl.
    total_frames = 0
    episodes_meta_rows: List[Dict[str, Any]] = []
    for ep in built_episodes:
        data_path = os.path.join(data_dir, f"episode_{ep['episode_index']:06d}.jsonl")
        write_jsonl(data_path, ep["rows"])
        total_frames += ep["length"]
        episodes_meta_rows.append(
            {
                "episode_index": ep["episode_index"],
                "length": ep["length"],
                "tasks": ep["tasks"],
            }
        )

    write_jsonl(os.path.join(meta_dir, "episodes.jsonl"), episodes_meta_rows)
    write_jsonl(
        os.path.join(meta_dir, "tasks.jsonl"),
        [{"task_index": idx, "task": name} for name, idx in tasks_index.items()],
    )

    if img_shape is None:
        img_shape = [0, 0, 3]
    info = {
        "codebase_version": "v2.0",
        "robot_type": "terafold",
        "fps": int(fps),
        "total_episodes": len(built_episodes),
        "total_frames": total_frames,
        "total_tasks": len(tasks_index),
        "chunks_size": 1000,
        "data_path": "data/episode_{episode_index:06d}.jsonl",
        "video_path": None,
        "features": {
            IMAGE_KEY: {
                "dtype": "image",
                "shape": img_shape,
                "names": ["height", "width", "channel"],
            },
            STATE_KEY: {
                "dtype": "float32",
                "shape": [state_dim],
                "names": [f"state_{i}" for i in range(state_dim)],
            },
            ACTION_KEY: {
                "dtype": "float32",
                "shape": [action_dim],
                "names": [f"action_{i}" for i in range(action_dim)],
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    write_json(os.path.join(meta_dir, "info.json"), info)

    summary = {
        "out_dir": os.path.abspath(out_dir),
        "num_episodes": len(built_episodes),
        "num_frames": total_frames,
        "num_tasks": len(tasks_index),
        "state_dim": state_dim,
        "action_dim": action_dim,
        "image_shape": img_shape,
        "fps": int(fps),
        "info_json": os.path.join(meta_dir, "info.json"),
        "lerobot_native_export": _maybe_native_export(out_dir, info),
        "training_commands": print_lerobot_training_commands(out_dir),
    }
    return summary


def _maybe_native_export(out_dir: str, info: dict) -> Dict[str, Any]:
    """Best-effort: note whether a real LeRobotDataset export is possible.

    We never hard-fail here. If ``lerobot`` is importable we record that a native
    conversion could be run; full automated conversion is intentionally left to
    the LeRobot CLI to avoid pinning an internal lerobot API.
    """
    try:
        import importlib.util

        spec = importlib.util.find_spec("lerobot")
    except Exception:
        spec = None
    if spec is None:
        return {
            "attempted": False,
            "available": False,
            "note": "lerobot not installed; install with: pip install -e '.[lerobot]'",
        }
    return {
        "attempted": False,
        "available": True,
        "note": (
            "lerobot is installed. Convert this intermediate dataset with the "
            "LeRobot dataset tools, or load meta/info.json + data/*.jsonl directly."
        ),
    }


def print_lerobot_training_commands(dataset_dir: str) -> str:
    """Return ready-to-run ACT and SmolVLA training command strings."""
    dataset_dir = os.path.abspath(dataset_dir)
    lines = [
        "# --- LeRobot training commands -------------------------------------",
        f"# Dataset (intermediate TeraFold export): {dataset_dir}",
        "# 1. Install LeRobot:  pip install -e '.[lerobot]'",
        "# 2. (If needed) convert/push this intermediate dataset to a LeRobotDataset.",
        "",
        "# Train ACT:",
        "python -m lerobot.scripts.train \\",
        "    --policy.type=act \\",
        f"    --dataset.repo_id=terafold/cloth_fold \\",
        f"    --dataset.root={dataset_dir} \\",
        "    --output_dir=outputs/train/act_cloth_fold \\",
        "    --batch_size=8 \\",
        "    --steps=20000",
        "",
        "# Train SmolVLA:",
        "python -m lerobot.scripts.train \\",
        "    --policy.type=smolvla \\",
        f"    --dataset.repo_id=terafold/cloth_fold \\",
        f"    --dataset.root={dataset_dir} \\",
        "    --output_dir=outputs/train/smolvla_cloth_fold \\",
        "    --batch_size=8 \\",
        "    --steps=20000",
    ]
    return "\n".join(lines)
