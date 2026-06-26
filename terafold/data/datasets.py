"""Read access to recorded TeraFold episodes (numpy only).

These helpers turn the on-disk episode directories written by
:mod:`terafold.data.recorder` into Python objects for analysis, residual-model
training, and benchmarking. Everything here is pure stdlib + numpy so the
Stage-0 pipeline never needs torch / pandas to load its own demonstrations.

``PUBLIC_DATASET_NOTES`` documents how *other-embodiment* public datasets (e.g.
LeRobot / Open-X) relate to TeraFold: they are useful for pretraining,
representation learning, and success scoring — NOT for direct low-level control
of our specific robot, whose action space differs.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterator, List, Optional, Tuple

from terafold.data.episode_schema import (
    EpisodePaths,
    list_episode_dirs,
    read_json,
    read_jsonl,
)
from terafold.physics.cloth_state import FoldState
from terafold.robot.base import RobotAction, RobotObservation

__all__ = [
    "load_episodes",
    "iter_episode_steps",
    "EpisodeStepDataset",
    "PUBLIC_DATASET_NOTES",
]


PUBLIC_DATASET_NOTES: Dict[str, str] = {
    "_general": (
        "Public manipulation datasets (LeRobot hub, Open-X-Embodiment, DROID, "
        "RoboMimic, etc.) are collected on OTHER embodiments with OTHER action "
        "spaces. Treat them as pretraining / representation / scoring data, NOT "
        "as a direct source of low-level commands for the TeraFold robot."
    ),
    "lerobot": (
        "LeRobot hub datasets share our intermediate schema (observation.images.*, "
        "observation.state, action). Useful to pretrain ACT / SmolVLA backbones; "
        "re-map or fine-tune the action head to our DoF before deployment."
    ),
    "open_x_embodiment": (
        "Massive multi-robot corpus. Good for visual/representation pretraining. "
        "Action spaces are heterogeneous — never replay actions on our robot."
    ),
    "cloth_folding_public": (
        "Cloth/fabric datasets (e.g. cloth manipulation benchmarks) are valuable "
        "for keypoint and success-model pretraining even when the gripper differs."
    ),
}


def load_episodes(episodes_dir: str) -> List[Dict[str, Any]]:
    """Load a light summary of every episode under ``episodes_dir``.

    Each entry is a dict with ``root``, ``metadata``, ``result``, ``num_frames``,
    ``num_actions`` and ``frame_paths``. Heavy per-step data is left on disk;
    use :func:`iter_episode_steps` or :class:`EpisodeStepDataset` to stream it.
    """
    episodes: List[Dict[str, Any]] = []
    for paths in list_episode_dirs(episodes_dir):
        if not os.path.exists(paths.metadata):
            continue
        metadata = read_json(paths.metadata)
        result = read_json(paths.result) if os.path.exists(paths.result) else None
        frame_paths = _frame_paths(paths)
        num_actions = sum(1 for r in read_jsonl(paths.actions) if r)
        episodes.append(
            {
                "root": paths.root,
                "metadata": metadata,
                "result": result,
                "num_frames": len(frame_paths),
                "num_actions": num_actions,
                "frame_paths": frame_paths,
            }
        )
    return episodes


def _frame_paths(paths: EpisodePaths) -> List[str]:
    """Sorted list of frame image paths actually present on disk."""
    frames_dir = paths.frames_dir
    if not os.path.isdir(frames_dir):
        return []
    names = sorted(
        n for n in os.listdir(frames_dir) if n.endswith(".png")
    )
    return [os.path.join(frames_dir, n) for n in names]


def _coerce_obs(d: dict) -> Optional[RobotObservation]:
    return RobotObservation.from_dict(d) if d else None


def _coerce_action(d: dict) -> Optional[RobotAction]:
    return RobotAction.from_dict(d) if d else None


def _coerce_fold_state(d: dict) -> Optional[FoldState]:
    return FoldState.from_dict(d) if d else None


def iter_episode_steps(
    episode_dir: str,
) -> Iterator[Tuple[Optional[RobotObservation], Optional[RobotAction], Optional[FoldState]]]:
    """Yield ``(observation, action, fold_state)`` for each recorded step.

    Missing components (e.g. the action on the initial perception step) are
    yielded as ``None``. The three jsonl streams are aligned by line index.
    """
    paths = EpisodePaths(episode_dir)
    obs_rows = list(read_jsonl(paths.observations))
    act_rows = list(read_jsonl(paths.actions))
    kp_rows = list(read_jsonl(paths.keypoints))
    n = max(len(obs_rows), len(act_rows), len(kp_rows))
    for i in range(n):
        obs = _coerce_obs(obs_rows[i]) if i < len(obs_rows) else None
        act = _coerce_action(act_rows[i]) if i < len(act_rows) else None
        fs = _coerce_fold_state(kp_rows[i]) if i < len(kp_rows) else None
        yield obs, act, fs


class EpisodeStepDataset:
    """A flat, indexable view over every step of every episode in a directory.

    ``__getitem__`` returns a dict with ``observation``, ``action``,
    ``fold_state``, ``episode_dir``, ``frame_index`` and ``image_path``. Pure
    numpy: no torch dependency.
    """

    def __init__(self, episodes_dir: str) -> None:
        self.episodes_dir = episodes_dir
        self._index: List[Tuple[str, int]] = []
        for paths in list_episode_dirs(episodes_dir):
            if not os.path.exists(paths.metadata):
                continue
            n = sum(1 for _ in read_jsonl(paths.observations))
            for i in range(n):
                self._index.append((paths.root, i))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx < 0:
            idx += len(self._index)
        if not 0 <= idx < len(self._index):
            raise IndexError(idx)
        episode_dir, frame_index = self._index[idx]
        paths = EpisodePaths(episode_dir)
        obs_rows = list(read_jsonl(paths.observations))
        act_rows = list(read_jsonl(paths.actions))
        kp_rows = list(read_jsonl(paths.keypoints))
        obs = _coerce_obs(obs_rows[frame_index]) if frame_index < len(obs_rows) else None
        act = _coerce_action(act_rows[frame_index]) if frame_index < len(act_rows) else None
        fs = _coerce_fold_state(kp_rows[frame_index]) if frame_index < len(kp_rows) else None
        return {
            "observation": obs,
            "action": act,
            "fold_state": fs,
            "episode_dir": episode_dir,
            "frame_index": frame_index,
            "image_path": paths.frame_path(frame_index),
        }
