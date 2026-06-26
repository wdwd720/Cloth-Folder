"""Episode data schema: the on-disk contract for recorded demonstrations.

An episode directory looks like::

    data/episodes/<task>/episode_000001/
        metadata.json        # EpisodeMetadata
        frames/
            top_000000.png    # top-down camera frames (one per recorded step)
            top_000001.png
        observations.jsonl    # one RobotObservation dict per line
        actions.jsonl         # one RobotAction dict per line
        keypoints.jsonl       # one FoldState dict per line (perception)
        plan.json             # the geometric/learned FoldPlan used
        safety.json           # safety config + any violations encountered
        result.json           # EpisodeResult (success/failure + metrics)
        notes.txt             # free-form operator notes

This module defines the metadata/result dataclasses, the canonical file names,
an :class:`EpisodePaths` helper, and JSON(L) read/write utilities. It is pure
stdlib + numpy so the recorder and exporter share one definition.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional

# Canonical file/dir names.
META_FILE = "metadata.json"
FRAMES_DIR = "frames"
OBS_FILE = "observations.jsonl"
ACTIONS_FILE = "actions.jsonl"
KEYPOINTS_FILE = "keypoints.jsonl"
PLAN_FILE = "plan.json"
SAFETY_FILE = "safety.json"
RESULT_FILE = "result.json"
NOTES_FILE = "notes.txt"
FRAME_PREFIX = "top_"
FRAME_EXT = ".png"


class FailureMode(str, Enum):
    """Standardized failure taxonomy for cloth folding."""

    NONE = "none"
    MISSED_GRASP = "missed_grasp"
    CLOTH_SLIPPED = "cloth_slipped"
    BAD_PERCEPTION = "bad_perception"
    COLLISION_RISK = "collision_risk"
    FOLD_INCOMPLETE = "fold_incomplete"
    CLOTH_BUNCHED = "cloth_bunched"
    GRIPPER_FAILED = "gripper_failed"
    HUMAN_INTERRUPTED = "human_interrupted"


@dataclass
class EpisodeMetadata:
    """Top-level descriptive metadata for one episode."""

    task_name: str
    task_instruction: str
    robot_type: str
    camera_type: str
    start_time: str  # ISO 8601
    operator: str = "unknown"
    cloth_type: str = "small_towel"
    variation_tags: List[str] = field(default_factory=list)
    calibration_files: Dict[str, Optional[str]] = field(default_factory=dict)
    episode_index: Optional[int] = None
    num_frames: int = 0
    end_time: Optional[str] = None
    dry_run: bool = True
    terafold_version: str = "0.0.1"
    # Filled in once known (mirrors result.json for quick scanning).
    success: Optional[bool] = None
    failure_mode: str = FailureMode.NONE.value

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "EpisodeMetadata":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class EpisodeResult:
    """Outcome + metrics for an episode (mirrors result.json)."""

    success: Optional[bool] = None
    failure_mode: str = FailureMode.NONE.value
    metrics: Dict[str, Any] = field(default_factory=dict)
    scorer: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "EpisodeResult":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class EpisodePaths:
    """Resolves the canonical file paths inside one episode directory."""

    root: str

    @property
    def metadata(self) -> str:
        return os.path.join(self.root, META_FILE)

    @property
    def frames_dir(self) -> str:
        return os.path.join(self.root, FRAMES_DIR)

    @property
    def observations(self) -> str:
        return os.path.join(self.root, OBS_FILE)

    @property
    def actions(self) -> str:
        return os.path.join(self.root, ACTIONS_FILE)

    @property
    def keypoints(self) -> str:
        return os.path.join(self.root, KEYPOINTS_FILE)

    @property
    def plan(self) -> str:
        return os.path.join(self.root, PLAN_FILE)

    @property
    def safety(self) -> str:
        return os.path.join(self.root, SAFETY_FILE)

    @property
    def result(self) -> str:
        return os.path.join(self.root, RESULT_FILE)

    @property
    def notes(self) -> str:
        return os.path.join(self.root, NOTES_FILE)

    def frame_path(self, index: int) -> str:
        return os.path.join(self.frames_dir, f"{FRAME_PREFIX}{index:06d}{FRAME_EXT}")

    def ensure_dirs(self) -> None:
        os.makedirs(self.frames_dir, exist_ok=True)


# --------------------------------------------------------------------------
# JSON / JSONL helpers (shared by recorder, replay, exporter).
# --------------------------------------------------------------------------


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def read_json(path: str) -> Any:
    with open(path) as f:
        return json.load(f)


def append_jsonl(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")


def write_jsonl(path: str, rows: List[Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def read_jsonl(path: str) -> Iterator[dict]:
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def next_episode_dir(task_root: str) -> EpisodePaths:
    """Return paths for the next ``episode_NNNNNN`` directory under ``task_root``."""
    os.makedirs(task_root, exist_ok=True)
    existing = [
        d
        for d in os.listdir(task_root)
        if d.startswith("episode_") and os.path.isdir(os.path.join(task_root, d))
    ]
    indices = []
    for d in existing:
        try:
            indices.append(int(d.split("_")[1]))
        except (IndexError, ValueError):
            continue
    nxt = (max(indices) + 1) if indices else 1
    return EpisodePaths(os.path.join(task_root, f"episode_{nxt:06d}"))


def list_episode_dirs(task_root: str) -> List[EpisodePaths]:
    """All episode dirs under ``task_root`` sorted by index."""
    if not os.path.isdir(task_root):
        return []
    dirs = sorted(
        d for d in os.listdir(task_root) if d.startswith("episode_")
    )
    return [EpisodePaths(os.path.join(task_root, d)) for d in dirs]


__all__ = [
    "FailureMode",
    "EpisodeMetadata",
    "EpisodeResult",
    "EpisodePaths",
    "write_json",
    "read_json",
    "append_jsonl",
    "write_jsonl",
    "read_jsonl",
    "next_episode_dir",
    "list_episode_dirs",
    "META_FILE",
    "FRAMES_DIR",
    "OBS_FILE",
    "ACTIONS_FILE",
    "KEYPOINTS_FILE",
    "PLAN_FILE",
    "SAFETY_FILE",
    "RESULT_FILE",
    "NOTES_FILE",
]
