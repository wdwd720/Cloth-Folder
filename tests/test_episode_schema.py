"""Episode schema writes/reads valid metadata and resolves canonical paths."""

from __future__ import annotations

import os

from terafold.data.episode_schema import (
    EpisodeMetadata,
    EpisodePaths,
    EpisodeResult,
    FailureMode,
    list_episode_dirs,
    next_episode_dir,
    read_json,
    write_json,
)


def test_episode_paths(tmp_path):
    p = EpisodePaths(str(tmp_path / "episode_000001"))
    assert p.metadata.endswith("metadata.json")
    assert p.frame_path(3).endswith("top_000003.png")
    p.ensure_dirs()
    assert os.path.isdir(p.frames_dir)


def test_metadata_roundtrip(tmp_path):
    meta = EpisodeMetadata(
        task_name="fold_towel_half_right_to_left",
        task_instruction="fold the small towel in half from right to left",
        robot_type="mock",
        camera_type="mock",
        start_time="2026-06-25T00:00:00",
        operator="tester",
        variation_tags=["blue", "rotated"],
    )
    path = str(tmp_path / "metadata.json")
    write_json(path, meta.to_dict())
    loaded = EpisodeMetadata.from_dict(read_json(path))
    assert loaded.task_name == meta.task_name
    assert loaded.variation_tags == ["blue", "rotated"]
    assert loaded.dry_run is True


def test_result_failure_modes():
    r = EpisodeResult(success=False, failure_mode=FailureMode.MISSED_GRASP.value)
    assert r.to_dict()["failure_mode"] == "missed_grasp"


def test_next_episode_dir_increments(tmp_path):
    root = str(tmp_path / "task")
    first = next_episode_dir(root)
    first.ensure_dirs()
    assert first.root.endswith("episode_000001")
    second = next_episode_dir(root)
    second.ensure_dirs()
    assert second.root.endswith("episode_000002")
    assert len(list_episode_dirs(root)) == 2
