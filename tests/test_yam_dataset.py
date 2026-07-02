"""Dummy YAM episode: on-disk structure, contents, and round-trip loading."""

from __future__ import annotations

import json

import pytest

from terafold.yam.dataset import create_dummy_episode, load_episode

TASK = "fold the towel in half neatly"


@pytest.fixture()
def episode(tmp_path):
    out = tmp_path / "towel_fold_smoke" / "episode_000001"
    res = create_dummy_episode(out, frames=3, seed=0)
    assert res["status"] == "ok"
    return out


def test_dummy_episode_directory_structure(episode):
    assert (episode / "metadata.json").is_file()
    assert (episode / "states.jsonl").is_file()
    assert (episode / "actions.jsonl").is_file()
    for cam in ("top", "left", "right"):
        frames = sorted((episode / "cameras" / cam).glob("frame_*.png"))
        assert len(frames) == 3, f"camera {cam} should have 3 frames"
        assert frames[0].name == "frame_000000.png"


def test_dummy_episode_metadata_contract(episode):
    meta = json.loads((episode / "metadata.json").read_text())
    assert meta["task"] == TASK
    assert meta["success"] is None
    assert meta["hardware_commanded"] is False
    assert meta["dummy"] is True
    assert meta["camera_order"] == ["top", "left", "right"]
    assert meta["norm_tag"] == "yam_dual_molmoact2"
    assert meta["arms"] == ["left_yam", "right_yam"]
    assert meta["num_frames"] == 3
    assert meta["action_dim"] == meta["state_dim"] == len(meta["state_layout"])


def test_dummy_episode_states_and_actions(episode):
    meta = json.loads((episode / "metadata.json").read_text())
    states = [json.loads(x) for x in (episode / "states.jsonl").read_text().splitlines()]
    actions = [json.loads(x) for x in (episode / "actions.jsonl").read_text().splitlines()]
    assert len(states) == len(actions) == 3

    ts = [s["timestamp"] for s in states]
    assert ts == sorted(ts) and ts[0] < ts[-1], "timestamps must increase"

    for s in states:
        assert len(s["state_vector"]) == meta["state_dim"]
        for arm in ("left_yam", "right_yam"):
            assert len(s["arms"][arm]["joint_pos_rad"]) == 6
            assert 0.0 <= s["arms"][arm]["gripper"] <= 1.0
    for a in actions:
        assert len(a["action"]) == meta["action_dim"]
        assert a["hardware_commanded"] is False


def test_dummy_episode_is_deterministic(tmp_path):
    a = create_dummy_episode(tmp_path / "a", frames=2, seed=7)
    b = create_dummy_episode(tmp_path / "b", frames=2, seed=7)
    assert a["status"] == b["status"] == "ok"
    sa = (tmp_path / "a" / "actions.jsonl").read_text().splitlines()
    sb = (tmp_path / "b" / "actions.jsonl").read_text().splitlines()
    assert [json.loads(x)["action"] for x in sa] == [json.loads(x)["action"] for x in sb]


def test_load_episode_roundtrip(episode):
    ep = load_episode(episode)
    assert ep["metadata"]["task"] == TASK
    assert len(ep["states"]) == len(ep["actions"]) == 3
    assert set(ep["frames_by_camera"]) == {"top", "left", "right"}
    assert all(len(v) == 3 for v in ep["frames_by_camera"].values())


def test_dummy_episode_respects_gripper_count(tmp_path):
    from terafold.yam.config import YamDualConfig

    cfg = YamDualConfig(grippers_per_arm=2)
    create_dummy_episode(tmp_path / "ep", frames=1, config=cfg)
    meta = json.loads((tmp_path / "ep" / "metadata.json").read_text())
    state = json.loads((tmp_path / "ep" / "states.jsonl").read_text().splitlines()[0])
    action = json.loads((tmp_path / "ep" / "actions.jsonl").read_text().splitlines()[0])
    assert meta["state_dim"] == 2 * (6 + 2) == len(state["state_vector"])
    assert meta["action_dim"] == len(action["action"])


def test_load_episode_missing_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_episode(tmp_path / "nope")


def test_load_episode_detects_inconsistency(episode):
    (episode / "cameras" / "top" / "frame_000002.png").unlink()
    with pytest.raises(ValueError):
        load_episode(episode)
