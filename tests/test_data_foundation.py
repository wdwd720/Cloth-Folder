"""Tests for the structured-episode data foundation (schema → logger → export).

These MUST pass with only numpy + pyyaml installed (no cv2/torch/pandas). The
LeRobot export is exercised through its pure-stdlib fallback path.
"""

from __future__ import annotations

import os

from terafold.data.episode_schema import append_jsonl, read_json, read_jsonl
from terafold.data.schema import Episode, EpisodeFrame, action_vector
from terafold.data.episode_logger import (
    episode_from_motion_log,
    read_episode,
    write_episode,
)
from terafold.data.lerobot_export import export_episodes_to_lerobot
from terafold.data.summary import episode_summary, summary_markdown


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _write_motion_log(path: str) -> None:
    """A small but representative CommandLogger-style motion log."""
    events = [
        ("image_ghost_request", {
            "robot": "waveshare7", "port": "/dev/tty.usb", "clearance_m": 0.1,
            "enable_motion": False, "acknowledge": False,
            "perception": "towel detected", "confidence": 0.91,
            "direction": "right_to_left",
        }),
        ("protocol_confirm", {
            "sdk_available": True, "opened": True, "baudrate": 1000000,
            "confirmed": True, "responders": [1, 2], "read_ok": True,
            "read_example": {"1": 2048},
        }),
        # two consecutive writes to DISTINCT servos -> one merged timestep
        ("write_position", {"id": 2, "target": 2100, "speed": 300, "acc": 20, "ok": True}),
        ("write_position", {"id": 1, "target": 1900, "speed": 300, "acc": 20, "ok": True}),
        # repeated servo 1 -> flushes, starts a new timestep
        ("moved", {"label": "sweep", "id": 1, "target": 1950}),
        ("refused", {"reason": "protocol_unconfirmed"}),
    ]
    for i, (ev, data) in enumerate(events):
        append_jsonl(path, {"event": ev, "data": data, "timestamp": 1000.0 + i, "seq": i})


# --------------------------------------------------------------------------
# 1) schema
# --------------------------------------------------------------------------


def test_frame_roundtrip():
    f = EpisodeFrame(
        timestamp=12.5, task="fold", joint_targets_raw={1: 100, 2: 200},
        dry_run=True, contact_enabled=False, perception={"confidence": 0.8},
    )
    g = EpisodeFrame.from_dict(f.to_dict())
    assert g.timestamp == 12.5
    assert g.task == "fold"
    assert g.joint_targets_raw == {1: 100, 2: 200}
    assert g.dry_run is True and g.contact_enabled is False


def test_episode_roundtrip_and_validate_empty():
    ep = Episode(meta={"task": "fold"}, frames=[EpisodeFrame(timestamp=0.0)])
    back = Episode.from_dict(ep.to_dict())
    assert back.meta["task"] == "fold"
    assert len(back.frames) == 1
    # empty episode is flagged
    assert "episode has no frames" in Episode(meta={}, frames=[]).validate()
    # a normal episode validates clean
    assert ep.validate() == []


def test_action_vector_sorts_by_servo_id():
    f = EpisodeFrame(timestamp=0.0, joint_targets_raw={3: 30, 1: 10, 2: 20})
    assert action_vector(f) == [10, 20, 30]
    # also works after a JSON-style round trip (string keys)
    g = EpisodeFrame.from_dict({"timestamp": 0.0, "joint_targets_raw": {"3": 30, "1": 10, "2": 20}})
    assert action_vector(g) == [10, 20, 30]


def test_validate_catches_misordered_action():
    f = EpisodeFrame(timestamp=0.0, joint_targets_raw={1: 10, 2: 20}, action_raw=[20, 10])
    problems = Episode(meta={}, frames=[f]).validate()
    assert any("ordered by ascending servo id" in p for p in problems)


# --------------------------------------------------------------------------
# 2) episode_logger
# --------------------------------------------------------------------------


def test_episode_from_motion_log(tmp_path):
    log = str(tmp_path / "real_image_ghost_fold_x.jsonl")
    _write_motion_log(log)
    ep = episode_from_motion_log(log)

    assert len(ep.frames) >= 1
    # dry-run + contact captured from the ghost request
    assert ep.meta["dry_run"] is True
    assert ep.meta["contact_enabled"] is False
    assert ep.meta["fold_direction"] == "right_to_left"

    # the merged write frame holds both servos, ordered by id
    merged = [f for f in ep.frames if set(int(k) for k in f.joint_targets_raw) == {1, 2}]
    assert merged, "expected a frame merging the two consecutive distinct-servo writes"
    assert action_vector(merged[0]) == [1900, 2100]

    # a refusal frame is captured
    assert any(f.failure_reason == "protocol_unconfirmed" for f in ep.frames)
    # episode is internally consistent
    assert ep.validate() == []


def test_write_and_read_episode(tmp_path):
    log = str(tmp_path / "ep.jsonl")
    _write_motion_log(log)
    ep = episode_from_motion_log(log)

    out = str(tmp_path / "episode_dir")
    write_episode(ep, out)
    assert os.path.exists(os.path.join(out, "meta.json"))
    assert os.path.exists(os.path.join(out, "frames.jsonl"))

    back = read_episode(out)
    assert len(back.frames) == len(ep.frames)
    assert back.meta["dry_run"] is True


# --------------------------------------------------------------------------
# 3) lerobot_export
# --------------------------------------------------------------------------


def _synthetic_episode(direction: str) -> Episode:
    frames = [
        EpisodeFrame(timestamp=0.0, task="fold", fold_direction=direction,
                     perception={"confidence": 0.9}, dry_run=True),
        # use servo ids out of order to prove the union is sorted
        EpisodeFrame(timestamp=1.0, task="fold",
                     joint_positions_raw={3: 2000, 1: 1000, 2: 1500},
                     joint_targets_raw={3: 2100, 1: 1100, 2: 1600}, dry_run=True),
    ]
    return Episode(meta={"task": "fold", "dry_run": True, "contact_enabled": False,
                         "calibration": {"present": False}}, frames=frames)


def test_export_episodes_to_lerobot(tmp_path):
    eps = [_synthetic_episode("right_to_left"), _synthetic_episode("left_to_right")]
    out = str(tmp_path / "lerobot_ds")
    summary = export_episodes_to_lerobot(eps, out, fps=10)

    info_path = os.path.join(out, "meta", "info.json")
    assert os.path.exists(info_path)
    info = read_json(info_path)

    # action / state features ordered by ascending servo id
    assert info["features"]["action"]["names"] == ["servo_1", "servo_2", "servo_3"]
    assert info["features"]["observation.state"]["names"] == ["servo_1", "servo_2", "servo_3"]

    # missing calibration recorded explicitly
    assert info["calibration"]["present"] is False
    assert summary["calibration_present"] is False
    assert summary["episodes"] == 2
    assert summary["frames"] == 4
    assert summary["lerobot_installed"] in (True, False)
    assert summary["format"] in ("parquet", "jsonl")

    # dry-run episodes export; per-episode meta records dry_run + missing calibration
    rows = list(read_jsonl(os.path.join(out, "meta", "episodes.jsonl")))
    assert len(rows) == 2
    assert all(r["dry_run"] is True for r in rows)
    assert all(r["calibration"]["present"] is False for r in rows)

    assert os.path.exists(os.path.join(out, "README.md"))


def test_export_from_motion_log_dir(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    _write_motion_log(str(log_dir / "run_a.jsonl"))
    out = str(tmp_path / "ds")
    summary = export_episodes_to_lerobot(str(log_dir), out, fps=10)
    assert summary["episodes"] == 1
    assert summary["frames"] >= 1
    assert os.path.exists(os.path.join(out, "meta", "info.json"))


# --------------------------------------------------------------------------
# 4) summary
# --------------------------------------------------------------------------


def test_summary_over_motion_log(tmp_path):
    log = str(tmp_path / "ep.jsonl")
    _write_motion_log(log)

    s = episode_summary(log)
    assert s["n_frames"] > 0
    assert s["dry_run"] is True
    assert s["contact_enabled"] is False
    assert s["n_writes"] >= 1
    assert s["perception_confidence"] == 0.91

    md = summary_markdown(log)
    assert "Episode summary" in md
    assert "dry-run: True" in md


def test_summary_over_episode_dir(tmp_path):
    ep = _synthetic_episode("right_to_left")
    out = str(tmp_path / "edir")
    write_episode(ep, out)
    s = episode_summary(out)
    assert s["n_frames"] == 2
    assert s["dry_run"] is True
