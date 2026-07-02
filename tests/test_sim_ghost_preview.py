"""Tests for the SIM image-driven air-only ghost-fold preview.

These run with ONLY numpy + pyyaml installed: matplotlib is optional, so the
suite never asserts that the PNG was rendered. One test forces the no-matplotlib
path to prove the metadata JSON is still written and ``rendered`` is ``False``.
"""

from __future__ import annotations

import builtins
import json
import os

from terafold.data.episode_schema import write_json
from terafold.robot.joint_map import JointMap
from terafold.sim.ghost_preview import run_sim_real_ghost_fold


def _tiny_plan() -> dict:
    """A minimal FoldPlan-shaped dict: trajectory + corners + direction."""
    return {
        "task_name": "fold_towel_half_right_to_left",
        "calibrated": False,
        "metadata": {"direction": "right_to_left"},
        "fold_state": {
            "keypoints": {
                "top_left": [0.17, 0.12],
                "top_right": [0.53, 0.12],
                "bottom_right": [0.53, 0.38],
                "bottom_left": [0.17, 0.38],
                "grasp": [0.53, 0.25],
                "place": [0.17, 0.25],
            }
        },
        "trajectory": {
            "waypoints": [
                [0.53, 0.25, 0.085],
                [0.53, 0.25, 0.0],     # would touch the table — must be lifted
                [0.35, 0.25, 0.04],
                [0.17, 0.25, 0.012],
            ],
            "gripper": [1.0, 0.0, 0.0, 1.0],  # a hard close — must be softened
            "phases": ["pregrasp", "grasp", "arc", "place"],
        },
    }


def _fake_joint_map() -> JointMap:
    return JointMap(robot="sim_arm",
                    servo_joint_map={1: {"joint": "base_yaw", "sign": 1}})


def _write_plan(tmp_path) -> str:
    p = str(tmp_path / "plan.json")
    write_json(p, _tiny_plan())
    return p


def test_preview_with_joint_map_is_air_only(tmp_path):
    plan = _write_plan(tmp_path)
    out = str(tmp_path / "preview.png")
    res = run_sim_real_ghost_fold(plan, joint_map=_fake_joint_map(), out=out,
                                  height_clearance_m=0.10, on_log=lambda _m: None)

    # Air-only, contact LOCKED, no kinematics.
    assert res["contact_disabled"] is True
    assert res["touches_table"] is False
    assert res["missing_kinematics"] is True
    assert res["min_z"] >= 0.10 - 1e-9
    assert res["num_waypoints"] == 4

    # Image-derived direction + active servo ids from the joint map.
    assert res["fold_direction"] == "right_to_left"
    assert res["active_servo_ids"] == [1]
    assert res["joint_map_present"] is True

    # Warnings are present (kinematics + uncalibrated plan).
    assert isinstance(res["warnings"], list) and res["warnings"]
    assert any("kinematics" in w.lower() for w in res["warnings"])
    assert any("calibration" in w.lower() for w in res["warnings"])

    # Metadata JSON is always written and round-trips.
    assert os.path.exists(res["meta_json"])
    with open(res["meta_json"]) as f:
        on_disk = json.load(f)
    assert on_disk["contact_disabled"] is True
    assert on_disk["touches_table"] is False
    assert on_disk["fold_direction"] == "right_to_left"

    # rendered is a bool either way (PNG is optional).
    assert isinstance(res["rendered"], bool)


def test_preview_without_joint_map_warns_about_servo_ids(tmp_path):
    plan = _write_plan(tmp_path)
    out = str(tmp_path / "nojm.png")
    res = run_sim_real_ghost_fold(plan, joint_map=None, out=out,
                                  on_log=lambda _m: None)

    assert res["joint_map_present"] is False
    assert res["active_servo_ids"] == []
    assert any("servo id" in w.lower() for w in res["warnings"])
    assert os.path.exists(res["meta_json"])


def test_preview_works_without_matplotlib(tmp_path, monkeypatch):
    """Force the no-matplotlib path: PNG skipped, metadata still written."""
    real_import = builtins.__import__

    def _blocked_import(name, *args, **kwargs):
        if name == "matplotlib" or name.startswith("matplotlib."):
            raise ImportError("matplotlib disabled for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)

    plan = _write_plan(tmp_path)
    out = str(tmp_path / "nomatplotlib.mp4")  # .mp4 should be re-pointed to .png
    res = run_sim_real_ghost_fold(plan, joint_map=_fake_joint_map(), out=out,
                                  on_log=lambda _m: None)

    assert res["rendered"] is False
    assert res["png"] is None
    assert any("matplotlib" in w.lower() for w in res["warnings"])
    # Metadata is ALWAYS written, even with no plotting backend.
    assert os.path.exists(res["meta_json"])
    assert res["meta_json"].endswith(".mp4.meta.json")
    assert res["contact_disabled"] is True


def test_preview_defaults_meta_next_to_out_and_touches_table_false(tmp_path):
    """A plan whose waypoints dip below the table must NOT touch it after lifting."""
    plan = _write_plan(tmp_path)
    out = str(tmp_path / "sub" / "deep_preview.png")
    res = run_sim_real_ghost_fold(plan, joint_map=_fake_joint_map(), out=out,
                                  height_clearance_m=0.15, on_log=lambda _m: None)
    assert res["touches_table"] is False
    assert res["min_z"] >= 0.15 - 1e-9
    assert res["meta_json"] == out + ".meta.json"
    assert os.path.exists(res["meta_json"])
