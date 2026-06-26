"""Virtual fold simulation: plan loading, waypoint parsing, rendering, safety."""

from __future__ import annotations

import inspect
import os

import numpy as np


def _plan_json(tmp_path):
    from terafold.config.schema import TaskConfig
    from terafold.physics.cloth_state import ClothKeypoints, FoldState
    from terafold.planning.fold_plan import plan_fold
    from terafold.planning.fold_task import FoldTask

    task = FoldTask.from_config(TaskConfig())
    kp = ClothKeypoints([60, 70], [190, 60], [200, 180], [70, 190])
    plan = plan_fold(FoldState(keypoints=kp, frame="image", image_shape=(256, 256)), task)
    path = str(tmp_path / "plan.json")
    plan.save(path)
    return path, plan


def test_sim_loads_plan_json(tmp_path):
    from terafold.sim.fold_sim import load_plan

    path, _ = _plan_json(tmp_path)
    plan = load_plan(path)
    assert "trajectory" in plan and "fold_state" in plan


def test_waypoints_parsed_correctly(tmp_path):
    from terafold.sim.fold_sim import load_plan, parse_trajectory

    path, plan = _plan_json(tmp_path)
    wp, grip, times, phases, scene = parse_trajectory(load_plan(path))
    assert wp.shape == (plan.trajectory.num_waypoints, 3)
    assert np.allclose(wp, plan.trajectory.waypoints)
    assert len(grip) == wp.shape[0] and len(times) == wp.shape[0]
    assert len(phases) == wp.shape[0]
    assert scene["direction"] == "right_to_left"


def test_ee_only_produces_video(tmp_path):
    from terafold.sim.fold_sim import run_sim_fold

    path, _ = _plan_json(tmp_path)
    out = str(tmp_path / "fold_ee.gif")  # gif avoids the ffmpeg requirement in CI
    m = run_sim_fold(path, out=out, level="ee-only", fps=12, max_seconds=3, on_log=lambda x: None)
    assert m["status"] == "ok" and m["renderer"] == "2d"
    assert m["frames"] > 0
    assert os.path.exists(m["out"]) and os.path.getsize(m["out"]) > 0
    assert os.path.exists(m["meta_json"])


def test_cloth_proxy_produces_video(tmp_path):
    from terafold.sim.fold_sim import run_sim_fold

    path, _ = _plan_json(tmp_path)
    out = str(tmp_path / "fold_cloth.gif")
    m = run_sim_fold(path, out=out, level="cloth-proxy", fps=12, max_seconds=3, on_log=lambda x: None)
    assert m["status"] == "ok"
    assert os.path.exists(m["out"])


def test_missing_mujoco_gives_install_message(tmp_path, monkeypatch):
    import terafold.sim.fold_sim as sim

    monkeypatch.setattr(sim, "have_mujoco", lambda: False)
    path, _ = _plan_json(tmp_path)
    m = sim.run_sim_fold(path, out=str(tmp_path / "arm.gif"), level="arm-ik", on_log=lambda x: None)
    assert m["status"] == "missing_dependency"
    assert 'pip install -e ".[sim]"' in m["install_command"]
    assert "MuJoCo" in m["message"]


def test_sim_does_not_require_real_robot_adapter(tmp_path):
    # Runs fully without any robot argument / adapter.
    from terafold.sim.fold_sim import run_sim_fold

    path, _ = _plan_json(tmp_path)
    m = run_sim_fold(path, out=str(tmp_path / "f.gif"), level="ee-only", fps=8, max_seconds=2,
                     on_log=lambda x: None)
    assert m["status"] == "ok"
    assert m["control"] == "none (simulation only)"
    assert m["simulation_only"] is True


def test_sim_never_sends_motor_commands(tmp_path):
    from terafold.sim.fold_sim import run_sim_fold

    path, _ = _plan_json(tmp_path)
    m = run_sim_fold(path, out=str(tmp_path / "f.gif"), level="ee-only", fps=8, max_seconds=2,
                     on_log=lambda x: None)
    assert m["motor_commands_sent"] == 0

    # Structural: the sim module never touches a robot adapter or sends actions.
    import terafold.sim.fold_sim as sim

    src = inspect.getsource(sim)
    for forbidden in ("send_action", "SO101Adapter", "LeArmAdapter", "GenericArmAdapter",
                      "MockRobot", "require_motion_enabled", "execute_trajectory"):
        assert forbidden not in src, f"sim module must not reference {forbidden!r}"
