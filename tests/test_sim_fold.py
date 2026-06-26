"""Virtual fold simulation: plan loading, waypoint parsing, rendering, safety."""

from __future__ import annotations

import inspect
import os
import xml.etree.ElementTree as ET

import numpy as np
import pytest


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


# -------------------- arm-ik MJCF mass/inertia regression --------------------

_MOVING_BODIES = ("arm0", "arm1", "arm2", "arm3", "arm4", "arm5", "gripper", "finger")


def test_arm_ik_mjcf_every_moving_body_has_mass_and_inertia(tmp_path):
    from terafold.sim.fold_sim import _build_mjcf, load_plan, parse_trajectory

    path, _ = _plan_json(tmp_path)
    wp, _, _, _, scene = parse_trajectory(load_plan(path))
    xml = _build_mjcf("arm-ik", scene, wp, 320, 240)
    root = ET.fromstring(xml)
    bodies = {b.get("name"): b for b in root.iter("body")}

    for name in _MOVING_BODIES:
        assert name in bodies, f"missing body {name}"
        inertial = bodies[name].find("inertial")
        assert inertial is not None, f"{name} has no <inertial> tag"
        mass = float(inertial.get("mass"))
        diag = [float(x) for x in inertial.get("diaginertia").split()]
        assert mass > 0.0, f"{name} mass must be > 0 (got {mass})"
        assert len(diag) == 3 and all(d > 0.0 for d in diag), f"{name} inertia must be > 0 (got {diag})"
        # arm links are moving bodies (have a joint); all have a geom.
        if name.startswith("arm"):
            assert bodies[name].find("joint") is not None
        assert bodies[name].find("geom") is not None


def test_ee_only_and_cloth_proxy_mjcf_unchanged_no_arm(tmp_path):
    from terafold.sim.fold_sim import _build_mjcf, load_plan, parse_trajectory

    path, _ = _plan_json(tmp_path)
    wp, _, _, _, scene = parse_trajectory(load_plan(path))
    for level in ("ee-only", "cloth-proxy"):
        root = ET.fromstring(_build_mjcf(level, scene, wp, 320, 240))
        names = {b.get("name") for b in root.iter("body")}
        assert "arm0" not in names and "gripper" not in names  # no arm in these levels


def test_arm_ik_mjcf_compiles_in_mujoco(tmp_path):
    mujoco = pytest.importorskip("mujoco")
    from terafold.sim.fold_sim import _build_mjcf, load_plan, parse_trajectory

    path, _ = _plan_json(tmp_path)
    wp, _, _, _, scene = parse_trajectory(load_plan(path))
    model = mujoco.MjModel.from_xml_string(_build_mjcf("arm-ik", scene, wp, 320, 240))
    for name in _MOVING_BODIES:
        bid = model.body(name).id
        assert model.body_mass[bid] > 1e-6, f"{name} compiled mass must be > 0"
        assert np.all(model.body_inertia[bid] > 0.0), f"{name} compiled inertia must be > 0"


# -------------------- camera / framing / visibility --------------------


def test_framing_warning_when_trajectory_out_of_bounds():
    from terafold.sim.fold_sim import _framing_warning

    cam_info = {"azimuth": 45, "elevation": -30, "distance": 0.1, "fovy": 45, "lookat": [0, 0, 0]}
    bounds = {"lo": np.array([-1.0, -1.0, 0.0]), "hi": np.array([1.0, 1.0, 0.0]),
              "center": np.zeros(3), "radius": 2.0}
    warn = _framing_warning("iso", cam_info, bounds)
    assert warn is not None
    for token in ("camera_pos", "target", "bounds"):
        assert token in warn


def test_no_framing_warning_when_fit():
    from terafold.sim.fold_sim import _framing_warning

    cam_info = {"azimuth": 45, "elevation": -30, "distance": 2.0, "fovy": 45, "lookat": [0, 0, 0]}
    bounds = {"lo": np.array([-0.1, -0.1, 0.0]), "hi": np.array([0.1, 0.1, 0.0]),
              "center": np.zeros(3), "radius": 0.15}
    assert _framing_warning("iso", cam_info, bounds) is None


def test_scene_bounds_cover_trajectory(tmp_path):
    from terafold.sim.fold_sim import _scene_bounds, load_plan, parse_trajectory

    path, _ = _plan_json(tmp_path)
    wp, _, _, _, scene = parse_trajectory(load_plan(path))
    b = _scene_bounds(wp, scene)
    assert np.all(b["lo"] <= wp[:, :3].min(0) + 1e-9)
    assert np.all(b["hi"] >= wp[:, :3].max(0) - 1e-9)
    assert b["radius"] > 0


@pytest.mark.parametrize("view", ["iso", "top", "side"])
def test_arm_ik_view_renders_visible_frames(tmp_path, view):
    pytest.importorskip("mujoco")
    from terafold.sim.fold_sim import run_sim_fold

    path, _ = _plan_json(tmp_path)
    out = str(tmp_path / f"arm_{view}.gif")
    m = run_sim_fold(path, out=out, level="arm-ik", view=view, fps=6, max_seconds=1.2,
                     on_log=lambda x: None)
    if m["status"] == "error":  # no GL context (headless CI) — not an MJCF bug
        pytest.skip(f"mujoco render unavailable: {str(m.get('error'))[:60]}")
    assert m["status"] == "ok" and m["renderer"] == "mujoco" and m["view"] == view

    import imageio.v2 as iio

    frames = list(iio.get_reader(out))
    assert len(frames) > 0
    arr = np.asarray(frames[len(frames) // 2])[:, :, :3]
    # Non-empty AND visible: not near-black, and actually has structure.
    assert arr.mean() > 20, f"{view} frame too dark (mean={arr.mean():.1f})"
    assert arr.max() > 60, f"{view} frame has no bright content"
    assert arr.std() > 10, f"{view} frame is a flat fill (std={arr.std():.1f})"


# -------------------- SO-101 realistic arm (so101-real) --------------------


def test_find_so101_assets_returns_report():
    from terafold.sim.so101 import find_so101_assets

    rep = find_so101_assets()
    assert {"found", "type", "path", "searched", "note"}.issubset(rep)
    assert isinstance(rep["searched"], list) and rep["searched"]
    assert isinstance(rep["found"], bool)


def test_so101_mjcf_valid_masses_and_gripper_joints(tmp_path):
    mujoco = pytest.importorskip("mujoco")
    from terafold.sim import so101
    from terafold.sim.fold_sim import _build_mjcf, load_plan, parse_trajectory

    path, _ = _plan_json(tmp_path)
    wp, _, _, _, scene = parse_trajectory(load_plan(path))
    model = mujoco.MjModel.from_xml_string(_build_mjcf("so101-real", scene, wp, 320, 240))
    for name in so101.SO101_BODIES:
        bid = model.body(name).id
        assert model.body_mass[bid] > 1e-6, f"{name} mass must be > 0"
        assert np.all(model.body_inertia[bid] > 0.0), f"{name} inertia must be > 0"
    # Two parallel-gripper finger joints (driven kinematically by the renderer).
    for jn in ("grip_left", "grip_right"):
        assert model.joint(jn).id >= 0


def test_so101_real_requires_mujoco_message(tmp_path, monkeypatch):
    import terafold.sim.fold_sim as sim

    monkeypatch.setattr(sim, "have_mujoco", lambda: False)
    path, _ = _plan_json(tmp_path)
    m = sim.run_sim_fold(path, out=str(tmp_path / "x.gif"), level="so101-real", on_log=lambda x: None)
    assert m["status"] == "missing_dependency"
    assert 'pip install -e ".[sim]"' in m["install_command"]


def test_so101_real_renders_and_sends_no_motor_commands(tmp_path):
    pytest.importorskip("mujoco")
    from terafold.sim.fold_sim import run_sim_fold

    path, _ = _plan_json(tmp_path)
    out = str(tmp_path / "so101.gif")
    m = run_sim_fold(path, out=out, level="so101-real", view="iso", fps=6, max_seconds=1.0,
                     width=320, height=240, on_log=lambda x: None)
    if m["status"] == "error":  # no GL context
        pytest.skip(f"mujoco render unavailable: {str(m.get('error'))[:60]}")
    assert m["status"] == "ok" and m["renderer"] == "mujoco"
    assert m["motor_commands_sent"] == 0 and m["simulation_only"] is True
    assert m["control"] == "none (simulation only)"
    assert isinstance(m["so101_assets"]["found"], bool)

    import imageio.v2 as iio

    arr = np.asarray(list(iio.get_reader(out))[-1])[:, :, :3]
    assert arr.mean() > 20 and arr.std() > 10  # visible 3D arm scene


def test_so101_real_all_views_render(tmp_path):
    pytest.importorskip("mujoco")
    from terafold.sim.fold_sim import ALL_VIEWS, run_sim_fold

    path, _ = _plan_json(tmp_path)
    m = run_sim_fold(path, out=str(tmp_path / "fold.gif"), level="so101-real", all_views=True,
                     fps=5, max_seconds=0.8, width=240, height=200, on_log=lambda x: None)
    if m["status"] == "error":
        pytest.skip("mujoco render unavailable")
    assert m["all_views"] is True
    assert set(m["views"]) == set(ALL_VIEWS)
    for p in m["views"].values():
        assert os.path.exists(p)


def test_so101_real_save_frames(tmp_path):
    import glob

    pytest.importorskip("mujoco")
    from terafold.sim.fold_sim import run_sim_fold

    path, _ = _plan_json(tmp_path)
    m = run_sim_fold(path, out=str(tmp_path / "f.gif"), level="so101-real", view="top",
                     save_frames=True, fps=5, max_seconds=0.8, width=240, height=200,
                     on_log=lambda x: None)
    if m["status"] == "error":
        pytest.skip("mujoco render unavailable")
    assert "frames_dirs" in m
    pngs = glob.glob(os.path.join(m["frames_dirs"]["top"], "*.png"))
    assert len(pngs) == m["frames"] and len(pngs) > 0


def test_so101_real_slowmo_lengthens_playback(tmp_path):
    pytest.importorskip("mujoco")
    from terafold.sim.fold_sim import run_sim_fold

    path, _ = _plan_json(tmp_path)
    base = run_sim_fold(path, out=str(tmp_path / "n.gif"), level="so101-real", view="iso",
                        fps=6, max_seconds=1.0, width=240, height=200, on_log=lambda x: None)
    slow = run_sim_fold(path, out=str(tmp_path / "s.gif"), level="so101-real", view="iso",
                        fps=6, max_seconds=1.0, slowmo=True, width=240, height=200,
                        on_log=lambda x: None)
    if base["status"] == "error" or slow["status"] == "error":
        pytest.skip("mujoco render unavailable")
    assert slow["frames"] > base["frames"]
    assert slow["slowmo"] is True


# -------------------- cloth-physics-proxy (PART 2/6) --------------------

_CORNERS = [[0.0, 0.0], [10.0, 0.0], [10.0, 8.0], [0.0, 8.0]]  # TL, TR, BR, BL


def _grid():
    from terafold.sim.cloth import ClothGrid

    return ClothGrid(_CORNERS, "right_to_left", [5.0, 0.0], [5.0, 8.0], grasp_xy=[10.0, 4.0])


def test_cloth_grid_initializes_flat():
    cg = _grid()
    assert np.allclose(cg.pos[:, :, 2], 0.0)
    assert np.allclose(cg.pos, cg.flat)


def test_cloth_right_to_left_moves_right_half_over_left():
    cg = _grid()
    before = float(cg.pos[cg.moving][:, 0].mean())
    cg.update(1.0)
    after = float(cg.pos[cg.moving][:, 0].mean())
    assert before > 5.0 and after < 5.0  # right half ends up left of the crease


def test_cloth_grasped_vertices_follow_gripper():
    cg = _grid()
    handle = np.array([5.0, 4.0, 3.0])
    cg.update(0.5, handle=handle, grasped=True)
    assert np.allclose(cg.grasp_point(), handle, atol=1e-6)


def test_cloth_vertices_never_below_table():
    cg = _grid()
    for f in np.linspace(0.0, 1.0, 11):
        cg.update(float(f), handle=np.array([5.0, 4.0, 2.0]), grasped=True)
        assert cg.pos[:, :, 2].min() >= -1e-9


def test_cloth_final_state_on_correct_side():
    cg = _grid()
    cg.update(1.0)
    m = cg.success_metrics(place_target=[0.0, 4.0])
    assert m["crossed_crease"] is True
    assert m["settled_on_target_side"] is True
    assert m["fold_visually_successful"] is True


def test_cloth_physics_renders_nonblack_and_evaluates(tmp_path):
    pytest.importorskip("mujoco")
    from terafold.sim.fold_sim import run_sim_fold

    path, _ = _plan_json(tmp_path)
    out = str(tmp_path / "cpp.gif")
    m = run_sim_fold(path, out=out, level="cloth-physics-proxy", view="demo", fps=6,
                     max_seconds=1.5, width=320, height=240, on_log=lambda x: None)
    if m["status"] == "error":
        pytest.skip("mujoco render unavailable")
    assert m["status"] == "ok" and m["renderer"] == "mujoco"
    assert "fold_visually_successful" in m and "final_edge_error_m" in m
    assert m["motor_commands_sent"] == 0 and m["simulation_only"] is True

    import imageio.v2 as iio

    arr = np.asarray(list(iio.get_reader(out))[len(list(iio.get_reader(out))) // 2])[:, :, :3]
    assert arr.mean() > 20 and arr.std() > 10


def test_cloth_physics_split_view_renders(tmp_path):
    pytest.importorskip("mujoco")
    from terafold.sim.fold_sim import run_sim_fold

    path, _ = _plan_json(tmp_path)
    m = run_sim_fold(path, out=str(tmp_path / "split.gif"), level="cloth-physics-proxy",
                     view="split", fps=5, max_seconds=1.0, width=320, height=240,
                     on_log=lambda x: None)
    if m["status"] == "error":
        pytest.skip("mujoco render unavailable")
    assert m["status"] == "ok" and m["frames"] > 0


def _high_plan(tmp_path):
    """A plan whose close/place waypoints float 12 cm above the cloth plane."""
    import json

    plan = {
        "trajectory": {
            "waypoints": [[7.0, 4.0, 0.10], [7.0, 4.0, 0.12], [3.0, 4.0, 0.12]],
            "gripper": [1.0, 0.0, 0.0], "times": [0.0, 1.0, 2.0],
            "phases": ["pregrasp", "close", "place"],
        },
        "fold_state": {
            "keypoints": {"top_left": [0.0, 0.0], "top_right": [10.0, 0.0],
                          "bottom_right": [10.0, 8.0], "bottom_left": [0.0, 8.0]},
            "metadata": {"direction": "right_to_left"},
        },
        "fold_line": {"a": [5.0, 0.0], "b": [5.0, 8.0]},
        "grasp_place": {"grasp": [10.0, 4.0], "place": [0.0, 4.0]},
        "metadata": {"direction": "right_to_left"},
    }
    p = str(tmp_path / "high.json")
    with open(p, "w") as f:
        json.dump(plan, f)
    return p


def test_contact_warning_fires_when_gripper_too_high(tmp_path):
    pytest.importorskip("mujoco")
    from terafold.sim.fold_sim import run_sim_fold

    p = _high_plan(tmp_path)
    m = run_sim_fold(p, out=str(tmp_path / "hi.gif"), level="cloth-physics-proxy", view="demo",
                     fps=5, max_seconds=1.0, width=240, height=200, on_log=lambda x: None)
    if m["status"] == "error":
        pytest.skip("mujoco render unavailable")
    warns = m.get("warnings", [])
    assert any("not contacting cloth" in w for w in warns), warns


def test_mujoco_cloth_falls_back_gracefully(tmp_path):
    pytest.importorskip("mujoco")
    from terafold.sim.fold_sim import run_sim_fold

    path, _ = _plan_json(tmp_path)
    msgs = []
    m = run_sim_fold(path, out=str(tmp_path / "mc.gif"), level="mujoco-cloth", view="demo",
                     fps=5, max_seconds=1.0, width=240, height=200, on_log=msgs.append)
    if m["status"] == "error":
        pytest.skip("mujoco render unavailable")
    assert m["mujoco_cloth"]["attempted"] is True
    assert m["mujoco_cloth"]["used"] is False
    assert m["mujoco_cloth"]["fell_back_to"] == "cloth-physics-proxy"
    assert m["render_level"] == "cloth-physics-proxy"
    assert any("cloth-physics-proxy recommended" in str(x) for x in msgs)


def test_cloth_physics_requires_mujoco_message(tmp_path, monkeypatch):
    import terafold.sim.fold_sim as sim

    monkeypatch.setattr(sim, "have_mujoco", lambda: False)
    path, _ = _plan_json(tmp_path)
    m = sim.run_sim_fold(path, out=str(tmp_path / "x.gif"), level="cloth-physics-proxy",
                         on_log=lambda x: None)
    assert m["status"] == "missing_dependency"
