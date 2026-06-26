"""demo-image (markers/claude/model), trajectory export, calibration, real-motion gate."""

from __future__ import annotations

import json
import os

import numpy as np

from terafold.demo import run_demo_image, run_demo_today

TASK = os.path.abspath("configs/task_fold_towel_half.yaml")


def _marked_image_file(tmp_path, size=256):
    from terafold.vision.imageio import imwrite
    from terafold.vision.synthetic_cloth import SyntheticClothConfig, render_towel_scene

    rng = np.random.default_rng(7)
    s = render_towel_scene(rng, SyntheticClothConfig(image_size=size, marker_mode=True), marker_mode=True)
    path = str(tmp_path / "towel.png")
    imwrite(path, s.image)
    return path


CLAUDE_JSON = json.dumps({
    "top_left": [40, 30], "top_right": [220, 35], "bottom_left": [45, 200], "bottom_right": [215, 205],
    "grasp_point": [215, 120], "place_point": [45, 120], "fold_line": [[130, 30], [130, 205]],
    "confidence": 0.85, "notes": "towel",
})


def test_demo_image_markers(tmp_path):
    img = _marked_image_file(tmp_path)
    res = run_demo_image(
        img, mode="markers", task_path=TASK,
        overlay_out=str(tmp_path / "ov.png"), save_json=str(tmp_path / "res.json"),
    )
    assert res["status"] == "ok"
    assert res["perception"] == "markers"
    assert res["confidence"] >= 0.9
    assert res["dry_run_only"] is True  # uncalibrated
    assert os.path.exists(res["overlay_path"]) and os.path.exists(res["json_path"])
    # The real-motion gate must list unmet requirements (no calibration, etc.).
    assert len(res["real_motion"]["missing"]) >= 1


def test_demo_image_claude_mocked(tmp_path):
    img = _marked_image_file(tmp_path)
    res = run_demo_image(
        img, mode="claude", task_path=TASK, claude_responder=lambda im: CLAUDE_JSON,
        overlay_out=str(tmp_path / "ov.png"), save_json=str(tmp_path / "res.json"),
    )
    assert res["perception"] == "claude"
    assert abs(res["confidence"] - 0.85) < 1e-6


def test_demo_image_model_no_checkpoint_is_fallback(tmp_path):
    img = _marked_image_file(tmp_path)
    res = run_demo_image(img, mode="model", task_path=TASK, overlay_out=str(tmp_path / "ov.png"),
                         save_json=str(tmp_path / "res.json"))
    assert res["perception"] == "classical_fallback"
    # Fallback perception must block real motion.
    assert any("perception" in m for m in res["real_motion"]["missing"])


def test_export_trajectory_creates_csv(tmp_path):
    # Build a real plan, save it, export trajectory to CSV.
    from terafold.config.schema import TaskConfig
    from terafold.physics.cloth_state import ClothKeypoints, FoldState
    from terafold.planning.fold_plan import plan_fold
    from terafold.planning.fold_task import FoldTask
    from terafold.data.trajectory_export import export_trajectory

    task = FoldTask.from_config(TaskConfig())
    kp = ClothKeypoints([60, 70], [190, 60], [200, 180], [70, 190])
    plan = plan_fold(FoldState(keypoints=kp, frame="image", image_shape=(256, 256)), task)
    pj = str(tmp_path / "plan.json")
    plan.save(pj)
    out = str(tmp_path / "traj.csv")
    res = export_trajectory(pj, out)
    assert res["rows"] == plan.trajectory.num_waypoints
    with open(out) as f:
        lines = f.read().splitlines()
    assert lines[0] == "waypoint,x,y,z,gripper,speed,timestamp,phase"
    assert len(lines) == res["rows"] + 1


def test_calibrate_table_from_image(tmp_path):
    from terafold.camera.homography import load_homography
    from terafold.camera.table_calibration import calibrate_table_from_image

    out = str(tmp_path / "homography.json")
    res = calibrate_table_from_image(
        [[10, 10], [290, 12], [292, 190], [8, 188]],
        [[0, 0], [0.7, 0], [0.7, 0.5], [0, 0.5]], out,
    )
    assert res["reprojection_error_m"] < 1e-3
    assert load_homography(out).image_calibrated is True


def test_no_real_motion_without_calibration(tmp_path):
    # so101 + both flags but uncalibrated/markers-only -> refused.
    res = run_demo_today(
        task_path=TASK, camera="mock", robot="so101", mode="markers",
        dry_run=False, enable_motion=True, acknowledge=True, out=str(tmp_path / "ep"),
    )
    assert res["status"] == "real_motion_refused"
    assert any("calibrat" in m for m in res["missing"])


def test_no_real_motion_when_perception_is_fallback(tmp_path):
    # mode model with no checkpoint -> classical fallback -> real motion refused.
    res = run_demo_today(
        task_path=TASK, camera="mock", robot="so101", mode="model",
        dry_run=False, enable_motion=True, acknowledge=True, out=str(tmp_path / "ep"),
    )
    assert res["status"] == "real_motion_refused"
    assert any("perception" in m for m in res["missing"])
