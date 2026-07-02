"""Tests for terafold.calibration (pure numpy; cv2 not required).

These exercise the table homography, camera-intrinsics graceful fallback, the
robot<-table Umeyama fit, the safe (dry-run) touch-calibration workflow, and
held-out validation. Everything here runs with only numpy + pyyaml installed.
"""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from terafold.calibration.camera_intrinsics import calibrate_camera
from terafold.calibration.robot_table_transform import (
    fit_robot_table_transform,
    robot_touch_calibration,
    umeyama,
)
from terafold.calibration.table_homography import (
    apply_homography,
    calibrate_table,
    reprojection_rms,
    solve_homography,
)
from terafold.calibration.validation import validate_table_calibration
from terafold.data.episode_schema import read_json, write_json
from terafold.math.transforms import compute_homography

ROBOT = "physical_7dof_waveshare"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _table_grid():
    """A spread of >=4 known table points (meters)."""
    xs = np.linspace(0.0, 0.5, 3)
    ys = np.linspace(0.0, 0.4, 3)
    return np.array([[x, y] for x in xs for y in ys], dtype=np.float64)


def _ground_truth_table_to_image():
    """A well-conditioned table->image homography (the 'true' camera)."""
    table_corners = np.array([[0, 0], [0.5, 0], [0.5, 0.4], [0, 0.4]], float)
    image_corners = np.array([[100, 80], [540, 75], [560, 460], [90, 470]], float)
    return compute_homography(table_corners, image_corners)


def _rotation_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# ---------------------------------------------------------------------------
# table_homography
# ---------------------------------------------------------------------------


def test_solve_homography_recovers_synthetic():
    H_ti = _ground_truth_table_to_image()
    table_pts = _table_grid()
    image_pts = apply_homography(H_ti, table_pts)  # project table -> image

    H = solve_homography(image_pts, table_pts)  # recover image -> table
    recovered = apply_homography(H, image_pts)
    assert np.allclose(recovered, table_pts, atol=1e-8)
    assert reprojection_rms(H, image_pts, table_pts) < 1e-9


def test_apply_homography_returns_nx2():
    H = np.eye(3)
    out = apply_homography(H, [[1.0, 2.0], [3.0, 4.0]])
    assert out.shape == (2, 2)
    # A single point still comes back as (1, 2).
    one = apply_homography(H, [5.0, 6.0])
    assert one.shape == (1, 2)


def test_calibrate_table_writes_valid_artifact(tmp_path):
    H_ti = _ground_truth_table_to_image()
    table_pts = _table_grid()
    image_pts = apply_homography(H_ti, table_pts)
    out = tmp_path / "table_homography.yaml"

    res = calibrate_table(image_pts, table_pts, str(out), operator="tester")
    assert res["out"] == str(out)
    assert res["valid_for_hover"] is True
    assert res["valid_for_contact"] is True
    assert res["rms_error_m"] < 1e-9
    assert res["operator"] == "tester"

    # File round-trips and carries the discovery flags + 3x3 H.
    saved = yaml.safe_load(out.read_text())
    assert np.asarray(saved["H"]).shape == (3, 3)
    assert saved["valid_for_hover"] is True
    assert "valid_threshold_m" in saved


def test_calibrate_table_image_metadata(tmp_path):
    table_pts = _table_grid()
    image_pts = apply_homography(_ground_truth_table_to_image(), table_pts)
    out = tmp_path / "h.yaml"
    res = calibrate_table(image_pts, table_pts, str(out), image="frames/cal.png")
    assert res["image"] == {"path": "frames/cal.png"}


def test_calibrate_table_wrong_counts_raise(tmp_path):
    out = str(tmp_path / "h.yaml")
    img = np.zeros((3, 2))  # only 3 points
    tab = np.zeros((3, 2))
    with pytest.raises(ValueError):
        calibrate_table(img, tab, out)
    # Length mismatch.
    with pytest.raises(ValueError):
        calibrate_table(np.zeros((5, 2)), np.zeros((4, 2)), out)
    with pytest.raises(ValueError):
        solve_homography(np.zeros((3, 2)), np.zeros((3, 2)))


# ---------------------------------------------------------------------------
# camera_intrinsics (graceful, optional)
# ---------------------------------------------------------------------------


def test_calibrate_camera_unavailable_without_images(tmp_path):
    # No images match -> 'unavailable' regardless of whether cv2 is installed.
    res = calibrate_camera(str(tmp_path / "no_such_*.png"), str(tmp_path / "K.yaml"))
    assert res["status"] == "unavailable"
    assert "install" in res and "reason" in res
    # Nothing was written.
    assert not (tmp_path / "K.yaml").exists()


def test_calibrate_camera_empty_glob_unavailable(tmp_path):
    res = calibrate_camera("", str(tmp_path / "K.yaml"))
    assert res["status"] == "unavailable"


# ---------------------------------------------------------------------------
# robot_table_transform: umeyama
# ---------------------------------------------------------------------------


def test_umeyama_recovers_rotation_translation():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(8, 3))
    R_true = _rotation_z(0.6)
    t_true = np.array([0.3, -0.2, 0.1])
    dst = (R_true @ src.T).T + t_true

    R, t, s = umeyama(src, dst, with_scale=False)
    assert np.allclose(R, R_true, atol=1e-9)
    assert np.allclose(t, t_true, atol=1e-9)
    assert abs(s - 1.0) < 1e-9
    pred = (R @ src.T).T + t
    rms = float(np.sqrt(((pred - dst) ** 2).sum(axis=1).mean()))
    assert rms < 1e-9


def test_umeyama_handles_2d_padding():
    src = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    R_true = _rotation_z(0.3)
    t_true = np.array([0.5, -0.4, 0.0])
    dst = (R_true @ np.c_[src, np.zeros(4)].T).T + t_true
    R, t, s = umeyama(src, dst)  # 2D src padded internally
    assert R.shape == (3, 3) and t.shape == (3,)
    assert np.allclose(R, R_true, atol=1e-9)


# ---------------------------------------------------------------------------
# robot_table_transform: fit + persist
# ---------------------------------------------------------------------------


def _write_touch_points(path, table_xy, R, t):
    pts = []
    for (x, y) in table_xy:
        rob = R @ np.array([x, y, 0.0]) + t
        pts.append({"table_xy": [float(x), float(y)], "robot_xyz": [float(v) for v in rob]})
    write_json(str(path), {"points": pts})


def test_fit_robot_table_transform_valid(tmp_path):
    R_true = _rotation_z(0.4)
    t_true = np.array([0.10, 0.20, 0.0])
    table_xy = [(0.0, 0.0), (0.5, 0.0), (0.5, 0.4), (0.0, 0.4), (0.25, 0.2)]
    tp = tmp_path / "touch_points.json"
    _write_touch_points(tp, table_xy, R_true, t_true)

    out = tmp_path / "robot_table_transform.yaml"
    res = fit_robot_table_transform(str(tp), str(out))
    assert res["valid_for_hover"] is True
    assert res["valid_for_contact"] is True
    assert res["rms_error_m"] < 1e-9
    assert np.allclose(np.asarray(res["R"]), R_true, atol=1e-9)
    assert np.allclose(np.asarray(res["t"]), t_true, atol=1e-9)

    saved = yaml.safe_load(out.read_text())
    assert saved["valid_for_hover"] is True
    assert "rms_error_m" in saved


def test_fit_robot_table_transform_too_few_points_raises(tmp_path):
    tp = tmp_path / "few.json"
    write_json(str(tp), {"points": [
        {"table_xy": [0.0, 0.0], "robot_xyz": [0.0, 0.0, 0.0]},
        {"table_xy": [0.1, 0.0], "robot_xyz": [0.1, 0.0, 0.0]},
    ]})
    with pytest.raises(ValueError):
        fit_robot_table_transform(str(tp), str(tmp_path / "out.yaml"))


# ---------------------------------------------------------------------------
# robot_touch_calibration: SAFE workflow
# ---------------------------------------------------------------------------


def test_robot_touch_calibration_dry_run_moves_nothing():
    res = robot_touch_calibration(ROBOT)
    assert res["status"] == "dry_run"
    assert res["moves_hardware"] is False
    assert isinstance(res["procedure"], list) and res["procedure"]


def test_robot_touch_calibration_saves_points(tmp_path):
    out = tmp_path / "points.json"
    points = [
        {"table_xy": [0.0, 0.0], "robot_xyz": [0.1, 0.2, 0.0]},
        {"table_xy": [0.5, 0.0], "robot_xyz": [0.6, 0.2, 0.0]},
        {"table_xy": [0.5, 0.4], "robot_xyz": [0.6, 0.6, 0.0]},
    ]
    res = robot_touch_calibration(ROBOT, out=str(out), points=points)
    assert res["status"] == "saved"
    assert res["moves_hardware"] is False
    assert res["num_points"] == 3
    saved = read_json(str(out))
    assert saved["num_points"] == 3
    assert saved["points"][0]["table_xy"] == [0.0, 0.0]


def test_robot_touch_calibration_real_mode_refused_without_flags():
    res = robot_touch_calibration(ROBOT, dry_run=False, enable_motion=False, acknowledge=False)
    assert res["status"] == "refused"
    assert res["allowed"] is False
    assert res["moves_hardware"] is False


def test_robot_touch_calibration_real_mode_refused_unconfirmed_protocol():
    class _Backend:
        protocol_confirmed = False

    res = robot_touch_calibration(
        ROBOT, dry_run=False, enable_motion=True, acknowledge=True, backend=_Backend()
    )
    assert res["status"] == "refused"
    assert "protocol" in res["reason"].lower()


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def test_validate_perfect_calibration(tmp_path):
    H_ti = _ground_truth_table_to_image()
    table_pts = _table_grid()
    image_pts = apply_homography(H_ti, table_pts)
    cal = tmp_path / "table_homography.yaml"
    calibrate_table(image_pts, table_pts, str(cal))

    # Held-out points generated from the SAME ground-truth camera.
    holdout_table = np.array([[0.1, 0.1], [0.4, 0.3], [0.2, 0.35]], float)
    holdout_image = apply_homography(H_ti, holdout_table)
    res = validate_table_calibration(str(cal), holdout_image, holdout_table)
    assert res["valid_for_hover"] is True
    assert res["valid_for_contact"] is True
    assert res["rms_error_m"] < 1e-6


def test_validate_high_error_marks_invalid(tmp_path):
    H_ti = _ground_truth_table_to_image()
    table_pts = _table_grid()
    image_pts = apply_homography(H_ti, table_pts)
    cal = tmp_path / "table_homography.yaml"
    calibrate_table(image_pts, table_pts, str(cal))

    # A held-out image point whose claimed table coordinate is way off (10 cm+).
    holdout_image = apply_homography(H_ti, np.array([[0.2, 0.2]], float))
    holdout_table_wrong = np.array([[0.2 + 0.5, 0.2 + 0.5]], float)
    res = validate_table_calibration(str(cal), holdout_image, holdout_table_wrong)
    assert res["valid_for_hover"] is False
    assert res["valid_for_contact"] is False
    assert res["rms_error_m"] > res["valid_threshold_m"]
