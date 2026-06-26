"""HSV markers, camera discovery (mocked), robot discovery + generic adapter."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest


def _marked_image(seed=4, size=256):
    from terafold.vision.synthetic_cloth import SyntheticClothConfig, render_towel_scene

    rng = np.random.default_rng(seed)
    s = render_towel_scene(rng, SyntheticClothConfig(image_size=size, marker_mode=True), marker_mode=True)
    return s.image, s.keypoints


# -------------------- HSV markers (Part 5) --------------------


def test_hsv_detects_all_corners():
    img, true_kp = _marked_image()
    from terafold.vision.marker_detector import detect_markers_hsv

    res = detect_markers_hsv(img)
    assert res.all_corners_found and not res.missing_colors
    err = np.linalg.norm(res.keypoints.corners - true_kp.corners, axis=1)
    assert err.max() < 4.0


def test_hsv_reports_missing_colors():
    from terafold.vision.marker_detector import detect_markers_hsv

    res = detect_markers_hsv(np.full((120, 120, 3), 128, np.uint8))  # gray -> nothing
    assert not res.all_corners_found
    assert set(res.missing_colors) == {"red", "blue", "green", "orange"}
    assert res.keypoints is None


def test_parse_marker_map():
    from terafold.vision.marker_detector import DEFAULT_MARKER_MAP, parse_marker_map

    assert parse_marker_map(None) == DEFAULT_MARKER_MAP
    m = parse_marker_map("red:top_left,yellow:bottom_right")
    assert m == {"red": "top_left", "yellow": "bottom_right"}
    with pytest.raises(ValueError):
        parse_marker_map("notacolor:top_left")
    with pytest.raises(ValueError):
        parse_marker_map("red:not_a_corner")


def test_debug_masks_returned():
    from terafold.vision.marker_detector import detect_markers_hsv

    img, _ = _marked_image()
    res = detect_markers_hsv(img, return_masks=True)
    assert set(res.masks) == {"red", "blue", "green", "orange"}


# -------------------- Camera discovery (Part 1) --------------------


def test_scan_cameras_graceful_without_cv2(monkeypatch):
    import terafold.camera.discovery as disc

    monkeypatch.setitem(sys.modules, "cv2", None)  # makes `import cv2` raise
    report = disc.scan_cameras(max_index=2)
    assert report["available"] is False and report["cv2"] is False
    assert "opencv" in report["hint"].lower()


def test_scan_cameras_with_mocked_probe(monkeypatch):
    import terafold.camera.discovery as disc

    monkeypatch.setitem(sys.modules, "cv2", types.ModuleType("cv2"))  # top import succeeds

    def fake_probe(idx, out, warmup=3, min_blur=25.0):
        if idx == 1:
            return {"index": 1, "opened": True, "width": 640, "height": 480,
                    "brightness": 120.0, "blur": 90.0, "usable": True}
        return {"index": idx, "opened": False, "error": "could not open"}

    monkeypatch.setattr(disc, "_probe_index", fake_probe)
    report = disc.scan_cameras(max_index=2, isolate=False)
    assert report["available"] is True
    assert report["usable_indices"] == [1]
    cams = {c["index"]: c for c in report["cameras"]}
    assert cams[1]["usable"] and not cams[0]["opened"]


# -------------------- Robot discovery (Part 6) --------------------


def test_robot_scan_sends_no_commands():
    from terafold.robot.discovery import scan_serial_ports

    report = scan_serial_ports(probe_readonly=False)
    assert "no motor commands" in report["note"].lower()
    assert all(p["probe"] is None for p in report["ports"])  # nothing opened/written
    assert len(report["next_steps"]) >= 5


def test_robot_info_template_has_fields():
    from terafold.robot.discovery import robot_info_template_markdown

    md = robot_info_template_markdown()
    for field in ("Arm brand", "Baud rate", "Servo", "Gripper", "Workspace"):
        assert field in md


# -------------------- Generic adapter (Part 7) --------------------


def test_generic_adapter_refuses_real_motion():
    from terafold.robot.generic_adapter import GenericArmAdapter
    from terafold.robot.safety import SafetyError

    with pytest.raises(SafetyError):
        GenericArmAdapter(dry_run=False).connect()  # no verified backend


def test_generic_adapter_dry_run_and_export(tmp_path):
    from terafold.robot.base import RobotAction
    from terafold.robot.generic_adapter import GenericArmAdapter

    arm = GenericArmAdapter(dry_run=True)
    arm.connect()
    assert arm.is_connected
    arm.send_action(RobotAction(target_ee_pose=[0.3, 0.25, 0.1, 3.14, 0, 0], gripper=1.0,
                                metadata={"phase": "pregrasp"}))
    out = arm.export_trajectory(str(tmp_path / "traj.csv"))
    import os

    assert os.path.exists(out)
    with open(out) as f:
        header = f.readline()
    assert "x,y,z,gripper,speed,timestamp,phase" in header
