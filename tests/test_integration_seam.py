"""End-to-end integration: calibration artifacts on disk → capability unlock.

Proves the seam between the calibration package (which WRITES artifacts) and the
capability system (which DISCOVERS them and unlocks levels). This is the
"the system gets more capable as you add *validated* artifacts" contract.
"""

from __future__ import annotations

import json

import yaml

import terafold.robot.capabilities as cap
from terafold.calibration.table_homography import calibrate_table
from terafold.calibration.robot_table_transform import fit_robot_table_transform
from terafold.robot.joint_map import JointMap
from terafold.robot.safety_state import SafetyLevel

ROBOT = "physical_7dof_waveshare"


class ConfirmedBackend:
    protocol_confirmed = True

    def available(self):
        return True

    def probe_protocol(self, ids=None):
        return {"opened": True, "responders": list(ids or [1, 2, 5, 6]),
                "read_ok": True, "confirmed": True}

    def read_all(self, ids):
        return {i: 1500 for i in ids}

    def close(self):
        pass


def _write_artifacts(tmp_path):
    th = tmp_path / "table_homography.yaml"
    ci = tmp_path / "camera_intrinsics.yaml"
    rtt = tmp_path / "robot_table_transform.yaml"
    touch = tmp_path / "touch.json"

    # Real homography fit (rms ~0 ⇒ valid_for_hover True).
    calibrate_table([[0, 0], [100, 0], [100, 100], [0, 100]],
                    [[0, 0], [0.3, 0], [0.3, 0.3], [0, 0.3]], str(th), timestamp=1.0)
    # Real robot↔table transform fit.
    json.dump({"points": [
        {"table_xy": [0, 0], "robot_xyz": [0.1, 0.2, 0.0]},
        {"table_xy": [0.3, 0], "robot_xyz": [0.4, 0.2, 0.0]},
        {"table_xy": [0.3, 0.2], "robot_xyz": [0.4, 0.4, 0.0]},
        {"table_xy": [0, 0.2], "robot_xyz": [0.1, 0.4, 0.0]}]}, open(touch, "w"))
    fit_robot_table_transform(str(touch), str(rtt), timestamp=1.0)
    # Minimal valid camera intrinsics (capabilities only reads the validity flag).
    with open(ci, "w") as f:
        yaml.safe_dump({"K": [[600, 0, 320], [0, 600, 240], [0, 0, 1]],
                        "rms_px": 0.4, "valid_for_hover": True, "valid_for_contact": True}, f)
    return {"joint_map": str(tmp_path / "jm.yaml"), "kinematics": str(tmp_path / "k.yaml"),
            "camera_intrinsics": str(ci), "table_homography": str(th),
            "robot_table_transform": str(rtt)}


def test_full_calibration_chain_unlocks_hover_not_contact(tmp_path, monkeypatch):
    paths = _write_artifacts(tmp_path)
    monkeypatch.setattr(cap, "artifact_paths", lambda robot: paths)
    jm = JointMap(ROBOT, {6: {"joint": "base_yaw", "sign": 1, "raw_min": 1200, "raw_max": 1800}})
    monkeypatch.setattr(cap, "load_joint_map", lambda r: jm)

    caps = cap.probe_capabilities(ROBOT, do_probe=True, backend=ConfirmedBackend())

    # Calibration artifacts were discovered from REAL files written by the
    # calibration package — proving the key names line up.
    assert caps.table_homography_valid is True
    assert caps.robot_table_transform_valid is True
    assert caps.camera_intrinsics_valid is True
    assert caps.calibration_valid_for_hover is True

    # Full valid chain + protocol + joint map ⇒ calibrated hover unlocks.
    assert caps.unlocked_level == int(SafetyLevel.CALIBRATED_HOVER)
    # …but contact is STILL locked (no unlock flag, no hover logs/compliance).
    assert caps.contact_allowed is False
    soft = next(s for s in caps.level_status if s["level"] == int(SafetyLevel.SOFT_CONTACT))
    assert soft["unlocked"] is False


def test_invalid_homography_keeps_hover_locked(tmp_path, monkeypatch):
    paths = _write_artifacts(tmp_path)
    # Corrupt the homography to be invalid for hover.
    with open(paths["table_homography"]) as f:
        d = yaml.safe_load(f)
    d["valid_for_hover"] = False
    with open(paths["table_homography"], "w") as f:
        yaml.safe_dump(d, f)
    monkeypatch.setattr(cap, "artifact_paths", lambda robot: paths)
    jm = JointMap(ROBOT, {6: {"joint": "base_yaw", "sign": 1}})
    monkeypatch.setattr(cap, "load_joint_map", lambda r: jm)

    caps = cap.probe_capabilities(ROBOT, do_probe=True, backend=ConfirmedBackend())
    assert caps.table_homography_valid is False
    # Falls back to the ghost-fold level — hover stays locked on the bad calibration.
    assert caps.unlocked_level == int(SafetyLevel.GHOST_FOLD)
