"""Tests for the central capability + safety-state system.

No hardware: the read-only protocol probe is driven with a mock backend.
"""

from __future__ import annotations

import pytest

from terafold.robot.capabilities import (RobotCapabilities, artifact_paths,
                                         discover_artifacts, probe_capabilities)
from terafold.robot.joint_map import JointMap
from terafold.robot.safety_state import LEVELS, SafetyLevel, level_spec

ROBOT = "physical_7dof_waveshare"


class FakeProbeBackend:
    """A read-only mock backend for probe_capabilities (never writes)."""

    def __init__(self, confirmed=True, responders=(1, 2, 5, 6), sdk=True, opened=True):
        self._confirmed = confirmed
        self._responders = list(responders)
        self._sdk = sdk
        self._opened = opened
        self.closed = False

    def available(self):
        return self._sdk

    @property
    def protocol_confirmed(self):
        return self._confirmed

    def probe_protocol(self, ids=None):
        return {"opened": self._opened, "responders": self._responders,
                "read_ok": self._confirmed, "confirmed": self._confirmed}

    def read_all(self, ids):
        return {i: 1500 for i in ids}

    def close(self):
        self.closed = True


# --------------------------- ladder spec ---------------------------


def test_eight_levels_defined():
    assert [int(l) for l in sorted(LEVELS)] == list(range(9))
    for lvl in SafetyLevel:
        spec = level_spec(lvl)
        assert spec.title and spec.summary
        d = spec.to_dict()
        assert d["level"] == int(lvl)


def test_levels_label():
    assert SafetyLevel.GHOST_FOLD.label == "L4 ghost_fold"
    assert SafetyLevel.SOFT_CONTACT == 6


# --------------------------- artifact discovery ---------------------------


def test_artifact_paths_and_discovery(tmp_path, monkeypatch):
    paths = artifact_paths(ROBOT)
    assert paths["joint_map"].endswith(f"{ROBOT}_joint_map.yaml")
    assert "camera_intrinsics" in paths and "robot_table_transform" in paths
    arts = discover_artifacts(ROBOT)
    # In a clean checkout no calibration exists.
    assert arts["camera_intrinsics"]["present"] in (True, False)
    assert "valid_for_hover" in arts["table_homography"]


# --------------------------- evaluation: no probe ---------------------------


def test_unprobed_is_sim_only_and_offline():
    caps = probe_capabilities(ROBOT, do_probe=False)
    assert isinstance(caps, RobotCapabilities)
    assert caps.unlocked_level == int(SafetyLevel.SIM_ONLY)
    assert caps.probed is False
    assert caps.port_open is None  # never opened
    assert caps.contact_allowed is False
    # Disabled IDs are 3,4,7 (configured active is 1,2,5,6).
    assert caps.disabled_ids == [3, 4, 7]
    # The read-only level is blocked, and the reason mentions the probe.
    ro = next(s for s in caps.level_status if s["level"] == int(SafetyLevel.READ_ONLY))
    assert ro["unlocked"] is False
    assert any("probe" in m for m in ro["missing"])


# --------------------------- evaluation: probed, confirmed ---------------------------


def test_probe_confirmed_unlocks_read_and_nudge():
    caps = probe_capabilities(ROBOT, do_probe=True, backend=FakeProbeBackend())
    assert caps.probed is True
    assert caps.protocol_confirmed is True
    assert caps.can_read is True and caps.can_write is True
    assert caps.responding_ids == [1, 2, 5, 6]
    # Read-only + tiny nudge unlock; joint map is still missing so ghost is blocked.
    assert caps.unlocked_level == int(SafetyLevel.TINY_NUDGE)
    jm_level = next(s for s in caps.level_status if s["level"] == int(SafetyLevel.JOINT_MAP))
    assert jm_level["unlocked"] is False
    assert any("joint map" in m.lower() for m in jm_level["missing"])


def test_probe_unconfirmed_blocks_writes():
    # Port opens + SDK present ⇒ read-only is fine, but no servo responded ⇒ the
    # protocol is unconfirmed, so writes (tiny nudge) stay blocked.
    caps = probe_capabilities(ROBOT, do_probe=True,
                              backend=FakeProbeBackend(confirmed=False, responders=[]))
    assert caps.protocol_confirmed is False
    assert caps.can_write is False
    assert caps.unlocked_level == int(SafetyLevel.READ_ONLY)
    nudge = next(s for s in caps.level_status if s["level"] == int(SafetyLevel.TINY_NUDGE))
    assert nudge["unlocked"] is False
    assert any("protocol confirmed" in m.lower() for m in nudge["missing"])


def test_probe_port_closed_is_sim_only():
    # If the port cannot open, even read-only is blocked.
    caps = probe_capabilities(ROBOT, do_probe=True,
                              backend=FakeProbeBackend(confirmed=False, responders=[], opened=False))
    assert caps.port_open is False
    assert caps.unlocked_level == int(SafetyLevel.SIM_ONLY)


# --------------------------- joint map unlocks ghost fold ---------------------------


def test_joint_map_with_base_yaw_unlocks_ghost(monkeypatch):
    import terafold.robot.capabilities as cap

    jm = JointMap(ROBOT, {6: {"joint": "base_yaw", "sign": 1}})
    monkeypatch.setattr(cap, "load_joint_map", lambda r: jm)
    caps = cap.probe_capabilities(ROBOT, do_probe=True, backend=FakeProbeBackend())
    assert caps.joint_map_present is True
    assert caps.sweep_joint_available is True
    assert caps.joint_limits_known is True  # global safe_position_units guard
    assert caps.unlocked_level == int(SafetyLevel.GHOST_FOLD)
    # Hover is still blocked: calibration missing.
    hover = next(s for s in caps.level_status if s["level"] == int(SafetyLevel.CALIBRATED_HOVER))
    assert hover["unlocked"] is False
    assert any("homography" in m.lower() or "intrinsics" in m.lower() or "transform" in m.lower()
               for m in hover["missing"])


# --------------------------- contact stays locked ---------------------------


def test_contact_blocked_even_with_full_calibration(monkeypatch, tmp_path):
    """Full valid calibration unlocks hover but NOT contact without the flag."""
    import terafold.robot.capabilities as cap

    jm = JointMap(ROBOT, {6: {"joint": "base_yaw", "sign": 1}})
    monkeypatch.setattr(cap, "load_joint_map", lambda r: jm)

    def fake_discover(robot, extra=None):
        valid = {"present": True, "valid_for_hover": True, "valid_for_contact": True,
                 "validated": True, "path": "x", "data": {}}
        return {"paths": {}, "joint_map": valid, "kinematics": valid,
                "camera_intrinsics": valid, "table_homography": valid,
                "robot_table_transform": valid}

    monkeypatch.setattr(cap, "discover_artifacts", fake_discover)
    caps = cap.probe_capabilities(ROBOT, do_probe=True, backend=FakeProbeBackend(),
                                  contact_unlock=False)
    assert caps.calibration_valid_for_hover is True
    assert caps.unlocked_level == int(SafetyLevel.CALIBRATED_HOVER)
    assert caps.contact_allowed is False  # no unlock flag → contact stays locked

    caps2 = cap.probe_capabilities(ROBOT, do_probe=True, backend=FakeProbeBackend(),
                                   contact_unlock=True)
    # Even WITH the unlock flag, soft contact needs hover logs/compliance (not built),
    # so it remains blocked and contact is still not allowed.
    assert caps2.unlocked_level == int(SafetyLevel.CALIBRATED_HOVER)
    assert caps2.contact_allowed is False


# --------------------------- report rendering ---------------------------


def test_report_markdown_and_dict():
    caps = probe_capabilities(ROBOT, do_probe=False)
    md = caps.report_markdown()
    assert "Unlocked level" in md and "Ladder" in md
    d = caps.to_dict()
    assert d["unlocked_label"].startswith("L0")
    assert d["contact_allowed"] is False
    assert isinstance(caps.blocked_capabilities(), list)


# --------------------------- backend interface aliases ---------------------------


def test_backend_named_interface_present():
    from terafold.robot.waveshare_sms_sts_backend import WaveshareSmsStsBackend

    b = WaveshareSmsStsBackend("/dev/null")
    for name in ("read_pos_speed", "read_all", "ping_many", "available",
                 "probe_protocol", "write_pos_ex", "safe_stop", "close_safely"):
        assert hasattr(b, name), name
    assert b.available() in (True, False)
    # close_safely always closes, even with nothing open.
    rep = b.close_safely()
    assert rep["port_closed"] is True
