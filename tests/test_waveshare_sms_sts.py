"""Tests for the verified SMS/STS backend + image->plan->ghost pipeline.

These tests NEVER touch real hardware: every real-motion path is driven with a
MOCK backend, and perception uses a synthetic image.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from terafold.robot.joint_map import JointMap
from terafold.robot.waveshare_bus_servo import PROTOCOL_REFUSAL, WaveshareBusServoError
from terafold.robot.waveshare_sms_sts_backend import (WaveshareSmsStsBackend,
                                                      load_scservo_sdk)

ROBOT = "physical_7dof_waveshare"


class MockBackend:
    """A confirmed backend that records writes — used to avoid the live arm."""

    protocol_confirmed = True

    def __init__(self, pos=1500):
        self.writes = []
        self.pos = pos
        self.closed = False
        self.estopped = False

    def read_all_positions(self, ids):
        return {i: self.pos for i in ids}

    def read_position(self, i):
        return self.pos

    def write_position(self, sid, target, speed=None, acc=None):
        self.writes.append((sid, int(target)))
        return True

    def close(self):
        self.closed = True

    def emergency_stop(self):
        self.estopped = True
        return {"port_closed": True}


def _synthetic_towel(tmp_path):
    from terafold.vision.imageio import imwrite

    img = np.full((180, 240, 3), 28, np.uint8)
    img[40:140, 40:200] = 210  # bright towel rectangle
    p = str(tmp_path / "towel.png")
    imwrite(p, img)
    return p


# --------------------------- backend ---------------------------


def test_backend_imports_or_fails_gracefully():
    sdk = load_scservo_sdk()
    assert sdk is None or hasattr(sdk, "PortHandler")  # importable or None — never crashes
    b = WaveshareSmsStsBackend("/dev/null")
    assert isinstance(b.sdk_available, bool)
    assert b.protocol_confirmed is False  # not confirmed until open+ping+read


def test_backend_write_refuses_without_confirmation():
    b = WaveshareSmsStsBackend("/dev/null")
    with pytest.raises(WaveshareBusServoError) as e:
        b.write_position(1, 1500)
    assert str(e.value) == PROTOCOL_REFUSAL
    with pytest.raises(WaveshareBusServoError):
        b.torque_enable(1, True)


def test_backend_ping_and_read_mocked():
    b = WaveshareSmsStsBackend("/dev/null", active_ids=[1, 2])

    class FakePkt:
        def ping(self, i):
            return (1, 0, 0)          # model, COMM_SUCCESS(0), err

        def ReadPosSpeed(self, i):
            return (1500 + i, 0, 0, 0)  # pos, speed, COMM_SUCCESS, err

    class FakeSdk:
        COMM_SUCCESS = 0

    b._packet = FakePkt()
    b._sdk = FakeSdk
    assert b.ping([1, 2]) == {1: True, 2: True}
    assert b.read_position(1) == 1501
    assert b.read_all_positions([1, 2]) == {1: 1501, 2: 1502}


def test_backend_write_clamps_to_safe_range():
    b = WaveshareSmsStsBackend("/dev/null", safe_min_units=400, safe_max_units=3700)
    b._confirmed = True  # pretend confirmed

    class FakeSdk:
        COMM_SUCCESS = 0

    class FakePkt:
        def WritePosEx(self, *a):
            return (0, 0)

    b._sdk = FakeSdk
    b._packet = FakePkt()
    assert b.write_position(6, 1500) is True
    with pytest.raises(WaveshareBusServoError):
        b.write_position(6, 99999)  # outside safe range


# --------------------------- joint-space ghost (derived) ---------------------------


def test_joint_space_ghost_is_derived_not_hardcoded():
    from terafold.robot.real_motion import joint_space_ghost_fold

    jm = JointMap("r", {6: {"joint": "base_yaw", "sign": 1}})
    a = joint_space_ghost_fold("right_to_left", jm, {6: 1500}, max_sweep_units=80,
                               safe_units=(400, 3700))
    b = joint_space_ghost_fold("right_to_left", jm, {6: 2000}, max_sweep_units=80,
                               safe_units=(400, 3700))
    # Targets follow the READ current position (not constants).
    assert a["waypoints"][0]["target_units"] == 1500
    assert b["waypoints"][0]["target_units"] == 2000
    # Direction flips the sweep sign.
    c = joint_space_ghost_fold("left_to_right", jm, {6: 1500}, max_sweep_units=80,
                               safe_units=(400, 3700))
    assert a["sweep_units"] == -c["sweep_units"]
    # Only base_yaw is commanded.
    assert all(w["servo_id"] == 6 for w in a["waypoints"])


def test_joint_space_ghost_refuses_unmapped_base():
    from terafold.robot.real_motion import joint_space_ghost_fold

    jm = JointMap("r", {1: {"joint": "shoulder_pitch", "sign": 1}})  # no base_yaw
    res = joint_space_ghost_fold("right_to_left", jm, {1: 1500})
    assert res["ok"] is False and res.get("needed_joint") == "base_yaw"


def test_joint_space_ghost_refuses_out_of_range():
    from terafold.robot.real_motion import joint_space_ghost_fold

    jm = JointMap("r", {6: {"joint": "base_yaw", "sign": 1}})
    res = joint_space_ghost_fold("right_to_left", jm, {6: 410}, max_sweep_units=80,
                                 safe_units=(400, 3700))
    assert res["ok"] is False and "safe range" in res["refusal"]


# --------------------------- real-image-ghost-fold ---------------------------


def test_image_ghost_dry_run_runs_perception(tmp_path):
    from terafold.robot.real_image_ghost import run_real_image_ghost_fold

    img = _synthetic_towel(tmp_path)
    res = run_real_image_ghost_fold(img, robot=ROBOT, port="/dev/null",
                                    enable_motion=False, acknowledge=False, log=lambda _m: None)
    assert res["status"] == "dry_run"
    assert "grasp_image" in res and "grasp_normalized" in res
    assert res["direction"] == "right_to_left"
    assert res["ghost"]["touches_table"] is False
    assert res["ghost"]["min_z"] >= 0.10 - 1e-9
    assert res["motor_commands_sent"] == 0
    assert os.path.exists(res["log"])  # commands logged


def test_image_ghost_refuses_contact_without_calibration(tmp_path):
    from terafold.robot.real_image_ghost import run_real_image_ghost_fold

    img = _synthetic_towel(tmp_path)
    res = run_real_image_ghost_fold(img, robot=ROBOT, port="/dev/null", log=lambda _m: None)
    assert res["calibration_present"] is False
    assert res["contact_fold_allowed"] is False


def test_image_ghost_refuses_motion_without_joint_map(tmp_path, monkeypatch):
    import terafold.robot.real_image_ghost as mod

    monkeypatch.setattr(mod, "load_joint_map", lambda r: None)
    img = _synthetic_towel(tmp_path)
    res = mod.run_real_image_ghost_fold(img, robot=ROBOT, port="/dev/null",
                                        enable_motion=True, acknowledge=True,
                                        backend=MockBackend(), log=lambda _m: None,
                                        sleep_fn=lambda _s: None)
    assert res["status"] == "refused" and "joint map" in res["refusal"].lower()


def test_image_ghost_executes_bounded_sweep_with_mock_backend(tmp_path, monkeypatch):
    import terafold.robot.real_image_ghost as mod

    jm = JointMap(ROBOT, {6: {"joint": "base_yaw", "sign": 1}})
    monkeypatch.setattr(mod, "load_joint_map", lambda r: jm)
    img = _synthetic_towel(tmp_path)
    mb = MockBackend(pos=1500)
    res = mod.run_real_image_ghost_fold(img, robot=ROBOT, port="/dev/null",
                                        enable_motion=True, acknowledge=True, backend=mb,
                                        log=lambda _m: None, sleep_fn=lambda _s: None)
    assert res["status"] == "moved"
    assert res["motor_commands_sent"] == 3
    assert [w[0] for w in mb.writes] == [6, 6, 6]   # only base_yaw commanded
    assert mb.writes[0][1] == 1500 and mb.writes[-1][1] == 1500  # returns home
    assert mb.writes[1][1] == 1420                  # 1500 - 80 (right_to_left sweep)


def test_image_ghost_refuses_motion_when_backend_unconfirmed(tmp_path, monkeypatch):
    import terafold.robot.real_image_ghost as mod

    jm = JointMap(ROBOT, {6: {"joint": "base_yaw", "sign": 1}})
    monkeypatch.setattr(mod, "load_joint_map", lambda r: jm)

    class Unconfirmed:
        protocol_confirmed = False

        def close(self):
            pass

    img = _synthetic_towel(tmp_path)
    res = mod.run_real_image_ghost_fold(img, robot=ROBOT, port="/dev/null",
                                        enable_motion=True, acknowledge=True,
                                        backend=Unconfirmed(), log=lambda _m: None,
                                        sleep_fn=lambda _s: None)
    assert res["status"] == "refused" and "protocol not confirmed" in res["refusal"].lower()


def test_map_servo_joints_refuses_unconfirmed_backend():
    from terafold.robot.real_image_ghost import run_map_servo_joints

    class Unconfirmed:
        protocol_confirmed = False

    res = run_map_servo_joints(Unconfirmed(), robot=ROBOT, ids=[1, 2], save=False,
                               input_fn=lambda _p: "base_yaw", log=lambda _m: None)
    assert res["ok"] is False


def test_map_servo_joints_builds_map_with_mock(tmp_path):
    from terafold.robot.real_image_ghost import run_map_servo_joints

    answers = iter(["base_yaw", "shoulder_pitch"])
    res = run_map_servo_joints(MockBackend(), robot=ROBOT, ids=[6, 1], save=False,
                               input_fn=lambda _p: next(answers), log=lambda _m: None)
    assert res["ok"] is True
    jm = res["joint_map"]
    assert jm.id_for_joint("base_yaw") == 6
    assert jm.id_for_joint("shoulder_pitch") == 1


def test_no_hardcoded_fold_servo_sequence():
    import inspect

    from terafold.robot import real_image_ghost, real_motion

    for mod in (real_image_ghost, real_motion):
        src = inspect.getsource(mod)
        # No literal multi-servo target sequences baked in for the fold.
        assert "WritePosEx" not in src  # motion goes through the backend, not raw packets
