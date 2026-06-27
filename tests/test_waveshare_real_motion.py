"""Safety tests for the physical 7-DOF Waveshare bus-servo arm workflow.

Core invariant: nothing moves hardware by default, and every read/move refuses
until a verified bus-servo protocol is wired (there is none in the repo).
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from terafold.cli import app

runner = CliRunner()
ROBOT = "physical_7dof_waveshare"


def _out(result):
    s = result.output or ""
    try:
        s += result.stderr  # raises when stdout/stderr are mixed
    except Exception:
        pass
    return s


# --------------------------- config + adapter ---------------------------


def test_config_loads_7dof_disabled_by_default():
    from terafold.robot.arm_config import load_arm_config

    cfg = load_arm_config(ROBOT)
    assert cfg.dof == 7 and len(cfg.joint_names) == 7
    assert cfg.motion_default == "disabled" and not cfg.motion_enabled_by_default
    assert cfg.max_delta_per_test_deg <= 3.0 and cfg.table_clearance_m >= 0.10


def test_adapter_fails_safely_when_protocol_unknown():
    from terafold.robot.waveshare_bus_servo import (PROTOCOL_REFUSAL,
                                                    WaveshareBusServoAdapter,
                                                    WaveshareBusServoError)

    ad = WaveshareBusServoAdapter(dof=7)
    assert ad.protocol_confirmed is False and ad.supports_motion is False
    assert ad.read_positions()["supported"] is False
    assert ad.read_positions()["reason"] == PROTOCOL_REFUSAL
    assert ad.scan_servo_ids()["supported"] is False
    assert ad.read_position(1) is None
    with pytest.raises(WaveshareBusServoError):
        ad.write_position(1, 10.0)
    with pytest.raises(WaveshareBusServoError):
        ad.torque_enable(True)


def test_emergency_stop_calls_torque_off_if_available():
    from terafold.robot.waveshare_bus_servo import WaveshareBusServoAdapter

    class FakeBackend:
        def __init__(self):
            self.torque_off_called = False

        def torque_off(self):
            self.torque_off_called = True

    backend = FakeBackend()
    ad = WaveshareBusServoAdapter(command_backend=backend)
    assert ad.protocol_confirmed is True
    res = ad.emergency_stop()
    assert backend.torque_off_called is True
    assert res["torque_off_attempted"] and res["torque_off_ok"] and res["port_closed"]


def test_emergency_stop_safe_without_backend_prints_manual():
    from terafold.robot.waveshare_bus_servo import WaveshareBusServoAdapter

    res = WaveshareBusServoAdapter().emergency_stop()
    assert res["torque_off_attempted"] is False
    assert res["port_closed"] is True
    assert any("CUT POWER" in m for m in res["manual_instructions"])


# --------------------------- nudge / delta gates ---------------------------


def test_check_nudge_delta_refuses_large():
    from terafold.robot.real_motion import check_nudge_delta
    from terafold.robot.safety import SafetyError

    check_nudge_delta(3.0, 5.0)            # ok
    check_nudge_delta(8.0, 5.0, allow_larger=True)  # ok with override
    with pytest.raises(SafetyError):
        check_nudge_delta(9.0, 5.0)


def test_servo_nudge_refuses_without_flags():
    r = runner.invoke(app, ["servo-nudge", "--robot", ROBOT, "--servo-id", "1", "--delta-deg", "3"])
    assert r.exit_code != 0
    assert "Real motion is blocked" in _out(r)


def test_servo_nudge_refuses_large_delta():
    r = runner.invoke(app, ["servo-nudge", "--robot", ROBOT, "--servo-id", "1", "--delta-deg", "9",
                            "--enable-motion", "--i-understand-this-moves-hardware"])
    assert r.exit_code != 0
    assert "exceeds the hard limit" in _out(r)


def test_servo_nudge_refuses_at_protocol_gate():
    r = runner.invoke(app, ["servo-nudge", "--robot", ROBOT, "--servo-id", "1", "--delta-deg", "2",
                            "--enable-motion", "--i-understand-this-moves-hardware",
                            "--allow-open-loop-nudge"])
    assert r.exit_code != 0
    assert "protocol not confirmed" in _out(r)


# --------------------------- ghost fold ---------------------------


def test_ghost_fold_lifts_above_table_and_never_hard_closes(tmp_path):
    from terafold.robot.real_motion import ghost_fold_trajectory

    g = ghost_fold_trajectory("runs/demo_image/claude_result.json", height_clearance_m=0.10)
    assert g["num_waypoints"] > 0
    assert g["min_z"] >= 0.10 - 1e-9 and g["touches_table"] is False
    assert all(w["xyz"][2] >= 0.10 - 1e-9 for w in g["waypoints"])
    assert all(w["gripper"] >= 0.5 for w in g["waypoints"])  # never a hard pinch


def test_real_ghost_fold_dry_run_does_not_move():
    r = runner.invoke(app, ["real-ghost-fold", "--robot", ROBOT,
                            "--plan-json", "runs/demo_image/claude_result.json",
                            "--height-clearance-m", "0.10", "--speed", "slow", "--dry-run"])
    assert r.exit_code == 0
    out = _out(r)
    assert "DRY-RUN" in out and "nothing moved" in out
    assert "touches_table=False" in out


def test_real_ghost_fold_real_refuses_no_kinematics():
    r = runner.invoke(app, ["real-ghost-fold", "--robot", ROBOT,
                            "--plan-json", "runs/demo_image/claude_result.json",
                            "--enable-motion", "--i-understand-this-moves-hardware"])
    assert r.exit_code != 0
    assert "kinematics are unknown" in _out(r)


# --------------------------- teach / replay ---------------------------


def test_teach_replay_demo_save_and_load(tmp_path):
    from terafold.robot.real_motion import (JointDemo, load_joint_demo,
                                            record_teach_demo, save_joint_demo)

    class FakeAdapter:
        protocol_confirmed = True

        def read_positions(self):
            return {"supported": True, "positions": {1: 100, 2: 200, 3: 300}}

    demo = record_teach_demo(FakeAdapter(), robot="r", dof=3, joint_names=["a", "b", "c"],
                             poses=["home", "lift"], input_fn=lambda _p: "", log=lambda _m: None)
    assert demo.has_positions()
    path = str(tmp_path / "demo.json")
    save_joint_demo(path, demo)
    loaded = load_joint_demo(path)
    assert isinstance(loaded, JointDemo)
    assert loaded.waypoints[0]["servo_positions"] == {"1": 100, "2": 200, "3": 300} \
        or loaded.waypoints[0]["servo_positions"] == {1: 100, 2: 200, 3: 300}
    assert [w["name"] for w in loaded.waypoints] == ["home", "lift"]


def test_replay_dry_run_does_not_move(tmp_path):
    from terafold.robot.real_motion import JointDemo, save_joint_demo

    demo = JointDemo(robot=ROBOT, dof=7, joint_names=["j"],
                     waypoints=[{"name": "home", "servo_positions": {1: 500}, "gripper": 1.0}],
                     protocol_confirmed=True)
    p = str(tmp_path / "d.json")
    save_joint_demo(p, demo)
    r = runner.invoke(app, ["replay-joint-demo", "--robot", ROBOT, "--demo", p,
                            "--speed", "very_slow", "--dry-run"])
    assert r.exit_code == 0 and "DRY-RUN" in _out(r)


# --------------------------- image / table gate ---------------------------


def test_image_real_motion_gate_refuses_without_calibration():
    from terafold.robot.real_motion import IMAGE_CALIBRATION_REFUSAL, image_real_motion_gate

    gate = image_real_motion_gate(enable_motion=True, acknowledge=True)
    assert gate["allowed"] is False
    assert gate["message"] == IMAGE_CALIBRATION_REFUSAL
    assert any("calibration" in m for m in gate["missing"])


def test_real_image_fold_cli_refuses_without_calibration(tmp_path):
    img = tmp_path / "x.png"
    img.write_bytes(b"\x89PNG\r\n")
    r = runner.invoke(app, ["real-image-fold", "--robot", ROBOT, "--image", str(img),
                            "--enable-motion", "--i-understand-this-moves-hardware"])
    assert r.exit_code != 0
    assert "requires table calibration" in _out(r)


def test_servo_scan_refuses_without_protocol():
    r = runner.invoke(app, ["servo-scan", "--robot", ROBOT, "--port", "auto", "--read-only"])
    assert r.exit_code != 0
    assert "protocol not confirmed" in _out(r)
