"""Tests for the reusable motion core: interpolation, safety gate, execution.

Numpy/pyyaml only — no cv2/matplotlib/torch/serial. All hardware interaction is
through a recording :class:`MockBackend`; nothing here moves anything.
"""

from __future__ import annotations

import pytest

from terafold.robot.interpolation import (interpolate_raw_joint_targets,
                                          minimum_jerk_interpolation,
                                          trapezoidal_profile)
from terafold.robot.motion_core import (MotionPlan, MotionStep, ReadbackMonitor,
                                        abort_with_reason, return_to_start_best_effort,
                                        safe_multi_joint_move, safe_single_joint_move)
from terafold.robot.safety_gate import (MotionGate, MotionRequest,
                                        validate_targets_against_limits)


# ============================ interpolation ============================


def test_interpolate_raw_endpoints_exact_and_length():
    start = {1: 500, 2: 1000}
    goal = {1: 600, 2: 800}
    frames = interpolate_raw_joint_targets(start, goal, steps=4)
    assert len(frames) == 5  # inclusive of start and goal -> steps + 1
    assert frames[0] == {1: 500, 2: 1000}
    assert frames[-1] == {1: 600, 2: 800}
    # All targets are ints and monotone between endpoints for each servo.
    assert all(isinstance(v, int) for f in frames for v in f.values())
    seq1 = [f[1] for f in frames]
    assert seq1 == sorted(seq1)  # 500 -> 600 ascending


def test_interpolate_missing_id_holds_position():
    # Servo 2 only present in start -> held; servo 3 only in goal -> held.
    frames = interpolate_raw_joint_targets({1: 500, 2: 900}, {1: 700, 3: 1200}, steps=2)
    assert all(f[2] == 900 for f in frames)
    assert all(f[3] == 1200 for f in frames)


def test_min_jerk_monotonic_and_endpoints():
    out = minimum_jerk_interpolation(0.0, 100.0, steps=20)
    assert len(out) == 21
    assert out[0] == pytest.approx(0.0)
    assert out[-1] == pytest.approx(100.0)
    # Strictly non-decreasing for an increasing goal.
    assert all(b >= a - 1e-9 for a, b in zip(out, out[1:]))
    # Symmetric S-curve: midpoint sits at half the displacement.
    assert out[10] == pytest.approx(50.0, abs=1e-6)


def test_trapezoidal_endpoints_monotonic_and_lengths():
    out = trapezoidal_profile(10.0, 20.0, steps=10, accel_frac=0.25)
    assert len(out) == 11
    assert out[0] == pytest.approx(10.0)
    assert out[-1] == pytest.approx(20.0)
    assert all(b >= a - 1e-9 for a, b in zip(out, out[1:]))


def test_trapezoidal_accel_frac_extremes():
    # accel_frac=0 -> linear; accel_frac>=0.5 clamps to triangular. Both stay valid.
    lin = trapezoidal_profile(0.0, 10.0, steps=10, accel_frac=0.0)
    assert lin[5] == pytest.approx(5.0)
    tri = trapezoidal_profile(0.0, 10.0, steps=10, accel_frac=0.9)  # clamped to 0.5
    assert tri[0] == pytest.approx(0.0) and tri[-1] == pytest.approx(10.0)
    assert all(b >= a - 1e-9 for a, b in zip(tri, tri[1:]))


def test_interpolation_rejects_zero_steps():
    for fn in (lambda: interpolate_raw_joint_targets({1: 0}, {1: 1}, 0),
               lambda: minimum_jerk_interpolation(0.0, 1.0, 0),
               lambda: trapezoidal_profile(0.0, 1.0, 0)):
        with pytest.raises(ValueError):
            fn()


# ============================ safety gate ============================


class FakeCaps:
    def __init__(self, joint_map_present=False, calibration_valid_for_hover=False,
                 contact_allowed=False):
        self.joint_map_present = joint_map_present
        self.calibration_valid_for_hover = calibration_valid_for_hover
        self.contact_allowed = contact_allowed


def _req(**kw):
    base = dict(robot="r", servo_targets={1: 1500}, speed=300, acc=20)
    base.update(kw)
    return MotionRequest(**base)


def test_validate_targets_against_limits():
    assert validate_targets_against_limits({1: 1500, 2: 3000}, (400, 3700)) == []
    bad = validate_targets_against_limits({1: 100, 2: 5000}, (400, 3700))
    assert len(bad) == 2
    assert any("100" in m for m in bad) and any("5000" in m for m in bad)


def test_dry_run_allowed_and_flagged_non_real():
    res = MotionGate().check(_req(dry_run=True))
    assert res["allowed"] is True
    assert res["checks"]["real_motion"] is False
    assert res["checks"]["dry_run"] is True
    assert res["refusals"] == []


def test_missing_flags_block_real():
    res = MotionGate().check(_req(dry_run=False, enable_motion=False, acknowledge=False))
    assert res["allowed"] is False
    assert res["checks"]["both_flags"] is False
    assert any("enable_motion" in r for r in res["refusals"])

    # Only one flag is still not enough.
    res2 = MotionGate().check(_req(dry_run=False, enable_motion=True, acknowledge=False))
    assert res2["allowed"] is False


def test_real_with_both_flags_allowed():
    res = MotionGate().check(_req(dry_run=False, enable_motion=True, acknowledge=True))
    assert res["allowed"] is True
    assert res["checks"]["both_flags"] is True
    assert res["checks"]["real_motion"] is True


def test_target_outside_limits_blocked():
    res = MotionGate().check(_req(servo_targets={1: 50}, safe_units=(400, 3700)))
    assert res["allowed"] is False
    assert res["checks"]["targets_within_limits"] is False
    assert any("outside safe raw range" in r for r in res["refusals"])


def test_disabled_id_blocked():
    res = MotionGate().check(_req(servo_targets={1: 1500, 3: 1500}, active_ids=[1, 2]))
    assert res["allowed"] is False
    assert res["checks"]["only_active_ids"] is False
    assert any("servo 3 is not in the active set" in r for r in res["refusals"])


def test_contact_always_refused():
    # Even a clean, fully-flagged request is refused when it requests contact.
    res = MotionGate().check(_req(dry_run=False, enable_motion=True, acknowledge=True,
                                  contact=True), capabilities=FakeCaps())
    assert res["allowed"] is False
    assert any("contact folding is LOCKED" in r for r in res["refusals"])
    # Refused even with no capabilities supplied.
    res2 = MotionGate().check(_req(contact=True))
    assert res2["allowed"] is False


def test_capabilities_none_only_basic_checks_with_note():
    # requires_joint_map is ignored when capabilities is None; a note explains.
    res = MotionGate().check(_req(requires_joint_map=True, requires_calibration=True))
    assert res["allowed"] is True
    assert res["checks"]["capabilities_verified"] is False
    assert "not verified" in res["note"].lower()
    assert "joint_map_present" not in res["checks"]


def test_requires_joint_map_blocks_when_missing():
    res = MotionGate().check(_req(requires_joint_map=True), capabilities=FakeCaps(joint_map_present=False))
    assert res["allowed"] is False
    assert res["checks"]["joint_map_present"] is False
    res_ok = MotionGate().check(_req(requires_joint_map=True),
                                capabilities=FakeCaps(joint_map_present=True))
    assert res_ok["allowed"] is True


def test_requires_calibration_blocks_when_invalid():
    res = MotionGate().check(_req(requires_calibration=True),
                             capabilities=FakeCaps(calibration_valid_for_hover=False))
    assert res["allowed"] is False
    assert res["checks"]["calibration_valid_for_hover"] is False


# ============================ motion core ============================


class MockBackend:
    """Records writes; reports a confirmed protocol; reads back the last target."""

    def __init__(self, protocol_confirmed=True):
        self.protocol_confirmed = protocol_confirmed
        self.writes = []
        self.positions = {}

    def write_position(self, servo_id, target, speed=None, acc=None):
        self.writes.append((int(servo_id), int(target), speed, acc))
        self.positions[int(servo_id)] = int(target)
        return True

    def read_pos_speed(self, servo_id):
        return (self.positions.get(int(servo_id), 0), 0)

    def read_position(self, servo_id):
        return self.positions.get(int(servo_id))


class RecordingLog:
    def __init__(self):
        self.events = []

    def log(self, event, data=None):
        self.events.append((event, data))


def test_single_move_refuses_when_gate_not_ok_writes_nothing():
    be = MockBackend()
    step = MotionStep(servo_id=1, target_units=1500, speed=300, acc=20, label="x")
    res = safe_single_joint_move(be, step, gate_ok=False)
    assert res["wrote"] is False and res["refused"] is True and res["ok"] is False
    assert be.writes == []  # absolutely nothing was sent


def test_single_move_writes_once_when_gate_ok():
    be = MockBackend()
    log = RecordingLog()
    step = MotionStep(servo_id=5, target_units=1600, speed=200, acc=10, label="sweep")
    res = safe_single_joint_move(be, step, gate_ok=True, log=log)
    assert res["wrote"] is True and res["ok"] is True
    assert be.writes == [(5, 1600, 200, 10)]
    assert any(ev == "single_joint_move" for ev, _ in log.events)


def test_single_move_refuses_when_protocol_unconfirmed():
    be = MockBackend(protocol_confirmed=False)
    step = MotionStep(servo_id=1, target_units=1500, speed=300, acc=20)
    res = safe_single_joint_move(be, step, gate_ok=True)
    assert res["wrote"] is False and res["refused"] is True
    assert be.writes == []


def test_readback_settled_true_when_matches():
    be = MockBackend()
    be.write_position(1, 1500)  # now read_pos_speed -> (1500, 0)
    assert ReadbackMonitor.settled(be, 1, 1500, tol=8, speed_tol=20) is True
    # Far from target -> not settled.
    assert ReadbackMonitor.settled(be, 1, 2000, tol=8) is False


def test_readback_speed_gate_and_position_fallback():
    class MovingBackend:
        protocol_confirmed = True

        def read_pos_speed(self, sid):
            return (1500, 100)  # at target but still moving fast

    assert ReadbackMonitor.settled(MovingBackend(), 1, 1500, speed_tol=20) is False

    class PosOnlyBackend:
        protocol_confirmed = True

        def read_position(self, sid):
            return 1503

    # No speed available -> speed gate skipped; within tol -> settled.
    assert ReadbackMonitor.settled(PosOnlyBackend(), 1, 1500, tol=8) is True


def test_multi_joint_move_writes_all_steps():
    be = MockBackend()
    calls = []
    plan = MotionPlan(steps=[
        MotionStep(1, 1500, 300, 20, "a"),
        MotionStep(1, 1600, 300, 20, "b"),
        MotionStep(1, 1500, 300, 20, "c"),
    ], note="sweep+return")
    res = safe_multi_joint_move(be, plan, gate_ok=True, sleep_fn=lambda s: calls.append(s))
    assert res["ok"] is True and res["wrote"] is True and res["steps"] == 3
    assert [w[1] for w in be.writes] == [1500, 1600, 1500]
    assert len(calls) == 3 and all(s > 0 for s in calls)  # dwelled once per write
    assert all(r["settled"] for r in res["results"])  # mock reads back instantly


def test_multi_joint_move_refused_writes_nothing():
    be = MockBackend()
    plan = MotionPlan(steps=[MotionStep(1, 1500, 300, 20)])
    res = safe_multi_joint_move(be, plan, gate_ok=False, sleep_fn=lambda s: None)
    assert res["refused"] is True and res["wrote"] is False and res["steps"] == 0
    assert be.writes == []


def test_return_to_start_writes_start_targets():
    be = MockBackend()
    log = RecordingLog()
    res = return_to_start_best_effort(be, {1: 500, 2: 1000}, speed=200, acc=10, log=log)
    assert res["ok"] is True and res["wrote"] is True
    assert sorted(be.writes) == [(1, 500, 200, 10), (2, 1000, 200, 10)]
    assert any(ev == "return_to_start" for ev, _ in log.events)


def test_return_to_start_refuses_unconfirmed_protocol():
    be = MockBackend(protocol_confirmed=False)
    res = return_to_start_best_effort(be, {1: 500}, speed=200, acc=10)
    assert res["wrote"] is False and be.writes == []


def test_return_to_start_swallows_errors():
    class ErrBackend:
        protocol_confirmed = True

        def write_position(self, sid, target, speed=None, acc=None):
            raise RuntimeError("boom")

    res = return_to_start_best_effort(ErrBackend(), {1: 500, 2: 600}, speed=200, acc=10)
    assert res["ok"] is False and res["wrote"] is False
    assert set(res["errors"]) == {1, 2}  # both captured, nothing raised


def test_abort_with_reason():
    log = RecordingLog()
    res = abort_with_reason("e-stop pressed", log=log)
    assert res["aborted"] is True and res["wrote"] is False and res["ok"] is False
    assert res["reason"] == "e-stop pressed"
    assert log.events[0][0] == "abort"
