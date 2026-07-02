"""Schema tests for the enriched joint map (raw home/limits, sign, confidence)."""

from __future__ import annotations

from terafold.robot.joint_map import JOINT_ENTRY_FIELDS, JointMap
from terafold.robot.real_image_ghost import run_map_servo_joints

ROBOT = "physical_7dof_waveshare"


class RichMockBackend:
    """Confirmed backend whose readback MOVES on a nudge (so sign is inferable)."""

    protocol_confirmed = True

    def __init__(self):
        self.writes = []
        self._pos = {6: 1500, 1: 2000}
        self._nudged = set()

    def read_position(self, sid):
        return self._pos.get(sid, 1500)

    def write_position(self, sid, target, speed=None, acc=None):
        self.writes.append((sid, int(target)))
        self._pos[sid] = int(target)  # readback reflects the commanded move
        return True


def test_rich_joint_map_schema_populated_and_valid():
    answers = iter(["base_yaw", "shoulder_pitch"])
    res = run_map_servo_joints(RichMockBackend(), robot=ROBOT, ids=[6, 1], save=False,
                               delta_units=40, input_fn=lambda _p: next(answers),
                               log=lambda _m: None, now=123.0)
    assert res["ok"] is True
    jm: JointMap = res["joint_map"]
    assert jm.validate() == []  # schema clean
    e6 = jm.servo_joint_map[6]
    for f in ("joint", "role", "sign", "raw_home", "raw_min", "raw_max",
              "last_observed", "confidence", "motion_notes"):
        assert f in e6, f
    assert e6["joint"] == "base_yaw" and e6["role"] == "base"
    assert e6["raw_home"] == 1500
    assert e6["raw_min"] <= e6["raw_home"] <= e6["raw_max"]
    # Sign inferred +1 from a positive nudge that actually moved.
    assert e6["sign"] == 1 and e6["confidence"] == "high"
    # Disabled/missing IDs recorded (3,4,7 are not active and not mapped here).
    assert set(res["disabled_ids"]) >= {3, 4, 7}
    assert jm.has_raw_limits() is True
    assert jm.raw_limits_for_id(6) == (e6["raw_min"], e6["raw_max"])


def test_joint_map_roundtrip_keeps_rich_fields():
    jm = JointMap(ROBOT, {6: {"joint": "base_yaw", "sign": -1, "raw_home": 1246,
                              "raw_min": 946, "raw_max": 1546, "role": "base"}},
                  disabled_ids=[3, 4, 7], operator_notes="cable branch suspect on 3,4,7")
    d = jm.to_dict()
    jm2 = JointMap.from_dict(d)
    assert jm2.sign_for_id(6) == -1
    assert jm2.raw_limits_for_id(6) == (946, 1546)
    assert jm2.disabled_ids == [3, 4, 7]
    assert jm2.operator_notes.startswith("cable branch")
    assert jm2.validate() == []


def test_joint_map_validate_catches_bad_entries():
    jm = JointMap(ROBOT, {6: {"joint": "", "sign": 3, "raw_min": 2000, "raw_max": 1000}})
    problems = jm.validate()
    assert any("joint name" in p for p in problems)
    assert any("sign" in p for p in problems)
    assert any("raw_min > raw_max" in p for p in problems)


def test_joint_entry_fields_constant_exposed():
    assert "raw_home" in JOINT_ENTRY_FIELDS and "confidence" in JOINT_ENTRY_FIELDS
