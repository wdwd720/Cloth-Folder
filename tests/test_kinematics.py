"""Tests for the kinematics scaffold.

Numpy/pyyaml only — no cv2/torch/scipy. Covers raw-unit conversion, toy FK vs
analytic 2R, damped-least-squares IK (reachable + unreachable), the custom-arm
model's IK refusal, and the model-status report.
"""

from __future__ import annotations

import math

import numpy as np

from terafold.kinematics import (
    CustomArmModel,
    DHParam,
    RawUnitScale,
    deg_to_raw,
    fk_dh,
    ik_dls_planar,
    raw_to_deg,
    report_markdown,
    robot_model_status,
    toy_planar_fk,
)

ROBOT = "physical_7dof_waveshare"


# ----------------------------------------------------------------------
# raw_units
# ----------------------------------------------------------------------


def test_raw_units_roundtrip_when_known():
    scale = RawUnitScale.confirmed(home=2048.0, scale_deg_per_unit=360.0 / 4096.0, sign=1)
    for units in (2048, 2148, 1800, 3000):
        deg = raw_to_deg(units, scale)
        assert deg is not None
        back = deg_to_raw(deg, scale)
        assert back == units  # exact round-trip (modulo integer rounding)
    # Home maps to zero degrees.
    assert abs(raw_to_deg(2048, scale)) < 1e-9


def test_raw_units_sign_applies():
    scale = RawUnitScale.confirmed(home=2048.0, scale_deg_per_unit=0.1, sign=-1)
    # With sign -1, a count above home yields negative degrees.
    assert raw_to_deg(2148, scale) < 0
    assert deg_to_raw(raw_to_deg(2148, scale), scale) == 2148


def test_raw_units_none_when_unknown():
    unknown = RawUnitScale()  # default: scale_deg_per_unit=None, known=False
    assert raw_to_deg(2048, unknown) is None
    assert deg_to_raw(10.0, unknown) is None
    # A populated-but-guessed scale is still "unknown" (known=False).
    guess = RawUnitScale.single_turn_guess()
    assert guess.scale_deg_per_unit is not None
    assert guess.known is False
    assert raw_to_deg(2048, guess) is None
    assert deg_to_raw(10.0, guess) is None


# ----------------------------------------------------------------------
# fk
# ----------------------------------------------------------------------


def test_toy_planar_fk_matches_analytic():
    l1, l2 = 1.0, 0.5
    for q1, q2 in [(0.0, 0.0), (0.3, -0.4), (1.1, 0.7), (-0.6, 1.2)]:
        x, y = toy_planar_fk(l1, l2, q1, q2)
        ex = l1 * math.cos(q1) + l2 * math.cos(q1 + q2)
        ey = l1 * math.sin(q1) + l2 * math.sin(q1 + q2)
        assert abs(x - ex) < 1e-12
        assert abs(y - ey) < 1e-12


def test_fk_dh_planar_matches_toy_fk():
    """A 2-link planar DH chain reproduces the analytic 2R end-effector position."""
    l1, l2 = 1.0, 0.5
    params = [DHParam(a=l1), DHParam(a=l2)]
    for q1, q2 in [(0.0, 0.0), (0.3, -0.4), (1.1, 0.7)]:
        T = fk_dh(params, [q1, q2])
        assert T.shape == (4, 4)
        # Bottom row of a homogeneous transform.
        assert np.allclose(T[3], [0, 0, 0, 1])
        x, y = toy_planar_fk(l1, l2, q1, q2)
        assert abs(T[0, 3] - x) < 1e-9
        assert abs(T[1, 3] - y) < 1e-9
        assert abs(T[2, 3]) < 1e-9  # planar arm stays in z=0


# ----------------------------------------------------------------------
# ik_dls
# ----------------------------------------------------------------------


def test_ik_dls_reachable_recovers_pose():
    l1, l2 = 1.0, 0.5
    q_true = (0.4, 0.8)
    target = toy_planar_fk(l1, l2, *q_true)
    res = ik_dls_planar(target, l1, l2, tol=1e-4)
    assert res["ok"] is True
    assert res["error"] <= 1e-4
    x, y = toy_planar_fk(l1, l2, res["q"][0], res["q"][1])
    assert math.hypot(x - target[0], y - target[1]) < 1e-4
    assert 1 <= res["iters"] <= 200


def test_ik_dls_unreachable_returns_not_ok():
    l1, l2 = 1.0, 0.5
    # Far beyond the l1 + l2 = 1.5 reach.
    res = ik_dls_planar((5.0, 5.0), l1, l2, tol=1e-4)
    assert res["ok"] is False
    assert res["error"] > 1e-4


# ----------------------------------------------------------------------
# custom_arm_model
# ----------------------------------------------------------------------


def test_custom_arm_model_refuses_ik():
    model = CustomArmModel.from_robot(ROBOT)
    assert model.robot == ROBOT
    assert model.validated is False
    assert model.can_do_contact_ik() is False
    refusal = model.ik_refused()
    assert refusal["ok"] is False
    assert refusal["refused"] is True
    assert refusal["validated"] is False
    assert "REFUSED" in refusal["reason"]
    assert "validated" in refusal["reason"].lower()
    # ik() returns the same refusal while unvalidated (no joint map on disk).
    assert model.ik(target=(0.1, 0.2, 0.3)) == refusal


# ----------------------------------------------------------------------
# model_status
# ----------------------------------------------------------------------


def test_robot_model_status_reflects_unvalidated_custom_arm():
    status = robot_model_status(ROBOT)
    assert status["robot"] == ROBOT
    assert status["validated"] is False
    assert status["allowed_for_contact"] is False
    # IK must be refused / called out as unvalidated.
    lowered = status["ik_status"].lower()
    assert "refused" in lowered or "unvalidated" in lowered
    # No joint map on disk for this arm -> nothing mapped, raw->angle unknown.
    assert status["joint_map_present"] in (True, False)
    if not status["joint_map_present"]:
        assert status["known_joint_names"] == []
        assert status["raw_limits_known"] is False
        assert "UNKNOWN" in status["raw_to_angle_status"]
    # The markdown report mentions IK and never claims a validated model.
    md = report_markdown(status)
    assert "IK" in md
    assert "validated model: **False**" in md
    assert "cartesian-ik" not in status["allowed_capabilities"]
