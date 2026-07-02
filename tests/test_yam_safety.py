"""YAM safety clamps + shadow-mode gates: pure functions, no hardware possible."""

from __future__ import annotations

import pytest

from terafold.yam import safety
from terafold.yam.config import load_yam_config
from terafold.yam.safety import (
    HardwareExecutionForbidden,
    ShadowModeViolation,
    assert_no_hardware_execution,
    assert_shadow_mode,
    clamp_joint_delta,
    clamp_gripper_delta,
    hardware_execution_allowed,
    validate_workspace_bounds,
)


# -- clamps -----------------------------------------------------------------


def test_large_joint_actions_are_clamped():
    current = [0.0, 0.0, 0.0]
    target = [1.0, -2.0, 0.02]  # first two are wildly too big
    out = clamp_joint_delta(current, target, max_delta_rad=0.05)
    assert out == pytest.approx([0.05, -0.05, 0.02])


def test_small_joint_actions_pass_through():
    out = clamp_joint_delta([0.1, -0.2], [0.12, -0.21], max_delta_rad=0.05)
    assert out == pytest.approx([0.12, -0.21])


def test_clamp_joint_delta_rejects_length_mismatch():
    with pytest.raises(ValueError):
        clamp_joint_delta([0.0], [0.0, 0.0], 0.05)


def test_clamp_joint_delta_rejects_negative_limit():
    with pytest.raises(ValueError):
        clamp_joint_delta([0.0], [0.0], -0.1)


def test_gripper_delta_is_clamped_both_directions():
    assert clamp_gripper_delta(0.5, 1.0, 0.1) == pytest.approx(0.6)
    assert clamp_gripper_delta(0.5, 0.0, 0.1) == pytest.approx(0.4)
    assert clamp_gripper_delta(0.5, 0.55, 0.1) == pytest.approx(0.55)


def test_workspace_bounds_inside_and_outside():
    bounds = {"x": [0.1, 0.8], "y": [-0.5, 0.5], "z": [0.01, 0.6]}
    assert validate_workspace_bounds((0.4, 0.0, 0.3), bounds) == []
    violations = validate_workspace_bounds((0.9, -0.7, 0.3), bounds)
    assert len(violations) == 2
    assert any("x=" in v for v in violations)
    assert any("y=" in v for v in violations)


def test_workspace_bounds_bad_point_shape():
    assert validate_workspace_bounds((0.1, 0.2), {"x": [0, 1]}) != []


# -- shadow mode ------------------------------------------------------------


def test_reference_config_passes_shadow_mode():
    assert_shadow_mode(load_yam_config())  # must not raise


def test_shadow_mode_refuses_autonomy_dict_and_object():
    with pytest.raises(ShadowModeViolation):
        assert_shadow_mode({"autonomous_execution_enabled": True})

    class Cfg:
        autonomous_execution_enabled = True

    with pytest.raises(ShadowModeViolation):
        assert_shadow_mode(Cfg())


def test_hardware_execution_defaults_false():
    assert safety.HARDWARE_EXECUTION_IMPLEMENTED is False
    assert hardware_execution_allowed() is False
    # Even both dangerous flags cannot unlock execution this sprint.
    assert hardware_execution_allowed(True, True) is False


def test_assert_no_hardware_execution_is_noop_for_shadow_work():
    assert_no_hardware_execution()  # no flags -> fine


@pytest.mark.parametrize("enable,ack", [(True, False), (False, True), (True, True)])
def test_any_execution_attempt_is_refused_this_sprint(enable, ack):
    with pytest.raises(HardwareExecutionForbidden):
        assert_no_hardware_execution(enable_yam_motion=enable,
                                     i_understand_this_moves_the_yam=ack)


def test_future_execution_still_requires_both_flags(monkeypatch):
    """Even when a future sprint implements execution, one flag is never enough."""
    monkeypatch.setattr(safety, "HARDWARE_EXECUTION_IMPLEMENTED", True)
    with pytest.raises(HardwareExecutionForbidden):
        safety.assert_no_hardware_execution(enable_yam_motion=True)
    with pytest.raises(HardwareExecutionForbidden):
        safety.assert_no_hardware_execution(i_understand_this_moves_the_yam=True)
    # both explicit flags -> permitted only in that future sprint
    safety.assert_no_hardware_execution(enable_yam_motion=True,
                                        i_understand_this_moves_the_yam=True)


def test_yam_package_has_no_hardware_or_heavy_imports_at_import_time():
    """Importing every terafold.yam module must pull in no SDK/torch/rerun.

    Run in a subprocess so fake modules injected by other tests can't leak in.
    """
    import os
    import subprocess
    import sys

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = (
        "import sys\n"
        "mods = ['terafold.yam.safety', 'terafold.yam.config', 'terafold.yam.assets',\n"
        "        'terafold.yam.dataset', 'terafold.yam.molmoact2_smoke',\n"
        "        'terafold.yam.shadow', 'terafold.yam.rerun_logger']\n"
        "import importlib\n"
        "[importlib.import_module(m) for m in mods]\n"
        "forbidden = [m for m in ('serial', 'torch', 'transformers', 'rerun',\n"
        "                         'dynamixel_sdk', 'can') if m in sys.modules]\n"
        "assert not forbidden, f'heavy/hardware modules imported: {forbidden}'\n"
    )
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                          cwd=repo)
    assert proc.returncode == 0, proc.stderr
