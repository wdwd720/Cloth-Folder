"""Safety: motion is blocked by default; checks reject unsafe commands."""

from __future__ import annotations

import os

import pytest

from terafold.config.schema import WorkspaceConfig
from terafold.robot.safety import (
    STOP_FILE,
    SafetyChecker,
    SafetyConfig,
    SafetyError,
    require_motion_enabled,
)


def _checker():
    ws = WorkspaceConfig(x_min_m=0, x_max_m=0.5, y_min_m=0, y_max_m=0.5, z_min_m=0, z_max_m=0.2)
    return SafetyChecker(SafetyConfig(workspace=ws))


def test_motion_blocked_without_flags():
    with pytest.raises(SafetyError):
        require_motion_enabled(False, False)
    with pytest.raises(SafetyError):
        require_motion_enabled(True, False)
    with pytest.raises(SafetyError):
        require_motion_enabled(False, True)


def test_motion_allowed_with_both_flags():
    # Should not raise.
    require_motion_enabled(True, True)


def test_check_xyz_inside_ok():
    chk = _checker()
    chk.check_xyz([0.25, 0.25, 0.1])  # no raise


def test_check_xyz_outside_raises():
    chk = _checker()
    with pytest.raises(SafetyError):
        chk.check_xyz([1.0, 0.25, 0.1])
    with pytest.raises(SafetyError):
        chk.check_xyz([0.25, 0.25, 5.0])


def test_stop_file_detected(tmp_path):
    chk = _checker()
    stop = tmp_path / STOP_FILE
    assert not chk.is_stop_requested(cwd=str(tmp_path))
    stop.write_text("stop")
    assert chk.is_stop_requested(cwd=str(tmp_path))
    os.remove(stop)
