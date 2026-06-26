"""Mock robot simulation + workspace enforcement."""

from __future__ import annotations

import numpy as np
import pytest

from terafold.config.schema import WorkspaceConfig
from terafold.robot.base import RobotAction
from terafold.robot.mock_robot import MockRobot
from terafold.robot.safety import SafetyError


def _robot():
    ws = WorkspaceConfig(x_min_m=0.0, x_max_m=0.5, y_min_m=0.0, y_max_m=0.5, z_min_m=0.0, z_max_m=0.2)
    rob = MockRobot(workspace=ws, dry_run=True)
    rob.connect()
    return rob


def test_connect_disconnect():
    rob = _robot()
    assert rob.is_connected
    rob.disconnect()
    assert not rob.is_connected


def test_in_bounds_action_returns_observation():
    rob = _robot()
    act = RobotAction(target_ee_pose=[0.25, 0.25, 0.1, np.pi, 0, 0], gripper=1.0)
    obs = rob.send_action(act)
    assert obs.ee_pose is not None
    assert np.allclose(obs.ee_pose[:3], [0.25, 0.25, 0.1], atol=1e-6)
    rob.disconnect()


def test_out_of_bounds_action_raises():
    rob = _robot()
    bad = RobotAction(target_ee_pose=[5.0, 0.25, 0.1, np.pi, 0, 0], gripper=1.0)
    with pytest.raises(SafetyError):
        rob.send_action(bad)
    rob.disconnect()


def test_gripper_state_tracked():
    rob = _robot()
    rob.close_gripper()
    assert rob.get_observation().gripper_state <= 0.5
    rob.open_gripper()
    assert rob.get_observation().gripper_state >= 0.5
    rob.disconnect()


def test_default_is_dry_run():
    assert MockRobot().dry_run is True
