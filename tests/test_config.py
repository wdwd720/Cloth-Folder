"""Config schema validation catches invalid values."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from terafold.config.schema import (
    FoldConfig,
    TaskConfig,
    WorkspaceConfig,
)


def test_defaults_are_valid_and_safe():
    cfg = TaskConfig()
    assert cfg.task_name
    assert cfg.robot.dry_run_default is True  # safety default
    assert cfg.fold.num_waypoints >= 4


def test_workspace_rejects_inverted_bounds():
    with pytest.raises(ValidationError):
        WorkspaceConfig(x_min_m=1.0, x_max_m=0.0)
    with pytest.raises(ValidationError):
        WorkspaceConfig(z_min_m=0.5, z_max_m=0.1)


def test_fold_arc_must_clear_lift():
    with pytest.raises(ValidationError):
        FoldConfig(lift_height_m=0.10, arc_height_m=0.05)


def test_fold_rejects_nonpositive_heights():
    with pytest.raises(ValidationError):
        FoldConfig(lift_height_m=-0.01)


def test_cloth_rejects_nonpositive_size():
    with pytest.raises(ValidationError):
        TaskConfig(cloth={"expected_width_m": 0.0})


def test_workspace_contains():
    ws = WorkspaceConfig()
    assert ws.contains_xy(0.35, 0.25)
    assert not ws.contains_xy(-1.0, 0.25)
    assert ws.contains_xyz(0.35, 0.25, 0.1)
    assert not ws.contains_xyz(0.35, 0.25, 10.0)


def test_task_forbids_unknown_keys():
    with pytest.raises(ValidationError):
        TaskConfig(unknown_field=123)
