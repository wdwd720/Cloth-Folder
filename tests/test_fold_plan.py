"""Fold planner: trajectory ordering, workspace containment, reflection target."""

from __future__ import annotations

import numpy as np

from terafold.config.schema import TaskConfig
from terafold.math.geometry import reflect_point_across_line
from terafold.physics.cloth_state import ClothKeypoints, FoldState
from terafold.planning.fold_plan import FoldPlan, plan_fold
from terafold.planning.fold_task import FoldTask
from terafold.planning.trajectory import FOLD_PHASES, trajectory_to_actions


def _plan():
    task = FoldTask.from_config(TaskConfig())
    kp = ClothKeypoints([60, 70], [190, 60], [200, 180], [70, 190])
    fs = FoldState(keypoints=kp, frame="image", image_shape=(256, 256))
    return plan_fold(fs, task), task


def test_plan_has_enough_waypoints():
    plan, task = _plan()
    assert plan.trajectory.num_waypoints >= task.fold.num_waypoints


def test_trajectory_phase_order_is_a_subsequence():
    plan, _ = _plan()
    # Collapse consecutive duplicate phases, then check it is a subsequence of FOLD_PHASES.
    collapsed = []
    for p in plan.trajectory.phases:
        if not collapsed or collapsed[-1] != p:
            collapsed.append(p)
    idx = 0
    for p in collapsed:
        while idx < len(FOLD_PHASES) and FOLD_PHASES[idx] != p:
            idx += 1
        assert idx < len(FOLD_PHASES), f"phase {p} out of order: {collapsed}"


def test_trajectory_stays_in_workspace():
    plan, task = _plan()
    for p in plan.trajectory.waypoints:
        assert task.workspace.contains_xyz(float(p[0]), float(p[1]), float(p[2]))


def test_times_monotonic_and_arc_lifts():
    plan, task = _plan()
    assert np.all(np.diff(plan.trajectory.times) > 0)
    # The arc must rise above the table by at least the lift height.
    assert plan.trajectory.max_z() >= task.fold.lift_height_m - 1e-6


def test_place_is_reflection_of_grasp():
    plan, _ = _plan()
    fl = plan.fold_line
    expected = reflect_point_across_line(plan.grasp_place.grasp, fl.point, fl.direction)
    assert np.allclose(plan.grasp_place.place, expected, atol=1e-6)


def test_plan_serialization_roundtrip():
    plan, _ = _plan()
    plan2 = FoldPlan.from_dict(plan.to_dict())
    assert np.allclose(plan2.trajectory.waypoints, plan.trajectory.waypoints)
    assert plan2.task_name == plan.task_name


def test_actions_match_waypoints():
    plan, _ = _plan()
    actions = trajectory_to_actions(plan.trajectory)
    assert len(actions) == plan.trajectory.num_waypoints
    assert actions[0].target_ee_pose is not None
