"""Fold direction is authoritative for grasp/place — even over a detector keypoint.

For ``right_to_left``: grasp on the RIGHT edge, place reflects to the LEFT.
For ``left_to_right``: grasp on the LEFT edge, place reflects to the RIGHT.
A detector (Claude/learned) grasp on the wrong side is ignored.
"""

from __future__ import annotations

import numpy as np

from terafold.config.schema import TaskConfig
from terafold.physics.cloth_state import ClothKeypoints, FoldState
from terafold.physics.fold_geometry import compute_fold_line, grasp_place_from_keypoints
from terafold.planning.fold_plan import plan_fold
from terafold.planning.fold_task import FoldTask

# TL, TR, BR, BL — axis-aligned so "right" == larger x.
CORNERS = ([0.0, 0.0], [10.0, 0.0], [10.0, 8.0], [0.0, 8.0])


def _kp(grasp=None, place=None):
    return ClothKeypoints(*CORNERS, grasp=grasp, place=place)


def test_right_to_left_grasp_right_place_left():
    kp = _kp()
    fl = compute_fold_line(kp, "right_to_left")
    gp = grasp_place_from_keypoints(kp, direction="right_to_left")
    assert gp.grasp[0] > fl.point[0]  # grasp on the RIGHT of the crease
    assert gp.place[0] < fl.point[0]  # place reflects to the LEFT


def test_left_to_right_grasp_left_place_right():
    kp = _kp()
    fl = compute_fold_line(kp, "left_to_right")
    gp = grasp_place_from_keypoints(kp, direction="left_to_right")
    assert gp.grasp[0] < fl.point[0]  # grasp on the LEFT
    assert gp.place[0] > fl.point[0]  # place reflects to the RIGHT


def test_conflicting_detector_grasp_is_ignored():
    # Detector (e.g. Claude) puts grasp on the LEFT for a right_to_left fold.
    kp = _kp(grasp=[1.0, 4.0], place=[9.0, 4.0])
    fl = compute_fold_line(kp, "right_to_left")
    gp = grasp_place_from_keypoints(kp, direction="right_to_left", grasp_override=kp.grasp)
    assert gp.grasp[0] > fl.point[0]  # recomputed onto the RIGHT edge
    assert gp.place[0] < fl.point[0]
    assert "rejected_overrides" in gp.metadata


def test_consistent_detector_grasp_is_honored():
    kp = _kp()
    fl = compute_fold_line(kp, "right_to_left")
    good = np.array([9.5, 5.0])  # on the right side, refining the midpoint
    gp = grasp_place_from_keypoints(kp, direction="right_to_left", grasp_override=good)
    assert np.allclose(gp.grasp, good)
    assert gp.place[0] < fl.point[0]


def test_plan_fold_overrides_wrong_direction_claude_grasp():
    # Full pipeline: Claude labels a left-edge grasp for a right_to_left task.
    task = FoldTask.from_config(TaskConfig())  # fold_towel_half_right_to_left
    assert task.direction == "right_to_left"
    kp = ClothKeypoints([60, 70], [190, 60], [200, 180], [70, 190],
                        grasp=[65, 130], place=[195, 120])  # wrong (left-to-right)
    plan = plan_fold(FoldState(keypoints=kp, frame="image", image_shape=(256, 256)), task)
    crease_x = plan.fold_line.point[0]
    assert plan.grasp_place.grasp[0] > crease_x  # forced onto the RIGHT
    assert plan.grasp_place.place[0] < crease_x  # reflects to the LEFT


def test_place_is_reflection_of_grasp_all_directions():
    from terafold.math.geometry import reflect_point_across_line

    for direction in ("right_to_left", "left_to_right", "top_to_bottom", "bottom_to_top"):
        kp = _kp()
        gp = grasp_place_from_keypoints(kp, direction=direction)
        fl = compute_fold_line(kp, direction)
        expected = reflect_point_across_line(gp.grasp, fl.point, fl.direction)
        assert np.allclose(gp.place, expected, atol=1e-9)
