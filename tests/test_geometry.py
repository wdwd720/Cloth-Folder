"""Geometry primitives."""

from __future__ import annotations

import numpy as np

from terafold.math.geometry import (
    fold_line_from_corners,
    midpoint,
    order_corners_clockwise,
    polygon_area,
    polygon_centroid,
    project_point_onto_line,
)


def test_polygon_area_unit_square():
    sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    assert abs(polygon_area(sq) - 1.0) < 1e-9


def test_polygon_area_order_independent():
    sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    assert abs(polygon_area(sq) - polygon_area(sq[::-1])) < 1e-9


def test_polygon_centroid_square():
    sq = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], float)
    assert np.allclose(polygon_centroid(sq), [1.0, 1.0], atol=1e-9)


def test_midpoint():
    assert np.allclose(midpoint([0, 0], [2, 4]), [1, 2])


def test_projection_onto_line():
    c = np.array([0.0, 0.0])
    d = np.array([1.0, 0.0])
    assert np.allclose(project_point_onto_line([3.0, 5.0], c, d), [3.0, 0.0])


def test_vertical_crease_direction():
    # An axis-aligned rectangle's right-to-left crease is vertical (||(0,1)||).
    p, d = fold_line_from_corners([0, 0], [4, 0], [0, 2], [4, 2], "right_to_left")
    assert abs(d[0]) < 1e-9 and abs(abs(d[1]) - 1.0) < 1e-9
    assert np.allclose(p, [2.0, 1.0])


def test_order_corners_clockwise():
    pts = np.array([[4, 2], [0, 0], [0, 2], [4, 0]], float)  # scrambled
    ordered = order_corners_clockwise(pts)
    assert ordered.shape == (4, 2)
    # First corner is the top-left-most (min x+y).
    assert np.allclose(ordered[0], [0, 0])
