"""Reflection math — the heart of the fold geometry."""

from __future__ import annotations

import numpy as np

from terafold.math.geometry import fold_line_from_corners, reflect_point_across_line
from terafold.math.transforms import apply_affine_2d, rigid_transform_2d


def test_reflect_twice_is_identity():
    rng = np.random.default_rng(0)
    c = np.array([1.0, -2.0])
    d = np.array([0.4, 0.9])
    for _ in range(20):
        p = rng.normal(size=2) * 5
        r = reflect_point_across_line(p, c, d)
        rr = reflect_point_across_line(r, c, d)
        assert np.allclose(rr, p, atol=1e-9)


def test_point_on_line_unchanged():
    c = np.array([2.0, 3.0])
    d = np.array([0.0, 1.0])
    for t in (-3.0, 0.0, 1.5, 7.0):
        p = c + d * t
        assert np.allclose(reflect_point_across_line(p, c, d), p, atol=1e-9)


def test_right_edge_maps_near_left_edge():
    # Vertical crease at x=0; a point on the right maps to the mirror on the left.
    c = np.array([0.0, 0.0])
    d = np.array([0.0, 1.0])
    p = np.array([3.0, 1.0])
    r = reflect_point_across_line(p, c, d)
    assert np.allclose(r, [-3.0, 1.0])


def test_reflection_works_in_3d():
    c = np.array([0.0, 0.0, 0.0])
    d = np.array([0.0, 0.0, 1.0])  # z-axis line
    p = np.array([2.0, 0.0, 5.0])
    r = reflect_point_across_line(p, c, d)
    assert np.allclose(r, [-2.0, 0.0, 5.0])


def test_fold_line_stable_under_translation():
    tl, tr, bl, br = (
        np.array([0.0, 0.0]),
        np.array([4.0, 0.0]),
        np.array([0.0, 2.0]),
        np.array([4.0, 2.0]),
    )
    p0, d0 = fold_line_from_corners(tl, tr, bl, br, "right_to_left")
    shift = np.array([10.0, -5.0])
    p1, d1 = fold_line_from_corners(tl + shift, tr + shift, bl + shift, br + shift, "right_to_left")
    # Direction unchanged; the crease point translates with the cloth.
    assert np.allclose(np.abs(d0), np.abs(d1), atol=1e-9)
    assert np.allclose(p1 - p0, shift, atol=1e-9)


def test_fold_line_stable_under_rotation():
    tl, tr, bl, br = (
        np.array([0.0, 0.0]),
        np.array([4.0, 0.0]),
        np.array([0.0, 2.0]),
        np.array([4.0, 2.0]),
    )
    p0, d0 = fold_line_from_corners(tl, tr, bl, br, "right_to_left")
    R = rigid_transform_2d(np.deg2rad(30), 1.0, -2.0)
    rtl, rtr, rbl, rbr = (apply_affine_2d(R, x) for x in (tl, tr, bl, br))
    p1, d1 = fold_line_from_corners(rtl, rtr, rbl, rbr, "right_to_left")
    # The crease direction rotates with the cloth (compare absolute cosine).
    cos = abs(float(np.dot(d0, d1)) / (np.linalg.norm(d0) * np.linalg.norm(d1)))
    rotated_d0 = apply_affine_2d(rigid_transform_2d(np.deg2rad(30), 0, 0), d0)
    cos_rot = abs(float(np.dot(rotated_d0, d1)) / (np.linalg.norm(rotated_d0) * np.linalg.norm(d1)))
    assert cos_rot > 0.999
