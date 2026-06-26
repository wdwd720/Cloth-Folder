"""Homography solve / apply / invert consistency."""

from __future__ import annotations

import numpy as np

from terafold.math.frames import pixel_to_table_xy, table_xy_to_pixel
from terafold.math.transforms import (
    apply_homography,
    compute_homography,
    homography_reprojection_error,
    invert_homography,
)


def _square_to_rect():
    src = np.array([[0, 0], [200, 0], [200, 200], [0, 200]], float)
    dst = np.array([[0.0, 0.0], [0.7, 0.0], [0.7, 0.5], [0.0, 0.5]], float)
    return src, dst


def test_homography_fits_correspondences():
    src, dst = _square_to_rect()
    H = compute_homography(src, dst)
    assert np.allclose(apply_homography(H, src), dst, atol=1e-6)
    assert homography_reprojection_error(H, src, dst) < 1e-6


def test_homography_roundtrip_inverse():
    src, dst = _square_to_rect()
    H = compute_homography(src, dst)
    Hinv = invert_homography(H)
    back = apply_homography(Hinv, dst)
    assert np.allclose(back, src, atol=1e-6)


def test_pixel_table_roundtrip():
    src, dst = _square_to_rect()
    H = compute_homography(src, dst)
    uv = np.array([123.0, 77.0])
    xy = pixel_to_table_xy(H, uv)
    uv2 = table_xy_to_pixel(H, xy)
    assert np.allclose(uv, uv2, atol=1e-6)


def test_homography_overdetermined_least_squares():
    # 6 correspondences (perfect) still recover the exact mapping.
    src = np.array([[0, 0], [200, 0], [200, 200], [0, 200], [100, 0], [100, 200]], float)
    H_true = compute_homography(src[:4], np.array(
        [[0, 0], [0.7, 0], [0.7, 0.5], [0, 0.5]], float))
    dst = apply_homography(H_true, src)
    H = compute_homography(src, dst)
    assert homography_reprojection_error(H, src, dst) < 1e-6
