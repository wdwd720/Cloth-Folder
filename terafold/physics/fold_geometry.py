"""Cloth-specific fold geometry built on top of the pure :mod:`terafold.math`.

This is the geometric *prior*: given detected cloth keypoints it computes the
crease, the grasp point, and the place point (by reflection across the crease).
It also predicts what the cloth *should* look like after a perfect fold, which
the success scorer compares against the observed result.

Crucially, none of this assumes a perfect rectangle: the crease is derived from
edge midpoints (robust to skew), grasp/place strategies operate on whatever
corners the perception layer reports, and a ``rectangularity`` confidence is
surfaced so the planner / residual model can down-weight the prior when the
cloth is clearly deformed.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from terafold.math.geometry import (
    midpoint,
    reflect_point_across_line,
)
from terafold.physics.cloth_state import (
    ClothKeypoints,
    FoldLine,
    GraspPlacePair,
)

__all__ = [
    "compute_fold_line",
    "compute_grasp_point",
    "compute_place_point",
    "reflect_keypoints_across_fold",
    "predict_folded_corners",
    "predict_folded_footprint",
    "grasp_place_from_keypoints",
    "GRASP_STRATEGIES",
]

GRASP_STRATEGIES = (
    "right_edge_midpoint",
    "left_edge_midpoint",
    "top_edge_midpoint",
    "bottom_edge_midpoint",
    "right_top_corner",
    "right_bottom_corner",
    "center",
)


def compute_fold_line(kp: ClothKeypoints, direction: str = "right_to_left") -> FoldLine:
    """Crease line for the fold, derived from corner geometry (skew-robust)."""
    return FoldLine.from_corners(kp, direction=direction)


def compute_grasp_point(kp: ClothKeypoints, strategy: str = "right_edge_midpoint") -> np.ndarray:
    """Initial grasp point from a named strategy.

    The default for a right-to-left half fold is the *right edge midpoint*,
    which is more rotationally stable than grasping a corner (see
    :func:`terafold.physics.quasi_static.estimate_grasp_stability`). The learned
    residual / keypoint model can override this with a data-driven grasp.
    """
    strategy = strategy.lower()
    if strategy == "right_edge_midpoint":
        return kp.mid_right
    if strategy == "left_edge_midpoint":
        return kp.mid_left
    if strategy == "top_edge_midpoint":
        return kp.mid_top
    if strategy == "bottom_edge_midpoint":
        return kp.mid_bottom
    if strategy == "right_top_corner":
        return kp.top_right.copy()
    if strategy == "right_bottom_corner":
        return kp.bottom_right.copy()
    if strategy == "center":
        return kp.center
    raise ValueError(f"unknown grasp strategy {strategy!r}; options={GRASP_STRATEGIES}")


def compute_place_point(grasp: np.ndarray, fold_line: FoldLine) -> np.ndarray:
    """Place point = reflection of the grasp across the crease (the fold target)."""
    return reflect_point_across_line(grasp, fold_line.point, fold_line.direction)


def reflect_keypoints_across_fold(
    kp: ClothKeypoints, fold_line: FoldLine, side: str = "right"
) -> ClothKeypoints:
    """Predict post-fold corners by reflecting the *moving* half across the crease.

    For a right-to-left fold the right half (TR, BR) reflects onto the left
    half; the stationary corners (TL, BL) keep their positions. The result is
    the idealized folded shape used by the success scorer.
    """
    side = side.lower()
    tl, tr, br, bl = kp.top_left, kp.top_right, kp.bottom_right, kp.bottom_left

    def refl(p):
        return reflect_point_across_line(p, fold_line.point, fold_line.direction)

    if side == "right":
        new_tr, new_br = refl(tr), refl(br)
        new_tl, new_bl = tl.copy(), bl.copy()
    elif side == "left":
        new_tl, new_bl = refl(tl), refl(bl)
        new_tr, new_br = tr.copy(), br.copy()
    elif side == "top":
        new_tl, new_tr = refl(tl), refl(tr)
        new_bl, new_br = bl.copy(), br.copy()
    elif side == "bottom":
        new_bl, new_br = refl(bl), refl(br)
        new_tl, new_tr = tl.copy(), tr.copy()
    else:
        raise ValueError(f"unknown side {side!r}")
    return ClothKeypoints(
        top_left=new_tl, top_right=new_tr, bottom_right=new_br, bottom_left=new_bl
    )


def predict_folded_footprint(
    kp: ClothKeypoints, direction: str = "right_to_left"
) -> ClothKeypoints:
    """Predict the cloth *footprint* after a perfect half fold.

    When the moving flap is folded over the crease it lands on top of the
    stationary half, so the post-fold outline the camera sees is exactly the
    **stationary half** quad (area ~ half the original). This is the idealized
    "after" shape the success scorer compares against the observed result.

    The four returned points are ordered TL, TR, BR, BL of the *folded* shape.
    """
    half_top = midpoint(kp.top_left, kp.top_right)
    half_bottom = midpoint(kp.bottom_right, kp.bottom_left)
    half_left = midpoint(kp.bottom_left, kp.top_left)
    half_right = midpoint(kp.top_right, kp.bottom_right)
    if direction == "right_to_left":  # left half remains
        return ClothKeypoints(kp.top_left, half_top, half_bottom, kp.bottom_left)
    if direction == "left_to_right":  # right half remains
        return ClothKeypoints(half_top, kp.top_right, kp.bottom_right, half_bottom)
    if direction == "top_to_bottom":  # bottom half remains
        return ClothKeypoints(half_left, half_right, kp.bottom_right, kp.bottom_left)
    if direction == "bottom_to_top":  # top half remains
        return ClothKeypoints(kp.top_left, kp.top_right, half_right, half_left)
    raise ValueError(f"unknown direction {direction!r}")


def predict_folded_corners(
    kp: ClothKeypoints, direction: str = "right_to_left"
) -> ClothKeypoints:
    """Alias for :func:`predict_folded_footprint` (the post-fold outline)."""
    return predict_folded_footprint(kp, direction)


def grasp_place_from_keypoints(
    kp: ClothKeypoints,
    direction: str = "right_to_left",
    grasp_strategy: str = "right_edge_midpoint",
    grasp_override: Optional[np.ndarray] = None,
) -> GraspPlacePair:
    """Full geometric grasp/place pair for a fold.

    If ``grasp_override`` is given (e.g. a learned grasp keypoint), it is used
    instead of the strategy heuristic; the place point is always the reflection
    of the (chosen) grasp across the crease.
    """
    fold_line = compute_fold_line(kp, direction)
    grasp = (
        np.asarray(grasp_override, dtype=np.float64).reshape(-1)
        if grasp_override is not None
        else compute_grasp_point(kp, grasp_strategy)
    )
    place = compute_place_point(grasp, fold_line)
    return GraspPlacePair(grasp=grasp, place=place, confidence=float(kp.rectangularity()))
