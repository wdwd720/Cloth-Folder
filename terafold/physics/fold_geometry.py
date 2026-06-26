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
    as_vector,
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
    "DIRECTION_GRASP_STRATEGY",
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

# The fold DIRECTION determines which edge is the moving (grasp) edge. This is
# authoritative — it overrides a config strategy or a detector keypoint that
# would grab the wrong side. For ``fold_towel_half_right_to_left`` the right half
# moves, so the grasp is the right-edge midpoint and the place is its reflection
# across the vertical centre crease (landing on the left edge).
DIRECTION_GRASP_STRATEGY = {
    "right_to_left": "right_edge_midpoint",
    "left_to_right": "left_edge_midpoint",
    "top_to_bottom": "top_edge_midpoint",
    "bottom_to_top": "bottom_edge_midpoint",
}


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


def _crease_normal(fold_line: FoldLine) -> np.ndarray:
    """In-plane normal to the crease (perpendicular to its direction)."""
    d = as_vector(fold_line.direction)[:2]
    return np.array([-d[1], d[0]], dtype=np.float64)


def _side_sign(point, fold_line: FoldLine) -> float:
    """Signed offset of ``point`` from the crease along the crease normal.

    Same sign == same side of the crease. Used to decide whether a candidate
    grasp is on the moving (correct) side for the fold direction.
    """
    p = as_vector(point)[:2]
    return float(np.dot(p - as_vector(fold_line.point)[:2], _crease_normal(fold_line)))


def grasp_place_from_keypoints(
    kp: ClothKeypoints,
    direction: str = "right_to_left",
    grasp_strategy: Optional[str] = None,
    grasp_override: Optional[np.ndarray] = None,
) -> GraspPlacePair:
    """Full geometric grasp/place pair for a fold — DIRECTION is authoritative.

    The grasp must lie on the *moving* edge implied by ``direction`` (e.g. the
    right edge for ``right_to_left``). A config ``grasp_strategy`` or a detector
    ``grasp_override`` (e.g. a Claude/learned grasp keypoint) is honored ONLY if
    it lands on that moving side; if it conflicts with the task direction it is
    ignored and the direction-derived edge midpoint is used instead. The place
    point is always the reflection of the chosen grasp across the crease.
    """
    fold_line = compute_fold_line(kp, direction)

    # 1. Canonical grasp: the moving-edge midpoint chosen by the fold direction.
    moving_strategy = DIRECTION_GRASP_STRATEGY.get(direction, grasp_strategy or "right_edge_midpoint")
    grasp = compute_grasp_point(kp, moving_strategy)
    moving_sign = _side_sign(grasp, fold_line)
    rejected: list[str] = []

    # 2. A config strategy may refine WHERE on the moving edge to grasp, but only
    #    if it stays on the correct side.
    if grasp_strategy and grasp_strategy != moving_strategy:
        try:
            cand = compute_grasp_point(kp, grasp_strategy)
            if _side_sign(cand, fold_line) * moving_sign > 0:
                grasp = cand
            else:
                rejected.append(f"strategy {grasp_strategy!r} (wrong side for {direction})")
        except ValueError:
            pass

    # 3. A detector grasp (markers / learned / Claude) is honored only when it
    #    agrees with the task direction — never trusted blindly.
    if grasp_override is not None:
        ov = as_vector(grasp_override)[:2]
        if _side_sign(ov, fold_line) * moving_sign > 0:
            grasp = ov
        else:
            rejected.append(f"detector grasp {ov.tolist()} (wrong side for {direction})")

    place = compute_place_point(grasp, fold_line)
    meta = {"direction": direction, "moving_strategy": moving_strategy}
    if rejected:
        meta["rejected_overrides"] = rejected
    return GraspPlacePair(
        grasp=grasp, place=place, confidence=float(kp.rectangularity()), metadata=meta
    )
