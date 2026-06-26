"""Fold quality metrics — the geometric backbone of success scoring.

These metrics work purely from cloth state (keypoints / polygons / masks) and
form the interpretable, learning-free baseline that the learned success model
(Model 2) is trained to agree with and eventually surpass.

Metrics produced (matching the spec):

* ``overlap_ratio``        — IoU of the idealized folded shape vs the observed
                             after-shape (1.0 = perfect).
* ``corner_error_m``       — best-matched corner distance between predicted and
                             observed after-corners.
* ``edge_alignment_error`` — how far the moving edge lands from the target edge
                             under reflection (0 = perfect crease alignment).
* ``visible_area_ratio``   — after-footprint / before-footprint (~0.5 for a
                             half fold).
* ``wrinkle_score``        — texture roughness inside the cloth (0 = flat).

Convex polygon intersection uses Sutherland-Hodgman clipping (both cloth halves
are convex quads), so no external geometry library is needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations
from typing import Dict, Optional

import numpy as np

from terafold.math.geometry import polygon_area
from terafold.physics.cloth_state import ClothKeypoints, FoldLine, FoldState
from terafold.physics.fold_geometry import (
    compute_fold_line,
    predict_folded_corners,
)

__all__ = [
    "FoldQualityMetrics",
    "sutherland_hodgman_clip",
    "polygon_iou",
    "corner_match_error",
    "edge_alignment_error",
    "visible_area_ratio",
    "wrinkle_score_from_mask",
    "wrinkle_score_from_image",
    "compute_fold_quality",
    "stationary_half_polygon",
    "moving_half_polygon",
]

_MOVING_SIDE = {
    "right_to_left": "right",
    "left_to_right": "left",
    "top_to_bottom": "top",
    "bottom_to_top": "bottom",
}


@dataclass
class FoldQualityMetrics:
    overlap_ratio: float = 0.0
    corner_error_m: float = float("inf")
    edge_alignment_error: float = float("inf")
    visible_area_ratio: float = 0.0
    wrinkle_score: float = 1.0
    success: Optional[bool] = None
    extra: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "overlap_ratio": self.overlap_ratio,
            "corner_error_m": self.corner_error_m,
            "edge_alignment_error": self.edge_alignment_error,
            "visible_area_ratio": self.visible_area_ratio,
            "wrinkle_score": self.wrinkle_score,
            "success": self.success,
            "extra": self.extra,
        }


# --------------------------------------------------------------------------
# Polygon helpers
# --------------------------------------------------------------------------


def sutherland_hodgman_clip(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    """Clip ``subject`` polygon by convex ``clip`` polygon. Returns vertices (M, 2)."""
    subject = [np.asarray(p, dtype=np.float64) for p in subject]
    clip = [np.asarray(p, dtype=np.float64) for p in clip]
    # Ensure clip polygon is counter-clockwise for a consistent "inside" test.
    if _signed_area(clip) < 0:
        clip = clip[::-1]

    output = subject
    for i in range(len(clip)):
        a = clip[i]
        b = clip[(i + 1) % len(clip)]
        edge = b - a
        inputs = output
        output = []
        if not inputs:
            break
        for j in range(len(inputs)):
            cur = inputs[j]
            prev = inputs[j - 1]
            cur_in = np.cross(edge, cur - a) >= 0
            prev_in = np.cross(edge, prev - a) >= 0
            if cur_in:
                if not prev_in:
                    output.append(_seg_intersect(prev, cur, a, b))
                output.append(cur)
            elif prev_in:
                output.append(_seg_intersect(prev, cur, a, b))
    return np.array(output) if output else np.zeros((0, 2))


def _signed_area(poly) -> float:
    pts = np.asarray(poly, dtype=np.float64)
    if len(pts) < 3:
        return 0.0
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _seg_intersect(p1, p2, a, b) -> np.ndarray:
    """Intersection of segment p1->p2 with the infinite line a->b."""
    d1 = p2 - p1
    d2 = b - a
    denom = np.cross(d1, d2)
    if abs(denom) < 1e-12:
        return p2.copy()
    t = np.cross(a - p1, d2) / denom
    return p1 + t * d1


def polygon_iou(poly_a: np.ndarray, poly_b: np.ndarray) -> float:
    """Intersection-over-union of two convex polygons."""
    a = np.asarray(poly_a, dtype=np.float64)
    b = np.asarray(poly_b, dtype=np.float64)
    if a.shape[0] < 3 or b.shape[0] < 3:
        return 0.0
    inter = sutherland_hodgman_clip(a, b)
    if inter.shape[0] < 3:
        return 0.0
    area_i = polygon_area(inter)
    area_a = polygon_area(a)
    area_b = polygon_area(b)
    union = area_a + area_b - area_i
    return float(area_i / union) if union > 1e-12 else 0.0


def stationary_half_polygon(kp: ClothKeypoints, direction: str = "right_to_left") -> np.ndarray:
    """The half of the cloth that does NOT move during the fold (as a quad)."""
    if direction in ("right_to_left",):  # left half stays
        return np.stack([kp.top_left, kp.mid_top, kp.mid_bottom, kp.bottom_left])
    if direction in ("left_to_right",):  # right half stays
        return np.stack([kp.mid_top, kp.top_right, kp.bottom_right, kp.mid_bottom])
    if direction in ("top_to_bottom",):  # bottom half stays
        return np.stack([kp.mid_left, kp.mid_right, kp.bottom_right, kp.bottom_left])
    if direction in ("bottom_to_top",):  # top half stays
        return np.stack([kp.top_left, kp.top_right, kp.mid_right, kp.mid_left])
    raise ValueError(f"unknown direction {direction!r}")


def moving_half_polygon(kp: ClothKeypoints, direction: str = "right_to_left") -> np.ndarray:
    """The half of the cloth that moves during the fold (as a quad)."""
    if direction in ("right_to_left",):  # right half moves
        return np.stack([kp.mid_top, kp.top_right, kp.bottom_right, kp.mid_bottom])
    if direction in ("left_to_right",):
        return np.stack([kp.top_left, kp.mid_top, kp.mid_bottom, kp.bottom_left])
    if direction in ("top_to_bottom",):
        return np.stack([kp.top_left, kp.top_right, kp.mid_right, kp.mid_left])
    if direction in ("bottom_to_top",):
        return np.stack([kp.mid_left, kp.mid_right, kp.bottom_right, kp.bottom_left])
    raise ValueError(f"unknown direction {direction!r}")


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def corner_match_error(pred_kp: ClothKeypoints, obs_kp: ClothKeypoints) -> float:
    """Mean corner distance under the best of the 24 corner-label assignments.

    Folded-shape corner labels can be permuted relative to the prediction, so we
    minimize over permutations rather than assuming a fixed correspondence.
    """
    pred = pred_kp.corners
    obs = obs_kp.corners
    best = float("inf")
    for perm in permutations(range(4)):
        d = float(np.linalg.norm(pred - obs[list(perm)], axis=1).mean())
        best = min(best, d)
    return best


def edge_alignment_error(kp: ClothKeypoints, fold_line: FoldLine, direction: str) -> float:
    """Distance the moving edge midpoint lands from the target edge midpoint.

    For a perfect rectangle folded along its center crease this is 0.
    """
    from terafold.math.geometry import reflect_point_across_line

    side = _MOVING_SIDE.get(direction, "right")
    if side == "right":
        moving_mid, target_mid = kp.mid_right, kp.mid_left
    elif side == "left":
        moving_mid, target_mid = kp.mid_left, kp.mid_right
    elif side == "top":
        moving_mid, target_mid = kp.mid_top, kp.mid_bottom
    else:
        moving_mid, target_mid = kp.mid_bottom, kp.mid_top
    reflected = reflect_point_across_line(moving_mid, fold_line.point, fold_line.direction)
    return float(np.linalg.norm(reflected - target_mid))


def visible_area_ratio(
    after_area: float, before_area: float
) -> float:
    """``after / before`` footprint ratio (clamped to a sane range)."""
    if before_area <= 1e-9:
        return 0.0
    return float(np.clip(after_area / before_area, 0.0, 2.0))


def wrinkle_score_from_mask(mask: np.ndarray) -> float:
    """Boundary-roughness wrinkle proxy in [0, 1] from a binary mask.

    Compares the mask perimeter to that of an equal-area circle (isoperimetric
    ratio). A smooth folded cloth has a low score; a bunched one has a high one.
    """
    m = (np.asarray(mask) > 0).astype(np.float64)
    area = float(m.sum())
    if area < 1.0:
        return 1.0
    # Perimeter via gradient (count boundary pixels).
    gx = np.abs(np.diff(m, axis=1, prepend=m[:, :1]))
    gy = np.abs(np.diff(m, axis=0, prepend=m[:1, :]))
    perimeter = float((gx + gy > 0).sum())
    circle_perim = 2.0 * np.sqrt(np.pi * area)
    ratio = perimeter / max(circle_perim, 1e-6)
    # ratio ~1 for a disk, larger for ragged boundaries. Map to [0, 1].
    return float(np.clip((ratio - 1.0) / 2.0, 0.0, 1.0))


def wrinkle_score_from_image(image: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Gradient-energy wrinkle proxy in [0, 1] from intensity texture."""
    img = np.asarray(image, dtype=np.float64)
    if img.ndim == 3:
        img = img.mean(axis=2)
    gx = np.diff(img, axis=1, prepend=img[:, :1])
    gy = np.diff(img, axis=0, prepend=img[:1, :])
    energy = np.sqrt(gx**2 + gy**2)
    if mask is not None:
        m = np.asarray(mask) > 0
        if m.sum() < 1:
            return 0.0
        val = float(energy[m].mean())
    else:
        val = float(energy.mean())
    # Normalize: typical flat-cloth gradient ~ a few intensity units.
    return float(np.clip(val / 40.0, 0.0, 1.0))


def compute_fold_quality(
    before: FoldState,
    after: FoldState,
    success_cfg=None,
    direction: str = "right_to_left",
) -> FoldQualityMetrics:
    """Compute all fold-quality metrics and (if a success config is given) success.

    Works in whatever frame the FoldStates are expressed in; pass table-frame
    states for metric (meter) thresholds to be meaningful.
    """
    before_kp = before.keypoints
    after_kp = after.keypoints

    fold_line = before.fold_line or compute_fold_line(before_kp, direction)
    predicted_after = predict_folded_corners(before_kp, direction)

    # Overlap: idealized folded shape vs observed after shape (IoU).
    overlap = polygon_iou(predicted_after.corners, after_kp.corners)
    corner_err = corner_match_error(predicted_after, after_kp)
    edge_err = edge_alignment_error(before_kp, fold_line, direction)

    # Visible area: prefer masks, fall back to polygon areas.
    before_area = before.mask.area() if before.mask else polygon_area(before_kp.corners)
    after_area = after.mask.area() if after.mask else polygon_area(after_kp.corners)
    var = visible_area_ratio(after_area, before_area)

    # Wrinkle: prefer mask boundary roughness, fall back to stored proxy.
    if after.mask is not None and after.mask.mask is not None:
        wrinkle = wrinkle_score_from_mask(after.mask.mask)
    elif after.mask is not None and after.mask.wrinkle_proxy is not None:
        wrinkle = float(np.clip(after.mask.wrinkle_proxy, 0.0, 1.0))
    else:
        wrinkle = 0.0

    metrics = FoldQualityMetrics(
        overlap_ratio=overlap,
        corner_error_m=corner_err,
        edge_alignment_error=edge_err,
        visible_area_ratio=var,
        wrinkle_score=wrinkle,
    )

    if success_cfg is not None:
        metrics.success = bool(
            corner_err <= success_cfg.max_corner_error_m
            and overlap >= success_cfg.min_overlap_ratio
            and wrinkle <= success_cfg.max_wrinkle_score
            and var >= success_cfg.min_visible_area_ratio
        )
    return metrics
