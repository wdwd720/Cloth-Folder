"""Lightweight quasi-static cloth heuristics.

This is NOT a cloth simulator. It is a small set of interpretable functions
that encode the physical intuitions in the TeraFold spec:

* Cloth resists stretching far more than bending.
* Folding needs *enough* lift to reduce table friction/dragging...
* ...but *too much* lift causes uncontrolled swinging / slip.
* Grasping near corners increases rotational instability; an edge midpoint is
  more stable for a half-fold.
* The release point should be nudged opposite the expected slip direction.
* Fold quality depends on final keypoint alignment + visible-mask overlap.

Each function returns a scalar (or small vector) that is both usable as a
planning prior *and* as an input feature for the residual learner — the whole
point is that the learner can correct these later.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from terafold.physics.cloth_state import ClothKeypoints
from terafold.physics.friction import estimate_stiffness, estimate_table_friction

__all__ = [
    "estimate_required_lift_height",
    "estimate_arc_height",
    "estimate_slip_compensation",
    "estimate_grasp_stability",
    "estimate_fold_energy_proxy",
    "physics_feature_vector",
    "PHYSICS_FEATURE_NAMES",
]


def estimate_required_lift_height(
    cloth_size: Tuple[float, float],
    stiffness_hint: str = "medium",
    friction_hint: str = "medium",
    base_lift_m: float = 0.02,
) -> float:
    """Estimate the lift height (m) needed to fold without dragging.

    Intuition encoded:
    * Larger cloth (longer moving flap) needs more lift to clear the table.
    * Higher friction needs more lift (more dragging otherwise).
    * Higher bending stiffness needs *slightly* more lift (a stiff flap stands
      up and would otherwise scrape), but stiffness matters less than size.

    The fold "flap" length for a half fold is ~ half the cloth dimension along
    the fold direction; we use the larger dimension as a conservative proxy.
    """
    w, h = float(cloth_size[0]), float(cloth_size[1])
    flap = 0.5 * max(w, h)
    mu = estimate_table_friction(friction_hint)
    stiff = estimate_stiffness(stiffness_hint)
    # Lift scales with flap length and friction; stiffness adds a small bump.
    lift = base_lift_m + 0.25 * flap * (0.5 + mu) + 0.01 * stiff
    # Keep within sane bounds (1 cm .. 20 cm).
    return float(np.clip(lift, 0.01, 0.20))


def estimate_arc_height(lift_height_m: float, cloth_size: Tuple[float, float]) -> float:
    """Apex height of the fold arc — above the lift height, scaled by flap length.

    Too-high arcs cause swinging/slip, so the bonus over the lift height is
    capped relative to the flap length.
    """
    flap = 0.5 * max(float(cloth_size[0]), float(cloth_size[1]))
    arc = lift_height_m + min(0.5 * flap, 0.06)
    return float(max(arc, lift_height_m + 0.005))


def estimate_slip_compensation(
    grasp: np.ndarray,
    place: np.ndarray,
    cloth_mask=None,
    table_friction: float = 0.4,
    gain: float = 0.25,
) -> np.ndarray:
    """Vector (same units as grasp/place) to add to the *place* point.

    The cloth tends to slip *forward* (continuing along the fold travel
    direction) as it is released, so we place slightly *short* of the target —
    i.e. nudge the place point back toward the grasp side. Magnitude grows with
    travel distance and friction (more friction -> more stored tension ->
    more snap-back/slip on release).
    """
    grasp = np.asarray(grasp, dtype=np.float64).reshape(-1)
    place = np.asarray(place, dtype=np.float64).reshape(-1)
    travel = place - grasp
    dist = float(np.linalg.norm(travel))
    if dist < 1e-9:
        return np.zeros_like(place)
    direction = travel / dist
    magnitude = gain * dist * float(np.clip(table_friction, 0.0, 1.0))
    # Nudge back toward grasp (place short of target) to counter forward slip.
    return -direction * magnitude


def estimate_grasp_stability(
    keypoints: ClothKeypoints, candidate_grasp: np.ndarray
) -> float:
    """Score a candidate grasp in [0, 1] (1 = very stable).

    Heuristics:
    * Grasping at an edge *midpoint* is most stable for a half fold.
    * Grasping near a *corner* is least stable (the flap rotates).
    * Grasping near the cloth *center* is unstable for a half fold (you'd fold
      along an unintended line) — penalized.
    """
    g = np.asarray(candidate_grasp, dtype=np.float64).reshape(-1)
    corners = keypoints.corners
    center = keypoints.center
    diag = float(np.linalg.norm(corners.max(axis=0) - corners.min(axis=0))) + 1e-9

    # Distance to nearest corner (closer = less stable).
    d_corner = float(min(np.linalg.norm(corners - g, axis=1))) / diag
    # Distance to nearest edge midpoint (closer = more stable).
    mids = np.stack(
        [keypoints.mid_top, keypoints.mid_right, keypoints.mid_bottom, keypoints.mid_left]
    )
    d_mid = float(min(np.linalg.norm(mids - g, axis=1))) / diag
    # Distance to center (too close = unstable for half fold).
    d_center = float(np.linalg.norm(center - g)) / diag

    corner_term = np.clip(d_corner * 2.0, 0.0, 1.0)  # reward being away from corners
    mid_term = np.clip(1.0 - d_mid * 2.0, 0.0, 1.0)  # reward being near a midpoint
    center_term = np.clip(d_center * 2.0, 0.0, 1.0)  # reward being away from center

    score = 0.4 * corner_term + 0.4 * mid_term + 0.2 * center_term
    return float(np.clip(score, 0.0, 1.0))


def estimate_fold_energy_proxy(
    before_keypoints: ClothKeypoints, after_keypoints: ClothKeypoints
) -> float:
    """A bending-dominated "work" proxy for the fold.

    Approximates the energy as the summed squared displacement of the moving
    corners (stretching would cost far more, but a fold is mostly rigid-body
    motion of a flap, so displacement is a reasonable proxy). Normalized by the
    cloth diagonal so it is scale-independent.
    """
    b = before_keypoints.corners
    a = after_keypoints.corners
    diag = float(np.linalg.norm(b.max(axis=0) - b.min(axis=0))) + 1e-9
    disp = np.linalg.norm(a - b, axis=1) / diag
    return float((disp**2).sum())


PHYSICS_FEATURE_NAMES = [
    "cloth_width",
    "cloth_height",
    "cloth_diag",
    "rectangularity",
    "friction",
    "stiffness",
    "required_lift",
    "arc_height",
    "grasp_stability",
    "slip_comp_mag",
    "travel_distance",
]


def physics_feature_vector(
    keypoints: ClothKeypoints,
    grasp: np.ndarray,
    place: np.ndarray,
    cloth_size: Tuple[float, float],
    stiffness_hint: str = "medium",
    friction_hint: str = "medium",
    slip_comp: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Assemble the interpretable physics features for the residual learner.

    Order matches :data:`PHYSICS_FEATURE_NAMES`.
    """
    w, h = float(cloth_size[0]), float(cloth_size[1])
    diag = float(np.hypot(w, h))
    mu = estimate_table_friction(friction_hint)
    stiff = estimate_stiffness(stiffness_hint)
    lift = estimate_required_lift_height((w, h), stiffness_hint, friction_hint)
    arc = estimate_arc_height(lift, (w, h))
    stab = estimate_grasp_stability(keypoints, grasp)
    if slip_comp is None:
        slip_comp = estimate_slip_compensation(grasp, place, None, mu)
    slip_mag = float(np.linalg.norm(slip_comp))
    travel = float(np.linalg.norm(np.asarray(place) - np.asarray(grasp)))
    return np.array(
        [w, h, diag, keypoints.rectangularity(), mu, stiff, lift, arc, stab, slip_mag, travel],
        dtype=np.float64,
    )
