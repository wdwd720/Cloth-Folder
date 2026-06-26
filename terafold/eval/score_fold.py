"""Score a fold from a before/after image pair.

This is the evaluation chokepoint used by the recorder, the benchmark, and any
rollout that wants a numeric verdict on a completed fold. It works with **numpy
alone**: perception falls back to the classical
:class:`~terafold.vision.infer_keypoints.FallbackKeypointPredictor` when no
learned checkpoint / torch is available, and the success verdict comes from the
geometric :func:`~terafold.physics.fold_quality.compute_fold_quality` metrics.

An optional learned success model (Model 2) may be supplied to refine the
probability; torch is only touched in that branch.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from terafold.config.schema import SuccessConfig
from terafold.physics.cloth_state import ClothKeypoints, FoldState
from terafold.physics.fold_quality import compute_fold_quality

__all__ = ["score_fold"]


def _to_table_fold_state(fs: FoldState, frames) -> FoldState:
    """Map an image-frame :class:`FoldState`'s corners into table (meter) space.

    Only the four corners are remapped (the derived midpoints / fold line follow
    from them); the mask is kept for area-ratio / wrinkle proxies, whose values
    are scale-cancelling ratios. Returns ``fs`` unchanged when ``frames`` is not
    image-calibrated.
    """
    if frames is None or not getattr(frames, "image_calibrated", False):
        return fs
    corners = fs.keypoints.corners
    mapped = np.array([frames.pixel_to_table_xy(uv) for uv in corners], dtype=np.float64)
    kp = ClothKeypoints(
        top_left=mapped[0],
        top_right=mapped[1],
        bottom_right=mapped[2],
        bottom_left=mapped[3],
        confidence=fs.keypoints.confidence,
    )
    return FoldState(keypoints=kp, mask=fs.mask, frame="table", image_shape=fs.image_shape)


def score_fold(
    before_image: np.ndarray,
    after_image: np.ndarray,
    predictor: Any = None,
    success_cfg: Optional[SuccessConfig] = None,
    success_model: Any = None,
    frames: Any = None,
    direction: str = "right_to_left",
) -> Dict[str, Any]:
    """Score a fold given the before/after images.

    Parameters
    ----------
    before_image, after_image:
        ``(H, W, 3)`` uint8 RGB frames of the cloth before and after the fold.
    predictor:
        A perception predictor exposing ``predict(image) -> FoldState``. When
        ``None`` the best available one is fetched via
        :func:`get_keypoint_predictor` (learned if a checkpoint + torch exist,
        else the numpy fallback).
    success_cfg:
        Thresholds for the boolean verdict. Defaults to :class:`SuccessConfig`'s
        conservative values. Meter thresholds are only meaningful when ``frames``
        is image-calibrated (otherwise metrics are in pixels).
    success_model:
        Optional learned success classifier (torch module or callable). When
        given, its probability is blended with the geometric overlap.
    frames:
        Optional :class:`~terafold.math.frames.FrameTransforms`. If image-
        calibrated, states are converted to table (meter) coordinates first.
    direction:
        Fold direction (default ``"right_to_left"``).

    Returns
    -------
    dict with keys ``success`` (bool), ``success_prob`` (float in [0, 1]),
    ``overlap_ratio``, ``corner_error_m``, ``edge_alignment_error``,
    ``visible_area_ratio``, ``wrinkle_score``.
    """
    if predictor is None:
        from terafold.vision.infer_keypoints import get_keypoint_predictor

        predictor = get_keypoint_predictor()
    if success_cfg is None:
        success_cfg = SuccessConfig()

    before_fs = predictor.predict(before_image)
    after_fs = predictor.predict(after_image)

    before_fs = _to_table_fold_state(before_fs, frames)
    after_fs = _to_table_fold_state(after_fs, frames)

    metrics = compute_fold_quality(
        before_fs, after_fs, success_cfg=success_cfg, direction=direction
    )

    # Geometric probability: idealized-fold IoU is already a [0, 1] confidence.
    geo_prob = float(np.clip(metrics.overlap_ratio, 0.0, 1.0))
    success_prob = geo_prob

    if success_model is not None:
        try:
            from terafold.vision.success_model import predict_success_prob

            model_prob = predict_success_prob(
                success_model, before_fs, after_fs,
                success_cfg=success_cfg, direction=direction,
            )
            success_prob = 0.5 * (geo_prob + float(model_prob))
        except Exception:
            # A missing/broken success model must never break geometric scoring.
            success_prob = geo_prob

    success = metrics.success
    if success is None:
        success = bool(success_prob >= 0.5)

    return {
        "success": bool(success),
        "success_prob": float(success_prob),
        "overlap_ratio": float(metrics.overlap_ratio),
        "corner_error_m": float(metrics.corner_error_m),
        "edge_alignment_error": float(metrics.edge_alignment_error),
        "visible_area_ratio": float(metrics.visible_area_ratio),
        "wrinkle_score": float(metrics.wrinkle_score),
    }
