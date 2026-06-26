"""Keypoint inference + perception predictors.

Provides two ``predict(image) -> FoldState`` predictors:

* :class:`KeypointPredictor` -- runs the learned U-Net (Model 1). Requires torch.
* :class:`FallbackKeypointPredictor` -- numpy-only classical perception
  (color-marker detection, else mask segmentation + min-area-rect corners).

:func:`get_keypoint_predictor` returns the learned predictor when a checkpoint
and torch are available, otherwise the fallback. It **always** returns a working
predictor whose ``.predict`` fills the four corners + a fold line.

torch is imported lazily inside the learned paths; the module imports with numpy
alone so the Stage-0 pipeline (and the geometric planner) never needs torch.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

from terafold.physics.cloth_state import (
    ClothKeypoints,
    FoldLine,
    FoldState,
    GraspPlacePair,
)
from terafold.physics.fold_geometry import grasp_place_from_keypoints
from terafold.vision.cloth_mask import mask_state_from_mask, segment_cloth
from terafold.vision.keypoint_dataset import (
    HEATMAP_SIZE,
    IMAGE_SIZE,
    _resize_rgb,
    heatmaps_to_keypoints,
)
from terafold.vision.marker_detector import keypoints_from_markers

__all__ = [
    "load_keypoint_model",
    "infer_keypoints",
    "KeypointPredictor",
    "FallbackKeypointPredictor",
    "get_keypoint_predictor",
]


# --------------------------------------------------------------------------
# Shared FoldState assembly
# --------------------------------------------------------------------------


def _complete_keypoints(
    kp: ClothKeypoints, direction: str = "right_to_left"
) -> tuple[ClothKeypoints, FoldLine, GraspPlacePair]:
    """Ensure corners + grasp/place/fold endpoints + a fold line are all filled."""
    fold_line = FoldLine.from_corners(kp, direction=direction)
    grasp_override = kp.grasp if kp.grasp is not None else None
    gp = grasp_place_from_keypoints(
        kp, direction=direction, grasp_override=grasp_override
    )
    full = ClothKeypoints(
        top_left=kp.top_left,
        top_right=kp.top_right,
        bottom_right=kp.bottom_right,
        bottom_left=kp.bottom_left,
        grasp=kp.grasp if kp.grasp is not None else gp.grasp,
        place=kp.place if kp.place is not None else gp.place,
        fold_a=kp.fold_a if kp.fold_a is not None else fold_line.a,
        fold_b=kp.fold_b if kp.fold_b is not None else fold_line.b,
        confidence=kp.confidence,
    )
    return full, fold_line, gp


def _ensure_nondegenerate(kp: ClothKeypoints, h: int, w: int) -> ClothKeypoints:
    """Replace degenerate/empty detections with a full-image quad.

    When the cloth cannot be found (blank/uniform image, failed segmentation),
    ``corners_from_mask`` may return collapsed or non-finite corners. Building a
    crease from those raises ``ValueError`` in ``fold_line_from_corners``. The
    contract is that the predictor never crashes, so we fall back to the image
    bounds (and let grasp/place be recomputed) rather than propagating garbage.
    """
    c = kp.corners
    if (not np.all(np.isfinite(c))) or float(np.ptp(c[:, 0])) < 1.0 or float(np.ptp(c[:, 1])) < 1.0:
        return ClothKeypoints(
            top_left=[0.0, 0.0],
            top_right=[w - 1.0, 0.0],
            bottom_right=[w - 1.0, h - 1.0],
            bottom_left=[0.0, h - 1.0],
        )
    return kp


def _fold_state_from_keypoints(
    kp: ClothKeypoints, image: np.ndarray, direction: str = "right_to_left"
) -> FoldState:
    """Build a complete image-frame :class:`FoldState` from raw keypoints."""
    h, w = np.asarray(image).shape[:2]
    kp = _ensure_nondegenerate(kp, int(h), int(w))
    full, fold_line, gp = _complete_keypoints(kp, direction=direction)
    try:
        mask = segment_cloth(image)
        mask_state = mask_state_from_mask(mask, image=image)
    except Exception:
        mask_state = None
    return FoldState(
        keypoints=full,
        fold_line=fold_line,
        mask=mask_state,
        grasp_place=gp,
        frame="image",
        image_shape=(int(h), int(w)),
    )


# --------------------------------------------------------------------------
# Learned model
# --------------------------------------------------------------------------


def load_keypoint_model(checkpoint: str):
    """Load a trained keypoint model from ``checkpoint`` (lazy torch).

    The returned model carries ``.image_size`` / ``.heatmap_size`` attributes
    derived from the saved config (defaults applied if absent).
    """
    try:
        import torch
    except Exception as exc:
        raise ImportError(
            "load_keypoint_model requires torch. Install it with "
            "`pip install -e '.[torch]'` (or `pip install torch`)."
        ) from exc
    from terafold.vision.keypoint_model import build_keypoint_model

    ckpt = torch.load(checkpoint, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        config = ckpt.get("config", {})
        state = ckpt["state_dict"]
    else:
        config = {}
        state = ckpt
    num_kp = int(config.get("num_keypoints", 8))
    model = build_keypoint_model(num_kp)
    model.load_state_dict(state)
    model.eval()
    model.image_size = int(config.get("image_size", IMAGE_SIZE))
    model.heatmap_size = int(config.get("heatmap_size", HEATMAP_SIZE))
    return model


def infer_keypoints(model, image: np.ndarray) -> ClothKeypoints:
    """Run the model on a single image and decode keypoints in image pixels."""
    import torch

    img = np.asarray(image)
    h, w = img.shape[:2]
    image_size = int(getattr(model, "image_size", IMAGE_SIZE))
    resized = _resize_rgb(img, (image_size, image_size))
    chw = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1) / 255.0
    tensor = torch.from_numpy(np.ascontiguousarray(chw))[None]
    with torch.no_grad():
        heatmaps = model(tensor)[0].cpu().numpy()
    # Decode in model image-size space, then rescale to original pixels.
    kp = heatmaps_to_keypoints(heatmaps, image_size=image_size)
    arr = kp.to_array()
    finite = np.all(np.isfinite(arr), axis=1)
    arr[finite, 0] *= w / float(image_size)
    arr[finite, 1] *= h / float(image_size)
    return ClothKeypoints.from_array(arr)


class KeypointPredictor:
    """Learned perception: ``predict(image) -> FoldState`` via the U-Net."""

    def __init__(self, checkpoint: str, device: Optional[str] = None) -> None:
        self.checkpoint = checkpoint
        self.device = device
        self.model = load_keypoint_model(checkpoint)
        if device is not None:
            self.model = self.model.to(device)

    def predict(
        self, image: np.ndarray, direction: str = "right_to_left"
    ) -> FoldState:
        kp = infer_keypoints(self.model, image)
        return _fold_state_from_keypoints(kp, image, direction=direction)


# --------------------------------------------------------------------------
# Classical fallback (numpy only)
# --------------------------------------------------------------------------


class FallbackKeypointPredictor:
    """Numpy-only perception: markers if present, else mask + min-area-rect."""

    def predict(
        self, image: np.ndarray, direction: str = "right_to_left"
    ) -> FoldState:
        img = np.asarray(image)
        kp: Optional[ClothKeypoints] = None
        try:
            kp = keypoints_from_markers(img)
        except Exception:
            kp = None
        if kp is None:
            from terafold.vision.cloth_mask import corners_from_mask

            mask = segment_cloth(img)
            kp = corners_from_mask(mask)
        return _fold_state_from_keypoints(kp, img, direction=direction)


# --------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------


def get_keypoint_predictor(checkpoint: Optional[str] = None):
    """Return the best available predictor; always usable.

    Learned :class:`KeypointPredictor` if ``checkpoint`` exists and torch is
    importable, otherwise the numpy :class:`FallbackKeypointPredictor`.
    """
    if checkpoint and os.path.exists(checkpoint):
        try:
            import torch  # noqa: F401

            return KeypointPredictor(checkpoint)
        except Exception:
            # Missing torch or a bad checkpoint must never break perception.
            pass
    return FallbackKeypointPredictor()
