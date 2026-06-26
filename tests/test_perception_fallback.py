"""Regression tests: perception/camera robustness contracts.

These pin three bugs found by adversarial review:
* the classical fallback predictor must never crash (even on blank/black input),
* the mock camera's ground truth must stay aligned with image flips.
"""

from __future__ import annotations

import numpy as np

from terafold.camera.mock_camera import MockCamera
from terafold.config.schema import CameraConfig
from terafold.vision.cloth_mask import segment_cloth
from terafold.vision.infer_keypoints import get_keypoint_predictor


def test_fallback_predictor_never_crashes():
    predictor = get_keypoint_predictor(None)
    cases = {
        "blank": np.full((128, 128, 3), 120, np.uint8),
        "black": np.zeros((96, 96, 3), np.uint8),
        "white": np.full((100, 140, 3), 255, np.uint8),
        "noise": np.random.default_rng(0).integers(0, 255, (110, 90, 3)).astype(np.uint8),
    }
    for name, img in cases.items():
        fs = predictor.predict(img)
        assert np.all(np.isfinite(fs.keypoints.corners)), name
        assert fs.fold_line is not None, name


def test_fallback_does_not_import_torch():
    import sys

    sys.modules.pop("torch", None)
    predictor = get_keypoint_predictor(None)
    predictor.predict(np.full((64, 64, 3), 90, np.uint8))
    assert "torch" not in sys.modules  # numpy-only path


def _gt_alignment(flip_h: bool) -> float:
    cam = MockCamera(CameraConfig(type="mock", flip_horizontal=flip_h), seed=11, image_size=128)
    cam.connect()
    frame = cam.read()
    gt = cam.last_ground_truth.mask.mask > 0
    seg = segment_cloth(frame.image) > 0
    inter = float((gt & seg).sum())
    union = float((gt | seg).sum())
    return inter / max(union, 1.0)


def test_mock_camera_ground_truth_tracks_flip():
    assert _gt_alignment(False) > 0.8
    assert _gt_alignment(True) > 0.8  # GT must follow the flipped image
