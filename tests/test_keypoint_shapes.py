"""Keypoint model + heatmap encoding tensor shapes."""

from __future__ import annotations

import numpy as np
import pytest

from terafold.physics.cloth_state import DETECTOR_KEYPOINTS, ClothKeypoints
from terafold.vision.keypoint_dataset import (
    HEATMAP_SIZE,
    IMAGE_SIZE,
    heatmaps_to_keypoints,
    keypoints_to_heatmaps,
)


def test_heatmap_encoding_shape():
    kp = ClothKeypoints(
        [40, 50], [200, 45], [205, 200], [45, 205],
        grasp=[205, 125], place=[45, 125], fold_a=[125, 45], fold_b=[125, 205],
    )
    hm = keypoints_to_heatmaps(kp, image_size=IMAGE_SIZE, out_size=HEATMAP_SIZE, sigma=2.0)
    assert hm.shape == (len(DETECTOR_KEYPOINTS), HEATMAP_SIZE, HEATMAP_SIZE)
    assert hm.dtype == np.float32 or hm.dtype == np.float64
    assert hm.max() > 0.5


def test_heatmap_decoding_recovers_points():
    kp = ClothKeypoints([40, 50], [200, 45], [205, 200], [45, 205])
    hm = keypoints_to_heatmaps(kp, image_size=IMAGE_SIZE, out_size=HEATMAP_SIZE, sigma=2.0)
    decoded = heatmaps_to_keypoints(hm, image_size=IMAGE_SIZE)
    # Recovered corners should be within a few pixels (heatmap quantization).
    err = np.linalg.norm(decoded.corners - kp.corners, axis=1)
    assert err.max() < IMAGE_SIZE / HEATMAP_SIZE * 3


def test_keypoint_model_forward_shape():
    torch = pytest.importorskip("torch")
    from terafold.vision.keypoint_model import build_keypoint_model

    model = build_keypoint_model(num_keypoints=len(DETECTOR_KEYPOINTS))
    model.eval()
    x = torch.zeros(2, 3, IMAGE_SIZE, IMAGE_SIZE)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, len(DETECTOR_KEYPOINTS), HEATMAP_SIZE, HEATMAP_SIZE)
