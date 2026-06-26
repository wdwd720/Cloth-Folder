"""Synthetic cloth generator produces in-bounds labels and loadable images."""

from __future__ import annotations

import os

import numpy as np

from terafold.data.episode_schema import read_json
from terafold.physics.cloth_state import FoldState
from terafold.vision.imageio import imread
from terafold.vision.synthetic_cloth import generate_synthetic_dataset, render_towel_scene
from terafold.vision.synthetic_cloth import SyntheticClothConfig


def test_render_labels_inside_bounds():
    rng = np.random.default_rng(0)
    cfg = SyntheticClothConfig(image_size=128)
    for _ in range(10):
        sample = render_towel_scene(rng, cfg)
        h, w = sample.image.shape[:2]
        assert sample.image.dtype == np.uint8
        for corner in sample.keypoints.corners:
            assert 0 <= corner[0] < w, corner
            assert 0 <= corner[1] < h, corner
        # mask covers some, but not all, of the image
        frac = (np.asarray(sample.mask) > 0).mean()
        assert 0.02 < frac < 0.98


def test_generate_dataset_writes_files(tmp_path):
    out = str(tmp_path / "synth")
    generate_synthetic_dataset(out, 5, image_size=96, seed=1)
    index = read_json(os.path.join(out, "index.json"))
    assert index["num"] == 5
    item = index["items"][0]
    img = imread(os.path.join(out, item["image"]))
    assert img.shape[0] == 96 and img.shape[2] == 3
    label = read_json(os.path.join(out, item["label"]))
    fs = FoldState.from_dict(label["fold_state"])
    assert fs.keypoints.corners.shape == (4, 2)


def test_dataset_labels_in_bounds(tmp_path):
    out = str(tmp_path / "synth2")
    generate_synthetic_dataset(out, 8, image_size=100, seed=2)
    index = read_json(os.path.join(out, "index.json"))
    for item in index["items"]:
        label = read_json(os.path.join(out, item["label"]))
        fs = FoldState.from_dict(label["fold_state"])
        for c in fs.keypoints.corners:
            assert 0 <= c[0] < 100 and 0 <= c[1] < 100
