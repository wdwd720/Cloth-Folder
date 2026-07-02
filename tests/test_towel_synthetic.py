"""Tests for OPTIONAL synthetic towel augmentation (terafold.towel.synthetic).

Runs with numpy + pyyaml only: images are written via the pure-Python PNG codec
and labels go through the towel schema foundation. Synthetic data is augmentation
only — real images are the default — but every generated label must still be a
valid, training-usable towel label with exact in-bounds corners.
"""

from __future__ import annotations

import os

from terafold.towel import dataset as ds
from terafold.towel import schema as sc
from terafold.towel.synthetic import generate_towel_dataset


def _silent(_m: str) -> None:
    pass


def test_generate_synthetic_dataset(tmp_path):
    out = str(tmp_path / "towel_synth_v0")
    size = 128
    res = generate_towel_dataset(5, out, image_size=size, seed=0, log=_silent)

    assert res["out"] == out
    assert res["num_images"] == 5
    assert res["kind"] == "synthetic"

    # Manifest: synthetic kind, 5 samples.
    man = ds.load_manifest(out)
    assert man["kind"] == "synthetic"
    assert len(man["samples"]) == 5
    assert man["num_labeled_usable"] == 5

    for sample in man["samples"]:
        # Image + label files exist on disk.
        assert os.path.exists(os.path.join(out, sample["image"]))
        assert os.path.exists(os.path.join(out, sample["label"]))
        assert sample["source"] == "synthetic"

        lab = ds.load_label(out, sample)
        assert sc.validate_label(lab) == []
        assert lab["source"] == "synthetic"
        # Every label is usable for training.
        assert sc.is_usable(lab) is True

        # Corners present and strictly inside the image bounds.
        corners = sc.label_corners_xy(lab)
        assert corners is not None and len(corners) == 4
        for x, y in corners:
            assert 0 <= x <= size - 1
            assert 0 <= y <= size - 1


def test_synthetic_is_deterministic_across_dirs(tmp_path):
    out_a = str(tmp_path / "a")
    out_b = str(tmp_path / "b")
    generate_towel_dataset(3, out_a, image_size=96, seed=7, log=_silent)
    generate_towel_dataset(3, out_b, image_size=96, seed=7, log=_silent)

    man_a = ds.load_manifest(out_a)
    man_b = ds.load_manifest(out_b)

    # Same seed -> identical first image bytes (hash) across different out dirs.
    h_a = sc.image_hash(os.path.join(out_a, man_a["samples"][0]["image"]))
    h_b = sc.image_hash(os.path.join(out_b, man_b["samples"][0]["image"]))
    assert h_a == h_b
    # And the manifest's recorded hash matches the file on disk.
    assert man_a["samples"][0]["hash"] == h_a
