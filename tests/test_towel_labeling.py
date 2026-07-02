"""Tests for terafold.towel.labeling — local 4-corner labeling helpers.

Uses PNG images (written via the pure-Python codec) so no OpenCV/Pillow is needed,
and drives the labeling functions through ``input_fn`` so nothing opens a GUI.
"""

from __future__ import annotations

import os

import numpy as np

from terafold.towel import dataset as ds
from terafold.towel import labeling as lb
from terafold.towel import schema as sc
from terafold.vision.imageio import imwrite


def _png(path, w=80, h=60, val=200):
    img = np.full((h, w, 3), 30, np.uint8)
    img[10:h - 10, 10:w - 10] = val  # a bright rectangle (distinct per `val`)
    imwrite(path, img)
    return path


def _make_dataset(tmp_path, n=3):
    raw = tmp_path / "raw"
    raw.mkdir()
    for i in range(n):
        _png(str(raw / f"img_{i}.png"), val=80 + i * 30)
    out = str(tmp_path / "towel_real_v0")
    ds.create_towel_real_dataset(str(raw), out)
    return out


def _feeder(responses):
    """Return an input_fn that yields each response in order, ignoring the prompt."""
    it = iter(responses)

    def fn(prompt=""):
        return next(it)

    return fn


# --------------------------- label_towel_image ---------------------------


def test_label_image_with_corners_sets_usable_and_writes_overlay(tmp_path):
    out = _make_dataset(tmp_path, n=1)
    man = ds.load_manifest(out)
    s = man["samples"][0]
    img_path = os.path.join(out, s["image"])
    lab_path = os.path.join(out, s["label"])

    res = lb.label_towel_image(
        img_path, lab_path, corners=[[10, 10], [60, 10], [60, 40], [10, 40]],
    )

    assert res["usable"] is True
    assert res["state"] == "flat_unfolded"
    assert res["corners"] == [[10, 10], [60, 10], [60, 40], [10, 40]]
    # Overlay rendered (PNG backend always available) and on disk.
    assert res["overlay"] and os.path.exists(res["overlay"])

    lab = ds.load_label(out, s)
    assert lab["usable_for_training"] is True
    assert sc.validate_label(lab) == []
    assert sc.is_usable(lab) is True
    assert lab["corners"] == {"tl": [10, 10], "tr": [60, 10], "br": [60, 40], "bl": [10, 40]}
    assert lab["bbox"] == [10, 10, 60, 40]


def test_label_image_terminal_corner_parsing(tmp_path):
    out = _make_dataset(tmp_path, n=1)
    man = ds.load_manifest(out)
    s = man["samples"][0]
    img_path = os.path.join(out, s["image"])
    lab_path = os.path.join(out, s["label"])

    # use_gui=False forces the terminal path: one 'x,y' string per corner.
    fn = _feeder(["10,10", "60,10", "60,40", "10,40"])
    res = lb.label_towel_image(img_path, lab_path, input_fn=fn, use_gui=False)

    assert res["usable"] is True
    assert res["corners"] == [[10, 10], [60, 10], [60, 40], [10, 40]]
    lab = ds.load_label(out, s)
    assert sc.is_usable(lab) is True
    assert lab["corners"]["br"] == [60, 40]


def test_label_image_keeps_state_and_sets_fold_axis(tmp_path):
    out = _make_dataset(tmp_path, n=1)
    man = ds.load_manifest(out)
    s = man["samples"][0]
    img_path = os.path.join(out, s["image"])
    lab_path = os.path.join(out, s["label"])

    res = lb.label_towel_image(
        img_path, lab_path, state="partially_folded",
        corners=[[10, 10], [60, 10], [60, 40], [10, 40]], fold_axis="vertical",
    )
    assert res["state"] == "partially_folded"
    lab = ds.load_label(out, s)
    assert lab["state"] == "partially_folded"
    assert lab["fold_axis"] == "vertical"
    assert sc.validate_label(lab) == []


# --------------------------- label_towel_folder ---------------------------


def test_label_folder_skip_bad_and_coords(tmp_path):
    out = _make_dataset(tmp_path, n=3)
    man = ds.load_manifest(out)
    s_skip, s_bad, s_coords = man["samples"]

    # One response per pending sample, in manifest order: skip, bad_view, coords.
    fn = _feeder(["skip", "bad_view", "10,10 60,10 60,40 10,40"])
    res = lb.label_towel_folder(out, input_fn=fn)

    assert res == {"total": 3, "labeled": 1, "skipped": 1, "marked_bad": 1}

    # Skipped sample stays unlabeled.
    lab_skip = ds.load_label(out, s_skip)
    assert lab_skip["usable_for_training"] is False
    assert sc.is_usable(lab_skip) is False

    # bad_view sample: state set, non-usable, excluded from training.
    lab_bad = ds.load_label(out, s_bad)
    assert lab_bad["state"] == "bad_view"
    assert lab_bad["usable_for_training"] is False
    assert sc.is_usable(lab_bad) is False

    # coords sample becomes usable.
    lab_ok = ds.load_label(out, s_coords)
    assert lab_ok["usable_for_training"] is True
    assert sc.is_usable(lab_ok) is True
    assert lab_ok["corners"] == {"tl": [10, 10], "tr": [60, 10], "br": [60, 40], "bl": [10, 40]}

    # Manifest usable count was recomputed (only the coords sample).
    assert ds.load_manifest(out)["num_labeled_usable"] == 1


def test_label_folder_only_processes_pending(tmp_path):
    out = _make_dataset(tmp_path, n=2)
    man = ds.load_manifest(out)
    # Pre-label the first sample as usable so the folder loop skips it.
    first = man["samples"][0]
    lb.label_towel_image(
        os.path.join(out, first["image"]), os.path.join(out, first["label"]),
        corners=[[10, 10], [60, 10], [60, 40], [10, 40]],
    )

    fn = _feeder(["10,10 60,10 60,40 10,40"])  # only one pending sample remains
    res = lb.label_towel_folder(out, input_fn=fn)
    assert res["total"] == 1 and res["labeled"] == 1
    assert ds.load_manifest(out)["num_labeled_usable"] == 2
