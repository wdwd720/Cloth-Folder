"""Tests for the towel dataset foundation: schema, dataset, overlay, merge, YOLO export.

Uses PNG images (written via the pure-Python codec) so no OpenCV/Pillow is needed.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from terafold.towel import dataset as ds
from terafold.towel import merge as mg
from terafold.towel import overlay as ov
from terafold.towel import schema as sc
from terafold.towel import yolo_export as yx
from terafold.vision.imageio import imwrite


def _png(path, w=80, h=60, val=200):
    img = np.full((h, w, 3), 30, np.uint8)
    img[10:h - 10, 10:w - 10] = val  # a bright rectangle (distinct per `val`)
    imwrite(path, img)
    return path


def _make_raw_folder(tmp_path, n=3, dupe=False):
    d = tmp_path / "raw"
    d.mkdir()
    for i in range(n):
        _png(str(d / f"img_{i}.png"), val=100 + i * 20)
    if dupe:
        _png(str(d / "dup.png"), val=100)  # same content as img_0
    return str(d)


def _label_sample(root, sample, state="flat_unfolded", corners=None, visible=None):
    lab = ds.load_label(root, sample)
    lab["state"] = state
    if corners is not None:
        lab["corners"] = {k: corners[i] for i, k in enumerate(sc.CORNER_ORDER)}
    if visible is not None:
        lab["visible"] = visible
    lab["usable_for_training"] = True
    ds.save_label(root, sample, lab)
    return lab


# --------------------------- schema ---------------------------


def test_label_template_matches_spec():
    lab = sc.new_label("images/x.jpg")
    assert lab["state"] == "flat_unfolded"
    assert lab["bbox"] is None
    assert set(lab["corners"]) == {"tl", "tr", "br", "bl"}
    assert all(lab["corners"][k] is None for k in sc.CORNER_ORDER)
    assert lab["visible"] == {"tl": True, "tr": True, "br": True, "bl": True}
    assert lab["source"] == "user_photo"
    assert lab["usable_for_training"] is False
    assert sc.validate_label(lab) == []


def test_validate_catches_bad_state_and_corners():
    lab = sc.new_label("images/x.jpg")
    lab["state"] = "banana"
    lab["corners"]["tl"] = [1, 2, 3]
    problems = sc.validate_label(lab)
    assert any("state" in p for p in problems)
    assert any("corner tl" in p for p in problems)


def test_image_size_and_hash_png(tmp_path):
    p = _png(str(tmp_path / "a.png"), w=123, h=45, val=100)
    assert sc.image_size(p) == (123, 45)
    h1 = sc.image_hash(p)
    p2 = _png(str(tmp_path / "b.png"), w=123, h=45, val=200)
    assert sc.image_hash(p2) != h1   # different content → different hash


def test_is_usable_requires_trainable_state_and_corners():
    lab = sc.new_label("images/x.jpg")
    lab["usable_for_training"] = True
    assert sc.is_usable(lab) is False  # no corners yet
    lab["corners"] = {"tl": [0, 0], "tr": [10, 0], "br": [10, 10], "bl": [0, 10]}
    assert sc.is_usable(lab) is True
    lab["state"] = "bad_view"
    assert sc.is_usable(lab) is False  # reject state excluded


# --------------------------- dataset ---------------------------


def test_create_dataset_structure_and_dedup(tmp_path):
    raw = _make_raw_folder(tmp_path, n=3, dupe=True)
    out = str(tmp_path / "towel_real_v0")
    res = ds.create_towel_real_dataset(raw, out)
    assert res["num_images"] == 3          # the duplicate was skipped
    assert res["skipped_duplicates"] == 1
    p = ds.dataset_paths(out)
    assert os.path.isdir(p["images"]) and os.path.isdir(p["labels"])
    assert os.path.exists(p["manifest"]) and os.path.exists(p["readme"])
    man = ds.load_manifest(out)
    assert man["kind"] == "real" and len(man["samples"]) == 3
    s0 = man["samples"][0]
    assert s0["width"] == 80 and s0["height"] == 60 and s0["source"] == "user_photo"
    lab = ds.load_label(out, s0)
    assert sc.validate_label(lab) == [] and lab["usable_for_training"] is False


def test_create_dataset_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ds.create_towel_real_dataset(str(tmp_path / "nope"), str(tmp_path / "o"))


# --------------------------- overlay ---------------------------


def test_overlay_renders_png(tmp_path):
    raw = _make_raw_folder(tmp_path, n=1)
    out = str(tmp_path / "d")
    ds.create_towel_real_dataset(raw, out)
    man = ds.load_manifest(out)
    s = man["samples"][0]
    lab = _label_sample(out, s, corners=[[12, 12], [68, 12], [68, 48], [12, 48]])
    res = ov.draw_towel_overlay(os.path.join(out, s["image"]), lab,
                                str(tmp_path / "ov.png"))
    assert res["rendered"] is True and os.path.exists(res["out"])


# --------------------------- merge ---------------------------


def _labeled_dataset(tmp_path, name, n=4, source="user_photo", state="flat_unfolded"):
    raw = tmp_path / f"raw_{name}"
    raw.mkdir()
    for i in range(n):
        _png(str(raw / f"{name}_{i}.png"), val=40 + i * 30 + (hash(name) % 50))
    out = str(tmp_path / name)
    ds.create_towel_real_dataset(str(raw), out, source=source)
    man = ds.load_manifest(out)
    for s in man["samples"]:
        _label_sample(out, s, state=state,
                      corners=[[12, 12], [68, 12], [68, 48], [12, 48]])
    return out


def test_merge_dedup_split_and_real_only(tmp_path):
    real = _labeled_dataset(tmp_path, "real", n=6, source="user_photo")
    synth = _labeled_dataset(tmp_path, "synth", n=4, source="synthetic")
    out = str(tmp_path / "combined")
    res = mg.merge_towel_datasets([real, synth], out)
    assert res["num_images"] == 10
    assert sum(res["by_split"].values()) == 10
    assert res["by_source"]["real_user"] == 6 and res["by_source"]["synthetic"] == 4

    # real-only drops synthetic.
    out2 = str(tmp_path / "combined_real")
    res2 = mg.merge_towel_datasets([real, synth], out2, real_only=True)
    assert res2["num_images"] == 6 and res2["dropped_synthetic"] == 4
    assert res2["by_source"]["synthetic"] == 0


def test_merge_dedup_by_hash_across_inputs(tmp_path):
    a = _labeled_dataset(tmp_path, "aa", n=3)
    # Second dataset reuses the SAME raw images → identical hashes.
    res = mg.merge_towel_datasets([a, a], str(tmp_path / "dd"))
    assert res["num_images"] == 3 and res["dropped_duplicates"] == 3


# --------------------------- YOLO export ---------------------------


def test_yolo_export_format_and_reject(tmp_path):
    real = _labeled_dataset(tmp_path, "rr", n=5, source="user_photo")
    man = ds.load_manifest(real)
    # Mark one sample bad_view (should be rejected from export).
    _label_sample(real, man["samples"][0], state="bad_view")
    combined = str(tmp_path / "comb")
    mg.merge_towel_datasets([real], combined)
    yolo = str(tmp_path / "yolo")
    res = yx.export_yolo_towel_pose(combined, yolo)
    # 5 labeled, 1 bad_view → only 4 usable reached the merge → 4 exported.
    assert res["exported"] == 4
    assert os.path.exists(res["yaml"])
    with open(res["yaml"]) as f:
        y = f.read()
    assert "kpt_shape: [4, 3]" in y and "0: towel" in y and "flip_idx" in y
    # A label line: class + 4 bbox + 4*3 kpts = 17 tokens.
    split_dir = "train" if res["by_split"]["train"] else ("val" if res["by_split"]["val"] else "test")
    lab_dir = os.path.join(yolo, "labels", split_dir)
    any_txt = [f for f in os.listdir(lab_dir) if f.endswith(".txt")][0]
    tokens = open(os.path.join(lab_dir, any_txt)).read().split()
    assert len(tokens) == 17 and tokens[0] == "0"
    # by-source counts present.
    assert res["by_source"]["real_user"] == 4
