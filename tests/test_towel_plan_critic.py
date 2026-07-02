"""Tests for the rule-based fold-plan critic + its trainable (numpy) scaffold.

Everything here runs with numpy + pyyaml only: images are written via the
pure-Python PNG codec and training is forced onto the numpy backend, so the
optional torch path is never required.
"""

from __future__ import annotations

import json
import os

import numpy as np

from terafold.towel import dataset as ds
from terafold.towel import plan_critic as pc
from terafold.towel import schema as sc
from terafold.vision.imageio import imwrite


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _png(path, w=80, h=60, val=200):
    img = np.full((h, w, 3), 30, np.uint8)
    img[10:h - 10, 10:w - 10] = val
    imwrite(path, img)
    return path


def _good_features():
    return {
        "corners": [[10, 10], [70, 10], [70, 50], [10, 50]],
        "aspect_ratio": 60.0 / 40.0,
        "state": "flat_unfolded",
        "keypoint_confidence": 0.95,
        "grasp_point": [40, 10],            # mid top edge — far from any corner
        "place_point": [40, 50],
        "fold_direction": "right_to_left",
        "fold_axis": [[40, 10], [40, 50]],  # non-degenerate vertical axis
        "visible": {"tl": True, "tr": True, "br": True, "bl": True},
    }


# --------------------------------------------------------------------------
# rule-based critic
# --------------------------------------------------------------------------


def test_low_confidence_flags_reason_and_low_prob():
    res = pc.critique_plan({"keypoint_confidence": 0.3})
    assert "low_confidence" in res["risk_reasons"]
    assert res["success_probability"] < 0.5
    assert res["risk_score"] == 1.0 - res["success_probability"]


def test_bad_view_is_hard_fail():
    res = pc.critique_plan({"state": "bad_view"})
    assert "bad_view" in res["risk_reasons"]
    assert res["success_probability"] < 0.05  # forced to ~0


def test_other_reject_states_are_hard_fail():
    for st in ("multiple_towels", "not_towel"):
        res = pc.critique_plan({"state": st})
        assert st in res["risk_reasons"]
        assert res["success_probability"] < 0.05


def test_good_plan_scores_high_with_no_hard_fail():
    res = pc.critique_plan(_good_features())
    assert res["success_probability"] > 0.6
    assert not any(r in pc.HARD_FAIL_REASONS for r in res["risk_reasons"])
    # a clean plan should raise no risk reasons at all
    assert res["risk_reasons"] == []


def test_already_folded_and_occlusion_and_geometry_and_grasp():
    # folded_success -> already_folded
    feats = _good_features()
    feats["state"] = "folded_success"
    assert "already_folded" in pc.critique_plan(feats)["risk_reasons"]

    # a hidden corner -> towel_too_occluded
    feats = _good_features()
    feats["visible"]["br"] = False
    assert "towel_too_occluded" in pc.critique_plan(feats)["risk_reasons"]

    # collinear / near-zero-area corners -> corner_geometry_bad
    feats = _good_features()
    feats["corners"] = [[10, 10], [30, 10], [50, 10], [70, 10]]
    feats["aspect_ratio"] = None
    assert "corner_geometry_bad" in pc.critique_plan(feats)["risk_reasons"]

    # only 3 corners present -> corner_geometry_bad
    feats = _good_features()
    feats["corners"] = [[10, 10], [70, 10], [70, 50]]
    assert "corner_geometry_bad" in pc.critique_plan(feats)["risk_reasons"]

    # grasp sitting on a corner -> grasp_too_close_to_corner
    feats = _good_features()
    feats["grasp_point"] = [10, 10]
    assert "grasp_too_close_to_corner" in pc.critique_plan(feats)["risk_reasons"]

    # missing fold axis -> fold_axis_bad
    feats = _good_features()
    feats["fold_axis"] = None
    assert "fold_axis_bad" in pc.critique_plan(feats)["risk_reasons"]


def test_critique_handles_empty_and_none_input():
    for arg in (None, {}):
        res = pc.critique_plan(arg)
        assert 0.0 <= res["success_probability"] <= 1.0
        assert isinstance(res["risk_reasons"], list)


# --------------------------------------------------------------------------
# trainable scaffold (numpy backend, end-to-end)
# --------------------------------------------------------------------------


def _make_towel_dataset(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for i in range(6):
        _png(str(raw / f"img_{i}.png"), val=60 + i * 25)
    out = str(tmp_path / "towel_ds")
    ds.create_towel_real_dataset(str(raw), out)
    man = ds.load_manifest(out)
    samples = man["samples"]
    good_corners = [[12, 12], [68, 12], [68, 48], [12, 48]]
    # 4 usable positives.
    for s in samples[:4]:
        lab = ds.load_label(out, s)
        lab["state"] = "flat_unfolded"
        lab["corners"] = {k: good_corners[i] for i, k in enumerate(sc.CORNER_ORDER)}
        lab["usable_for_training"] = True
        ds.save_label(out, s, lab)
    # 2 reject negatives (no corners).
    for s, st in zip(samples[4:], ("bad_view", "not_towel")):
        lab = ds.load_label(out, s)
        lab["state"] = st
        ds.save_label(out, s, lab)
    return out


def test_build_train_eval_end_to_end_numpy(tmp_path):
    towel = _make_towel_dataset(tmp_path)

    # build
    crit = str(tmp_path / "critic")
    bres = pc.build_plan_critic_dataset(towel, crit, log=lambda *_: None)
    assert bres["rows"] == 6
    assert bres["positives"] == 4 and bres["negatives"] == 2
    assert os.path.exists(os.path.join(crit, "critic_dataset.jsonl"))
    assert os.path.exists(os.path.join(crit, "meta.json"))

    # train (force the numpy backend so torch is never required)
    model_dir = str(tmp_path / "model")
    tres = pc.train_plan_critic(crit, model_dir, log=lambda *_: None, backend="numpy")
    assert tres["backend"] == "numpy"
    assert tres["n_rows"] == 6
    model_path = tres["model"]
    assert os.path.exists(model_path) and model_path.endswith("model.pt")
    # numpy model.pt is JSON-portable.
    with open(model_path) as f:
        saved = json.load(f)
    assert saved["feature_names"] == pc.FEATURE_NAMES
    # heuristic labels are linearly separable -> the tiny model should fit them.
    assert tres["train_acc"] >= 0.8

    # eval -> Markdown report
    eval_dir = str(tmp_path / "eval")
    eres = pc.eval_plan_critic(crit, model_path, eval_dir, log=lambda *_: None)
    assert eres["status"] == "ok"
    assert eres["n"] == 6
    assert 0.0 <= eres["accuracy"] <= 1.0
    assert os.path.exists(eres["report"]) and eres["report"].endswith(".md")
    report_txt = open(eres["report"]).read()
    assert "Plan-critic evaluation" in report_txt and "accuracy" in report_txt
