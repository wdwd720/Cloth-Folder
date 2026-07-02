"""Tests for the outer_visible_corners_any_state Claude LABEL policy.

Covers: the new prompt/policy, the loose-bounding-box detector (folded / partially
folded / rounded-corner / bbox cases), provenance of the policy name, --force
relabel, and that the DEFAULT policy behavior is unchanged. No SDK/network — the
labeler is driven through canned-JSON responders.
"""

from __future__ import annotations

import json
import os

import numpy as np

from terafold.towel import claude_pseudolabel as cl
from terafold.towel import overlay as ov
from terafold.towel.dataset import create_towel_real_dataset, load_label, load_manifest
from terafold.vision.imageio import imwrite

W = H = 120
POLICY = "outer_visible_corners_any_state"

# An axis-aligned bounding-box quad (the lazy / loose label we want to catch).
BBOX_CORNERS = {"tl": [10, 10], "tr": [110, 10], "br": [110, 110], "bl": [10, 110]}
# A genuinely rotated/skewed quad that hugs an angled towel (NOT its bbox).
ROTATED_CORNERS = {"tl": [22, 12], "tr": [102, 24], "br": [92, 104], "bl": [14, 96]}


def _silent(*_a) -> None:
    pass


def _json(corners, state="folded_success", conf=0.9, visible=None):
    return json.dumps({
        "state": state, "usable_for_training": True, "bbox": [10, 10, 110, 110],
        "corners": corners, "visible": visible or {"tl": True, "tr": True, "br": True, "bl": True},
        "fold_axis": "vertical", "grasp_edge": "right", "place_edge": "left",
        "confidence": conf, "risk_reasons": [], "notes": "ok",
    })


# --------------------------- prompt / policy registry ---------------------------


def test_label_policies_registered():
    assert "outer_visible_corners_any_state" in cl.LABEL_POLICIES
    assert "default" in cl.LABEL_POLICIES


def test_outer_prompt_has_rules_and_self_check():
    p = cl.LABEL_POLICY_PROMPTS[POLICY]
    assert "Do NOT label the bounding box" in p
    assert "visible outer corners" in p.lower()
    assert "QUALITY SELF-CHECK" in p
    assert "intersection of the two outer edge" in p   # rounded-corner rule
    assert "not just the bounding box" in p.lower()


def test_labeler_selects_policy_prompt():
    lab = cl.ClaudeTowelLabeler(label_policy=POLICY, responder=lambda p, w, h: "{}")
    assert lab.label_policy == POLICY
    assert "Do NOT label the bounding box" in lab.user_prompt
    # default keeps the original prompt
    d = cl.ClaudeTowelLabeler(responder=lambda p, w, h: "{}")
    assert d.label_policy == "default" and "Do NOT label the bounding box" not in d.user_prompt


# --------------------------- loose bbox detector ---------------------------


def test_bbox_detector_flags_loose_box_only_under_new_policy():
    new = cl.parse_and_build_pseudolabel(_json(BBOX_CORNERS), "images/x.png", W, H, label_policy=POLICY)
    assert "corners_equal_bbox" in new["risk_reasons"]
    # default policy must be unchanged (no new risk added)
    default = cl.parse_and_build_pseudolabel(_json(BBOX_CORNERS), "images/x.png", W, H)
    assert "corners_equal_bbox" not in default["risk_reasons"]


def test_rotated_quad_not_flagged_as_bbox():
    lab = cl.parse_and_build_pseudolabel(_json(ROTATED_CORNERS), "images/x.png", W, H, label_policy=POLICY)
    assert "corners_equal_bbox" not in lab["risk_reasons"]
    assert cl.corners_look_like_bbox([ROTATED_CORNERS[k] for k in ("tl", "tr", "br", "bl")], W, H) is False
    assert cl.corners_look_like_bbox([BBOX_CORNERS[k] for k in ("tl", "tr", "br", "bl")], W, H) is True


def test_folded_and_partial_accepted_states_keep_corners():
    for state in ("folded_success", "partially_folded"):
        lab = cl.parse_and_build_pseudolabel(_json(ROTATED_CORNERS, state=state), "images/x.png",
                                             W, H, label_policy=POLICY)
        assert lab["state"] == state
        assert lab["label_policy"] == POLICY
        # corners preserved (4 present), not rejected just for being folded
        assert all(lab["corners"][k] is not None for k in ("tl", "tr", "br", "bl"))


def test_rounded_corner_label_stored():
    # A rounded-corner towel: Claude returns the edge-intersection points (a normal
    # rotated quad). It should label fine, not be flagged as bbox.
    lab = cl.parse_and_build_pseudolabel(_json(ROTATED_CORNERS, state="flat_unfolded"),
                                         "images/x.png", W, H, label_policy=POLICY)
    assert "corners_equal_bbox" not in lab["risk_reasons"]
    assert lab["confidence"] == 0.9


# --------------------------- provenance + force ---------------------------


def _real(tmp_path, n=3):
    raw = tmp_path / "raw"
    raw.mkdir()
    for i in range(n):
        imwrite(str(raw / f"p{i}.png"), np.full((H, W, 3), 60 + i * 30, np.uint8))
    out = str(tmp_path / "real")
    create_towel_real_dataset(str(raw), out, source="user_photo", log=_silent)
    return out


def test_policy_saved_in_provenance(tmp_path):
    real = _real(tmp_path, n=2)
    out = str(tmp_path / "lab")
    cl.claude_label_towel_folder(real, out, responder=lambda p, w, h: _json(ROTATED_CORNERS),
                                 label_policy=POLICY, log=_silent)
    man = load_manifest(out)
    assert man["label_policy"] == POLICY and man["prompt_name"] == POLICY
    s0 = man["samples"][0]
    assert s0["label_policy"] == POLICY
    assert load_label(out, s0)["label_policy"] == POLICY


def test_force_relabels_overwriting_old(tmp_path):
    real = _real(tmp_path, n=3)
    out = str(tmp_path / "lab")
    r1 = cl.claude_label_towel_folder(real, out, responder=lambda p, w, h: _json(ROTATED_CORNERS),
                                      label_policy=POLICY, log=_silent)
    assert r1["num_images"] == 3
    # resume alone would reuse; --force overrides reuse and relabels everything.
    r2 = cl.claude_label_towel_folder(real, out, responder=lambda p, w, h: _json(ROTATED_CORNERS),
                                      label_policy=POLICY, resume=True, force=True, log=_silent)
    assert r2["resumed"] == 0 and r2["num_images"] == 3
    # without force, resume reuses prior labels
    r3 = cl.claude_label_towel_folder(real, out, responder=lambda p, w, h: _json(ROTATED_CORNERS),
                                      label_policy=POLICY, resume=True, log=_silent)
    assert r3["resumed"] == 3


def test_unknown_label_policy_errors(tmp_path):
    real = _real(tmp_path, n=1)
    res = cl.claude_label_towel_folder(real, str(tmp_path / "o"),
                                       responder=lambda p, w, h: _json(ROTATED_CORNERS),
                                       label_policy="bogus", log=_silent)
    assert res["status"] == "error" and "label_policies" in res


def test_default_policy_unchanged(tmp_path):
    # Default policy must still embed label_policy='default' and not add bbox risk.
    real = _real(tmp_path, n=1)
    out = str(tmp_path / "lab")
    cl.claude_label_towel_folder(real, out, responder=lambda p, w, h: _json(BBOX_CORNERS),
                                 log=_silent)  # default policy
    lab = load_label(out, load_manifest(out)["samples"][0])
    assert lab["label_policy"] == "default"
    assert "corners_equal_bbox" not in lab["risk_reasons"]


# --------------------------- better overlay ---------------------------


def test_overlay_renders_with_labels_and_hidden_corner(tmp_path):
    img = str(tmp_path / "img.png")
    imwrite(img, np.full((H, W, 3), 200, np.uint8))
    label = {
        "corners": {"tl": [22, 12], "tr": [102, 24], "br": [92, 104], "bl": [14, 96]},
        "visible": {"tl": True, "tr": True, "br": False, "bl": True},  # one hidden
        "bbox": [14, 12, 102, 104], "fold_axis": "vertical",
    }
    res = ov.draw_towel_overlay(img, label, str(tmp_path / "ov.png"), draw_dims=10)
    assert res["rendered"] is True and os.path.exists(res["out"])
