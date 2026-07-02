"""Tests for the crop-first real-photo towel pipeline.

Covers the new ``crop-real-towel-candidates`` localization step (Claude is asked for
ONLY the striped towel's bounding box, ignoring the bed/pillow/room), the tight
padded crop + remap metadata, the reject rule, the new
``outer_visible_corners_striped_towel_only`` label policy, and the end-to-end
crop -> label -> remap-back-to-full-image flow with overlays on BOTH the crop and
the original image.

NO network / NO SDK: Claude is driven via injected canned-JSON ``responder``s.
Images are PNGs written by the pure-Python codec, so no OpenCV/Pillow is needed.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from terafold.towel import claude_pseudolabel as cl
from terafold.towel import crop_candidates as cc
from terafold.towel.dataset import load_label, load_manifest
from terafold.towel.raw_ingest import load_input_images
from terafold.vision.imageio import imread, imwrite

STRIPED = "outer_visible_corners_striped_towel_only"


def _silent(*_a) -> None:
    pass


def _solid(path, val):
    imwrite(str(path), np.full((40, 40, 3), val, np.uint8))


def _mean(path):
    return round(float(np.asarray(imread(str(path))).mean()))


def _full_image_folder(tmp_path, n=3, W=200, H=150, tx=(60, 140), ty=(40, 110)):
    """A folder of full photos with a bright 'towel' rectangle on a dark surface."""
    d = tmp_path / "full"
    d.mkdir()
    for i in range(n):
        img = np.full((H, W, 3), 30, np.uint8)
        img[ty[0]:ty[1], tx[0]:tx[1]] = 200
        imwrite(str(d / f"p{i}.png"), img)
    return str(d)


def _loc_json(bbox=(60, 40, 140, 110), found=True, is_target=True, conf=0.9):
    return json.dumps({
        "found": found, "is_target": is_target, "bbox": list(bbox),
        "confidence": conf, "reason": "striped towel on the bed",
        "rejected_regions": ["bed sheet", "pillow", "headboard"],
    })


def _corner_json(corners, state="flat_unfolded", is_target=True, conf=0.9, visible=None):
    return json.dumps({
        "is_target": is_target, "state": state, "usable_for_training": True,
        "bbox": None,  # let the builder derive the bbox from the corners
        "corners": corners,
        "visible": visible or {"tl": True, "tr": True, "br": True, "bl": True},
        "fold_axis": "vertical", "grasp_edge": "right", "place_edge": "left",
        "confidence": conf, "risk_reasons": [], "notes": "ok",
    })


# ============================= localization prompt =============================


def test_localize_prompt_targets_towel_and_lists_ignores():
    p = cc.build_localize_prompt("beige and white striped towel", 200, 150)
    assert "beige and white striped towel" in p
    # requirement #7 sentence is present verbatim
    assert "Do not label the bed sheet, pillow, mattress, blanket, or room rectangle." in p
    # the explicit ignore list (#2)
    for token in ("bed sheet", "pillow", "blanket", "headboard", "floor", "body part"):
        assert token in p
    assert "is_target" in p and "found" in p


def test_localize_system_prompt_is_geometry_only():
    assert "bounding box" in cc.LOCALIZE_SYSTEM_PROMPT.lower()
    assert "never robot commands" in cc.LOCALIZE_SYSTEM_PROMPT.lower()


# ============================= parse_localization =============================


def test_parse_localization_accepts_valid_box():
    loc = cc.parse_localization(_loc_json(), 200, 150, min_confidence=0.5)
    assert loc["ok"] is True
    assert loc["bbox"] == [60.0, 40.0, 140.0, 110.0]
    assert loc["rejected_regions"]


@pytest.mark.parametrize("raw,reason", [
    (json.dumps({"found": False, "confidence": 0.0}), "target_not_found"),
    (_loc_json(is_target=False), "not_target"),
    (_loc_json(conf=0.2), "low_confidence<0.5"),
    (json.dumps({"found": True, "is_target": True, "bbox": [5, 5, 5, 5], "confidence": 0.9}),
     "invalid_bbox"),
])
def test_parse_localization_reject_rule(raw, reason):
    loc = cc.parse_localization(raw, 200, 150, min_confidence=0.5)
    assert loc["ok"] is False
    assert loc["reject_reason"] == reason


def test_parse_localization_unparseable_raises():
    with pytest.raises(cl.ClaudeTowelError):
        cc.parse_localization("not json at all", 200, 150)


def test_parse_localization_null_target_not_rejected():
    # An explicit JSON null for is_target/found must NOT silently reject a found towel
    # (matches the labeler's null -> True handling).
    a = cc.parse_localization(
        json.dumps({"found": True, "is_target": None, "bbox": [60, 40, 140, 110],
                    "confidence": 0.9}), 200, 150, 0.5)
    assert a["ok"] is True
    b = cc.parse_localization(
        json.dumps({"found": None, "is_target": True, "bbox": [60, 40, 140, 110],
                    "confidence": 0.9}), 200, 150, 0.5)
    assert b["ok"] is True


# ============================= crop geometry / remap =============================


def test_compute_crop_box_pads_and_clamps():
    # box [60,40,140,110] in a 200x150 image, pad 0.1 -> +/-8 in x, +/-7 in y
    x1, y1, x2, y2 = cc.compute_crop_box([60, 40, 140, 110], 200, 150, pad_frac=0.1)
    assert (x1, y1, x2, y2) == (52, 33, 148, 117)
    # padding never escapes the image bounds
    bx1, by1, bx2, by2 = cc.compute_crop_box([0, 0, 200, 150], 200, 150, pad_frac=0.5)
    assert (bx1, by1) == (0, 0) and (bx2, by2) == (200, 150)


def test_compute_crop_box_nondegenerate():
    x1, y1, x2, y2 = cc.compute_crop_box([10, 10, 10, 10], 100, 100, pad_frac=0.0)
    assert x2 > x1 and y2 > y1


def test_remap_roundtrip_no_scale():
    crop = {"x_off": 52, "y_off": 33, "scale_x": 1.0, "scale_y": 1.0}
    assert cc.remap_xy([8, 7], crop) == [60.0, 40.0]
    corners = {"tl": [8, 7], "tr": [88, 7], "br": [88, 77], "bl": [8, 77]}
    out = cc.remap_corners(corners, crop)
    assert out["tl"] == [60.0, 40.0] and out["br"] == [140.0, 110.0]
    assert cc.remap_bbox([8, 7, 88, 77], crop) == [60.0, 40.0, 140.0, 110.0]


def test_remap_with_downscale():
    # region 200 wide downscaled to 100 -> scale 0.5; crop center maps to orig center
    crop = {"x_off": 100, "y_off": 100, "scale_x": 0.5, "scale_y": 0.5}
    assert cc.remap_xy([50, 50], crop) == [200.0, 200.0]


def test_remap_passes_through_none_corner():
    crop = {"x_off": 10, "y_off": 10, "scale_x": 1.0, "scale_y": 1.0}
    out = cc.remap_corners({"tl": [1, 1], "tr": None, "br": [5, 5], "bl": None}, crop)
    assert out["tr"] is None and out["bl"] is None and out["tl"] == [11.0, 11.0]


# ============================= crop orchestration =============================


def test_crop_real_towel_candidates_end_to_end(tmp_path):
    inp = _full_image_folder(tmp_path, n=3)
    out = str(tmp_path / "crops")
    res = cc.crop_real_towel_candidates(inp, out, responder=lambda p, w, h: _loc_json(),
                                        pad_frac=0.1, max_crop_size=None, log=_silent)
    assert res["status"] == "ok" and res["num_crops"] == 3 and res["num_rejected"] == 0

    man = load_manifest(out)
    assert man["kind"] == cc.CROPS_MANIFEST_KIND
    assert man["target_description"] == cc.DEFAULT_TARGET_DESCRIPTION
    e0 = man["images"][0]
    # crop file exists and is the tight region (96x84 after 0.1 padding)
    assert os.path.exists(os.path.join(out, e0["image"]))
    assert e0["width"] == 96 and e0["height"] == 84
    cm = e0["crop"]
    assert cm["x_off"] == 52 and cm["y_off"] == 33
    assert cm["orig_image"].endswith(".png") and os.path.exists(cm["orig_image"])
    # localization overlay drawn on the ORIGINAL image
    assert e0["localize_overlay"] and os.path.exists(os.path.join(out, e0["localize_overlay"]))


def test_crop_dataset_is_consumable_by_loader(tmp_path):
    inp = _full_image_folder(tmp_path, n=2)
    out = str(tmp_path / "crops")
    cc.crop_real_towel_candidates(inp, out, responder=lambda p, w, h: _loc_json(),
                                  max_crop_size=None, log=_silent)
    loaded = load_input_images(out)
    assert len(loaded["images"]) == 2
    # the crop metadata must survive into the loader output (so the labeler can remap)
    assert all("crop" in im for im in loaded["images"])


def test_crop_records_rejections(tmp_path):
    inp = _full_image_folder(tmp_path, n=2)
    out = str(tmp_path / "crops")
    res = cc.crop_real_towel_candidates(
        inp, out, responder=lambda p, w, h: _loc_json(is_target=False), log=_silent)
    assert res["num_crops"] == 0 and res["num_rejected"] == 2
    man = load_manifest(out)
    assert len(man["rejected"]) == 2
    assert man["rejected"][0]["reject_reason"] == "not_target"


def test_crop_resume_and_force(tmp_path):
    inp = _full_image_folder(tmp_path, n=3)
    out = str(tmp_path / "crops")
    r1 = cc.crop_real_towel_candidates(inp, out, responder=lambda p, w, h: _loc_json(),
                                       max_crop_size=None, log=_silent)
    assert r1["num_crops"] == 3
    # resume reuses existing crops (no new Claude calls counted as crops)
    r2 = cc.crop_real_towel_candidates(inp, out, responder=lambda p, w, h: _loc_json(),
                                       max_crop_size=None, resume=True, log=_silent)
    assert r2["resumed"] == 3 and r2["num_crops"] == 3
    # force re-crops everything (resume ignored)
    r3 = cc.crop_real_towel_candidates(inp, out, responder=lambda p, w, h: _loc_json(),
                                       max_crop_size=None, resume=True, force=True, log=_silent)
    assert r3["resumed"] == 0 and r3["num_crops"] == 3


def test_crop_no_api_key_graceful(tmp_path):
    inp = _full_image_folder(tmp_path, n=1)
    res = cc.crop_real_towel_candidates(inp, str(tmp_path / "o"), log=_silent)
    # no responder/client/key -> graceful no_api_key (never raises, never prints a key)
    assert res["status"] == "no_api_key" and res["instructions"]


# ============================= striped-towel label policy =============================


def test_striped_policy_registered_with_required_text():
    assert STRIPED in cl.LABEL_POLICIES
    p = cl.LABEL_POLICY_PROMPTS[STRIPED]
    assert "Do not label the bed sheet, pillow, mattress, blanket, or room rectangle." in p
    assert "REJECT RULE" in p
    assert "Do NOT label the bounding box" in p


def test_striped_reject_rule_when_not_target():
    lab = cl.parse_and_build_pseudolabel(
        _corner_json({"tl": [10, 10], "tr": [90, 12], "br": [88, 90], "bl": [12, 88]},
                     is_target=False),
        "images/x.png", 100, 100, label_policy=STRIPED)
    assert "not_striped_towel" in lab["risk_reasons"]
    assert lab["state"] == "not_towel"
    assert lab["is_target"] is False
    assert lab["usable_for_training"] is False


def test_striped_keeps_corners_when_target():
    lab = cl.parse_and_build_pseudolabel(
        _corner_json({"tl": [10, 10], "tr": [90, 12], "br": [88, 90], "bl": [12, 88]}),
        "images/x.png", 100, 100, label_policy=STRIPED)
    assert "not_striped_towel" not in lab["risk_reasons"]
    assert lab["is_target"] is True
    assert all(lab["corners"][k] is not None for k in ("tl", "tr", "br", "bl"))


def test_striped_flags_loose_bbox():
    lab = cl.parse_and_build_pseudolabel(
        _corner_json({"tl": [10, 10], "tr": [90, 10], "br": [90, 90], "bl": [10, 90]}),
        "images/x.png", 100, 100, label_policy=STRIPED)
    assert "corners_equal_bbox" in lab["risk_reasons"]


def test_default_policy_unaffected_by_is_target():
    # The reject rule and bbox flag are gated on the striped policy only.
    lab = cl.parse_and_build_pseudolabel(
        _corner_json({"tl": [10, 10], "tr": [90, 10], "br": [90, 90], "bl": [10, 90]},
                     is_target=False),
        "images/x.png", 100, 100)  # default policy
    assert "not_striped_towel" not in lab["risk_reasons"]
    assert "corners_equal_bbox" not in lab["risk_reasons"]


# ============================= crop -> label -> remap end-to-end =============================


def test_label_on_crops_remaps_back_to_full_image(tmp_path):
    inp = _full_image_folder(tmp_path, n=2)
    crops = str(tmp_path / "crops")
    cc.crop_real_towel_candidates(inp, crops, responder=lambda p, w, h: _loc_json(),
                                  pad_frac=0.1, max_crop_size=None, log=_silent)
    # Crop region is x_off=52,y_off=33. The towel (orig [60..140,40..110]) sits at
    # crop coords x:[8..88], y:[7..77]. Claude returns those crop-space corners.
    crop_corners = {"tl": [8, 7], "tr": [88, 7], "br": [88, 77], "bl": [8, 77]}
    out = str(tmp_path / "labeled")
    res = cl.claude_label_towel_folder(
        crops, out, responder=lambda p, w, h: _corner_json(crop_corners),
        label_policy=STRIPED, min_confidence=0.65, force=True, log=_silent)
    assert res["status"] == "ok" and res["num_images"] == 2

    man = load_manifest(out)
    s0 = man["samples"][0]
    lab = load_label(out, s0)
    # crop-space corners stay canonical (match the copied crop image)
    assert lab["corners"]["tl"] == [8.0, 7.0]
    # remapped full-image corners land back on the ORIGINAL towel rectangle
    fi = lab["full_image"]["corners"]
    assert fi["tl"] == [60.0, 40.0] and fi["br"] == [140.0, 110.0]
    assert lab["crop"]["x_off"] == 52
    # overlays on BOTH crop and full image
    assert s0["overlay"] and os.path.exists(os.path.join(out, s0["overlay"]))
    assert s0["full_overlay"] and os.path.exists(os.path.join(out, s0["full_overlay"]))
    assert "crop" in s0


def test_label_without_crop_metadata_has_no_full_image(tmp_path):
    # A plain (non-crop) folder must be completely unaffected: no full_image block.
    d = tmp_path / "plain"
    d.mkdir()
    imwrite(str(d / "a.png"), np.full((100, 100, 3), 180, np.uint8))
    out = str(tmp_path / "lab")
    cl.claude_label_towel_folder(
        str(d), out,
        responder=lambda p, w, h: _corner_json(
            {"tl": [10, 10], "tr": [90, 12], "br": [88, 90], "bl": [12, 88]}),
        label_policy=STRIPED, force=True, log=_silent)
    s0 = load_manifest(out)["samples"][0]
    lab = load_label(out, s0)
    assert "full_image" not in lab and "crop" not in lab
    assert s0["full_overlay"] is None and "crop" not in s0


# ============================= resume id-collision regression =============================
# A new image added before existing ones on --resume must NOT reuse an id a resumed
# sample owns (which would silently overwrite that sample's image/label on disk).


def test_label_resume_new_image_does_not_clobber(tmp_path):
    d = tmp_path / "imgs"
    d.mkdir()
    _solid(d / "b.png", 100)
    _solid(d / "c.png", 150)
    out = str(tmp_path / "lab")
    resp = lambda p, w, h: _corner_json(  # noqa: E731
        {"tl": [2, 2], "tr": [35, 2], "br": [35, 35], "bl": [2, 35]})
    cl.claude_label_towel_folder(str(d), out, responder=resp, label_policy=STRIPED, log=_silent)
    _solid(d / "a.png", 50)  # sorts FIRST -> would grab id 000001 under the old scheme
    cl.claude_label_towel_folder(str(d), out, responder=resp, label_policy=STRIPED,
                                 resume=True, log=_silent)
    man = load_manifest(out)
    ids = [s["id"] for s in man["samples"]]
    imgs = [s["image"] for s in man["samples"]]
    assert len(ids) == 3 and len(set(ids)) == 3            # no id collision
    assert len(set(imgs)) == 3                             # no aliased image files
    # all three distinct source colors survive (b=100 was NOT overwritten by a=50)
    assert sorted(_mean(os.path.join(out, s["image"])) for s in man["samples"]) == [50, 100, 150]


def test_crop_resume_new_image_does_not_clobber(tmp_path):
    d = tmp_path / "full"
    d.mkdir()
    _solid(d / "b.png", 100)
    _solid(d / "c.png", 150)
    out = str(tmp_path / "crops")
    resp = lambda p, w, h: json.dumps(  # noqa: E731
        {"found": True, "is_target": True, "bbox": [5, 5, 35, 35], "confidence": 0.9})
    cc.crop_real_towel_candidates(str(d), out, responder=resp, max_crop_size=None,
                                  pad_frac=0.0, log=_silent)
    _solid(d / "a.png", 50)  # sorts FIRST
    cc.crop_real_towel_candidates(str(d), out, responder=resp, max_crop_size=None,
                                  pad_frac=0.0, resume=True, log=_silent)
    man = load_manifest(out)
    ids = [e["id"] for e in man["images"]]
    imgs = [e["image"] for e in man["images"]]
    assert len(ids) == 3 and len(set(ids)) == 3            # no id collision
    assert len(set(imgs)) == 3                             # no aliased crop files
    # each crop's pixels still match its own source color (b=100 not clobbered by a=50)
    assert sorted(_mean(os.path.join(out, e["image"])) for e in man["images"]) == [50, 100, 150]
