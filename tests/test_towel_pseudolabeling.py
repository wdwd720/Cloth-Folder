"""Tests for the external-image + Claude pseudo-labeling pipeline.

NO network and NO SDK: the Kaggle download is exercised via its graceful
"unavailable" branch and an injected ``download_fn``; Claude is exercised via an
injected ``responder`` (canned JSON). Images are PNGs written by the pure-Python
codec, so no OpenCV/Pillow is needed.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from terafold.towel import claude_pseudolabel as cl
from terafold.towel import filtering as ft
from terafold.towel import raw_ingest as ri
from terafold.towel import review as rv
from terafold.towel import schema as sc
from terafold.towel import yolo_export as yx
from terafold.towel.dataset import load_label, load_manifest
from terafold.vision.imageio import imwrite


def _silent(*_a) -> None:
    pass


def _png(path, w=120, h=90, val=190):
    img = np.full((h, w, 3), 25, np.uint8)
    img[10:h - 10, 10:w - 10] = val
    imwrite(path, img)
    return path


def _raw_source_folder(tmp_path, names):
    d = tmp_path / "src"
    d.mkdir()
    for i, name in enumerate(names):
        _png(str(d / name), val=60 + i * 25)
    return str(d)


# ============================ raw ingest + manifest ============================


def test_ingest_raw_folder_manifest_schema(tmp_path):
    src = _raw_source_folder(tmp_path, ["white_towel_1.png", "hotel_towel_2.png", "cat.png"])
    out = str(tmp_path / "raw")
    res = ri.ingest_raw_folder(src, out, source="kaggle", dataset_slug="o/s",
                               license_note="CC-BY", log=_silent)
    assert res["status"] == "ok" and res["num_images"] == 3
    man = ri.load_raw_manifest(out)
    assert man["kind"] == ri.RAW_MANIFEST_KIND and man["source"] == "kaggle"
    assert man["dataset_slug"] == "o/s" and man["license_note"] == "CC-BY"
    assert ri.validate_raw_manifest(man) == []
    im0 = man["images"][0]
    assert im0["width"] == 120 and im0["height"] == 90
    assert im0["hash"] and im0["original"].endswith(".png") and im0["source"] == "kaggle"


def test_ingest_skips_tiny_and_dedups(tmp_path):
    d = tmp_path / "src"
    d.mkdir()
    _png(str(d / "ok.png"), w=120, h=90, val=100)
    _png(str(d / "dup.png"), w=120, h=90, val=100)   # identical bytes -> dedup
    _png(str(d / "tiny.png"), w=20, h=20, val=200)   # below min_size -> skipped
    out = str(tmp_path / "raw")
    res = ri.ingest_raw_folder(str(d), out, source="openimages", min_size=64, log=_silent)
    assert res["num_images"] == 1
    assert res["skipped_duplicates"] == 1 and res["skipped_small"] == 1


def test_validate_raw_manifest_flags_problems():
    bad = {"kind": "nope", "images": [{"id": "1"}]}
    problems = ri.validate_raw_manifest(bad)
    assert any("kind" in p for p in problems)
    assert any("source" in p for p in problems)
    assert any("image" in p for p in problems)  # missing image/original/hash keys


# ============================ kaggle importer ============================


def test_kaggle_gracefully_unavailable_without_client(tmp_path, monkeypatch):
    monkeypatch.setattr(ri, "have_kagglehub", lambda: False)
    monkeypatch.setattr(ri, "have_kaggle_cli", lambda: False)
    res = ri.import_kaggle_towel_dataset("owner/slug", str(tmp_path / "k"), log=_silent)
    assert res["status"] == "unavailable"
    assert "kaggle" in res["install"]
    assert any("pip install kaggle" in line for line in res["instructions"])


def test_kaggle_ingest_via_injected_downloader(tmp_path):
    src = _raw_source_folder(tmp_path, ["towel_a.png", "towel_b.png"])

    def fake_download(dataset, dest, log):
        assert dataset == "owner/slug"
        return src

    out = str(tmp_path / "k")
    res = ri.import_kaggle_towel_dataset("owner/slug", out, download_fn=fake_download, log=_silent)
    assert res["status"] == "ok" and res["num_images"] == 2
    man = ri.load_raw_manifest(out)
    assert man["source"] == "kaggle" and man["dataset_slug"] == "owner/slug"


def test_kaggle_src_dir_offline(tmp_path):
    src = _raw_source_folder(tmp_path, ["towel_x.png"])
    out = str(tmp_path / "k")
    res = ri.import_kaggle_towel_dataset("owner/slug", out, src_dir=src, log=_silent)
    assert res["status"] == "ok" and res["num_images"] == 1


# ============================ open images ============================


def test_openimages_manual_instructions(tmp_path):
    res = ri.import_openimages_towels(str(tmp_path / "oi"), log=_silent)
    assert res["status"] == "manual_instructions"
    assert any("Towel" in line for line in res["instructions"])
    assert any("corner" in line.lower() for line in res["instructions"])  # explains box != corners


def test_openimages_ingest_with_src_dir(tmp_path):
    src = _raw_source_folder(tmp_path, ["img1.png", "img2.png"])
    out = str(tmp_path / "oi")
    res = ri.import_openimages_towels(out, src_dir=src, log=_silent)
    assert res["status"] == "ok" and res["num_images"] == 2
    assert ri.load_raw_manifest(out)["source"] == "openimages"


# ============================ candidate filtering ============================


def test_filter_by_filename_keyword(tmp_path):
    src = _raw_source_folder(tmp_path, ["white_towel.png", "bath_towel.png", "random_dog.png"])
    raw = str(tmp_path / "raw")
    ri.ingest_raw_folder(src, raw, source="kaggle", log=_silent)
    out = str(tmp_path / "cand")
    res = ft.filter_towel_images(raw, out, mode="filename", log=_silent)
    assert res["status"] == "ok"
    assert res["kept"] == 2 and res["rejected"] == 1  # dog dropped
    man = load_raw_manifest_safe(out)
    assert man["filter_mode"] == "filename" and man["num_images"] == 2


def test_filter_vlm_seam_keeps_non_keyword(tmp_path):
    src = _raw_source_folder(tmp_path, ["mystery_1.png", "mystery_2.png"])
    raw = str(tmp_path / "raw")
    ri.ingest_raw_folder(src, raw, source="openimages", log=_silent)
    out = str(tmp_path / "cand")
    # VLM says the first is a towel, the second is not.
    seen = {"n": 0}

    def vlm(path):
        seen["n"] += 1
        return seen["n"] == 1

    res = ft.filter_towel_images(raw, out, mode="filename_or_vlm", vlm_responder=vlm, log=_silent)
    assert res["kept"] == 1 and res["vlm_used"] is True


def load_raw_manifest_safe(root):
    return ri.load_raw_manifest(root)


# ============================ Claude parser / validator ============================

W, H = 120, 90
CLEAN = {
    "state": "flat_unfolded", "usable_for_training": True,
    "bbox": [12, 12, 108, 78],
    "corners": {"tl": [12, 12], "tr": [108, 12], "br": [108, 78], "bl": [12, 78]},
    "visible": {"tl": True, "tr": True, "br": True, "bl": True},
    "fold_axis": "vertical", "grasp_edge": "right", "place_edge": "left",
    "confidence": 0.95, "risk_reasons": [], "notes": "clean flat towel",
}


def test_parser_valid_label_pending():
    lab = cl.parse_and_build_pseudolabel(json.dumps(CLEAN), "images/1.png", W, H, min_confidence=0.75)
    assert lab["pseudolabel"] is True and lab["source"] == "claude_pseudolabel"
    assert lab["state"] == "flat_unfolded"
    assert lab["approval_status"] == "pending" and lab["review_needed"] is False
    assert lab["usable_for_training"] is False     # never usable before approval
    assert sc.validate_label(lab) == []
    assert sc.label_corners_xy(lab) is not None


def test_parser_invalid_json_rejected():
    with pytest.raises(cl.ClaudeTowelError):
        cl.parse_and_build_pseudolabel("not json at all", "images/1.png", W, H)


def test_parser_out_of_bounds_flagged():
    bad = {**CLEAN, "corners": {**CLEAN["corners"], "tr": [9999, 9999]}}
    lab = cl.parse_and_build_pseudolabel(json.dumps(bad), "images/1.png", W, H)
    assert "out_of_bounds" in lab["risk_reasons"]
    assert lab["approval_status"] == "review_needed"
    # clamped back into bounds for storage
    assert lab["corners"]["tr"][0] <= W - 1 and lab["corners"]["tr"][1] <= H - 1


def test_parser_bad_state_not_trainable():
    bad = {**CLEAN, "state": "not_towel", "confidence": 0.2}
    lab = cl.parse_and_build_pseudolabel(json.dumps(bad), "images/1.png", W, H)
    assert lab["approval_status"] == "review_needed"
    assert any(r.startswith("non_trainable_state") for r in lab["risk_reasons"])
    assert sc.is_usable(lab) is False


def test_parser_self_crossing_geometry_flagged():
    bad = {**CLEAN, "corners": {"tl": [12, 12], "tr": [108, 12], "br": [12, 78], "bl": [108, 78]}}
    lab = cl.parse_and_build_pseudolabel(json.dumps(bad), "images/1.png", W, H)
    assert "geometry_impossible" in lab["risk_reasons"]
    assert lab["geometry_error"] == 1.0


def test_parser_low_confidence_flagged():
    low = {**CLEAN, "confidence": 0.40}
    lab = cl.parse_and_build_pseudolabel(json.dumps(low), "images/1.png", W, H, min_confidence=0.75)
    assert any(r.startswith("low_confidence") for r in lab["risk_reasons"])
    assert lab["approval_status"] == "review_needed"


# ============================ Claude folder labeling ============================

SCENARIOS = {
    "000001": CLEAN,                                            # clean flat -> pending
    "000002": {**CLEAN, "state": "not_towel", "confidence": 0.2},   # reject state
    "000003": {**CLEAN, "confidence": 0.50},                   # low confidence
    "000004": {**CLEAN, "corners": {"tl": [12, 12], "tr": [108, 12],
                                    "br": [12, 78], "bl": [108, 78]}},  # self-cross
}


def _responder(image_path, w, h):
    sid = os.path.splitext(os.path.basename(image_path))[0]
    return json.dumps(SCENARIOS[sid])


def _raw_with_scenarios(tmp_path, names):
    src = _raw_source_folder(tmp_path, names)
    raw = str(tmp_path / "raw")
    ri.ingest_raw_folder(src, raw, source="kaggle", log=_silent)
    return raw


def test_claude_label_missing_key_graceful(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    raw = _raw_with_scenarios(tmp_path, ["01.png"])
    res = cl.claude_label_towel_folder(raw, str(tmp_path / "ps"), log=_silent)  # no responder/key
    assert res["status"] == "no_api_key"
    assert any("ANTHROPIC_API_KEY" in line for line in res["instructions"])


def test_claude_label_folder_writes_labels_and_overlays(tmp_path):
    raw = _raw_with_scenarios(tmp_path, ["01.png", "02.png", "03.png", "04.png"])
    out = str(tmp_path / "ps")
    res = cl.claude_label_towel_folder(raw, out, responder=_responder, log=_silent)
    assert res["status"] == "ok" and res["num_images"] == 4
    # 1 clean pending + 3 flagged for review
    assert res["review_needed"] == 3 and res["pending"] == 1
    man = load_manifest(out)
    assert man["kind"] == "pseudolabeled" and man["pseudolabel"] is True
    # every sample has a label JSON; the clean one rendered an overlay
    for s in man["samples"]:
        lab = load_label(out, s)
        assert lab["pseudolabel"] is True and lab["usable_for_training"] is False
        if s["id"] == "000001":
            assert os.path.exists(os.path.join(out, s["overlay"]))


# ============================ review + export gating ============================


def _pseudo_dataset(tmp_path, names=("01.png", "02.png", "03.png", "04.png")):
    raw = _raw_with_scenarios(tmp_path, list(names))
    out = str(tmp_path / "ps")
    cl.claude_label_towel_folder(raw, out, responder=_responder, log=_silent)
    return out


def test_export_excludes_pseudolabels_by_default(tmp_path):
    ps = _pseudo_dataset(tmp_path)
    yolo = str(tmp_path / "yolo")
    res = yx.export_yolo_towel_pose(ps, yolo, log=_silent)   # default include="none"
    assert res["exported"] == 0
    assert res["skipped_pseudolabels"] == 4


def test_export_approved_only_includes_approved(tmp_path):
    ps = _pseudo_dataset(tmp_path)
    # Auto-approve high-confidence + good geometry: only the clean flat one qualifies.
    a = rv.approve_high_confidence_pseudolabels(ps, min_confidence=0.90,
                                                max_geometry_error=0.10, log=_silent)
    assert a["auto_approved"] == 1 and a["needs_review"] == 3
    yolo = str(tmp_path / "yolo")
    res = yx.export_yolo_towel_pose(ps, yolo, include_pseudolabels="approved_only", log=_silent)
    assert res["exported"] == 1
    # Provenance is preserved through pseudo-labeling: these images came from kaggle,
    # so they report under the kaggle bucket (pseudolabel-ness is gated separately).
    assert res["by_source"]["kaggle"] == 1
    assert res["skipped_unapproved_pseudo"] == 3


def test_high_confidence_autoapprove_only_valid_geometry(tmp_path):
    ps = _pseudo_dataset(tmp_path)
    rv.approve_high_confidence_pseudolabels(ps, min_confidence=0.90,
                                            max_geometry_error=0.10, log=_silent)
    man = load_manifest(ps)
    status = {s["id"]: load_label(ps, s)["approval_status"] for s in man["samples"]}
    assert status["000001"] == "auto_approved"   # clean, high-conf, good geometry
    assert status["000002"] == "review_needed"   # not_towel
    assert status["000003"] == "review_needed"   # low confidence
    assert status["000004"] == "review_needed"   # self-crossing geometry


def test_review_status_affects_export(tmp_path):
    ps = _pseudo_dataset(tmp_path, names=("01.png", "02.png"))  # clean flat + not_towel

    # Operator approves everything they see.
    answers = iter(["approve", "approve", "approve", "approve"])
    rv.review_pseudolabels(ps, input_fn=lambda _p: next(answers, "skip"), log=_silent)

    lab1 = load_label(ps, load_manifest(ps)["samples"][0])
    lab2 = load_label(ps, load_manifest(ps)["samples"][1])
    assert lab1["approval_status"] == "approved" and lab1["usable_for_training"] is True
    # Approving a not_towel must NOT make it trainable.
    assert lab2["approval_status"] == "approved" and lab2["usable_for_training"] is False
    assert os.path.exists(os.path.join(ps, "review_report.md"))

    yolo = str(tmp_path / "yolo")
    res = yx.export_yolo_towel_pose(ps, yolo, include_pseudolabels="approved_only", log=_silent)
    assert res["exported"] == 1  # only the trainable, approved flat towel


def test_export_rejects_invalid_include_mode(tmp_path):
    ps = _pseudo_dataset(tmp_path, names=("01.png",))
    with pytest.raises(ValueError):
        yx.export_yolo_towel_pose(ps, str(tmp_path / "y"), include_pseudolabels="all")


def test_review_report_counts(tmp_path):
    ps = _pseudo_dataset(tmp_path)

    def answer(_prompt):
        return "skip"  # leave everything pending; just smoke-test the report writer

    res = rv.review_pseudolabels(ps, input_fn=answer, log=_silent)
    assert res["status"] == "ok"
    assert os.path.exists(res["report"])
    with open(res["report"]) as f:
        txt = f.read()
    assert "approved" in txt and "need review" in txt
