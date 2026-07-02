"""Regression tests for the shared input-shape loader.

`create-towel-real-dataset` writes a towel-dataset manifest (`samples`), while the
external/generated pipeline used a raw-image manifest (`images`). These tests pin
that BOTH `filter-towel-images` and `claude-label-towel-folder` now accept every
shape: real-user datasets, raw/candidate manifests, bare image folders, and dirs
with an images/ subfolder — preserving provenance and never auto-approving.
"""

from __future__ import annotations

import json
import os

import numpy as np

from terafold.towel import claude_pseudolabel as cl
from terafold.towel import filtering as ft
from terafold.towel import raw_ingest as ri
from terafold.towel.dataset import create_towel_real_dataset, load_label, load_manifest
from terafold.vision.imageio import imwrite


def _silent(*_a) -> None:
    pass


def _clean(w, h):
    return {
        "state": "flat_unfolded", "usable_for_training": True,
        "bbox": [6, 6, w - 6, h - 6],
        "corners": {"tl": [6, 6], "tr": [w - 6, 6], "br": [w - 6, h - 6], "bl": [6, h - 6]},
        "visible": {"tl": True, "tr": True, "br": True, "bl": True},
        "fold_axis": "vertical", "grasp_edge": "right", "place_edge": "left",
        "confidence": 0.9, "risk_reasons": [], "notes": "ok",
    }


def _responder(path, w, h):
    return json.dumps(_clean(int(w), int(h)))


def _real_dataset(tmp_path, n=5, source="user_photo"):
    raw = tmp_path / "raw_jpg"
    raw.mkdir()
    for i in range(n):
        imwrite(str(raw / f"photo_{i}.png"), np.full((90, 120, 3), 40 + i * 20, np.uint8))
    out = str(tmp_path / "towel_real_v1")
    create_towel_real_dataset(str(raw), out, source=source, log=_silent)
    return out


def _bare_folder(tmp_path, n=3):
    d = tmp_path / "bare"
    d.mkdir()
    for i in range(n):
        imwrite(str(d / f"{i + 1:06d}.png"), np.full((80, 100, 3), 60 + i * 30, np.uint8))
    return str(d)


# --------------------------- loader ---------------------------


def test_loader_handles_real_dataset(tmp_path):
    real = _real_dataset(tmp_path, n=5)
    ld = ri.load_input_images(real)
    assert ld["kind"] == "real" and ld["source"] == "user_photo"
    assert len(ld["images"]) == 5
    assert all(e["source"] == "user_photo" for e in ld["images"])


def test_loader_handles_raw_manifest(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    for i in range(2):
        imwrite(str(src / f"towel_{i}.png"), np.full((90, 120, 3), 100 + i * 30, np.uint8))
    raw = str(tmp_path / "raw")
    ri.ingest_raw_folder(str(src), raw, source="kaggle", log=_silent)
    ld = ri.load_input_images(raw)
    assert ld["kind"] == ri.RAW_MANIFEST_KIND and ld["source"] == "kaggle"
    assert len(ld["images"]) == 2


def test_loader_handles_bare_folder_and_images_subdir(tmp_path):
    bare = _bare_folder(tmp_path, n=3)
    ld = ri.load_input_images(bare)
    assert ld["kind"] == "folder" and len(ld["images"]) == 3
    assert all(e.get("hash") and e.get("width") for e in ld["images"])

    # dir with images/ subfolder but no manifest
    nested = tmp_path / "nested"
    (nested / "images").mkdir(parents=True)
    imwrite(str(nested / "images" / "a.png"), np.full((70, 70, 3), 120, np.uint8))
    ld2 = ri.load_input_images(str(nested))
    assert ld2["kind"] == "folder" and len(ld2["images"]) == 1
    assert ld2["images"][0]["image"] == "images/a.png"


# --------------------------- claude-label on real dataset (the bug) ---------------------------


def test_create_real_dataset_is_claude_labelable(tmp_path):
    real = _real_dataset(tmp_path, n=5)
    out = str(tmp_path / "labeled")
    res = cl.claude_label_towel_folder(real, out, responder=_responder, log=_silent)
    assert res["status"] == "ok" and res["num_images"] == 5      # was 0 before the fix
    man = load_manifest(out)
    for s in man["samples"]:
        label = load_label(out, s)
        assert s["source"] == "user_photo"                       # provenance preserved
        assert label["pseudolabel"] is True
        assert label["usable_for_training"] is False             # NOT auto-approved
        assert label["approval_status"] in ("pending", "review_needed")


def test_bare_folder_is_claude_labelable(tmp_path):
    bare = _bare_folder(tmp_path, n=3)
    out = str(tmp_path / "labeled")
    res = cl.claude_label_towel_folder(bare, out, responder=_responder, log=_silent)
    assert res["status"] == "ok" and res["num_images"] == 3
    assert load_manifest(out)["samples"][0]["source"] == "external"


def test_claude_label_empty_dir_graceful(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    res = cl.claude_label_towel_folder(str(empty), str(tmp_path / "o"),
                                       responder=_responder, log=_silent)
    assert res["status"] == "error" and "no images" in res["message"]


# --------------------------- filter --mode all / --trust-source ---------------------------


def test_filter_mode_all_keeps_numbered_filenames(tmp_path):
    real = _real_dataset(tmp_path, n=5)   # images named 000001.png ... (no "towel" in name)
    out = str(tmp_path / "cand_all")
    res = ft.filter_towel_images(real, out, mode="all", max_images=105, log=_silent)
    assert res["status"] == "ok" and res["kept"] == 5
    assert set(res["reasons"]) == {"trusted_source"}
    files = sorted(os.listdir(os.path.join(out, "images")))
    assert files[0] == "000001.png" and len(files) == 5
    # provenance preserved into the candidate manifest
    assert load_manifest(out)["source"] == "user_photo" or \
        ri.load_raw_manifest(out)["source"] == "user_photo"


def test_filter_filename_mode_drops_non_keyword(tmp_path):
    # The original bug also hid this: real photos aren't named "*towel*", so the
    # keyword filter keeps none — which is exactly why --mode all exists.
    real = _real_dataset(tmp_path, n=5)
    res = ft.filter_towel_images(real, str(tmp_path / "c"), mode="filename", log=_silent)
    assert res["status"] == "ok" and res["kept"] == 0


def test_filter_mode_all_on_bare_folder(tmp_path):
    bare = _bare_folder(tmp_path, n=3)
    res = ft.filter_towel_images(bare, str(tmp_path / "c"), mode="all", log=_silent)
    assert res["kept"] == 3


def test_trust_source_cli_equals_mode_all(tmp_path):
    from typer.testing import CliRunner

    from terafold.cli import app

    real = _real_dataset(tmp_path, n=4)
    runner = CliRunner()
    out_a = str(tmp_path / "cand_mode_all")
    out_t = str(tmp_path / "cand_trust")
    r1 = runner.invoke(app, ["filter-towel-images", "--input", real, "--out", out_a, "--mode", "all"])
    r2 = runner.invoke(app, ["filter-towel-images", "--input", real, "--out", out_t, "--trust-source"])
    assert r1.exit_code == 0 and r2.exit_code == 0, (r1.output, r2.output)
    man_a = ri.load_raw_manifest(out_a)
    man_t = ri.load_raw_manifest(out_t)
    assert man_a["num_images"] == man_t["num_images"] == 4
    assert man_a["filter_mode"] == man_t["filter_mode"] == "all"


def test_filter_unknown_mode_errors(tmp_path):
    real = _real_dataset(tmp_path, n=2)
    res = ft.filter_towel_images(real, str(tmp_path / "c"), mode="bogus", log=_silent)
    assert res["status"] == "error" and "all" in res["modes"]
