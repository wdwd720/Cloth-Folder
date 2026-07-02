"""Tests for legal/controlled external towel image import (web_import).

NO network and NO optional deps: folder import uses PNGs written by the
pure-Python codec; the huggingface path is exercised via its graceful
manual-instructions branch (no `datasets` / no `hf_repo`).
"""

from __future__ import annotations

import numpy as np

from terafold.towel import dataset as ds
from terafold.towel import schema as sc
from terafold.towel import web_import as wi
from terafold.vision.imageio import imwrite


def _silent(*_a) -> None:
    pass


def _png(path, w=64, h=48, val=180):
    img = np.full((h, w, 3), 20, np.uint8)
    img[8:h - 8, 8:w - 8] = val
    imwrite(path, img)
    return path


def _folder(tmp_path, n=3):
    d = tmp_path / "export"
    d.mkdir()
    for i in range(n):
        _png(str(d / f"img_{i}.png"), val=70 + i * 30)
    return str(d)


# --------------------------- folder / local export ---------------------------


def test_folder_import_creates_external_unlabeled(tmp_path):
    src = _folder(tmp_path, n=3)
    out = str(tmp_path / "towel_external_v0")
    res = wi.import_towel_web_dataset(
        "folder", out, src_dir=src, license_note="CC-BY test", log=_silent
    )
    assert res["status"] == "ok"
    assert res["num_images"] == 3
    assert res["source"] == "folder"
    assert res["kind"] == "external"

    man = ds.load_manifest(out)
    assert man["kind"] == "external"
    assert man["source"] == "external"
    assert man["license_note"] == "CC-BY test"
    assert len(man["samples"]) == 3

    # Imported images are valid label templates but UNLABELED (no 4 corners yet).
    for _s, lab in ds.iter_samples(out):
        assert sc.validate_label(lab) == []
        assert lab["source"] == "external"
        assert sc.is_usable(lab) is False
    assert list(ds.iter_usable(out)) == []


def test_roboflow_alias_uses_external_kind(tmp_path):
    src = _folder(tmp_path, n=2)
    out = str(tmp_path / "rf")
    res = wi.import_towel_web_dataset("roboflow", out, src_dir=src, log=_silent)
    assert res["status"] == "ok" and res["num_images"] == 2
    assert ds.load_manifest(out)["kind"] == "external"


def test_nested_export_is_flattened(tmp_path):
    # Roboflow/Kaggle-style nested export with no top-level images.
    src = tmp_path / "nested"
    (src / "train" / "images").mkdir(parents=True)
    (src / "valid" / "images").mkdir(parents=True)
    _png(str(src / "train" / "images" / "a.png"), val=90)
    _png(str(src / "valid" / "images" / "b.png"), val=150)
    out = str(tmp_path / "flat")
    res = wi.import_towel_web_dataset("kaggle", out, src_dir=str(src), log=_silent)
    assert res["status"] == "ok" and res["num_images"] == 2
    assert ds.load_manifest(out)["kind"] == "external"


def test_folder_without_src_dir_returns_manual(tmp_path):
    res = wi.import_towel_web_dataset("kaggle", str(tmp_path / "o"), log=_silent)
    assert res["status"] == "manual_instructions"
    assert res["instructions"]


def test_folder_missing_src_dir_errors(tmp_path):
    res = wi.import_towel_web_dataset(
        "folder", str(tmp_path / "o"), src_dir=str(tmp_path / "nope"), log=_silent
    )
    assert res["status"] == "error"


# --------------------------- open_images ---------------------------


def test_open_images_manual_instructions(tmp_path):
    res = wi.import_towel_web_dataset("open_images", str(tmp_path / "oi"), log=_silent)
    assert res["status"] == "manual_instructions"
    assert res["instructions"]
    assert any("Towel" in line for line in res["instructions"])


# --------------------------- huggingface (graceful) ---------------------------


def test_huggingface_without_repo_returns_manual(tmp_path):
    # No hf_repo (and likely no `datasets`) -> manual_instructions + install hint.
    res = wi.import_towel_web_dataset("huggingface", str(tmp_path / "hf"), log=_silent)
    assert res["status"] == "manual_instructions"
    assert "install" in res and "datasets" in res["install"]
    assert res["instructions"]


# --------------------------- unknown source ---------------------------


def test_unknown_source_errors(tmp_path):
    res = wi.import_towel_web_dataset("bing", str(tmp_path / "x"), log=_silent)
    assert res["status"] == "error"
    assert "huggingface" in res["supported"]
