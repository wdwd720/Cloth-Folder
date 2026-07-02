"""Legal/controlled import of *external* towel images into a TeraFold dataset.

This module NEVER scrapes random web pages. It only ingests images the user has
already obtained legitimately:

* ``folder`` / ``roboflow`` / ``kaggle`` — a local export folder the user
  downloaded (Roboflow/Kaggle exports are accepted as-is; nested ``train/valid``
  layouts are flattened automatically).
* ``huggingface`` — streamed from a named HF dataset repo (requires the optional
  ``datasets`` package and a ``--hf-repo``).
* ``open_images`` — automatic download is intentionally NOT implemented; the
  importer returns the exact manual steps to fetch the Open Images *Towel* class
  with the official downloader / FiftyOne, then re-import as a local ``folder``.

Every imported dataset is tagged ``kind='external'`` with provenance + a
``license_note`` recorded in its manifest. Imported images are **unlabeled** —
they still need our 4-corner towel labels before they can feed YOLO pose
training. Missing optional deps degrade gracefully (a status dict, never a
raise); the module imports with numpy + pyyaml only.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import tempfile
from typing import Any, Callable, Dict, Iterator, Optional

from terafold.towel.dataset import create_towel_real_dataset
from terafold.towel.schema import IMAGE_EXTS

__all__ = ["import_towel_web_dataset", "SUPPORTED_SOURCES", "LOCAL_EXPORT_SOURCES"]

# Sources whose ``src_dir`` is a local export folder the user already downloaded.
LOCAL_EXPORT_SOURCES = ("folder", "roboflow", "kaggle")
SUPPORTED_SOURCES = ("folder", "roboflow", "kaggle", "huggingface", "open_images")


def _have_datasets() -> bool:
    """True if the optional Hugging Face ``datasets`` package is importable."""
    try:
        return importlib.util.find_spec("datasets") is not None
    except Exception:
        return False


def _has_top_level_images(d: str) -> bool:
    try:
        names = os.listdir(d)
    except OSError:
        return False
    return any(os.path.splitext(n)[1].lower() in IMAGE_EXTS for n in names)


def _iter_image_files(root: str) -> Iterator[str]:
    for dirpath, _dirs, files in os.walk(root):
        for name in sorted(files):
            if os.path.splitext(name)[1].lower() in IMAGE_EXTS:
                yield os.path.join(dirpath, name)


def import_towel_web_dataset(
    source: str,
    out: str,
    query: str = "towel",
    max_images: int = 500,
    src_dir: Optional[str] = None,
    hf_repo: Optional[str] = None,
    license_note: str = "",
    log: Callable[[str], None] = print,
) -> Dict[str, Any]:
    """Import external towel images into ``out`` as a ``kind='external'`` dataset.

    Returns ``{status:'ok', out, num_images, source, ...}`` on success, a
    ``{status:'manual_instructions', ...}`` dict when the user must take a manual
    step (download an export, install ``datasets``, etc.), or
    ``{status:'error', ...}`` for an unknown source / bad input. Never raises on a
    missing optional dependency.
    """
    src = (source or "").strip().lower()
    if src in LOCAL_EXPORT_SOURCES:
        return _import_local_export(src, out, src_dir, license_note, log)
    if src == "huggingface":
        return _import_huggingface(out, hf_repo, max_images, license_note, log)
    if src == "open_images":
        return _open_images_instructions(query)
    return {
        "status": "error",
        "message": f"unknown source {source!r}",
        "supported": list(SUPPORTED_SOURCES),
    }


# --------------------------------------------------------------------------
# folder / roboflow / kaggle — local export folders
# --------------------------------------------------------------------------


def _import_local_export(
    source: str,
    out: str,
    src_dir: Optional[str],
    license_note: str,
    log: Callable[[str], None],
) -> Dict[str, Any]:
    if not src_dir:
        return {
            "status": "manual_instructions",
            "source": source,
            "instructions": [
                f"No --src-dir given. Download your {source} towel export to a local "
                "folder first (this importer never scrapes the web), then re-run:",
                f"  python3 -m terafold import-towel-web --source {source} "
                "--src-dir <path-to-your-downloaded-export> --out " + out,
                "The folder may contain images directly or in nested "
                "train/valid/test subfolders (both are accepted).",
            ],
        }
    if not os.path.isdir(src_dir):
        return {
            "status": "error",
            "source": source,
            "message": f"src_dir not found or not a directory: {src_dir}",
            "supported": list(SUPPORTED_SOURCES),
        }

    # Roboflow/Kaggle exports often nest images under split subfolders. If there
    # are no images at the top level, flatten everything into a staging dir.
    staging: Optional[str] = None
    images_dir = src_dir
    if not _has_top_level_images(src_dir):
        files = list(_iter_image_files(src_dir))
        if not files:
            return {
                "status": "error",
                "source": source,
                "message": f"no images ({', '.join(IMAGE_EXTS)}) found under {src_dir}",
            }
        staging = tempfile.mkdtemp(prefix="terafold_webimport_")
        for i, f in enumerate(files):
            ext = os.path.splitext(f)[1].lower() or ".png"
            shutil.copyfile(f, os.path.join(staging, f"{i:06d}{ext}"))
        images_dir = staging
        log(f"[web-import] flattened {len(files)} images from nested {source} export")

    note = license_note or f"{source} export (external, user-provided local download)"
    try:
        res = create_towel_real_dataset(
            images_dir, out, source="external", kind="external",
            license_note=note, log=log,
        )
    finally:
        if staging:
            shutil.rmtree(staging, ignore_errors=True)

    return {
        "status": "ok",
        "out": out,
        "num_images": res["num_images"],
        "skipped_duplicates": res.get("skipped_duplicates", 0),
        "source": source,
        "kind": "external",
        "license_note": note,
        "manifest": res.get("manifest"),
        "labeled": False,
        "needs_labeling": True,
    }


# --------------------------------------------------------------------------
# huggingface — streamed image dataset (optional `datasets` dep)
# --------------------------------------------------------------------------


def _import_huggingface(
    out: str,
    hf_repo: Optional[str],
    max_images: int,
    license_note: str,
    log: Callable[[str], None],
) -> Dict[str, Any]:
    if not _have_datasets() or not hf_repo:
        return {
            "status": "manual_instructions",
            "source": "huggingface",
            "install": "pip install datasets",
            "have_datasets": _have_datasets(),
            "hf_repo": hf_repo,
            "instructions": [
                "Streaming a Hugging Face image dataset needs the optional 'datasets' "
                "package AND a repo id.",
                "  1. pip install datasets",
                "  2. Pick a legally usable towel image dataset on "
                "https://huggingface.co/datasets",
                "  3. Re-run: python3 -m terafold import-towel-web --source huggingface "
                f"--hf-repo <owner/dataset> --out {out} [--max-images {max_images}]",
            ],
        }

    import datasets  # type: ignore  (lazy; only reached when importable)

    from terafold.data.hf_streaming import _extract_images, _pick_top_image
    from terafold.vision.imageio import imwrite

    staging = tempfile.mkdtemp(prefix="terafold_hfimport_")
    saved = 0
    decode_errors = 0
    try:
        try:
            stream = datasets.load_dataset(hf_repo, split="train", streaming=True)
        except Exception as exc:  # offline / bad repo / no 'train' split
            return {
                "status": "error",
                "source": "huggingface",
                "hf_repo": hf_repo,
                "install": "pip install datasets",
                "message": f"could not open HF dataset {hf_repo!r}: {exc}",
            }

        for row in stream:
            if saved >= max_images:
                break
            try:
                arr = _pick_top_image(_extract_images(row))
            except Exception:
                arr = None
            if arr is None:
                decode_errors += 1
                continue
            try:
                imwrite(os.path.join(staging, f"{saved:06d}.png"), arr)
                saved += 1
            except Exception:
                decode_errors += 1

        if saved == 0:
            return {
                "status": "error",
                "source": "huggingface",
                "hf_repo": hf_repo,
                "num_images": 0,
                "message": f"no decodable images found while streaming {hf_repo!r}",
            }

        note = license_note or f"huggingface:{hf_repo} (external, streamed)"
        res = create_towel_real_dataset(
            staging, out, source="external", kind="external",
            license_note=note, log=log,
        )
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    return {
        "status": "ok",
        "out": out,
        "num_images": res["num_images"],
        "skipped_duplicates": res.get("skipped_duplicates", 0),
        "source": "huggingface",
        "hf_repo": hf_repo,
        "kind": "external",
        "license_note": note,
        "manifest": res.get("manifest"),
        "decode_errors": decode_errors,
        "labeled": False,
        "needs_labeling": True,
    }


# --------------------------------------------------------------------------
# open_images — manual download only (no network access here)
# --------------------------------------------------------------------------


def _open_images_instructions(query: str) -> Dict[str, Any]:
    return {
        "status": "manual_instructions",
        "source": "open_images",
        "class_name": "Towel",
        "query": query,
        "instructions": [
            "Automatic Open Images download is intentionally NOT implemented "
            "(this importer never accesses the network).",
            "Download the Open Images 'Towel' class manually, then re-import it as a "
            "local folder:",
            "",
            "Option A - FiftyOne (recommended):",
            "  pip install fiftyone",
            "  python -c \"import fiftyone.zoo as foz; "
            "foz.load_zoo_dataset('open-images-v7', split='train', "
            "label_types=['detections'], classes=['Towel'], max_samples=500, "
            "dataset_dir='open_images_towel')\"",
            "  # images land under open_images_towel/data/",
            "",
            "Option B - official downloader:",
            "  See https://storage.googleapis.com/openimages/web/download.html ; "
            "filter the 'Towel' class and download its images to a local folder.",
            "",
            "Then re-run this importer pointing at the downloaded images:",
            "  python3 -m terafold import-towel-web --source folder "
            "--src-dir <downloaded-open-images-towel-folder>",
        ],
    }
