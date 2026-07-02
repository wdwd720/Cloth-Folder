"""Ingest *external* real towel images into a RAW image dataset.

The external-image pipeline starts here. A "raw" dataset is deliberately simpler
than a labeled towel dataset — it is just deduplicated images + provenance, with
NO labels yet::

    data/towel_kaggle_raw_v0/
        images/000001.jpg ...
        manifest.json            # raw image manifest (source, slug, hash, w/h, license)

Later stages turn it into candidates (:mod:`terafold.towel.filtering`) and then
pseudo-labels (:mod:`terafold.towel.claude_pseudolabel`).

Two real-image sources are supported here:

* **Kaggle** — via ``kagglehub`` or the ``kaggle`` CLI if installed. When neither
  is available (or no credentials), the importer fails *gracefully* with the exact
  setup commands — it never raises and never touches the network implicitly.
* **Open Images** — automatic bulk download is intentionally NOT implemented;
  instead a clear manual path (FiftyOne / official downloader) is printed, and the
  importer ingests an already-downloaded folder. Open Images bboxes do not give
  towel *corners*, so Claude labeling + human review is still required.

Everything is stdlib + numpy-free at import time; ``kagglehub``/``kaggle`` are
lazy, optional, and probed (never imported eagerly).
"""

from __future__ import annotations

import importlib.util
import os
import shutil
from typing import Any, Callable, Dict, Iterator, List, Optional

from terafold.data.episode_schema import write_json
from terafold.towel.schema import IMAGE_EXTS, image_hash, image_size

__all__ = [
    "RAW_MANIFEST_KIND", "DEFAULT_MIN_SIZE", "ingest_raw_folder", "load_raw_manifest",
    "validate_raw_manifest", "import_kaggle_towel_dataset", "import_openimages_towels",
    "KAGGLE_SETUP", "have_kagglehub", "have_kaggle_cli", "load_input_images",
]

# Provenance keys carried through every stage so generated/real lineage survives.
_PROVENANCE_KEYS = ("generation_model", "generation_prompt", "prompt_template_name",
                    "category_target", "target_state", "training_track", "cost_estimate",
                    "license")

RAW_MANIFEST_KIND = "raw_images"
DEFAULT_MIN_SIZE = 64  # px: skip images whose smaller side is below this

KAGGLE_SETUP = [
    "Kaggle access is not configured. Install a client and add credentials:",
    "  pip install kaggle kagglehub",
    "  # then place kaggle.json at ~/.kaggle/kaggle.json (chmod 600), OR set:",
    "  export KAGGLE_USERNAME=...   export KAGGLE_KEY=...",
    "  # get the token from https://www.kaggle.com/settings -> 'Create New API Token'",
]


def _noop(_m: str) -> None:
    pass


# --------------------------------------------------------------------------
# Raw image manifest
# --------------------------------------------------------------------------


def raw_manifest_paths(root: str) -> Dict[str, str]:
    return {
        "root": root,
        "images": os.path.join(root, "images"),
        "manifest": os.path.join(root, "manifest.json"),
    }


def load_raw_manifest(root: str) -> Dict[str, Any]:
    from terafold.data.episode_schema import read_json

    return read_json(raw_manifest_paths(root)["manifest"])


def validate_raw_manifest(manifest: Dict[str, Any]) -> List[str]:
    """Return a list of problems with a raw image manifest (empty ⇒ valid)."""
    problems: List[str] = []
    if not isinstance(manifest, dict):
        return ["manifest is not a JSON object"]
    if manifest.get("kind") != RAW_MANIFEST_KIND:
        problems.append(f"kind must be {RAW_MANIFEST_KIND!r}")
    if not manifest.get("source"):
        problems.append("missing 'source'")
    imgs = manifest.get("images")
    if not isinstance(imgs, list):
        return problems + ["'images' must be a list"]
    for i, im in enumerate(imgs):
        if not isinstance(im, dict):
            problems.append(f"images[{i}] is not an object")
            continue
        for key in ("id", "image", "original", "hash", "source"):
            if not im.get(key):
                problems.append(f"images[{i}] missing {key!r}")
        for key in ("width", "height"):
            if key not in im:
                problems.append(f"images[{i}] missing {key!r}")
    return problems


def _iter_image_files(root: str) -> Iterator[str]:
    for dirpath, _dirs, files in os.walk(root):
        for name in sorted(files):
            if os.path.splitext(name)[1].lower() in IMAGE_EXTS:
                yield os.path.join(dirpath, name)


def _has_images(d: str) -> bool:
    try:
        return any(os.path.splitext(n)[1].lower() in IMAGE_EXTS for n in os.listdir(d))
    except OSError:
        return False


def load_input_images(input_dir: str, default_source: str = "external") -> Dict[str, Any]:
    """Normalize ANY supported input shape into a common image list.

    Accepts (and auto-detects) all of:

    * a RAW / candidate manifest (``kind: raw_images`` with an ``images`` list),
    * a towel dataset manifest from ``create-towel-real-dataset`` / merge / pseudo-label
      (``kind: real|external|synthetic|merged|pseudolabeled`` with a ``samples`` list),
    * a plain folder of images,
    * a dataset directory whose images live in an ``images/`` subfolder (no manifest).

    Returns ``{"base_dir", "kind", "source", "images": [...]}`` where every entry has
    at least ``image`` (path relative to ``base_dir``), ``original``, ``hash``,
    ``width``, ``height`` and ``source`` — so the filter / Claude-label stages can
    treat real-user datasets, generated datasets, and bare folders uniformly. Image
    provenance is preserved (``real_user`` etc.). Never raises on a bad/missing
    manifest — it falls back to scanning for image files.
    """
    from terafold.data.episode_schema import read_json

    base = input_dir
    man_path = os.path.join(input_dir, "manifest.json")
    man: Optional[Dict[str, Any]] = None
    if os.path.exists(man_path):
        try:
            loaded = read_json(man_path)
            man = loaded if isinstance(loaded, dict) else None
        except Exception:
            man = None

    if man is not None and isinstance(man.get("images"), list):
        # Already a raw / candidate manifest — pass entries through, fill source.
        src = man.get("source") or default_source
        images = [{**e, "source": e.get("source") or src} for e in man["images"]]
        return {"base_dir": base, "kind": man.get("kind", RAW_MANIFEST_KIND),
                "source": src, "images": images}

    if man is not None and isinstance(man.get("samples"), list):
        # A towel dataset (create-towel-real-dataset / merge / pseudolabeled).
        src = man.get("source") or default_source
        images = []
        for s in man["samples"]:
            entry = {
                "id": s.get("id"),
                "image": s.get("image"),
                "original": s.get("original", ""),
                "hash": s.get("hash"),
                "width": s.get("width"),
                "height": s.get("height"),
                "source": s.get("source") or src,
            }
            for k in _PROVENANCE_KEYS:
                if s.get(k) is not None:
                    entry[k] = s[k]
            images.append(entry)
        return {"base_dir": base, "kind": man.get("kind", "towel"),
                "source": src, "images": images}

    # No usable manifest — scan for image files (images/ subfolder preferred).
    images_sub = os.path.join(input_dir, "images")
    if os.path.isdir(images_sub) and _has_images(images_sub):
        scan_dir, prefix = images_sub, "images"
    else:
        scan_dir, prefix = input_dir, ""
    images = []
    if os.path.isdir(scan_dir):
        for name in sorted(os.listdir(scan_dir)):
            if os.path.splitext(name)[1].lower() not in IMAGE_EXTS:
                continue
            rel = f"{prefix}/{name}" if prefix else name
            abspath = os.path.join(input_dir, rel)
            try:
                h = image_hash(abspath)
            except OSError:
                continue
            size = image_size(abspath)
            w, hgt = size if size else (None, None)
            images.append({"image": rel, "original": os.path.abspath(abspath),
                           "hash": h, "width": w, "height": hgt, "source": default_source})
    return {"base_dir": base, "kind": "folder", "source": default_source, "images": images}


def ingest_raw_folder(
    src_dir: str,
    out: str,
    source: str,
    dataset_slug: Optional[str] = None,
    license_note: str = "",
    max_images: Optional[int] = None,
    min_size: int = DEFAULT_MIN_SIZE,
    kind_note: str = "",
    log: Callable[[str], None] = _noop,
) -> Dict[str, Any]:
    """Copy images from ``src_dir`` (recursively) into a RAW dataset at ``out``.

    Recursively finds JPG/JPEG/PNG/WebP/BMP, skips tiny images, deduplicates by
    content hash, records width/height + provenance, writes ``manifest.json``.
    Originals are copied (bytes, never decoded) — the source folder is untouched.
    """
    if not os.path.isdir(src_dir):
        return {"status": "error", "message": f"src_dir not found: {src_dir}", "source": source}

    paths = raw_manifest_paths(out)
    os.makedirs(paths["images"], exist_ok=True)

    images: List[Dict[str, Any]] = []
    seen: Dict[str, str] = {}
    skipped_small = skipped_dupes = skipped_unreadable = 0
    next_id = 1

    for src in _iter_image_files(src_dir):
        if max_images is not None and len(images) >= max_images:
            break
        try:
            h = image_hash(src)
        except OSError:
            skipped_unreadable += 1
            log(f"[warn] unreadable, skipping: {src}")
            continue
        if h in seen:
            skipped_dupes += 1
            continue
        size = image_size(src)
        if size is not None:
            w, hgt = size
            if min(w, hgt) < int(min_size):
                skipped_small += 1
                continue
        else:
            w = hgt = None  # unknown size: keep it (don't penalize odd formats)

        sid = f"{next_id:06d}"
        ext = os.path.splitext(src)[1].lower() or ".jpg"
        img_rel = f"images/{sid}{ext}"
        try:
            shutil.copyfile(src, os.path.join(out, img_rel))
        except OSError:
            skipped_unreadable += 1
            log(f"[warn] copy failed, skipping: {src}")
            continue
        seen[h] = sid
        images.append({
            "id": sid,
            "image": img_rel,
            "original": os.path.abspath(src),
            "hash": h,
            "width": w,
            "height": hgt,
            "source": source,
            "license": license_note,
        })
        next_id += 1

    manifest = {
        "dataset": os.path.basename(os.path.normpath(out)),
        "kind": RAW_MANIFEST_KIND,
        "source": source,
        "dataset_slug": dataset_slug,
        "license_note": license_note,
        "provenance_note": kind_note,
        "min_size": int(min_size),
        "num_images": len(images),
        "skipped_small": skipped_small,
        "skipped_duplicates": skipped_dupes,
        "skipped_unreadable": skipped_unreadable,
        "labeled": False,
        "needs_labeling": True,
        "images": images,
    }
    write_json(paths["manifest"], manifest)
    log(f"Ingested {len(images)} raw {source} images -> {out} "
        f"({skipped_dupes} dupes, {skipped_small} tiny, {skipped_unreadable} bad skipped)")
    return {
        "status": "ok",
        "out": out,
        "source": source,
        "dataset_slug": dataset_slug,
        "num_images": len(images),
        "skipped_small": skipped_small,
        "skipped_duplicates": skipped_dupes,
        "skipped_unreadable": skipped_unreadable,
        "manifest": paths["manifest"],
        "license_note": license_note,
        "labeled": False,
        "needs_labeling": True,
    }


# --------------------------------------------------------------------------
# Kaggle
# --------------------------------------------------------------------------


def have_kagglehub() -> bool:
    try:
        return importlib.util.find_spec("kagglehub") is not None
    except Exception:
        return False


def have_kaggle_cli() -> bool:
    # The `kaggle` package ships a CLI *and* an importable module. Either is fine.
    if shutil.which("kaggle"):
        return True
    try:
        return importlib.util.find_spec("kaggle") is not None
    except Exception:
        return False


def _download_kaggle(dataset: str, dest: str, log: Callable[[str], None]) -> Optional[str]:
    """Download a Kaggle dataset to a local folder. Returns the folder or None.

    Tries ``kagglehub`` first (no manual unzip), then the ``kaggle`` CLI. Any
    failure (no creds / network / bad slug) is swallowed and reported as None so
    the caller can degrade to setup instructions.
    """
    if have_kagglehub():
        try:
            import kagglehub  # type: ignore

            log(f"[kaggle] downloading {dataset} via kagglehub ...")
            path = kagglehub.dataset_download(dataset)
            if path and os.path.isdir(path):
                return path
        except Exception as exc:  # pragma: no cover - network/creds dependent
            log(f"[kaggle] kagglehub failed: {exc}")

    if have_kaggle_cli():
        os.makedirs(dest, exist_ok=True)
        try:
            import subprocess

            log(f"[kaggle] downloading {dataset} via kaggle CLI ...")
            subprocess.run(
                ["kaggle", "datasets", "download", "-d", dataset, "-p", dest, "--unzip"],
                check=True, capture_output=True, text=True,
            )
            return dest
        except Exception as exc:  # pragma: no cover - network/creds dependent
            log(f"[kaggle] kaggle CLI failed: {exc}")
    return None


def import_kaggle_towel_dataset(
    dataset: str,
    out: str,
    max_images: int = 2000,
    min_size: int = DEFAULT_MIN_SIZE,
    license_note: str = "",
    src_dir: Optional[str] = None,
    download_fn: Optional[Callable[[str, str, Callable[[str], None]], Optional[str]]] = None,
    log: Callable[[str], None] = _noop,
) -> Dict[str, Any]:
    """Import a Kaggle towel dataset into a RAW image dataset at ``out``.

    Resolution order for the source images:
      1. ``src_dir`` — an already-downloaded Kaggle export folder (offline path),
      2. ``download_fn`` — an injected downloader (test seam),
      3. ``kagglehub`` / ``kaggle`` CLI if installed and credentialed.

    When no client is available it returns ``status='unavailable'`` with exact
    setup instructions (``KAGGLE_SETUP``) — it never raises and never prints keys.
    """
    note = license_note or f"kaggle:{dataset} (see the dataset page for its license)"

    if src_dir:
        if not os.path.isdir(src_dir):
            return {"status": "error", "source": "kaggle",
                    "message": f"src_dir not found: {src_dir}"}
        folder = src_dir
    else:
        downloader = download_fn or _download_kaggle
        if download_fn is None and not (have_kagglehub() or have_kaggle_cli()):
            return {
                "status": "unavailable",
                "source": "kaggle",
                "dataset_slug": dataset,
                "install": "pip install kaggle kagglehub",
                "instructions": list(KAGGLE_SETUP) + [
                    "",
                    "Already downloaded it? Point the importer at the local export:",
                    f"  python3 -m terafold import-kaggle-towel-dataset --dataset {dataset} "
                    f"--src-dir <local-export-folder> --out {out}",
                ],
            }
        folder = downloader(dataset, os.path.join(out, "_kaggle_download"), log)
        if not folder or not os.path.isdir(folder):
            return {
                "status": "error",
                "source": "kaggle",
                "dataset_slug": dataset,
                "message": f"could not download Kaggle dataset {dataset!r} "
                           "(check credentials / network / slug).",
                "instructions": list(KAGGLE_SETUP),
            }

    res = ingest_raw_folder(
        folder, out, source="kaggle", dataset_slug=dataset, license_note=note,
        max_images=max_images, min_size=min_size,
        kind_note="Kaggle dataset; verify the dataset's license before redistribution.",
        log=log,
    )
    # Clean up an internal download staging dir (never the user's --src-dir).
    if not src_dir:
        staging = os.path.join(out, "_kaggle_download")
        if os.path.isdir(staging):
            shutil.rmtree(staging, ignore_errors=True)
    return res


# --------------------------------------------------------------------------
# Open Images
# --------------------------------------------------------------------------


def _openimages_instructions(out: str, max_images: int) -> Dict[str, Any]:
    return {
        "status": "manual_instructions",
        "source": "openimages",
        "class_name": "Towel",
        "instructions": [
            "Automatic Open Images bulk download is intentionally NOT implemented here.",
            "Download the Open Images 'Towel' class once, then re-run with --src-dir.",
            "",
            "Option A — FiftyOne (recommended):",
            "  pip install fiftyone",
            "  python -c \"import fiftyone.zoo as foz; "
            "foz.load_zoo_dataset('open-images-v7', split='train', "
            "label_types=['detections'], classes=['Towel'], "
            f"max_samples={max_images}, dataset_dir='open_images_towel')\"",
            "  # images land under open_images_towel/data/",
            "",
            "Option B — official downloader:",
            "  https://storage.googleapis.com/openimages/web/download.html "
            "(filter the 'Towel' class).",
            "",
            "Then ingest the downloaded folder:",
            f"  python3 -m terafold import-openimages-towels --src-dir "
            f"<open_images_towel/data> --out {out} --max-images {max_images}",
            "",
            "NOTE: Open Images provides image-level labels / bounding boxes only — NOT "
            "towel corner keypoints. You still need Claude labeling + human review to get "
            "the 4 corners this pipeline trains on.",
        ],
        "license_note": "Open Images images are CC-BY (per-image); annotations are CC-BY 4.0.",
    }


def import_openimages_towels(
    out: str,
    max_images: int = 2000,
    min_size: int = DEFAULT_MIN_SIZE,
    src_dir: Optional[str] = None,
    license_note: str = "",
    log: Callable[[str], None] = _noop,
) -> Dict[str, Any]:
    """Import Open Images 'Towel' images into a RAW dataset.

    With ``--src-dir`` (a locally-downloaded Open Images Towel folder) it ingests
    using the shared raw-manifest schema. Without it, it returns the exact manual
    download steps and explains that Open Images gives boxes — not towel corners —
    so Claude labeling + human review is still required.
    """
    if not src_dir:
        return _openimages_instructions(out, max_images)
    if not os.path.isdir(src_dir):
        return {"status": "error", "source": "openimages",
                "message": f"src_dir not found: {src_dir}"}
    note = license_note or "Open Images v7 'Towel' (images CC-BY; annotations CC-BY 4.0)"
    return ingest_raw_folder(
        src_dir, out, source="openimages", dataset_slug="open-images-v7/Towel",
        license_note=note, max_images=max_images, min_size=min_size,
        kind_note="Open Images boxes are NOT towel corners — Claude+human labeling required.",
        log=log,
    )
