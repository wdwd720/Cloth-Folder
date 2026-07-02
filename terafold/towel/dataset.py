"""Create and read a real-image towel dataset (the default data path).

A dataset directory looks like::

    data/towel_real_v0/
        images/000001.jpg ...
        labels/000001.json ...      # one towel label per image (schema.new_label)
        manifest.json               # dataset descriptor + per-sample metadata
        README.md

The manifest is the index every later stage reads. Image *bytes are copied* (never
decoded), so this works with no OpenCV/Pillow installed and supports JPEG towels.
"""

from __future__ import annotations

import os
import shutil
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from terafold.data.episode_schema import read_json, write_json
from terafold.towel.schema import (IMAGE_EXTS, TOWEL_STATES, image_hash, image_size,
                                   is_usable, new_label)

__all__ = [
    "dataset_paths", "create_towel_real_dataset", "load_manifest", "save_manifest",
    "load_label", "save_label", "iter_samples", "iter_usable", "add_image",
    "recount_usable", "LABEL_SCHEMA",
]

LABEL_SCHEMA = "towel_v1"


def _noop(_m: str) -> None:
    pass


def dataset_paths(root: str) -> Dict[str, str]:
    return {
        "root": root,
        "images": os.path.join(root, "images"),
        "labels": os.path.join(root, "labels"),
        "manifest": os.path.join(root, "manifest.json"),
        "readme": os.path.join(root, "README.md"),
    }


def _list_images(images_dir: str) -> List[str]:
    out = []
    for name in sorted(os.listdir(images_dir)):
        if os.path.splitext(name)[1].lower() in IMAGE_EXTS:
            out.append(os.path.join(images_dir, name))
    return out


def load_manifest(root: str) -> Dict[str, Any]:
    return read_json(dataset_paths(root)["manifest"])


def save_manifest(root: str, manifest: Dict[str, Any]) -> str:
    p = dataset_paths(root)["manifest"]
    write_json(p, manifest)
    return p


def load_label(root: str, sample: Dict[str, Any]) -> Dict[str, Any]:
    return read_json(os.path.join(root, sample["label"]))


def save_label(root: str, sample: Dict[str, Any], label: Dict[str, Any]) -> str:
    p = os.path.join(root, sample["label"])
    write_json(p, label)
    return p


def iter_samples(root: str) -> Iterator[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """Yield ``(sample_meta, label_dict)`` for every sample in the manifest."""
    man = load_manifest(root)
    for s in man.get("samples", []):
        yield s, load_label(root, s)


def iter_usable(root: str) -> Iterator[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """Yield only samples whose label is usable for training (trainable + 4 corners)."""
    for s, lab in iter_samples(root):
        if is_usable(lab):
            yield s, lab


def add_image(
    root: str, src_path: str, sample_id: str, source: str = "user_photo",
    copy: bool = True, state: str = "flat_unfolded",
) -> Optional[Dict[str, Any]]:
    """Copy one image into a dataset and write its label template.

    Returns the manifest sample dict, or ``None`` if the file is unreadable.
    """
    paths = dataset_paths(root)
    os.makedirs(paths["images"], exist_ok=True)
    os.makedirs(paths["labels"], exist_ok=True)
    ext = os.path.splitext(src_path)[1].lower() or ".jpg"
    img_rel = os.path.join("images", f"{sample_id}{ext}")
    lab_rel = os.path.join("labels", f"{sample_id}.json")
    dst_img = os.path.join(root, img_rel)
    try:
        shutil.copyfile(src_path, dst_img)
    except OSError:
        return None
    size = image_size(dst_img)
    width, height = (size if size else (None, None))
    label = new_label(img_rel.replace(os.sep, "/"), source=source, state=state)
    write_json(os.path.join(root, lab_rel), label)
    return {
        "id": sample_id,
        "image": img_rel.replace(os.sep, "/"),
        "label": lab_rel.replace(os.sep, "/"),
        "original": os.path.abspath(src_path),
        "hash": image_hash(dst_img),
        "width": width,
        "height": height,
        "source": source,
    }


def create_towel_real_dataset(
    images_dir: str,
    out: str,
    source: str = "user_photo",
    copy: bool = True,
    log: Callable[[str], None] = _noop,
    kind: str = "real",
    license_note: str = "",
) -> Dict[str, Any]:
    """Build a towel dataset from a folder of real images.

    * copies/normalizes images into ``out/images`` (bytes copied — no decode),
    * writes an unlabeled label template per image,
    * de-duplicates by content hash within the input,
    * writes ``manifest.json`` + a ``README.md``.

    Labels are NOT required yet (``usable_for_training`` defaults False).
    """
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"images dir not found: {images_dir}")
    paths = dataset_paths(out)
    os.makedirs(paths["images"], exist_ok=True)
    os.makedirs(paths["labels"], exist_ok=True)

    files = _list_images(images_dir)
    samples: List[Dict[str, Any]] = []
    seen_hashes: Dict[str, str] = {}
    skipped_dupes = 0
    next_id = 1
    for src in files:
        try:
            h = image_hash(src)
        except OSError:
            log(f"[warn] unreadable, skipping: {src}")
            continue
        if h in seen_hashes:
            skipped_dupes += 1
            log(f"[dupe] {os.path.basename(src)} == {seen_hashes[h]} (skipped)")
            continue
        sid = f"{next_id:06d}"
        sample = add_image(out, src, sid, source=source, copy=copy)
        if sample is None:
            log(f"[warn] copy failed, skipping: {src}")
            continue
        seen_hashes[h] = os.path.basename(src)
        samples.append(sample)
        next_id += 1

    manifest = {
        "dataset": os.path.basename(os.path.normpath(out)),
        "kind": kind,                       # real | external | synthetic | merged
        "label_schema": LABEL_SCHEMA,
        "states": TOWEL_STATES,
        "source": source,
        "license_note": license_note,
        "num_images": len(samples),
        "num_input_files": len(files),
        "skipped_duplicates": skipped_dupes,
        "num_labeled_usable": 0,            # filled by labeling/merge later
        "samples": samples,
    }
    save_manifest(out, manifest)
    _write_readme(out, manifest)
    log(f"Created {kind} dataset '{manifest['dataset']}': {len(samples)} images "
        f"({skipped_dupes} duplicates skipped) -> {out}")
    return {"out": out, "num_images": len(samples), "skipped_duplicates": skipped_dupes,
            "manifest": paths["manifest"], "samples": samples}


def _write_readme(root: str, manifest: Dict[str, Any]) -> None:
    paths = dataset_paths(root)
    lines = [
        f"# Towel dataset — {manifest['dataset']}",
        "",
        f"- kind: **{manifest['kind']}**  (real images are the default training path)",
        f"- source: {manifest.get('source')}",
        f"- images: {manifest['num_images']}  (duplicates skipped: "
        f"{manifest.get('skipped_duplicates', 0)})",
        f"- label schema: {manifest['label_schema']}",
        "",
        "## States",
        ", ".join(manifest["states"]),
        "",
        "## Layout",
        "- `images/` — the towel images (bytes copied, not re-encoded).",
        "- `labels/` — one JSON per image (4 corners tl/tr/br/bl, state, etc.).",
        "- `manifest.json` — dataset index + per-sample metadata (hash, width, height, source).",
        "",
        "## Labeling",
        "Images start UNLABELED (`usable_for_training: false`). Add 4-corner labels:",
        "",
        "```bash",
        f"python3 -m terafold label-towel-folder --dataset {root}",
        "```",
        "",
        "Only `usable_for_training: true` samples with a trainable state "
        "(flat/wrinkled/partially_folded/folded_success) and 4 corners are exported "
        "to YOLO pose training. `bad_view` / `multiple_towels` / `not_towel` are excluded.",
    ]
    with open(paths["readme"], "w") as f:
        f.write("\n".join(lines) + "\n")


def recount_usable(root: str) -> int:
    """Recompute and persist ``num_labeled_usable`` in the manifest."""
    man = load_manifest(root)
    n = 0
    for s in man.get("samples", []):
        try:
            if is_usable(load_label(root, s)):
                n += 1
        except FileNotFoundError:
            continue
    man["num_labeled_usable"] = n
    save_manifest(root, man)
    return n
