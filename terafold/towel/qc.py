"""Quality control for generated (or any raw) towel image datasets.

Generated images are not automatically good. This QC pass runs *before* training
eligibility and drops the obvious junk so we don't waste Claude tokens or pollute
the trainset:

* file validity / corrupt-image removal (can't open or size it ⇒ drop),
* image dimension check (too small ⇒ drop),
* exact-duplicate removal (content SHA-1),
* near-duplicate removal (perceptual average-hash, Hamming distance),
* metadata-completeness check (generation lineage present).

Kept images are *copied* into a candidate dataset using the raw-manifest schema
(so ``claude-label-towel-folder`` can consume it directly), with generation
lineage preserved per image. Originals are never modified. Rejections are recorded
with reasons. Perceptual hashing needs an image backend (cv2/Pillow or the
built-in PNG codec); if an image can't be decoded it's treated as corrupt.
"""

from __future__ import annotations

import os
import shutil
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from terafold.data.episode_schema import read_json, write_json
from terafold.towel.raw_ingest import RAW_MANIFEST_KIND, raw_manifest_paths
from terafold.towel.schema import image_hash, image_size

__all__ = ["DEFAULT_MIN_SIZE", "DEFAULT_NEAR_DUP_HAMMING", "GENERATION_META_FIELDS",
           "average_hash", "hamming", "qc_filter_generated_images"]

DEFAULT_MIN_SIZE = 64
DEFAULT_NEAR_DUP_HAMMING = 5  # <= this many differing bits ⇒ "near duplicate"

# Generation lineage we expect on a generated sample (for completeness scoring).
GENERATION_META_FIELDS = ("generation_model", "generation_prompt",
                          "prompt_template_name", "category_target")


def _noop(_m: str) -> None:
    pass


def _resize_mean(gray: np.ndarray, side: int) -> np.ndarray:
    """Downsample a 2-D array to ``side x side`` by block-averaging (any input size)."""
    h, w = gray.shape
    ys = np.linspace(0, h, side + 1).astype(int)
    xs = np.linspace(0, w, side + 1).astype(int)
    out = np.zeros((side, side), dtype=np.float64)
    for i in range(side):
        y0, y1 = ys[i], max(ys[i] + 1, ys[i + 1])
        for j in range(side):
            x0, x1 = xs[j], max(xs[j] + 1, xs[j + 1])
            out[i, j] = gray[y0:y1, x0:x1].mean()
    return out


def average_hash(path: str, side: int = 8) -> Optional[int]:
    """64-bit perceptual average-hash of an image, or None if it can't be decoded."""
    try:
        from terafold.vision.imageio import imread

        rgb = np.asarray(imread(path)).astype(np.float64)
    except Exception:
        return None
    if rgb.ndim == 3:
        gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    else:
        gray = rgb
    small = _resize_mean(gray, side)
    bits = (small > small.mean()).flatten()
    h = 0
    for i, b in enumerate(bits):
        if b:
            h |= (1 << i)
    return h


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def qc_filter_generated_images(
    input_dir: str,
    out: str,
    min_size: int = DEFAULT_MIN_SIZE,
    max_images: Optional[int] = None,
    near_dup_hamming: int = DEFAULT_NEAR_DUP_HAMMING,
    dedup_perceptual: bool = True,
    require_metadata: bool = False,
    log: Callable[[str], None] = _noop,
) -> Dict[str, Any]:
    """Run QC on a generated/raw dataset at ``input_dir`` → candidate dataset at ``out``.

    Returns a summary dict. ``require_metadata=True`` rejects samples missing
    generation lineage; otherwise incompleteness is only counted.
    """
    man_path = os.path.join(input_dir, "manifest.json")
    if not os.path.exists(man_path):
        return {"status": "error", "message": f"no manifest in {input_dir}"}
    src = read_json(man_path)
    images = src.get("images", [])

    paths = raw_manifest_paths(out)
    os.makedirs(paths["images"], exist_ok=True)

    kept: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    by_reason: Dict[str, int] = {}
    seen_hashes: Dict[str, str] = {}
    kept_ahashes: List[int] = []
    incomplete_meta = 0
    next_id = 1

    def _reject(im, reason):
        rejected.append({**im, "qc_reason": reason})
        by_reason[reason] = by_reason.get(reason, 0) + 1

    for im in images:
        if max_images is not None and len(kept) >= max_images:
            break
        rel = im.get("image")
        src_path = os.path.join(input_dir, rel) if rel else None

        # 1) file validity
        if not src_path or not os.path.exists(src_path):
            _reject(im, "missing_file")
            continue

        # 2) corrupt / undecodable (size unknown ⇒ can't trust the file)
        size = image_size(src_path)
        if size is None:
            _reject(im, "corrupt_or_unreadable")
            continue
        w, h = size

        # 3) dimension check
        if min(w, h) < int(min_size):
            _reject(im, "too_small")
            continue

        # 4) metadata completeness
        missing_meta = [k for k in GENERATION_META_FIELDS if not im.get(k)]
        if missing_meta:
            incomplete_meta += 1
            if require_metadata:
                _reject(im, "incomplete_metadata")
                continue

        # 5) exact-duplicate (content hash)
        chash = im.get("hash") or image_hash(src_path)
        if chash in seen_hashes:
            _reject(im, "exact_duplicate")
            continue

        # 6) near-duplicate (perceptual average-hash)
        if dedup_perceptual:
            ah = average_hash(src_path)
            if ah is None:
                _reject(im, "corrupt_or_unreadable")
                continue
            if any(hamming(ah, prev) <= int(near_dup_hamming) for prev in kept_ahashes):
                _reject(im, "near_duplicate")
                continue
            kept_ahashes.append(ah)

        # keep it
        sid = f"{next_id:06d}"
        ext = os.path.splitext(rel)[1].lower() or ".png"
        img_rel = f"images/{sid}{ext}"
        try:
            shutil.copyfile(src_path, os.path.join(out, img_rel))
        except OSError:
            _reject(im, "copy_failed")
            continue
        seen_hashes[chash] = sid
        entry = {
            "id": sid, "image": img_rel, "original": im.get("original", src_path),
            "hash": chash, "width": w, "height": h,
            "source": im.get("source", src.get("source", "openai_generated")),
            "qc_passed": True,
        }
        for k in (*GENERATION_META_FIELDS, "target_state", "cost_estimate", "license"):
            if im.get(k) is not None:
                entry[k] = im[k]
        kept.append(entry)
        by_reason["kept"] = by_reason.get("kept", 0) + 1
        next_id += 1

    manifest = {
        "dataset": os.path.basename(os.path.normpath(out)),
        "kind": RAW_MANIFEST_KIND,
        "qc_of": os.path.basename(os.path.normpath(input_dir)),
        "source": src.get("source", "openai_generated"),
        "generation_model": src.get("generation_model"),
        "min_size": int(min_size),
        "near_dup_hamming": int(near_dup_hamming),
        "dedup_perceptual": bool(dedup_perceptual),
        "num_input": len(images),
        "num_images": len(kept),
        "num_rejected": len(rejected),
        "incomplete_metadata": incomplete_meta,
        "reasons": by_reason,
        "labeled": False,
        "needs_labeling": True,
        "images": kept,
        "rejected": rejected,
    }
    write_json(paths["manifest"], manifest)
    log(f"QC kept {len(kept)} / {len(images)} generated images -> {out} "
        f"[{len(rejected)} rejected: {by_reason}]")
    return {
        "status": "ok", "out": out, "num_input": len(images), "kept": len(kept),
        "rejected": len(rejected), "reasons": by_reason,
        "incomplete_metadata": incomplete_meta, "manifest": paths["manifest"],
    }
