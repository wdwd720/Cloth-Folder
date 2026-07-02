"""Filter a RAW image dataset down to likely-towel CANDIDATES.

Between bulk ingest (:mod:`terafold.towel.raw_ingest`) and the (paid) Claude
labeling pass, this stage cheaply throws away images that clearly are not single
flat towels, so we don't spend API tokens on junk.

Two modes:

* ``filename``        — keep images whose path/filename contains a towel keyword
  (``towel``, ``bath_towel``, ``hotel_towel``, ``white_towel``, ``linen``, ...).
* ``filename_or_vlm`` — filename first; if an image classifier / VLM responder is
  provided, ambiguous images are additionally screened by it (optional, injected).

It never deletes the originals: kept images are *copied* into a new candidate
dataset (same raw-manifest schema) with a ``filter_reason`` per image; rejected /
non-image files are recorded in the manifest but not copied.
"""

from __future__ import annotations

import os
import shutil
from typing import Any, Callable, Dict, List, Optional

from terafold.data.episode_schema import write_json
from terafold.towel.raw_ingest import (RAW_MANIFEST_KIND, _PROVENANCE_KEYS,
                                       load_input_images, raw_manifest_paths)
from terafold.towel.schema import image_hash, image_size

__all__ = ["DEFAULT_KEYWORDS", "FILTER_MODES", "filter_towel_images"]

# Path/filename substrings that strongly suggest a towel image.
DEFAULT_KEYWORDS = [
    "towel", "towels", "bath_towel", "bath-towel", "hand_towel", "hotel_towel",
    "white_towel", "beach_towel", "linen", "washcloth", "terry",
]

FILTER_MODES = ("filename", "filename_or_vlm", "all")
# Floor below which an image is "tiny" (only applied in trusted `all` mode when the
# caller did not pass an explicit --min-size).
_TINY_FLOOR = 16


def _noop(_m: str) -> None:
    pass


def _matches_keyword(path: str, keywords: List[str]) -> Optional[str]:
    low = path.replace("\\", "/").lower()
    for kw in keywords:
        if kw in low:
            return kw
    return None


def filter_towel_images(
    input_dir: str,
    out: str,
    mode: str = "filename_or_vlm",
    max_images: int = 1000,
    keywords: Optional[List[str]] = None,
    min_size: int = 0,
    vlm_responder: Optional[Callable[[str], bool]] = None,
    log: Callable[[str], None] = _noop,
) -> Dict[str, Any]:
    """Filter ANY supported input shape at ``input_dir`` into a candidate dataset.

    Args:
        input_dir: a raw/candidate manifest, a ``create-towel-real-dataset`` (or
            merged/pseudolabeled) dataset, a plain image folder, or a dir with an
            ``images/`` subfolder — all auto-detected.
        out: output candidate dataset directory.
        mode: ``filename`` | ``filename_or_vlm`` | ``all``. ``all`` keeps every valid
            image from a trusted source (only broken/tiny images are skipped).
        max_images: cap on kept candidates.
        keywords: override the towel keyword list.
        min_size: small-image cut (0 = default; ``all`` mode applies a 16 px tiny floor).
        vlm_responder: optional ``fn(image_path) -> bool``; consulted only in
            ``filename_or_vlm`` mode for images that fail the keyword test.
        log: logging sink.

    Provenance is preserved (``real_user`` for create-towel-real-dataset outputs,
    generation lineage for generated datasets). Originals are never modified.
    """
    mode = (mode or "filename_or_vlm").strip().lower()
    if mode not in FILTER_MODES:
        return {"status": "error", "message": f"unknown mode {mode!r}",
                "modes": list(FILTER_MODES)}
    if not os.path.isdir(input_dir):
        return {"status": "error", "message": f"input not found: {input_dir}"}

    loaded = load_input_images(input_dir, default_source="external")
    src_images = loaded["images"]
    if not src_images:
        return {"status": "error",
                "message": f"no images found in {input_dir} (manifest, images/, or folder)"}

    kw = keywords or DEFAULT_KEYWORDS
    paths = raw_manifest_paths(out)
    os.makedirs(paths["images"], exist_ok=True)
    eff_min = int(min_size) if min_size else (_TINY_FLOOR if mode == "all" else 0)

    kept: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    seen_hashes: Dict[str, str] = {}
    by_reason: Dict[str, int] = {}
    use_vlm = mode == "filename_or_vlm" and vlm_responder is not None
    next_id = 1

    def _reject(im, why):
        rejected.append({**im, "filter_reason": why})
        by_reason[why] = by_reason.get(why, 0) + 1

    for im in src_images:
        if len(kept) >= max_images:
            break
        rel = im.get("image")
        src_path = os.path.join(input_dir, rel) if rel else None
        original = im.get("original", "")

        # 1) obvious bad / missing file.
        if not src_path or not os.path.exists(src_path):
            _reject(im, "missing_file")
            continue

        # 2) validity + tiny check (broken => can't be sized).
        w, h = im.get("width"), im.get("height")
        if not (w and h):
            size = image_size(src_path)
            w, h = size if size else (None, None)
        if not (w and h):
            _reject(im, "corrupt_or_unreadable")
            continue
        if eff_min and min(w, h) < eff_min:
            _reject(im, "too_small")
            continue

        # 3) selection: trust-all, else keyword / optional VLM screen.
        if mode == "all":
            reason: Optional[str] = "trusted_source"
        else:
            hit = _matches_keyword(original or rel, kw) or _matches_keyword(rel or "", kw)
            reason = None
            if hit:
                reason = f"keyword:{hit}"
            elif use_vlm:
                try:
                    if vlm_responder(src_path):
                        reason = "vlm:towel"
                except Exception as exc:  # a flaky classifier must not abort the run
                    log(f"[warn] VLM screen failed for {rel}: {exc}")
            if reason is None:
                _reject(im, "no_keyword_match" if not use_vlm else "vlm_rejected")
                continue

        # 4) dedup across kept candidates.
        hsh = im.get("hash") or image_hash(src_path)
        if hsh in seen_hashes:
            _reject(im, "duplicate")
            continue

        sid = f"{next_id:06d}"
        ext = os.path.splitext(rel)[1].lower() or ".jpg"
        img_rel = f"images/{sid}{ext}"
        try:
            shutil.copyfile(src_path, os.path.join(out, img_rel))
        except OSError:
            _reject(im, "copy_failed")
            continue
        seen_hashes[hsh] = sid
        entry = {
            "id": sid,
            "image": img_rel,
            "original": original,
            "hash": hsh,
            "width": w,
            "height": h,
            "source": im.get("source") or loaded["source"],
            "license": im.get("license", ""),
            "filter_reason": reason,
        }
        for k in _PROVENANCE_KEYS:           # carry generated/real lineage forward
            if im.get(k) is not None:
                entry[k] = im[k]
        kept.append(entry)
        by_reason[reason.split(":")[0]] = by_reason.get(reason.split(":")[0], 0) + 1
        next_id += 1

    manifest = {
        "dataset": os.path.basename(os.path.normpath(out)),
        "kind": RAW_MANIFEST_KIND,
        "candidate_of": os.path.basename(os.path.normpath(input_dir)),
        "input_kind": loaded["kind"],
        "source": loaded["source"],
        "filter_mode": mode,
        "vlm_used": use_vlm,
        "keywords": kw,
        "num_input": len(src_images),
        "num_images": len(kept),
        "num_rejected": len(rejected),
        "reasons": by_reason,
        "labeled": False,
        "needs_labeling": True,
        "images": kept,
        "rejected": rejected,
    }
    write_json(paths["manifest"], manifest)
    log(f"Filtered {len(kept)} candidates (of {manifest['num_input']}) -> {out} "
        f"[{len(rejected)} rejected; mode={mode}]")
    return {
        "status": "ok",
        "out": out,
        "mode": mode,
        "num_input": manifest["num_input"],
        "kept": len(kept),
        "rejected": len(rejected),
        "reasons": by_reason,
        "vlm_used": use_vlm,
        "manifest": paths["manifest"],
    }
