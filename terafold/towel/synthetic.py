"""OPTIONAL synthetic towel augmentation — *not* the main data path.

Real user/external photos are the DEFAULT training data for towel
perception. This module exists only to manufacture a small, deterministic
batch of trivially-easy top-down towel images (a bright quadrilateral on a
contrasting background) whose four corners are known *exactly*. It is useful
for pre-training / smoke-testing the YOLO-pose pipeline before real labels
exist — never as a substitute for real data.

Everything here is numpy + stdlib only. Images are written through
:func:`terafold.vision.imageio.imwrite`, so the pure-Python PNG codec works
with no OpenCV/Pillow installed. Labels are produced via the towel
:mod:`~terafold.towel.schema` foundation and the resulting manifest is tagged
``kind='synthetic'`` so downstream merges can keep real images first-class
(and drop these via ``--real-only``).
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List

import numpy as np

from terafold.data.episode_schema import write_json
from terafold.towel.dataset import LABEL_SCHEMA, dataset_paths, save_manifest
from terafold.towel.schema import (TOWEL_STATES, bbox_from_corners, image_hash,
                                   is_usable, new_label, validate_label)
from terafold.vision.imageio import imwrite

__all__ = ["generate_towel_dataset"]

# States we cycle through for the synthetic quads (all trainable, with corners).
_SYNTH_STATES: List[str] = ["flat_unfolded", "wrinkled_unfolded", "partially_folded"]


def _noop(_m: str) -> None:
    pass


# --------------------------------------------------------------------------
# Geometry (numpy-only): build a known quadrilateral inside the frame.
# --------------------------------------------------------------------------


def _fit_inside(corners: np.ndarray, size: int, margin: float) -> np.ndarray:
    """Scale ``corners`` toward their centroid until inside ``[margin, size-1-margin]``."""
    lo = float(margin)
    hi = float(size - 1 - margin)
    c = corners.mean(axis=0)
    for _ in range(60):
        mn = corners.min(axis=0)
        mx = corners.max(axis=0)
        scale = 1.0
        for k in range(2):
            if mx[k] > hi:
                scale = min(scale, (hi - c[k]) / (mx[k] - c[k] + 1e-9))
            if mn[k] < lo:
                scale = min(scale, (c[k] - lo) / (c[k] - mn[k] + 1e-9))
        if scale >= 1.0:
            break
        corners = c + (corners - c) * (scale * 0.97)
    return np.clip(corners, lo, hi)


def _make_corners(rng: np.random.Generator, size: int) -> np.ndarray:
    """Random rotated/sheared rectangle corners ordered TL, TR, BR, BL.

    Returns an ``(4, 2)`` float array of ``[x, y]`` pixel coordinates that are
    guaranteed to sit a few pixels inside the image border.
    """
    w = rng.uniform(0.42, 0.62) * size
    h = rng.uniform(0.42, 0.62) * size
    base = np.array(
        [[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]],
        dtype=np.float64,
    )

    # Mild shear so the quad is not a perfect axis-aligned rectangle.
    shear = rng.uniform(-0.18, 0.18)
    base = base @ np.array([[1.0, shear], [0.0, 1.0]]).T

    # Mild rotation.
    ang = np.deg2rad(rng.uniform(-22.0, 22.0))
    ca, sa = np.cos(ang), np.sin(ang)
    base = base @ np.array([[ca, -sa], [sa, ca]]).T

    # Translate to a jittered centre.
    cx = size / 2 + rng.uniform(-1.0, 1.0) * 0.06 * size
    cy = size / 2 + rng.uniform(-1.0, 1.0) * 0.06 * size
    corners = base + np.array([cx, cy])
    return _fit_inside(corners, size, margin=4.0)


def _polygon_mask(corners: np.ndarray, h: int, w: int) -> np.ndarray:
    """Crossing-number rasterization of a convex quad to a bool ``(H, W)`` mask."""
    poly = np.asarray(corners, dtype=np.float64)
    n = poly.shape[0]
    ys = np.arange(h, dtype=np.float64)[:, None]
    xs = np.arange(w, dtype=np.float64)[None, :]
    inside = np.zeros((h, w), dtype=bool)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        cond = (yi > ys) != (yj > ys)
        denom = yj - yi
        if abs(denom) < 1e-12:
            j = i
            continue
        xint = (xj - xi) * (ys - yi) / denom + xi
        inside ^= cond & (xs < xint)
        j = i
    return inside


def _render(rng: np.random.Generator, corners: np.ndarray, size: int) -> np.ndarray:
    """Paint a white/off-white quad on a contrasting background -> uint8 RGB."""
    # Contrasting (darkish, slightly colored) background + mild noise.
    bg = rng.uniform(30.0, 80.0, size=3)
    img = np.broadcast_to(bg[None, None, :], (size, size, 3)).astype(np.float64).copy()
    img += rng.normal(0.0, 4.0, size=(size, size, 3))

    # White / off-white towel + mild noise.
    mask = _polygon_mask(corners, size, size)
    cloth = rng.uniform(205.0, 245.0, size=3)
    cloth_field = cloth[None, None, :] + rng.normal(0.0, 6.0, size=(size, size, 3))
    img = np.where(mask[:, :, None], cloth_field, img)

    return np.clip(img, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def generate_towel_dataset(
    num: int,
    out: str,
    image_size: int = 256,
    seed: int = 0,
    log: Callable[[str], None] = print,
) -> Dict[str, Any]:
    """Render ``num`` deterministic synthetic towel images + exact corner labels.

    SYNTHETIC AUGMENTATION ONLY — real images are the default training data.

    Each sample is a bright quadrilateral (slightly rotated/sheared, mild
    noise) on a contrasting background, written as a real PNG. Because we
    construct the quad, the four corners (tl, tr, br, bl) are known exactly and
    written into a fully-labeled, training-usable towel label
    (``source='synthetic'``, ``usable_for_training=True``). The manifest is
    tagged ``kind='synthetic'``.

    Args:
        num: number of images to generate.
        out: output dataset directory.
        image_size: square image side in pixels.
        seed: seed for ``numpy.random.default_rng`` (fully deterministic).
        log: line logger (defaults to ``print``).

    Returns:
        ``{"out": out, "num_images": num, "kind": "synthetic"}``.
    """
    log("=" * 70)
    log("SYNTHETIC TOWEL AUGMENTATION (pretraining/smoke-test only)")
    log("Real user/external photos are the DEFAULT training data — not this.")
    log("=" * 70)

    rng = np.random.default_rng(seed)
    paths = dataset_paths(out)
    os.makedirs(paths["images"], exist_ok=True)
    os.makedirs(paths["labels"], exist_ok=True)

    samples: List[Dict[str, Any]] = []
    for i in range(int(num)):
        sid = f"{i + 1:06d}"
        img_rel = f"images/{sid}.png"
        lab_rel = f"labels/{sid}.json"

        corners = _make_corners(rng, image_size)
        img = _render(rng, corners, image_size)
        img_abs = os.path.join(out, img_rel)
        imwrite(img_abs, img)

        # Known corners -> integer pixel coords, clipped strictly in-bounds.
        corners_int = [
            [int(np.clip(round(x), 0, image_size - 1)),
             int(np.clip(round(y), 0, image_size - 1))]
            for x, y in corners
        ]
        state = _SYNTH_STATES[i % len(_SYNTH_STATES)]
        label = new_label(img_rel, source="synthetic", state=state)
        label["corners"] = {
            "tl": corners_int[0], "tr": corners_int[1],
            "br": corners_int[2], "bl": corners_int[3],
        }
        label["bbox"] = bbox_from_corners(corners_int)
        label["usable_for_training"] = True
        label["notes"] = "synthetic augmentation (auto-generated, exact corners)"

        problems = validate_label(label)
        if problems:  # should never happen; guard against silent bad labels
            raise ValueError(f"synthetic label {sid} invalid: {problems}")
        if not is_usable(label):
            raise ValueError(f"synthetic label {sid} not usable")

        write_json(os.path.join(out, lab_rel), label)

        samples.append({
            "id": sid,
            "image": img_rel,
            "label": lab_rel,
            "original": "",  # synthetic: no source file
            "hash": image_hash(img_abs),
            "width": int(image_size),
            "height": int(image_size),
            "source": "synthetic",
        })

    manifest = {
        "dataset": os.path.basename(os.path.normpath(out)),
        "kind": "synthetic",
        "label_schema": LABEL_SCHEMA,
        "states": TOWEL_STATES,
        "source": "synthetic",
        "license_note": "synthetic augmentation — generated, no real photos",
        "num_images": len(samples),
        "num_input_files": 0,
        "skipped_duplicates": 0,
        "num_labeled_usable": len(samples),
        "generator": "terafold.towel.synthetic",
        "seed": int(seed),
        "image_size": int(image_size),
        "samples": samples,
    }
    save_manifest(out, manifest)
    _write_readme(out, manifest)

    log(f"Wrote {len(samples)} SYNTHETIC towel images + exact labels -> {out}")
    return {"out": out, "num_images": len(samples), "kind": "synthetic"}


def _write_readme(root: str, manifest: Dict[str, Any]) -> None:
    paths = dataset_paths(root)
    lines = [
        f"# SYNTHETIC towel dataset — {manifest['dataset']}",
        "",
        "**SYNTHETIC AUGMENTATION / PRETRAINING ONLY.**",
        "Real user/external photos are the DEFAULT training data. These images are",
        "auto-generated bright quadrilaterals on a contrasting background, used only",
        "to pre-train / smoke-test the towel-pose pipeline before real labels exist.",
        "",
        f"- kind: **{manifest['kind']}**",
        f"- source: {manifest['source']}",
        f"- images: {manifest['num_images']}  (all labeled & usable, exact corners)",
        f"- label schema: {manifest['label_schema']}",
        f"- generator: {manifest['generator']}  (seed={manifest['seed']}, "
        f"image_size={manifest['image_size']})",
        "",
        "## Layout",
        "- `images/` — generated PNG towels.",
        "- `labels/` — one JSON per image (exact 4 corners tl/tr/br/bl, state, bbox).",
        "- `manifest.json` — dataset index (kind=synthetic) + per-sample metadata.",
        "",
        "Merge AFTER real data and prefer `--real-only` for final training:",
        "",
        "```bash",
        f"python3 -m terafold merge-towel --inputs data/towel_real_v0,{root} "
        "--out data/towel_merged_v0",
        "```",
    ]
    with open(paths["readme"], "w") as f:
        f.write("\n".join(lines) + "\n")
