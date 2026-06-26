"""Synthetic top-down towel scene generator (numpy/scipy only).

This is the Stage-0 data engine: it renders realistic-enough top-down images of a
single towel lying on a table, together with **exact, in-bounds** labels (four
corners + grasp + place + the two crease endpoints). The keypoint detector
trains on this output directly, so correctness of the labels matters as much as
visual realism.

Everything is vectorized numpy. Realism comes from cheap, smooth random fields:

* lighting / vignetting   -> low-frequency multiplicative field
* wrinkles                -> mid-frequency multiplicative field over the cloth
* shadows                 -> blurred, shifted copy of the cloth mask
* table texture           -> low-frequency colour field + fine gaussian noise
* camera noise            -> additive gaussian noise over the whole frame
* non-rectangular cloth   -> small random corner perturbations
* wavy edges              -> per-edge sinusoidal displacement (tapered at corners)
* partial occlusion       -> an occasional occluder blob over the cloth
* colored markers         -> small disks at keypoints when ``marker_mode``

No torch / cv2 / PIL. Images are written through
:func:`terafold.vision.imageio.imwrite` so the pure-python PNG codec works when
no image backend is installed.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage

from terafold.physics.cloth_state import (
    ClothKeypoints,
    ClothMaskState,
    FoldLine,
    FoldState,
)
from terafold.physics.fold_geometry import predict_folded_footprint
from terafold.math.geometry import reflect_point_across_line
from terafold.vision.imageio import imread, imwrite

__all__ = [
    "SyntheticClothConfig",
    "SyntheticSample",
    "render_towel_scene",
    "make_before_after_pair",
    "generate_synthetic_dataset",
    "generate_fold_pairs",
]


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


@dataclass
class SyntheticClothConfig:
    """Tunable ranges for the synthetic towel renderer.

    Defaults are calibrated for ``image_size`` ~256 and produce visually varied
    but always in-bounds towels. All "frac" values are fractions of
    ``image_size``; colours are RGB in 0..255.
    """

    image_size: int = 256
    # Cloth extent (fraction of the image side).
    min_frac: float = 0.30
    max_frac: float = 0.70
    min_aspect: float = 0.6  # height / width
    max_aspect: float = 1.6
    # Pose randomization.
    max_rotation_deg: float = 35.0
    center_jitter_frac: float = 0.08
    # Shape randomization.
    corner_jitter_frac: float = 0.05  # fraction of min(w, h)
    edge_wave_amp_frac: float = 0.02  # fraction of image side
    points_per_edge: int = 8
    margin_px: float = 3.0
    # Colours (RGB ranges).
    table_color_min: Tuple[float, float, float] = (60.0, 70.0, 80.0)
    table_color_max: Tuple[float, float, float] = (140.0, 150.0, 160.0)
    cloth_color_min: Tuple[float, float, float] = (90.0, 90.0, 110.0)
    cloth_color_max: Tuple[float, float, float] = (235.0, 235.0, 240.0)
    # Lighting / shading fields.
    light_coarse: int = 5
    light_amp: float = 0.18
    wrinkle_coarse: int = 14
    wrinkle_amp: float = 0.16
    # Shadow.
    shadow_strength: float = 0.35
    shadow_blur: float = 4.0
    shadow_offset: Tuple[float, float] = (4.0, 4.0)  # (dy, dx) in px
    # Noise.
    table_noise: float = 5.0
    cloth_noise: float = 6.0
    camera_noise: float = 3.0
    # Occlusion.
    occlusion_prob: float = 0.25
    occlusion_frac: float = 0.18  # blob radius as fraction of cloth size
    # Markers.
    marker_mode: bool = False
    marker_radius_frac: float = 0.018
    # Before/after pair shaping.
    fail_prob: float = 0.45
    success_jitter_frac: float = 0.01
    fail_jitter_frac: float = 0.09
    # Success thresholds (image-frame pixels) used to label fold pairs.
    success_min_overlap: float = 0.75
    success_max_corner_err_frac: float = 0.06
    success_max_wrinkle: float = 0.6
    success_min_visible_ratio: float = 0.30


@dataclass
class SyntheticSample:
    """One rendered scene with its ground-truth labels."""

    image: np.ndarray  # (H, W, 3) uint8 RGB
    keypoints: ClothKeypoints
    fold_line: FoldLine
    mask: np.ndarray  # (H, W) uint8 0/255
    polygon: np.ndarray  # (N, 2) float, cloth outline (x, y)
    meta: Dict


# --------------------------------------------------------------------------
# Smooth random fields
# --------------------------------------------------------------------------


def _smooth_field(
    rng: np.random.Generator,
    h: int,
    w: int,
    coarse: int = 6,
    low: float = 0.0,
    high: float = 1.0,
    blur: float = 0.0,
) -> np.ndarray:
    """A smooth (H, W) field from upsampled coarse random values."""
    coarse = max(2, int(coarse))
    c = rng.uniform(low, high, size=(coarse, coarse))
    f = ndimage.zoom(c, (h / coarse, w / coarse), order=3)
    f = f[:h, :w]
    if f.shape != (h, w):
        f = np.pad(
            f,
            ((0, max(0, h - f.shape[0])), (0, max(0, w - f.shape[1]))),
            mode="edge",
        )[:h, :w]
    if blur > 0:
        f = ndimage.gaussian_filter(f, blur)
    return f


# --------------------------------------------------------------------------
# Cloth geometry
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


def _random_corners(rng: np.random.Generator, cfg: SyntheticClothConfig) -> np.ndarray:
    """Random cloth corners ordered TL, TR, BR, BL (x right, y down)."""
    size = cfg.image_size
    w = rng.uniform(cfg.min_frac, cfg.max_frac) * size
    aspect = rng.uniform(cfg.min_aspect, cfg.max_aspect)
    h = float(np.clip(w * aspect, cfg.min_frac * size, cfg.max_frac * size))

    base = np.array(
        [[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]],
        dtype=np.float64,
    )
    # Non-rectangular deformation.
    base = base + rng.normal(0.0, cfg.corner_jitter_frac * min(w, h), size=(4, 2))

    # Rotation.
    ang = np.deg2rad(rng.uniform(-cfg.max_rotation_deg, cfg.max_rotation_deg))
    ca, sa = np.cos(ang), np.sin(ang)
    R = np.array([[ca, -sa], [sa, ca]])
    base = base @ R.T

    # Translation.
    cx = size / 2 + rng.uniform(-1.0, 1.0) * cfg.center_jitter_frac * size
    cy = size / 2 + rng.uniform(-1.0, 1.0) * cfg.center_jitter_frac * size
    corners = base + np.array([cx, cy])

    wave_room = cfg.edge_wave_amp_frac * size + 2.0
    marker_room = (cfg.marker_radius_frac * size + 1.0) if cfg.marker_mode else 0.0
    margin = cfg.margin_px + wave_room + marker_room
    return _fit_inside(corners, size, margin)


def _keypoints_from_corners(corners: np.ndarray) -> ClothKeypoints:
    """Build a fully-populated :class:`ClothKeypoints` for a right-to-left fold."""
    tl, tr, br, bl = corners
    kp = ClothKeypoints(top_left=tl, top_right=tr, bottom_right=br, bottom_left=bl)
    fold_line = FoldLine.from_corners(kp, direction="right_to_left")
    grasp = kp.mid_right
    place = reflect_point_across_line(grasp, fold_line.point, fold_line.direction)
    return ClothKeypoints(
        top_left=tl,
        top_right=tr,
        bottom_right=br,
        bottom_left=bl,
        grasp=grasp,
        place=place,
        fold_a=kp.mid_top,
        fold_b=kp.mid_bottom,
    )


def _wavy_polygon(
    rng: np.random.Generator, corners: np.ndarray, cfg: SyntheticClothConfig
) -> np.ndarray:
    """Dense cloth outline with sinusoidal edge waviness (corners stay fixed)."""
    size = cfg.image_size
    amp = cfg.edge_wave_amp_frac * size
    n = len(corners)
    ppe = max(2, int(cfg.points_per_edge))
    out: List[np.ndarray] = []
    for i in range(n):
        a = corners[i]
        b = corners[(i + 1) % n]
        edge = b - a
        L = float(np.linalg.norm(edge))
        if L < 1e-6:
            out.append(a.copy())
            continue
        normal = np.array([-edge[1], edge[0]]) / L
        phase = rng.uniform(0.0, 2.0 * np.pi)
        freq = rng.uniform(1.0, 3.0)
        ts = np.linspace(0.0, 1.0, ppe, endpoint=False)
        taper = 4.0 * ts * (1.0 - ts)  # 0 at corners, 1 at edge centre
        disp = amp * np.sin(2.0 * np.pi * freq * ts + phase) * taper
        pts = a[None, :] + ts[:, None] * edge[None, :] + disp[:, None] * normal[None, :]
        out.append(pts)
    return np.concatenate(out, axis=0)


def _polygon_mask(polygon: np.ndarray, h: int, w: int) -> np.ndarray:
    """Vectorized crossing-number rasterization of a polygon to a bool (H, W) mask."""
    poly = np.asarray(polygon, dtype=np.float64)
    n = poly.shape[0]
    ys = np.arange(h, dtype=np.float64)[:, None]  # (H, 1)
    xs = np.arange(w, dtype=np.float64)[None, :]  # (1, W)
    inside = np.zeros((h, w), dtype=bool)
    px = poly[:, 0]
    py = poly[:, 1]
    j = n - 1
    for i in range(n):
        yi, yj = py[i], py[j]
        xi, xj = px[i], px[j]
        cond = (yi > ys) != (yj > ys)  # (H, 1) broadcast
        denom = yj - yi
        if abs(denom) < 1e-12:
            j = i
            continue
        xint = (xj - xi) * (ys - yi) / denom + xi  # (H, 1)
        inside ^= cond & (xs < xint)
        j = i
    return inside


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def _draw_markers(
    img: np.ndarray, kp: ClothKeypoints, cfg: SyntheticClothConfig
) -> np.ndarray:
    """Paint small colored disks at each keypoint (in-place safe copy)."""
    from terafold.vision.marker_detector import DEFAULT_MARKER_COLORS

    size = cfg.image_size
    r = max(2, int(round(cfg.marker_radius_frac * size)))
    h, w = img.shape[:2]
    ys = np.arange(h)[:, None]
    xs = np.arange(w)[None, :]
    lookup = {
        "top_left": kp.top_left,
        "top_right": kp.top_right,
        "bottom_right": kp.bottom_right,
        "bottom_left": kp.bottom_left,
        "grasp": kp.grasp,
        "place": kp.place,
        "fold_a": kp.fold_a,
        "fold_b": kp.fold_b,
    }
    for name, pt in lookup.items():
        if pt is None or name not in DEFAULT_MARKER_COLORS:
            continue
        color = np.asarray(DEFAULT_MARKER_COLORS[name], dtype=np.float64)
        disk = (xs - pt[0]) ** 2 + (ys - pt[1]) ** 2 <= r * r
        img[disk] = color
    return img


def _paint_scene(
    rng: np.random.Generator,
    cfg: SyntheticClothConfig,
    corners: np.ndarray,
    marker_kp: Optional[ClothKeypoints] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render one cloth quad into a full table scene -> (image_u8, mask_u8, polygon)."""
    H = W = cfg.image_size

    # Table base colour + low-frequency lighting + fine texture noise.
    table_color = rng.uniform(
        np.asarray(cfg.table_color_min), np.asarray(cfg.table_color_max)
    )
    light = _smooth_field(
        rng, H, W, coarse=cfg.light_coarse, low=1.0 - cfg.light_amp, high=1.0 + cfg.light_amp
    )
    img = table_color[None, None, :] * light[:, :, None]
    img = img + rng.normal(0.0, cfg.table_noise, size=(H, W, 3))

    # Cloth polygon + mask.
    polygon = _wavy_polygon(rng, corners, cfg)
    mask = _polygon_mask(polygon, H, W)

    # Soft drop shadow (outside the cloth).
    m = mask.astype(np.float64)
    sh = ndimage.shift(m, cfg.shadow_offset, order=0, mode="constant")
    sh = ndimage.gaussian_filter(sh, cfg.shadow_blur)
    sh = np.clip(sh * (1.0 - m), 0.0, 1.0)
    img = img * (1.0 - sh[:, :, None] * cfg.shadow_strength)

    # Cloth colour + lighting + wrinkles.
    cloth_color = rng.uniform(
        np.asarray(cfg.cloth_color_min), np.asarray(cfg.cloth_color_max)
    )
    wr = _smooth_field(
        rng,
        H,
        W,
        coarse=cfg.wrinkle_coarse,
        low=1.0 - cfg.wrinkle_amp,
        high=1.0 + cfg.wrinkle_amp,
    )
    cloth = cloth_color[None, None, :] * (light * wr)[:, :, None]
    cloth = cloth + rng.normal(0.0, cfg.cloth_noise, size=(H, W, 3))
    img = np.where(mask[:, :, None], cloth, img)

    # Optional partial occlusion: an occluder blob over part of the cloth.
    if rng.random() < cfg.occlusion_prob:
        cloth_pts = np.argwhere(mask)
        if cloth_pts.shape[0] > 0:
            cy, cx = cloth_pts[rng.integers(cloth_pts.shape[0])]
            scale = np.linalg.norm(corners.max(axis=0) - corners.min(axis=0))
            rad = max(3.0, cfg.occlusion_frac * scale)
            ys = np.arange(H)[:, None]
            xs = np.arange(W)[None, :]
            occ = (xs - cx) ** 2 + (ys - cy) ** 2 <= rad * rad
            occ_color = rng.uniform(40.0, 200.0, size=3)
            img[occ] = occ_color
            mask[occ] = False  # occluded cloth is not visible

    # Markers.
    img = np.clip(img, 0, 255)
    if marker_kp is not None:
        img = _draw_markers(img, marker_kp, cfg)

    # Camera noise.
    img = img + rng.normal(0.0, cfg.camera_noise, size=(H, W, 3))
    img_u8 = np.clip(img, 0, 255).astype(np.uint8)
    mask_u8 = (mask.astype(np.uint8)) * 255
    return img_u8, mask_u8, polygon


# --------------------------------------------------------------------------
# Public scene API
# --------------------------------------------------------------------------


def render_towel_scene(
    rng: np.random.Generator,
    cfg: SyntheticClothConfig,
    marker_mode: bool = False,
) -> SyntheticSample:
    """Render one random towel scene with full ground-truth labels."""
    marker_mode = bool(marker_mode or cfg.marker_mode)
    corners = _random_corners(rng, cfg)
    kp = _keypoints_from_corners(corners)
    img, mask, polygon = _paint_scene(rng, cfg, corners, kp if marker_mode else None)
    fold_line = FoldLine.from_corners(kp, direction="right_to_left")
    meta = {
        "image_size": int(cfg.image_size),
        "marker_mode": marker_mode,
        "rectangularity": float(kp.rectangularity()),
        "width_px": float(kp.width()),
        "height_px": float(kp.height()),
    }
    return SyntheticSample(
        image=img, keypoints=kp, fold_line=fold_line, mask=mask, polygon=polygon, meta=meta
    )


def _fold_state_for(
    sample_kp: ClothKeypoints,
    fold_line: FoldLine,
    mask_u8: np.ndarray,
    polygon: np.ndarray,
    image_size: int,
    keep_mask: bool = True,
) -> FoldState:
    mask_state = ClothMaskState(
        polygon=polygon,
        mask_shape=(image_size, image_size),
        visible_area_px=float((mask_u8 > 0).sum()),
        mask=(mask_u8 if keep_mask else None),
    )
    return FoldState(
        keypoints=sample_kp,
        fold_line=fold_line,
        mask=mask_state,
        frame="image",
        image_shape=(image_size, image_size),
    )


def _success_thresholds(cfg: SyntheticClothConfig):
    @dataclass
    class _T:
        max_corner_error_m: float
        min_overlap_ratio: float
        max_wrinkle_score: float
        min_visible_area_ratio: float

    return _T(
        max_corner_error_m=cfg.success_max_corner_err_frac * cfg.image_size,
        min_overlap_ratio=cfg.success_min_overlap,
        max_wrinkle_score=cfg.success_max_wrinkle,
        min_visible_area_ratio=cfg.success_min_visible_ratio,
    )


def make_before_after_pair(
    rng: np.random.Generator, cfg: SyntheticClothConfig
) -> Tuple[np.ndarray, np.ndarray, bool, Dict]:
    """Render a before/after fold pair and score it.

    Returns ``(before_img, after_img, success, metrics)`` where ``success`` and
    ``metrics`` come from :func:`terafold.physics.fold_quality.compute_fold_quality`
    so the labels match how the success scorer is trained/evaluated.
    """
    from terafold.physics.fold_quality import compute_fold_quality

    before = render_towel_scene(rng, cfg)
    before_kp = before.keypoints
    folded = predict_folded_footprint(before_kp, direction="right_to_left")
    after_corners = folded.corners.copy()

    scale = float(np.linalg.norm(before_kp.corners.max(axis=0) - before_kp.corners.min(axis=0)))
    intended_fail = rng.random() < cfg.fail_prob
    jitter = (cfg.fail_jitter_frac if intended_fail else cfg.success_jitter_frac) * scale
    after_corners = after_corners + rng.normal(0.0, jitter, size=(4, 2))
    after_corners = _fit_inside(after_corners, cfg.image_size, cfg.margin_px + 2.0)

    after_kp_full = _keypoints_from_corners(after_corners)
    after_img, after_mask, after_poly = _paint_scene(rng, cfg, after_corners, None)
    after_fl = FoldLine.from_corners(after_kp_full, direction="right_to_left")

    before_fs = _fold_state_for(
        before_kp, before.fold_line, before.mask, before.polygon, cfg.image_size
    )
    after_fs = _fold_state_for(
        after_kp_full, after_fl, after_mask, after_poly, cfg.image_size
    )

    metrics = compute_fold_quality(
        before_fs, after_fs, success_cfg=_success_thresholds(cfg), direction="right_to_left"
    )
    md = metrics.to_dict()
    return before.image, after_img, bool(md["success"]), md


# --------------------------------------------------------------------------
# Dataset writers
# --------------------------------------------------------------------------


def _write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def generate_synthetic_dataset(
    out_dir: str,
    num: int,
    image_size: int = 256,
    seed: int = 0,
    marker_mode: bool = False,
) -> dict:
    """Generate ``num`` labelled towel scenes under ``out_dir``.

    On-disk layout (consumed by ``keypoint_dataset``)::

        <out>/images/<i:06d>.png   RGB uint8
        <out>/masks/<i:06d>.png    binary 0/255
        <out>/labels/<i:06d>.json  {"fold_state", "marker_mode", "image_size", "index"}
        <out>/index.json           {"num", "image_size", "marker_mode", "items", "meta"}
    """
    rng = np.random.default_rng(seed)
    cfg = SyntheticClothConfig(image_size=image_size, marker_mode=marker_mode)

    images_dir = os.path.join(out_dir, "images")
    masks_dir = os.path.join(out_dir, "masks")
    labels_dir = os.path.join(out_dir, "labels")
    for d in (images_dir, masks_dir, labels_dir):
        os.makedirs(d, exist_ok=True)

    items: List[dict] = []
    for i in range(num):
        sample = render_towel_scene(rng, cfg, marker_mode=marker_mode)
        img_rel = f"images/{i:06d}.png"
        mask_rel = f"masks/{i:06d}.png"
        label_rel = f"labels/{i:06d}.json"

        imwrite(os.path.join(out_dir, img_rel), sample.image)
        imwrite(os.path.join(out_dir, mask_rel), sample.mask)

        fs = _fold_state_for(
            sample.keypoints,
            sample.fold_line,
            sample.mask,
            sample.polygon,
            image_size,
            keep_mask=False,
        )
        label = {
            "fold_state": fs.to_dict(),
            "marker_mode": bool(marker_mode),
            "image_size": int(image_size),
            "index": int(i),
        }
        _write_json(os.path.join(out_dir, label_rel), label)
        items.append({"image": img_rel, "mask": mask_rel, "label": label_rel})

    summary = {
        "num": int(num),
        "image_size": int(image_size),
        "marker_mode": bool(marker_mode),
        "items": items,
        "meta": {"seed": int(seed), "generator": "terafold.vision.synthetic_cloth"},
    }
    _write_json(os.path.join(out_dir, "index.json"), summary)
    return summary


def generate_fold_pairs(
    out_dir: str,
    num: int,
    image_size: int = 256,
    seed: int = 0,
) -> dict:
    """Generate ``num`` before/after fold pairs under ``out_dir``.

    Layout (consumed by ``train_success``)::

        <out>/pairs/<i:06d>/before.png
        <out>/pairs/<i:06d>/after.png
        <out>/pairs/<i:06d>/label.json  {"success": bool, "metrics": {...}, "index": int}
        <out>/index.json
    """
    rng = np.random.default_rng(seed)
    cfg = SyntheticClothConfig(image_size=image_size)

    items: List[dict] = []
    n_success = 0
    for i in range(num):
        before_img, after_img, success, metrics = make_before_after_pair(rng, cfg)
        rel = f"pairs/{i:06d}"
        pair_dir = os.path.join(out_dir, rel)
        os.makedirs(pair_dir, exist_ok=True)
        imwrite(os.path.join(pair_dir, "before.png"), before_img)
        imwrite(os.path.join(pair_dir, "after.png"), after_img)
        label = {"success": bool(success), "metrics": metrics, "index": int(i)}
        _write_json(os.path.join(pair_dir, "label.json"), label)
        n_success += int(success)
        items.append(
            {
                "before": f"{rel}/before.png",
                "after": f"{rel}/after.png",
                "label": f"{rel}/label.json",
            }
        )

    summary = {
        "num": int(num),
        "image_size": int(image_size),
        "num_success": int(n_success),
        "items": items,
        "meta": {"seed": int(seed), "generator": "terafold.vision.synthetic_cloth"},
    }
    _write_json(os.path.join(out_dir, "index.json"), summary)
    return summary
