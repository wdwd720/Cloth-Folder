"""Keypoint heatmap dataset for the cloth detector (numpy/scipy only).

This turns the on-disk synthetic dataset written by
:mod:`terafold.vision.synthetic_cloth` into ``(image, heatmap)`` training
samples for the U-Net keypoint model (Model 1).

The core :class:`KeypointDataset` is **numpy only**: it loads PNGs through
:mod:`terafold.vision.imageio`, resizes them, applies numpy/scipy augmentations
that keep the keypoint labels consistent, and renders Gaussian target heatmaps.
Only :func:`build_torch_dataset` (a thin wrapper used by training) imports torch,
and it does so lazily so this module always imports with numpy alone.

Conventions:

* keypoints are ``(x, y) == (col, row)`` pixel coordinates, ordered by
  :data:`terafold.physics.cloth_state.DETECTOR_KEYPOINTS`.
* heatmaps are ``(K, h, w)`` float32 in ``[0, 1]`` with a Gaussian peak at each
  keypoint; channels for missing keypoints are all-zero.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage

from terafold.physics.cloth_state import (
    DETECTOR_KEYPOINTS,
    ClothKeypoints,
    FoldState,
)
from terafold.vision.imageio import imread

__all__ = [
    "IMAGE_SIZE",
    "HEATMAP_SIZE",
    "HEATMAP_SIGMA",
    "NUM_KEYPOINTS",
    "render_heatmap",
    "keypoints_to_heatmaps",
    "heatmaps_to_keypoints",
    "Augmentor",
    "KeypointSample",
    "KeypointDataset",
    "build_torch_dataset",
]

IMAGE_SIZE: int = 256
HEATMAP_SIZE: int = 64
HEATMAP_SIGMA: float = 2.0
NUM_KEYPOINTS: int = len(DETECTOR_KEYPOINTS)


# --------------------------------------------------------------------------
# Image resize (bilinear, numpy only)
# --------------------------------------------------------------------------


def _resize_rgb(img: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    """Bilinear resize an ``(H, W, 3)`` (or ``(H, W)``) image to ``out_hw``."""
    arr = np.asarray(img, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    h_in, w_in = arr.shape[:2]
    h_out, w_out = int(out_hw[0]), int(out_hw[1])
    if (h_in, w_in) == (h_out, w_out):
        return arr.copy()
    # Sample centers mapped back to input coordinates.
    ys = (np.arange(h_out) + 0.5) * (h_in / h_out) - 0.5
    xs = (np.arange(w_out) + 0.5) * (w_in / w_out) - 0.5
    ys = np.clip(ys, 0, h_in - 1)
    xs = np.clip(xs, 0, w_in - 1)
    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    y1 = np.minimum(y0 + 1, h_in - 1)
    x1 = np.minimum(x0 + 1, w_in - 1)
    wy = (ys - y0)[:, None, None]
    wx = (xs - x0)[None, :, None]
    top = arr[y0][:, x0] * (1 - wx) + arr[y0][:, x1] * wx
    bot = arr[y1][:, x0] * (1 - wx) + arr[y1][:, x1] * wx
    out = top * (1 - wy) + bot * wy
    return out


# --------------------------------------------------------------------------
# Heatmap encode / decode
# --------------------------------------------------------------------------


def render_heatmap(
    out_hw: Tuple[int, int], point_xy: Optional[np.ndarray], sigma: float
) -> np.ndarray:
    """A single Gaussian heatmap of shape ``out_hw`` peaked at ``point_xy``.

    ``point_xy`` is ``(x, y)`` in the heatmap's own coordinate frame. ``None`` or
    a NaN point yields an all-zero map.
    """
    h, w = int(out_hw[0]), int(out_hw[1])
    hm = np.zeros((h, w), dtype=np.float32)
    if point_xy is None:
        return hm
    x, y = float(point_xy[0]), float(point_xy[1])
    if not np.isfinite(x) or not np.isfinite(y):
        return hm
    s = max(float(sigma), 1e-3)
    ys = np.arange(h, dtype=np.float64)[:, None]
    xs = np.arange(w, dtype=np.float64)[None, :]
    g = np.exp(-((xs - x) ** 2 + (ys - y) ** 2) / (2.0 * s * s))
    return g.astype(np.float32)


def keypoints_to_heatmaps(
    kp: ClothKeypoints,
    image_size: int = IMAGE_SIZE,
    out_size: int = HEATMAP_SIZE,
    sigma: float = HEATMAP_SIGMA,
) -> np.ndarray:
    """Render ``(K, out_size, out_size)`` target heatmaps from keypoints.

    Keypoints are given in ``image_size`` pixel coordinates and scaled into the
    ``out_size`` heatmap grid. Missing keypoints produce all-zero channels.
    """
    arr = kp.to_array()  # (K, 2) in DETECTOR_KEYPOINTS order, NaN for missing
    scale = float(out_size) / float(image_size)
    out = np.zeros((arr.shape[0], int(out_size), int(out_size)), dtype=np.float32)
    for i in range(arr.shape[0]):
        x, y = arr[i]
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        out[i] = render_heatmap((out_size, out_size), (x * scale, y * scale), sigma)
    return out


def heatmaps_to_keypoints(
    heatmaps: np.ndarray, image_size: int = IMAGE_SIZE, eps: float = 1e-6
) -> ClothKeypoints:
    """Decode ``(K, h, w)`` heatmaps into keypoints via soft-argmax.

    Coordinates are returned in ``image_size`` pixel space. A channel whose peak
    is effectively zero is treated as a missing keypoint (NaN).
    """
    hm = np.asarray(heatmaps, dtype=np.float64)
    if hm.ndim != 3:
        raise ValueError(f"expected (K, h, w) heatmaps, got {hm.shape}")
    k, h, w = hm.shape
    arr = np.full((k, 2), np.nan, dtype=np.float64)
    ys = np.arange(h, dtype=np.float64)[:, None]
    xs = np.arange(w, dtype=np.float64)[None, :]
    sx = float(image_size) / float(w)
    sy = float(image_size) / float(h)
    for i in range(k):
        ch = np.clip(hm[i], 0.0, None)
        if float(ch.max()) <= eps:
            continue
        total = float(ch.sum())
        if total <= eps:
            continue
        x = float((ch * xs).sum() / total)
        y = float((ch * ys).sum() / total)
        arr[i] = [x * sx, y * sy]
    if k != len(DETECTOR_KEYPOINTS):
        # Pad/truncate defensively to the canonical channel count.
        fixed = np.full((len(DETECTOR_KEYPOINTS), 2), np.nan)
        fixed[: min(k, len(DETECTOR_KEYPOINTS))] = arr[: len(DETECTOR_KEYPOINTS)]
        arr = fixed
    return ClothKeypoints.from_array(arr)


# --------------------------------------------------------------------------
# Augmentation (numpy / scipy, label-consistent)
# --------------------------------------------------------------------------


@dataclass
class Augmentor:
    """Label-consistent numpy/scipy image augmentations.

    Geometric transforms (rotation + translation) are applied to both the image
    and the keypoints; photometric transforms (brightness, color jitter, noise,
    shadow, occlusion) only touch the image. ``__call__`` returns a new
    ``(image_u8, keypoints)`` pair and never mutates its inputs.
    """

    max_rotation_deg: float = 15.0
    max_translate_frac: float = 0.06
    brightness_range: Tuple[float, float] = (0.8, 1.2)
    color_jitter: float = 0.1
    noise_std: float = 5.0
    shadow_prob: float = 0.3
    shadow_strength: float = 0.35
    occlusion_prob: float = 0.2
    occlusion_frac: float = 0.15
    p_geometric: float = 0.9

    def _rotate_translate(
        self, img: np.ndarray, arr: np.ndarray, rng: np.random.Generator
    ) -> Tuple[np.ndarray, np.ndarray]:
        h, w = img.shape[:2]
        theta = np.deg2rad(rng.uniform(-self.max_rotation_deg, self.max_rotation_deg))
        tx = rng.uniform(-1.0, 1.0) * self.max_translate_frac * w
        ty = rng.uniform(-1.0, 1.0) * self.max_translate_frac * h
        ca, sa = np.cos(theta), np.sin(theta)
        R = np.array([[ca, -sa], [sa, ca]])  # xy rotation
        c_xy = np.array([(w - 1) / 2.0, (h - 1) / 2.0])
        c_rc = c_xy[::-1]
        t_rc = np.array([ty, tx])
        # Image: input_rc = M_rc @ out_rc + offset, with M_rc == R.
        offset = c_rc - R @ (c_rc + t_rc)
        out_img = np.empty_like(img, dtype=np.float64)
        for ch in range(img.shape[2]):
            out_img[:, :, ch] = ndimage.affine_transform(
                img[:, :, ch].astype(np.float64),
                R,
                offset=offset,
                order=1,
                mode="nearest",
            )
        # Keypoints forward map: q = R @ (p - c) + c + t.
        new = arr.copy()
        finite = np.all(np.isfinite(arr), axis=1)
        if finite.any():
            p = arr[finite] - c_xy[None, :]
            q = p @ R.T + c_xy[None, :] + np.array([tx, ty])[None, :]
            new[finite] = q
        return out_img, new

    def __call__(
        self, image: np.ndarray, kp: ClothKeypoints, rng: np.random.Generator
    ) -> Tuple[np.ndarray, ClothKeypoints]:
        img = np.asarray(image, dtype=np.float64).copy()
        if img.ndim == 2:
            img = np.repeat(img[:, :, None], 3, axis=2)
        h, w = img.shape[:2]
        arr = kp.to_array()

        if rng.random() < self.p_geometric:
            img, arr = self._rotate_translate(img, arr, rng)

        # Brightness.
        img *= rng.uniform(*self.brightness_range)
        # Per-channel color jitter.
        if self.color_jitter > 0:
            img *= rng.uniform(
                1.0 - self.color_jitter, 1.0 + self.color_jitter, size=3
            )[None, None, :]
        # Shadow: a smooth low-frequency darkening field.
        if rng.random() < self.shadow_prob:
            coarse = rng.uniform(0.0, 1.0, size=(3, 3))
            field2 = ndimage.zoom(coarse, (h / 3.0, w / 3.0), order=1)[:h, :w]
            field2 = field2 / (field2.max() + 1e-9)
            img *= 1.0 - self.shadow_strength * field2[:, :, None]
        # Occlusion blob.
        if rng.random() < self.occlusion_prob:
            cx = rng.uniform(0, w)
            cy = rng.uniform(0, h)
            rad = max(3.0, self.occlusion_frac * min(h, w))
            ys = np.arange(h)[:, None]
            xs = np.arange(w)[None, :]
            occ = (xs - cx) ** 2 + (ys - cy) ** 2 <= rad * rad
            img[occ] = rng.uniform(30.0, 220.0, size=3)
        # Sensor noise.
        if self.noise_std > 0:
            img += rng.normal(0.0, self.noise_std, size=img.shape)

        img_u8 = np.clip(img, 0, 255).astype(np.uint8)
        return img_u8, ClothKeypoints.from_array(arr)


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------


@dataclass
class KeypointSample:
    """One training sample: CHW image + target heatmaps + keypoints."""

    image_chw: np.ndarray  # (3, image_size, image_size) float32 in [0, 1]
    heatmaps: np.ndarray  # (K, heatmap_size, heatmap_size) float32
    keypoints: ClothKeypoints
    meta: Dict = field(default_factory=dict)


class KeypointDataset:
    """Numpy dataset over a synthetic-cloth directory (see ``synthetic_cloth``).

    Each item is a :class:`KeypointSample`. No torch is required; use
    :func:`build_torch_dataset` to get a torch ``Dataset`` for training.
    """

    def __init__(
        self,
        synth_dir: str,
        augment: bool = True,
        image_size: int = IMAGE_SIZE,
        heatmap_size: int = HEATMAP_SIZE,
        sigma: float = HEATMAP_SIGMA,
        augmentor: Optional[Augmentor] = None,
        seed: int = 0,
    ) -> None:
        self.synth_dir = str(synth_dir)
        self.augment = bool(augment)
        self.image_size = int(image_size)
        self.heatmap_size = int(heatmap_size)
        self.sigma = float(sigma)
        self.augmentor = augmentor or Augmentor()
        self._base_seed = int(seed)

        index_path = os.path.join(self.synth_dir, "index.json")
        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"no index.json in {self.synth_dir!r}; generate data with "
                "terafold.vision.synthetic_cloth.generate_synthetic_dataset"
            )
        with open(index_path) as f:
            self.index = json.load(f)
        self.items: List[dict] = list(self.index.get("items", []))

    def __len__(self) -> int:
        return len(self.items)

    def _load_raw(self, i: int) -> Tuple[np.ndarray, ClothKeypoints, dict]:
        item = self.items[i]
        img = imread(os.path.join(self.synth_dir, item["image"]))
        with open(os.path.join(self.synth_dir, item["label"])) as f:
            label = json.load(f)
        fs = FoldState.from_dict(label["fold_state"])
        return img, fs.keypoints, label

    def __getitem__(self, i: int) -> KeypointSample:
        img, kp, label = self._load_raw(i)
        h_in, w_in = img.shape[:2]

        # Resize image + scale keypoints into ``image_size``.
        img_r = _resize_rgb(img, (self.image_size, self.image_size))
        arr = kp.to_array()
        finite = np.all(np.isfinite(arr), axis=1)
        arr[finite, 0] *= self.image_size / float(w_in)
        arr[finite, 1] *= self.image_size / float(h_in)
        kp = ClothKeypoints.from_array(arr)
        img_r = np.clip(img_r, 0, 255).astype(np.uint8)

        if self.augment:
            rng = np.random.default_rng(self._base_seed * 1_000_003 + i)
            img_r, kp = self.augmentor(img_r, kp, rng)

        heatmaps = keypoints_to_heatmaps(
            kp, self.image_size, self.heatmap_size, self.sigma
        )
        image_chw = (
            np.asarray(img_r, dtype=np.float32).transpose(2, 0, 1) / 255.0
        )
        meta = {
            "index": int(label.get("index", i)),
            "marker_mode": bool(label.get("marker_mode", False)),
            "image_size": self.image_size,
            "heatmap_size": self.heatmap_size,
        }
        return KeypointSample(
            image_chw=image_chw, heatmaps=heatmaps, keypoints=kp, meta=meta
        )


def build_torch_dataset(synth_dir: str, augment: bool = True, **kwargs):
    """Wrap :class:`KeypointDataset` as a torch ``Dataset`` (lazy torch import).

    Each item is ``{"image": (3,H,W) float32, "heatmaps": (K,h,w) float32}``.
    """
    try:
        import torch
        from torch.utils.data import Dataset
    except Exception as exc:  # pragma: no cover - exercised only without torch
        raise ImportError(
            "build_torch_dataset requires torch. Install it with "
            "`pip install -e '.[torch]'` (or `pip install torch`)."
        ) from exc

    base = KeypointDataset(synth_dir, augment=augment, **kwargs)

    class _TorchKeypointDataset(Dataset):
        def __init__(self, ds: KeypointDataset) -> None:
            self.ds = ds

        def __len__(self) -> int:
            return len(self.ds)

        def __getitem__(self, idx: int) -> Dict[str, "torch.Tensor"]:
            s = self.ds[idx]
            return {
                "image": torch.from_numpy(np.ascontiguousarray(s.image_chw)),
                "heatmaps": torch.from_numpy(np.ascontiguousarray(s.heatmaps)),
            }

    return _TorchKeypointDataset(base)
