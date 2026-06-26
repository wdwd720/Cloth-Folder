"""Colored-marker detection (numpy only).

When the cloth is tagged with small colored stickers at known keypoints (the
``marker_mode`` rendered by :mod:`terafold.vision.synthetic_cloth`, or physical
fiducials on a real towel), this module recovers keypoint pixel locations by
nearest-color thresholding + centroiding. It is a cheap, dependency-free
ground-truth/fallback path that does not need a learned detector.

Coordinates are returned as ``(u, v) == (x, y) == (col, row)`` to match the
convention used by :class:`terafold.physics.cloth_state.ClothKeypoints`.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from terafold.physics.cloth_state import ClothKeypoints

__all__ = [
    "DEFAULT_MARKER_COLORS",
    "detect_color_markers",
    "keypoints_from_markers",
]

# Distinct, well-separated RGB colors for the eight detector keypoints.
DEFAULT_MARKER_COLORS: Dict[str, tuple] = {
    "top_left": (230, 30, 30),  # red
    "top_right": (30, 200, 30),  # green
    "bottom_right": (40, 60, 230),  # blue
    "bottom_left": (235, 220, 20),  # yellow
    "grasp": (230, 30, 230),  # magenta
    "place": (30, 220, 220),  # cyan
    "fold_a": (245, 140, 20),  # orange
    "fold_b": (150, 60, 220),  # purple
}


def detect_color_markers(
    image: np.ndarray,
    colors: Optional[Dict[str, tuple]] = None,
    tol: float = 60.0,
    min_pixels: int = 3,
) -> Dict[str, np.ndarray]:
    """Locate colored markers by nearest-color thresholding + centroid.

    Returns ``{name: (u, v)}`` only for markers with at least ``min_pixels``
    matching pixels within Euclidean color distance ``tol``.
    """
    if colors is None:
        colors = DEFAULT_MARKER_COLORS
    img = np.asarray(image, dtype=np.float64)
    if img.ndim != 3 or img.shape[2] < 3:
        raise ValueError("detect_color_markers expects an (H, W, 3) RGB image")
    img = img[:, :, :3]
    h, w = img.shape[:2]
    ys = np.arange(h, dtype=np.float64)[:, None]
    xs = np.arange(w, dtype=np.float64)[None, :]

    out: Dict[str, np.ndarray] = {}
    for name, color in colors.items():
        c = np.asarray(color, dtype=np.float64)
        dist = np.sqrt(((img - c[None, None, :]) ** 2).sum(axis=2))
        sel = dist <= tol
        count = int(sel.sum())
        if count < min_pixels:
            continue
        # Weight by closeness to the target color for a robust centroid.
        wmap = np.where(sel, np.clip(tol - dist, 0.0, None), 0.0)
        total = float(wmap.sum())
        if total <= 0:
            continue
        u = float((wmap * xs).sum() / total)
        v = float((wmap * ys).sum() / total)
        out[name] = np.array([u, v], dtype=np.float64)
    return out


def keypoints_from_markers(
    image: np.ndarray, colors: Optional[Dict[str, tuple]] = None
) -> Optional[ClothKeypoints]:
    """Build :class:`ClothKeypoints` from detected markers.

    Requires all four corners to be present; ``None`` is returned otherwise so
    callers can fall back to mask-based corner estimation.
    """
    found = detect_color_markers(image, colors=colors)
    required = ("top_left", "top_right", "bottom_right", "bottom_left")
    if not all(name in found for name in required):
        return None
    return ClothKeypoints(
        top_left=found["top_left"],
        top_right=found["top_right"],
        bottom_right=found["bottom_right"],
        bottom_left=found["bottom_left"],
        grasp=found.get("grasp"),
        place=found.get("place"),
        fold_a=found.get("fold_a"),
        fold_b=found.get("fold_b"),
    )
