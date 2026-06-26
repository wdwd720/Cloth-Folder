"""Colored-marker detection — RGB nearest-color (legacy) + HSV blob (robust).

Two detectors live here:

* :func:`detect_color_markers` / :func:`keypoints_from_markers` — the original
  numpy nearest-RGB-color centroiding. Cheap and used by the synthetic data path.
* :func:`detect_markers_hsv` / :func:`keypoints_from_markers_hsv` — the robust
  HSV-blob detector for the *real* demo: convert to HSV, threshold per color,
  keep the largest blob per color, centroid it, and report exactly which colors
  were missing. This is the fastest reliable path for a physical towel tagged
  with 4 colored corner stickers.

Physical marker convention (configurable via a marker map):
    red -> top_left, blue -> top_right, green -> bottom_left, orange/yellow -> bottom_right

Coordinates are returned as ``(u, v) == (x, y) == (col, row)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from terafold.physics.cloth_state import ClothKeypoints

__all__ = [
    "DEFAULT_MARKER_COLORS",
    "DEFAULT_MARKER_MAP",
    "COLOR_HSV_RANGES",
    "detect_color_markers",
    "keypoints_from_markers",
    "MarkerDetectionResult",
    "detect_markers_hsv",
    "keypoints_from_markers_hsv",
    "parse_marker_map",
    "rgb_to_hsv",
]

# Corner/keypoint -> RGB. Physical convention: TL red, TR blue, BL green, BR orange.
DEFAULT_MARKER_COLORS: Dict[str, tuple] = {
    "top_left": (230, 30, 30),  # red
    "top_right": (40, 60, 230),  # blue
    "bottom_left": (30, 200, 30),  # green
    "bottom_right": (245, 140, 20),  # orange
    "grasp": (230, 30, 230),  # magenta
    "place": (30, 220, 220),  # cyan
    "fold_a": (235, 220, 20),  # yellow
    "fold_b": (150, 60, 220),  # purple
}

# Color name -> corner. The default is the physical convention above.
DEFAULT_MARKER_MAP: Dict[str, str] = {
    "red": "top_left",
    "blue": "top_right",
    "green": "bottom_left",
    "orange": "bottom_right",
}

# Color name -> list of (h_lo, h_hi, s_min, v_min). H in degrees [0,360], S/V in [0,1].
# Red wraps around 0, so it has two hue windows.
COLOR_HSV_RANGES: Dict[str, List[Tuple[float, float, float, float]]] = {
    "red": [(0.0, 12.0, 0.45, 0.30), (348.0, 360.0, 0.45, 0.30)],
    "orange": [(13.0, 44.0, 0.45, 0.35)],
    "yellow": [(45.0, 70.0, 0.40, 0.35)],
    "green": [(80.0, 160.0, 0.35, 0.25)],
    "cyan": [(165.0, 195.0, 0.35, 0.30)],
    "blue": [(200.0, 255.0, 0.35, 0.25)],
    "purple": [(256.0, 290.0, 0.30, 0.25)],
    "magenta": [(291.0, 347.0, 0.35, 0.25)],
}


# --------------------------------------------------------------------------
# Legacy RGB nearest-color detector (kept for the synthetic / fallback path)
# --------------------------------------------------------------------------


def detect_color_markers(
    image: np.ndarray,
    colors: Optional[Dict[str, tuple]] = None,
    tol: float = 60.0,
    min_pixels: int = 3,
) -> Dict[str, np.ndarray]:
    """Locate markers by nearest-RGB-color thresholding + weighted centroid."""
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
        if int(sel.sum()) < min_pixels:
            continue
        wmap = np.where(sel, np.clip(tol - dist, 0.0, None), 0.0)
        total = float(wmap.sum())
        if total <= 0:
            continue
        out[name] = np.array(
            [float((wmap * xs).sum() / total), float((wmap * ys).sum() / total)]
        )
    return out


def keypoints_from_markers(
    image: np.ndarray, colors: Optional[Dict[str, tuple]] = None
) -> Optional[ClothKeypoints]:
    """Build :class:`ClothKeypoints` from nearest-RGB markers (all 4 corners required)."""
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


# --------------------------------------------------------------------------
# HSV blob detector (robust, real-demo path)
# --------------------------------------------------------------------------


def rgb_to_hsv(image: np.ndarray):
    """Vectorized RGB(uint8/float) -> (H[0,360], S[0,1], V[0,1]) float arrays."""
    img = np.asarray(image, dtype=np.float64)[:, :, :3] / 255.0
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    mx = img.max(axis=2)
    mn = img.min(axis=2)
    diff = mx - mn
    h = np.zeros_like(mx)
    mask = diff > 1e-9
    rm = mask & (mx == r)
    h[rm] = (60.0 * ((g[rm] - b[rm]) / diff[rm]) + 360.0) % 360.0
    gm = mask & (mx == g) & ~rm
    h[gm] = 60.0 * ((b[gm] - r[gm]) / diff[gm]) + 120.0
    bm = mask & (mx == b) & ~rm & ~gm
    h[bm] = 60.0 * ((r[bm] - g[bm]) / diff[bm]) + 240.0
    s = np.where(mx > 1e-9, diff / np.where(mx > 1e-9, mx, 1.0), 0.0)
    return h, s, mx


def _color_mask(h, s, v, color: str) -> np.ndarray:
    ranges = COLOR_HSV_RANGES.get(color.lower())
    if ranges is None:
        raise ValueError(f"unknown marker color {color!r}; known: {sorted(COLOR_HSV_RANGES)}")
    mask = np.zeros(h.shape, dtype=bool)
    for h_lo, h_hi, s_min, v_min in ranges:
        mask |= (h >= h_lo) & (h <= h_hi) & (s >= s_min) & (v >= v_min)
    return mask


def _largest_blob_centroid(mask: np.ndarray, min_area: int) -> Optional[Tuple[float, float, int]]:
    """Centroid (u,v) + area of the largest connected blob, or None if too small."""
    try:
        from scipy import ndimage

        labels, n = ndimage.label(mask)
        if n == 0:
            return None
        sizes = ndimage.sum(mask, labels, index=np.arange(1, n + 1))
        k = int(np.argmax(sizes)) + 1
        area = int(sizes[k - 1])
        if area < min_area:
            return None
        cy, cx = ndimage.center_of_mass(mask, labels, k)
        return float(cx), float(cy), area
    except Exception:
        ys, xs = np.nonzero(mask)
        if xs.size < min_area:
            return None
        return float(xs.mean()), float(ys.mean()), int(xs.size)


def parse_marker_map(spec: Optional[str]) -> Dict[str, str]:
    """Parse ``"red:top_left,blue:top_right,..."`` into a color->corner dict."""
    if not spec:
        return dict(DEFAULT_MARKER_MAP)
    out: Dict[str, str] = {}
    valid_corners = {"top_left", "top_right", "bottom_left", "bottom_right",
                     "grasp", "place", "fold_a", "fold_b"}
    for pair in spec.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if ":" not in pair:
            raise ValueError(f"bad marker-map entry {pair!r}; expected color:corner")
        color, corner = (p.strip().lower() for p in pair.split(":", 1))
        if color not in COLOR_HSV_RANGES:
            raise ValueError(f"unknown color {color!r} in marker-map")
        if corner not in valid_corners:
            raise ValueError(f"unknown corner {corner!r} in marker-map")
        out[color] = corner
    return out


@dataclass
class MarkerDetectionResult:
    """Outcome of an HSV marker detection."""

    corners: Dict[str, np.ndarray]  # corner_name -> (u, v) for detected corners
    found_colors: Dict[str, np.ndarray]  # color -> (u, v)
    missing_colors: List[str]  # colors in the map with no usable blob
    marker_map: Dict[str, str]
    masks: Dict[str, np.ndarray] = field(default_factory=dict)
    keypoints: Optional[ClothKeypoints] = None

    @property
    def all_corners_found(self) -> bool:
        return all(
            c in self.corners
            for c in ("top_left", "top_right", "bottom_left", "bottom_right")
        )


def detect_markers_hsv(
    image: np.ndarray,
    marker_map: Optional[Dict[str, str]] = None,
    min_area: int = 12,
    return_masks: bool = False,
) -> MarkerDetectionResult:
    """Detect colored corner markers via HSV blobs; report missing colors.

    Uses the largest blob per color and never claims success for a color whose
    blob is missing or below ``min_area`` — those land in ``missing_colors``.
    """
    img = np.asarray(image)
    if img.ndim != 3 or img.shape[2] < 3:
        raise ValueError("detect_markers_hsv expects an (H, W, 3) RGB image")
    marker_map = marker_map or dict(DEFAULT_MARKER_MAP)
    h, s, v = rgb_to_hsv(img)

    found_colors: Dict[str, np.ndarray] = {}
    corners: Dict[str, np.ndarray] = {}
    missing: List[str] = []
    masks: Dict[str, np.ndarray] = {}
    for color, corner in marker_map.items():
        mask = _color_mask(h, s, v, color)
        if return_masks:
            masks[color] = mask
        blob = _largest_blob_centroid(mask, min_area)
        if blob is None:
            missing.append(color)
            continue
        uv = np.array([blob[0], blob[1]], dtype=np.float64)
        found_colors[color] = uv
        corners[corner] = uv

    keypoints = None
    needed = ("top_left", "top_right", "bottom_left", "bottom_right")
    if all(c in corners for c in needed):
        keypoints = ClothKeypoints(
            top_left=corners["top_left"],
            top_right=corners["top_right"],
            bottom_right=corners["bottom_right"],
            bottom_left=corners["bottom_left"],
            grasp=corners.get("grasp"),
            place=corners.get("place"),
            fold_a=corners.get("fold_a"),
            fold_b=corners.get("fold_b"),
        )
    return MarkerDetectionResult(
        corners=corners, found_colors=found_colors, missing_colors=missing,
        marker_map=marker_map, masks=masks, keypoints=keypoints,
    )


def keypoints_from_markers_hsv(
    image: np.ndarray, marker_map: Optional[Dict[str, str]] = None, min_area: int = 12
) -> Optional[ClothKeypoints]:
    """HSV detection that returns keypoints only when all 4 corners are present."""
    return detect_markers_hsv(image, marker_map=marker_map, min_area=min_area).keypoints
