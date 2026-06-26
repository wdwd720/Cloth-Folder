"""Classical cloth segmentation and corner extraction (numpy/scipy only).

This is the perception fallback used when no learned keypoint model is available
(and a building block the success scorer can lean on). It segments the towel
from the table by color contrast, extracts a boundary polygon, and recovers four
ordered corners via a minimum-area enclosing rectangle over the convex hull.

No torch / cv2 / PIL. Uses ``scipy.ndimage`` for connected components and
hole-filling.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import ndimage

from terafold.math.geometry import order_corners_clockwise, polygon_perimeter
from terafold.physics.cloth_state import ClothKeypoints, ClothMaskState

__all__ = [
    "segment_cloth",
    "mask_to_polygon",
    "corners_from_mask",
    "mask_state_from_mask",
    "estimate_wrinkle_proxy",
]


# --------------------------------------------------------------------------
# Segmentation
# --------------------------------------------------------------------------


def _otsu_threshold(values: np.ndarray, bins: int = 64) -> float:
    """Otsu's threshold on a 1D array of nonnegative values."""
    v = values.ravel()
    vmax = float(v.max())
    if vmax <= 1e-9:
        return 0.0
    hist, edges = np.histogram(v, bins=bins, range=(0.0, vmax))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return 0.5 * vmax
    p = hist / total
    centers = 0.5 * (edges[:-1] + edges[1:])
    omega = np.cumsum(p)
    mu = np.cumsum(p * centers)
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    denom = np.where(denom < 1e-12, 1e-12, denom)
    sigma_b = (mu_t * omega - mu) ** 2 / denom
    idx = int(np.argmax(sigma_b))
    return float(centers[idx])


def segment_cloth(image: np.ndarray, method: str = "auto") -> np.ndarray:
    """Segment the cloth from the table -> ``(H, W)`` uint8 mask (0 / 255).

    The background table color is estimated from the image border, a per-pixel
    color-distance map is thresholded (Otsu), and the largest filled connected
    component is returned.
    """
    img = np.asarray(image, dtype=np.float64)
    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=2)
    img = img[:, :, :3]
    h, w = img.shape[:2]

    # Background color from a border frame (table is assumed to surround cloth).
    bw = max(1, int(round(0.06 * min(h, w))))
    border = np.concatenate(
        [
            img[:bw].reshape(-1, 3),
            img[-bw:].reshape(-1, 3),
            img[:, :bw].reshape(-1, 3),
            img[:, -bw:].reshape(-1, 3),
        ],
        axis=0,
    )
    bg = np.median(border, axis=0)

    dist = np.sqrt(((img - bg[None, None, :]) ** 2).sum(axis=2))
    thr = _otsu_threshold(dist)
    mask = dist > max(thr, 8.0)

    # Clean up: fill holes, keep the largest connected component.
    mask = ndimage.binary_fill_holes(mask)
    labels, n = ndimage.label(mask)
    if n > 1:
        sizes = ndimage.sum(np.ones_like(labels), labels, index=np.arange(1, n + 1))
        largest = int(np.argmax(sizes)) + 1
        mask = labels == largest
    elif n == 0:
        return np.zeros((h, w), dtype=np.uint8)
    mask = ndimage.binary_fill_holes(mask)
    return (mask.astype(np.uint8)) * 255


# --------------------------------------------------------------------------
# Polygon / hull
# --------------------------------------------------------------------------


def _convex_hull(points: np.ndarray) -> np.ndarray:
    """Andrew's monotone chain convex hull -> ordered (M, 2) CCW-ish polygon."""
    pts = np.unique(np.asarray(points, dtype=np.float64), axis=0)
    if pts.shape[0] <= 3:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = np.array(lower[:-1] + upper[:-1], dtype=np.float64)
    return hull


def mask_to_polygon(mask: np.ndarray, max_points: int = 64) -> np.ndarray:
    """Boundary polygon of a binary mask as ``(N, 2)`` (x, y), up to ``max_points``.

    Returns the convex hull of the mask boundary pixels (ordered, robust to
    small ragged edges), subsampled to ``max_points``.
    """
    m = np.asarray(mask) > 0
    if not m.any():
        return np.zeros((0, 2), dtype=np.float64)
    # Boundary pixels: foreground minus its erosion.
    eroded = ndimage.binary_erosion(m, border_value=0)
    boundary = m & ~eroded
    rows, cols = np.nonzero(boundary)
    if rows.size == 0:
        rows, cols = np.nonzero(m)
    pts = np.stack([cols.astype(np.float64), rows.astype(np.float64)], axis=1)  # (x, y)
    hull = _convex_hull(pts)
    if hull.shape[0] > max_points:
        idx = np.linspace(0, hull.shape[0] - 1, max_points).round().astype(int)
        hull = hull[np.unique(idx)]
    return hull


# --------------------------------------------------------------------------
# Corner extraction (min-area rectangle)
# --------------------------------------------------------------------------


def _min_area_rect(hull: np.ndarray) -> np.ndarray:
    """Minimum-area enclosing rectangle of a convex hull -> 4 corners (x, y)."""
    if hull.shape[0] < 3:
        # Degenerate: pad bbox.
        mn = hull.min(axis=0)
        mx = hull.max(axis=0)
        return np.array([[mn[0], mn[1]], [mx[0], mn[1]], [mx[0], mx[1]], [mn[0], mx[1]]])

    best_area = np.inf
    best_rect = None
    n = hull.shape[0]
    for i in range(n):
        edge = hull[(i + 1) % n] - hull[i]
        L = float(np.linalg.norm(edge))
        if L < 1e-9:
            continue
        ux = edge / L
        uy = np.array([-ux[1], ux[0]])
        proj_x = hull @ ux
        proj_y = hull @ uy
        minx, maxx = proj_x.min(), proj_x.max()
        miny, maxy = proj_y.min(), proj_y.max()
        area = (maxx - minx) * (maxy - miny)
        if area < best_area:
            best_area = area
            corners = np.array(
                [
                    minx * ux + miny * uy,
                    maxx * ux + miny * uy,
                    maxx * ux + maxy * uy,
                    minx * ux + maxy * uy,
                ]
            )
            best_rect = corners
    if best_rect is None:
        mn = hull.min(axis=0)
        mx = hull.max(axis=0)
        return np.array([[mn[0], mn[1]], [mx[0], mn[1]], [mx[0], mx[1]], [mn[0], mx[1]]])
    return best_rect


def corners_from_mask(mask: np.ndarray) -> ClothKeypoints:
    """Recover four ordered corners (TL, TR, BR, BL) from a binary cloth mask."""
    m = np.asarray(mask) > 0
    if not m.any():
        z = np.zeros(2)
        return ClothKeypoints(top_left=z, top_right=z, bottom_right=z, bottom_left=z)
    rows, cols = np.nonzero(m)
    pts = np.stack([cols.astype(np.float64), rows.astype(np.float64)], axis=1)
    hull = _convex_hull(pts)
    rect = _min_area_rect(hull)
    ordered = order_corners_clockwise(rect)  # [TL, TR, BR, BL]
    return ClothKeypoints(
        top_left=ordered[0],
        top_right=ordered[1],
        bottom_right=ordered[2],
        bottom_left=ordered[3],
    )


# --------------------------------------------------------------------------
# Mask state + wrinkle proxy
# --------------------------------------------------------------------------


def estimate_wrinkle_proxy(image: np.ndarray, mask: np.ndarray) -> float:
    """Texture-energy wrinkle proxy in [0, 1] over the masked cloth region."""
    img = np.asarray(image, dtype=np.float64)
    if img.ndim == 3:
        img = img[:, :, :3].mean(axis=2)
    m = np.asarray(mask) > 0
    if m.sum() < 1:
        return 0.0
    gx = ndimage.sobel(img, axis=1)
    gy = ndimage.sobel(img, axis=0)
    energy = np.sqrt(gx**2 + gy**2)
    val = float(energy[m].mean())
    # Sobel magnitudes on flat cloth are small; normalize to a sane scale.
    return float(np.clip(val / 160.0, 0.0, 1.0))


def mask_state_from_mask(
    mask: np.ndarray, image: Optional[np.ndarray] = None
) -> ClothMaskState:
    """Build a :class:`ClothMaskState` (polygon, bbox, curvature, wrinkle) from a mask."""
    m = np.asarray(mask)
    mb = m > 0
    polygon = mask_to_polygon(m)
    if mb.any():
        rows, cols = np.nonzero(mb)
        bbox = np.array(
            [float(cols.min()), float(rows.min()), float(cols.max()), float(rows.max())]
        )
        visible = float(mb.sum())
    else:
        bbox = None
        visible = 0.0

    # Curvature proxy: deviation of perimeter from an equal-area circle.
    curvature = None
    if polygon.shape[0] >= 3:
        from terafold.math.geometry import polygon_area

        area = polygon_area(polygon)
        if area > 1e-6:
            perim = polygon_perimeter(polygon, closed=True)
            circle_perim = 2.0 * np.sqrt(np.pi * area)
            curvature = float(np.clip(perim / max(circle_perim, 1e-6) - 1.0, 0.0, None))

    wrinkle = (
        estimate_wrinkle_proxy(image, m) if image is not None and mb.any() else None
    )

    return ClothMaskState(
        polygon=polygon if polygon.shape[0] >= 3 else None,
        mask_shape=tuple(m.shape[:2]),
        visible_area_px=visible,
        bbox=bbox,
        contour_curvature=curvature,
        wrinkle_proxy=wrinkle,
        mask=m,
    )
