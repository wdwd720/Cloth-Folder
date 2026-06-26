"""Pure-numpy overlay rasterization for cloth perception and fold plans.

Draws circles, lines, polylines, keypoints, crease lines, grasp/place arrows and
full fold-plan overlays directly into ``(H, W, 3)`` uint8 RGB arrays. No
matplotlib / cv2 / PIL — just numpy rasterization. Image files are written via
:func:`terafold.vision.imageio.imwrite`.

All point coordinates are ``(x, y) == (col, row)``.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from terafold.physics.cloth_state import ClothKeypoints, FoldLine, FoldState, GraspPlacePair
from terafold.vision.imageio import ensure_uint8_rgb, imwrite

__all__ = [
    "draw_circle",
    "draw_line",
    "draw_polyline",
    "draw_keypoints",
    "draw_fold_line",
    "draw_grasp_place",
    "draw_fold_plan",
    "save_visualization",
]

KEYPOINT_COLORS = {
    "top_left": (230, 30, 30),
    "top_right": (30, 200, 30),
    "bottom_right": (40, 60, 230),
    "bottom_left": (235, 220, 20),
    "grasp": (230, 30, 230),
    "place": (30, 220, 220),
    "fold_a": (245, 140, 20),
    "fold_b": (150, 60, 220),
}


def _as_rgb(img: np.ndarray) -> np.ndarray:
    """Return a writable contiguous uint8 RGB copy."""
    return ensure_uint8_rgb(img).copy()


def _color(c) -> np.ndarray:
    return np.clip(np.asarray(c, dtype=np.float64), 0, 255).astype(np.uint8)


def draw_circle(
    img: np.ndarray, center, radius: float = 4, color=(255, 0, 0), filled: bool = True
) -> np.ndarray:
    """Draw a (filled) circle centered at ``center`` ``(x, y)``."""
    out = _as_rgb(img)
    h, w = out.shape[:2]
    cx, cy = float(center[0]), float(center[1])
    r = max(0.5, float(radius))
    ys = np.arange(h)[:, None]
    xs = np.arange(w)[None, :]
    d2 = (xs - cx) ** 2 + (ys - cy) ** 2
    if filled:
        sel = d2 <= r * r
    else:
        inner = (r - 1.5) ** 2
        sel = (d2 <= r * r) & (d2 >= inner)
    out[sel] = _color(color)
    return out


def draw_line(
    img: np.ndarray, p0, p1, color=(255, 255, 255), thickness: int = 1
) -> np.ndarray:
    """Draw a straight line from ``p0`` to ``p1`` (both ``(x, y)``)."""
    out = _as_rgb(img)
    h, w = out.shape[:2]
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])
    n = int(max(2, round(max(abs(x1 - x0), abs(y1 - y0)) + 1)))
    xs = np.linspace(x0, x1, n)
    ys = np.linspace(y0, y1, n)
    col = _color(color)
    t = max(0, int(thickness) - 1)
    for dx in range(-t, t + 1):
        for dy in range(-t, t + 1):
            xi = np.clip(np.round(xs + dx).astype(int), 0, w - 1)
            yi = np.clip(np.round(ys + dy).astype(int), 0, h - 1)
            out[yi, xi] = col
    return out


def draw_polyline(
    img: np.ndarray,
    pts: Sequence,
    color=(255, 255, 255),
    closed: bool = False,
    thickness: int = 1,
) -> np.ndarray:
    """Draw a connected sequence of points."""
    out = _as_rgb(img)
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape[0] < 2:
        return out
    for i in range(pts.shape[0] - 1):
        out = draw_line(out, pts[i], pts[i + 1], color=color, thickness=thickness)
    if closed:
        out = draw_line(out, pts[-1], pts[0], color=color, thickness=thickness)
    return out


def draw_keypoints(
    img: np.ndarray,
    kp: ClothKeypoints,
    radius: float = 4,
    draw_edges: bool = True,
    colors: Optional[dict] = None,
) -> np.ndarray:
    """Overlay cloth keypoints (and corner outline) on an image."""
    out = _as_rgb(img)
    colors = colors or KEYPOINT_COLORS
    if draw_edges:
        out = draw_polyline(out, kp.corners, color=(255, 255, 255), closed=True, thickness=1)
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
        if pt is None:
            continue
        out = draw_circle(out, pt, radius=radius, color=colors.get(name, (255, 0, 0)))
    return out


def draw_fold_line(
    img: np.ndarray, fold_line: FoldLine, color=(255, 255, 0), thickness: int = 2
) -> np.ndarray:
    """Draw the crease line (uses stored endpoints, else extends through the image)."""
    out = _as_rgb(img)
    if fold_line.a is not None and fold_line.b is not None:
        a, b = fold_line.a, fold_line.b
    else:
        h, w = out.shape[:2]
        span = float(np.hypot(h, w))
        a = fold_line.point - fold_line.direction * span
        b = fold_line.point + fold_line.direction * span
    return draw_line(out, a, b, color=color, thickness=thickness)


def _draw_arrow(img, p0, p1, color, thickness=2):
    out = draw_line(img, p0, p1, color=color, thickness=thickness)
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    v = p1 - p0
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        return out
    v = v / n
    perp = np.array([-v[1], v[0]])
    head = max(4.0, 0.18 * n)
    left = p1 - v * head + perp * head * 0.5
    right = p1 - v * head - perp * head * 0.5
    out = draw_line(out, p1, left, color=color, thickness=thickness)
    out = draw_line(out, p1, right, color=color, thickness=thickness)
    return out


def draw_grasp_place(
    img: np.ndarray,
    gp: GraspPlacePair,
    grasp_color=(230, 30, 230),
    place_color=(30, 220, 220),
    radius: float = 5,
) -> np.ndarray:
    """Draw the grasp point, place point, and an arrow between them."""
    out = _as_rgb(img)
    out = _draw_arrow(out, gp.grasp, gp.place, color=(255, 255, 255), thickness=2)
    out = draw_circle(out, gp.grasp, radius=radius, color=grasp_color)
    out = draw_circle(out, gp.place, radius=radius, color=place_color)
    return out


def draw_fold_plan(img: np.ndarray, plan, frames=None) -> np.ndarray:
    """Overlay a :class:`FoldPlan`: keypoints, crease, grasp/place, trajectory.

    Geometry is drawn in pixel space. Table-frame entities (the plan's
    grasp/place/trajectory) are projected to pixels via ``frames`` when it is
    calibrated; otherwise the plan's image-frame perception
    (``plan.image_fold_state``) is used directly.
    """
    out = _as_rgb(img)

    calibrated = bool(frames is not None and getattr(frames, "image_calibrated", False))

    def to_px(xy):
        if calibrated:
            return np.asarray(frames.table_xy_to_pixel(xy), dtype=np.float64).reshape(-1)
        return np.asarray(xy, dtype=np.float64).reshape(-1)

    # Image-frame perception (always available, already in pixels).
    img_fs: Optional[FoldState] = getattr(plan, "image_fold_state", None)
    if img_fs is not None:
        out = draw_keypoints(out, img_fs.keypoints, radius=4)
        if img_fs.fold_line is not None:
            out = draw_fold_line(out, img_fs.fold_line)
        if img_fs.grasp_place is not None:
            out = draw_grasp_place(out, img_fs.grasp_place)
        elif img_fs.keypoints.grasp is not None and img_fs.keypoints.place is not None:
            out = _draw_arrow(
                out, img_fs.keypoints.grasp, img_fs.keypoints.place,
                color=(255, 255, 255), thickness=2,
            )

    # Table-frame entities, projected when calibrated.
    if calibrated:
        gp = getattr(plan, "grasp_place", None)
        if gp is not None:
            out = draw_grasp_place(
                out, GraspPlacePair(grasp=to_px(gp.grasp), place=to_px(gp.place))
            )
        fl = getattr(plan, "fold_line", None)
        if fl is not None and fl.a is not None and fl.b is not None:
            out = draw_line(out, to_px(fl.a), to_px(fl.b), color=(255, 255, 0), thickness=2)
        traj = getattr(plan, "trajectory", None)
        if traj is not None and getattr(traj, "waypoints", None) is not None:
            wp = np.asarray(traj.waypoints, dtype=np.float64)
            if wp.shape[0] >= 2:
                px = np.array([to_px(p[:2]) for p in wp])
                out = draw_polyline(out, px, color=(255, 120, 0), thickness=1)
    return out


def save_visualization(path: str, img: np.ndarray) -> None:
    """Write an overlay image to ``path`` via the pure image I/O backend."""
    imwrite(path, ensure_uint8_rgb(img))
