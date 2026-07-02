"""Draw a labeling preview overlay: 4 corners, bbox, fold axis, state.

Rendering needs to decode the image, which requires OpenCV or Pillow (or a PNG via
the pure codec). If none can decode the image, the overlay is skipped gracefully
(``rendered=False``) — the label itself is still saved by the caller.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from terafold.towel.schema import CORNER_ORDER, bbox_from_corners, label_corners_xy

__all__ = ["draw_towel_overlay"]

# Distinct colors per corner (RGB): tl green, tr yellow, br red, bl blue.
_CORNER_COLORS = {"tl": (0, 220, 0), "tr": (230, 210, 0), "br": (230, 40, 40), "bl": (40, 110, 240)}


def _load_rgb(image_path: str):
    """Decode an image to an (H,W,3) uint8 RGB array, or None if impossible."""
    try:
        from terafold.vision.imageio import have_cv2, have_pil

        ext = os.path.splitext(image_path)[1].lower()
        if have_cv2() or have_pil() or ext == ".png":
            from terafold.vision.imageio import imread

            return imread(image_path)
    except Exception:
        return None
    return None


def draw_towel_overlay(
    image_path: str, label: Dict[str, Any], out_path: str, draw_dims: int = 8,
) -> Dict[str, Any]:
    """Render corners/bbox/fold-axis onto a copy of the image and save it.

    Returns ``{"rendered": bool, "out": path|None, "reason": str}``. Never raises on
    a missing image backend — it just reports ``rendered=False``.
    """
    import numpy as np

    rgb = _load_rgb(image_path)
    if rgb is None:
        return {"rendered": False, "out": None,
                "reason": "no image backend for this format (install terafold[vision] for JPEG)"}
    img = np.array(rgb, dtype=np.uint8).copy()
    h, w = img.shape[0], img.shape[1]
    # Scale dot size with the image so corners are clearly visible on big photos.
    r = min(40, max(int(draw_dims), min(w, h) // 60, 4))

    def _dot(cx, cy, color, rad):
        x0, x1 = max(0, cx - rad), min(w, cx + rad + 1)
        y0, y1 = max(0, cy - rad), min(h, cy + rad + 1)
        img[y0:y1, x0:x1] = color

    def _line(p0, p1, color, thick=2):
        (x0, y0), (x1, y1) = p0, p1
        n = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
        for t in range(n + 1):
            x = int(round(x0 + (x1 - x0) * t / max(1, n)))
            y = int(round(y0 + (y1 - y0) * t / max(1, n)))
            _dot(x, y, color, thick)

    visible = label.get("visible") or {}
    corners = label.get("corners") or {}
    pts = {}
    for k in CORNER_ORDER:
        v = corners.get(k)
        if v is not None and len(v) == 2:
            pts[k] = (int(round(v[0])), int(round(v[1])))

    # Quad outline tl->tr->br->bl->tl (the visible towel silhouette).
    seq = [k for k in CORNER_ORDER if k in pts]
    if len(seq) == 4:
        order = ["tl", "tr", "br", "bl", "tl"]
        for a, b in zip(order, order[1:]):
            _line(pts[a], pts[b], (255, 255, 255), max(2, r // 4))

    # Big, ringed, labeled corner dots. Visible corners get a filled colored dot with
    # a white outline; hidden corners get a hollow gray ring + an X so review is easy.
    for k, (cx, cy) in pts.items():
        is_vis = bool(visible.get(k, True))
        _dot(cx, cy, (255, 255, 255), r + 2)        # white outline ring
        if is_vis:
            _dot(cx, cy, _CORNER_COLORS[k], r)
        else:
            _dot(cx, cy, (60, 60, 60), r)            # hollow-ish dark fill for hidden
            _line((cx - r, cy - r), (cx + r, cy + r), (255, 0, 0), max(1, r // 4))
            _line((cx - r, cy + r), (cx + r, cy - r), (255, 0, 0), max(1, r // 4))
        _corner_text(img, k.upper() + ("" if is_vis else "?"), cx, cy, r, _CORNER_COLORS[k])

    # bbox (from label or derived from corners).
    cxy = label_corners_xy(label)
    bbox = label.get("bbox")
    if bbox is None and cxy is not None:
        bbox = bbox_from_corners(cxy)
    if bbox is not None and len(bbox) == 4:
        x1, y1, x2, y2 = (int(round(v)) for v in bbox)
        _line((x1, y1), (x2, y1), (180, 180, 180), 1)
        _line((x2, y1), (x2, y2), (180, 180, 180), 1)
        _line((x2, y2), (x1, y2), (180, 180, 180), 1)
        _line((x1, y2), (x1, y1), (180, 180, 180), 1)

    # Fold axis: accept [[x,y],[x,y]] or {"p0":..,"p1":..} or "vertical"/"horizontal".
    fa = label.get("fold_axis")
    fa_line = _fold_axis_points(fa, w, h, cxy)
    if fa_line is not None:
        _line(fa_line[0], fa_line[1], (255, 0, 255), 2)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    from terafold.vision.imageio import imwrite

    imwrite(out_path, img)
    return {"rendered": True, "out": out_path, "reason": ""}


def _corner_text(img, text: str, cx: int, cy: int, r: int, color) -> None:
    """Draw a small corner label (tl/tr/br/bl) near a dot, if cv2 is available.

    Degrades to a no-op when OpenCV isn't installed — the dots still render."""
    try:
        from terafold.vision.imageio import have_cv2

        if not have_cv2():
            return
        import cv2  # type: ignore

        h, w = img.shape[0], img.shape[1]
        scale = max(0.4, r / 14.0)
        thick = max(1, r // 8)
        ox, oy = cx + r + 2, max(int(12 * scale), cy - r - 2)
        ox = min(ox, w - int(28 * scale))
        oy = min(max(oy, int(14 * scale)), h - 2)
        # dark shadow then bright text for legibility on any background
        cv2.putText(img, text, (ox + 1, oy + 1), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (0, 0, 0), thick + 1, cv2.LINE_AA)
        cv2.putText(img, text, (ox, oy), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    tuple(int(c) for c in color), thick, cv2.LINE_AA)
    except Exception:
        return


def _fold_axis_points(fa: Any, w: int, h: int, corners_xy) -> Optional[tuple]:
    if fa is None:
        return None
    if isinstance(fa, (list, tuple)) and len(fa) == 2 and all(
            isinstance(p, (list, tuple)) and len(p) == 2 for p in fa):
        return ((int(fa[0][0]), int(fa[0][1])), (int(fa[1][0]), int(fa[1][1])))
    if isinstance(fa, dict) and "p0" in fa and "p1" in fa:
        return ((int(fa["p0"][0]), int(fa["p0"][1])), (int(fa["p1"][0]), int(fa["p1"][1])))
    if isinstance(fa, str) and corners_xy is not None:
        xs = [p[0] for p in corners_xy]
        ys = [p[1] for p in corners_xy]
        cx = int(sum(xs) / 4)
        cy = int(sum(ys) / 4)
        if fa.startswith("vert"):
            return ((cx, int(min(ys))), (cx, int(max(ys))))
        if fa.startswith("horiz"):
            return ((int(min(xs)), cy), (int(max(xs)), cy))
    return None
