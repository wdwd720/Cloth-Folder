"""Towel label schema, states, and lightweight image utilities.

The per-image label JSON is the contract the whole pipeline shares. It matches the
spec exactly (corners as {tl,tr,br,bl}, visibility, fold axis, grasp/place edges,
source, usable_for_training, notes). Image dimensions are kept in the dataset
*manifest* (not the label) so the label stays exactly as specified.

Everything here is stdlib-only (hashing + header parsing). JPEG/PNG dimensions are
read from the file header so we never need OpenCV/Pillow just to size an image.
"""

from __future__ import annotations

import hashlib
import os
import struct
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "TOWEL_STATES", "TRAINABLE_STATES", "REJECT_STATES", "CORNER_ORDER", "SOURCES",
    "SOURCE_GROUP", "SOURCE_GROUPS", "IMAGE_EXTS", "APPROVAL_STATES", "APPROVED_STATES",
    "new_label",
    "validate_label", "is_usable", "label_corners_xy", "bbox_from_corners",
    "image_size", "image_hash", "is_pseudolabel", "is_approved_pseudolabel",
    "source_group", "polygon_area_xy", "quad_is_simple", "quad_geometry_error",
]

# The 7 towel states (in the spec's order).
TOWEL_STATES: List[str] = [
    "flat_unfolded",
    "wrinkled_unfolded",
    "partially_folded",
    "folded_success",
    "bad_view",
    "multiple_towels",
    "not_towel",
]

# States that can carry a valid 4-corner pose and feed YOLO pose training.
TRAINABLE_STATES = ["flat_unfolded", "wrinkled_unfolded", "partially_folded", "folded_success"]
# States that are explicitly excluded from training export.
REJECT_STATES = ["bad_view", "multiple_towels", "not_towel"]

CORNER_ORDER = ("tl", "tr", "br", "bl")  # click/label order

# Provenance sources and the coarse group used for per-source reporting.
# Real-user photos, legacy "external" imports, optional synthetic augmentation,
# the external real-image ingest sources (kaggle/openimages/roboflow), the
# OpenAI-generated "realism bridge" layer, and the Claude Vision pseudo-label
# source (used when an image has no more specific provenance). Adding sources
# only *widens* what validates.
SOURCES = ["user_photo", "external", "synthetic", "kaggle", "openimages",
           "roboflow", "openai_generated", "claude_pseudolabel", "openai_pseudolabel"]
SOURCE_GROUP = {
    "user_photo": "real_user",
    "external": "external",
    "synthetic": "synthetic",
    "kaggle": "kaggle",
    "openimages": "openimages",
    "roboflow": "roboflow",
    "openai_generated": "openai_generated",
    "claude_pseudolabel": "claude_pseudolabel",
    "openai_pseudolabel": "openai_pseudolabel",
}

# Per-source report buckets (stable key order for export/merge summaries).
SOURCE_GROUPS = ["real_user", "external", "synthetic", "kaggle", "openimages",
                 "roboflow", "openai_generated", "claude_pseudolabel", "openai_pseudolabel"]

# Approval lifecycle for a (machine) pseudo-label. Human labels never carry one.
#   pending        — freshly produced, not yet reviewed
#   review_needed  — auto-flagged for a human (low confidence / risky state / geometry)
#   approved       — a human approved it (review-pseudolabels)
#   auto_approved  — passed the high-confidence + geometry auto-gate
#   rejected       — a human (or the gate) rejected it; never exported
APPROVAL_STATES = ["pending", "review_needed", "approved", "auto_approved", "rejected"]
# The two statuses that allow a pseudo-label to reach training export.
APPROVED_STATES = ("approved", "auto_approved")

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def new_label(image_rel: str, source: str = "user_photo", state: str = "flat_unfolded") -> Dict[str, Any]:
    """Return a fresh, *unlabeled* towel label template (matches the spec exactly)."""
    return {
        "image": image_rel,
        "state": state,
        "bbox": None,
        "corners": {"tl": None, "tr": None, "br": None, "bl": None},
        "visible": {"tl": True, "tr": True, "br": True, "bl": True},
        "fold_axis": None,
        "grasp_edge": None,
        "place_edge": None,
        "source": source,
        "usable_for_training": False,
        "notes": "",
    }


def validate_label(label: Dict[str, Any]) -> List[str]:
    """Return a list of schema problems (empty ⇒ valid)."""
    problems: List[str] = []
    if not isinstance(label, dict):
        return ["label is not a JSON object"]
    if not label.get("image"):
        problems.append("missing 'image'")
    state = label.get("state")
    if state not in TOWEL_STATES:
        problems.append(f"state {state!r} not in {TOWEL_STATES}")
    corners = label.get("corners")
    if not isinstance(corners, dict) or set(corners) != set(CORNER_ORDER):
        problems.append("corners must be an object with keys tl,tr,br,bl")
    else:
        for k, v in corners.items():
            if v is not None and not (isinstance(v, (list, tuple)) and len(v) == 2):
                problems.append(f"corner {k} must be null or [x, y]")
    vis = label.get("visible")
    if not isinstance(vis, dict) or set(vis) != set(CORNER_ORDER):
        problems.append("visible must be an object with keys tl,tr,br,bl")
    if label.get("source") not in SOURCES:
        problems.append(f"source {label.get('source')!r} not in {SOURCES}")
    if not isinstance(label.get("usable_for_training"), bool):
        problems.append("usable_for_training must be a bool")
    # Pseudo-label extras are optional, but if present they must be well-formed.
    if "approval_status" in label and label["approval_status"] not in APPROVAL_STATES:
        problems.append(f"approval_status {label['approval_status']!r} not in {APPROVAL_STATES}")
    if "confidence" in label and label["confidence"] is not None:
        c = label["confidence"]
        if not (isinstance(c, (int, float)) and 0.0 <= float(c) <= 1.0):
            problems.append("confidence must be a number in [0, 1]")
    return problems


def label_corners_xy(label: Dict[str, Any]) -> Optional[List[List[float]]]:
    """Return the 4 corners as ``[[x,y]*4]`` in tl,tr,br,bl order, or None if any missing."""
    corners = label.get("corners") or {}
    out: List[List[float]] = []
    for k in CORNER_ORDER:
        v = corners.get(k)
        if v is None or len(v) != 2:
            return None
        out.append([float(v[0]), float(v[1])])
    return out


def bbox_from_corners(corners_xy: List[List[float]]) -> List[float]:
    """Axis-aligned ``[x1, y1, x2, y2]`` enclosing the 4 corners."""
    xs = [p[0] for p in corners_xy]
    ys = [p[1] for p in corners_xy]
    return [min(xs), min(ys), max(xs), max(ys)]


def is_usable(label: Dict[str, Any]) -> bool:
    """True if the label is flagged usable AND has a trainable state with 4 corners."""
    if not label.get("usable_for_training"):
        return False
    if label.get("state") not in TRAINABLE_STATES:
        return False
    return label_corners_xy(label) is not None


# ----------------------------------------------------------------------
# Pseudo-label helpers (Claude Vision labels). A pseudo-label is a normal
# towel label with ``pseudolabel: true`` plus a confidence + approval status.
# It is NEVER usable_for_training until a human (or the auto-gate) approves it.
# ----------------------------------------------------------------------


def is_pseudolabel(label: Dict[str, Any]) -> bool:
    """True if this label was machine-generated (Claude/OpenAI Vision), not human-drawn."""
    return (bool(label.get("pseudolabel"))
            or label.get("source") in ("claude_pseudolabel", "openai_pseudolabel"))


def is_approved_pseudolabel(label: Dict[str, Any]) -> bool:
    """True if a pseudo-label has been approved (manually or by the auto-gate)."""
    return is_pseudolabel(label) and label.get("approval_status") in APPROVED_STATES


def source_group(sample: Dict[str, Any], label: Dict[str, Any]) -> str:
    """Report bucket for a sample, by *image provenance* (where the image came
    from), so generated/kaggle/openimages images stay distinguishable even after
    Claude pseudo-labeling. Whether a sample is a (machine) pseudo-label is a
    SEPARATE axis tracked by :func:`is_pseudolabel` and gated independently at
    export time — it does not collapse the provenance bucket. The
    ``claude_pseudolabel`` bucket is used only when an image has no more specific
    provenance (its source is literally ``claude_pseudolabel``)."""
    src = sample.get("source") or label.get("source") or "external"
    return SOURCE_GROUP.get(src, "external")


# ----------------------------------------------------------------------
# Pure-python quad geometry (no numpy) — shared by the Claude label
# validator and the high-confidence auto-approve geometry gate.
# ----------------------------------------------------------------------


def polygon_area_xy(pts: List[List[float]]) -> float:
    """Absolute shoelace area of a polygon given as ``[[x,y], ...]``."""
    n = len(pts)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def _ccw(a, b, c) -> float:
    return (c[1] - a[1]) * (b[0] - a[0]) - (b[1] - a[1]) * (c[0] - a[0])


def _proper_intersect(p1, p2, p3, p4) -> bool:
    """True if open segment p1p2 properly crosses open segment p3p4."""
    d1, d2 = _ccw(p3, p4, p1), _ccw(p3, p4, p2)
    d3, d4 = _ccw(p1, p2, p3), _ccw(p1, p2, p4)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def quad_is_simple(corners_xy: List[List[float]]) -> bool:
    """True if the tl,tr,br,bl quad is *simple* (non-self-intersecting): the two
    diagonals (tl-br and tr-bl) cross and the opposite edges do not."""
    if corners_xy is None or len(corners_xy) != 4:
        return False
    tl, tr, br, bl = corners_xy
    # Opposite-edge pairs must NOT cross; diagonals MUST cross for a convex order.
    if _proper_intersect(tl, tr, br, bl):  # top edge vs bottom edge
        return False
    if _proper_intersect(tr, br, bl, tl):  # right edge vs left edge
        return False
    return _proper_intersect(tl, br, tr, bl)


def quad_geometry_error(corners_xy: List[List[float]]) -> float:
    """A 0..1 'how wrong is this quad' score (0 = clean convex quad, 1 = broken).

    * self-intersecting / degenerate ⇒ 1.0 (unusable),
    * otherwise ``1 - quad_area / convex_hull_area`` (concavity), so a convex
      towel rectangle scores ~0 and a folded-over / concave outline scores high.
    """
    if corners_xy is None or len(corners_xy) != 4:
        return 1.0
    if not quad_is_simple(corners_xy):
        return 1.0
    area = polygon_area_xy(corners_xy)
    hull_area = polygon_area_xy(_convex_hull(corners_xy))
    if hull_area <= 1e-9 or area <= 1e-9:
        return 1.0
    return max(0.0, min(1.0, 1.0 - area / hull_area))


def _convex_hull(pts: List[List[float]]) -> List[List[float]]:
    """Andrew's monotone-chain convex hull of up to a handful of points."""
    pts = sorted([[float(p[0]), float(p[1])] for p in pts])
    if len(pts) <= 2:
        return pts

    def build(seq):
        out: List[List[float]] = []
        for p in seq:
            while len(out) >= 2 and _ccw(out[-2], out[-1], p) <= 0:
                out.pop()
            out.append(p)
        return out

    lower = build(pts)
    upper = build(reversed(pts))
    return lower[:-1] + upper[:-1]


# ----------------------------------------------------------------------
# Image utilities — header-only sizing + content hashing (no heavy deps).
# ----------------------------------------------------------------------


def image_hash(path: str) -> str:
    """SHA-1 of the raw file bytes (for cross-dataset de-duplication)."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _png_size(data: bytes) -> Optional[Tuple[int, int]]:
    if len(data) >= 24 and data[:8] == b"\x89PNG\r\n\x1a\n" and data[12:16] == b"IHDR":
        w, h = struct.unpack(">II", data[16:24])
        return int(w), int(h)
    return None


def _jpeg_size(data: bytes) -> Optional[Tuple[int, int]]:
    # Walk JPEG markers to a Start-Of-Frame (SOF0..SOF15 except DHT/DAC/RST/SOS).
    if data[:2] != b"\xff\xd8":
        return None
    i = 2
    n = len(data)
    while i + 9 < n:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            h = struct.unpack(">H", data[i + 5:i + 7])[0]
            w = struct.unpack(">H", data[i + 7:i + 9])[0]
            return int(w), int(h)
        if i + 4 > n:
            break
        seg_len = struct.unpack(">H", data[i + 2:i + 4])[0]
        i += 2 + seg_len
    return None


def image_size(path: str) -> Optional[Tuple[int, int]]:
    """Return ``(width, height)`` from the file header without decoding pixels.

    Supports PNG and JPEG via header parsing. Falls back to OpenCV/Pillow if present
    (for BMP/WebP/odd files). Returns ``None`` if the size cannot be determined.
    """
    try:
        with open(path, "rb") as f:
            head = f.read(2 * 1024 * 1024)  # generous: covers the JPEG SOF
    except OSError:
        return None
    ext = os.path.splitext(path)[1].lower()
    if ext == ".png" or head[:8] == b"\x89PNG\r\n\x1a\n":
        s = _png_size(head)
        if s:
            return s
    if ext in (".jpg", ".jpeg") or head[:2] == b"\xff\xd8":
        s = _jpeg_size(head)
        if s:
            return s
    # Fallback for formats we don't header-parse.
    try:
        from terafold.vision.imageio import have_cv2, have_pil

        if have_pil():
            from PIL import Image

            with Image.open(path) as im:
                return int(im.width), int(im.height)
        if have_cv2():
            import cv2

            img = cv2.imread(path)
            if img is not None:
                return int(img.shape[1]), int(img.shape[0])
    except Exception:
        return None
    return None
