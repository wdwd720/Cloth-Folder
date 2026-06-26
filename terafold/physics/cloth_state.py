"""Cloth state representations shared across perception, physics, and planning.

Three levels of fidelity, per the TeraFold design:

* Low-dimensional **keypoint** state (:class:`ClothKeypoints`): corners, center,
  edge midpoints — cheap, learnable, and the primary interface for V0.
* **Mask / polygon** state (:class:`ClothMaskState`): segmentation mask, polygon
  boundary, visible area, curvature + wrinkle proxies.
* Optional future **mesh** state (:class:`ClothMeshState`): a grid of surface
  points with rest lengths and an approximate deformation energy — scaffolding
  for a later quasi-static / differentiable cloth model.

Plus the fold-specific structures :class:`FoldLine`, :class:`GraspPlacePair`,
and the aggregate :class:`FoldState`.

All of these are pure-numpy dataclasses with JSON-friendly ``to_dict`` /
``from_dict`` so they can be serialized into episode files. None of them import
torch / opencv.

A note on frames: a :class:`FoldState` carries a ``frame`` tag (``"image"`` or
``"table"``). Geometry (reflection, crease) is valid in either frame; the
planner converts image-frame perception into table-frame metric coordinates
before computing robot motion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from terafold.math.geometry import fold_line_from_corners, midpoint, polygon_area

# Canonical heatmap channels predicted by the keypoint detector (Model 1).
# Order matters: detector output channel i corresponds to DETECTOR_KEYPOINTS[i].
DETECTOR_KEYPOINTS: List[str] = [
    "top_left",
    "top_right",
    "bottom_right",
    "bottom_left",
    "grasp",
    "place",
    "fold_a",
    "fold_b",
]

__all__ = [
    "DETECTOR_KEYPOINTS",
    "ClothKeypoints",
    "ClothMaskState",
    "ClothMeshState",
    "FoldLine",
    "GraspPlacePair",
    "FoldState",
]


def _pt(x) -> Optional[np.ndarray]:
    if x is None:
        return None
    return np.asarray(x, dtype=np.float64).reshape(-1)


def _to_list(x) -> Optional[list]:
    if x is None:
        return None
    return np.asarray(x, dtype=np.float64).tolist()


@dataclass
class ClothKeypoints:
    """Corner + derived keypoints for a (possibly non-rectangular) cloth.

    The four corners are required; ``grasp``, ``place`` and the two fold-line
    endpoints (``fold_a``, ``fold_b``) are optional because they may either be
    predicted directly by the detector or computed by the geometric planner.
    ``center`` and the four edge midpoints are *derived* properties so they
    never drift out of sync with the corners.
    """

    top_left: np.ndarray
    top_right: np.ndarray
    bottom_right: np.ndarray
    bottom_left: np.ndarray
    grasp: Optional[np.ndarray] = None
    place: Optional[np.ndarray] = None
    fold_a: Optional[np.ndarray] = None
    fold_b: Optional[np.ndarray] = None
    confidence: Optional[Dict[str, float]] = None

    def __post_init__(self) -> None:
        self.top_left = _pt(self.top_left)
        self.top_right = _pt(self.top_right)
        self.bottom_right = _pt(self.bottom_right)
        self.bottom_left = _pt(self.bottom_left)
        self.grasp = _pt(self.grasp)
        self.place = _pt(self.place)
        self.fold_a = _pt(self.fold_a)
        self.fold_b = _pt(self.fold_b)

    # -- derived geometry ----------------------------------------------
    @property
    def corners(self) -> np.ndarray:
        """(4, 2) corners in order TL, TR, BR, BL."""
        return np.stack([self.top_left, self.top_right, self.bottom_right, self.bottom_left])

    @property
    def center(self) -> np.ndarray:
        return self.corners.mean(axis=0)

    @property
    def mid_top(self) -> np.ndarray:
        return midpoint(self.top_left, self.top_right)

    @property
    def mid_right(self) -> np.ndarray:
        return midpoint(self.top_right, self.bottom_right)

    @property
    def mid_bottom(self) -> np.ndarray:
        return midpoint(self.bottom_right, self.bottom_left)

    @property
    def mid_left(self) -> np.ndarray:
        return midpoint(self.bottom_left, self.top_left)

    @property
    def edge_midpoints(self) -> Dict[str, np.ndarray]:
        return {
            "top": self.mid_top,
            "right": self.mid_right,
            "bottom": self.mid_bottom,
            "left": self.mid_left,
        }

    def width(self) -> float:
        """Mean horizontal edge length (TL->TR and BL->BR)."""
        top = np.linalg.norm(self.top_right - self.top_left)
        bottom = np.linalg.norm(self.bottom_right - self.bottom_left)
        return float(0.5 * (top + bottom))

    def height(self) -> float:
        """Mean vertical edge length (TL->BL and TR->BR)."""
        left = np.linalg.norm(self.bottom_left - self.top_left)
        right = np.linalg.norm(self.bottom_right - self.top_right)
        return float(0.5 * (left + right))

    def rectangularity(self) -> float:
        """Ratio of polygon area to bounding-box area in [0, 1] (1 == axis box).

        A proxy for "how rectangular" the detected cloth is; the planner can use
        it to decide how much to trust the rectangular prior vs. learned points.
        """
        c = self.corners
        poly = polygon_area(c)
        bbox = (c[:, 0].max() - c[:, 0].min()) * (c[:, 1].max() - c[:, 1].min())
        return float(poly / bbox) if bbox > 1e-9 else 0.0

    # -- (de)serialization ---------------------------------------------
    def to_array(self) -> np.ndarray:
        """Stack detector keypoints into ``(8, 2)`` in :data:`DETECTOR_KEYPOINTS` order.

        Missing optional points are filled with NaN.
        """
        lookup = {
            "top_left": self.top_left,
            "top_right": self.top_right,
            "bottom_right": self.bottom_right,
            "bottom_left": self.bottom_left,
            "grasp": self.grasp,
            "place": self.place,
            "fold_a": self.fold_a,
            "fold_b": self.fold_b,
        }
        out = np.full((len(DETECTOR_KEYPOINTS), 2), np.nan)
        for i, name in enumerate(DETECTOR_KEYPOINTS):
            v = lookup[name]
            if v is not None:
                out[i] = v
        return out

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "ClothKeypoints":
        """Inverse of :meth:`to_array` (``(8, 2)`` in canonical order)."""
        arr = np.asarray(arr, dtype=np.float64)
        if arr.shape != (len(DETECTOR_KEYPOINTS), 2):
            raise ValueError(f"expected {(len(DETECTOR_KEYPOINTS), 2)}, got {arr.shape}")
        d = {name: arr[i] for i, name in enumerate(DETECTOR_KEYPOINTS)}

        def maybe(v):
            return None if np.any(np.isnan(v)) else v

        return cls(
            top_left=d["top_left"],
            top_right=d["top_right"],
            bottom_right=d["bottom_right"],
            bottom_left=d["bottom_left"],
            grasp=maybe(d["grasp"]),
            place=maybe(d["place"]),
            fold_a=maybe(d["fold_a"]),
            fold_b=maybe(d["fold_b"]),
        )

    def to_dict(self) -> dict:
        return {
            "top_left": _to_list(self.top_left),
            "top_right": _to_list(self.top_right),
            "bottom_right": _to_list(self.bottom_right),
            "bottom_left": _to_list(self.bottom_left),
            "grasp": _to_list(self.grasp),
            "place": _to_list(self.place),
            "fold_a": _to_list(self.fold_a),
            "fold_b": _to_list(self.fold_b),
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ClothKeypoints":
        return cls(
            top_left=d["top_left"],
            top_right=d["top_right"],
            bottom_right=d["bottom_right"],
            bottom_left=d["bottom_left"],
            grasp=d.get("grasp"),
            place=d.get("place"),
            fold_a=d.get("fold_a"),
            fold_b=d.get("fold_b"),
            confidence=d.get("confidence"),
        )


@dataclass
class FoldLine:
    """A crease line in 2D, stored as point + unit direction with endpoints."""

    point: np.ndarray
    direction: np.ndarray
    a: Optional[np.ndarray] = None  # endpoint for drawing
    b: Optional[np.ndarray] = None  # endpoint for drawing

    def __post_init__(self) -> None:
        self.point = _pt(self.point)
        d = _pt(self.direction)
        n = float(np.linalg.norm(d))
        if n < 1e-12:
            raise ValueError("FoldLine direction must be non-zero")
        self.direction = d / n
        self.a = _pt(self.a)
        self.b = _pt(self.b)

    @classmethod
    def from_corners(cls, kp: "ClothKeypoints", direction: str = "right_to_left") -> "FoldLine":
        point, dvec = fold_line_from_corners(
            kp.top_left, kp.top_right, kp.bottom_left, kp.bottom_right, direction=direction
        )
        # Endpoints for visualization: extend across the cloth bbox.
        c = kp.corners
        diag = float(np.linalg.norm(c.max(axis=0) - c.min(axis=0)))
        a = point - dvec * diag * 0.5
        b = point + dvec * diag * 0.5
        return cls(point=point, direction=dvec, a=a, b=b)

    def to_dict(self) -> dict:
        return {
            "point": _to_list(self.point),
            "direction": _to_list(self.direction),
            "a": _to_list(self.a),
            "b": _to_list(self.b),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FoldLine":
        return cls(point=d["point"], direction=d["direction"], a=d.get("a"), b=d.get("b"))


@dataclass
class GraspPlacePair:
    """A grasp point and its target place point, with optional heights/scores."""

    grasp: np.ndarray
    place: np.ndarray
    grasp_z: float = 0.0
    place_z: float = 0.0
    confidence: float = 1.0
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.grasp = _pt(self.grasp)
        self.place = _pt(self.place)

    def to_dict(self) -> dict:
        return {
            "grasp": _to_list(self.grasp),
            "place": _to_list(self.place),
            "grasp_z": self.grasp_z,
            "place_z": self.place_z,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GraspPlacePair":
        return cls(
            grasp=d["grasp"],
            place=d["place"],
            grasp_z=float(d.get("grasp_z", 0.0)),
            place_z=float(d.get("place_z", 0.0)),
            confidence=float(d.get("confidence", 1.0)),
            metadata=d.get("metadata", {}),
        )


@dataclass
class ClothMaskState:
    """Segmentation mask + polygon boundary and derived shape descriptors."""

    polygon: Optional[np.ndarray] = None  # (N, 2)
    mask_shape: Optional[tuple] = None  # (H, W) of the mask it came from
    visible_area_px: Optional[float] = None
    bbox: Optional[np.ndarray] = None  # (x_min, y_min, x_max, y_max)
    contour_curvature: Optional[float] = None  # mean abs turning proxy
    wrinkle_proxy: Optional[float] = None  # 0 = flat, higher = more wrinkled
    mask: Optional[np.ndarray] = field(default=None, repr=False)  # (H, W) uint8/bool

    def __post_init__(self) -> None:
        if self.polygon is not None:
            self.polygon = np.asarray(self.polygon, dtype=np.float64)
        if self.mask is not None:
            self.mask = np.asarray(self.mask)
            if self.mask_shape is None:
                self.mask_shape = tuple(self.mask.shape[:2])

    def area(self) -> float:
        """Best available visible area estimate (mask pixels or polygon area)."""
        if self.visible_area_px is not None:
            return float(self.visible_area_px)
        if self.mask is not None:
            return float((self.mask > 0).sum())
        if self.polygon is not None and self.polygon.shape[0] >= 3:
            return polygon_area(self.polygon)
        return 0.0

    def to_dict(self) -> dict:
        # Note: the dense mask is intentionally NOT serialized into JSON; it is
        # stored alongside as an image / .npy by the recorder when needed.
        return {
            "polygon": _to_list(self.polygon),
            "mask_shape": list(self.mask_shape) if self.mask_shape else None,
            "visible_area_px": self.visible_area_px,
            "bbox": _to_list(self.bbox),
            "contour_curvature": self.contour_curvature,
            "wrinkle_proxy": self.wrinkle_proxy,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ClothMaskState":
        return cls(
            polygon=d.get("polygon"),
            mask_shape=tuple(d["mask_shape"]) if d.get("mask_shape") else None,
            visible_area_px=d.get("visible_area_px"),
            bbox=d.get("bbox"),
            contour_curvature=d.get("contour_curvature"),
            wrinkle_proxy=d.get("wrinkle_proxy"),
        )


@dataclass
class ClothMeshState:
    """Optional future mesh state: a grid of surface points + rest lengths.

    Not used by the V0 planner, but defined so a quasi-static / differentiable
    cloth model can be slotted in without changing downstream interfaces.
    """

    grid: np.ndarray  # (R, C, 2 or 3)
    rest_lengths: Optional[np.ndarray] = None  # (R, C, 2) horiz/vert rest lengths
    stiffness: float = 1.0

    def __post_init__(self) -> None:
        self.grid = np.asarray(self.grid, dtype=np.float64)

    def deformation_energy_proxy(self) -> float:
        """Sum of squared edge-length deviations from rest (0 if no rest lengths)."""
        if self.rest_lengths is None:
            return 0.0
        g = self.grid
        energy = 0.0
        # Horizontal edges.
        if g.shape[1] > 1:
            dx = np.linalg.norm(g[:, 1:] - g[:, :-1], axis=-1)
            energy += float(((dx - self.rest_lengths[:, :-1, 0]) ** 2).sum())
        # Vertical edges.
        if g.shape[0] > 1:
            dy = np.linalg.norm(g[1:, :] - g[:-1, :], axis=-1)
            energy += float(((dy - self.rest_lengths[:-1, :, 1]) ** 2).sum())
        return self.stiffness * energy


@dataclass
class FoldState:
    """Aggregate cloth state for one observation: keypoints + mask + fold line."""

    keypoints: ClothKeypoints
    fold_line: Optional[FoldLine] = None
    mask: Optional[ClothMaskState] = None
    grasp_place: Optional[GraspPlacePair] = None
    frame: str = "image"  # "image" or "table"
    image_shape: Optional[tuple] = None  # (H, W)
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "frame": self.frame,
            "image_shape": list(self.image_shape) if self.image_shape else None,
            "keypoints": self.keypoints.to_dict(),
            "fold_line": self.fold_line.to_dict() if self.fold_line else None,
            "mask": self.mask.to_dict() if self.mask else None,
            "grasp_place": self.grasp_place.to_dict() if self.grasp_place else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FoldState":
        return cls(
            keypoints=ClothKeypoints.from_dict(d["keypoints"]),
            fold_line=FoldLine.from_dict(d["fold_line"]) if d.get("fold_line") else None,
            mask=ClothMaskState.from_dict(d["mask"]) if d.get("mask") else None,
            grasp_place=GraspPlacePair.from_dict(d["grasp_place"])
            if d.get("grasp_place")
            else None,
            frame=d.get("frame", "image"),
            image_shape=tuple(d["image_shape"]) if d.get("image_shape") else None,
            metadata=d.get("metadata", {}),
        )
