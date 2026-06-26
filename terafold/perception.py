"""Unified perception entry point for the image/live demos.

One function, :func:`perceive`, runs the requested perception mode and returns a
:class:`PerceptionResult` that is explicit about what actually happened —
including whether a fallback was used, which marker colors were missing, and a
confidence score. Both ``demo-image`` and ``demo-today`` go through here so they
report perception identically.

Modes:
  * ``markers`` — HSV colored-marker detection (configurable color->corner map).
  * ``model``   — learned keypoint model (``--checkpoint``), else classical fallback.
  * ``claude``  — Claude Vision labels image geometry (keypoints only), else fallback.

Honesty rules (Part 5 / Part 9): if marker detection fails we do NOT pretend it
worked — the perception label becomes ``marker_failed_fallback_classical`` and
``fallback_used`` is True, which the real-motion gate refuses.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from terafold.physics.cloth_state import ClothKeypoints, FoldLine, FoldState

# Perception labels that represent a *trusted* (non-fallback) detection.
TRUSTED_PERCEPTIONS = ("markers", "learned_keypoint_model", "claude")
FALLBACK_PERCEPTIONS = (
    "marker_failed_fallback_classical",
    "classical_fallback",
    "claude_failed_fallback_classical",
)

__all__ = ["PerceptionResult", "perceive", "TRUSTED_PERCEPTIONS", "FALLBACK_PERCEPTIONS"]


@dataclass
class PerceptionResult:
    fold_state: FoldState
    mode: str
    perception: str  # explicit label (see module docstring)
    confidence: float
    detected_corners: List[List[float]]
    missing_colors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    fallback_used: bool = False
    notes: str = ""

    def is_trusted(self, threshold: float) -> bool:
        """True only when a trusted detector ran above the confidence threshold."""
        return (
            not self.fallback_used
            and self.perception in TRUSTED_PERCEPTIONS
            and self.confidence >= threshold
        )

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "perception": self.perception,
            "confidence": self.confidence,
            "detected_corners": self.detected_corners,
            "missing_colors": self.missing_colors,
            "warnings": self.warnings,
            "fallback_used": self.fallback_used,
            "notes": self.notes,
        }


def _fold_state(image: np.ndarray, kp: ClothKeypoints, direction: str) -> FoldState:
    """Assemble an image-frame FoldState (keypoints + fold line + best-effort mask)."""
    h, w = np.asarray(image).shape[:2]
    fold_line = FoldLine.from_corners(kp, direction=direction)
    mask_state = None
    try:
        from terafold.vision.cloth_mask import mask_state_from_mask, segment_cloth

        mask_state = mask_state_from_mask(segment_cloth(image), image=image)
    except Exception:
        mask_state = None
    return FoldState(
        keypoints=kp, fold_line=fold_line, mask=mask_state,
        frame="image", image_shape=(int(h), int(w)),
    )


def _confidence_from_rect(kp: ClothKeypoints, lo: float, hi: float) -> float:
    rect = float(np.clip(kp.rectangularity(), 0.0, 1.0))
    return float(np.clip(lo + (hi - lo) * rect, 0.0, 1.0))


def perceive(
    image: np.ndarray,
    mode: str = "markers",
    direction: str = "right_to_left",
    checkpoint: Optional[str] = None,
    marker_map: Optional[Dict[str, str]] = None,
    debug_markers_dir: Optional[str] = None,
    claude_labeler: Any = None,
) -> PerceptionResult:
    """Run perception in ``mode`` and return an explicit :class:`PerceptionResult`."""
    image = np.asarray(image)
    mode = (mode or "markers").lower()

    if mode == "markers":
        return _perceive_markers(image, direction, marker_map, debug_markers_dir)
    if mode == "model":
        return _perceive_model(image, direction, checkpoint)
    if mode == "claude":
        return _perceive_claude(image, direction, claude_labeler)
    raise ValueError(f"unknown perception mode {mode!r}; use markers|model|claude")


def _fallback_classical(image: np.ndarray, direction: str):
    from terafold.vision.infer_keypoints import FallbackKeypointPredictor

    fs = FallbackKeypointPredictor().predict(image, direction=direction)
    return fs.keypoints


def _perceive_markers(image, direction, marker_map, debug_markers_dir) -> PerceptionResult:
    from terafold.vision.marker_detector import detect_markers_hsv

    want_masks = bool(debug_markers_dir)
    res = detect_markers_hsv(image, marker_map=marker_map, return_masks=want_masks)
    if want_masks:
        _save_debug_masks(debug_markers_dir, res)

    if res.all_corners_found and res.keypoints is not None:
        kp = res.keypoints
        return PerceptionResult(
            fold_state=_fold_state(image, kp, direction),
            mode="markers", perception="markers",
            confidence=0.92,
            detected_corners=kp.corners.tolist(),
            missing_colors=res.missing_colors,
            notes="All 4 corner markers detected.",
        )

    # Markers failed — do NOT pretend. Fall back to the classical detector.
    kp = _fallback_classical(image, direction)
    warn = (
        f"Marker detection FAILED (missing colors: {res.missing_colors or 'unknown'}). "
        "Using the classical fallback — DRY-RUN ONLY; real motion is refused."
    )
    return PerceptionResult(
        fold_state=_fold_state(image, kp, direction),
        mode="markers", perception="marker_failed_fallback_classical",
        confidence=0.25,
        detected_corners=kp.corners.tolist(),
        missing_colors=res.missing_colors,
        warnings=[warn], fallback_used=True,
        notes="Marker detection failed; classical fallback used.",
    )


def _perceive_model(image, direction, checkpoint) -> PerceptionResult:
    from terafold.vision.infer_keypoints import LEARNED_KIND, resolve_keypoint_predictor

    predictor, info = resolve_keypoint_predictor(checkpoint)
    fs = predictor.predict(image, direction=direction)
    kp = fs.keypoints
    if info["kind"] == LEARNED_KIND:
        return PerceptionResult(
            fold_state=_fold_state(image, kp, direction),
            mode="model", perception="learned_keypoint_model",
            confidence=_confidence_from_rect(kp, 0.55, 0.92),
            detected_corners=kp.corners.tolist(),
            notes=f"learned keypoint model ({checkpoint}).",
        )
    warn = (
        f"No learned model in use ({info['reason']}); classical fallback. "
        "DRY-RUN ONLY; real motion is refused."
    )
    return PerceptionResult(
        fold_state=_fold_state(image, kp, direction),
        mode="model", perception="classical_fallback",
        confidence=_confidence_from_rect(kp, 0.30, 0.55),
        detected_corners=kp.corners.tolist(),
        warnings=[warn], fallback_used=True,
        notes="classical fallback (no learned model).",
    )


def _perceive_claude(image, direction, claude_labeler) -> PerceptionResult:
    if claude_labeler is None:
        from terafold.vision.claude_labeler import ClaudeKeypointLabeler

        claude_labeler = ClaudeKeypointLabeler()
    try:
        label = claude_labeler.label(image)
        kp = label.to_keypoints()
        return PerceptionResult(
            fold_state=_fold_state(image, kp, direction),
            mode="claude", perception="claude",
            confidence=float(label.confidence),
            detected_corners=kp.corners.tolist(),
            notes=f"Claude Vision labels: {label.notes[:120]}",
        )
    except Exception as exc:
        kp = _fallback_classical(image, direction)
        return PerceptionResult(
            fold_state=_fold_state(image, kp, direction),
            mode="claude", perception="claude_failed_fallback_classical",
            confidence=0.25,
            detected_corners=kp.corners.tolist(),
            warnings=[f"Claude labeling failed ({exc}); classical fallback. DRY-RUN ONLY."],
            fallback_used=True,
            notes="Claude labeling failed; classical fallback used.",
        )


def _save_debug_masks(out_dir: str, res) -> None:
    from terafold.vision.imageio import imwrite

    os.makedirs(out_dir, exist_ok=True)
    for color, mask in res.masks.items():
        imwrite(os.path.join(out_dir, f"{color}_mask.png"), (mask > 0).astype("uint8") * 255)
