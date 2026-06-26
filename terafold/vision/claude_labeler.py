"""Claude Vision keypoint labeling — IMAGE GEOMETRY ONLY.

This is an *optional* perception fallback. Claude Vision is asked to return only
the pixel geometry of the cloth (corners, grasp/place points, fold line) as
JSON. It is **never** asked for — and the output is never interpreted as — robot
commands, joint angles, or motor/serial signals. TeraFold always computes the
fold plan and enforces safety; Claude only labels what it sees.

Safety / hygiene:
* The Anthropic SDK is a LAZY, optional import — TeraFold imports fine without it.
* The API key comes from ``ANTHROPIC_API_KEY`` (never hard-coded, never printed).
* The returned JSON is validated (pydantic), clamped to image bounds, and
  rejected on impossible geometry (degenerate / self-crossing quad, too-small
  cloth, out-of-bounds grasp/place, or too-low confidence).
"""

from __future__ import annotations

import base64
import json
import os
from typing import Any, Callable, List, Optional

import numpy as np
from pydantic import BaseModel, Field

from terafold.physics.cloth_state import ClothKeypoints

__all__ = ["ClaudeLabel", "ClaudeKeypointLabeler", "ClaudeLabelError"]

DEFAULT_MODEL = "claude-opus-4-8"

SDK_INSTALL_HINT = "The Anthropic SDK is required for --mode claude. Install it:\n  python3 -m pip install anthropic"
KEY_HINT = "ANTHROPIC_API_KEY is not set. Provide a key:\n  export ANTHROPIC_API_KEY=..."

SYSTEM_PROMPT = (
    "You are a vision labeling assistant for a cloth-folding robot. You ONLY label "
    "image geometry. You must NEVER output robot commands, joint angles, motor or "
    "serial/PWM signals, or any control instructions — only pixel coordinates. "
    "Respond with JSON only, no prose outside the JSON."
)

USER_PROMPT = """Label the towel/cloth in this top-down image. Image size is {w} wide by {h} tall (pixels), origin top-left, x=column, y=row.

Identify the towel/cloth — NOT the table, NOT a person, NOT the robot. If colored corner markers (e.g. red/blue/green/yellow stickers) are present, use them. If there are no markers, infer the visible cloth rectangle. If you are uncertain, lower the confidence. If the image does not clearly show a cloth, return a low confidence and explain in notes.

Return ONLY this JSON (no prose, no code fences):
{{
  "top_left": [u, v],
  "top_right": [u, v],
  "bottom_left": [u, v],
  "bottom_right": [u, v],
  "grasp_point": [u, v],
  "place_point": [u, v],
  "fold_line": [[u1, v1], [u2, v2]],
  "confidence": 0.0_to_1.0,
  "notes": "short explanation"
}}"""


class ClaudeLabelError(ValueError):
    """Raised when Claude's labels are missing, malformed, or geometrically impossible."""


def _pt(name: str):
    return Field(..., min_length=2, max_length=2, description=name)


class ClaudeLabel(BaseModel):
    """Validated Claude Vision label (pixel geometry only)."""

    top_left: List[float] = _pt("top_left")
    top_right: List[float] = _pt("top_right")
    bottom_left: List[float] = _pt("bottom_left")
    bottom_right: List[float] = _pt("bottom_right")
    grasp_point: List[float] = _pt("grasp_point")
    place_point: List[float] = _pt("place_point")
    fold_line: List[List[float]] = Field(..., min_length=2, max_length=2)
    confidence: float = 0.0
    notes: str = ""

    def to_keypoints(self) -> ClothKeypoints:
        return ClothKeypoints(
            top_left=self.top_left,
            top_right=self.top_right,
            bottom_right=self.bottom_right,
            bottom_left=self.bottom_left,
            grasp=self.grasp_point,
            place=self.place_point,
            fold_a=self.fold_line[0],
            fold_b=self.fold_line[1],
        )


class ClaudeKeypointLabeler:
    """Label cloth keypoints from an image via Claude Vision (geometry only)."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        client: Any = None,
        responder: Optional[Callable[[np.ndarray], str]] = None,
        min_confidence: float = 0.30,
        min_area_frac: float = 0.01,
        max_tokens: int = 1024,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.client = client
        # responder(image) -> raw text. Injection seam for tests / custom transports.
        self._responder = responder
        self.min_confidence = float(min_confidence)
        self.min_area_frac = float(min_area_frac)
        self.max_tokens = int(max_tokens)

    # -- public ---------------------------------------------------------
    def label(self, image: np.ndarray) -> ClaudeLabel:
        img = np.asarray(image)
        h, w = img.shape[:2]
        raw = self._responder(img) if self._responder is not None else self._call_api(img)
        data = _parse_json(raw)
        try:
            label = ClaudeLabel.model_validate(data)
        except Exception as exc:
            raise ClaudeLabelError(f"Claude JSON failed schema validation: {exc}") from exc
        return _validate_geometry(label, w, h, self.min_confidence, self.min_area_frac)

    # -- transport (lazy SDK) -------------------------------------------
    def _call_api(self, image: np.ndarray) -> str:
        api_key = self.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if self.client is None and not api_key:
            raise ClaudeLabelError(KEY_HINT)
        try:
            import anthropic
        except Exception as exc:  # pragma: no cover - exercised only without SDK
            raise ClaudeLabelError(SDK_INSTALL_HINT) from exc

        from terafold.vision.imageio import _png_encode, ensure_uint8_rgb

        h, w = np.asarray(image).shape[:2]
        png = _png_encode(ensure_uint8_rgb(image))
        b64 = base64.standard_b64encode(png).decode("utf-8")

        client = self.client or anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": "image/png", "data": b64},
                        },
                        {"type": "text", "text": USER_PROMPT.format(w=int(w), h=int(h))},
                    ],
                }
            ],
        )
        return _extract_text(message)


# --------------------------------------------------------------------------
# Parsing + geometric validation
# --------------------------------------------------------------------------


def _extract_text(message: Any) -> str:
    """Concatenate text from the response content blocks (object or dict shaped)."""
    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")
    parts: List[str] = []
    for block in content or []:
        btype = getattr(block, "type", None) or (block.get("type") if isinstance(block, dict) else None)
        if btype == "text":
            text = getattr(block, "text", None) or (block.get("text") if isinstance(block, dict) else None)
            if text:
                parts.append(text)
    return "\n".join(parts).strip()


def _parse_json(raw: str) -> dict:
    if not raw or not raw.strip():
        raise ClaudeLabelError("Claude returned an empty response.")
    text = raw.strip()
    if "```" in text:  # strip code fences if the model added them
        segs = text.split("```")
        for seg in segs:
            seg = seg.strip()
            if seg.startswith("json"):
                seg = seg[4:].strip()
            if seg.startswith("{"):
                text = seg
                break
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ClaudeLabelError(f"No JSON object found in Claude response: {raw[:120]!r}")
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ClaudeLabelError(f"Malformed JSON from Claude: {exc}") from exc


def _seg_intersect(p1, p2, p3, p4) -> bool:
    """True if segment p1p2 properly intersects p3p4 (used for self-crossing check)."""
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) - (b[1] - a[1]) * (c[0] - a[0])

    d1, d2 = ccw(p3, p4, p1), ccw(p3, p4, p2)
    d3, d4 = ccw(p1, p2, p3), ccw(p1, p2, p4)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def _validate_geometry(
    label: ClaudeLabel, w: int, h: int, min_conf: float, min_area_frac: float
) -> ClaudeLabel:
    from terafold.math.geometry import polygon_area

    margin_x, margin_y = 0.06 * w, 0.06 * h
    corners = {
        "top_left": label.top_left, "top_right": label.top_right,
        "bottom_left": label.bottom_left, "bottom_right": label.bottom_right,
    }
    points = {**corners, "grasp_point": label.grasp_point, "place_point": label.place_point,
              "fold_a": label.fold_line[0], "fold_b": label.fold_line[1]}

    # Reject grossly out-of-bounds points (beyond a small margin), then clamp.
    for name, p in points.items():
        if not (np.all(np.isfinite(p))):
            raise ClaudeLabelError(f"{name} is not finite")
        if not (-margin_x <= p[0] <= w - 1 + margin_x and -margin_y <= p[1] <= h - 1 + margin_y):
            raise ClaudeLabelError(f"{name} {p} is outside the image bounds ({w}x{h})")

    def clamp(p):
        return [float(np.clip(p[0], 0, w - 1)), float(np.clip(p[1], 0, h - 1))]

    label.top_left, label.top_right = clamp(label.top_left), clamp(label.top_right)
    label.bottom_left, label.bottom_right = clamp(label.bottom_left), clamp(label.bottom_right)
    label.grasp_point, label.place_point = clamp(label.grasp_point), clamp(label.place_point)
    label.fold_line = [clamp(label.fold_line[0]), clamp(label.fold_line[1])]
    label.confidence = float(np.clip(label.confidence, 0.0, 1.0))

    quad = [label.top_left, label.top_right, label.bottom_right, label.bottom_left]
    diag = float(np.hypot(w, h))

    # Corners must not be (nearly) identical.
    cs = list(corners.values())
    for i in range(4):
        for j in range(i + 1, 4):
            if float(np.linalg.norm(np.array(cs[i]) - np.array(cs[j]))) < 0.02 * diag:
                raise ClaudeLabelError("Two cloth corners are nearly identical (degenerate quad).")

    # Quad must be simple (non-self-crossing): TL-TR-BR-BL with diagonals crossing.
    if not _seg_intersect(label.top_left, label.bottom_right, label.top_right, label.bottom_left):
        raise ClaudeLabelError("Cloth quadrilateral is self-crossing / not simple.")

    # Cloth area must be a meaningful fraction of the image.
    area = polygon_area(np.array(quad))
    if area < min_area_frac * (w * h):
        raise ClaudeLabelError(
            f"Cloth area too small ({area:.0f}px < {min_area_frac:.1%} of image)."
        )

    if label.confidence < min_conf:
        raise ClaudeLabelError(
            f"Claude confidence too low ({label.confidence:.2f} < {min_conf:.2f}): {label.notes[:80]}"
        )
    return label
