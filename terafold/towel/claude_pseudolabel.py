"""Pseudo-label external towel images with Claude Vision — GEOMETRY ONLY.

This turns a RAW/candidate image dataset into a *pseudo-labeled* towel dataset.
Claude is asked, per image, for STRICT JSON describing the one main towel: its
state, 4 outer corners (tl/tr/br/bl), bbox, fold axis, grasp/place edges, a
confidence, and risk reasons. Claude is **never** asked for — and its output is
never read as — robot commands, joint angles, or motor/serial signals.

Safety / hygiene (mirrors :mod:`terafold.vision.claude_labeler`):

* The Anthropic SDK is a LAZY, optional import; this module imports without it.
* The API key is read ONLY from ``ANTHROPIC_API_KEY`` — never hard-coded, never
  printed, never written to disk. A missing key fails *gracefully* with setup
  instructions (no traceback).
* Pseudo-labels are NEVER trusted: every label is validated (strict JSON, in-bounds
  coordinates, sane quad geometry) and starts ``usable_for_training=False`` with an
  ``approval_status`` of ``pending`` / ``review_needed``. Nothing reaches YOLO
  export until a human (``review-pseudolabels``) or the explicit high-confidence
  gate (``approve-high-confidence-pseudolabels``) approves it.
* Overlays + confidence are saved for every image so a human can audit quickly.
* Per-image API errors are caught and logged; ``resume=True`` skips images already
  labeled, so a long run can be restarted cheaply.
"""

from __future__ import annotations

import base64
import json
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

from terafold.data.episode_schema import read_json, write_json
from terafold.towel.dataset import dataset_paths
from terafold.towel.overlay import draw_towel_overlay
from terafold.towel.raw_ingest import load_input_images
from terafold.towel.schema import (CORNER_ORDER, TOWEL_STATES, TRAINABLE_STATES,
                                   image_size, quad_geometry_error, quad_is_simple,
                                   validate_label)

__all__ = [
    "DEFAULT_MODEL", "AUTO_OK_STATES", "ClaudeTowelError", "ClaudeTowelLabeler",
    "parse_and_build_pseudolabel", "claude_label_towel_folder", "KEY_HINT", "SDK_HINT",
    "LABEL_POLICIES", "LABEL_POLICY_PROMPTS", "DEFAULT_LABEL_POLICY", "corners_look_like_bbox",
    "STRICT_OUTER_POLICIES", "STRIPED_TOWEL_POLICIES", "label_image_folder",
]

DEFAULT_MODEL = "claude-opus-4-8"

# Flat-ish trainable states a corner-pose label makes sense for. folded_success is
# trainable in the schema but the spec routes it to human review, so it's excluded.
AUTO_OK_STATES = ["flat_unfolded", "wrinkled_unfolded", "partially_folded"]

KEY_HINT = (
    "ANTHROPIC_API_KEY is not set. Claude pseudo-labeling needs an Anthropic API key:\n"
    "  export ANTHROPIC_API_KEY=...    # get one at https://console.anthropic.com/\n"
    "The key is read only from the environment — it is never stored or printed."
)
SDK_HINT = "The Anthropic SDK is required. Install it:\n  python3 -m pip install anthropic"

SYSTEM_PROMPT = (
    "You are a vision labeling assistant for a cloth-folding robot. You ONLY label "
    "image geometry for ONE towel. You must NEVER output robot commands, joint "
    "angles, motor/serial/PWM signals, or control instructions — only pixel "
    "coordinates and the requested fields. Respond with STRICT JSON only: no prose, "
    "no markdown, no code fences."
)

USER_PROMPT = """Label the ONE main towel in this image. Image is {w} px wide by {h} px tall; origin is top-left, x=column (0..{w}), y=row (0..{h}).

Guidance:
- Label only ONE main towel — the largest/clearest one. If there are several distinct towels, use state "multiple_towels".
- Prefer a flat or slightly wrinkled white / off-white hotel/bath towel lying on a surface.
- If it is a folded stack, decorative/animal towel sculpture, rolled, hanging, worn, or you cannot tell it is a towel, set usable_for_training=false and the appropriate state ("not_towel", "folded_success", or "bad_view").
- "corners" are the 4 OUTER boundary corners of the towel as it lies on the surface, in this order: tl (top-left), tr (top-right), br (bottom-right), bl (bottom-left). Use the visible outer edge.
- If corners are occluded or you are uncertain, give a LOW confidence and say so in risk_reasons — do not guess precise corners.

Return ONLY this JSON object:
{{
  "state": "flat_unfolded | wrinkled_unfolded | partially_folded | folded_success | bad_view | multiple_towels | not_towel",
  "usable_for_training": true,
  "bbox": [x1, y1, x2, y2],
  "corners": {{"tl": [x, y], "tr": [x, y], "br": [x, y], "bl": [x, y]}},
  "visible": {{"tl": true, "tr": true, "br": true, "bl": true}},
  "fold_axis": "vertical | horizontal | null",
  "grasp_edge": "top | bottom | left | right | null",
  "place_edge": "top | bottom | left | right | null",
  "confidence": 0.0,
  "risk_reasons": ["short strings"],
  "notes": "one short sentence"
}}"""

# Stronger policy: label the 4 VISIBLE outer corners of the towel's CURRENT shape
# with pixel-level accuracy (NOT a loose bounding box), accepting folded states.
OUTER_VISIBLE_PROMPT = """Label the 4 VISIBLE OUTER CORNERS of the ONE main towel's CURRENT visible outer shape, with PIXEL-LEVEL accuracy. Image is {w} px wide by {h} px tall; origin is top-left, x=column (0..{w}), y=row (0..{h}).

MAIN RULE: place each keypoint exactly on a real outer corner of the towel's visible silhouette.
- Do NOT label the bounding box. The 4 points must trace the towel's true outline, not an axis-aligned rectangle around it.
- Do NOT place any corner on the background, table, or outside the towel.
- Do NOT guess hidden ORIGINAL corners of an unfolded towel — label the corners of what is VISIBLE now.
- Do NOT reject just because the towel is folded.
- Folded towel / folded stack: label the 4 outer corners of the visible folded rectangle/square (state "folded_success").
- Partially folded towel: label the 4 outer corners of the visible outer silhouette (state "partially_folded").
- Flat / wrinkled towel: label its 4 outer corners (state "flat_unfolded" / "wrinkled_unfolded").
- Rounded corner: place the keypoint at the visual intersection of the two outer edge directions (where the straight edges would meet).
- If a corner is hidden, cut off by the frame, covered by a hand/body, or otherwise impossible to place precisely: set that corner's "visible" flag to false AND lower the overall confidence.

Corner order (in image coordinates):
- tl = the visible corner that is top-left in the image
- tr = top-right visible corner
- br = bottom-right visible corner
- bl = bottom-left visible corner

QUALITY SELF-CHECK — before you finalize, verify ALL of these (if any fails, fix the points or lower confidence):
1. Each point sits on a real towel corner, not on the background.
2. The 4 points form the actual visible towel outline (a quadrilateral that hugs the towel).
3. The points are NOT just the bounding box of the towel.
4. Each edge of the quad tightly follows the corresponding towel edge.
5. If zooming into a corner would move where you'd place the point, set a LOW confidence.

Use state "multiple_towels", "bad_view", or "not_towel" (with usable_for_training=false) if there isn't one clear towel with a labelable visible outline.

Return ONLY this JSON object:
{{
  "state": "flat_unfolded | wrinkled_unfolded | partially_folded | folded_success | bad_view | multiple_towels | not_towel",
  "usable_for_training": true,
  "bbox": [x1, y1, x2, y2],
  "corners": {{"tl": [x, y], "tr": [x, y], "br": [x, y], "bl": [x, y]}},
  "visible": {{"tl": true, "tr": true, "br": true, "bl": true}},
  "fold_axis": "vertical | horizontal | null",
  "grasp_edge": "top | bottom | left | right | null",
  "place_edge": "top | bottom | left | right | null",
  "confidence": 0.0,
  "risk_reasons": ["short strings"],
  "notes": "one short sentence"
}}"""

# Crop-first policy: the image is a TIGHT CROP around ONE beige/white striped towel
# (produced by ``crop-real-towel-candidates``). Label the 4 visible outer corners of
# THAT towel only, and reject if the crop is not actually the striped towel.
STRIPED_TOWEL_PROMPT = """Label the 4 VISIBLE OUTER CORNERS of the ONE beige/white striped towel in this image, with PIXEL-LEVEL accuracy. Image is {w} px wide by {h} px tall; origin is top-left, x=column (0..{w}), y=row (0..{h}).

The target object is the beige/white striped towel. Do not label the bed sheet, pillow, mattress, blanket, or room rectangle.

This image is a tight crop around ONE towel, but a little of the surface around it may still be visible. Label ONLY the striped towel.

MAIN RULE: place each keypoint exactly on a real outer corner of the striped towel's visible silhouette.
- Do NOT label the bounding box. The 4 points must trace the towel's true outline, not an axis-aligned rectangle around it.
- Do NOT place any corner on the bed sheet, pillow, blanket, headboard, floor, or any background — only on the striped towel.
- Do NOT guess hidden ORIGINAL corners of an unfolded towel — label the corners of what is VISIBLE now.
- Do NOT reject just because the towel is folded.
- Folded towel: label the 4 outer corners of the visible folded rectangle (state "folded_success").
- Partially folded towel: label the 4 outer corners of the visible outer silhouette (state "partially_folded").
- Flat / wrinkled towel: label its 4 outer corners (state "flat_unfolded" / "wrinkled_unfolded").
- Rounded corner: place the keypoint at the visual intersection of the two outer edge directions (where the straight edges would meet).
- If a corner is hidden, cut off by the frame, or covered: set that corner's "visible" flag to false AND lower confidence.

REJECT RULE: set is_target=false, usable_for_training=false, and state "not_towel" if the main object in this crop is NOT the beige/white striped towel (e.g. it is the bed sheet, a pillow, a blanket, the headboard, the floor, or a plain room rectangle). Use "multiple_towels" if there are several distinct towels, or "bad_view" if you cannot place corners.

Corner order (in image coordinates):
- tl = the visible corner that is top-left in the image
- tr = top-right visible corner
- br = bottom-right visible corner
- bl = bottom-left visible corner

QUALITY SELF-CHECK — before you finalize, verify ALL of these (if any fails, fix the points or lower confidence):
1. Each point sits on a real corner of the STRIPED TOWEL, not on the bed/background.
2. The 4 points form the actual visible towel outline (a quadrilateral that hugs the towel).
3. The points are NOT just the bounding box of the towel.
4. Each edge of the quad tightly follows the corresponding towel edge.
5. The object you labeled really is the beige/white striped towel (else set is_target=false).

Return ONLY this JSON object:
{{
  "is_target": true,
  "state": "flat_unfolded | wrinkled_unfolded | partially_folded | folded_success | bad_view | multiple_towels | not_towel",
  "usable_for_training": true,
  "bbox": [x1, y1, x2, y2],
  "corners": {{"tl": [x, y], "tr": [x, y], "br": [x, y], "bl": [x, y]}},
  "visible": {{"tl": true, "tr": true, "br": true, "bl": true}},
  "fold_axis": "vertical | horizontal | null",
  "grasp_edge": "top | bottom | left | right | null",
  "place_edge": "top | bottom | left | right | null",
  "confidence": 0.0,
  "risk_reasons": ["short strings"],
  "notes": "one short sentence"
}}"""

DEFAULT_LABEL_POLICY = "default"
LABEL_POLICY_PROMPTS = {
    "default": USER_PROMPT,
    "outer_visible_corners_any_state": OUTER_VISIBLE_PROMPT,
    "outer_visible_corners_striped_towel_only": STRIPED_TOWEL_PROMPT,
}
LABEL_POLICIES = tuple(LABEL_POLICY_PROMPTS)

# Policies that demand a pixel-accurate visible OUTER quad (not a loose bbox). For
# these the "corners are just the axis-aligned bbox" lazy-answer flag applies.
STRICT_OUTER_POLICIES = (
    "outer_visible_corners_any_state",
    "outer_visible_corners_striped_towel_only",
    "striped_towel_visible_outer_corners",      # OpenAI labeler's striped-towel policy
)

# Policies that label ONLY the beige/white striped towel: an explicit "not the
# striped towel" answer rejects (state -> not_towel). Shared by the Claude crop-first
# policy and the OpenAI second-opinion labeler.
STRIPED_TOWEL_POLICIES = (
    "outer_visible_corners_striped_towel_only",
    "striped_towel_visible_outer_corners",
)

_MEDIA_TYPES = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
    ".webp": "image/webp", ".gif": "image/gif",
}


class ClaudeTowelError(ValueError):
    """Raised when Claude's response is missing, empty, or not valid JSON."""


def _sid_minter(reserved=None) -> Callable[[], str]:
    """Return a callable that mints zero-padded ids ('000001', '000002', ...),
    skipping any id in ``reserved`` (and any it has already handed out).

    Used so a ``--resume`` run gives NEW images a fresh id that can never collide
    with an id a resumed sample still owns — otherwise the new image's
    ``images/{sid}`` / ``labels/{sid}.json`` would silently overwrite the resumed
    sample's files. With an empty ``reserved`` it yields 000001, 000002, ... exactly
    like the old positional scheme, so non-resume runs are unchanged.
    """
    used = set(reserved or ())
    counter = [0]

    def _next() -> str:
        while True:
            counter[0] += 1
            sid = f"{counter[0]:06d}"
            if sid not in used:
                used.add(sid)
                return sid

    return _next


# --------------------------------------------------------------------------
# Transport (lazy SDK)
# --------------------------------------------------------------------------


class ClaudeTowelLabeler:
    """Get one strict-JSON towel label from Claude Vision for one image."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        client: Any = None,
        responder: Optional[Callable[[str, int, int], str]] = None,
        max_tokens: int = 1024,
        label_policy: str = DEFAULT_LABEL_POLICY,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.client = client
        # responder(image_path, w, h) -> raw text. Test / custom-transport seam.
        self._responder = responder
        self.max_tokens = int(max_tokens)
        self.label_policy = label_policy if label_policy in LABEL_POLICY_PROMPTS else DEFAULT_LABEL_POLICY
        self.user_prompt = LABEL_POLICY_PROMPTS[self.label_policy]
        self.last_usage: Dict[str, int] = {"input_tokens": 0, "output_tokens": 0}

    def have_transport(self) -> bool:
        """True if we can actually obtain a response (responder, client, or key)."""
        if self._responder is not None or self.client is not None:
            return True
        return bool(self.api_key or os.environ.get("ANTHROPIC_API_KEY"))

    def raw_response(self, image_path: str, w: int, h: int) -> str:
        self.last_usage = {"input_tokens": 0, "output_tokens": 0}
        if self._responder is not None:
            return self._responder(image_path, w, h)
        return self._call_api(image_path, w, h)

    def _call_api(self, image_path: str, w: int, h: int) -> str:
        api_key = self.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if self.client is None and not api_key:
            raise ClaudeTowelError(KEY_HINT)
        try:
            import anthropic
        except Exception as exc:  # pragma: no cover - exercised only without SDK
            raise ClaudeTowelError(SDK_HINT) from exc

        media_type, b64 = _encode_image(image_path)
        client = self.client or anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image",
                     "source": {"type": "base64", "media_type": media_type, "data": b64}},
                    {"type": "text", "text": self.user_prompt.format(w=int(w), h=int(h))},
                ],
            }],
        )
        self.last_usage = _usage(message)
        return _extract_text(message)


def _encode_image(image_path: str) -> Tuple[str, str]:
    """Return ``(media_type, base64)`` for the Anthropic image block.

    For jpeg/png/webp/gif the original bytes are sent as-is (no decode → no
    cv2/Pillow needed). Other formats (e.g. bmp) are decoded + re-encoded to PNG.
    """
    ext = os.path.splitext(image_path)[1].lower()
    media = _MEDIA_TYPES.get(ext)
    if media is not None:
        with open(image_path, "rb") as f:
            data = f.read()
        return media, base64.standard_b64encode(data).decode("utf-8")
    # Fallback: decode then PNG-encode (needs an image backend).
    from terafold.vision.imageio import _png_encode, ensure_uint8_rgb, imread

    png = _png_encode(ensure_uint8_rgb(imread(image_path)))
    return "image/png", base64.standard_b64encode(png).decode("utf-8")


def _usage(message: Any) -> Dict[str, int]:
    u = getattr(message, "usage", None)
    if u is None and isinstance(message, dict):
        u = message.get("usage")
    if u is None:
        return {"input_tokens": 0, "output_tokens": 0}
    get = (lambda k: getattr(u, k, 0)) if not isinstance(u, dict) else (lambda k: u.get(k, 0))
    return {"input_tokens": int(get("input_tokens") or 0),
            "output_tokens": int(get("output_tokens") or 0)}


def _extract_text(message: Any) -> str:
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
    """Strictly extract a single JSON object from a vision model's response text.

    Provider-neutral: this helper is shared by the Claude and OpenAI labelers, so its
    error messages never name a specific provider (the caller adds provider context)."""
    if not raw or not raw.strip():
        raise ClaudeTowelError("Empty response (no text) from the vision model.")
    text = raw.strip()
    if "```" in text:  # strip code fences if the model added them anyway
        for seg in text.split("```"):
            seg = seg.strip()
            if seg.startswith("json"):
                seg = seg[4:].strip()
            if seg.startswith("{"):
                text = seg
                break
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ClaudeTowelError(f"No JSON object found in the model response: {raw[:120]!r}")
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError as exc:
        raise ClaudeTowelError(f"Malformed JSON in the model response: {exc}") from exc
    if not isinstance(data, dict):
        raise ClaudeTowelError("The model response JSON is not an object.")
    return data


# --------------------------------------------------------------------------
# Build + validate a pseudo-label from the parsed JSON
# --------------------------------------------------------------------------


def _as_bool(v: Any, default: bool = True) -> bool:
    """Coerce a model-supplied truthiness value robustly.

    Handles real JSON booleans, ``None`` (-> ``default``), numbers, and off-spec
    STRINGED booleans like ``"false"`` / ``"no"`` / ``"0"`` (which ``bool()`` would
    wrongly treat as True, silently skipping the striped-towel reject rule)."""
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() not in ("false", "no", "0", "off", "n", "none", "null", "")
    return bool(v)


def _num_pair(v: Any) -> Optional[List[float]]:
    if isinstance(v, (list, tuple)) and len(v) == 2:
        try:
            return [float(v[0]), float(v[1])]
        except (TypeError, ValueError):
            return None
    return None


def _in_bounds(p: List[float], w: int, h: int, margin: float) -> bool:
    return (-margin * w <= p[0] <= (w - 1) + margin * w
            and -margin * h <= p[1] <= (h - 1) + margin * h)


def corners_look_like_bbox(corners_xy: List[List[float]], w: int, h: int,
                           tol_frac: float = 0.02) -> bool:
    """True if the 4 corners are (nearly) the axis-aligned bounding box — the
    signature of a lazy "I just labeled the bbox" answer rather than the towel's
    true outline. Tolerance scales with the image diagonal. A genuinely rotated /
    skewed towel quad will NOT match its axis-aligned bbox, so this is a useful
    (soft) flag — though a perfectly top-down axis-aligned towel can also match."""
    from terafold.towel.schema import bbox_from_corners

    if corners_xy is None or len(corners_xy) != 4:
        return False
    x1, y1, x2, y2 = bbox_from_corners(corners_xy)
    expected = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]  # tl, tr, br, bl
    tol = tol_frac * ((w ** 2 + h ** 2) ** 0.5)
    return all(((c[0] - e[0]) ** 2 + (c[1] - e[1]) ** 2) ** 0.5 <= tol
               for c, e in zip(corners_xy, expected))


def parse_and_build_pseudolabel(
    raw: str,
    image_rel: str,
    w: int,
    h: int,
    model: str = DEFAULT_MODEL,
    min_confidence: float = 0.75,
    bounds_margin: float = 0.06,
    label_policy: str = DEFAULT_LABEL_POLICY,
    backend: str = "claude",
) -> Dict[str, Any]:
    """Parse a vision model's raw text and build a validated pseudo-label dict.

    Shared by the Claude and OpenAI labelers (``backend`` selects only the
    provenance fields ``source`` / ``label_method`` / model key — the geometry,
    validation, and risk logic are identical). Field-name aliases are accepted so
    either model's JSON works: ``risks`` ⇆ ``risk_reasons`` and
    ``is_target_towel_visible`` ⇆ ``is_target``.

    Raises :class:`ClaudeTowelError` only for *unparseable* JSON. Everything else
    (bad state, missing/out-of-bounds corners, impossible geometry, low confidence)
    is recorded in ``risk_reasons`` and routed to ``approval_status='review_needed'``
    — never silently trusted, never auto-usable.
    """
    data = _parse_json(raw)  # may raise ClaudeTowelError
    risk: List[str] = []
    risk.extend(str(r) for r in (data.get("risk_reasons") or data.get("risks") or []) if str(r).strip())

    # -- state --
    state = data.get("state")
    if state not in TOWEL_STATES:
        risk.append(f"invalid_state:{state!r}")
        state = "bad_view"
    if state in ("bad_view", "multiple_towels", "not_towel"):
        risk.append(f"non_trainable_state:{state}")
    elif state == "folded_success":
        risk.append("folded_success_needs_review")

    # -- target check (striped-towel reject rule) --
    # ``is_target`` (alias ``is_target_towel_visible``) defaults True (older prompts
    # omit it); only an explicit False means "this is NOT the striped towel" -> reject.
    raw_is_target = data.get("is_target", data.get("is_target_towel_visible", True))
    is_target = _as_bool(raw_is_target, default=True)
    # An explicit non-empty reject_reason is recorded for the reviewer.
    reject_reason = data.get("reject_reason")
    if reject_reason and str(reject_reason).strip().lower() not in ("", "none", "null", "n/a"):
        risk.append(f"reject_reason:{str(reject_reason).strip()[:80]}")
    if label_policy in STRIPED_TOWEL_POLICIES and not is_target:
        risk.append("not_striped_towel")
        if state in TRAINABLE_STATES:
            state = "not_towel"
            risk.append("non_trainable_state:not_towel")

    # -- corners (tl,tr,br,bl) --
    raw_corners = data.get("corners") or {}
    corners: Dict[str, Optional[List[float]]] = {}
    present: List[List[float]] = []
    for k in CORNER_ORDER:
        p = _num_pair(raw_corners.get(k))
        corners[k] = p
        if p is not None:
            present.append(p)
    if len(present) < 4:
        risk.append("corners_missing")

    # -- bounds check + clamp --
    margin = float(bounds_margin)
    for k, p in list(corners.items()):
        if p is None:
            continue
        if not _in_bounds(p, w, h, margin):
            risk.append("out_of_bounds")
        corners[k] = [min(max(p[0], 0.0), float(w - 1)), min(max(p[1], 0.0), float(h - 1))]

    corners_xy = [corners[k] for k in CORNER_ORDER] if len(present) == 4 else None

    # -- geometry --
    geom_err: Optional[float] = None
    if corners_xy is not None:
        if not quad_is_simple(corners_xy):
            risk.append("geometry_impossible")
        geom_err = quad_geometry_error(corners_xy)
        # Under the strict outer-corner policy, flag corners that are just the
        # axis-aligned bbox (a lazy "I labeled the box" answer) for review. Gated on
        # the policy so the default policy's behavior is unchanged.
        if label_policy in STRICT_OUTER_POLICIES and \
                corners_look_like_bbox(corners_xy, w, h):
            risk.append("corners_equal_bbox")

    # -- bbox (validate or derive) --
    bbox = _valid_bbox(data.get("bbox"), w, h)
    if bbox is None and corners_xy is not None:
        from terafold.towel.schema import bbox_from_corners

        bbox = bbox_from_corners(corners_xy)
        if data.get("bbox") is not None:
            risk.append("bbox_invalid")
    elif data.get("bbox") is not None and bbox is None:
        risk.append("bbox_invalid")

    # -- confidence --
    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    if confidence < float(min_confidence):
        risk.append(f"low_confidence<{min_confidence:g}")

    # -- visible flags --
    raw_vis = data.get("visible") or {}
    visible = {k: _as_bool(raw_vis.get(k, True), default=True) for k in CORNER_ORDER}

    review_needed = bool(risk)
    is_openai = backend == "openai"
    label: Dict[str, Any] = {
        "image": image_rel,
        "state": state,
        "bbox": bbox,
        "corners": corners,
        "visible": visible,
        "fold_axis": data.get("fold_axis"),
        "grasp_edge": data.get("grasp_edge"),
        "place_edge": data.get("place_edge"),
        "source": "openai_pseudolabel" if is_openai else "claude_pseudolabel",
        "usable_for_training": False,   # NEVER usable until approved
        "notes": str(data.get("notes", ""))[:500],
        # pseudo-label extras
        "pseudolabel": True,
        "label_backend": backend,
        "label_method": "openai_vision" if is_openai else "claude_vision",
        "model": model,
        # Back-compat: keep claude_model on Claude labels (older readers expect it).
        **({} if is_openai else {"claude_model": model}),
        "label_policy": label_policy,
        "is_target": is_target,
        "confidence": confidence,
        "geometry_error": geom_err,
        "risk_reasons": sorted(set(risk)),
        "approval_status": "review_needed" if review_needed else "pending",
        "review_needed": review_needed,
    }
    return label


def _valid_bbox(b: Any, w: int, h: int) -> Optional[List[float]]:
    if not (isinstance(b, (list, tuple)) and len(b) == 4):
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in b)
    except (TypeError, ValueError):
        return None
    if not (x2 > x1 and y2 > y1):
        return None
    x1 = min(max(x1, 0.0), float(w - 1))
    x2 = min(max(x2, 0.0), float(w - 1))
    y1 = min(max(y1, 0.0), float(h - 1))
    y2 = min(max(y2, 0.0), float(h - 1))
    if not (x2 > x1 and y2 > y1):
        return None
    return [x1, y1, x2, y2]


# --------------------------------------------------------------------------
# Folder orchestration
# --------------------------------------------------------------------------


def claude_label_towel_folder(
    input_dir: str,
    out: str,
    max_images: int = 500,
    model: str = DEFAULT_MODEL,
    min_confidence: float = 0.75,
    resume: bool = False,
    force: bool = False,
    label_policy: str = DEFAULT_LABEL_POLICY,
    api_key: Optional[str] = None,
    client: Any = None,
    responder: Optional[Callable[[str, int, int], str]] = None,
    cost_per_mtok_in: float = 3.0,
    cost_per_mtok_out: float = 15.0,
    log: Callable[[str], None] = print,
) -> Dict[str, Any]:
    """Pseudo-label a raw/candidate dataset into a towel dataset at ``out`` with Claude.

    ``label_policy`` selects the Claude prompt (``default`` keeps the original
    behavior; ``outer_visible_corners_any_state`` demands pixel-accurate visible
    outer corners and accepts folded states). ``force`` (a.k.a. relabel) ignores any
    existing labels and re-labels every image, overwriting old (possibly loose) ones.

    Returns a status dict. ``status='no_api_key'`` (with setup instructions) when no
    key/transport is available — it does NOT raise and does NOT print the key.
    """
    if label_policy not in LABEL_POLICY_PROMPTS:
        return {"status": "error", "message": f"unknown label_policy {label_policy!r}",
                "label_policies": list(LABEL_POLICIES)}
    labeler = ClaudeTowelLabeler(model=model, api_key=api_key, client=client,
                                 responder=responder, label_policy=label_policy)
    return label_image_folder(
        labeler, input_dir, out, backend="claude", key_hint=KEY_HINT, max_images=max_images,
        model=model, min_confidence=min_confidence, resume=resume, force=force,
        label_policy=label_policy, cost_per_mtok_in=cost_per_mtok_in,
        cost_per_mtok_out=cost_per_mtok_out,
        cost_note="rough estimate; verify current Anthropic pricing", log=log)


def label_image_folder(
    labeler: Any,
    input_dir: str,
    out: str,
    *,
    backend: str = "claude",
    key_hint: str = KEY_HINT,
    max_images: int = 500,
    model: str = DEFAULT_MODEL,
    min_confidence: float = 0.75,
    resume: bool = False,
    force: bool = False,
    label_policy: str = DEFAULT_LABEL_POLICY,
    cost_per_mtok_in: float = 3.0,
    cost_per_mtok_out: float = 15.0,
    cost_note: str = "rough estimate; verify current provider pricing",
    log: Callable[[str], None] = print,
) -> Dict[str, Any]:
    """Backend-agnostic folder labeling core shared by the Claude and OpenAI labelers.

    ``labeler`` is any object exposing ``raw_response(image_path, w, h) -> str`` and a
    ``last_usage`` dict (e.g. :class:`ClaudeTowelLabeler` or the OpenAI labeler). The
    geometry/validation/overlay/crop-remap and the safe pseudo-label gating are
    identical across backends; ``backend`` only switches the provenance fields.
    Never raises on a missing key/transport — returns ``status='no_api_key'`` with
    ``key_hint`` and never prints the key.
    """
    if not os.path.isdir(input_dir):
        return {"status": "error", "message": f"input not found: {input_dir}"}
    # Accept ANY supported input shape: raw/candidate manifest, a
    # create-towel-real-dataset / merged / pseudolabeled dataset (samples list), a
    # plain image folder, a crop dataset, or a dir with an images/ subfolder.
    src = load_input_images(input_dir, default_source="external")
    images = src.get("images", [])
    if not images:
        return {"status": "error",
                "message": f"no images found in {input_dir} "
                           "(expected a manifest, an images/ folder, or image files)"}

    if not labeler.have_transport():
        return {"status": "no_api_key", "instructions": key_hint.splitlines()}
    # --force / --relabel overrides resume reuse so old labels are overwritten.
    reuse = resume and not force

    paths = dataset_paths(out)
    os.makedirs(paths["images"], exist_ok=True)
    os.makedirs(paths["labels"], exist_ok=True)
    overlays_dir = os.path.join(out, "overlays")
    os.makedirs(overlays_dir, exist_ok=True)

    prior = _load_existing(out) if reuse else {}
    # Reserve every id a resumed sample still owns so a NEW image never reuses it and
    # clobbers that sample's images/{sid} / labels/{sid}.json on disk.
    _next_sid = _sid_minter({p["sample"].get("id") for p in prior.values()
                             if isinstance(p.get("sample"), dict) and p["sample"].get("id")})
    samples: List[Dict[str, Any]] = []
    counts = {"labeled": 0, "review_needed": 0, "pending": 0, "resumed": 0, "errors": 0}
    tok_in = tok_out = 0
    n_attempt = 0

    for im in images:
        if n_attempt >= max_images:
            break
        rel = im.get("image")
        if not rel:
            continue
        n_attempt += 1
        image_source = im.get("source", src.get("source", "external"))

        # Resume: reuse an existing label for the same source image (by hash).
        # --force/--relabel disables this so old labels are overwritten.
        h = im.get("hash")
        if reuse and h and h in prior:
            sample = dict(prior[h]["sample"])
            counts["resumed"] += 1
            _bump(counts, prior[h]["label"])
            samples.append(sample)
            continue

        # Assign a fresh, collision-free id only for images we actually (re)label.
        sid = _next_sid()
        ext = os.path.splitext(rel)[1].lower() or ".jpg"
        img_rel = f"images/{sid}{ext}"
        lab_rel = f"labels/{sid}.json"

        # Copy the source image into the dataset.
        try:
            import shutil
            shutil.copyfile(os.path.join(input_dir, rel), os.path.join(out, img_rel))
        except OSError as exc:
            counts["errors"] += 1
            log(f"[warn] could not copy {rel}: {exc}")
            continue

        w, hh = im.get("width"), im.get("height")
        if not (w and hh):
            size = image_size(os.path.join(out, img_rel))
            if size:
                w, hh = size
        if not (w and hh):
            counts["errors"] += 1
            log(f"[warn] unknown image size for {rel}; skipping (cannot bound-check).")
            continue

        try:
            raw = labeler.raw_response(os.path.join(out, img_rel), int(w), int(hh))
            label = parse_and_build_pseudolabel(
                raw, img_rel, int(w), int(hh), model=model, min_confidence=min_confidence,
                label_policy=label_policy, backend=backend)
        except ClaudeTowelError as exc:
            counts["errors"] += 1
            log(f"[warn] {rel}: {exc}")
            continue
        except Exception as exc:  # transport / unexpected — keep going, never crash the run
            counts["errors"] += 1
            log(f"[warn] {rel}: {backend} vision call failed: {exc}")
            continue

        tok_in += labeler.last_usage.get("input_tokens", 0)
        tok_out += labeler.last_usage.get("output_tokens", 0)

        problems = validate_label(label)
        if problems:  # should not happen, but never write a schema-invalid label
            label.setdefault("risk_reasons", []).append("schema_invalid")
            label["approval_status"] = "review_needed"
            label["review_needed"] = True
            log(f"[warn] {rel}: built label had schema problems: {problems}")

        # Carry the generation training track onto the LABEL so it survives merge
        # (which copies the label dict) and reaches export/auto-triage.
        if im.get("training_track") is not None:
            label["training_track"] = im["training_track"]
            label["category_target"] = im.get("category_target")

        # Crop-first inputs carry ``crop`` metadata: the corners were labeled in CROP
        # coordinates, so remap them back to the ORIGINAL full image and stash that on
        # the label (the crop-space corners stay canonical for the copied crop image).
        full_image_label = _attach_full_image_remap(label, im.get("crop"))

        write_json(os.path.join(out, lab_rel), label)
        overlay_rel = f"overlays/{sid}_overlay.png"
        ov = draw_towel_overlay(os.path.join(out, img_rel), label,
                                os.path.join(out, overlay_rel))

        # Overlay #2 (crop-first only): the remapped corners drawn on the full image.
        full_overlay_rel = None
        if full_image_label is not None:
            orig_path = label["full_image"].get("orig_image")
            cand = f"full_overlays/{sid}_full_overlay.png"
            if orig_path and os.path.exists(orig_path):
                fov = draw_towel_overlay(orig_path, full_image_label,
                                         os.path.join(out, cand))
                if fov.get("rendered"):
                    full_overlay_rel = cand

        counts["labeled"] += 1
        _bump(counts, label)

        from terafold.towel.schema import SOURCE_GROUP

        sample = {
            "id": sid,
            "image": img_rel,
            "label": lab_rel,
            "overlay": overlay_rel if ov.get("rendered") else None,
            "full_overlay": full_overlay_rel,
            "hash": h,
            "width": w, "height": hh,
            "source": image_source,            # provenance of the IMAGE (e.g. openai_generated)
            "source_group": SOURCE_GROUP.get(image_source, "external"),
            "image_source": image_source,
            "pseudolabel": True,               # this sample carries a machine label
            "label_backend": backend,
            "label_policy": label_policy,
            "confidence": label["confidence"],
            "approval_status": label["approval_status"],
        }
        if im.get("crop") is not None:
            sample["crop"] = im["crop"]
        # Carry through generation provenance (so OpenAI-generated images keep their
        # prompt/category/track lineage after pseudo-labeling). Unknown keys absent.
        for k in ("generation_model", "generation_prompt", "prompt_template_name",
                  "category_target", "training_track", "cost_estimate", "license"):
            if im.get(k) is not None:
                sample[k] = im[k]
        samples.append(sample)

    est_cost = (tok_in / 1e6) * cost_per_mtok_in + (tok_out / 1e6) * cost_per_mtok_out
    is_openai = backend == "openai"
    manifest = {
        "dataset": os.path.basename(os.path.normpath(out)),
        "kind": "pseudolabeled",
        "label_schema": "towel_v1",
        "states": TOWEL_STATES,
        "source": "openai_pseudolabel" if is_openai else "claude_pseudolabel",
        "pseudolabel": True,
        "backend": backend,
        "model": model,
        # Back-compat: keep claude_model on Claude datasets (older readers expect it).
        **({} if is_openai else {"claude_model": model}),
        "label_policy": label_policy,
        "prompt_name": label_policy,
        "min_confidence": float(min_confidence),
        "forced_relabel": bool(force),
        "input_dataset": os.path.basename(os.path.normpath(input_dir)),
        "input_source": src.get("source"),
        "num_images": len(samples),
        "num_review_needed": counts["review_needed"],
        "num_pending": counts["pending"],
        "num_errors": counts["errors"],
        "num_resumed": counts["resumed"],
        "trainable_states": TRAINABLE_STATES,
        "usage": {"input_tokens": tok_in, "output_tokens": tok_out},
        "cost_estimate_usd": round(est_cost, 4),
        "cost_assumption": {"per_mtok_in": cost_per_mtok_in, "per_mtok_out": cost_per_mtok_out,
                            "note": cost_note},
        "warning": "Pseudo-labels are machine-generated and may be WRONG. Review before training.",
        "samples": samples,
    }
    write_json(paths["manifest"], manifest)
    _write_readme(out, manifest)
    log(f"Pseudo-labeled {counts['labeled']} images -> {out} "
        f"({counts['review_needed']} need review, {counts['pending']} pending, "
        f"{counts['errors']} errors, {counts['resumed']} resumed)")
    if tok_in or tok_out:
        log(f"  tokens in/out: {tok_in}/{tok_out}  est cost ~${est_cost:.3f} (approx)")
    return {
        "status": "ok",
        "out": out,
        "num_images": len(samples),
        "review_needed": counts["review_needed"],
        "pending": counts["pending"],
        "errors": counts["errors"],
        "resumed": counts["resumed"],
        "usage": manifest["usage"],
        "cost_estimate_usd": manifest["cost_estimate_usd"],
        "manifest": paths["manifest"],
    }


def _attach_full_image_remap(label: Dict[str, Any], crop: Optional[Dict[str, Any]]
                             ) -> Optional[Dict[str, Any]]:
    """For a crop-first sample, remap the crop-space corners/bbox back to the ORIGINAL
    full image and stash them on ``label`` under ``crop`` / ``full_image``.

    The crop-space corners stay canonical on the label (they match the copied crop
    image). Returns a lightweight label dict whose corners/bbox are in ORIGINAL image
    coordinates — suitable for drawing the full-image overlay — or ``None`` when there
    is no crop metadata (so non-crop inputs are completely unaffected).
    """
    if not crop:
        return None
    from terafold.towel.crop_candidates import remap_bbox, remap_corners

    full_corners = remap_corners(label.get("corners") or {}, crop)
    full_bbox = remap_bbox(label.get("bbox"), crop)
    label["crop"] = crop
    label["full_image"] = {
        "orig_image": crop.get("orig_image"),
        "orig_width": crop.get("orig_width"),
        "orig_height": crop.get("orig_height"),
        "corners": full_corners,
        "bbox": full_bbox,
    }
    return {
        "corners": full_corners,
        "visible": label.get("visible"),
        "bbox": full_bbox,
        "fold_axis": label.get("fold_axis"),
    }


def _bump(counts: Dict[str, int], label: Dict[str, Any]) -> None:
    status = label.get("approval_status")
    if status == "review_needed":
        counts["review_needed"] += 1
    elif status == "pending":
        counts["pending"] += 1


def _load_existing(out: str) -> Dict[str, Any]:
    """Map ``image hash -> {sample, label}`` for an existing pseudolabel dataset."""
    man_path = dataset_paths(out)["manifest"]
    if not os.path.exists(man_path):
        return {}
    prior: Dict[str, Any] = {}
    try:
        man = read_json(man_path)
    except Exception:
        return {}
    for s in man.get("samples", []):
        h = s.get("hash")
        if not h:
            continue
        try:
            label = read_json(os.path.join(out, s["label"]))
        except Exception:
            continue
        prior[h] = {"sample": s, "label": label}
    return prior


def _write_readme(root: str, manifest: Dict[str, Any]) -> None:
    paths = dataset_paths(root)
    vision = "OpenAI Vision" if manifest.get("backend") == "openai" else "Claude Vision"
    model = manifest.get("model") or manifest.get("claude_model") or "?"
    lines = [
        f"# Pseudo-labeled towel dataset — {manifest['dataset']}",
        "",
        f"> ⚠️ These labels were generated by **{vision}** and may be WRONG. "
        "They start `usable_for_training: false` and are excluded from YOLO export "
        "until approved by `review-pseudolabels` or `approve-high-confidence-pseudolabels`.",
        "",
        f"- model: {model}",
        f"- images: {manifest['num_images']}  "
        f"(need review: {manifest['num_review_needed']}, pending: {manifest['num_pending']})",
        f"- min confidence asked: {manifest['min_confidence']}",
        f"- token usage: in {manifest['usage']['input_tokens']} / "
        f"out {manifest['usage']['output_tokens']}  (~${manifest['cost_estimate_usd']} est)",
        "",
        "## Next",
        "```bash",
        f"python3 -m terafold review-pseudolabels --dataset {root}",
        f"# or, carefully: python3 -m terafold approve-high-confidence-pseudolabels --dataset {root}",
        "```",
    ]
    with open(paths["readme"], "w") as f:
        f.write("\n".join(lines) + "\n")
