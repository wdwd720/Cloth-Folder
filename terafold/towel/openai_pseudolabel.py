"""Pseudo-label towel images with OpenAI vision (Responses API) — GEOMETRY ONLY.

A SECOND-OPINION labeler alongside :mod:`terafold.towel.claude_pseudolabel`, for
when Claude's corner labels drift onto the bed / headboard / crop border instead of
the actual beige/white striped towel. It shares the EXACT same towel label schema,
validation, geometry checks, overlay style, and human-review gating — only the
vision backend differs (OpenAI's Responses API with image input instead of Claude).

Safety / hygiene mirror :mod:`terafold.towel.claude_pseudolabel`:

* the OpenAI SDK is a LAZY, optional import (this module imports without it);
* the API key is read ONLY from ``OPENAI_API_KEY`` — never hard-coded, never
  printed, never written to disk. A missing key fails *gracefully* with setup
  instructions (no traceback, no key leak);
* the model is asked ONLY for image geometry (pixel coordinates + the requested
  fields) — NEVER robot commands, joint angles, or motor/serial signals;
* every label starts ``usable_for_training=False`` with ``approval_status``
  ``pending`` / ``review_needed`` — nothing reaches YOLO export until a human (or
  the explicit high-confidence gate) approves it;
* per-image API errors are caught and logged so a long run never crashes.

Tests drive this through the ``responder`` (canned JSON text) and ``client`` (a fake
Responses client) injection seams and never make real paid calls.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional

from terafold.towel.claude_pseudolabel import (STRIPED_TOWEL_POLICIES, ClaudeTowelError,
                                               SYSTEM_PROMPT, _encode_image,
                                               label_image_folder)

__all__ = [
    "DEFAULT_OPENAI_MODEL", "OPENAI_LABEL_POLICY", "OPENAI_KEY_HINT", "OPENAI_SDK_HINT",
    "OPENAI_SYSTEM_PROMPT", "OPENAI_USER_PROMPT", "OpenAITowelError",
    "OpenAITowelLabeler", "openai_label_towel_folder",
]

DEFAULT_OPENAI_MODEL = "gpt-5.5"
OPENAI_LABEL_POLICY = "striped_towel_visible_outer_corners"

OPENAI_KEY_HINT = (
    "OPENAI_API_KEY is not set. OpenAI towel labeling needs an OpenAI API key:\n"
    "  export OPENAI_API_KEY=...        # https://platform.openai.com/api-keys\n"
    "The key is read only from the environment — it is never stored or printed."
)
OPENAI_SDK_HINT = "The OpenAI SDK is required:\n  python3 -m pip install openai"

# Geometry-only safety framing is provider-agnostic; reuse the shared system prompt.
OPENAI_SYSTEM_PROMPT = SYSTEM_PROMPT

# One striped-towel labeling prompt. Pins the target to the beige/white striped towel,
# lists the distractors to ignore, gives the visible-outer-corner rule, and demands
# the exact JSON shape the spec asks for.
OPENAI_USER_PROMPT = """Label the ONE beige/white striped towel in this image. Image is {w} px wide by {h} px tall; origin is top-left, x=column (0..{w}), y=row (0..{h}).

The target object is the beige/white striped towel. Label ONLY that towel.
IGNORE and never label any of these (they are NOT the target):
  - the bed sheet
  - the mattress cover
  - pillows
  - blankets / duvets / comforters
  - the headboard
  - the floor
  - any body part / feet / hands

CORNER RULE:
- Label the 4 visible outer corners of the towel's CURRENT visible outer shape.
- Do NOT label the image border, crop border, bed rectangle, or bounding box.
- If folded, label the 4 outer corners of the visible folded rectangle.
- If partially folded, label the 4 outer corners of the visible towel silhouette.
- If a corner is rounded, place it at the visual intersection of the two outer towel edges.
- Corner order in image coordinates: tl = top-left, tr = top-right, br = bottom-right, bl = bottom-left.
- If a corner is hidden, cut off by the frame, or covered, set its "visible" flag to false and lower confidence.

REJECT RULE: if the beige/white striped towel is NOT clearly visible (only the bed/sheet/pillow/blanket/headboard/floor is present, or you cannot place real towel corners), set is_target_towel_visible=false, give a short reject_reason, and use state "not_towel" or "bad_view".

Return ONLY this JSON object (no prose, no markdown, no code fences):
{{
  "is_target_towel_visible": true,
  "reject_reason": null,
  "state": "flat_unfolded | wrinkled_unfolded | partially_folded | folded_success | not_towel | bad_view",
  "corners": {{"tl": [x, y], "tr": [x, y], "br": [x, y], "bl": [x, y]}},
  "visible": {{"tl": true, "tr": true, "br": true, "bl": true}},
  "confidence": 0.0,
  "risks": ["short strings"]
}}"""


class OpenAITowelError(ClaudeTowelError):
    """Raised when OpenAI's response is missing, unusable, or the key/SDK is absent.

    Subclasses :class:`ClaudeTowelError` so the shared folder-labeling core
    (which catches ``ClaudeTowelError``) handles OpenAI errors uniformly.
    """


# --------------------------------------------------------------------------
# Response helpers (tolerate SDK objects OR plain dicts, for easy mocking)
# --------------------------------------------------------------------------


def _get(obj: Any, key: str) -> Any:
    """Read ``key`` from an SDK object (attr) or a plain dict (item) — None if absent."""
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _openai_output_text(response: Any) -> str:
    """Extract assistant text from a Responses API result, robustly (req #6).

    Handles every shape the Responses API / SDK returns: the ``output_text``
    convenience accessor, a plain string, and walking ``output[].content[]`` for
    ``output_text`` / ``text`` blocks (and surfacing a ``refusal`` so it lands in the
    debug snippet rather than looking like a silent empty). Works on SDK objects and
    on plain dicts (mocks). Finding the JSON *inside* the returned text is then handled
    by :func:`_parse_json`."""
    if isinstance(response, str):
        return response.strip()
    txt = _get(response, "output_text")
    if txt:
        return str(txt).strip()

    parts: List[str] = []
    for item in (_get(response, "output") or []):
        # A reasoning item has no content text; a message item has a content list.
        content = _get(item, "content")
        if isinstance(content, str):
            parts.append(content)
            continue
        for block in (content or []):
            btype = _get(block, "type")
            t = _get(block, "text")
            if t is None and btype in (None, "output_text", "text"):
                t = _get(block, "value")          # some shapes nest under "value"
            if t:
                parts.append(str(t))
            else:
                refusal = _get(block, "refusal")  # surface refusals for debugging
                if refusal:
                    parts.append(str(refusal))
    # Last resort: some wrappers expose a top-level message/content directly.
    if not parts:
        direct = _get(response, "content")
        if isinstance(direct, str):
            parts.append(direct)
    return "\n".join(p for p in parts if p).strip()


def _openai_usage(response: Any) -> Dict[str, int]:
    u = getattr(response, "usage", None)
    if u is None and isinstance(response, dict):
        u = response.get("usage")
    if u is None:
        return {"input_tokens": 0, "output_tokens": 0}
    get = (lambda k: getattr(u, k, 0)) if not isinstance(u, dict) else (lambda k: u.get(k, 0))
    return {"input_tokens": int(get("input_tokens") or 0),
            "output_tokens": int(get("output_tokens") or 0)}


# --------------------------------------------------------------------------
# Transport (lazy SDK) — OpenAI Responses API with an image input
# --------------------------------------------------------------------------


class OpenAITowelLabeler:
    """Get one strict-JSON towel label from OpenAI Vision for one image."""

    def __init__(
        self,
        model: str = DEFAULT_OPENAI_MODEL,
        api_key: Optional[str] = None,
        client: Any = None,
        responder: Optional[Callable[[str, int, int], str]] = None,
        max_tokens: int = 1024,
        label_policy: str = OPENAI_LABEL_POLICY,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.client = client
        # responder(image_path, w, h) -> raw text. Test / custom-transport seam.
        self._responder = responder
        self.max_tokens = int(max_tokens)
        self.label_policy = label_policy
        self.user_prompt = OPENAI_USER_PROMPT
        self.last_usage: Dict[str, int] = {"input_tokens": 0, "output_tokens": 0}

    def have_transport(self) -> bool:
        """True if we can actually obtain a response (responder, client, or key)."""
        if self._responder is not None or self.client is not None:
            return True
        return bool(self.api_key or os.environ.get("OPENAI_API_KEY"))

    def raw_response(self, image_path: str, w: int, h: int) -> str:
        self.last_usage = {"input_tokens": 0, "output_tokens": 0}
        if self._responder is not None:
            return self._responder(image_path, w, h)
        return self._call_api(image_path, w, h)

    def _call_api(self, image_path: str, w: int, h: int) -> str:
        api_key = self.api_key or os.environ.get("OPENAI_API_KEY")
        if self.client is None and not api_key:
            raise OpenAITowelError(OPENAI_KEY_HINT)
        client = self.client
        if client is None:
            try:
                from openai import OpenAI
            except Exception as exc:  # pragma: no cover - exercised only without SDK
                raise OpenAITowelError(OPENAI_SDK_HINT) from exc
            client = OpenAI(api_key=api_key)

        media_type, b64 = _encode_image(image_path)
        data_url = f"data:{media_type};base64,{b64}"
        prompt = self.user_prompt.format(w=int(w), h=int(h))
        response = client.responses.create(
            model=self.model,
            instructions=OPENAI_SYSTEM_PROMPT,
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": data_url},
                ],
            }],
            max_output_tokens=self.max_tokens,
        )
        self.last_usage = _openai_usage(response)
        return _openai_output_text(response)


# --------------------------------------------------------------------------
# Folder orchestration (delegates to the shared backend-agnostic core)
# --------------------------------------------------------------------------


def openai_label_towel_folder(
    input_dir: str,
    out: str,
    max_images: int = 500,
    model: str = DEFAULT_OPENAI_MODEL,
    min_confidence: float = 0.75,
    resume: bool = False,
    force: bool = False,
    label_policy: str = OPENAI_LABEL_POLICY,
    multipass: bool = False,
    crop_padding: float = 0.08,
    verify: bool = True,
    max_crop_size: Optional[int] = 1024,
    debug: bool = False,
    target_description: str = "beige and white striped towel",
    api_key: Optional[str] = None,
    client: Any = None,
    responder: Optional[Callable[[str, int, int], str]] = None,
    stage_responder: Optional[Callable[[str, str, int, int], str]] = None,
    cost_per_mtok_in: float = 0.0,
    cost_per_mtok_out: float = 0.0,
    log: Callable[[str], None] = print,
) -> Dict[str, Any]:
    """Pseudo-label a dataset of towel images with OpenAI Vision into ``out``.

    Accepts the same input shapes as the Claude labeler (raw/candidate manifest,
    real/merged/pseudolabeled dataset, crop dataset, plain image folder). ``force``
    re-labels every image; ``--max-images`` caps the run for cheap testing.

    ``multipass=True`` switches to the multi-stage HIGH-PRECISION pipeline (localize →
    crop+zoom → corner-label → remap → verify → geometry-gate), trading more
    tokens/calls per image for tight, YOLO-ready corners in messy scenes;
    ``crop_padding`` sets the crop margin and ``verify`` toggles the verification pass.

    Returns a status dict — ``status='no_api_key'`` (with setup instructions) when no
    key/transport is available; it never raises and never prints the key.
    """
    # The OpenAI labeler always sends the striped-towel prompt, so only a striped
    # policy is valid — any other would silently disable the "not the striped towel"
    # reject gate while still asking the model is_target_towel_visible. Fail loudly.
    if label_policy not in STRIPED_TOWEL_POLICIES:
        return {"status": "error",
                "message": f"--label-policy must be one of {list(STRIPED_TOWEL_POLICIES)} "
                           f"for the OpenAI striped-towel labeler (got {label_policy!r})",
                "label_policies": list(STRIPED_TOWEL_POLICIES)}
    if multipass:
        from terafold.towel.openai_multipass import label_towel_folder_multipass

        return label_towel_folder_multipass(
            input_dir, out, target_description=target_description, max_images=max_images,
            model=model, min_confidence=min_confidence, crop_padding=crop_padding,
            max_crop_size=max_crop_size, verify=verify, label_policy=label_policy,
            resume=resume, force=force, debug=debug, api_key=api_key, client=client,
            stage_responder=stage_responder, log=log)
    labeler = OpenAITowelLabeler(model=model, api_key=api_key, client=client,
                                 responder=responder, label_policy=label_policy)
    return label_image_folder(
        labeler, input_dir, out, backend="openai", key_hint=OPENAI_KEY_HINT,
        max_images=max_images, model=model, min_confidence=min_confidence, resume=resume,
        force=force, label_policy=label_policy, cost_per_mtok_in=cost_per_mtok_in,
        cost_per_mtok_out=cost_per_mtok_out,
        cost_note="token counts only; verify current OpenAI pricing for this model", log=log)
