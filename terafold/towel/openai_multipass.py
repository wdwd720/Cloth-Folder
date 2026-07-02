"""Multi-stage HIGH-PRECISION OpenAI towel labeling (the ``--multipass`` path).

Single-pass corner labeling on a full real-world photo tends to produce loose,
bounding-box-ish corners — the model "sees" the whole scene and brackets the towel
instead of tracing it. This pipeline trades tokens for accuracy: it spends multiple
vision calls per image to pin the 4 corners tightly on the striped towel.

Per image (geometry only — never robot commands):

  1. LOCALIZE   — full image -> tight bbox of ONLY the beige/white striped towel
                  (ignore bed sheet/mattress/pillow/blanket/headboard/floor/body).
  2. CROP+ZOOM  — crop that bbox (small padding), save the zoomed crop, and run a
                  SECOND vision call on the crop alone.
  3. CORNERS    — on the crop, label the 4 VISIBLE OUTER corners of the towel's
                  current shape (not hidden originals, not the crop/image border,
                  not an axis-aligned bbox; rounded corner -> edge intersection).
  4. REMAP      — map the crop keypoints back to ORIGINAL image coordinates.
  5. VERIFY     — send the original image + the proposed keypoints back and ask:
                  all on the towel? any on background? just a loose bbox? tight
                  enough for YOLO? -> accept / correct (new corners) / reject.
  6. GEOMETRY   — automatic checks: corners ≈ crop/image bbox, quad far outside the
                  detected towel region, bbox area >> towel region, corner on the
                  image edge -> flag (review) or reject.
  7. SAVE       — localization overlay, crop, crop-corner overlay, full-image
                  remapped overlay, verification JSON, and the final label JSON.

Safety / hygiene mirror :mod:`terafold.towel.openai_pseudolabel`: the OpenAI SDK is
a lazy optional import, the key is read ONLY from ``OPENAI_API_KEY`` (never stored
/ printed), and every label starts ``usable_for_training=False`` (``pending`` /
``review_needed`` / ``rejected``) — nothing reaches training without human review.
Tests drive all three stages through a ``stage_responder`` seam (or a fake Responses
``client``) and never make real paid calls.
"""

from __future__ import annotations

import os
import shutil
from typing import Any, Callable, Dict, List, Optional, Tuple

from terafold.data.episode_schema import read_json, write_json
from terafold.towel.claude_pseudolabel import (STRIPED_TOWEL_POLICIES, ClaudeTowelError,
                                               _as_bool, _encode_image, _num_pair, _parse_json,
                                               _sid_minter, _valid_bbox, corners_look_like_bbox,
                                               parse_and_build_pseudolabel)
from terafold.towel.crop_candidates import (DEFAULT_TARGET_DESCRIPTION, LOCALIZE_SYSTEM_PROMPT,
                                            _crop_array, _localize_overlay, build_localize_prompt,
                                            compute_crop_box, remap_bbox, remap_corners)
from terafold.towel.dataset import dataset_paths
from terafold.towel.openai_pseudolabel import (DEFAULT_OPENAI_MODEL, OPENAI_KEY_HINT,
                                               OPENAI_LABEL_POLICY, OPENAI_SDK_HINT,
                                               OPENAI_SYSTEM_PROMPT, OPENAI_USER_PROMPT,
                                               OpenAITowelError, _openai_output_text, _openai_usage)
from terafold.towel.overlay import draw_towel_overlay
from terafold.towel.raw_ingest import load_input_images
from terafold.towel.schema import (CORNER_ORDER, TOWEL_STATES, TRAINABLE_STATES,
                                   bbox_from_corners, image_size, quad_is_simple, validate_label)

__all__ = [
    "VERIFY_SYSTEM_PROMPT", "build_verify_prompt", "parse_verification",
    "multipass_geometry_checks", "MultipassOpenAILabeler", "label_towel_folder_multipass",
    "EDGE_FRAC", "BBOX_AREA_RATIO",
]

# Geometry-gate thresholds.
EDGE_FRAC = 0.01          # a corner within 1% of the image border is "on the edge" -> reject
BBOX_AREA_RATIO = 1.8     # quad bbox area > 1.8x the detected towel bbox -> spilled outside
OUTSIDE_MARGIN_FRAC = 0.15  # quad extending >15% of the towel bbox beyond it -> flag

VERIFY_SYSTEM_PROMPT = (
    "You are a meticulous QA checker for towel corner keypoints used to train a "
    "cloth-folding robot's perception. You ONLY judge image geometry for ONE towel "
    "and NEVER output robot commands. Respond with STRICT JSON only: no prose, no "
    "markdown, no code fences."
)


def build_verify_prompt(w: int, h: int, corners_xy: List[List[float]]) -> str:
    """Build the verification prompt for the 4 proposed corners (original coords)."""
    tl, tr, br, bl = corners_xy
    pt = lambda p: f"[{round(p[0], 1)}, {round(p[1], 1)}]"  # noqa: E731
    return f"""This image is {w} px wide by {h} px tall (origin top-left). Four proposed corner keypoints for the beige/white striped towel are drawn on it as colored dots connected by a white quadrilateral:
  tl={pt(tl)}, tr={pt(tr)}, br={pt(br)}, bl={pt(bl)}

Judge the STRIPED TOWEL ONLY. Ignore the bed sheet, mattress cover, pillow, blanket, headboard, floor, and any body part / feet / hands.

Answer these checks:
- all_on_towel: are ALL 4 points exactly on real outer corners of the striped towel?
- any_on_background: is ANY point on the bed/mattress/pillow/blanket/headboard/floor/body/other background?
- is_loose_bbox: are these just a loose axis-aligned bounding box rather than the towel's true visible outline?
- tight_enough_for_yolo: are the corners tight and accurate enough for YOLO pose training?

Decide a verdict:
- "accept" if the points are already good.
- "correct" if they are close but should be nudged — then give corrected_corners ON the real towel corners, in ORIGINAL image pixel coordinates (0..{w}, 0..{h}).
- "reject" if the striped towel is not clearly present, the points are on background, or they cannot be fixed.

Return ONLY this JSON object:
{{
  "all_on_towel": true,
  "any_on_background": false,
  "is_loose_bbox": false,
  "tight_enough_for_yolo": true,
  "verdict": "accept | correct | reject",
  "corrected_corners": {{"tl": [x, y], "tr": [x, y], "br": [x, y], "bl": [x, y]}},
  "reason": "one short sentence"
}}"""


def parse_verification(raw: str) -> Dict[str, Any]:
    """Parse the verification JSON into a normalized dict.

    Returns ``{all_on_towel, any_on_background, is_loose_bbox, tight_enough_for_yolo,
    verdict, corrected_corners (dict|None), reason}``. Raises
    :class:`OpenAITowelError` only on unparseable JSON. Booleans are coerced robustly
    (a stringified ``"false"`` is not treated as True)."""
    data = _parse_json(raw)
    verdict = str(data.get("verdict", "")).strip().lower()
    if verdict not in ("accept", "correct", "reject"):
        verdict = "reject"  # unknown verdict -> be conservative
    corrected = None
    raw_cc = data.get("corrected_corners")
    if isinstance(raw_cc, dict):
        cc = {k: _num_pair(raw_cc.get(k)) for k in CORNER_ORDER}
        if all(cc[k] is not None for k in CORNER_ORDER):
            corrected = cc
    return {
        "all_on_towel": _as_bool(data.get("all_on_towel"), default=True),
        "any_on_background": _as_bool(data.get("any_on_background"), default=False),
        "is_loose_bbox": _as_bool(data.get("is_loose_bbox"), default=False),
        "tight_enough_for_yolo": _as_bool(data.get("tight_enough_for_yolo"), default=True),
        "verdict": verdict,
        "corrected_corners": corrected,
        "reason": str(data.get("reason", ""))[:300],
    }


# --------------------------------------------------------------------------
# Automatic geometry checks (no API; the YOLO-readiness gate)
# --------------------------------------------------------------------------


def _fills_frame(corners_xy: List[List[float]], w: int, h: int, tol_frac: float = 0.04) -> bool:
    """True if the 4 corners sit (nearly) on the FRAME corners of a ``w``×``h`` image —
    i.e. the model labeled the crop/image BORDER instead of the towel. Tolerance
    scales with the image diagonal."""
    frame = [[0.0, 0.0], [w - 1.0, 0.0], [w - 1.0, h - 1.0], [0.0, h - 1.0]]  # tl,tr,br,bl
    tol = tol_frac * ((w ** 2 + h ** 2) ** 0.5)
    return all(((c[0] - f[0]) ** 2 + (c[1] - f[1]) ** 2) ** 0.5 <= tol
               for c, f in zip(corners_xy, frame))


def _crop_border_tol(pad_frac: float) -> float:
    """Crop-frame tolerance (fraction of the crop diagonal) calibrated to the padding.

    ``compute_crop_box`` pads the towel bbox by ``pad_frac`` per side, so a CORRECTLY
    traced towel that fills its bbox sits ~``pad/(1+2·pad)`` of the crop diagonal in
    from each crop corner. Setting the tolerance to half that keeps a true crop-frame
    label (corners ≈ 0 inset) flagged while never hard-rejecting a correct tight label,
    regardless of how small ``--crop-padding`` is."""
    pad = max(0.0, float(pad_frac))
    return max(0.01, min(0.04, 0.5 * pad / (1.0 + 2.0 * pad)))


def multipass_geometry_checks(
    full_corners_xy: List[List[float]],
    full_bbox: List[float],
    w: int,
    h: int,
    loc_bbox: Optional[List[float]],
    crop_corners_xy: Optional[List[List[float]]] = None,
    crop_w: Optional[int] = None,
    crop_h: Optional[int] = None,
    crop_pad_frac: float = 0.08,
) -> Dict[str, Any]:
    """Automatic YOLO-readiness checks on the FINAL (original-coordinate) corners.

    Returns ``{"risks": [...], "reject": bool}``. HARD-REJECT: the corners fill the
    crop/image border (labeled the frame, not the towel), a corner on the image edge,
    an impossible (self-crossing) quad, or a quad whose area spills far outside the
    detected striped-towel region. SOFTER flags (axis-aligned/bbox-ish corners, quad
    slightly outside the towel bbox) are recorded for review but not auto-rejected — a
    genuinely top-down towel can be axis-aligned. The axis-aligned tolerance and the
    crop-frame tolerance both scale with the TOWEL size (not the full image), so a
    small towel in a big photo and a tightly-cropped towel are judged correctly."""
    risks: List[str] = []
    reject = False

    # Labeled the crop frame (the crop border) instead of the towel -> reject. The
    # tolerance tracks --crop-padding so a correct towel filling a tight crop is safe.
    if (crop_corners_xy is not None and crop_w and crop_h
            and _fills_frame(crop_corners_xy, int(crop_w), int(crop_h),
                             tol_frac=_crop_border_tol(crop_pad_frac))):
        risks.append("corners_fill_crop_border")
        reject = True
    if _fills_frame(full_corners_xy, w, h):
        risks.append("corners_fill_image_border")
        reject = True

    # Axis-aligned / bbox-ish corners (a loose-label signal) -> flag, not hard reject.
    # Scale the tolerance to the TOWEL (loc_bbox), not the full image, or a small towel
    # in a high-res photo would have a huge tolerance and be spuriously flagged.
    if loc_bbox:
        tol_w, tol_h = max(1.0, loc_bbox[2] - loc_bbox[0]), max(1.0, loc_bbox[3] - loc_bbox[1])
    else:
        tol_w, tol_h = float(w), float(h)
    if corners_look_like_bbox(full_corners_xy, tol_w, tol_h):
        risks.append("corners_axis_aligned_bbox")

    if not quad_is_simple(full_corners_xy):
        risks.append("geometry_impossible")
        reject = True

    eps_x, eps_y = EDGE_FRAC * w, EDGE_FRAC * h
    for k, (x, y) in zip(CORNER_ORDER, full_corners_xy):
        if x <= eps_x or x >= (w - 1) - eps_x or y <= eps_y or y >= (h - 1) - eps_y:
            risks.append(f"corner_on_image_edge:{k}")
            reject = True

    if loc_bbox and full_bbox:
        quad_area = max(0.0, (full_bbox[2] - full_bbox[0])) * max(0.0, (full_bbox[3] - full_bbox[1]))
        lw, lh = (loc_bbox[2] - loc_bbox[0]), (loc_bbox[3] - loc_bbox[1])
        loc_area = max(0.0, lw) * max(0.0, lh)
        if loc_area > 0 and quad_area > BBOX_AREA_RATIO * loc_area:
            risks.append("quad_area_exceeds_towel_region")
            reject = True
        mx, my = OUTSIDE_MARGIN_FRAC * lw, OUTSIDE_MARGIN_FRAC * lh
        if (full_bbox[0] < loc_bbox[0] - mx or full_bbox[1] < loc_bbox[1] - my
                or full_bbox[2] > loc_bbox[2] + mx or full_bbox[3] > loc_bbox[3] + my):
            risks.append("quad_outside_detected_towel")  # edges outside towel texture -> flag

    return {"risks": sorted(set(risks)), "reject": reject}


# --------------------------------------------------------------------------
# Transport — three Responses-API calls per image (lazy SDK)
# --------------------------------------------------------------------------


class MultipassOpenAILabeler:
    """Runs the localize / corner / verify OpenAI vision calls for one image.

    Test seams (no SDK / no network): ``stage_responder(stage, image_path, w, h) ->
    raw_json`` (``stage`` is ``"localize"`` / ``"corners"`` / ``"verify"``), or a fake
    ``client`` exposing ``responses.create(...) -> response.output_text``. Token usage
    accumulates across every call in ``self.usage``.
    """

    def __init__(
        self,
        model: str = DEFAULT_OPENAI_MODEL,
        target_description: str = DEFAULT_TARGET_DESCRIPTION,
        api_key: Optional[str] = None,
        client: Any = None,
        stage_responder: Optional[Callable[[str, str, int, int], str]] = None,
        max_tokens: int = 1500,
        max_retries: int = 2,
        json_mode: bool = True,
        debug: bool = False,
        debug_dir: Optional[str] = None,
        log: Callable[[str], None] = print,
    ) -> None:
        self.model = model
        self.target_description = target_description or DEFAULT_TARGET_DESCRIPTION
        self.api_key = api_key
        self.client = client
        self._stage_responder = stage_responder
        self.max_tokens = int(max_tokens)
        self.max_retries = max(0, int(max_retries))
        self.json_mode = bool(json_mode)
        self.debug = bool(debug)
        self.debug_dir = debug_dir
        self.log = log
        self.usage: Dict[str, int] = {"input_tokens": 0, "output_tokens": 0}

    def have_transport(self) -> bool:
        if self._stage_responder is not None or self.client is not None:
            return True
        return bool(self.api_key or os.environ.get("OPENAI_API_KEY"))

    # -- stage calls -------------------------------------------------------

    def localize(self, image_path: str, w: int, h: int) -> str:
        return self._call("localize", LOCALIZE_SYSTEM_PROMPT,
                          build_localize_prompt(self.target_description, int(w), int(h)),
                          image_path, w, h)

    def label_corners(self, crop_path: str, cw: int, ch: int) -> str:
        return self._call("corners", OPENAI_SYSTEM_PROMPT,
                          OPENAI_USER_PROMPT.format(w=int(cw), h=int(ch)), crop_path, cw, ch)

    def verify(self, overlay_path: str, w: int, h: int, corners_xy: List[List[float]]) -> str:
        return self._call("verify", VERIFY_SYSTEM_PROMPT,
                          build_verify_prompt(int(w), int(h), corners_xy), overlay_path, w, h)

    # -- debug helpers -----------------------------------------------------

    def _debug_log(self, stage: str, image_path: str, extracted: bool, attempt: int) -> None:
        if self.debug:
            self.log(f"[debug] provider=openai stage={stage} model={self.model} "
                     f"attempt={attempt + 1} image={os.path.basename(image_path)} "
                     f"text_extracted={extracted}")

    def _save_debug(self, stage: str, image_path: str, raw: Any, error: str) -> None:
        """Save a raw-response snippet for a failed stage. NEVER writes the API key
        (``raw`` is the model's response text only); failures here never break the run."""
        if not self.debug_dir:
            return
        try:
            os.makedirs(self.debug_dir, exist_ok=True)
            base = os.path.splitext(os.path.basename(image_path))[0]
            write_json(os.path.join(self.debug_dir, f"{base}_{stage}.json"), {
                "provider": "openai", "stage": stage, "model": self.model,
                "image": os.path.basename(image_path), "error": str(error)[:300],
                "response_snippet": (str(raw)[:2000] if raw else ""),
                "note": "raw model response snippet for debugging; the API key is never stored.",
            })
        except Exception:  # pragma: no cover - debug saving is strictly best-effort
            pass

    # -- transport ---------------------------------------------------------

    def _call(self, stage: str, system: str, user_text: str, image_path: str,
              w: int, h: int) -> str:
        if self._stage_responder is not None:
            raw = self._stage_responder(stage, image_path, int(w), int(h))
            self._debug_log(stage, image_path, extracted=bool(raw and str(raw).strip()), attempt=0)
            return raw
        return self._call_api(stage, system, user_text, image_path)

    def _call_api(self, stage: str, system: str, user_text: str, image_path: str) -> str:
        api_key = self.api_key or os.environ.get("OPENAI_API_KEY")
        if self.client is None and not api_key:
            raise OpenAITowelError(OPENAI_KEY_HINT)
        client = self.client
        if client is None:
            try:
                from openai import OpenAI
            except Exception as exc:  # pragma: no cover - only without SDK
                raise OpenAITowelError(OPENAI_SDK_HINT) from exc
            client = OpenAI(api_key=api_key)

        media_type, b64 = _encode_image(image_path)
        data_url = f"data:{media_type};base64,{b64}"
        attempts = 1 + self.max_retries
        last_raw = ""
        for attempt in range(attempts):
            text, max_tok = user_text, self.max_tokens
            if attempt > 0:
                # Retry on empty: lower complexity (demand strict JSON only) and give
                # more token headroom — reasoning models can spend the whole budget
                # before emitting any output text.
                text = (user_text + "\n\nIMPORTANT: Return STRICT JSON ONLY — exactly one "
                        "JSON object, no prose, no markdown, no code fences, no explanation.")
                max_tok = int(self.max_tokens * (attempt + 1))
            response = self._responses_create(client, system, text, data_url, max_tok)
            u = _openai_usage(response)
            self.usage["input_tokens"] += u.get("input_tokens", 0)
            self.usage["output_tokens"] += u.get("output_tokens", 0)
            raw = _openai_output_text(response)
            extracted = bool(raw and raw.strip())
            self._debug_log(stage, image_path, extracted=extracted, attempt=attempt)
            if extracted:
                return raw
            last_raw = raw or last_raw
        # Every attempt produced empty text -> OpenAI-branded error (never "Claude").
        self._save_debug(stage, image_path, last_raw, "empty response text after retries")
        raise OpenAITowelError(
            f"OpenAI returned an empty response for stage '{stage}' "
            f"(provider=openai, model={self.model}) after {attempts} attempt(s). "
            "Verify the model id supports image input and text output.")

    def _responses_create(self, client: Any, system: str, text: str, data_url: str,
                          max_tok: int) -> Any:
        """One Responses-API call, requesting structured JSON output when supported (#8).

        Only an "unsupported parameter" signal (an old SDK that rejects the ``text``
        kwarg, or an HTTP-400 BadRequest from the API) disables json mode and falls back
        — a genuine transient error (rate limit / connection / 5xx / auth) PROPAGATES so
        it is not masked into a duplicate call or a run-wide json-mode disable."""
        kwargs: Dict[str, Any] = dict(
            model=self.model, instructions=system,
            input=[{"role": "user", "content": [
                {"type": "input_text", "text": text},
                {"type": "input_image", "image_url": data_url}]}],
            max_output_tokens=int(max_tok))
        if self.json_mode:
            try:
                return client.responses.create(text={"format": {"type": "json_object"}}, **kwargs)
            except TypeError:
                self.json_mode = False          # SDK too old to accept the `text` kwarg
            except Exception as exc:
                if not _is_unsupported_param(exc):
                    raise                       # genuine transient/auth/server error -> propagate
                self.json_mode = False          # API rejected the json-mode param (HTTP 400)
        return client.responses.create(**kwargs)


# --------------------------------------------------------------------------
# Folder orchestration
# --------------------------------------------------------------------------


def label_towel_folder_multipass(
    input_dir: str,
    out: str,
    target_description: str = DEFAULT_TARGET_DESCRIPTION,
    max_images: int = 500,
    model: str = DEFAULT_OPENAI_MODEL,
    min_confidence: float = 0.75,
    crop_padding: float = 0.08,
    max_crop_size: Optional[int] = 1024,
    verify: bool = True,
    label_policy: str = OPENAI_LABEL_POLICY,
    resume: bool = False,
    force: bool = False,
    debug: bool = False,
    api_key: Optional[str] = None,
    client: Any = None,
    stage_responder: Optional[Callable[[str, str, int, int], str]] = None,
    log: Callable[[str], None] = print,
) -> Dict[str, Any]:
    """Multi-pass OpenAI towel labeling into ``out`` (see the module docstring).

    Returns a status dict. ``status='no_api_key'`` (with instructions) when no
    key/transport is available — it never raises and never prints the key. ``debug``
    turns on per-stage progress logging; raw-response snippets are written to
    ``out/debug/`` whenever a stage response cannot be parsed (the key is never saved).
    """
    if label_policy not in STRIPED_TOWEL_POLICIES:
        return {"status": "error",
                "message": f"--label-policy must be one of {list(STRIPED_TOWEL_POLICIES)} "
                           f"for multipass (got {label_policy!r})",
                "label_policies": list(STRIPED_TOWEL_POLICIES)}
    if not os.path.isdir(input_dir):
        return {"status": "error", "message": f"input not found: {input_dir}"}
    src = load_input_images(input_dir, default_source="external")
    images = src.get("images", [])
    if not images:
        return {"status": "error",
                "message": f"no images found in {input_dir} "
                           "(expected a manifest, an images/ folder, or image files)"}

    debug_dir = os.path.join(out, "debug")
    labeler = MultipassOpenAILabeler(model=model, target_description=target_description,
                                     api_key=api_key, client=client, stage_responder=stage_responder,
                                     debug=debug, debug_dir=debug_dir, log=log)
    if not labeler.have_transport():
        return {"status": "no_api_key", "instructions": OPENAI_KEY_HINT.splitlines()}

    reuse = resume and not force
    paths = dataset_paths(out)
    dirs = {
        "images": paths["images"], "labels": paths["labels"],
        "overlays": os.path.join(out, "overlays"),
        "crops": os.path.join(out, "crops"),
        "crop_overlays": os.path.join(out, "crop_overlays"),
        "localize_overlays": os.path.join(out, "localize_overlays"),
        "verify": os.path.join(out, "verify"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    if debug:
        log(f"[debug] provider=openai multipass start model={model} images={len(images)} "
            f"max_images={max_images} verify={verify} crop_padding={crop_padding}")

    prior = _load_prior(out) if reuse else {}
    _next_sid = _sid_minter({s.get("id") for s in prior.values() if s.get("id")})

    samples: List[Dict[str, Any]] = []
    counts = {"labeled": 0, "rejected": 0, "review_needed": 0, "pending": 0,
              "errors": 0, "resumed": 0, "reject_label": 0}
    n_attempt = 0

    for im in images:
        if n_attempt >= max_images:
            break
        rel = im.get("image")
        if not rel:
            continue
        n_attempt += 1
        src_img = os.path.join(input_dir, rel)
        image_source = im.get("source", src.get("source", "external"))
        h_hash = im.get("hash")

        if reuse and h_hash and h_hash in prior:
            samples.append(dict(prior[h_hash]))
            counts["resumed"] += 1
            _tally(counts, prior[h_hash].get("approval_status"))
            continue

        w, hh = im.get("width"), im.get("height")
        if not (w and hh):
            size = image_size(src_img)
            if size:
                w, hh = size
        if not (w and hh):
            counts["errors"] += 1
            log(f"[warn] unknown image size for {rel}; skipping.")
            continue

        try:
            outcome = _label_one(labeler, src_img, int(w), int(hh), out, dirs, _next_sid,
                                 model=model, min_confidence=min_confidence,
                                 crop_padding=crop_padding, max_crop_size=max_crop_size,
                                 verify=verify, label_policy=label_policy,
                                 target_description=target_description, log=log)
        except ClaudeTowelError as exc:  # OpenAITowelError (subclass) + any shared-parser error
            counts["errors"] += 1
            log(f"[warn] {rel}: {exc}")
            continue
        except Exception as exc:  # never crash the whole run
            counts["errors"] += 1
            log(f"[warn] {rel}: OpenAI multipass failed (provider=openai, model={model}): {exc}")
            continue

        if outcome.get("status") == "rejected_localization":
            counts["rejected"] += 1
            log(f"[reject] {rel}: localization — {outcome.get('reason')}")
            continue

        sample = outcome["sample"]
        sample.update({
            "hash": h_hash, "source": image_source,
            "image_source": image_source, "width": w, "height": hh,
        })
        samples.append(sample)
        counts["labeled"] += 1
        _tally(counts, sample.get("approval_status"))
        log(f"[label] {rel} -> {sample['id']} "
            f"({sample.get('approval_status')}; {len(sample.get('risk_reasons', []))} risks)")

    manifest = {
        "dataset": os.path.basename(os.path.normpath(out)),
        "kind": "pseudolabeled",
        "label_schema": "towel_v1",
        "states": TOWEL_STATES,
        "trainable_states": TRAINABLE_STATES,
        "source": "openai_pseudolabel",
        "pseudolabel": True,
        "backend": "openai",
        "pipeline": "multipass",
        "model": model,
        "label_policy": label_policy,
        "prompt_name": label_policy,
        "target_description": target_description or DEFAULT_TARGET_DESCRIPTION,
        "crop_padding": float(crop_padding),
        "verify": bool(verify),
        "min_confidence": float(min_confidence),
        "forced_relabel": bool(force),
        "input_dataset": os.path.basename(os.path.normpath(input_dir)),
        "input_source": src.get("source"),
        "num_input": len(images),
        "num_images": len(samples),
        "num_labeled": counts["labeled"],
        "num_rejected_localization": counts["rejected"],
        "num_review_needed": counts["review_needed"],
        "num_pending": counts["pending"],
        "num_label_rejected": counts["reject_label"],
        "num_errors": counts["errors"],
        "num_resumed": counts["resumed"],
        "usage": dict(labeler.usage),
        "warning": "Multipass pseudo-labels are machine-generated and may be WRONG. "
                   "Review before training.",
        "samples": samples,
    }
    write_json(paths["manifest"], manifest)
    _write_readme(out, manifest)
    log(f"Multipass-labeled {counts['labeled']} images -> {out} "
        f"({counts['rejected']} localization-rejected, {counts['review_needed']} need review, "
        f"{counts['reject_label']} label-rejected, {counts['errors']} errors, "
        f"{counts['resumed']} resumed)")
    u = labeler.usage
    if u.get("input_tokens") or u.get("output_tokens"):
        log(f"  tokens in/out: {u['input_tokens']}/{u['output_tokens']} "
            "(verify current OpenAI pricing for this model)")
    return {
        "status": "ok",
        "out": out,
        "num_input": len(images),
        "num_images": len(samples),
        "labeled": counts["labeled"],
        "rejected_localization": counts["rejected"],
        "review_needed": counts["review_needed"],
        "label_rejected": counts["reject_label"],
        "errors": counts["errors"],
        "resumed": counts["resumed"],
        "usage": dict(labeler.usage),
        "manifest": paths["manifest"],
    }


def _is_unsupported_param(exc: Exception) -> bool:
    """True if ``exc`` means the request PARAMS are unsupported (an HTTP-400 BadRequest /
    UnprocessableEntity), as opposed to a transient/auth/server error that must propagate.

    Detected structurally (no eager ``openai`` import): OpenAI SDK errors carry a
    ``status_code`` (e.g. ``BadRequestError.status_code == 400``)."""
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(exc, "status", None)
    if status in (400, 422):
        return True
    name = type(exc).__name__.lower()
    return "badrequest" in name or "unprocessable" in name or "unsupported" in name


def _safe_parse(stage: str, raw: str, image_path: str, labeler: "MultipassOpenAILabeler",
                parse_fn: Callable[[str], Any]) -> Any:
    """Parse a stage response; on failure save a debug snippet and raise an
    OpenAI-branded error (never a Claude-named message — the shared parser is
    provider-neutral and this adds the OpenAI provider context)."""
    try:
        return parse_fn(raw)
    except ClaudeTowelError as exc:
        labeler._save_debug(stage, image_path, raw, str(exc))
        raise OpenAITowelError(
            f"OpenAI {stage} response was not valid JSON "
            f"(provider=openai, model={labeler.model}): {exc}") from exc


def _label_one(
    labeler: MultipassOpenAILabeler, src_img: str, w: int, h: int, out: str,
    dirs: Dict[str, str], next_sid: Callable[[], str], *, model: str, min_confidence: float,
    crop_padding: float, max_crop_size: Optional[int], verify: bool, label_policy: str,
    target_description: str, log: Callable[[str], None],
) -> Dict[str, Any]:
    """Run the full multipass pipeline for ONE image; returns an outcome dict."""
    import numpy as np

    from terafold.towel.crop_candidates import parse_localization
    from terafold.vision.imageio import imread, imwrite

    # -- Stage 1: localize the striped towel (full image) ------------------
    raw_loc = labeler.localize(src_img, w, h)
    loc = _safe_parse("localize", raw_loc, src_img, labeler,
                      lambda r: parse_localization(r, w, h, min_confidence=min_confidence))
    if not loc["ok"]:
        return {"status": "rejected_localization",
                "reason": loc["reject_reason"], "confidence": loc["confidence"]}
    loc_bbox = loc["bbox"]

    sid = next_sid()
    ext = os.path.splitext(src_img)[1].lower() or ".jpg"

    # -- Stage 2: crop + zoom ---------------------------------------------
    box = compute_crop_box(loc_bbox, w, h, pad_frac=crop_padding)
    arr = np.array(imread(src_img), dtype=np.uint8)
    crop_arr, scale_x, scale_y = _crop_array(arr, box, max_crop_size)
    cw, ch = int(crop_arr.shape[1]), int(crop_arr.shape[0])
    crop_rel = f"crops/{sid}_crop.png"
    imwrite(os.path.join(out, crop_rel), crop_arr)
    cx1, cy1, cx2, cy2 = box
    crop_meta = {
        "orig_image": os.path.abspath(src_img), "orig_width": w, "orig_height": h,
        "x_off": cx1, "y_off": cy1, "region_width": cx2 - cx1, "region_height": cy2 - cy1,
        "crop_width": cw, "crop_height": ch, "scale_x": scale_x, "scale_y": scale_y,
        "pad_frac": float(crop_padding), "detected_bbox": [float(v) for v in loc_bbox],
        "localize_confidence": loc["confidence"],
    }
    # localization overlay on the ORIGINAL (detected bbox + padded crop region).
    localize_rel = f"localize_overlays/{sid}.png"
    _localize_overlay(src_img, loc_bbox, box, os.path.join(out, localize_rel))

    # -- Stage 3: corner labeling on the crop ------------------------------
    crop_path = os.path.join(out, crop_rel)
    raw_corners = labeler.label_corners(crop_path, cw, ch)
    crop_label = _safe_parse("corners", raw_corners, crop_path, labeler,
                             lambda r: parse_and_build_pseudolabel(
                                 r, crop_rel, cw, ch, model=model, min_confidence=min_confidence,
                                 label_policy=label_policy, backend="openai"))
    crop_corners_xy = _corners_xy(crop_label.get("corners"))

    # Crop-corner overlay (crop-space corners on the crop).
    crop_ov_rel = f"crop_overlays/{sid}_crop.png"
    cov = draw_towel_overlay(os.path.join(out, crop_rel), crop_label, os.path.join(out, crop_ov_rel))

    risk: set = set(crop_label.get("risk_reasons", []))
    reject = False
    if crop_label.get("state") in ("not_towel", "multiple_towels", "bad_view") or not crop_label.get("is_target", True):
        reject = True  # the corner stage itself says this crop is not the striped towel

    # -- Stage 4: remap crop keypoints to ORIGINAL coordinates -------------
    full_corners = remap_corners(crop_label.get("corners") or {}, crop_meta)
    full_corners_xy = _corners_xy(full_corners)
    if full_corners_xy is None:
        risk.add("corners_missing")
        reject = True
        full_bbox = loc_bbox
    else:
        full_bbox = bbox_from_corners(full_corners_xy)

    # -- Stage 5: verification pass ---------------------------------------
    verification: Dict[str, Any] = {"skipped": True}
    full_overlay_rel = f"overlays/{sid}_overlay.png"
    if full_corners_xy is not None:
        _render_full_overlay(src_img, full_corners, crop_label.get("visible"), full_bbox,
                             os.path.join(out, full_overlay_rel))
        if verify:
            overlay_path = os.path.join(out, full_overlay_rel)
            raw_v = labeler.verify(overlay_path, w, h, full_corners_xy)
            verification = _safe_parse("verify", raw_v, overlay_path, labeler, parse_verification)
            verification["skipped"] = False
            v_risk, v_reject, corrected = _apply_verification(verification, w, h)
            risk |= v_risk
            reject = reject or v_reject
            if corrected is not None:
                full_corners = corrected
                full_corners_xy = _corners_xy(full_corners)
                full_bbox = bbox_from_corners(full_corners_xy)
                risk.add("verify_corrected")
                _render_full_overlay(src_img, full_corners, crop_label.get("visible"), full_bbox,
                                     os.path.join(out, full_overlay_rel))
    write_json(os.path.join(out, f"verify/{sid}_verify.json"), verification)

    # -- Stage 6: automatic geometry checks --------------------------------
    if full_corners_xy is not None:
        geo = multipass_geometry_checks(full_corners_xy, full_bbox, w, h, loc_bbox,
                                        crop_corners_xy=crop_corners_xy, crop_w=cw, crop_h=ch,
                                        crop_pad_frac=crop_padding)
        risk |= set(geo["risks"])
        reject = reject or geo["reject"]

    # -- Stage 7: copy the ORIGINAL image + build the final label ----------
    img_rel = f"images/{sid}{ext}"
    shutil.copyfile(src_img, os.path.join(out, img_rel))

    approval = "rejected" if reject else ("review_needed" if risk else "pending")
    final = dict(crop_label)
    final.update({
        "image": img_rel,
        "corners": full_corners if full_corners_xy is not None else crop_label.get("corners"),
        "bbox": full_bbox,
        "pipeline": "multipass",
        "multipass": True,
        "crop": crop_meta,
        "crop_corners": crop_label.get("corners"),
        "localization": {"bbox": loc_bbox, "confidence": loc["confidence"],
                         "reason": loc.get("reason", "")},
        "verification": verification,
        "risk_reasons": sorted(risk),
        "approval_status": approval,
        "review_needed": approval != "pending",
        "usable_for_training": False,
    })
    problems = validate_label(final)
    if problems:
        final.setdefault("risk_reasons", [])
        final["risk_reasons"] = sorted(set(final["risk_reasons"]) | {"schema_invalid"})
        final["approval_status"] = "review_needed"
        final["review_needed"] = True
        log(f"[warn] {os.path.basename(src_img)}: final label schema problems: {problems}")
    # Re-sync from the (possibly downgraded) final label so the manifest sample and the
    # run counts never disagree with the on-disk label's approval_status.
    approval = final["approval_status"]

    lab_rel = f"labels/{sid}.json"
    write_json(os.path.join(out, lab_rel), final)

    sample = {
        "id": sid,
        "image": img_rel,
        "label": lab_rel,
        "overlay": full_overlay_rel,
        "crop": crop_rel,
        "crop_overlay": crop_ov_rel if cov.get("rendered") else None,
        "localize_overlay": localize_rel,
        "verify": f"verify/{sid}_verify.json",
        "pseudolabel": True,
        "label_backend": "openai",
        "pipeline": "multipass",
        "label_policy": label_policy,
        "state": final.get("state"),
        "confidence": final.get("confidence"),
        "approval_status": approval,
        "risk_reasons": final["risk_reasons"],
        "crop_meta": crop_meta,
    }
    return {"status": "ok", "sample": sample}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _apply_verification(v: Dict[str, Any], w: int, h: int) -> Tuple[set, bool, Optional[Dict[str, Any]]]:
    """Turn a verification result into (extra risks, hard-reject, corrected corners)."""
    risk: set = set()
    reject = False
    if not v["all_on_towel"]:
        risk.add("verify_not_all_on_towel")
        reject = True
    if v["any_on_background"]:
        risk.add("verify_on_background")
        reject = True
    if not v["tight_enough_for_yolo"]:
        risk.add("verify_not_tight_for_yolo")
        reject = True
    if v["is_loose_bbox"]:
        risk.add("verify_loose_bbox")
    # The model's free-text reason is preserved in the verification JSON, not folded
    # into risk_reasons (a positive reason must not force review).

    corrected = None
    if v["verdict"] == "reject":
        risk.add("verify_verdict_reject")
        reject = True
    elif v["verdict"] == "correct":
        if v["corrected_corners"] is not None:
            # Clamp corrected corners into bounds; if they parse we adopt them and clear
            # the hard-reject the "needs correcting" signals implied.
            cc = {k: [min(max(p[0], 0.0), float(w - 1)), min(max(p[1], 0.0), float(h - 1))]
                  for k, p in v["corrected_corners"].items()}
            corrected = cc
            reject = False  # the model gave fixed corners; geometry checks still gate them
            risk.discard("verify_not_all_on_towel")
            risk.discard("verify_not_tight_for_yolo")
            risk.discard("verify_loose_bbox")
        else:
            # The model said "needs correcting" but gave no usable corners — do NOT
            # silently keep the original corners as clean/pending; route to review.
            risk.add("verify_correct_no_corners")
    return risk, reject, corrected


def _corners_xy(corners: Optional[Dict[str, Any]]) -> Optional[List[List[float]]]:
    if not corners:
        return None
    out: List[List[float]] = []
    for k in CORNER_ORDER:
        v = corners.get(k)
        if v is None or len(v) != 2:
            return None
        out.append([float(v[0]), float(v[1])])
    return out


def _render_full_overlay(orig_path: str, corners: Dict[str, Any], visible: Any,
                         bbox: Any, out_path: str) -> None:
    draw_towel_overlay(orig_path, {"corners": corners, "visible": visible or {},
                                   "bbox": bbox, "fold_axis": None}, out_path)


def _tally(counts: Dict[str, int], approval: Optional[str]) -> None:
    if approval == "review_needed":
        counts["review_needed"] += 1
    elif approval == "pending":
        counts["pending"] += 1
    elif approval == "rejected":
        counts["reject_label"] += 1


def _load_prior(out: str) -> Dict[str, Any]:
    """Map ``original-image hash -> sample`` for an existing multipass dataset."""
    man_path = dataset_paths(out)["manifest"]
    if not os.path.exists(man_path):
        return {}
    try:
        man = read_json(man_path)
    except Exception:
        return {}
    prior: Dict[str, Any] = {}
    for s in man.get("samples", []):
        hsh = s.get("hash")
        if hsh and os.path.exists(os.path.join(out, s.get("image", ""))):
            prior[hsh] = s
    return prior


def _write_readme(root: str, manifest: Dict[str, Any]) -> None:
    paths = dataset_paths(root)
    lines = [
        f"# Multipass OpenAI towel labels — {manifest['dataset']}",
        "",
        "> ⚠️ Machine-generated by a **multi-stage OpenAI Vision** pipeline (localize → "
        "crop → corner-label → remap → verify → geometry-gate). Labels start "
        "`usable_for_training: false` and are excluded from YOLO export until approved.",
        "",
        f"- model: {manifest['model']}  pipeline: multipass  verify: {manifest['verify']}",
        f"- images: {manifest['num_images']}  "
        f"(review: {manifest['num_review_needed']}, pending: {manifest['num_pending']}, "
        f"label-rejected: {manifest['num_label_rejected']}, "
        f"localization-rejected: {manifest['num_rejected_localization']})",
        f"- crop padding: {manifest['crop_padding']}  min confidence: {manifest['min_confidence']}",
        f"- token usage: in {manifest['usage']['input_tokens']} / out {manifest['usage']['output_tokens']}",
        "",
        "## Artifacts per image",
        "- `localize_overlays/<id>.png` — detected towel bbox + crop region on the full image",
        "- `crops/<id>_crop.png` — the zoomed crop sent for corner labeling",
        "- `crop_overlays/<id>_crop.png` — corners drawn on the crop (crop coordinates)",
        "- `overlays/<id>_overlay.png` — final corners remapped onto the full image",
        "- `verify/<id>_verify.json` — the verification-pass result",
        "- `labels/<id>.json` — final label (corners in ORIGINAL image coordinates)",
        "",
        "## Next",
        "```bash",
        f"python3 -m terafold review-pseudolabels --dataset {root}",
        "```",
    ]
    with open(paths["readme"], "w") as f:
        f.write("\n".join(lines) + "\n")
