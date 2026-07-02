"""Crop-first real-photo towel pipeline — localize the target towel, THEN crop.

Real bedroom photos confuse a one-shot corner labeler: Claude keeps placing the 4
"corners" on the bed / headboard / room rectangle instead of the small striped
towel sitting on it. The fix is to make localization its OWN cheap step:

1. For each full image, ask Claude for ONLY the bounding box of the target object
   (e.g. "beige and white striped towel"), explicitly told to IGNORE the bed sheet,
   pillows, blankets, headboard, floor, and any body part / room rectangle.
2. If Claude says the box it found is NOT the target (``is_target=false``), or it
   cannot find one, or confidence is too low, the candidate is REJECTED — no crop.
3. Otherwise save a TIGHT crop (the box + a little padding) plus ``crop`` metadata
   that lets a later corner label in crop coordinates be remapped back to the
   ORIGINAL full-image coordinates.

The crop dataset this writes is a normal raw-style manifest (``images`` list), so
:func:`terafold.towel.claude_pseudolabel.claude_label_towel_folder` can label the
crops directly; because each entry carries ``crop`` metadata, the labeler remaps
the resulting corners back onto the full image and draws a full-image overlay too.

Safety / hygiene mirror :mod:`terafold.towel.claude_pseudolabel` exactly: the
Anthropic SDK is a lazy optional import, the key is read only from
``ANTHROPIC_API_KEY`` (never stored / printed), Claude is asked ONLY for image
geometry (a bounding box) — never robot commands — and per-image errors are caught
so a long run never crashes. Cropping needs to decode the image, so JPEG inputs
need an image backend (cv2/Pillow); PNGs work with the pure-Python codec.
"""

from __future__ import annotations

import math
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

from terafold.data.episode_schema import write_json
from terafold.towel.claude_pseudolabel import (DEFAULT_MODEL, KEY_HINT, SDK_HINT,
                                               ClaudeTowelError, SYSTEM_PROMPT,
                                               _encode_image, _extract_text, _parse_json,
                                               _sid_minter, _usage, _valid_bbox)
from terafold.towel.raw_ingest import load_input_images
from terafold.towel.schema import image_hash, image_size

__all__ = [
    "DEFAULT_TARGET_DESCRIPTION", "IGNORE_REGIONS", "LOCALIZE_SYSTEM_PROMPT",
    "build_localize_prompt", "ClaudeTowelLocator", "parse_localization",
    "compute_crop_box", "remap_xy", "remap_corners", "remap_bbox",
    "crop_real_towel_candidates", "CROPS_MANIFEST_KIND",
]

DEFAULT_TARGET_DESCRIPTION = "beige and white striped towel"
CROPS_MANIFEST_KIND = "towel_crops"

# Everything Claude must NOT mistake for the towel (requirement: ignore list).
IGNORE_REGIONS = [
    "the white bed sheet / mattress cover",
    "pillows",
    "blankets / duvets / comforters",
    "the headboard",
    "the floor",
    "any body part / feet / hands",
    "walls, furniture, or any large rectangular background region",
]

# Localization reuses the geometry-only safety framing but asks for a BOX only.
LOCALIZE_SYSTEM_PROMPT = (
    SYSTEM_PROMPT
    + " For this task you output ONLY a single bounding box (pixel coordinates) "
    "for one target object — never corners, never robot commands."
)


def build_localize_prompt(target_description: str, w: int, h: int) -> str:
    """Return the per-image localization prompt for ``target_description``.

    The prompt pins Claude to the target object and lists the distractors it must
    ignore, and asks it to self-report (``is_target``) so we can reject a box that
    is actually the bed / pillow / room rather than the towel.
    """
    target = (target_description or DEFAULT_TARGET_DESCRIPTION).strip()
    ignore = "\n".join(f"  - {r}" for r in IGNORE_REGIONS)
    return f"""Find the ONE {target} in this image and return ONLY a tight bounding box around it.

The target object is the {target}. Do not label the bed sheet, pillow, mattress, blanket, or room rectangle.

IGNORE and never select any of these (they are NOT the target):
{ignore}

Image is {w} px wide by {h} px tall; origin is top-left, x=column (0..{w}), y=row (0..{h}).

Rules:
- Return the bounding box of the {target} ONLY — tight around the towel's visible extent, not the surface it lies on.
- If you cannot clearly see the {target}, set found=false (do NOT return a box for the bed/pillow/room instead).
- Set is_target=true ONLY if the box you return is the {target}. If the most prominent rectangle is the bed/sheet/pillow/headboard/floor and NOT the towel, set is_target=false and found=false.
- "rejected_regions" lists the distractors you saw and deliberately did NOT box (e.g. "bed", "pillow", "headboard").

Return ONLY this JSON object:
{{
  "found": true,
  "is_target": true,
  "bbox": [x1, y1, x2, y2],
  "confidence": 0.0,
  "reason": "one short sentence on what you boxed and why it is the {target}",
  "rejected_regions": ["short strings"]
}}"""


# --------------------------------------------------------------------------
# Transport (lazy SDK) — mirrors ClaudeTowelLabeler but returns a BOX.
# --------------------------------------------------------------------------


class ClaudeTowelLocator:
    """Get one strict-JSON bounding box for the target towel from Claude Vision."""

    def __init__(
        self,
        target_description: str = DEFAULT_TARGET_DESCRIPTION,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        client: Any = None,
        responder: Optional[Callable[[str, int, int], str]] = None,
        max_tokens: int = 512,
    ) -> None:
        self.target_description = target_description or DEFAULT_TARGET_DESCRIPTION
        self.model = model
        self.api_key = api_key
        self.client = client
        # responder(image_path, w, h) -> raw text. Test / custom-transport seam.
        self._responder = responder
        self.max_tokens = int(max_tokens)
        self.last_usage: Dict[str, int] = {"input_tokens": 0, "output_tokens": 0}

    def have_transport(self) -> bool:
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
        prompt = build_localize_prompt(self.target_description, int(w), int(h))
        message = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=LOCALIZE_SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image",
                     "source": {"type": "base64", "media_type": media_type, "data": b64}},
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        self.last_usage = _usage(message)
        return _extract_text(message)


# --------------------------------------------------------------------------
# Parse + crop geometry
# --------------------------------------------------------------------------


def parse_localization(raw: str, w: int, h: int, min_confidence: float = 0.5) -> Dict[str, Any]:
    """Parse Claude's localization JSON into a normalized result dict.

    Returns ``{"ok", "found", "is_target", "bbox", "confidence", "reason",
    "rejected_regions", "reject_reason"}``. ``ok`` is True only when Claude found
    the target, said the box IS the target, the box is a valid in-bounds rectangle,
    and confidence ≥ ``min_confidence`` — the reject rule lives here.

    Raises :class:`ClaudeTowelError` only when the JSON is unparseable.
    """
    data = _parse_json(raw)  # may raise ClaudeTowelError

    # Treat a MISSING key and an explicit JSON null the same way (default True), to
    # match parse_and_build_pseudolabel — an off-spec null must not silently reject a
    # genuinely-found towel.
    raw_found = data.get("found", True)
    found = True if raw_found is None else bool(raw_found)
    raw_is_target = data.get("is_target", True)
    is_target = True if raw_is_target is None else bool(raw_is_target)
    try:
        confidence = max(0.0, min(1.0, float(data.get("confidence", 0.0))))
    except (TypeError, ValueError):
        confidence = 0.0
    reason = str(data.get("reason", ""))[:500]
    rejected = [str(r) for r in (data.get("rejected_regions") or []) if str(r).strip()]
    bbox = _valid_bbox(data.get("bbox"), w, h)

    reject_reason: Optional[str] = None
    if not found:
        reject_reason = "target_not_found"
    elif not is_target:
        reject_reason = "not_target"
    elif bbox is None:
        reject_reason = "invalid_bbox"
    elif confidence < float(min_confidence):
        reject_reason = f"low_confidence<{float(min_confidence):g}"

    return {
        "ok": reject_reason is None,
        "found": found,
        "is_target": is_target,
        "bbox": bbox,
        "confidence": confidence,
        "reason": reason,
        "rejected_regions": rejected,
        "reject_reason": reject_reason,
    }


def compute_crop_box(bbox: List[float], w: int, h: int, pad_frac: float = 0.08
                     ) -> Tuple[int, int, int, int]:
    """Pad a detected ``[x1,y1,x2,y2]`` box by ``pad_frac`` of its size and clamp
    to the image. Returns integer ``(x1, y1, x2, y2)`` with x2>x1 and y2>y1."""
    x1, y1, x2, y2 = (float(v) for v in bbox)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    pad = max(0.0, float(pad_frac))
    px, py = pad * (x2 - x1), pad * (y2 - y1)
    cx1 = int(math.floor(max(0.0, x1 - px)))
    cy1 = int(math.floor(max(0.0, y1 - py)))
    cx2 = int(math.ceil(min(float(w), x2 + px)))
    cy2 = int(math.ceil(min(float(h), y2 + py)))
    # Guarantee a non-degenerate crop even for a hairline detection.
    cx2 = min(w, max(cx2, cx1 + 1))
    cy2 = min(h, max(cy2, cy1 + 1))
    if cx2 <= cx1:
        cx1 = max(0, cx2 - 1)
    if cy2 <= cy1:
        cy1 = max(0, cy2 - 1)
    return cx1, cy1, cx2, cy2


def remap_xy(pt: List[float], crop: Dict[str, Any]) -> List[float]:
    """Map a point in CROP pixel coordinates back to ORIGINAL image coordinates."""
    sx = crop.get("scale_x") or 1.0
    sy = crop.get("scale_y") or 1.0
    x = float(crop.get("x_off", 0)) + float(pt[0]) / float(sx)
    y = float(crop.get("y_off", 0)) + float(pt[1]) / float(sy)
    return [x, y]


def remap_corners(corners: Dict[str, Any], crop: Dict[str, Any]) -> Dict[str, Any]:
    """Remap a ``{tl,tr,br,bl}`` corner dict from crop to original coordinates.

    ``None`` corners are passed through unchanged."""
    out: Dict[str, Any] = {}
    for k, v in (corners or {}).items():
        out[k] = remap_xy(v, crop) if (v is not None and len(v) == 2) else v
    return out


def remap_bbox(bbox: Optional[List[float]], crop: Dict[str, Any]) -> Optional[List[float]]:
    """Remap an ``[x1,y1,x2,y2]`` box from crop to original coordinates."""
    if bbox is None or len(bbox) != 4:
        return None
    x1, y1 = remap_xy([bbox[0], bbox[1]], crop)
    x2, y2 = remap_xy([bbox[2], bbox[3]], crop)
    return [x1, y1, x2, y2]


# --------------------------------------------------------------------------
# Crop a single image
# --------------------------------------------------------------------------


def _crop_array(arr, box: Tuple[int, int, int, int], max_crop_size: Optional[int]):
    """Slice ``box`` out of an (H,W,3) array, optionally nearest-neighbor downscaling
    so the longest side ≤ ``max_crop_size``. Returns ``(crop, scale_x, scale_y)``."""
    import numpy as np

    cx1, cy1, cx2, cy2 = box
    crop = np.ascontiguousarray(arr[cy1:cy2, cx1:cx2])
    rh, rw = crop.shape[0], crop.shape[1]
    if max_crop_size and rh > 0 and rw > 0 and max(rh, rw) > int(max_crop_size):
        s = float(max_crop_size) / float(max(rh, rw))
        cw_new = max(1, int(round(rw * s)))
        ch_new = max(1, int(round(rh * s)))
        ys = np.clip((np.arange(ch_new) * (rh / ch_new)).astype(int), 0, rh - 1)
        xs = np.clip((np.arange(cw_new) * (rw / cw_new)).astype(int), 0, rw - 1)
        crop = np.ascontiguousarray(crop[ys][:, xs])
    cw, ch = crop.shape[1], crop.shape[0]
    scale_x = cw / rw if rw else 1.0
    scale_y = ch / rh if rh else 1.0
    return crop, scale_x, scale_y


def _localize_overlay(orig_path: str, detected: List[float], crop_box: Tuple[int, int, int, int],
                      out_path: str) -> bool:
    """Draw the detected towel box (green) + padded crop region (yellow) on a copy of
    the ORIGINAL image so a human can verify Claude boxed the towel, not the bed."""
    try:
        import numpy as np

        from terafold.vision.imageio import imread, imwrite

        img = np.array(imread(orig_path), dtype=np.uint8).copy()
        h, w = img.shape[0], img.shape[1]
        thick = max(2, min(w, h) // 200)

        def _rect(box, color):
            x1, y1, x2, y2 = (int(round(v)) for v in box)
            x1, x2 = max(0, min(w - 1, x1)), max(0, min(w - 1, x2))
            y1, y2 = max(0, min(h - 1, y1)), max(0, min(h - 1, y2))
            for t in range(thick):
                img[max(0, y1 + t):y1 + t + 1, x1:x2 + 1] = color
                img[max(0, y2 - t):y2 - t + 1, x1:x2 + 1] = color
                img[y1:y2 + 1, max(0, x1 + t):x1 + t + 1] = color
                img[y1:y2 + 1, max(0, x2 - t):x2 - t + 1] = color

        _rect(crop_box, (230, 210, 0))         # padded crop region (yellow)
        _rect(detected, (0, 220, 0))           # detected towel box (green)
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
        imwrite(out_path, img)
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------
# Folder orchestration
# --------------------------------------------------------------------------


def crop_real_towel_candidates(
    input_dir: str,
    out: str,
    target_description: str = DEFAULT_TARGET_DESCRIPTION,
    max_images: int = 500,
    model: str = DEFAULT_MODEL,
    min_confidence: float = 0.5,
    pad_frac: float = 0.08,
    max_crop_size: Optional[int] = 1024,
    resume: bool = False,
    force: bool = False,
    api_key: Optional[str] = None,
    client: Any = None,
    responder: Optional[Callable[[str, int, int], str]] = None,
    cost_per_mtok_in: float = 3.0,
    cost_per_mtok_out: float = 15.0,
    log: Callable[[str], None] = print,
) -> Dict[str, Any]:
    """Localize the target towel in each image, crop tightly, and write a crop dataset.

    Returns a status dict. ``status='no_api_key'`` (with setup instructions) when no
    key/transport is available — it does NOT raise and does NOT print the key. Images
    where Claude does not confidently find the target (``is_target=false`` / not found
    / low confidence / bad box) are recorded as REJECTED and produce no crop.
    """
    if not os.path.isdir(input_dir):
        return {"status": "error", "message": f"input not found: {input_dir}"}
    src = load_input_images(input_dir, default_source="external")
    images = src.get("images", [])
    if not images:
        return {"status": "error",
                "message": f"no images found in {input_dir} "
                           "(expected a manifest, an images/ folder, or image files)"}

    locator = ClaudeTowelLocator(target_description=target_description, model=model,
                                 api_key=api_key, client=client, responder=responder)
    if not locator.have_transport():
        return {"status": "no_api_key", "instructions": KEY_HINT.splitlines()}

    images_dir = os.path.join(out, "images")
    overlays_dir = os.path.join(out, "localize_overlays")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(overlays_dir, exist_ok=True)

    reuse = resume and not force
    prior = _load_existing_crops(out) if reuse else {}
    # Reserve every id a resumed entry still owns so a NEW crop never reuses that id
    # and clobbers an existing images/{sid}.png whose manifest entry still points to it.
    _next_sid = _sid_minter({e.get("id") for e in prior.values() if e.get("id")})

    crops: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    counts = {"cropped": 0, "rejected": 0, "errors": 0, "resumed": 0}
    tok_in = tok_out = 0
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
        h = im.get("hash")

        # Resume: reuse an existing crop for the same source image (by hash).
        if reuse and h and h in prior:
            crops.append(dict(prior[h]))
            counts["resumed"] += 1
            continue

        w, hh = im.get("width"), im.get("height")
        if not (w and hh):
            size = image_size(src_img)
            if size:
                w, hh = size
        if not (w and hh):
            counts["errors"] += 1
            log(f"[warn] unknown image size for {rel}; skipping (cannot localize).")
            continue

        try:
            raw = locator.raw_response(src_img, int(w), int(hh))
            loc = parse_localization(raw, int(w), int(hh), min_confidence=min_confidence)
        except ClaudeTowelError as exc:
            counts["errors"] += 1
            log(f"[warn] {rel}: {exc}")
            continue
        except Exception as exc:  # transport / unexpected — keep going
            counts["errors"] += 1
            log(f"[warn] {rel}: localization call failed: {exc}")
            continue

        tok_in += locator.last_usage.get("input_tokens", 0)
        tok_out += locator.last_usage.get("output_tokens", 0)

        if not loc["ok"]:
            counts["rejected"] += 1
            rejected.append({
                "image": rel, "source": image_source, "hash": h,
                "reject_reason": loc["reject_reason"], "confidence": loc["confidence"],
                "reason": loc["reason"], "rejected_regions": loc["rejected_regions"],
            })
            log(f"[reject] {rel}: {loc['reject_reason']} (conf {loc['confidence']:.2f})")
            continue

        box = compute_crop_box(loc["bbox"], int(w), int(hh), pad_frac=pad_frac)
        sid = _next_sid()  # collision-free; minted only once we will actually write a crop
        crop_rel = f"images/{sid}.png"
        crop_path = os.path.join(out, crop_rel)
        try:
            import numpy as np

            from terafold.vision.imageio import imread, imwrite

            arr = np.array(imread(src_img), dtype=np.uint8)
            crop_arr, scale_x, scale_y = _crop_array(arr, box, max_crop_size)
            imwrite(crop_path, crop_arr)
        except Exception as exc:  # decode/encode needs a backend for JPEG
            counts["errors"] += 1
            log(f"[warn] {rel}: could not crop (need cv2/Pillow for this format?): {exc}")
            continue

        cx1, cy1, cx2, cy2 = box
        cw, ch = crop_arr.shape[1], crop_arr.shape[0]
        crop_meta = {
            "orig_image": os.path.abspath(src_img),
            "orig_width": int(w), "orig_height": int(hh),
            "x_off": cx1, "y_off": cy1,
            "region_width": cx2 - cx1, "region_height": cy2 - cy1,
            "crop_width": int(cw), "crop_height": int(ch),
            "scale_x": scale_x, "scale_y": scale_y,
            "pad_frac": float(pad_frac),
            "detected_bbox": [float(v) for v in loc["bbox"]],
            "target_description": target_description or DEFAULT_TARGET_DESCRIPTION,
            "localize_confidence": loc["confidence"],
            "localize_reason": loc["reason"],
        }

        overlay_rel = f"localize_overlays/{sid}_localize.png"
        rendered = _localize_overlay(src_img, loc["bbox"], box,
                                     os.path.join(out, overlay_rel))

        try:
            crop_hash = image_hash(crop_path)
        except OSError:
            crop_hash = None

        crops.append({
            "id": sid,
            "image": crop_rel,
            "original": os.path.abspath(src_img),
            "hash": crop_hash,
            "source_hash": h,                  # hash of the ORIGINAL (for resume)
            "width": int(cw), "height": int(ch),
            "source": image_source,
            "localize_overlay": overlay_rel if rendered else None,
            "crop": crop_meta,
        })
        counts["cropped"] += 1
        log(f"[crop] {rel} -> {crop_rel}  box={box} conf={loc['confidence']:.2f}")

    est_cost = (tok_in / 1e6) * cost_per_mtok_in + (tok_out / 1e6) * cost_per_mtok_out
    manifest = {
        "dataset": os.path.basename(os.path.normpath(out)),
        "kind": CROPS_MANIFEST_KIND,
        "source": src.get("source"),
        "input_dataset": os.path.basename(os.path.normpath(input_dir)),
        "target_description": target_description or DEFAULT_TARGET_DESCRIPTION,
        "claude_model": model,
        "min_confidence": float(min_confidence),
        "pad_frac": float(pad_frac),
        "max_crop_size": max_crop_size,
        "forced": bool(force),
        "num_input": len(images),
        "num_images": len(crops),
        "num_cropped": counts["cropped"],
        "num_rejected": counts["rejected"],
        "num_errors": counts["errors"],
        "num_resumed": counts["resumed"],
        "usage": {"input_tokens": tok_in, "output_tokens": tok_out},
        "cost_estimate_usd": round(est_cost, 4),
        "cost_assumption": {"per_mtok_in": cost_per_mtok_in, "per_mtok_out": cost_per_mtok_out,
                            "note": "rough estimate; verify current Anthropic pricing"},
        "note": "Crops are localized to the target towel; label them, then corners are "
                "remapped back to the original full image via each sample's 'crop' metadata.",
        "images": crops,
        "rejected": rejected,
    }
    write_json(os.path.join(out, "manifest.json"), manifest)
    _write_readme(out, manifest)
    log(f"Cropped {counts['cropped']} towel candidates -> {out} "
        f"({counts['rejected']} rejected, {counts['errors']} errors, {counts['resumed']} resumed)")
    if tok_in or tok_out:
        log(f"  tokens in/out: {tok_in}/{tok_out}  est cost ~${est_cost:.3f} (approx)")
    return {
        "status": "ok",
        "out": out,
        "num_input": len(images),
        "num_crops": len(crops),           # total crops in the dataset (incl. resumed)
        "num_new": counts["cropped"],      # newly cropped this run
        "num_rejected": counts["rejected"],
        "errors": counts["errors"],
        "resumed": counts["resumed"],
        "usage": manifest["usage"],
        "cost_estimate_usd": manifest["cost_estimate_usd"],
        "manifest": os.path.join(out, "manifest.json"),
    }


def _load_existing_crops(out: str) -> Dict[str, Any]:
    """Map ``original-image hash -> crop entry`` for an existing crops manifest."""
    man_path = os.path.join(out, "manifest.json")
    if not os.path.exists(man_path):
        return {}
    from terafold.data.episode_schema import read_json

    try:
        man = read_json(man_path)
    except Exception:
        return {}
    prior: Dict[str, Any] = {}
    for e in man.get("images", []):
        sh = e.get("source_hash")
        if sh and os.path.exists(os.path.join(out, e.get("image", ""))):
            prior[sh] = e
    return prior


def _write_readme(root: str, manifest: Dict[str, Any]) -> None:
    lines = [
        f"# Towel crop candidates — {manifest['dataset']}",
        "",
        "> Crop-first localization: Claude was asked ONLY for the bounding box of "
        f"the **{manifest['target_description']}** (ignoring bed/pillow/blanket/headboard/"
        "floor/body), then a tight padded crop was saved. Each crop carries `crop` "
        "metadata so corner labels are remapped back to the original full image.",
        "",
        f"- model: {manifest['claude_model']}",
        f"- crops: {manifest['num_images']}  "
        f"(rejected: {manifest['num_rejected']}, errors: {manifest['num_errors']})",
        f"- min confidence: {manifest['min_confidence']}  pad: {manifest['pad_frac']}",
        "",
        "## Next — label the crops (corners remap back to full images automatically)",
        "```bash",
        f"python3 -m terafold claude-label-towel-folder --input {root} \\",
        "  --out data/towel_real_labeled_v1_crops --min-confidence 0.65 \\",
        "  --label-policy outer_visible_corners_striped_towel_only --force",
        "```",
    ]
    with open(os.path.join(root, "README.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
