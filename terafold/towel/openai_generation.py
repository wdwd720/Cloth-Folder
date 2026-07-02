"""OpenAI image generation for the towel realism-bridge layer (Layer B).

Generates photorealistic hotel-towel images from the prompt curriculum
(:mod:`terafold.towel.prompt_bank`) into a generated dataset
(:mod:`terafold.towel.generated_dataset`), with a *safe, cost-aware* workflow:

* **dry-run** (default-friendly): builds + previews prompts and estimates cost
  with NO API calls and NO key required;
* **real** mode is heavily gated — it refuses unless ALL of: a key is present
  (`OPENAI_API_KEY`, read from env only, never printed), the explicit
  ``confirm`` acknowledgement is given, AND the predicted cost is within
  ``max_cost_usd``;
* writes the manifest **incrementally** (after each image) so an interrupted run
  can ``resume`` without re-paying for images already saved;
* per-image API failures are caught and recorded; a failure on the very first
  image aborts early (so a bad model name / auth never burns a whole budget).

The OpenAI SDK is a LAZY, optional import (it is not a core dependency). Tests
drive generation through the ``image_responder`` / ``client`` injection seams and
never make real paid calls.
"""

from __future__ import annotations

import base64
import os
from typing import Any, Callable, Dict, List, Optional

from terafold.towel import costing
from terafold.towel.generated_dataset import (gen_dataset_paths, init_generated_manifest,
                                              load_generated_manifest, new_generated_sample,
                                              save_generated_manifest)
from terafold.towel.prompt_bank import (DEFAULT_TASK, allocate_counts, build_prompt_curriculum,
                                        list_tasks)
from terafold.towel.schema import image_hash, image_size

__all__ = ["DEFAULT_MODEL", "CONFIRM_FLAG", "KEY_HINT", "SDK_HINT",
           "generate_openai_towel_dataset"]

DEFAULT_MODEL = costing.DEFAULT_MODEL  # "gpt-image-1"
CONFIRM_FLAG = "--yes-i-understand-this-uses-paid-api"

KEY_HINT = (
    "OPENAI_API_KEY is not set. Real image generation needs an OpenAI API key:\n"
    "  export OPENAI_API_KEY=...        # https://platform.openai.com/api-keys\n"
    "The key is read only from the environment — never stored or printed.\n"
    "Tip: run with --dry-run first to preview prompts + cost (no key, no charge)."
)
SDK_HINT = "The OpenAI SDK is required for real generation:\n  python3 -m pip install openai"


def _noop(_m: str) -> None:
    pass


def _utc_now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def generate_openai_towel_dataset(
    out: str,
    num_images: int,
    model: str = DEFAULT_MODEL,
    size: Any = 1024,
    task: str = DEFAULT_TASK,
    quality: str = costing.DEFAULT_QUALITY,
    seed: int = 0,
    dry_run: bool = True,
    max_cost_usd: Optional[float] = None,
    confirm: bool = False,
    resume: bool = False,
    preview_n: int = 12,
    api_key: Optional[str] = None,
    client: Any = None,
    image_responder: Optional[Callable[[str, str], bytes]] = None,
    clock: Callable[[], str] = _utc_now_iso,
    log: Callable[[str], None] = print,
) -> Dict[str, Any]:
    """Generate (or dry-run preview) an OpenAI towel dataset at ``out``.

    Returns a status dict; never raises on a missing key / SDK / budget breach and
    never prints the API key. ``status`` is one of: ``dry_run``, ``no_api_key``,
    ``needs_confirmation``, ``cost_exceeded``, ``sdk_missing``, ``generation_failed``,
    ``error``, or ``ok``.
    """
    if task not in list_tasks():
        return {"status": "error", "message": f"unknown task {task!r}", "tasks": list_tasks()}
    if int(num_images) <= 0:
        return {"status": "error", "message": "num_images must be > 0"}

    size_str = costing.size_string(size)
    records = build_prompt_curriculum(task, int(num_images), seed=int(seed))
    estimate = costing.estimate_cost(model, size, len(records), quality=quality)
    by_category = allocate_counts(task, int(num_images))

    # ---- DRY RUN: preview prompts + cost, no key, no API calls ---------------
    if dry_run:
        paths = gen_dataset_paths(out)
        os.makedirs(paths["root"], exist_ok=True)
        manifest = init_generated_manifest(out, model, size_str, task, seed, estimate, dry_run=True)
        manifest["num_requested"] = len(records)
        manifest["by_category"] = by_category
        manifest["planned_prompts"] = records
        manifest["preview"] = records[:preview_n]
        save_generated_manifest(out, manifest)
        log(f"[dry-run] would generate {len(records)} images ({task}, {size_str}) "
            f"with {model}; est cost ~${estimate['total_usd']} ({'known' if estimate['known_pricing'] else 'UNKNOWN'} pricing).")
        log(f"  by category: {by_category}")
        if max_cost_usd is not None and not costing.within_budget(estimate, max_cost_usd):
            log(f"  ⚠️ est ${estimate['total_usd']} EXCEEDS --max-cost-usd {max_cost_usd}; "
                "real run would refuse. Lower --num-images or raise the cap.")
        return {"status": "dry_run", "out": out, "num_requested": len(records),
                "by_category": by_category, "cost_estimate": estimate,
                "preview": records[:preview_n], "manifest": paths["manifest"]}

    # ---- REAL MODE gates -----------------------------------------------------
    injected = image_responder is not None or client is not None
    if not injected:
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            return {"status": "no_api_key", "instructions": KEY_HINT.splitlines()}
    if not confirm:
        return {
            "status": "needs_confirmation",
            "cost_estimate": estimate,
            "message": (f"Real generation of {len(records)} images would cost ~"
                        f"${estimate['total_usd']} (approx). Re-run with {CONFIRM_FLAG} "
                        "to proceed, or use --dry-run to preview."),
            "confirm_flag": CONFIRM_FLAG,
        }
    if not costing.within_budget(estimate, max_cost_usd):
        return {
            "status": "cost_exceeded",
            "cost_estimate": estimate,
            "max_cost_usd": max_cost_usd,
            "message": (f"Predicted cost ${estimate['total_usd']} exceeds --max-cost-usd "
                        f"{max_cost_usd}. Lower --num-images or raise the cap."),
        }

    # Resolve a client lazily (only if no responder was injected).
    gen_client = client
    if image_responder is None and gen_client is None:
        try:
            import openai  # noqa: F401
        except Exception:
            return {"status": "sdk_missing", "instructions": SDK_HINT.splitlines()}
        try:
            from openai import OpenAI

            gen_client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        except Exception as exc:
            return {"status": "error", "message": f"could not init OpenAI client: {exc}"}

    # ---- generate ------------------------------------------------------------
    paths = gen_dataset_paths(out)
    os.makedirs(paths["images"], exist_ok=True)
    prior = _load_prior(out) if resume else {}
    manifest = init_generated_manifest(out, model, size_str, task, seed, estimate, dry_run=False)
    manifest["num_requested"] = len(records)

    samples: List[Dict[str, Any]] = []
    counts = {"generated": 0, "resumed": 0, "errors": 0}
    cost_actual = 0.0
    unit_cost, _known = costing.per_image_usd(model, size, quality)

    for i, rec in enumerate(records):
        sid = f"{i + 1:06d}"
        img_rel = f"images/{sid}.png"
        img_abs = os.path.join(out, img_rel)

        if resume and sid in prior and os.path.exists(os.path.join(out, prior[sid].get("image", ""))):
            samples.append(prior[sid])
            counts["resumed"] += 1
            cost_actual += float(prior[sid].get("cost_estimate", 0.0) or 0.0)
            _save_progress(out, manifest, samples, counts, cost_actual)
            continue

        try:
            img_bytes = (image_responder(rec["prompt"], size_str) if image_responder is not None
                         else _generate_image_bytes(gen_client, model, rec["prompt"], size_str))
            if not img_bytes:
                raise RuntimeError("empty image bytes returned")
            with open(img_abs, "wb") as f:
                f.write(img_bytes)
        except Exception as exc:
            counts["errors"] += 1
            msg = _sanitize(str(exc))
            log(f"[warn] image {sid} failed: {msg}")
            # Fail fast on the very first image (bad model/auth/SDK) — don't burn budget.
            if counts["generated"] == 0 and counts["resumed"] == 0:
                _save_progress(out, manifest, samples, counts, cost_actual)
                return {"status": "generation_failed", "out": out,
                        "message": f"first image failed: {msg}", "errors": counts["errors"],
                        "manifest": paths["manifest"]}
            continue

        size_wh = image_size(img_abs)
        w, h = size_wh if size_wh else (None, None)
        sample = new_generated_sample(
            sid, img_rel, rec, model, image_hash(img_abs), w, h,
            cost_estimate=unit_cost, generation_time=clock(), size_str=size_str)
        samples.append(sample)
        counts["generated"] += 1
        cost_actual += unit_cost
        _save_progress(out, manifest, samples, counts, cost_actual)

    _write_readme(out, manifest)
    log(f"Generated {counts['generated']} images "
        f"({counts['resumed']} resumed, {counts['errors']} errors) -> {out} "
        f"(~${round(cost_actual, 4)} actual).")
    log(f"  by category: {manifest['by_category']}")
    return {"status": "ok", "out": out, "num_images": counts["generated"],
            "resumed": counts["resumed"], "errors": counts["errors"],
            "cost_actual_usd": round(cost_actual, 4), "cost_estimate": estimate,
            "by_category": manifest["by_category"], "manifest": paths["manifest"]}


# --------------------------------------------------------------------------
# Transport + helpers
# --------------------------------------------------------------------------


def _generate_image_bytes(client: Any, model: str, prompt: str, size_str: str) -> bytes:
    """Call the OpenAI images API and return raw image bytes (PNG)."""
    kwargs: Dict[str, Any] = {"model": model, "prompt": prompt, "size": size_str, "n": 1}
    # gpt-image-1 always returns b64 and rejects response_format; dall-e accepts it.
    if not str(model).startswith("gpt-image"):
        kwargs["response_format"] = "b64_json"
    resp = client.images.generate(**kwargs)
    data = getattr(resp, "data", None) or (resp.get("data") if isinstance(resp, dict) else None)
    if not data:
        raise RuntimeError("OpenAI response had no image data")
    d0 = data[0]
    b64 = getattr(d0, "b64_json", None) or (d0.get("b64_json") if isinstance(d0, dict) else None)
    if b64:
        return base64.b64decode(b64)
    url = getattr(d0, "url", None) or (d0.get("url") if isinstance(d0, dict) else None)
    if url:
        import urllib.request

        with urllib.request.urlopen(url, timeout=60) as r:  # noqa: S310 (https from OpenAI)
            return r.read()
    raise RuntimeError("OpenAI response had neither b64_json nor url")


def _save_progress(out, manifest, samples, counts, cost_actual) -> None:
    manifest["images"] = samples
    manifest["num_images"] = counts["generated"] + counts["resumed"]
    manifest["errors"] = counts["errors"]
    manifest["cost_actual_usd"] = round(cost_actual, 4)
    by_cat: Dict[str, int] = {}
    for s in samples:
        by_cat[s.get("category_target", "?")] = by_cat.get(s.get("category_target", "?"), 0) + 1
    manifest["by_category"] = by_cat
    save_generated_manifest(out, manifest)


def _load_prior(out: str) -> Dict[str, Any]:
    man_path = gen_dataset_paths(out)["manifest"]
    if not os.path.exists(man_path):
        return {}
    try:
        man = load_generated_manifest(out)
    except Exception:
        return {}
    return {s["id"]: s for s in man.get("images", []) if s.get("id")}


def _sanitize(msg: str) -> str:
    """Best-effort redaction so an error string can't leak a key."""
    key = os.environ.get("OPENAI_API_KEY")
    if key and key in msg:
        msg = msg.replace(key, "***")
    return msg[:300]


def _write_readme(root: str, manifest: Dict[str, Any]) -> None:
    paths = gen_dataset_paths(root)
    lines = [
        f"# OpenAI-generated towel dataset — {manifest['dataset']}",
        "",
        "> ⚠️ These images are **OpenAI-generated** (a realism *bridge* between "
        "synthetic and real data). They are NOT ground truth: every sample starts "
        "`usable_for_training: false`. Pseudo-label + human-review before export.",
        "",
        f"- model: {manifest['generation_model']}  size: {manifest['size']}  "
        f"task: {manifest['task']}  seed: {manifest['seed']}",
        f"- images: {manifest.get('num_images', 0)}  "
        f"(~${manifest.get('cost_actual_usd', 0)} approx actual cost)",
        f"- by category: {manifest.get('by_category', {})}",
        "",
        "## Next",
        "```bash",
        f"python3 -m terafold filter-generated-towel-images --input {root} --out data/towel_openai_qc_v0",
        f"python3 -m terafold claude-label-towel-folder --input {root} --out data/towel_openai_labeled_v0 --min-confidence 0.75",
        "```",
    ]
    with open(paths["readme"], "w") as f:
        f.write("\n".join(lines) + "\n")
