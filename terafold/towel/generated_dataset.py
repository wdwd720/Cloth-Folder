"""Dataset structure + manifest schema for OpenAI-generated towel images.

A generated dataset is a *superset* of the raw image dataset
(:mod:`terafold.towel.raw_ingest`), so it can be pseudo-labeled by
``claude-label-towel-folder`` directly — but it carries the full generation
lineage the spec requires per sample (model, prompt, template, category target,
cost, etc.). Layout::

    data/towel_openai_v0/
        images/000001.png ...
        manifest.json            # kind=openai_generated, rich per-sample metadata
        README.md

Generated images are NEVER ground truth: every sample starts ``label_status =
"unlabeled"``, ``approval_status = "pending"``, ``usable_for_training = false``.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional

from terafold.data.episode_schema import read_json, write_json
from terafold.towel.merge import split_for_hash

__all__ = [
    "GENERATED_KIND", "GENERATED_SOURCE", "REQUIRED_SAMPLE_FIELDS",
    "gen_dataset_paths", "new_generated_sample", "init_generated_manifest",
    "load_generated_manifest", "save_generated_manifest", "validate_generated_manifest",
    "summarize_generated_dataset",
]

GENERATED_KIND = "openai_generated"
GENERATED_SOURCE = "openai_generated"

# Per-sample metadata fields the spec requires.
REQUIRED_SAMPLE_FIELDS = (
    "id", "source", "generation_model", "generation_prompt", "prompt_template_name",
    "generation_time", "width", "height", "category_target", "training_track", "split",
    "cost_estimate", "label_status", "approval_status", "usable_for_training",
    "reviewer_notes",
)


def gen_dataset_paths(root: str) -> Dict[str, str]:
    return {
        "root": root,
        "images": os.path.join(root, "images"),
        "manifest": os.path.join(root, "manifest.json"),
        "readme": os.path.join(root, "README.md"),
    }


def new_generated_sample(
    sample_id: str,
    image_rel: str,
    prompt_record: Dict[str, Any],
    generation_model: str,
    image_hash: str,
    width: Optional[int],
    height: Optional[int],
    cost_estimate: float,
    generation_time: Optional[str],
    size_str: str,
) -> Dict[str, Any]:
    """Build one fully-populated generated-sample metadata dict."""
    split = split_for_hash(image_hash or sample_id)
    return {
        "id": sample_id,
        "image": image_rel,
        "original": f"openai_generated:{prompt_record.get('template_name', '?')}",
        "hash": image_hash,
        "width": width,
        "height": height,
        "size": size_str,
        "source": GENERATED_SOURCE,
        "generation_model": generation_model,
        "generation_prompt": prompt_record.get("prompt", ""),
        "prompt_template_name": prompt_record.get("template_name", ""),
        "category_target": prompt_record.get("category", ""),
        "target_state": prompt_record.get("target_state", ""),
        "training_track": prompt_record.get("training_track", "pose_positive"),
        "usable_hint": bool(prompt_record.get("usable_hint", False)),
        "generation_time": generation_time,
        "cost_estimate": round(float(cost_estimate), 5),
        "split": split,
        # Lifecycle: untrusted by default.
        "label_status": "unlabeled",
        "approval_status": "pending",
        "usable_for_training": False,
        "reviewer_notes": "",
    }


def init_generated_manifest(
    root: str, model: str, size_str: str, task: str, seed: int,
    cost_estimate: Dict[str, Any], dry_run: bool,
) -> Dict[str, Any]:
    return {
        "dataset": os.path.basename(os.path.normpath(root)),
        "kind": GENERATED_KIND,
        "source": GENERATED_SOURCE,
        "generation_model": model,
        "size": size_str,
        "task": task,
        "seed": int(seed),
        "dry_run": bool(dry_run),
        "cost_estimate": cost_estimate,
        "cost_actual_usd": 0.0,
        "num_requested": 0,
        "num_images": 0,
        "by_category": {},
        "warning": "OpenAI-generated images are NOT ground truth. Pseudo-label + "
                   "human-review before any export to training.",
        "images": [],
    }


def load_generated_manifest(root: str) -> Dict[str, Any]:
    return read_json(gen_dataset_paths(root)["manifest"])


def save_generated_manifest(root: str, manifest: Dict[str, Any]) -> str:
    p = gen_dataset_paths(root)["manifest"]
    write_json(p, manifest)
    return p


def validate_generated_manifest(manifest: Dict[str, Any]) -> List[str]:
    """Return a list of schema problems (empty ⇒ valid)."""
    problems: List[str] = []
    if not isinstance(manifest, dict):
        return ["manifest is not a JSON object"]
    if manifest.get("kind") != GENERATED_KIND:
        problems.append(f"kind must be {GENERATED_KIND!r}")
    if manifest.get("source") != GENERATED_SOURCE:
        problems.append(f"source must be {GENERATED_SOURCE!r}")
    if not manifest.get("generation_model"):
        problems.append("missing 'generation_model'")
    imgs = manifest.get("images")
    if not isinstance(imgs, list):
        return problems + ["'images' must be a list"]
    for i, im in enumerate(imgs):
        if not isinstance(im, dict):
            problems.append(f"images[{i}] is not an object")
            continue
        for key in REQUIRED_SAMPLE_FIELDS:
            if key not in im:
                problems.append(f"images[{i}] missing {key!r}")
        if im.get("source") != GENERATED_SOURCE:
            problems.append(f"images[{i}] source must be {GENERATED_SOURCE!r}")
        if im.get("usable_for_training") not in (False, True):
            problems.append(f"images[{i}] usable_for_training must be bool")
    return problems


def summarize_generated_dataset(
    root: str, out: Optional[str] = None, log: Callable[[str], None] = lambda _m: None,
) -> Dict[str, Any]:
    """Summarize a generated dataset: counts by category / status / size + cost."""
    manifest = load_generated_manifest(root)
    images = manifest.get("images", [])
    by_category: Dict[str, int] = {}
    by_label_status: Dict[str, int] = {}
    by_approval: Dict[str, int] = {}
    by_template: Dict[str, int] = {}
    by_size: Dict[str, int] = {}
    cost_total = 0.0
    usable = 0
    for im in images:
        by_category[im.get("category_target", "?")] = by_category.get(im.get("category_target", "?"), 0) + 1
        by_label_status[im.get("label_status", "?")] = by_label_status.get(im.get("label_status", "?"), 0) + 1
        by_approval[im.get("approval_status", "?")] = by_approval.get(im.get("approval_status", "?"), 0) + 1
        by_template[im.get("prompt_template_name", "?")] = by_template.get(im.get("prompt_template_name", "?"), 0) + 1
        by_size[im.get("size", "?")] = by_size.get(im.get("size", "?"), 0) + 1
        cost_total += float(im.get("cost_estimate", 0.0) or 0.0)
        if im.get("usable_for_training"):
            usable += 1
    summary = {
        "dataset": manifest.get("dataset"),
        "model": manifest.get("generation_model"),
        "task": manifest.get("task"),
        "num_images": len(images),
        "by_category": by_category,
        "by_label_status": by_label_status,
        "by_approval_status": by_approval,
        "by_size": by_size,
        "num_templates": len(by_template),
        "cost_actual_usd": round(cost_total, 4),
        "cost_estimate": manifest.get("cost_estimate"),
        "usable_for_training": usable,
    }
    log(f"{summary['dataset']}: {summary['num_images']} generated images "
        f"(~${summary['cost_actual_usd']}); by category: {by_category}")
    log(f"  label_status: {by_label_status}  approval: {by_approval}  usable: {usable}")
    if out:
        _write_summary_md(out, root, summary)
        summary["report"] = out
    return summary


def _write_summary_md(out: str, root: str, s: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    lines = [
        f"# Generated towel dataset summary — {s['dataset']}",
        "",
        f"- model: {s['model']}  task: {s['task']}",
        f"- images: {s['num_images']}  (usable_for_training: {s['usable_for_training']})",
        f"- approx actual cost: ${s['cost_actual_usd']}",
        "",
        "## By target category", *[f"- {k}: {v}" for k, v in sorted(s["by_category"].items())],
        "", "## By label status", *[f"- {k}: {v}" for k, v in sorted(s["by_label_status"].items())],
        "", "## By approval status", *[f"- {k}: {v}" for k, v in sorted(s["by_approval_status"].items())],
        "",
        "> Generated images are a realism *bridge*, not ground truth. Pseudo-label, "
        "review, and keep a real validation set. See docs/TOWEL_OPENAI_GENERATION.md.",
    ]
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
