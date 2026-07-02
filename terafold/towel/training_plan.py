"""Recommended staged training strategy + dataset-composition report.

Encodes the recommended 3-layer data strategy (synthetic → +OpenAI-generated →
+approved real/external) as an explicit, inspectable plan, and reports how the
*current* dataset compares to the recommended mix. Works on either a towel
dataset/merged manifest (``manifest.json`` with ``samples``) or an exported YOLO
pose dataset (``towel_pose.yaml`` with per-split label files).

The recommended counts are *ranges*, surfaced as guidance (not silently
hardcoded thresholds) — they can be overridden by the caller.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional

from terafold.data.episode_schema import read_json
from terafold.towel.dataset import dataset_paths, load_label
from terafold.towel.schema import (SOURCE_GROUPS, is_approved_pseudolabel, is_pseudolabel,
                                   is_usable, source_group)

__all__ = ["RECOMMENDED_MIX", "STAGES", "recommend_towel_training_plan"]

# Recommended first-pass target RANGES (guidance, overridable — not rigid gates).
RECOMMENDED_MIX = {
    "synthetic": (1000, 2000),
    "openai_generated": (500, 2000),
    "real_external_approved": (50, 300),
    "gold_eval": (30, 100),
}

STAGES = [
    ("Stage 0 — baseline", "Train on synthetic only. Sanity-check the pipeline + a "
     "smoke-quality detector. This is a baseline, not the deliverable."),
    ("Stage 1 — realism bridge", "Add OpenAI-generated (approved) images to synthetic. "
     "Expect a jump on realistic lighting/texture vs synthetic-only."),
    ("Stage 2 — real fine-tune", "Add approved real/external towels. This is what "
     "closes the gap to true hotel scenes."),
    ("Always", "Keep a REAL-ONLY validation set and a small curated GOLD eval set that "
     "are NEVER trained on, so generated data can't inflate your metrics."),
]


def _noop(_m: str) -> None:
    pass


def _composition_from_towel_dataset(data: str) -> Dict[str, Any]:
    man = read_json(dataset_paths(data)["manifest"])
    samples = man.get("samples", [])
    by_source = {g: 0 for g in SOURCE_GROUPS}
    usable = pseudo = approved_pseudo = 0
    for s in samples:
        try:
            label = load_label(data, s)
        except FileNotFoundError:
            continue
        by_source[source_group(s, label)] = by_source.get(source_group(s, label), 0) + 1
        if is_usable(label):
            usable += 1
        if is_pseudolabel(label):
            pseudo += 1
            if is_approved_pseudolabel(label):
                approved_pseudo += 1
    return {"kind": man.get("kind", "towel"), "total": len(samples), "by_source": by_source,
            "usable_for_training": usable, "pseudolabels": pseudo,
            "approved_pseudolabels": approved_pseudo}


def _composition_from_yolo(data: str) -> Dict[str, Any]:
    by_split = {}
    for split in ("train", "val", "test"):
        lab_dir = os.path.join(data, "labels", split)
        n = len([f for f in os.listdir(lab_dir) if f.endswith(".txt")]) if os.path.isdir(lab_dir) else 0
        if n:
            by_split[split] = n
    return {"kind": "yolo_pose", "total": sum(by_split.values()), "by_split": by_split}


def recommend_towel_training_plan(
    data: str, out: Optional[str] = None, log: Callable[[str], None] = print,
    recommended: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Report dataset composition + print the recommended staged training plan."""
    rec = recommended or RECOMMENDED_MIX
    if os.path.exists(dataset_paths(data)["manifest"]):
        comp = _composition_from_towel_dataset(data)
    elif os.path.exists(os.path.join(data, "towel_pose.yaml")):
        comp = _composition_from_yolo(data)
    else:
        return {"status": "error",
                "message": f"{data} is neither a towel dataset (manifest.json) "
                           "nor a YOLO pose dataset (towel_pose.yaml)"}

    gaps = _gaps(comp, rec)
    log(f"Dataset {os.path.basename(os.path.normpath(data))} ({comp['kind']}): "
        f"{comp['total']} samples")
    if "by_source" in comp:
        log("  by source: " + " ".join(f"{g}={n}" for g, n in comp["by_source"].items() if n))
        log(f"  usable_for_training: {comp['usable_for_training']}  "
            f"pseudolabels: {comp['pseudolabels']} (approved: {comp['approved_pseudolabels']})")
    if "by_split" in comp:
        log(f"  by split: {comp['by_split']}")
    log("")
    log("Recommended staged plan:")
    for name, desc in STAGES:
        log(f"  • {name}: {desc}")
    log("")
    log("Recommended first-pass target ranges (guidance):")
    for k, (lo, hi) in rec.items():
        log(f"  - {k}: {lo}–{hi}")
    if gaps:
        log("")
        log("Gaps vs recommended ranges:")
        for g in gaps:
            log(f"  ! {g}")
    log("")
    log("Then print the actual command (does NOT auto-launch training):")
    log("  python3 -m terafold print-towel-training-command --data <yolo_pose_dir> "
        "--model yolo11n-pose.pt --epochs 100 --imgsz 640")
    log("  # RunPod: run the SAME ultralytics command on a GPU pod after uploading the dataset.")

    result = {"status": "ok", "data": data, "composition": comp,
              "recommended": rec, "gaps": gaps, "stages": [s[0] for s in STAGES]}
    if out:
        _write_md(out, data, comp, rec, gaps)
        result["report"] = out
    return result


def _gaps(comp: Dict[str, Any], rec: Dict[str, Any]) -> List[str]:
    gaps: List[str] = []
    by_source = comp.get("by_source")
    if not by_source:
        return gaps
    synth = by_source.get("synthetic", 0)
    gen = by_source.get("openai_generated", 0)
    real = (by_source.get("real_user", 0) + by_source.get("external", 0)
            + by_source.get("kaggle", 0) + by_source.get("openimages", 0)
            + by_source.get("roboflow", 0))
    checks = [("synthetic", synth), ("openai_generated", gen), ("real_external_approved", real)]
    for key, have in checks:
        lo, hi = rec[key]
        if have < lo:
            gaps.append(f"{key}: have {have}, recommended ≥ {lo}")
    return gaps


def _write_md(out: str, data: str, comp: Dict[str, Any], rec, gaps) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    lines = [f"# Recommended towel training plan — {os.path.basename(os.path.normpath(data))}", ""]
    lines.append(f"- dataset kind: {comp['kind']}  total: {comp['total']}")
    if "by_source" in comp:
        lines.append(f"- by source: {comp['by_source']}")
        lines.append(f"- usable: {comp['usable_for_training']}  "
                     f"approved pseudolabels: {comp['approved_pseudolabels']}")
    if "by_split" in comp:
        lines.append(f"- by split: {comp['by_split']}")
    lines += ["", "## Staged strategy"]
    lines += [f"{i}. **{n}** — {d}" for i, (n, d) in enumerate(STAGES)]
    lines += ["", "## Recommended first-pass target ranges (guidance)"]
    lines += [f"- {k}: {lo}–{hi}" for k, (lo, hi) in rec.items()]
    if gaps:
        lines += ["", "## Gaps", *[f"- {g}" for g in gaps]]
    lines += ["", "> Generated data is a realism bridge, not ground truth. Keep a real-only "
              "validation set + curated gold eval set that are never trained on."]
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
