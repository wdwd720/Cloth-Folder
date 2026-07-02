"""Build a scene-usability *critic* dataset from a reviewed towel dataset.

The mixed / negative towel images (folded stacks, rolled, multiple, hanging,
not-a-towel, occluded) are useless for the 4-corner YOLO pose model but valuable
for a future **rejection / scene critic** that decides "is this a single flat
foldable towel?" before pose runs. Rather than throw them away, this turns a
reviewed/triaged towel dataset into a simple two-class manifest:

* ``usable``     — approved single flat/wrinkled towels (good pose scenes),
* ``not_usable`` — rejected, critic-negative-track, or non-foldable-state scenes.

Undecided (pending / review_needed) samples are skipped. Images are copied into
``out/images``; ``manifest.json`` records image path, state, approval_status,
rejection reason, source, training_track, and target_class.
"""

from __future__ import annotations

import os
import shutil
from typing import Any, Callable, Dict, List

from terafold.data.episode_schema import read_json, write_json
from terafold.towel.dataset import dataset_paths, load_manifest
from terafold.towel.schema import APPROVED_STATES, TRAINABLE_STATES, label_corners_xy

__all__ = ["build_towel_critic_from_review", "CRITIC_CLASSES"]

CRITIC_CLASSES = ("usable", "not_usable")
_FLAT_STATES = ("flat_unfolded", "wrinkled_unfolded")


def _noop(_m: str) -> None:
    pass


def _target_class(label: Dict[str, Any], track: str) -> str:
    """usable / not_usable / skip(None) for the critic."""
    status = label.get("approval_status")
    state = label.get("state")
    if track == "critic_negative":
        return "not_usable"
    if status == "rejected":
        return "not_usable"
    if status in APPROVED_STATES:
        # Approved AND a genuine flat foldable towel with corners ⇒ usable.
        if state in _FLAT_STATES and label_corners_xy(label) is not None:
            return "usable"
        return "not_usable"
    if state not in TRAINABLE_STATES:   # a confidently non-foldable state, even if unreviewed
        return "not_usable"
    return ""  # undecided ⇒ skip


def _rejection_reason(label: Dict[str, Any]) -> str:
    if label.get("triage_reasons"):
        return ", ".join(str(r) for r in label["triage_reasons"])
    if label.get("risk_reasons"):
        return ", ".join(str(r) for r in label["risk_reasons"])
    return ""


def build_towel_critic_from_review(
    dataset: str, out: str, log: Callable[[str], None] = _noop,
) -> Dict[str, Any]:
    """Build a two-class critic dataset from a reviewed towel ``dataset`` at ``out``."""
    if not os.path.exists(dataset_paths(dataset)["manifest"]):
        return {"status": "error", "message": f"no manifest in dataset: {dataset}"}
    man = load_manifest(dataset)
    images_dir = os.path.join(out, "images")
    os.makedirs(images_dir, exist_ok=True)

    samples: List[Dict[str, Any]] = []
    by_class = {c: 0 for c in CRITIC_CLASSES}
    skipped_undecided = 0
    next_id = 1

    for s in man.get("samples", []):
        try:
            label = read_json(os.path.join(dataset, s["label"]))
        except FileNotFoundError:
            continue
        track = s.get("training_track") or label.get("training_track") or "unspecified"
        cls = _target_class(label, track)
        if not cls:
            skipped_undecided += 1
            continue
        sid = f"{next_id:06d}"
        ext = os.path.splitext(s["image"])[1].lower() or ".png"
        img_rel = f"images/{sid}{ext}"
        try:
            shutil.copyfile(os.path.join(dataset, s["image"]), os.path.join(out, img_rel))
        except OSError:
            continue
        samples.append({
            "id": sid,
            "image": img_rel,
            "state": label.get("state"),
            "approval_status": label.get("approval_status"),
            "rejection_reason": _rejection_reason(label) if cls == "not_usable" else "",
            "source": s.get("source") or label.get("source"),
            "training_track": track,
            "target_class": cls,
        })
        by_class[cls] += 1
        next_id += 1

    manifest = {
        "dataset": os.path.basename(os.path.normpath(out)),
        "kind": "towel_scene_critic",
        "classes": list(CRITIC_CLASSES),
        "built_from": os.path.basename(os.path.normpath(dataset)),
        "num_images": len(samples),
        "by_class": by_class,
        "skipped_undecided": skipped_undecided,
        "note": "Two-class scene-usability critic data derived from reviewed towel "
                "labels. NOT pose data. usable = approved flat/wrinkled single towels; "
                "not_usable = rejected / critic-negative / non-foldable scenes.",
        "samples": samples,
    }
    write_json(os.path.join(out, "manifest.json"), manifest)
    log(f"Built critic dataset: {len(samples)} samples "
        f"(usable={by_class['usable']}, not_usable={by_class['not_usable']}, "
        f"{skipped_undecided} undecided skipped) -> {out}")
    return {"status": "ok", "out": out, "num_images": len(samples), "by_class": by_class,
            "skipped_undecided": skipped_undecided,
            "manifest": os.path.join(out, "manifest.json")}
