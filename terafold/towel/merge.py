"""Merge towel datasets into one, de-duplicating and splitting train/val/test.

* Only **usable, labeled** samples are merged by default (4 corners + trainable
  state). ``--real-only`` additionally drops synthetic samples.
* Duplicates are removed by image content hash (across all inputs).
* The split is deterministic and *stable per image* (bucketed by hash) so the same
  photo always lands in the same split, even as the dataset grows.
"""

from __future__ import annotations

import os
import shutil
from typing import Any, Callable, Dict, List, Tuple

from terafold.data.episode_schema import write_json
from terafold.towel.dataset import dataset_paths, iter_usable
from terafold.towel.schema import SOURCE_GROUP, SOURCE_GROUPS, TOWEL_STATES

__all__ = ["merge_towel_datasets", "split_for_hash"]


def _noop(_m: str) -> None:
    pass


def split_for_hash(h: str, ratios: Tuple[float, float, float] = (0.7, 0.2, 0.1)) -> str:
    """Deterministically bucket a content hash into train/val/test."""
    bucket = int(h[:8], 16) % 100
    train_cut = ratios[0] * 100
    val_cut = (ratios[0] + ratios[1]) * 100
    if bucket < train_cut:
        return "train"
    if bucket < val_cut:
        return "val"
    return "test"


def merge_towel_datasets(
    inputs: List[str],
    out: str,
    real_only: bool = False,
    split: Tuple[float, float, float] = (0.7, 0.2, 0.1),
    log: Callable[[str], None] = _noop,
) -> Dict[str, Any]:
    """Merge ``inputs`` (dataset dirs) into ``out``. Returns a summary dict."""
    if not inputs:
        raise ValueError("no input datasets given")
    paths = dataset_paths(out)
    os.makedirs(paths["images"], exist_ok=True)
    os.makedirs(paths["labels"], exist_ok=True)

    seen: Dict[str, str] = {}
    samples: List[Dict[str, Any]] = []
    by_split = {"train": 0, "val": 0, "test": 0}
    by_source = {g: 0 for g in SOURCE_GROUPS}
    dropped_dupes = 0
    dropped_synth = 0
    next_id = 1

    for ds in inputs:
        ds = ds.strip()
        if not ds:
            continue
        if not os.path.exists(dataset_paths(ds)["manifest"]):
            log(f"[warn] no manifest in {ds}; skipping.")
            continue
        for sample, label in iter_usable(ds):
            source = sample.get("source") or label.get("source") or "user_photo"
            if real_only and source == "synthetic":
                dropped_synth += 1
                continue
            h = sample.get("hash") or ""
            if h and h in seen:
                dropped_dupes += 1
                continue
            sid = f"{next_id:06d}"
            ext = os.path.splitext(sample["image"])[1].lower() or ".jpg"
            img_rel = f"images/{sid}{ext}"
            lab_rel = f"labels/{sid}.json"
            try:
                shutil.copyfile(os.path.join(ds, sample["image"]), os.path.join(out, img_rel))
            except OSError:
                log(f"[warn] could not copy {sample['image']} from {ds}; skipping.")
                continue
            new_label = dict(label)
            new_label["image"] = img_rel
            new_label["source"] = source
            write_json(os.path.join(out, lab_rel), new_label)
            sp = split_for_hash(h or sid, split)
            group = SOURCE_GROUP.get(source, "external")
            merged_sample = {
                "id": sid, "image": img_rel, "label": lab_rel, "hash": h,
                "width": sample.get("width"), "height": sample.get("height"),
                "source": source, "source_group": group, "split": sp,
                "state": new_label.get("state"), "origin_dataset": os.path.basename(os.path.normpath(ds)),
            }
            # Preserve generated-image lineage (training track) so track-aware export
            # / critic separation keeps working after a merge.
            track = sample.get("training_track") or new_label.get("training_track")
            if track is not None:
                merged_sample["training_track"] = track
            samples.append(merged_sample)
            if h:
                seen[h] = sid
            by_split[sp] += 1
            by_source[group] = by_source.get(group, 0) + 1
            next_id += 1

    manifest = {
        "dataset": os.path.basename(os.path.normpath(out)),
        "kind": "merged",
        "label_schema": "towel_v1",
        "states": TOWEL_STATES,
        "real_only": bool(real_only),
        "split_ratios": list(split),
        "num_images": len(samples),
        "by_split": by_split,
        "by_source": by_source,
        "dropped_duplicates": dropped_dupes,
        "dropped_synthetic": dropped_synth,
        "inputs": [os.path.basename(os.path.normpath(d)) for d in inputs if d.strip()],
        "samples": samples,
    }
    write_json(paths["manifest"], manifest)
    _write_readme(out, manifest)
    log(f"Merged {len(samples)} usable samples -> {out}  "
        f"(train/val/test = {by_split['train']}/{by_split['val']}/{by_split['test']}, "
        f"by source {by_source}, dropped {dropped_dupes} dupes"
        + (f", {dropped_synth} synthetic" if real_only else "") + ")")
    return {"out": out, "num_images": len(samples), "by_split": by_split,
            "by_source": by_source, "dropped_duplicates": dropped_dupes,
            "dropped_synthetic": dropped_synth, "manifest": paths["manifest"],
            "real_only": bool(real_only)}


def _write_readme(root: str, manifest: Dict[str, Any]) -> None:
    paths = dataset_paths(root)
    lines = [
        f"# Merged towel dataset — {manifest['dataset']}",
        "",
        f"- merged from: {', '.join(manifest['inputs'])}",
        f"- real_only: {manifest['real_only']}",
        f"- usable samples: {manifest['num_images']}",
        f"- split (train/val/test): {manifest['by_split']['train']}/"
        f"{manifest['by_split']['val']}/{manifest['by_split']['test']}",
        f"- by source: {manifest['by_source']}",
        f"- dropped duplicates: {manifest['dropped_duplicates']}  "
        f"dropped synthetic: {manifest['dropped_synthetic']}",
        "",
        "Only usable, labeled samples are included. Export to YOLO pose:",
        "",
        "```bash",
        f"python3 -m terafold export-yolo-towel-pose --data {root} --out data/yolo_towel_pose_v0",
        "```",
    ]
    with open(paths["readme"], "w") as f:
        f.write("\n".join(lines) + "\n")
