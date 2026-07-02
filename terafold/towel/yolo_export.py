"""Export a labeled towel dataset to YOLO **pose** format (4 corner keypoints).

One class (``towel``), 4 keypoints in tl,tr,br,bl order. Each label line is::

    0 cx cy w h  tlx tly v  trx try v  brx bry v  blx bly v

with everything normalized to ``[0, 1]`` and ``v`` the visibility flag
(2 = visible, 1 = labeled-but-occluded). Unlabeled / reject-state samples are
skipped. ``towel_pose.yaml`` declares ``kpt_shape: [4, 3]``.
"""

from __future__ import annotations

import os
import shutil
from typing import Any, Callable, Dict, List

from terafold.towel.dataset import dataset_paths, load_label, load_manifest
from terafold.towel.merge import split_for_hash
from terafold.towel.schema import (APPROVED_STATES, CORNER_ORDER, SOURCE_GROUPS,
                                   bbox_from_corners, image_size, is_pseudolabel,
                                   is_usable, label_corners_xy, source_group)

__all__ = ["export_yolo_towel_pose", "KPT_SHAPE", "FLIP_IDX", "INCLUDE_PSEUDOLABEL_MODES"]

# How pseudo-labels are treated. Default excludes them entirely; "approved_only"
# lets through pseudo-labels a human/auto-gate approved (unapproved still excluded).
INCLUDE_PSEUDOLABEL_MODES = ("none", "approved_only")

KPT_SHAPE = [4, 3]
# Horizontal-flip keypoint remap: tl<->tr (0<->1), br<->bl (2<->3).
FLIP_IDX = [1, 0, 3, 2]


def _noop(_m: str) -> None:
    pass


def _norm_line(corners_xy: List[List[float]], visible: Dict[str, Any], bbox, w: int, h: int) -> str:
    x1, y1, x2, y2 = bbox
    cx = ((x1 + x2) / 2.0) / w
    cy = ((y1 + y2) / 2.0) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h

    def cl(v):
        return f"{min(1.0, max(0.0, v)):.6f}"

    parts = ["0", cl(cx), cl(cy), cl(bw), cl(bh)]
    for k, (px, py) in zip(CORNER_ORDER, corners_xy):
        vis = 2 if visible.get(k, True) else 1
        parts += [cl(px / w), cl(py / h), str(vis)]
    return " ".join(parts)


def export_yolo_towel_pose(
    data: str, out: str, include_pseudolabels: str = "none",
    include_critic_negatives: bool = False,
    log: Callable[[str], None] = _noop,
) -> Dict[str, Any]:
    """Export usable samples of dataset ``data`` to a YOLO pose dataset at ``out``.

    ``include_pseudolabels``:
      * ``"none"`` (default) — exclude ALL machine pseudo-labels (Claude),
      * ``"approved_only"``  — additionally include pseudo-labels whose
        ``approval_status`` is approved / auto_approved. UNAPPROVED pseudo-labels
        are excluded in every mode.

    ``include_critic_negatives`` (default False): ``training_track == 'critic_negative'``
    samples are NOT pose data and are excluded from the pose export by default;
    pass True only to (re)build a pose set that intentionally includes them.

    The summary reports per-source, per-training-track, and pseudo-label
    approval-status breakdowns.
    """
    if include_pseudolabels not in INCLUDE_PSEUDOLABEL_MODES:
        raise ValueError(f"include_pseudolabels must be one of {INCLUDE_PSEUDOLABEL_MODES}")
    if not os.path.exists(dataset_paths(data)["manifest"]):
        raise FileNotFoundError(f"no manifest in dataset: {data}")
    manifest = load_manifest(data)
    samples = manifest.get("samples", [])

    by_source = {g: 0 for g in SOURCE_GROUPS}
    by_split = {"train": 0, "val": 0, "test": 0}
    by_training_track: Dict[str, int] = {}
    # Approval-status breakdown across the pseudo-labels we saw (any track).
    pseudo_status = {"approved": 0, "auto_approved": 0, "review_needed": 0,
                     "pending": 0, "rejected": 0}
    rejected = 0
    skipped_no_size = 0
    skipped_pseudolabels = 0          # excluded because of the include-pseudolabels policy
    skipped_unapproved_pseudo = 0     # pseudo-labels not yet approved
    skipped_critic_negatives = 0      # excluded because they are critic-track, not pose
    exported: List[Dict[str, Any]] = []

    for s in samples:
        try:
            label = load_label(data, s)
        except FileNotFoundError:
            rejected += 1
            continue
        track = s.get("training_track") or label.get("training_track")
        if is_pseudolabel(label):
            st = label.get("approval_status")
            if st in pseudo_status:
                pseudo_status[st] += 1
        # Critic-negative samples are not pose training data — exclude by default.
        if track == "critic_negative" and not include_critic_negatives:
            skipped_critic_negatives += 1
            continue
        # Pseudo-label gating happens BEFORE the usability check so an accidentally
        # usable-flagged-but-unapproved pseudo-label can never slip through.
        if is_pseudolabel(label):
            if include_pseudolabels != "approved_only":
                skipped_pseudolabels += 1
                continue
            if label.get("approval_status") not in APPROVED_STATES:
                skipped_unapproved_pseudo += 1
                continue
        if not is_usable(label):
            rejected += 1
            continue
        corners = label_corners_xy(label)
        if corners is None:
            rejected += 1
            continue
        w = s.get("width")
        h = s.get("height")
        if not (w and h):
            size = image_size(os.path.join(data, s["image"]))
            if size:
                w, h = size
        if not (w and h):
            skipped_no_size += 1
            log(f"[warn] no image size for {s['image']}; skipped.")
            continue
        bbox = label.get("bbox") or bbox_from_corners(corners)
        split = s.get("split") or split_for_hash(s.get("hash") or s["id"])
        ext = os.path.splitext(s["image"])[1].lower() or ".jpg"

        img_dst_dir = os.path.join(out, "images", split)
        lab_dst_dir = os.path.join(out, "labels", split)
        os.makedirs(img_dst_dir, exist_ok=True)
        os.makedirs(lab_dst_dir, exist_ok=True)
        stem = s["id"]
        try:
            shutil.copyfile(os.path.join(data, s["image"]), os.path.join(img_dst_dir, f"{stem}{ext}"))
        except OSError:
            rejected += 1
            continue
        line = _norm_line(corners, label.get("visible") or {}, bbox, int(w), int(h))
        with open(os.path.join(lab_dst_dir, f"{stem}.txt"), "w") as f:
            f.write(line + "\n")

        group = source_group(s, label)
        by_source[group] = by_source.get(group, 0) + 1
        by_split[split] += 1
        tkey = track or "unspecified"
        by_training_track[tkey] = by_training_track.get(tkey, 0) + 1
        exported.append({"id": stem, "split": split, "source_group": group,
                         "training_track": track})

    yaml_path = _write_yaml(out, by_split)
    summary = {
        "out": out, "yaml": yaml_path, "include_pseudolabels": include_pseudolabels,
        "include_critic_negatives": include_critic_negatives,
        "exported": len(exported), "rejected": rejected, "skipped_no_size": skipped_no_size,
        "skipped_pseudolabels": skipped_pseudolabels,
        "skipped_unapproved_pseudo": skipped_unapproved_pseudo,
        "skipped_critic_negatives": skipped_critic_negatives,
        "by_split": by_split, "by_source": by_source,
        "by_training_track": by_training_track,
        "pseudolabel_approval_status": pseudo_status, "kpt_shape": KPT_SHAPE,
    }
    log(f"Exported {len(exported)} YOLO pose samples -> {out}")
    log("  by source: " + " ".join(f"{g}={by_source[g]}" for g in SOURCE_GROUPS if by_source[g]))
    log(f"  by training_track: {by_training_track}")
    log(f"  by split : train={by_split['train']} val={by_split['val']} test={by_split['test']}")
    log(f"  rejected (unlabeled/bad): {rejected}")
    if skipped_pseudolabels or skipped_unapproved_pseudo or skipped_critic_negatives:
        log(f"  excluded: {skipped_pseudolabels} pseudo (policy='{include_pseudolabels}'), "
            f"{skipped_unapproved_pseudo} unapproved, {skipped_critic_negatives} critic-negative")
    log(f"  pseudo-label approval status: {pseudo_status}")
    return summary


def _write_yaml(out: str, by_split: Dict[str, int]) -> str:
    """Write ``towel_pose.yaml`` (kpt_shape [4,3]; test only if present)."""
    os.makedirs(out, exist_ok=True)
    abs_out = os.path.abspath(out)
    lines = [
        "# YOLO pose dataset — towel (4 corner keypoints: tl, tr, br, bl)",
        f"path: {abs_out}",
        "train: images/train",
        "val: images/val",
    ]
    if by_split.get("test", 0) > 0:
        lines.append("test: images/test")
    lines += [
        "",
        f"kpt_shape: [{KPT_SHAPE[0]}, {KPT_SHAPE[1]}]",
        f"flip_idx: {FLIP_IDX}",
        "",
        "names:",
        "  0: towel",
    ]
    path = os.path.join(out, "towel_pose.yaml")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path
