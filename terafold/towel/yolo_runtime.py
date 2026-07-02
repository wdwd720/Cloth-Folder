"""YOLO **pose** runtime for the towel pipeline: train command, inference, eval.

Three entry points:

* :func:`print_towel_training_command` — pure string building. Emits the
  copy-paste ``yolo pose train ...`` command (plus a fallback model + a RunPod
  note). No dependencies at all.
* :func:`infer_towel_pose` — run a trained pose model on one image and return the
  towel's 4 corners (tl, tr, br, bl), confidences, and a normalized pose. Saves an
  overlay PNG and a JSON sidecar.
* :func:`evaluate_towel_pose` — run the model over every *usable* labelled sample,
  compute per-corner pixel error / IoU / confidence stats broken down by source and
  state, and write a Markdown report.

Ultralytics is an OPTIONAL dependency: it is imported lazily inside the functions
that need it. When it is missing, those functions return a ``status="unavailable"``
dict with an install hint — they never raise at import or call time. The training
command builder is dependency-free and always works.
"""

from __future__ import annotations

import math
import os
from typing import Any, Callable, Dict, List, Optional

from terafold.data.episode_schema import write_json
from terafold.towel.dataset import iter_usable
from terafold.towel.overlay import draw_towel_overlay
from terafold.towel.schema import (CORNER_ORDER, SOURCE_GROUP, bbox_from_corners,
                                   image_size, label_corners_xy, new_label)

__all__ = [
    "print_towel_training_command",
    "infer_towel_pose",
    "evaluate_towel_pose",
    "DEFAULT_MODEL",
    "FALLBACK_MODEL",
]

DEFAULT_MODEL = "yolo26n-pose.pt"
FALLBACK_MODEL = "yolo11n-pose.pt"
_INSTALL_HINT = "pip install -e '.[yolo]'"


def _noop(_m: str) -> None:
    pass


# --------------------------------------------------------------------------
# 1) Training command (pure string building — no deps).
# --------------------------------------------------------------------------


def print_towel_training_command(
    data: str,
    model: str = DEFAULT_MODEL,
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    project: str = "runs/towel_pose",
    name: str = "yolo_towel_pose_v0",
) -> Dict[str, Any]:
    """Build the copy-paste ``yolo pose train`` command for a towel pose dataset.

    ``data`` is the exported YOLO-pose dataset directory (the one that contains
    ``towel_pose.yaml``). Returns the primary command (``model``), a ``fallback_command``
    that swaps in the always-available ``yolo11n-pose.pt`` weights, the resolved
    ``yaml`` path, and a short ``runpod_note``. Nothing is imported or executed.
    """
    yaml = os.path.join(data, "towel_pose.yaml")

    def _cmd(m: str) -> str:
        return (
            f"yolo pose train model={m} data={yaml} epochs={epochs} "
            f"imgsz={imgsz} batch={batch} project={project} name={name}"
        )

    runpod_note = (
        "The same command runs unchanged on a RunPod (or any cloud) GPU: "
        "`pip install ultralytics`, upload/sync this dataset directory, then run the "
        "command above. Use the fallback_command (yolo11n-pose.pt) if the yolo26 "
        "weights are unavailable for your ultralytics version."
    )
    return {
        "command": _cmd(model),
        "fallback_command": _cmd(FALLBACK_MODEL),
        "yaml": yaml,
        "runpod_note": runpod_note,
        "model": model,
        "project": project,
        "name": name,
    }


# --------------------------------------------------------------------------
# Small numpy/torch-agnostic helpers (only used on the inference path).
# --------------------------------------------------------------------------


def _to_np(x):
    """Coerce a torch tensor / array-like to a numpy array (or None)."""
    import numpy as np

    if x is None:
        return None
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        try:
            return np.asarray(x.numpy())
        except Exception:
            pass
    return np.asarray(x)


def _bbox_iou(a: List[float], b: List[float]) -> float:
    """Intersection-over-union of two ``[x1, y1, x2, y2]`` boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


# --------------------------------------------------------------------------
# 2) Inference on a single image.
# --------------------------------------------------------------------------


def infer_towel_pose(
    image: str,
    weights: str,
    out: Optional[str] = None,
    overlay_out: Optional[str] = None,
    conf: float = 0.25,
    log: Callable[[str], None] = print,
) -> Dict[str, Any]:
    """Run a trained YOLO pose model on ``image`` and return the towel pose.

    Returns a dict whose ``status`` is one of:

    * ``"unavailable"`` — ultralytics is not installed (with an ``install`` hint),
    * ``"no_detection"`` — the model found no towel,
    * ``"ok"`` — with ``bbox``, ``corners`` (tl/tr/br/bl ``[x, y]``),
      ``corner_confidence``, ``avg_keypoint_confidence``, ``normalized_corners``
      and ``predicted_state`` (always ``None`` here — classification is separate).

    When ``overlay_out`` is given, an overlay PNG is rendered via
    :func:`draw_towel_overlay` (gracefully skipped if no image backend can decode the
    image). When ``out`` is given, the result dict is written as JSON.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        return {
            "status": "unavailable",
            "install": _INSTALL_HINT,
            "message": (
                "ultralytics is not installed; cannot run YOLO pose inference. "
                f"Install the optional extra with: {_INSTALL_HINT}"
            ),
            "image": image,
            "weights": weights,
        }

    import numpy as np

    model = YOLO(weights)
    results = model(image, conf=conf, verbose=False)
    res = results[0] if isinstance(results, (list, tuple)) else results

    boxes = getattr(res, "boxes", None)
    kpts = getattr(res, "keypoints", None)
    n_boxes = 0 if boxes is None else len(boxes)
    if n_boxes == 0 or kpts is None:
        log(f"[infer] no towel detected in {image}")
        return {"status": "no_detection", "image": image, "weights": weights}

    xyxy = _to_np(getattr(boxes, "xyxy", None))
    bconf = _to_np(getattr(boxes, "conf", None))
    kxy = _to_np(getattr(kpts, "xy", None))
    kconf = _to_np(getattr(kpts, "conf", None))

    if kxy is None or kxy.shape[0] == 0:
        log(f"[infer] detection had no keypoints in {image}")
        return {"status": "no_detection", "image": image, "weights": weights}

    # Top detection = highest box confidence (fall back to the first row).
    idx = int(np.argmax(bconf)) if (bconf is not None and bconf.size) else 0

    bbox = [float(v) for v in xyxy[idx]] if xyxy is not None else None

    # Image size for normalization (orig_shape is (h, w)).
    orig = getattr(res, "orig_shape", None)
    if orig is not None and len(orig) >= 2:
        ih, iw = int(orig[0]), int(orig[1])
    else:
        size = image_size(image)
        iw, ih = (size if size else (0, 0))

    corner_pts = kxy[idx]
    corners: Dict[str, List[float]] = {}
    corner_confidence: Dict[str, Optional[float]] = {}
    normalized_corners: Dict[str, Optional[List[float]]] = {}
    for i, k in enumerate(CORNER_ORDER):
        if i >= len(corner_pts):
            corners[k] = None
            corner_confidence[k] = None
            normalized_corners[k] = None
            continue
        px, py = float(corner_pts[i][0]), float(corner_pts[i][1])
        corners[k] = [px, py]
        if kconf is not None and idx < kconf.shape[0] and i < kconf.shape[1]:
            corner_confidence[k] = float(kconf[idx][i])
        else:
            corner_confidence[k] = None
        normalized_corners[k] = (
            [px / iw, py / ih] if (iw and ih) else None
        )

    conf_vals = [c for c in corner_confidence.values() if c is not None]
    avg_kp_conf = float(sum(conf_vals) / len(conf_vals)) if conf_vals else None

    result: Dict[str, Any] = {
        "status": "ok",
        "bbox": bbox,
        "corners": corners,
        "corner_confidence": corner_confidence,
        "avg_keypoint_confidence": avg_kp_conf,
        "normalized_corners": normalized_corners,
        "predicted_state": None,
        "image": image,
        "weights": weights,
        "width": iw or None,
        "height": ih or None,
    }

    # Overlay: build a label dict carrying the predicted corners/bbox.
    if overlay_out:
        ov_label = new_label(image, source="user_photo", state="flat_unfolded")
        ov_label["corners"] = dict(corners)
        ov_label["bbox"] = bbox
        ov = draw_towel_overlay(image, ov_label, overlay_out)
        result["overlay"] = ov.get("out")
        result["overlay_rendered"] = bool(ov.get("rendered"))

    if out:
        result["out"] = out
        write_json(out, result)

    return result


# --------------------------------------------------------------------------
# 3) Evaluation over a labelled dataset.
# --------------------------------------------------------------------------


def _mean(xs: List[float]) -> Optional[float]:
    return float(sum(xs) / len(xs)) if xs else None


def _median(xs: List[float]) -> Optional[float]:
    if not xs:
        return None
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    if n % 2:
        return float(s[mid])
    return float((s[mid - 1] + s[mid]) / 2.0)


def _source_group(sample: Dict[str, Any], label: Dict[str, Any]) -> str:
    grp = sample.get("source_group")
    if grp:
        return grp
    src = sample.get("source") or label.get("source") or "external"
    return SOURCE_GROUP.get(src, "external")


def evaluate_towel_pose(
    data: str,
    weights: str,
    out: str,
    log: Callable[[str], None] = print,
    conf: float = 0.25,
    fail_norm_threshold: float = 0.10,
) -> Dict[str, Any]:
    """Evaluate a trained pose model on every usable sample of dataset ``data``.

    For each usable, labelled sample the model is run and per-corner pixel error is
    measured against the ground-truth corners. Results are aggregated into
    ``metrics`` (mean/median corner error, image-diagonal-normalized error, bbox IoU,
    confidence distribution, failure cases, and error broken down by source group and
    by towel state) and written as a Markdown report.

    ``out`` may be a directory (report goes to ``out/eval.md``) or a path ending in
    ``.md``. Returns ``{"status": "ok", "metrics": ..., "report": <path>}`` — or
    ``{"status": "unavailable", "install": ...}`` if ultralytics is missing.
    """
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        return {
            "status": "unavailable",
            "install": _INSTALL_HINT,
            "message": (
                "ultralytics is not installed; cannot evaluate a YOLO pose model. "
                f"Install the optional extra with: {_INSTALL_HINT}"
            ),
        }

    all_corner_err: List[float] = []
    all_norm_err: List[float] = []
    confidences: List[float] = []
    ious: List[float] = []
    failures: List[Dict[str, Any]] = []
    by_source: Dict[str, List[float]] = {}
    by_state: Dict[str, List[float]] = {}

    num_samples = 0
    num_detected = 0
    num_no_detection = 0

    for sample, label in iter_usable(data):
        num_samples += 1
        gt = label_corners_xy(label)
        if gt is None:
            continue
        img_path = os.path.join(data, sample["image"])
        state = label.get("state") or "unknown"
        group = _source_group(sample, label)

        # Image dimensions for normalization.
        w = sample.get("width")
        h = sample.get("height")
        if not (w and h):
            size = image_size(img_path)
            if size:
                w, h = size
        diag = math.hypot(float(w), float(h)) if (w and h) else None

        res = infer_towel_pose(img_path, weights, conf=conf, log=_noop)
        status = res.get("status")
        if status != "ok":
            num_no_detection += 1
            failures.append({
                "id": sample.get("id"), "image": sample.get("image"),
                "source": group, "state": state, "reason": status,
                "mean_px_error": None,
            })
            continue

        num_detected += 1
        pred = res.get("corners") or {}
        sample_errs: List[float] = []
        for i, k in enumerate(CORNER_ORDER):
            p = pred.get(k)
            if p is None:
                continue
            gx, gy = gt[i]
            e = math.hypot(float(p[0]) - gx, float(p[1]) - gy)
            sample_errs.append(e)
            all_corner_err.append(e)
            if diag:
                all_norm_err.append(e / diag)

        if not sample_errs:
            continue
        mean_e = float(sum(sample_errs) / len(sample_errs))
        norm_e = (mean_e / diag) if diag else None
        by_source.setdefault(group, []).append(mean_e)
        by_state.setdefault(state, []).append(mean_e)

        kp_conf = res.get("avg_keypoint_confidence")
        if kp_conf is not None:
            confidences.append(float(kp_conf))

        pred_bbox = res.get("bbox")
        gt_bbox = label.get("bbox") or bbox_from_corners(gt)
        if pred_bbox is not None and gt_bbox is not None:
            ious.append(_bbox_iou(pred_bbox, gt_bbox))

        if norm_e is not None and norm_e > fail_norm_threshold:
            failures.append({
                "id": sample.get("id"), "image": sample.get("image"),
                "source": group, "state": state, "reason": "high_error",
                "mean_px_error": round(mean_e, 2),
                "normalized_error": round(norm_e, 4),
            })

    def _grouped(d: Dict[str, List[float]]) -> Dict[str, Dict[str, Any]]:
        return {
            key: {"n": len(vals), "mean_px_error": _mean(vals),
                  "median_px_error": _median(vals)}
            for key, vals in sorted(d.items())
        }

    metrics: Dict[str, Any] = {
        "num_samples": num_samples,
        "num_detected": num_detected,
        "num_no_detection": num_no_detection,
        "mean_corner_px_error": _mean(all_corner_err),
        "median_corner_px_error": _median(all_corner_err),
        "normalized_corner_error": _mean(all_norm_err),
        "bbox_iou": _mean(ious),
        "confidence": {
            "min": (min(confidences) if confidences else None),
            "mean": _mean(confidences),
            "max": (max(confidences) if confidences else None),
        },
        "num_failure_cases": len(failures),
        "failure_cases": failures,
        "error_by_source": _grouped(by_source),
        "error_by_state": _grouped(by_state),
        "fail_norm_threshold": fail_norm_threshold,
    }

    report = _write_report(out, data, weights, metrics)
    log(f"Evaluated {num_detected}/{num_samples} usable samples -> {report}")
    return {"status": "ok", "metrics": metrics, "report": report}


def _fmt(v: Optional[float], nd: int = 2) -> str:
    return "n/a" if v is None else f"{v:.{nd}f}"


def _write_report(out: str, data: str, weights: str, metrics: Dict[str, Any]) -> str:
    """Write the Markdown eval report; ``out`` may be a dir or a ``.md`` path."""
    if out.lower().endswith(".md"):
        report = out
        os.makedirs(os.path.dirname(os.path.abspath(report)) or ".", exist_ok=True)
    else:
        os.makedirs(out, exist_ok=True)
        report = os.path.join(out, "eval.md")

    conf = metrics.get("confidence") or {}
    lines: List[str] = [
        "# Towel pose evaluation",
        "",
        f"- dataset: `{data}`",
        f"- weights: `{weights}`",
        f"- usable samples: {metrics['num_samples']}  "
        f"(detected {metrics['num_detected']}, no-detection {metrics['num_no_detection']})",
        "",
        "## Corner accuracy",
        "",
        f"- mean corner error: **{_fmt(metrics['mean_corner_px_error'])} px**",
        f"- median corner error: {_fmt(metrics['median_corner_px_error'])} px",
        f"- normalized corner error (by image diagonal): "
        f"{_fmt(metrics['normalized_corner_error'], 4)}",
        f"- mean bbox IoU: {_fmt(metrics['bbox_iou'], 4)}",
        "",
        "## Keypoint confidence",
        "",
        f"- min / mean / max: {_fmt(conf.get('min'), 3)} / "
        f"{_fmt(conf.get('mean'), 3)} / {_fmt(conf.get('max'), 3)}",
        "",
        "## Error by source",
        "",
        "| source | n | mean px error | median px error |",
        "| --- | ---: | ---: | ---: |",
    ]
    for src, st in metrics["error_by_source"].items():
        lines.append(
            f"| {src} | {st['n']} | {_fmt(st['mean_px_error'])} | {_fmt(st['median_px_error'])} |"
        )
    lines += [
        "",
        "## Error by state",
        "",
        "| state | n | mean px error | median px error |",
        "| --- | ---: | ---: | ---: |",
    ]
    for state, st in metrics["error_by_state"].items():
        lines.append(
            f"| {state} | {st['n']} | {_fmt(st['mean_px_error'])} | {_fmt(st['median_px_error'])} |"
        )

    failures = metrics.get("failure_cases") or []
    lines += [
        "",
        f"## Failure cases ({len(failures)}) — "
        f"normalized error > {metrics['fail_norm_threshold']}",
        "",
    ]
    if failures:
        lines += ["| id | source | state | reason | mean px error |",
                  "| --- | --- | --- | --- | ---: |"]
        for f in failures:
            lines.append(
                f"| {f.get('id')} | {f.get('source')} | {f.get('state')} | "
                f"{f.get('reason')} | {_fmt(f.get('mean_px_error'))} |"
            )
    else:
        lines.append("_None._")
    lines.append("")

    with open(report, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return report
