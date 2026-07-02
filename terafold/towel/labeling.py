"""Local 4-corner towel labeling helpers.

These functions turn raw images in a towel dataset into valid per-image labels
(corners ``tl,tr,br,bl`` + bbox + state) using the towel foundation:

* :func:`label_towel_image` writes one label. Corners can come from (a) an explicit
  list (the programmatic / test path), (b) clicking on a cv2 GUI window, or
  (c) typing ``x,y`` per corner in the terminal. The GUI/terminal paths are
  best-effort and degrade gracefully — a missing cv2 falls back to the terminal.
* :func:`label_towel_folder` walks a dataset manifest and labels every
  not-yet-usable sample, letting the operator skip an image or mark it as a reject
  state (``bad_view`` / ``multiple_towels`` / ``not_towel``).

Only ``numpy``/stdlib are needed: cv2 is lazily imported and never required, and
the overlay preview uses the foundation's PNG-capable backend.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional

from terafold.data.episode_schema import write_json
from terafold.towel.dataset import (load_label, load_manifest, recount_usable,
                                    save_label)
from terafold.towel.overlay import draw_towel_overlay
from terafold.towel.schema import (CORNER_ORDER, REJECT_STATES, bbox_from_corners,
                                   label_corners_xy, new_label, validate_label)

__all__ = ["label_towel_image", "label_towel_folder"]

_REJECT_SET = set(REJECT_STATES)


def label_towel_image(
    image: str,
    label: str,
    state: Optional[str] = None,
    corners: Optional[List[List[float]]] = None,
    fold_axis: Any = None,
    input_fn: Callable[[str], str] = input,
    log: Callable[[str], None] = print,
    use_gui: bool = True,
    overlay_out: Optional[str] = None,
) -> Dict[str, Any]:
    """Write a valid 4-corner towel label for ``image`` at JSON path ``label``.

    Args:
        image: path to the image being labeled.
        label: path to read/write the label JSON.
        state: towel state to set (else keep the existing/template state).
        corners: explicit ``[[x,y]*4]`` in ``tl,tr,br,bl`` order — the test path.
            If given it is used directly (no GUI/terminal prompts).
        fold_axis: optional fold-axis value stored on the label (else ``None``).
        input_fn: terminal input function (overridable for tests).
        log: logging sink.
        use_gui: try a cv2 click-window when ``corners`` is not provided.
        overlay_out: where to save the preview overlay (defaults next to the label).

    Returns:
        ``{"label_path", "overlay", "corners", "usable", "state"}``.
    """
    # Load an existing label if present, else build a fresh template.
    if os.path.exists(label):
        from terafold.data.episode_schema import read_json

        lab = read_json(label)
    else:
        image_rel = "images/" + os.path.basename(image)
        lab = new_label(image_rel)

    # Acquire the 4 corners in tl,tr,br,bl order.
    if corners is not None:
        pts = [[float(p[0]), float(p[1])] for p in corners]
    elif use_gui:
        pts = _collect_corners_gui(image, log)
        if pts is None:
            log("[labeling] GUI unavailable — falling back to terminal input.")
            pts = _collect_corners_terminal(input_fn, log)
    else:
        pts = _collect_corners_terminal(input_fn, log)

    if len(pts) != 4:
        raise ValueError(f"expected 4 corners (tl,tr,br,bl), got {len(pts)}")

    lab["corners"] = {k: pts[i] for i, k in enumerate(CORNER_ORDER)}
    lab["bbox"] = bbox_from_corners(pts)
    if fold_axis is not None:
        lab["fold_axis"] = fold_axis
    if state is not None:
        lab["state"] = state
    lab["usable_for_training"] = True

    problems = validate_label(lab)
    if problems:
        raise ValueError(f"label failed validation: {problems}")

    write_json(label, lab)

    # Best-effort overlay preview (never fatal if no image backend).
    if overlay_out is None:
        stem = os.path.splitext(os.path.basename(image))[0]
        overlay_out = os.path.join(os.path.dirname(label) or ".", f"{stem}_overlay.png")
    ov = draw_towel_overlay(image, lab, overlay_out)
    overlay = ov.get("out")
    if not ov.get("rendered"):
        log(f"[labeling] overlay skipped: {ov.get('reason', '')}")

    return {
        "label_path": label,
        "overlay": overlay,
        "corners": label_corners_xy(lab),
        "usable": True,
        "state": lab["state"],
    }


def label_towel_folder(
    dataset: str,
    input_fn: Callable[[str], str] = input,
    log: Callable[[str], None] = print,
) -> Dict[str, Any]:
    """Interactively label every not-yet-usable sample in a towel ``dataset``.

    For each pending sample ``input_fn`` returns one of:

    * ``skip`` (or empty) — leave the sample unlabeled;
    * ``bad_view`` / ``multiple_towels`` / ``not_towel`` — set that reject state and
      mark ``usable_for_training=False`` (excluded from training export);
    * ``x1,y1 x2,y2 x3,y3 x4,y4`` — four corner coords (``tl,tr,br,bl``) to label.

    Returns ``{"total", "labeled", "skipped", "marked_bad"}``.
    """
    man = load_manifest(dataset)
    pending = [s for s in man.get("samples", []) if not load_label(dataset, s).get("usable_for_training")]
    total = len(pending)
    labeled = skipped = marked_bad = 0

    for i, s in enumerate(pending, 1):
        img_path = os.path.join(dataset, s["image"])
        lab_path = os.path.join(dataset, s["label"])
        prompt = (
            f"[{i}/{total}] {s['image']} — corners 'x1,y1 x2,y2 x3,y3 x4,y4' "
            f"(tl,tr,br,bl) or skip/bad_view/multiple_towels/not_towel: "
        )
        resp = (input_fn(prompt) or "").strip()

        if resp == "" or resp == "skip":
            log(f"[{i}/{total}] skipped {s['image']}")
            skipped += 1
            continue

        if resp in _REJECT_SET:
            lab = load_label(dataset, s)
            lab["state"] = resp
            lab["usable_for_training"] = False
            save_label(dataset, s, lab)
            log(f"[{i}/{total}] marked {resp}: {s['image']}")
            marked_bad += 1
            continue

        try:
            pts = _parse_corner_line(resp)
        except ValueError as exc:
            log(f"[{i}/{total}] could not parse corners {resp!r} ({exc}) — skipping")
            skipped += 1
            continue

        label_towel_image(
            img_path, lab_path, corners=pts, input_fn=input_fn, log=log, use_gui=False,
        )
        log(f"[{i}/{total}] labeled {s['image']}")
        labeled += 1

    recount_usable(dataset)
    return {"total": total, "labeled": labeled, "skipped": skipped, "marked_bad": marked_bad}


# ----------------------------------------------------------------------
# Corner acquisition helpers.
# ----------------------------------------------------------------------


def _parse_xy(text: str) -> List[float]:
    """Parse a single ``'x,y'`` string into ``[x, y]`` floats."""
    parts = [p for p in text.replace(";", ",").strip().split(",") if p.strip() != ""]
    if len(parts) != 2:
        raise ValueError(f"expected 'x,y', got {text!r}")
    return [float(parts[0]), float(parts[1])]


def _parse_corner_line(text: str) -> List[List[float]]:
    """Parse ``'x1,y1 x2,y2 x3,y3 x4,y4'`` into ``[[x,y]*4]`` (tl,tr,br,bl)."""
    tokens = text.split()
    if len(tokens) != 4:
        raise ValueError(f"expected 4 'x,y' tokens, got {len(tokens)}")
    return [_parse_xy(tok) for tok in tokens]


def _collect_corners_terminal(
    input_fn: Callable[[str], str], log: Callable[[str], None],
) -> List[List[float]]:
    """Prompt for each corner (``tl``, ``tr``, ``br``, ``bl``) as ``x,y`` in the terminal."""
    log("Enter towel corners as 'x,y' (pixels), in order tl, tr, br, bl.")
    pts: List[List[float]] = []
    for k in CORNER_ORDER:
        raw = input_fn(f"{k} x,y: ")
        pts.append(_parse_xy(raw))
    return pts


def _collect_corners_gui(
    image_path: str, log: Callable[[str], None],
) -> Optional[List[List[float]]]:
    """Open a cv2 window and collect 4 clicked corners. Returns ``None`` if unavailable."""
    try:
        from terafold.vision.imageio import have_cv2

        if not have_cv2():
            return None
        import cv2  # type: ignore
        import numpy as np

        from terafold.vision.imageio import imread

        rgb = imread(image_path)
        bgr = np.ascontiguousarray(np.asarray(rgb)[:, :, ::-1]).copy()
        clicks: List[List[float]] = []

        def _on_mouse(event, x, y, flags, param):  # noqa: ANN001
            if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 4:
                clicks.append([float(x), float(y)])
                cv2.circle(bgr, (x, y), 5, (0, 0, 255), -1)
                cv2.imshow(win, bgr)

        win = "TeraFold: click corners tl, tr, br, bl"
        cv2.namedWindow(win)
        cv2.setMouseCallback(win, _on_mouse)
        cv2.imshow(win, bgr)
        log("Click 4 corners (tl, tr, br, bl). Press Esc to cancel.")
        while len(clicks) < 4:
            key = cv2.waitKey(20) & 0xFF
            if key == 27:  # Esc
                break
        cv2.destroyWindow(win)
        return clicks if len(clicks) == 4 else None
    except Exception:
        return None
