"""Human review + high-confidence auto-approval of Claude pseudo-labels.

Pseudo-labels are never trusted by default. Before any pseudo-labeled sample can
reach YOLO export it must be approved here:

* :func:`review_pseudolabels` — walk the dataset one sample at a time, show the
  overlay + state + confidence + risk reasons, and let the operator
  approve / reject / edit-state / edit-corners. Writes ``review_report.md``.
* :func:`approve_high_confidence_pseudolabels` — a NON-interactive gate that
  auto-approves ONLY labels that are a flat trainable towel, have 4 in-bounds
  corners, clear the confidence threshold, AND pass a geometry-error check.
  Everything else stays ``review_needed``. It loudly warns that pseudo-labels can
  be wrong.

Both update the per-image label JSON (``approval_status`` + ``usable_for_training``)
and the dataset manifest. Approving a label sets ``usable_for_training=true`` only
when it is actually a trainable 4-corner towel, so a mistaken "approve" on a
bad_view/not_towel can never leak into training.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional

from terafold.data.episode_schema import read_json, write_json
from terafold.towel.dataset import dataset_paths, load_manifest, save_manifest
from terafold.towel.labeling import _parse_corner_line
from terafold.towel.schema import (CORNER_ORDER, TOWEL_STATES, TRAINABLE_STATES,
                                   bbox_from_corners, is_pseudolabel, label_corners_xy,
                                   polygon_area_xy, quad_geometry_error, quad_is_simple)
from terafold.towel.claude_pseudolabel import AUTO_OK_STATES

__all__ = ["review_pseudolabels", "approve_high_confidence_pseudolabels",
           "auto_triage_pseudolabels", "TRIAGE_POLICIES"]

_PENDING = ("pending", "review_needed")
_BLOCKING_RISKS = ("geometry_impossible", "out_of_bounds", "corners_missing", "schema_invalid")

# --- triage policy knobs ---
TRIAGE_POLICIES = ("pose_positive_strict", "outer_visible_corners_any_state")

# pose_positive_strict: only flat/mildly-wrinkled, fully-unfolded towels.
_STRICT_REJECT_STATES = ("folded_success", "partially_folded", "multiple_towels",
                         "bad_view", "not_towel")
_STRICT_APPROVE_STATES = ("flat_unfolded", "wrinkled_unfolded")
_STRICT_APPROVE_SOURCES = ("openai_generated", "user_photo", "external",
                           "kaggle", "openimages", "roboflow")
# Severe risk substrings that disqualify a pose positive (matched case-insensitively).
_SEVERE_RISK_SUBSTRINGS = ("hang", "fold", "stack", "roll", "multiple", "occlud",
                           "person", "people", "hand", "off-frame", "off frame",
                           "cut off", "border")
_STRICT_MIN_AREA_FRAC = 0.01     # polygon area must be >= 1% of the image
_STRICT_MAX_BOUNDARY_FRAC = 0.985  # bbox spanning ~all of W and H ⇒ towel cut off / fills frame

# outer_visible_corners_any_state: accept ANY trainable towel state (including
# partially_folded / folded_success) as long as the 4 VISIBLE outer corners of the
# current shape are clean. The model learns the visible quad, not hidden original
# corners. Folding/rolling/stack words are NOT disqualifying here (folded is fine);
# only things that actually break the 4-visible-corner target are.
_ANY_STATE_ACCEPT = ("flat_unfolded", "wrinkled_unfolded", "partially_folded",
                     "folded_success")
_ANY_STATE_REJECT = ("multiple_towels", "bad_view", "not_towel")
_VISIBLE_SEVERE_RISKS = ("hand", "body", "person", "people", "multiple", "occlud",
                         "blur", "off-frame", "off frame", "cut off", "border")


def _noop(_m: str) -> None:
    pass


def _trainable_with_corners(label: Dict[str, Any]) -> bool:
    return label.get("state") in TRAINABLE_STATES and label_corners_xy(label) is not None


def _iter_pseudolabels(root: str):
    man = load_manifest(root)
    for s in man.get("samples", []):
        try:
            label = read_json(os.path.join(root, s["label"]))
        except FileNotFoundError:
            continue
        if is_pseudolabel(label):
            yield man, s, label


# --------------------------------------------------------------------------
# Interactive review
# --------------------------------------------------------------------------


def _matches_filters(s: Dict[str, Any], label: Dict[str, Any],
                     source_filter: Optional[str], category_filter: Optional[str]) -> bool:
    if source_filter is not None:
        src = s.get("source") or s.get("image_source") or label.get("source")
        if src != source_filter:
            return False
    if category_filter is not None:
        if (s.get("category_target") or label.get("category_target")) != category_filter:
            return False
    return True


def review_pseudolabels(
    dataset: str,
    input_fn: Callable[[str], str] = input,
    log: Callable[[str], None] = print,
    only_pending: bool = True,
    source_filter: Optional[str] = None,
    category_filter: Optional[str] = None,
) -> Dict[str, Any]:
    """Interactively approve / reject / edit Claude pseudo-labels in ``dataset``.

    Per sample, ``input_fn`` returns one of:
      * ``approve`` / ``a``  — mark approved (usable only if a trainable 4-corner towel)
      * ``reject``  / ``r``  — mark rejected (never exported)
      * ``skip``    / ``""`` — leave unchanged, move on
      * ``state:<towel_state>``                 — edit the state, then re-prompt
      * ``corners:x1,y1 x2,y2 x3,y3 x4,y4``     — edit the 4 corners, then re-prompt
      * ``quit`` / ``q``     — stop reviewing (remaining samples untouched)

    ``source_filter`` (e.g. ``openai_generated``) and ``category_filter`` (e.g.
    ``positive_flat``) restrict review to a subset.
    """
    if not os.path.exists(dataset_paths(dataset)["manifest"]):
        return {"status": "error", "message": f"no manifest in dataset: {dataset}"}
    man = load_manifest(dataset)
    samples = man.get("samples", [])

    approved = rejected = edited = skipped = 0
    quit_early = False
    reason_tally: Dict[str, int] = {}
    still_pending = 0

    for i, s in enumerate(samples, 1):
        lab_path = os.path.join(dataset, s["label"])
        try:
            label = read_json(lab_path)
        except FileNotFoundError:
            continue
        if not is_pseudolabel(label):
            continue
        if not _matches_filters(s, label, source_filter, category_filter):
            continue
        status = label.get("approval_status", "pending")
        for r in label.get("risk_reasons", []):
            reason_tally[r] = reason_tally.get(r, 0) + 1
        if only_pending and status not in _PENDING:
            continue
        if quit_early:
            still_pending += 1
            continue

        decided = False
        while not decided:
            log(f"[{i}/{len(samples)}] {s['image']}  state={label.get('state')} "
                f"conf={label.get('confidence')} status={label.get('approval_status')}")
            if label.get("risk_reasons"):
                log(f"    risks: {', '.join(label['risk_reasons'])}")
            if s.get("overlay"):
                log(f"    overlay: {os.path.join(dataset, s['overlay'])}")
            log(f"    corners: {label.get('corners')}")
            resp = (input_fn("    approve / reject / skip / state:<s> / "
                             "corners:x1,y1 x2,y2 x3,y3 x4,y4 / quit: ") or "").strip()
            low = resp.lower()

            if low in ("", "skip", "s"):
                skipped += 1
                decided = True
            elif low in ("quit", "q"):
                quit_early = True
                skipped += 1
                decided = True
            elif low in ("approve", "a"):
                _set_approval(label, "approved")
                write_json(lab_path, label)
                s["approval_status"] = "approved"
                approved += 1
                decided = True
                log(f"    -> approved (usable_for_training={label['usable_for_training']})")
            elif low in ("reject", "r"):
                _set_approval(label, "rejected")
                write_json(lab_path, label)
                s["approval_status"] = "rejected"
                rejected += 1
                decided = True
                log("    -> rejected")
            elif low.startswith("state:"):
                new_state = resp.split(":", 1)[1].strip()
                if new_state in TOWEL_STATES:
                    label["state"] = new_state
                    edited += 1
                    log(f"    state -> {new_state}")
                else:
                    log(f"    invalid state {new_state!r}; valid: {TOWEL_STATES}")
            elif low.startswith("corners:"):
                try:
                    pts = _parse_corner_line(resp.split(":", 1)[1].strip())
                    label["corners"] = {k: pts[j] for j, k in enumerate(CORNER_ORDER)}
                    label["bbox"] = bbox_from_corners(pts)
                    edited += 1
                    log("    corners updated")
                except ValueError as exc:
                    log(f"    could not parse corners ({exc})")
            else:
                log("    unrecognized; type approve/reject/skip/state:/corners:/quit")

    if only_pending:
        still_pending += sum(
            1 for _m, _s, lab in _iter_pseudolabels(dataset)
            if lab.get("approval_status") in _PENDING)

    _sync_manifest_status(dataset, man)
    save_manifest(dataset, man)
    report = _write_review_report(dataset, approved, rejected, still_pending, reason_tally)
    log(f"Review done: {approved} approved, {rejected} rejected, {edited} edits, "
        f"{skipped} skipped, {still_pending} still need review.")
    log(f"  report: {report}")
    return {"status": "ok", "approved": approved, "rejected": rejected, "edited": edited,
            "skipped": skipped, "needs_review": still_pending, "report": report}


def _set_approval(label: Dict[str, Any], status: str) -> None:
    label["approval_status"] = status
    label["review_needed"] = False
    if status in ("approved", "auto_approved"):
        label["usable_for_training"] = _trainable_with_corners(label)
    else:
        label["usable_for_training"] = False


# --------------------------------------------------------------------------
# High-confidence auto-approval (non-interactive)
# --------------------------------------------------------------------------


def approve_high_confidence_pseudolabels(
    dataset: str,
    min_confidence: float = 0.90,
    max_geometry_error: float = 0.10,
    source_filter: Optional[str] = None,
    category_filter: Optional[str] = None,
    log: Callable[[str], None] = print,
) -> Dict[str, Any]:
    """Auto-approve only high-confidence, geometrically-sound pseudo-labels.

    A label is auto-approved iff ALL hold:
      * state is a flat trainable towel (flat/wrinkled/partially_folded),
      * it has 4 corners,
      * ``confidence >= min_confidence``,
      * recomputed quad ``geometry_error <= max_geometry_error``,
      * no blocking risk (out_of_bounds / geometry_impossible / corners_missing).
    Everything else is left ``review_needed`` for a human. Pseudo-labels can be
    wrong — this is a convenience gate, not a substitute for review.
    ``source_filter`` / ``category_filter`` restrict the gate to a subset.
    """
    if not os.path.exists(dataset_paths(dataset)["manifest"]):
        return {"status": "error", "message": f"no manifest in dataset: {dataset}"}
    log("⚠️  WARNING: auto-approving pseudo-labels. Claude labels CAN BE WRONG; "
        "spot-check the overlays and prefer review-pseudolabels for anything important.")
    man = load_manifest(dataset)
    auto_approved = left_for_review = considered = 0

    for s in man.get("samples", []):
        lab_path = os.path.join(dataset, s["label"])
        try:
            label = read_json(lab_path)
        except FileNotFoundError:
            continue
        if not is_pseudolabel(label):
            continue
        if not _matches_filters(s, label, source_filter, category_filter):
            continue
        if label.get("approval_status") not in _PENDING:
            continue
        considered += 1
        ok, why = _auto_eligible(label, float(min_confidence), float(max_geometry_error))
        if ok:
            _set_approval(label, "auto_approved")
            label["auto_approve_note"] = f"conf>={min_confidence}, geom_err<={max_geometry_error}"
            write_json(lab_path, label)
            s["approval_status"] = "auto_approved"
            auto_approved += 1
        else:
            label["approval_status"] = "review_needed"
            label["review_needed"] = True
            label["usable_for_training"] = False
            label.setdefault("risk_reasons", [])
            write_json(lab_path, label)
            s["approval_status"] = "review_needed"
            left_for_review += 1

    _sync_manifest_status(dataset, man)
    save_manifest(dataset, man)
    log(f"Auto-approved {auto_approved} / {considered} pseudo-labels "
        f"(conf>={min_confidence}, geom_err<={max_geometry_error}); "
        f"{left_for_review} left for human review.")
    return {"status": "ok", "auto_approved": auto_approved, "considered": considered,
            "needs_review": left_for_review,
            "min_confidence": float(min_confidence),
            "max_geometry_error": float(max_geometry_error)}


def _auto_eligible(label: Dict[str, Any], min_conf: float, max_geom: float):
    if label.get("state") not in AUTO_OK_STATES:
        return False, "state_not_flat_trainable"
    corners_xy = label_corners_xy(label)
    if corners_xy is None:
        return False, "corners_missing"
    if any(r in _BLOCKING_RISKS for r in label.get("risk_reasons", [])):
        return False, "blocking_risk"
    conf = label.get("confidence")
    if conf is None or float(conf) < min_conf:
        return False, "low_confidence"
    if quad_geometry_error(corners_xy) > max_geom:
        return False, "geometry_error"
    return True, "ok"


# --------------------------------------------------------------------------
# Auto-triage (Deliverable 3) — policy-driven approve / reject / keep-for-review
# --------------------------------------------------------------------------


def _severe_risks_for(label: Dict[str, Any], substrings) -> List[str]:
    hits = []
    for r in label.get("risk_reasons", []):
        low = str(r).lower()
        if any(sub in low for sub in substrings):
            hits.append(str(r))
    return hits


def _severe_risks(label: Dict[str, Any]) -> List[str]:
    return _severe_risks_for(label, _SEVERE_RISK_SUBSTRINGS)


def _all_corners_visible(label: Dict[str, Any]) -> bool:
    vis = label.get("visible") or {}
    return all(vis.get(k, True) for k in CORNER_ORDER)


def _corners_in_bounds(corners_xy, w: int, h: int, tol: float = 1.5) -> bool:
    return all(-tol <= x <= (w - 1) + tol and -tol <= y <= (h - 1) + tol
               for x, y in corners_xy)


def _pose_positive_strict(label, sample, w, h, min_approve: float, min_keep: float):
    """Return (decision, reasons) for the pose_positive_strict policy.

    decision ∈ {"auto_reject", "auto_approve", "review"}.
    """
    reasons: List[str] = []
    state = label.get("state")
    track = sample.get("training_track") or label.get("training_track")
    corners_xy = label_corners_xy(label)
    conf = label.get("confidence")
    conf = float(conf) if conf is not None else 0.0

    # ---- hard rejections ----
    if track == "critic_negative":
        reasons.append("critic_negative_track")
    if state in _STRICT_REJECT_STATES:
        reasons.append(f"reject_state:{state}")
    if conf < min_keep:
        reasons.append(f"low_confidence<{min_keep:g}")
    if corners_xy is None:
        reasons.append("corners_missing")
    sev = _severe_risks(label)
    if sev:
        reasons.append("severe_risk:" + ",".join(sorted({s.split(":")[0] for s in sev}))[:60])
    if corners_xy is not None and w and h:
        if not _corners_in_bounds(corners_xy, int(w), int(h)):
            reasons.append("out_of_bounds")
        if not quad_is_simple(corners_xy):
            reasons.append("invalid_polygon")
        else:
            area = polygon_area_xy(corners_xy)
            if area < _STRICT_MIN_AREA_FRAC * (int(w) * int(h)):
                reasons.append("polygon_too_small")
            x1, y1, x2, y2 = bbox_from_corners(corners_xy)
            if (x2 - x1) >= _STRICT_MAX_BOUNDARY_FRAC * w and (y2 - y1) >= _STRICT_MAX_BOUNDARY_FRAC * h:
                reasons.append("touches_boundary")
    if reasons:
        return "auto_reject", reasons

    # ---- auto-approve (all must hold) ----
    ok = (state in _STRICT_APPROVE_STATES and conf >= min_approve
          and corners_xy is not None
          and (sample.get("source") or label.get("source")) in _STRICT_APPROVE_SOURCES
          and not sev)
    if ok:
        return "auto_approve", ["approved"]
    # In the uncertain band (e.g. 0.75 ≤ conf < 0.88, or unknown source) → human review.
    if conf < min_approve:
        return "review", [f"uncertain_confidence<{min_approve:g}"]
    return "review", ["needs_human"]


def _outer_visible_corners_any_state(label, sample, w, h, min_approve: float, min_keep: float):
    """Triage for the ``outer_visible_corners_any_state`` policy.

    Accepts flat / wrinkled / partially_folded / folded_success as long as the 4
    VISIBLE outer corners of the current shape are clean (exactly 4 in-bounds,
    non-occluded, simple non-ambiguous quad, one towel, decent confidence). Rejects
    multiple towels / bad_view / not_towel, hidden or cut-off corners, hands/body
    over corners, blur, and ambiguous corner order. Folding/rolling words are NOT
    disqualifying — a folded towel still has 4 visible outer corners to learn.
    decision ∈ {"auto_reject", "auto_approve", "review"}.
    """
    reasons: List[str] = []
    state = label.get("state")
    track = sample.get("training_track") or label.get("training_track")
    corners_xy = label_corners_xy(label)
    conf = label.get("confidence")
    conf = float(conf) if conf is not None else 0.0

    # ---- hard rejections ----
    if track == "critic_negative":
        reasons.append("critic_negative_track")
    if state in _ANY_STATE_REJECT or state not in _ANY_STATE_ACCEPT:
        reasons.append(f"reject_state:{state}")     # multiple/bad_view/not_towel/unknown
    if conf < min_keep:
        reasons.append(f"low_confidence<{min_keep:g}")
    if corners_xy is None:
        reasons.append("corners_missing")           # need exactly 4 visible keypoints
    elif not _all_corners_visible(label):
        reasons.append("corner_hidden")             # a target corner is flagged hidden
    sev = _severe_risks_for(label, _VISIBLE_SEVERE_RISKS)
    if sev:
        reasons.append("severe_risk:" + ",".join(sorted({s.split(":")[0] for s in sev}))[:60])
    if corners_xy is not None and w and h:
        if not _corners_in_bounds(corners_xy, int(w), int(h)):
            reasons.append("out_of_bounds")         # corner cut off by frame edge
        if not quad_is_simple(corners_xy):
            reasons.append("ambiguous_corner_order")  # self-crossing ⇒ order ambiguous
        else:
            area = polygon_area_xy(corners_xy)
            if area < _STRICT_MIN_AREA_FRAC * (int(w) * int(h)):
                reasons.append("polygon_too_small")
            x1, y1, x2, y2 = bbox_from_corners(corners_xy)
            if (x2 - x1) >= _STRICT_MAX_BOUNDARY_FRAC * w and (y2 - y1) >= _STRICT_MAX_BOUNDARY_FRAC * h:
                reasons.append("touches_boundary")
    if reasons:
        return "auto_reject", reasons

    # ---- auto-approve (all must hold) ----
    ok = (state in _ANY_STATE_ACCEPT and conf >= min_approve
          and corners_xy is not None and _all_corners_visible(label)
          and (sample.get("source") or label.get("source")) in _STRICT_APPROVE_SOURCES
          and not sev)
    if ok:
        return "auto_approve", ["approved"]
    # Uncertain (e.g. 0.75 ≤ conf < 0.88, or unknown source) → human review, never approve.
    if conf < min_approve:
        return "review", [f"uncertain_confidence<{min_approve:g}"]
    return "review", ["needs_human"]


# Policy name -> triage function.
_TRIAGE_FNS = {
    "pose_positive_strict": _pose_positive_strict,
    "outer_visible_corners_any_state": _outer_visible_corners_any_state,
}


def auto_triage_pseudolabels(
    dataset: str,
    policy: str = "pose_positive_strict",
    apply: bool = False,
    min_confidence_approve: float = 0.88,
    min_confidence_reject: float = 0.75,
    source_filter: Optional[str] = None,
    category_filter: Optional[str] = None,
    log: Callable[[str], None] = print,
) -> Dict[str, Any]:
    """Policy-driven triage of pseudo-labels into approve / reject / review.

    ``--dry-run`` (apply=False) computes + reports the triage and writes
    ``auto_triage_report.md`` but does NOT modify any label JSON. ``--apply`` also
    writes the updated labels + manifest. Only ``pending`` / ``review_needed``
    pseudo-labels are considered; it never blindly approves everything.
    """
    if policy not in TRIAGE_POLICIES:
        return {"status": "error", "message": f"unknown policy {policy!r}",
                "policies": list(TRIAGE_POLICIES)}
    if not os.path.exists(dataset_paths(dataset)["manifest"]):
        return {"status": "error", "message": f"no manifest in dataset: {dataset}"}
    man = load_manifest(dataset)

    counts = {"auto_approved": 0, "auto_rejected": 0, "review_needed": 0, "considered": 0}
    reason_hist: Dict[str, int] = {}
    by_source: Dict[str, Dict[str, int]] = {}
    by_category: Dict[str, Dict[str, int]] = {}
    by_track: Dict[str, Dict[str, int]] = {}

    _DECISION_KEY = {"auto_approve": "auto_approved", "auto_reject": "auto_rejected",
                     "review": "review_needed"}

    def _tally(bucket, key, decision):
        d = bucket.setdefault(key or "?", {"auto_approved": 0, "auto_rejected": 0,
                                           "review_needed": 0})
        d[_DECISION_KEY[decision]] += 1

    for s in man.get("samples", []):
        lab_path = os.path.join(dataset, s["label"])
        try:
            label = read_json(lab_path)
        except FileNotFoundError:
            continue
        if not is_pseudolabel(label):
            continue
        if not _matches_filters(s, label, source_filter, category_filter):
            continue
        if label.get("approval_status") not in _PENDING:
            continue
        counts["considered"] += 1
        w = s.get("width") or 0
        h = s.get("height") or 0
        decision, reasons = _TRIAGE_FNS[policy](
            label, s, w, h, float(min_confidence_approve), float(min_confidence_reject))

        for r in reasons:
            reason_hist[r] = reason_hist.get(r, 0) + 1
        _tally(by_source, s.get("source") or label.get("source"), decision)
        _tally(by_category, s.get("category_target") or label.get("category_target"), decision)
        _tally(by_track, s.get("training_track") or label.get("training_track"), decision)

        if decision == "auto_approve":
            counts["auto_approved"] += 1
            if apply:
                _set_approval(label, "auto_approved")
                label["triage_policy"] = policy
                write_json(lab_path, label)
                s["approval_status"] = label["approval_status"]
        elif decision == "auto_reject":
            counts["auto_rejected"] += 1
            if apply:
                _set_approval(label, "rejected")
                label["triage_policy"] = policy
                label["triage_reasons"] = reasons
                write_json(lab_path, label)
                s["approval_status"] = "rejected"
        else:
            counts["review_needed"] += 1
            if apply:
                label["approval_status"] = "review_needed"
                label["review_needed"] = True
                label["usable_for_training"] = False
                write_json(lab_path, label)
                s["approval_status"] = "review_needed"

    if apply:
        _sync_manifest_status(dataset, man)
        save_manifest(dataset, man)
    report = _write_triage_report(dataset, policy, apply, counts, reason_hist,
                                  by_source, by_category, by_track)
    mode = "APPLIED" if apply else "DRY-RUN (no labels modified)"
    log(f"Auto-triage [{policy}] {mode}: {counts['auto_approved']} approved, "
        f"{counts['auto_rejected']} rejected, {counts['review_needed']} need review "
        f"(of {counts['considered']}).")
    log(f"  reasons: {reason_hist}")
    log(f"  by source: { {k: v for k, v in by_source.items()} }")
    log(f"  report: {report}")
    return {"status": "ok", "policy": policy, "applied": bool(apply), **counts,
            "reason_histogram": reason_hist, "by_source": by_source,
            "by_category": by_category, "by_training_track": by_track, "report": report}


def _write_triage_report(root, policy, applied, counts, reasons, by_source, by_category, by_track):
    path = os.path.join(root, "auto_triage_report.md")
    common = sorted(reasons.items(), key=lambda kv: -kv[1])

    def _block(title, bucket):
        out = [f"## {title}"]
        for k, v in sorted(bucket.items()):
            out.append(f"- {k}: approved {v['auto_approved']}, rejected "
                       f"{v['auto_rejected']}, review {v['review_needed']}")
        return out + [""]

    lines = [
        f"# Auto-triage report — {os.path.basename(os.path.normpath(root))}",
        "",
        f"- policy: `{policy}`",
        f"- mode: {'APPLIED' if applied else 'dry-run (no labels modified)'}",
        f"- considered: {counts['considered']}",
        f"- **auto-approved: {counts['auto_approved']}**",
        f"- **auto-rejected: {counts['auto_rejected']}**",
        f"- **still need review: {counts['review_needed']}**",
        "",
        "## Reason histogram",
        *([f"- `{r}`: {n}" for r, n in common] or ["- (none)"]),
        "",
        *_block("By source", by_source),
        *_block("By category_target", by_category),
        *_block("By training_track", by_track),
        "> Auto-triage never blindly approves: only flat/wrinkled, in-bounds, "
        "high-confidence, plausible-geometry pose positives are approved.",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------


def _sync_manifest_status(root: str, man: Dict[str, Any]) -> None:
    """Refresh per-sample approval_status + the manifest usable count from labels."""
    n_usable = 0
    for s in man.get("samples", []):
        try:
            label = read_json(os.path.join(root, s["label"]))
        except FileNotFoundError:
            continue
        s["approval_status"] = label.get("approval_status", s.get("approval_status"))
        if label.get("usable_for_training") and label.get("state") in TRAINABLE_STATES \
                and label_corners_xy(label) is not None:
            n_usable += 1
    man["num_labeled_usable"] = n_usable


def _write_review_report(
    root: str, approved: int, rejected: int, needs_review: int, reasons: Dict[str, int],
) -> str:
    path = os.path.join(root, "review_report.md")
    common = sorted(reasons.items(), key=lambda kv: -kv[1])
    lines = [
        f"# Pseudo-label review report — {os.path.basename(os.path.normpath(root))}",
        "",
        f"- approved: **{approved}**",
        f"- rejected: **{rejected}**",
        f"- still need review: **{needs_review}**",
        "",
        "## Common risk / failure reasons",
    ]
    if common:
        lines += [f"- `{r}`: {n}" for r, n in common]
    else:
        lines.append("- (none recorded)")
    lines += [
        "",
        "> Pseudo-labels are machine-generated. Only approved samples are exported to "
        "YOLO (`export-yolo-towel-pose --include-pseudolabels approved_only`).",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path
