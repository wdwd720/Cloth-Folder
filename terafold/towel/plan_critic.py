"""Rule-based fold-plan critic (+ a trainable scaffold for later).

Given a small bag of *plan features* (towel corners, state, keypoint confidence,
grasp/place points, fold axis, visibility) the critic estimates how likely a fold
plan is to succeed and surfaces *why* it might fail. Today this is a transparent,
deterministic rule engine — no learning required, numpy/stdlib only — so it can run
inside the (contact-locked) ghost-fold preview and refuse obviously-bad plans.

The same feature schema also feeds a tiny **trainable** scaffold:

* :func:`build_plan_critic_dataset` turns a towel dataset into a per-sample feature
  table with a heuristic binary ``success`` label (``critic_dataset.jsonl``);
* :func:`train_plan_critic` fits a one-layer logistic regression (torch if present,
  otherwise pure-numpy gradient descent — same model dict either way);
* :func:`eval_plan_critic` scores rows and writes a small Markdown report.

Everything degrades gracefully: torch is imported lazily and only as an optional
accelerator. With only numpy + pyyaml installed the whole pipeline still runs via
the numpy backend. Scope is towel perception/planning only — no action policy, no
contact, no hardware.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from terafold.data.episode_schema import read_jsonl, write_json, write_jsonl
from terafold.towel.dataset import iter_samples
from terafold.towel.schema import (CORNER_ORDER, REJECT_STATES, bbox_from_corners,
                                   is_usable, label_corners_xy)

__all__ = [
    "RISK_REASONS",
    "HARD_FAIL_REASONS",
    "FEATURE_NAMES",
    "critique_plan",
    "build_plan_critic_dataset",
    "train_plan_critic",
    "eval_plan_critic",
]

# Every reason the critic can emit (stable order — also the report column order).
RISK_REASONS = [
    "low_confidence",
    "bad_view",
    "multiple_towels",
    "not_towel",
    "already_folded",
    "corner_geometry_bad",
    "towel_too_occluded",
    "fold_axis_bad",
    "grasp_too_close_to_corner",
]

# Reasons that should drive success_probability to ~0 regardless of anything else.
HARD_FAIL_REASONS = ("bad_view", "multiple_towels", "not_towel")

# Tuning constants (kept explicit so the rules are auditable).
MIN_AREA_RATIO = 0.10        # quad area / bbox area below this ⇒ near-degenerate
EXTREME_ASPECT = 6.0         # long/short edge ratio above this ⇒ not a towel-like quad
GRASP_CORNER_FRAC = 0.08     # grasp within this fraction of the image diag of a corner

# Ordered numeric features used by the trainable scaffold.
FEATURE_NAMES = [
    "keypoint_confidence",
    "has_4_corners",
    "area_ratio",
    "convex",
    "aspect_inv",
    "fold_axis_ok",
]


# ----------------------------------------------------------------------
# Geometry / feature helpers (defensive — never raise on junk input).
# ----------------------------------------------------------------------


def _corners_array(corners: Any) -> Optional[np.ndarray]:
    """Coerce corners (``4x2`` list or ``{tl,tr,br,bl}`` dict) to a ``(4,2)`` array."""
    if corners is None:
        return None
    try:
        if isinstance(corners, dict):
            pts = []
            for k in CORNER_ORDER:
                v = corners.get(k)
                if v is None or len(v) != 2:
                    return None
                pts.append([float(v[0]), float(v[1])])
            arr = np.asarray(pts, dtype=float)
        else:
            arr = np.asarray(corners, dtype=float)
    except (TypeError, ValueError):
        return None
    if arr.shape != (4, 2) or not np.all(np.isfinite(arr)):
        return None
    return arr


def _polygon_area(pts: np.ndarray) -> float:
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _is_convex(pts: np.ndarray) -> bool:
    """True if the 4 corners (in tl,tr,br,bl order) form a convex, non-self-crossing quad."""
    n = len(pts)
    pos = neg = False
    for i in range(n):
        a, b, c = pts[i], pts[(i + 1) % n], pts[(i + 2) % n]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if cross > 1e-9:
            pos = True
        elif cross < -1e-9:
            neg = True
    return not (pos and neg)


def _geometry_quality(pts: Optional[np.ndarray], aspect_hint: Any = None) -> Dict[str, Any]:
    """Return a quality report for the corner geometry (keys always present)."""
    d: Dict[str, Any] = {
        "num_corners": 0 if pts is None else int(len(pts)),
        "area": 0.0, "bbox_area": 0.0, "area_ratio": 0.0,
        "convex": False, "aspect": float("inf"), "ok": False, "issues": [],
    }
    if pts is None or len(pts) != 4:
        d["issues"] = ["not_4_corners"]
        return d
    x1, y1, x2, y2 = bbox_from_corners(pts.tolist())
    w, h = (x2 - x1), (y2 - y1)
    bbox_area = w * h
    area = _polygon_area(pts)
    d["area"], d["bbox_area"] = area, bbox_area
    d["area_ratio"] = (area / bbox_area) if bbox_area > 1e-9 else 0.0
    d["convex"] = _is_convex(pts)
    try:
        a = float(aspect_hint) if aspect_hint not in (None, 0) else 0.0
    except (TypeError, ValueError):
        a = 0.0
    if a > 0:
        ratio = max(a, 1.0 / a)
    else:
        long_, short_ = max(w, h), min(w, h)
        ratio = (long_ / short_) if short_ > 1e-9 else float("inf")
    d["aspect"] = ratio
    issues: List[str] = []
    if bbox_area <= 1e-9 or d["area_ratio"] < MIN_AREA_RATIO:
        issues.append("near_zero_area")
    if not d["convex"]:
        issues.append("non_convex")
    if ratio > EXTREME_ASPECT:
        issues.append("extreme_aspect")
    d["issues"] = issues
    d["ok"] = not issues
    return d


def _seg_len(p1: Sequence[float], p2: Sequence[float]) -> float:
    return math.hypot(float(p2[0]) - float(p1[0]), float(p2[1]) - float(p1[1]))


def _fold_axis_ok(fold_axis: Any) -> bool:
    """True if a non-degenerate fold axis is described (segment, points, angle, or name)."""
    if fold_axis is None or isinstance(fold_axis, bool):
        return False
    if isinstance(fold_axis, (int, float)):
        return math.isfinite(float(fold_axis))
    if isinstance(fold_axis, str):
        return bool(fold_axis.strip())
    if isinstance(fold_axis, dict):
        p1, p2 = fold_axis.get("p1"), fold_axis.get("p2")
        if p1 is not None and p2 is not None:
            try:
                return _seg_len(p1, p2) > 1e-6
            except (TypeError, ValueError, IndexError):
                return False
        return fold_axis.get("angle") is not None or bool(fold_axis.get("direction"))
    try:
        flat = np.asarray(fold_axis, dtype=float).ravel()
    except (TypeError, ValueError):
        return False
    if flat.size == 4 and np.all(np.isfinite(flat)):
        return math.hypot(flat[2] - flat[0], flat[3] - flat[1]) > 1e-6
    return False


def _image_diag(features: Dict[str, Any], pts: Optional[np.ndarray]) -> Optional[float]:
    """Best-effort image diagonal: explicit size first, else the towel's own bbox diag."""
    sz = features.get("image_size")
    try:
        if sz is not None and len(sz) == 2:
            return math.hypot(float(sz[0]), float(sz[1]))
    except (TypeError, ValueError):
        pass
    w, h = features.get("image_width"), features.get("image_height")
    try:
        if w and h:
            return math.hypot(float(w), float(h))
    except (TypeError, ValueError):
        pass
    d = features.get("image_diag")
    try:
        if d:
            return float(d)
    except (TypeError, ValueError):
        pass
    if pts is not None:
        x1, y1, x2, y2 = bbox_from_corners(pts.tolist())
        return math.hypot(x2 - x1, y2 - y1)
    return None


def _as_point(p: Any) -> Optional[Tuple[float, float]]:
    try:
        if p is not None and len(p) == 2:
            return float(p[0]), float(p[1])
    except (TypeError, ValueError):
        return None
    return None


# ----------------------------------------------------------------------
# The rule-based critic.
# ----------------------------------------------------------------------


def critique_plan(features: Optional[Dict[str, Any]], min_confidence: float = 0.75) -> Dict[str, Any]:
    """Score a fold plan from its features and explain the risk.

    ``features`` (all keys optional, parsed defensively):
      ``corners`` (``4x2`` list or ``{tl,tr,br,bl}``), ``aspect_ratio``, ``state``,
      ``keypoint_confidence``, ``grasp_point`` ``[x,y]``, ``place_point`` ``[x,y]``,
      ``fold_direction``, ``fold_axis``, ``visible`` (dict of bools), optional
      ``image_size``/``image_width``/``image_height`` for the grasp proximity rule.

    Returns ``{success_probability, risk_score, risk_reasons:[...], detail:{...}}``.
    ``success_probability = clip(prod(1 - penalty_i), 0, 1)``; a hard-fail state
    forces it to ~0.
    """
    features = dict(features or {})
    reasons: List[str] = []
    penalties: Dict[str, float] = {}
    detail: Dict[str, Any] = {}

    pts = _corners_array(features.get("corners"))
    state = features.get("state")
    conf = features.get("keypoint_confidence")
    visible = features.get("visible")
    fold_axis = features.get("fold_axis")
    grasp = _as_point(features.get("grasp_point"))

    def _add(reason: str, penalty: float) -> None:
        if reason not in penalties:
            reasons.append(reason)
        penalties[reason] = max(penalties.get(reason, 0.0), float(np.clip(penalty, 0.0, 1.0)))

    # --- confidence ----------------------------------------------------
    if conf is not None:
        try:
            cval = float(conf)
        except (TypeError, ValueError):
            cval = 0.0
        detail["keypoint_confidence"] = cval
        if cval < min_confidence:
            deficit = (min_confidence - cval) / max(min_confidence, 1e-6)
            _add("low_confidence", float(np.clip(deficit, 0.2, 1.0)))

    # --- state ---------------------------------------------------------
    if state is not None:
        detail["state"] = state
        if state in HARD_FAIL_REASONS:
            _add(state, 1.0)            # hard fail ⇒ probability ~0
        elif state == "folded_success":
            _add("already_folded", 0.9)

    # --- corner geometry ----------------------------------------------
    geom = _geometry_quality(pts, features.get("aspect_ratio"))
    detail["geometry"] = geom
    if not geom["ok"]:
        # Missing corners is a harder fail than a slightly-off but present quad.
        penalty = 0.85 if geom["num_corners"] != 4 else 0.8
        _add("corner_geometry_bad", penalty)

    # --- visibility / occlusion ---------------------------------------
    if isinstance(visible, dict) and visible:
        n_hidden = sum(1 for k in CORNER_ORDER if visible.get(k) is False)
        detail["hidden_corners"] = n_hidden
        if n_hidden >= 1:
            _add("towel_too_occluded", float(np.clip(0.35 + 0.2 * n_hidden, 0.0, 0.95)))

    # --- fold axis -----------------------------------------------------
    detail["fold_axis_ok"] = _fold_axis_ok(fold_axis)
    if not detail["fold_axis_ok"]:
        _add("fold_axis_bad", 0.6)

    # --- grasp proximity to a corner ----------------------------------
    diag = _image_diag(features, pts)
    if grasp is not None and pts is not None and diag and diag > 1e-6:
        dists = [math.hypot(grasp[0] - c[0], grasp[1] - c[1]) for c in pts]
        min_d = min(dists)
        detail["grasp_corner_min_dist"] = min_d
        detail["grasp_corner_min_frac"] = min_d / diag
        if min_d < GRASP_CORNER_FRAC * diag:
            _add("grasp_too_close_to_corner", 0.5)

    # --- combine -------------------------------------------------------
    prob = 1.0
    for r in reasons:
        prob *= (1.0 - penalties[r])
    if any(r in HARD_FAIL_REASONS for r in reasons):
        prob = 0.0
    prob = float(np.clip(prob, 0.0, 1.0))
    detail["penalties"] = penalties

    return {
        "success_probability": prob,
        "risk_score": float(np.clip(1.0 - prob, 0.0, 1.0)),
        "risk_reasons": reasons,
        "detail": detail,
    }


# ----------------------------------------------------------------------
# Trainable scaffold: dataset / train / eval.
# ----------------------------------------------------------------------


def _resolve_jsonl(data: str) -> str:
    """Accept either a directory holding ``critic_dataset.jsonl`` or the jsonl path."""
    if os.path.isdir(data):
        return os.path.join(data, "critic_dataset.jsonl")
    return data


def _row_features(row: Dict[str, Any]) -> List[float]:
    """Map a dataset/plan row to the fixed-order numeric feature vector."""
    try:
        conf = float(row.get("keypoint_confidence") or 0.0)
    except (TypeError, ValueError):
        conf = 0.0
    pts = _corners_array(row.get("corners"))
    if pts is not None:
        geom = _geometry_quality(pts, row.get("aspect_ratio"))
        has = 1.0
        area_ratio = float(geom["area_ratio"])
        convex = 1.0 if geom["convex"] else 0.0
        aspect = geom["aspect"]
        aspect_inv = (1.0 / aspect) if math.isfinite(aspect) and aspect > 0 else 0.0
    else:
        has = area_ratio = convex = aspect_inv = 0.0
    fa = 1.0 if _fold_axis_ok(row.get("fold_axis")) else 0.0
    return [conf, has, area_ratio, convex, aspect_inv, fa]


def build_plan_critic_dataset(
    data: str, out: str, log: Callable[[str], None] = print,
) -> Dict[str, Any]:
    """Derive a plan-critic feature table from a towel dataset.

    Includes every *usable* sample (positive candidates) plus every reject-state
    sample (``bad_view``/``multiple_towels``/``not_towel``) as negatives. Writes
    ``out/critic_dataset.jsonl`` + ``out/meta.json``.
    """
    os.makedirs(out, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    positives = negatives = 0

    for sample, label in iter_samples(data):
        usable = is_usable(label)
        state = label.get("state")
        if not (usable or state in REJECT_STATES):
            continue

        corners_xy = label_corners_xy(label)
        pts = _corners_array(corners_xy)
        if pts is not None:
            x1, y1, x2, y2 = bbox_from_corners(pts.tolist())
            w, h = (x2 - x1), (y2 - y1)
            aspect = (w / h) if h > 1e-9 else None
            cx = (x1 + x2) / 2.0
            fold_axis: Any = [[cx, y1], [cx, y2]]
            tl, tr, br, bl = pts
            grasp = [float((tr[0] + br[0]) / 2.0), float((tr[1] + br[1]) / 2.0)]
            place = [float((tl[0] + bl[0]) / 2.0), float((tl[1] + bl[1]) / 2.0)]
        else:
            aspect = None
            fold_axis = None
            grasp = None
            place = None

        geom_ok = _geometry_quality(pts, aspect)["ok"]
        success = 1 if (usable and geom_ok) else 0
        conf = 0.9 if usable else 0.4

        rows.append({
            "sample_id": sample.get("id"),
            "image": sample.get("image"),
            "state": state,
            "corners": (pts.tolist() if pts is not None else None),
            "aspect_ratio": aspect,
            "keypoint_confidence": conf,
            "grasp_point": grasp,
            "place_point": place,
            "fold_direction": "right_to_left",
            "fold_axis": fold_axis,
            "label": success,
        })
        positives += success
        negatives += (1 - success)

    jsonl = os.path.join(out, "critic_dataset.jsonl")
    write_jsonl(jsonl, rows)
    meta = {
        "dataset": "plan_critic",
        "source_dataset": os.path.basename(os.path.normpath(data)),
        "n_rows": len(rows),
        "positives": positives,
        "negatives": negatives,
        "feature_names": FEATURE_NAMES,
        "jsonl": jsonl,
    }
    write_json(os.path.join(out, "meta.json"), meta)
    log(f"Built plan-critic dataset: {len(rows)} rows "
        f"({positives} pos / {negatives} neg) -> {jsonl}")
    return {"out": out, "rows": len(rows), "positives": positives, "negatives": negatives}


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def _load_matrix(jsonl: str) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for row in read_jsonl(jsonl):
        X.append(_row_features(row))
        y.append(float(row.get("label", 0)))
    if not X:
        return np.zeros((0, len(FEATURE_NAMES)), float), np.zeros((0,), float)
    return np.asarray(X, dtype=float), np.asarray(y, dtype=float)


def _accuracy(w: np.ndarray, b: float, X: np.ndarray, y: np.ndarray) -> float:
    if len(y) == 0:
        return 0.0
    pred = (_sigmoid(X @ w + b) > 0.5).astype(float)
    return float((pred == y).mean())


def _train_numpy(X: np.ndarray, y: np.ndarray, epochs: int, lr: float) -> Tuple[np.ndarray, float]:
    n, d = X.shape
    w = np.zeros(d, dtype=float)
    b = 0.0
    if n == 0:
        return w, b
    for _ in range(epochs):
        p = _sigmoid(X @ w + b)
        err = p - y
        w -= lr * (X.T @ err) / n
        b -= lr * float(err.mean())
    return w, b


def _train_torch(X: np.ndarray, y: np.ndarray, epochs: int, lr: float) -> Tuple[np.ndarray, float]:
    import torch

    d = X.shape[1]
    model = torch.nn.Linear(d, 1)
    opt = torch.optim.Adam(model.parameters(), lr=max(lr * 0.1, 0.01))
    loss_fn = torch.nn.BCEWithLogitsLoss()
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(Xt), yt)
        loss.backward()
        opt.step()
    w = model.weight.detach().cpu().numpy().reshape(-1)
    b = float(model.bias.detach().cpu().numpy().reshape(-1)[0])
    return w, b


def train_plan_critic(
    data: str, out: str, log: Callable[[str], None] = print,
    backend: str = "auto", epochs: int = 400, lr: float = 0.5,
) -> Dict[str, Any]:
    """Fit a 1-layer logistic-regression plan critic.

    ``backend``: ``"auto"`` uses torch when importable, else numpy; pass ``"numpy"``
    or ``"torch"`` to force one. The saved ``model.pt`` is a portable dict either
    way (numpy → JSON, torch → ``torch.save``), so eval is backend-agnostic.
    """
    jsonl = _resolve_jsonl(data)
    X, y = _load_matrix(jsonl)

    use_torch = False
    if backend == "torch":
        use_torch = True
    elif backend == "auto":
        try:
            import torch  # noqa: F401
            use_torch = True
        except Exception:
            use_torch = False

    if use_torch and len(y) > 0:
        try:
            w, b = _train_torch(X, y, epochs, lr)
            backend_used = "torch"
        except Exception as exc:  # pragma: no cover - torch present but failed
            log(f"[warn] torch training failed ({exc}); falling back to numpy.")
            w, b = _train_numpy(X, y, epochs, lr)
            backend_used = "numpy"
    else:
        w, b = _train_numpy(X, y, epochs, lr)
        backend_used = "numpy"

    train_acc = _accuracy(w, b, X, y)
    model_path = out if out.endswith(".pt") else os.path.join(out, "model.pt")
    os.makedirs(os.path.dirname(os.path.abspath(model_path)) or ".", exist_ok=True)
    model_obj = {
        "backend": backend_used,
        "model": "logistic_regression",
        "feature_names": FEATURE_NAMES,
        "weights": [float(v) for v in w],
        "bias": float(b),
        "n_rows": int(len(y)),
        "train_acc": train_acc,
    }
    if backend_used == "torch":
        import torch
        torch.save(model_obj, model_path)
    else:
        write_json(model_path, model_obj)

    log(f"Trained plan critic ({backend_used}): n={len(y)}, train_acc={train_acc:.3f} "
        f"-> {model_path}")
    return {"out": out, "model": model_path, "backend": backend_used,
            "n_rows": int(len(y)), "train_acc": train_acc}


def _load_model(path: str) -> Dict[str, Any]:
    """Load a model.pt saved as JSON (numpy) or via torch.save (torch)."""
    try:
        with open(path, "r") as f:
            return json.loads(f.read())
    except (UnicodeDecodeError, ValueError):
        import torch
        return dict(torch.load(path, weights_only=False))


def eval_plan_critic(
    data: str, model: str, out: str, log: Callable[[str], None] = print,
) -> Dict[str, Any]:
    """Score every row, compute accuracy + a confusion summary, write a Markdown report."""
    jsonl = _resolve_jsonl(data)
    X, y = _load_matrix(jsonl)
    mdl = _load_model(model)
    w = np.asarray(mdl.get("weights", []), dtype=float)
    b = float(mdl.get("bias", 0.0))
    backend = mdl.get("backend", "numpy")

    n = int(len(y))
    if n == 0 or w.size != X.shape[1]:
        pred = np.zeros((n,), dtype=float)
    else:
        pred = (_sigmoid(X @ w + b) > 0.5).astype(float)

    tp = int(np.sum((pred == 1) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    accuracy = float((pred == y).mean()) if n else 0.0

    report_path = out if out.endswith(".md") else os.path.join(out, "eval.md")
    os.makedirs(os.path.dirname(os.path.abspath(report_path)) or ".", exist_ok=True)
    lines = [
        "# Plan-critic evaluation",
        "",
        f"- model: `{model}`  (backend: **{backend}**)",
        f"- rows scored: **{n}**",
        f"- accuracy: **{accuracy:.3f}**",
        "",
        "## Confusion (predicted vs. heuristic label)",
        "",
        "| | pred success | pred fail |",
        "|---|---|---|",
        f"| **actual success** | {tp} | {fn} |",
        f"| **actual fail** | {fp} | {tn} |",
        "",
        f"- true positives: {tp}  true negatives: {tn}",
        f"- false positives: {fp}  false negatives: {fn}",
        "",
        "Features: " + ", ".join(FEATURE_NAMES),
    ]
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    log(f"Evaluated plan critic: n={n}, accuracy={accuracy:.3f} -> {report_path}")
    return {
        "status": "ok",
        "accuracy": accuracy,
        "n": n,
        "report": report_path,
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }
