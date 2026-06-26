"""Aggregate evaluation: episode benchmarks and synthetic planner sweeps.

Two complementary entry points:

* :func:`benchmark_episodes` — scan recorded episode directories, read each
  ``result.json``, and aggregate the success rate + metric statistics.
* :func:`benchmark_planner_on_synthetic` — run the geometric planner over a
  synthetic dataset and score each plan against the *ideal* folded footprint
  (:func:`~terafold.physics.fold_geometry.predict_folded_footprint`), giving a
  hardware-free measure of perception + planning quality.

Pure numpy; perception degrades to the classical fallback, so the whole module
runs in the Stage-0 (numpy-only) pipeline.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np

from terafold.data.episode_schema import (
    EpisodeResult,
    list_episode_dirs,
    read_json,
)

__all__ = [
    "benchmark_episodes",
    "print_benchmark",
    "benchmark_planner_on_synthetic",
]

# Metrics aggregated across episodes / samples.
_METRIC_KEYS = [
    "overlap_ratio",
    "corner_error_m",
    "edge_alignment_error",
    "visible_area_ratio",
    "wrinkle_score",
]


def _stats(values: List[float]) -> Dict[str, float]:
    """Mean / std / min / max for a list of floats (NaN-safe, empty-safe)."""
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"),
                "min": float("nan"), "max": float("nan"), "n": 0}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n": int(arr.size),
    }


def _aggregate(successes: List[Optional[bool]], metric_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    known = [bool(s) for s in successes if s is not None]
    success_rate = float(np.mean(known)) if known else float("nan")
    metrics: Dict[str, Dict[str, float]] = {}
    for key in _METRIC_KEYS:
        metrics[key] = _stats([row.get(key) for row in metric_rows])
    return {
        "n": len(metric_rows),
        "n_with_label": len(known),
        "success_rate": success_rate,
        "metrics": metrics,
    }


def benchmark_episodes(
    episodes_dir: str,
    predictor: Any = None,
    success_cfg: Any = None,
) -> Dict[str, Any]:
    """Aggregate success + metric stats over recorded episodes under ``episodes_dir``.

    Reads each episode's ``result.json`` (an :class:`EpisodeResult`). When a
    result is missing but before/after frames are present, the fold is re-scored
    with :func:`~terafold.eval.score_fold.score_fold` (using ``predictor`` /
    ``success_cfg``) so partially-recorded episodes still contribute.
    """
    episode_paths = list_episode_dirs(episodes_dir)
    successes: List[Optional[bool]] = []
    metric_rows: List[Dict[str, Any]] = []

    for ep in episode_paths:
        success: Optional[bool] = None
        metrics: Dict[str, Any] = {}
        if os.path.exists(ep.result):
            try:
                result = EpisodeResult.from_dict(read_json(ep.result))
                success = result.success
                metrics = dict(result.metrics or {})
                # The scorer block may carry the per-metric values directly.
                if not any(k in metrics for k in _METRIC_KEYS) and result.scorer:
                    metrics = dict(result.scorer)
            except Exception:
                pass
        if not metrics:
            rescored = _rescore_episode(ep, predictor, success_cfg)
            if rescored is not None:
                metrics = rescored
                if success is None:
                    success = bool(rescored.get("success", False))
        successes.append(success)
        metric_rows.append(metrics)

    report = _aggregate(successes, metric_rows)
    report["episodes_dir"] = os.path.abspath(episodes_dir)
    return report


def _rescore_episode(ep, predictor, success_cfg) -> Optional[Dict[str, Any]]:
    """Re-score an episode from its first/last frame if a result is missing."""
    frames_dir = ep.frames_dir
    if not os.path.isdir(frames_dir):
        return None
    frame_files = sorted(
        f for f in os.listdir(frames_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    if len(frame_files) < 2:
        return None
    from terafold.eval.score_fold import score_fold
    from terafold.vision.imageio import imread

    before = imread(os.path.join(frames_dir, frame_files[0]))
    after = imread(os.path.join(frames_dir, frame_files[-1]))
    try:
        return score_fold(before, after, predictor=predictor, success_cfg=success_cfg)
    except Exception:
        return None


def print_benchmark(report: Dict[str, Any]) -> str:
    """Render a benchmark ``report`` dict as a human-readable table string."""
    lines: List[str] = []
    lines.append("=" * 56)
    lines.append("TeraFold benchmark")
    src = report.get("episodes_dir") or report.get("synth_dir") or ""
    if src:
        lines.append(f"  source: {src}")
    n = report.get("n", 0)
    n_lab = report.get("n_with_label", n)
    rate = report.get("success_rate", float("nan"))
    lines.append(f"  episodes/samples: {n}  (labeled: {n_lab})")
    lines.append(f"  success rate:     {rate:.3f}" if np.isfinite(rate) else "  success rate:     n/a")
    lines.append("-" * 56)
    lines.append(f"  {'metric':<24}{'mean':>8}{'std':>8}{'min':>8}")
    for key, st in report.get("metrics", {}).items():
        mean, std, mn = st.get("mean"), st.get("std"), st.get("min")
        if mean is None or not np.isfinite(mean):
            lines.append(f"  {key:<24}{'n/a':>8}")
        else:
            lines.append(f"  {key:<24}{mean:>8.3f}{std:>8.3f}{mn:>8.3f}")
    lines.append("=" * 56)
    return "\n".join(lines)


def _resolve_path(base: str, p: str) -> str:
    """Resolve a dataset-relative path against ``base`` (handles abs / nested)."""
    if os.path.isabs(p) and os.path.exists(p):
        return p
    for cand in (p, os.path.join(base, p), os.path.join(base, os.path.basename(p))):
        if os.path.exists(cand):
            return cand
    return os.path.join(base, p)


def benchmark_planner_on_synthetic(
    synth_dir: str,
    task: Any,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Run the geometric planner on a synthetic dataset and score vs the ideal fold.

    For each sample the *ground-truth* before-state (from the synthetic label)
    defines the ideal post-fold footprint via
    :func:`~terafold.physics.fold_geometry.predict_folded_footprint`. The planner
    perceives the same image and produces a plan; the perceived before-state is
    scored against that ideal "after", measuring perception + geometry accuracy
    without any hardware.
    """
    from terafold.physics.cloth_state import FoldState
    from terafold.physics.fold_geometry import predict_folded_footprint
    from terafold.physics.fold_quality import compute_fold_quality
    from terafold.planning.towel_half_fold import TowelHalfFoldPlanner
    from terafold.vision.imageio import imread

    index_path = os.path.join(synth_dir, "index.json")
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"No synthetic index.json under {synth_dir!r}. Generate one with "
            "terafold.vision.synthetic_cloth.generate_synthetic_dataset first."
        )
    index = read_json(index_path)
    items = index.get("items", [])
    if limit is not None:
        items = items[: int(limit)]

    direction = task.direction
    planner = TowelHalfFoldPlanner(task)

    successes: List[Optional[bool]] = []
    metric_rows: List[Dict[str, Any]] = []

    for item in items:
        try:
            image = imread(_resolve_path(synth_dir, item["image"]))
            label = read_json(_resolve_path(synth_dir, item["label"]))
            gt_before = FoldState.from_dict(label["fold_state"])
            ideal_after_kp = predict_folded_footprint(gt_before.keypoints, direction)
            after_fs = FoldState(keypoints=ideal_after_kp, frame=gt_before.frame)

            plan = planner.plan_from_image(image)
            perceived_before = plan.image_fold_state

            metrics = compute_fold_quality(
                perceived_before, after_fs,
                success_cfg=task.success, direction=direction,
            )
            row = {k: getattr(metrics, k) for k in _METRIC_KEYS}
            row["success"] = metrics.success
            metric_rows.append(row)
            successes.append(metrics.success)
        except Exception as exc:  # keep the sweep going on a single bad sample
            metric_rows.append({"error": str(exc)})
            successes.append(None)

    report = _aggregate(successes, metric_rows)
    report["synth_dir"] = os.path.abspath(synth_dir)
    return report
