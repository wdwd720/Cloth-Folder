"""Outcome metrics for a completed fold (numpy-only).

These answer the questions a fold-success detector cares about:

* did the moving corners actually **cross the crease** (the fold line)?
* how far are the post-fold corners from the **target** footprint?
* did the fold land on the **correct side** of the crease?
* is the fold a **success** given an edge-error threshold?

Pure numpy geometry on small corner arrays; no perception, no hardware.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from terafold.eval.perception_metrics import _line_point_dir, corner_error

__all__ = [
    "crossed_crease",
    "final_edge_error",
    "target_side_correct",
    "fold_success",
    "signed_side",
]


def _as_points(arr: Any) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 1:
        a = a.reshape(-1, 2)
    if a.ndim != 2 or a.shape[1] != 2:
        raise ValueError(f"expected an (N, 2) point array, got shape {a.shape}")
    return a


def signed_side(points: Any, fold_line: Any) -> np.ndarray:
    """Signed side (+1 / -1 / 0) of each point relative to the fold line.

    The sign is that of the 2-D cross product of the line direction with the
    point offset, so it is consistent for an undirected crease as long as the
    same line representation is used throughout.
    """
    pts = _as_points(points)
    p0, d, _ = _line_point_dir(fold_line)
    rel = pts - p0
    cross = d[0] * rel[:, 1] - d[1] * rel[:, 0]
    return np.sign(cross)


def crossed_crease(before_corners: Any, after_corners: Any, fold_line: Any,
                   eps: float = 1e-9) -> bool:
    """True iff at least one corner moved to the other side of the fold line.

    A corner counts as crossed when its signed side flips from clearly positive
    to clearly negative (or vice-versa); points sitting on the line (within
    ``eps``) are ignored.
    """
    before = _as_points(before_corners)
    after = _as_points(after_corners)
    p0, d, _ = _line_point_dir(fold_line)

    def side(pts: np.ndarray) -> np.ndarray:
        rel = pts - p0
        return d[0] * rel[:, 1] - d[1] * rel[:, 0]

    n = min(len(before), len(after))
    sb, sa = side(before[:n]), side(after[:n])
    for i in range(n):
        if abs(sb[i]) <= eps or abs(sa[i]) <= eps:
            continue
        if np.sign(sb[i]) != np.sign(sa[i]):
            return True
    return False


def final_edge_error(after: Any, target: Any) -> float:
    """Mean per-corner Euclidean error between the folded and target corners."""
    return float(corner_error(after, target)["mean"])


def target_side_correct(after_corners: Any, fold_line: Any, target_side: int) -> bool:
    """True iff the folded corners lie on the expected side of the crease.

    ``target_side`` is +1 or -1 (matching :func:`signed_side`). The check uses the
    centroid of the post-fold corners so a single noisy corner does not flip the
    verdict.
    """
    pts = _as_points(after_corners)
    centroid = pts.mean(axis=0, keepdims=True)
    s = signed_side(centroid, fold_line)[0]
    return bool(np.sign(s) == np.sign(float(target_side)) and s != 0.0)


def fold_success(edge_error: float, threshold: float) -> bool:
    """True iff the final edge error is within ``threshold`` (a successful fold)."""
    e = float(edge_error)
    return bool(np.isfinite(e) and e <= float(threshold))
