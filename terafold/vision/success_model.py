"""Learned fold-success model (Model 2).

The geometric :func:`terafold.physics.fold_quality.compute_fold_quality` metrics
are the interpretable, learning-free baseline. This module turns those metrics
(plus a few scale-invariant shape descriptors) into a fixed-length feature
vector via :func:`success_features` and provides a small MLP classifier
(:class:`SuccessNet` / :func:`build_success_model`) that is trained to agree with
— and eventually surpass — the geometric success label.

``success_features`` is pure numpy so the whole feature/inference path runs under
numpy alone. Torch is imported lazily inside the model builders/helpers, so this
module imports cleanly without torch installed.
"""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

from terafold.math.geometry import polygon_area
from terafold.physics.cloth_state import FoldState
from terafold.physics.fold_quality import compute_fold_quality

__all__ = [
    "SUCCESS_FEATURE_NAMES",
    "success_features",
    "build_success_model",
    "SuccessNet",
    "load_success_model",
    "predict_success_prob",
]

# Order is load-bearing: training and inference both rely on it. Keep additions
# at the end so existing checkpoints stay compatible.
SUCCESS_FEATURE_NAMES: List[str] = [
    "overlap_ratio",            # IoU of idealized vs observed after-shape
    "corner_error_norm",        # best-match corner error / before-diagonal
    "edge_alignment_norm",      # crease edge-alignment error / before-diagonal
    "visible_area_ratio",       # after-footprint / before-footprint (~0.5)
    "wrinkle_score",            # texture/boundary roughness in [0, 1]
    "before_rectangularity",    # how rectangular the unfolded cloth is
    "after_rectangularity",     # how rectangular the folded footprint is
    "before_aspect",            # before width/height (clamped)
    "after_aspect",             # after width/height (clamped)
    "width_ratio",              # after width / before width
    "height_ratio",             # after height / before height
    "center_shift_norm",        # centroid displacement / before-diagonal
]

SUCCESS_FEATURE_DIM = len(SUCCESS_FEATURE_NAMES)


def _safe_ratio(num: float, den: float, default: float = 0.0) -> float:
    return float(num / den) if abs(den) > 1e-9 else float(default)


def success_features(
    before_fs: FoldState,
    after_fs: FoldState,
    success_cfg: Any = None,
    direction: str = "right_to_left",
) -> np.ndarray:
    """Interpretable, scale-invariant success features for a fold (numpy only).

    Combines the geometric fold-quality metrics with a few shape descriptors so
    the learned model has a low-dimensional, physically meaningful input. The
    distance metrics are normalized by the before-cloth diagonal so the features
    are invariant to image resolution / table scale.

    Returns a ``float32`` array of length :data:`SUCCESS_FEATURE_DIM` whose order
    matches :data:`SUCCESS_FEATURE_NAMES`.
    """
    metrics = compute_fold_quality(
        before_fs, after_fs, success_cfg=success_cfg, direction=direction
    )

    before_kp = before_fs.keypoints
    after_kp = after_fs.keypoints

    before_diag = float(np.hypot(before_kp.width(), before_kp.height()))
    norm = before_diag if before_diag > 1e-9 else 1.0

    corner_err = metrics.corner_error_m
    if not np.isfinite(corner_err):
        corner_err = norm  # treat as "a whole cloth off" rather than infinite
    edge_err = metrics.edge_alignment_error
    if not np.isfinite(edge_err):
        edge_err = norm

    before_w, before_h = before_kp.width(), before_kp.height()
    after_w, after_h = after_kp.width(), after_kp.height()

    center_shift = float(np.linalg.norm(after_kp.center - before_kp.center))

    feats = np.array(
        [
            float(np.clip(metrics.overlap_ratio, 0.0, 1.0)),
            float(np.clip(corner_err / norm, 0.0, 4.0)),
            float(np.clip(edge_err / norm, 0.0, 4.0)),
            float(np.clip(metrics.visible_area_ratio, 0.0, 2.0)),
            float(np.clip(metrics.wrinkle_score, 0.0, 1.0)),
            float(np.clip(before_kp.rectangularity(), 0.0, 1.0)),
            float(np.clip(after_kp.rectangularity(), 0.0, 1.0)),
            float(np.clip(_safe_ratio(before_w, before_h, 1.0), 0.0, 4.0)),
            float(np.clip(_safe_ratio(after_w, after_h, 1.0), 0.0, 4.0)),
            float(np.clip(_safe_ratio(after_w, before_w, 0.0), 0.0, 2.0)),
            float(np.clip(_safe_ratio(after_h, before_h, 0.0), 0.0, 2.0)),
            float(np.clip(center_shift / norm, 0.0, 4.0)),
        ],
        dtype=np.float32,
    )
    return feats


# --------------------------------------------------------------------------
# Learned classifier (lazy torch)
# --------------------------------------------------------------------------


def _require_torch():
    """Import torch lazily with an actionable error message."""
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - only without torch
        raise ImportError(
            "PyTorch is required for the learned success model. Install it with "
            "`pip install torch` (or `pip install -e '.[torch]'`). The geometric "
            "success metric (terafold.physics.fold_quality) works without torch."
        ) from exc
    return torch, nn


def build_success_model(
    feature_dim: int = SUCCESS_FEATURE_DIM,
    hidden: tuple = (64, 32),
):
    """Build the success classifier as a torch ``nn.Module`` (lazy torch).

    Maps a :func:`success_features` vector to a single success logit (use a
    sigmoid for probability). Returns an instantiated module.
    """
    torch, nn = _require_torch()

    class _SuccessNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers: List[Any] = []
            d = int(feature_dim)
            for h in hidden:
                layers.append(nn.Linear(d, int(h)))
                layers.append(nn.ReLU())
                d = int(h)
            layers.append(nn.Linear(d, 1))
            self.net = nn.Sequential(*layers)
            self.feature_dim = int(feature_dim)
            self.hidden = tuple(int(h) for h in hidden)

        def forward(self, x):  # noqa: D401 - simple forward
            return self.net(x).squeeze(-1)

    return _SuccessNet()


def SuccessNet(feature_dim: int = SUCCESS_FEATURE_DIM, hidden: tuple = (64, 32)):
    """Alias for :func:`build_success_model` (the spec exposes ``SuccessNet``)."""
    return build_success_model(feature_dim=feature_dim, hidden=hidden)


def load_success_model(checkpoint: str):
    """Load a success model checkpoint saved by ``train_success`` (lazy torch)."""
    torch, _ = _require_torch()
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    feature_dim = int(cfg.get("feature_dim", SUCCESS_FEATURE_DIM))
    hidden = tuple(cfg.get("hidden", (64, 32)))
    model = build_success_model(feature_dim=feature_dim, hidden=hidden)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model


def predict_success_prob(
    model: Any,
    before_fs: FoldState,
    after_fs: FoldState,
    success_cfg: Any = None,
    direction: str = "right_to_left",
) -> float:
    """Return P(success) in [0, 1] for a before/after pair.

    ``model`` may be a torch module (run under ``no_grad``) or any callable
    mapping a feature vector to a raw logit. Features are computed with the
    numpy-only :func:`success_features`.
    """
    feat = success_features(before_fs, after_fs, success_cfg=success_cfg, direction=direction)
    if hasattr(model, "forward") or hasattr(model, "state_dict"):
        torch, _ = _require_torch()
        if hasattr(model, "eval"):
            model.eval()
        with torch.no_grad():
            t = torch.as_tensor(np.asarray(feat), dtype=torch.float32).unsqueeze(0)
            logit = float(model(t).reshape(-1)[0].item())
    elif callable(model):
        logit = float(np.asarray(model(feat), dtype=np.float64).reshape(-1)[0])
    else:
        raise TypeError(f"unsupported success model type: {type(model)!r}")
    return float(1.0 / (1.0 + np.exp(-np.clip(logit, -30.0, 30.0))))
