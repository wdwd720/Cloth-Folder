"""Learned residual correction on top of the geometric fold plan (Stage 3).

The geometric + physics planner (:mod:`terafold.planning.fold_plan`) produces a
strong prior: a crease, a grasp/place pair and pick-lift-arc-place heights. A
small learned **residual** model nudges those quantities to fix systematic
errors observed in recorded demonstrations (e.g. the gripper consistently
under-shoots the place point, or a stiffer towel needs a higher lift).

Design goals:

* **numpy-only import.** Torch is *only* needed to train or run a real neural
  residual; it is lazily imported inside the functions that need it. Importing
  this module — and applying a *zero* / callable / ``ResidualCorrection``
  residual — never requires torch.
* **Safety by construction.** Every residual is clamped to small, physically
  sane ranges so a mis-trained model can never command a wild motion.
* **Flexible input.** :func:`apply_residual_correction` accepts ``None`` (no
  correction), a :class:`ResidualCorrection`, a callable returning one, a
  checkpoint path (str), or a loaded torch model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids hard dependency
    from terafold.planning.fold_task import FoldTask

__all__ = [
    "RESIDUAL_FEATURE_NAMES",
    "RESIDUAL_OUTPUT_NAMES",
    "PLAN_FEATURE_KEYS",
    "ResidualCorrection",
    "zero_residual",
    "build_residual_features",
    "ResidualMLP",
    "load_residual_model",
    "predict_residual",
    "apply_residual_correction",
    "MAX_GRASP_DXY_M",
    "MAX_PLACE_DXY_M",
    "MAX_LIFT_DELTA_M",
    "MAX_ARC_DELTA_M",
    "MAX_RELEASE_DELTA_M",
]

# --------------------------------------------------------------------------
# Feature / output layout (the public contract for training & inference).
# --------------------------------------------------------------------------

#: Keys consumed from ``FoldPlan.features`` (see ``plan_fold``), in fixed order.
PLAN_FEATURE_KEYS: List[str] = [
    "cloth_width_m",
    "cloth_height_m",
    "cloth_diag_m",
    "rectangularity",
    "friction",
    "stiffness",
    "physics_required_lift_m",
    "physics_arc_height_m",
    "grasp_stability",
    "slip_comp_dx",
    "slip_comp_dy",
    "travel_distance_m",
    "lift_height_m",
    "arc_height_m",
    "release_height_m",
    "grasp_x",
    "grasp_y",
    "place_x",
    "place_y",
]

#: Length of ``FoldTask.task_embedding()`` (4 direction one-hot + 4 scalars).
_TASK_EMBED_DIM = 8

#: Ordered names of the residual feature vector built by
#: :func:`build_residual_features`. Length = len(PLAN_FEATURE_KEYS) + 8 = 27.
RESIDUAL_FEATURE_NAMES: List[str] = PLAN_FEATURE_KEYS + [
    f"task_emb_{i}" for i in range(_TASK_EMBED_DIM)
]

#: Ordered names of the residual model's raw output head.
RESIDUAL_OUTPUT_NAMES: List[str] = [
    "grasp_dx",
    "grasp_dy",
    "place_dx",
    "place_dy",
    "lift_delta",
    "arc_delta",
    "release_delta",
    "confidence",
    "predicted_success",
]

# --------------------------------------------------------------------------
# Safety clamps — a residual can only ever make a *small* correction.
# --------------------------------------------------------------------------

MAX_GRASP_DXY_M: float = 0.05
MAX_PLACE_DXY_M: float = 0.05
MAX_LIFT_DELTA_M: float = 0.05
MAX_ARC_DELTA_M: float = 0.05
MAX_RELEASE_DELTA_M: float = 0.02

# Absolute floors so a negative delta can never invert a height.
_MIN_LIFT_M: float = 0.005
_MIN_ARC_M: float = 0.005
_MIN_RELEASE_M: float = 0.0


@dataclass
class ResidualCorrection:
    """An additive correction to a geometric fold plan.

    All deltas are in metres (table frame). ``confidence`` and
    ``predicted_success`` are probabilities in ``[0, 1]``.
    """

    grasp_dxy: np.ndarray = field(default_factory=lambda: np.zeros(2))
    place_dxy: np.ndarray = field(default_factory=lambda: np.zeros(2))
    lift_delta: float = 0.0
    arc_delta: float = 0.0
    release_delta: float = 0.0
    confidence: float = 1.0
    predicted_success: float = 1.0

    def __post_init__(self) -> None:
        self.grasp_dxy = np.asarray(self.grasp_dxy, dtype=np.float64).reshape(-1)[:2]
        self.place_dxy = np.asarray(self.place_dxy, dtype=np.float64).reshape(-1)[:2]
        if self.grasp_dxy.size < 2:
            self.grasp_dxy = np.zeros(2)
        if self.place_dxy.size < 2:
            self.place_dxy = np.zeros(2)
        self.lift_delta = float(self.lift_delta)
        self.arc_delta = float(self.arc_delta)
        self.release_delta = float(self.release_delta)
        self.confidence = float(self.confidence)
        self.predicted_success = float(self.predicted_success)

    def clamped(self) -> "ResidualCorrection":
        """Return a copy with every quantity clamped to its safe range."""
        return ResidualCorrection(
            grasp_dxy=np.clip(self.grasp_dxy, -MAX_GRASP_DXY_M, MAX_GRASP_DXY_M),
            place_dxy=np.clip(self.place_dxy, -MAX_PLACE_DXY_M, MAX_PLACE_DXY_M),
            lift_delta=float(np.clip(self.lift_delta, -MAX_LIFT_DELTA_M, MAX_LIFT_DELTA_M)),
            arc_delta=float(np.clip(self.arc_delta, -MAX_ARC_DELTA_M, MAX_ARC_DELTA_M)),
            release_delta=float(
                np.clip(self.release_delta, -MAX_RELEASE_DELTA_M, MAX_RELEASE_DELTA_M)
            ),
            confidence=float(np.clip(self.confidence, 0.0, 1.0)),
            predicted_success=float(np.clip(self.predicted_success, 0.0, 1.0)),
        )

    def to_vector(self) -> np.ndarray:
        """Pack into the :data:`RESIDUAL_OUTPUT_NAMES` order (9,)."""
        return np.array(
            [
                self.grasp_dxy[0],
                self.grasp_dxy[1],
                self.place_dxy[0],
                self.place_dxy[1],
                self.lift_delta,
                self.arc_delta,
                self.release_delta,
                self.confidence,
                self.predicted_success,
            ],
            dtype=np.float64,
        )

    def to_dict(self) -> dict:
        return {
            "grasp_dx": float(self.grasp_dxy[0]),
            "grasp_dy": float(self.grasp_dxy[1]),
            "place_dx": float(self.place_dxy[0]),
            "place_dy": float(self.place_dxy[1]),
            "lift_delta": self.lift_delta,
            "arc_delta": self.arc_delta,
            "release_delta": self.release_delta,
            "confidence": self.confidence,
            "predicted_success": self.predicted_success,
        }

    @classmethod
    def from_vector(cls, vec: np.ndarray) -> "ResidualCorrection":
        v = np.asarray(vec, dtype=np.float64).reshape(-1)
        if v.size < len(RESIDUAL_OUTPUT_NAMES):
            raise ValueError(
                f"residual output vector must have >= {len(RESIDUAL_OUTPUT_NAMES)} "
                f"elements, got {v.size}"
            )
        return cls(
            grasp_dxy=v[0:2],
            place_dxy=v[2:4],
            lift_delta=v[4],
            arc_delta=v[5],
            release_delta=v[6],
            confidence=v[7],
            predicted_success=v[8],
        )


def zero_residual() -> ResidualCorrection:
    """An identity correction: no spatial change, full confidence."""
    return ResidualCorrection(
        grasp_dxy=np.zeros(2),
        place_dxy=np.zeros(2),
        lift_delta=0.0,
        arc_delta=0.0,
        release_delta=0.0,
        confidence=1.0,
        predicted_success=1.0,
    )


# --------------------------------------------------------------------------
# Feature engineering.
# --------------------------------------------------------------------------


def build_residual_features(features: Dict[str, float], task: "FoldTask") -> np.ndarray:
    """Build the fixed-length residual feature vector.

    Concatenates the :data:`PLAN_FEATURE_KEYS` values from ``plan.features``
    with ``task.task_embedding()``. The order matches
    :data:`RESIDUAL_FEATURE_NAMES`. Missing feature keys default to ``0.0``.
    """
    plan_vals = [float(features.get(k, 0.0)) for k in PLAN_FEATURE_KEYS]
    emb = np.asarray(task.task_embedding(), dtype=np.float64).reshape(-1)
    if emb.size < _TASK_EMBED_DIM:
        emb = np.concatenate([emb, np.zeros(_TASK_EMBED_DIM - emb.size)])
    else:
        emb = emb[:_TASK_EMBED_DIM]
    return np.concatenate([np.asarray(plan_vals, dtype=np.float64), emb])


# --------------------------------------------------------------------------
# Torch model (lazy). The module imports fine without torch; only building /
# loading / running a neural residual requires it.
# --------------------------------------------------------------------------


def _require_torch():
    """Import torch lazily with an actionable error."""
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - exercised only without torch
        raise ImportError(
            "PyTorch is required for the neural residual model. Install it with "
            "`pip install torch` (or `pip install -e '.[torch]'`). The geometric "
            "planner and zero/callable residuals work without torch."
        ) from exc
    return torch, nn


def ResidualMLP(
    in_dim: int = len(RESIDUAL_FEATURE_NAMES),
    hidden: Tuple[int, ...] = (64, 64),
    out_dim: int = len(RESIDUAL_OUTPUT_NAMES),
):
    """Build the residual MLP as a torch ``nn.Module`` (lazy torch).

    Returns an instantiated module mapping a residual feature vector to the
    :data:`RESIDUAL_OUTPUT_NAMES` head. The spatial/height outputs are raw
    deltas; the last two (confidence, predicted_success) are logits.
    """
    torch, nn = _require_torch()

    class _ResidualMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers: List[Any] = []
            d = int(in_dim)
            for h in hidden:
                layers.append(nn.Linear(d, int(h)))
                layers.append(nn.ReLU())
                d = int(h)
            layers.append(nn.Linear(d, int(out_dim)))
            self.net = nn.Sequential(*layers)
            self.in_dim = int(in_dim)
            self.hidden = tuple(int(h) for h in hidden)
            self.out_dim = int(out_dim)

        def forward(self, x):  # noqa: D401 - simple forward
            return self.net(x)

    return _ResidualMLP()


def load_residual_model(path: str):
    """Load a residual model checkpoint saved by ``train_residual`` (lazy torch)."""
    torch, _ = _require_torch()
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    in_dim = int(cfg.get("in_dim", len(RESIDUAL_FEATURE_NAMES)))
    hidden = tuple(cfg.get("hidden", (64, 64)))
    out_dim = int(cfg.get("out_dim", len(RESIDUAL_OUTPUT_NAMES)))
    model = ResidualMLP(in_dim=in_dim, hidden=hidden, out_dim=out_dim)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _correction_from_output(out: np.ndarray) -> ResidualCorrection:
    """Decode a raw 9-vector head into a :class:`ResidualCorrection`.

    The last two entries (confidence, predicted_success) are treated as logits
    and squashed with a sigmoid.
    """
    v = np.asarray(out, dtype=np.float64).reshape(-1)
    if v.size < len(RESIDUAL_OUTPUT_NAMES):
        raise ValueError(
            f"residual model must output {len(RESIDUAL_OUTPUT_NAMES)} values, got {v.size}"
        )
    probs = _sigmoid(v[7:9])
    return ResidualCorrection(
        grasp_dxy=v[0:2],
        place_dxy=v[2:4],
        lift_delta=v[4],
        arc_delta=v[5],
        release_delta=v[6],
        confidence=float(probs[0]),
        predicted_success=float(probs[1]),
    )


def predict_residual(model: Any, feat_vec: np.ndarray) -> ResidualCorrection:
    """Run a residual model on a feature vector and decode the correction.

    ``model`` may be a torch module (run under ``no_grad``) or a plain callable
    returning a raw 9-vector.
    """
    feat = np.asarray(feat_vec, dtype=np.float64).reshape(-1)
    if hasattr(model, "forward") or hasattr(model, "state_dict"):
        torch, _ = _require_torch()
        was_training = bool(getattr(model, "training", False))
        if hasattr(model, "eval"):
            model.eval()
        with torch.no_grad():
            t = torch.as_tensor(feat, dtype=torch.float32).unsqueeze(0)
            out = model(t).squeeze(0).detach().cpu().numpy()
        if was_training and hasattr(model, "train"):
            model.train()
    elif callable(model):
        out = np.asarray(model(feat), dtype=np.float64).reshape(-1)
    else:
        raise TypeError(f"unsupported residual model type: {type(model)!r}")
    return _correction_from_output(out)


# --------------------------------------------------------------------------
# The public entry point used by ``plan_fold``.
# --------------------------------------------------------------------------


def _resolve_correction(
    residual_model: Any, features: Dict[str, float], task: "FoldTask"
) -> ResidualCorrection:
    """Normalize any accepted ``residual_model`` input to a ResidualCorrection."""
    if residual_model is None:
        return zero_residual()
    if isinstance(residual_model, ResidualCorrection):
        return residual_model
    if isinstance(residual_model, str):
        # A checkpoint path -> load a torch model and run it.
        model = load_residual_model(residual_model)
        return predict_residual(model, build_residual_features(features, task))
    if hasattr(residual_model, "forward") or hasattr(residual_model, "state_dict"):
        # A loaded torch module.
        return predict_residual(residual_model, build_residual_features(features, task))
    if callable(residual_model):
        # A user callable. Prefer passing the features dict; accept either a
        # ResidualCorrection or a raw output vector back.
        out = residual_model(features)
        if isinstance(out, ResidualCorrection):
            return out
        return _correction_from_output(np.asarray(out, dtype=np.float64).reshape(-1))
    raise TypeError(f"unsupported residual_model type: {type(residual_model)!r}")


def apply_residual_correction(
    residual_model: Any,
    features: Dict[str, float],
    task: "FoldTask",
    grasp: np.ndarray,
    place: np.ndarray,
    lift_height: float,
    arc_height: float,
    release_height: float,
) -> Tuple[np.ndarray, np.ndarray, float, float, float, Dict[str, float]]:
    """Apply a (clamped) learned residual to a geometric plan.

    Works without torch for ``None`` / :class:`ResidualCorrection` / callable
    residuals. Returns the corrected ``(grasp, place, lift_height, arc_height,
    release_height, residual_info)`` where ``residual_info`` carries the applied
    deltas plus ``confidence`` and ``predicted_success``.
    """
    correction = _resolve_correction(residual_model, features, task).clamped()

    grasp = np.asarray(grasp, dtype=np.float64).reshape(-1).copy()
    place = np.asarray(place, dtype=np.float64).reshape(-1).copy()
    grasp[:2] = grasp[:2] + correction.grasp_dxy
    place[:2] = place[:2] + correction.place_dxy

    lift_height = float(max(_MIN_LIFT_M, lift_height + correction.lift_delta))
    arc_height = float(max(_MIN_ARC_M, arc_height + correction.arc_delta))
    release_height = float(max(_MIN_RELEASE_M, release_height + correction.release_delta))

    residual_info: Dict[str, float] = correction.to_dict()
    residual_info["confidence"] = correction.confidence
    residual_info["predicted_success"] = correction.predicted_success
    return grasp, place, lift_height, arc_height, release_height, residual_info
