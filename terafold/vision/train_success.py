"""Train the learned fold-success classifier (Model 2).

Consumes the ``generate_fold_pairs`` on-disk layout from
:mod:`terafold.vision.synthetic_cloth`::

    <data_dir>/pairs/<i:06d>/before.png
    <data_dir>/pairs/<i:06d>/after.png
    <data_dir>/pairs/<i:06d>/label.json   {"success": bool, "metrics": {...}}
    <data_dir>/index.json

For each pair the before/after images are perceived into :class:`FoldState`s
(via the always-available keypoint predictor), turned into the interpretable
:func:`terafold.vision.success_model.success_features` vector, and labelled with
the weak ``success`` label stored alongside the pair (from the geometric
fold-quality scorer). A small MLP is then trained to reproduce that label.

Torch is required to *train* and is imported lazily, so this module imports under
numpy alone. Feature extraction and dataset building are numpy-only.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from terafold.data.episode_schema import read_json, write_json
from terafold.vision.imageio import imread
from terafold.vision.success_model import (
    SUCCESS_FEATURE_DIM,
    SUCCESS_FEATURE_NAMES,
    build_success_model,
    success_features,
)

__all__ = ["build_success_dataset", "train_success"]


def _require_torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - only without torch
        raise ImportError(
            "PyTorch is required to train the success model. Install it with "
            "`pip install torch` (or `pip install -e '.[torch]'`). Dataset "
            "building (terafold.vision.train_success.build_success_dataset) and "
            "the geometric success metric work without torch."
        ) from exc
    return torch, nn


def _load_index(data_dir: str) -> List[dict]:
    """Resolve the list of pair items, tolerating a missing index.json."""
    index_path = os.path.join(data_dir, "index.json")
    if os.path.exists(index_path):
        summary = read_json(index_path)
        items = summary.get("items", [])
        if items:
            return items
    # Fall back to scanning the pairs directory.
    pairs_dir = os.path.join(data_dir, "pairs")
    items = []
    if os.path.isdir(pairs_dir):
        for name in sorted(os.listdir(pairs_dir)):
            rel = f"pairs/{name}"
            if os.path.exists(os.path.join(data_dir, rel, "before.png")):
                items.append(
                    {
                        "before": f"{rel}/before.png",
                        "after": f"{rel}/after.png",
                        "label": f"{rel}/label.json",
                    }
                )
    return items


def build_success_dataset(
    data_dir: str,
    direction: str = "right_to_left",
    limit: Optional[int] = None,
    predictor: Any = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build ``(features (N, D), labels (N,))`` from the fold-pair layout (numpy).

    Each before/after image pair is perceived into a :class:`FoldState`, reduced
    to a :func:`success_features` vector, and labelled with the weak geometric
    ``success`` flag stored in ``label.json``. The predictor (lazily fetched via
    ``get_keypoint_predictor`` if not provided) always works under numpy alone.
    """
    items = _load_index(data_dir)
    if limit is not None:
        items = items[: int(limit)]
    if not items:
        raise FileNotFoundError(
            f"No fold pairs found under {data_dir!r}. Generate them first with "
            "terafold.vision.synthetic_cloth.generate_fold_pairs(out_dir, num)."
        )

    if predictor is None:
        from terafold.vision.infer_keypoints import get_keypoint_predictor

        predictor = get_keypoint_predictor()

    feats: List[np.ndarray] = []
    labels: List[float] = []
    for item in items:
        before_img = imread(os.path.join(data_dir, item["before"]))
        after_img = imread(os.path.join(data_dir, item["after"]))
        label_data = read_json(os.path.join(data_dir, item["label"]))

        before_fs = predictor.predict(before_img)
        after_fs = predictor.predict(after_img)
        feats.append(success_features(before_fs, after_fs, direction=direction))
        labels.append(float(bool(label_data.get("success", False))))

    x = np.asarray(feats, dtype=np.float32)
    y = np.asarray(labels, dtype=np.float32)
    return x, y


def train_success(
    data_dir: str,
    out_dir: str,
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 1e-3,
    device: Optional[str] = None,
    val_frac: float = 0.1,
    limit: Optional[int] = None,
    direction: str = "right_to_left",
    seed: int = 0,
) -> Dict[str, Any]:
    """Train the success classifier and save ``<out>/model.pt`` + ``config.json``.

    Returns a history dict (per-epoch train/val loss and accuracy). Requires
    torch; raises a clear, actionable error if it is missing.
    """
    torch, nn = _require_torch()

    x_np, y_np = build_success_dataset(data_dir, direction=direction, limit=limit)
    n = int(x_np.shape[0])
    if n < 2:
        raise ValueError(
            f"Need at least 2 fold pairs to train, got {n}. Generate more with "
            "generate_fold_pairs()."
        )

    dev = torch.device(device) if device else torch.device("cpu")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = max(1, int(round(n * float(val_frac)))) if n > 1 else 0
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    if train_idx.size == 0:  # tiny datasets: train on everything
        train_idx = perm
        val_idx = perm[:0]

    x = torch.as_tensor(x_np, dtype=torch.float32, device=dev)
    y = torch.as_tensor(y_np, dtype=torch.float32, device=dev)

    model = build_success_model(feature_dim=int(x_np.shape[1])).to(dev)

    # Balance the BCE loss when the synthetic generator skews success/failure.
    n_pos = float(y_np[train_idx].sum())
    n_neg = float(train_idx.size - n_pos)
    pos_weight = torch.tensor(
        [n_neg / n_pos if n_pos > 0 else 1.0], dtype=torch.float32, device=dev
    )
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    def _eval(idx: np.ndarray) -> Tuple[float, float]:
        if idx.size == 0:
            return float("nan"), float("nan")
        model.eval()
        with torch.no_grad():
            ii = torch.as_tensor(idx, dtype=torch.long, device=dev)
            logits = model(x[ii])
            loss = loss_fn(logits, y[ii]).item()
            pred = (torch.sigmoid(logits) >= 0.5).float()
            acc = (pred == y[ii]).float().mean().item()
        return float(loss), float(acc)

    for _ in range(int(epochs)):
        model.train()
        ep = rng.permutation(train_idx)
        ep_loss = 0.0
        ep_correct = 0.0
        for start in range(0, ep.size, int(batch_size)):
            batch = ep[start : start + int(batch_size)]
            bi = torch.as_tensor(batch, dtype=torch.long, device=dev)
            opt.zero_grad()
            logits = model(x[bi])
            loss = loss_fn(logits, y[bi])
            loss.backward()
            opt.step()
            ep_loss += float(loss.item()) * batch.size
            with torch.no_grad():
                pred = (torch.sigmoid(logits) >= 0.5).float()
                ep_correct += float((pred == y[bi]).float().sum().item())
        history["train_loss"].append(ep_loss / max(1, train_idx.size))
        history["train_acc"].append(ep_correct / max(1, train_idx.size))
        vl, va = _eval(val_idx)
        history["val_loss"].append(vl)
        history["val_acc"].append(va)

    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "model.pt")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "feature_dim": int(x_np.shape[1]),
                "hidden": list(getattr(model, "hidden", (64, 32))),
                "feature_names": SUCCESS_FEATURE_NAMES,
                "direction": direction,
            },
        },
        model_path,
    )

    config = {
        "feature_dim": int(x_np.shape[1]),
        "feature_names": SUCCESS_FEATURE_NAMES,
        "num_samples": n,
        "num_train": int(train_idx.size),
        "num_val": int(val_idx.size),
        "num_success": int(y_np.sum()),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "direction": direction,
        "model_path": model_path,
    }
    write_json(os.path.join(out_dir, "config.json"), config)

    return {
        "history": history,
        "config": config,
        "model_path": model_path,
        "final_train_acc": history["train_acc"][-1] if history["train_acc"] else float("nan"),
        "final_val_acc": history["val_acc"][-1] if history["val_acc"] else float("nan"),
    }
