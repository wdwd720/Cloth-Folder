"""Training loop for the keypoint heatmap model (requires torch).

Trains the U-Net from :mod:`terafold.vision.keypoint_model` on a synthetic
dataset produced by :mod:`terafold.vision.synthetic_cloth`, then saves
``<out>/model.pt`` (weights + config) and ``<out>/config.json``.

torch is imported lazily inside :func:`train_keypoints`; the module itself
imports with numpy alone, and a clear, actionable error is raised if torch is
absent when training is actually attempted.
"""

from __future__ import annotations

import json
import os
from typing import Optional

from terafold.vision.keypoint_dataset import (
    HEATMAP_SIZE,
    IMAGE_SIZE,
    build_torch_dataset,
)
from terafold.vision.keypoint_model import NUM_KEYPOINTS, build_keypoint_model, heatmap_loss

__all__ = ["train_keypoints"]


def _require_torch():
    try:
        import torch  # noqa: F401

        return torch
    except Exception as exc:  # pragma: no cover - exercised only without torch
        raise ImportError(
            "train_keypoints requires torch. Install it with "
            "`pip install -e '.[torch]'` (or `pip install torch torchvision`)."
        ) from exc


def train_keypoints(
    data_dir: str,
    out_dir: str,
    epochs: int = 30,
    batch_size: int = 16,
    lr: float = 1e-3,
    device: Optional[str] = None,
    val_frac: float = 0.1,
    augment: bool = True,
    limit: Optional[int] = None,
) -> dict:
    """Train the keypoint model and save ``<out>/model.pt`` + ``config.json``.

    Returns a history dict with per-epoch train/val losses.
    """
    torch = _require_torch()
    from torch.utils.data import DataLoader, Subset, random_split

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    full = build_torch_dataset(data_dir, augment=augment, image_size=IMAGE_SIZE)
    if limit is not None:
        full = Subset(full, list(range(min(int(limit), len(full)))))
    n = len(full)
    if n == 0:
        raise ValueError(f"empty dataset at {data_dir!r}; nothing to train on")
    n_val = max(1, int(round(val_frac * n))) if n > 1 else 0
    n_train = n - n_val
    if n_val > 0:
        gen = torch.Generator().manual_seed(0)
        train_ds, val_ds = random_split(full, [n_train, n_val], generator=gen)
    else:
        train_ds, val_ds = full, None

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = (
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        if val_ds is not None
        else None
    )

    model = build_keypoint_model(NUM_KEYPOINTS).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": []}
    os.makedirs(out_dir, exist_ok=True)

    for epoch in range(int(epochs)):
        model.train()
        running = 0.0
        count = 0
        for batch in train_loader:
            img = batch["image"].to(dev)
            target = batch["heatmaps"].to(dev)
            opt.zero_grad()
            pred = model(img)
            loss = heatmap_loss(pred, target)
            loss.backward()
            opt.step()
            running += float(loss.item()) * img.shape[0]
            count += img.shape[0]
        train_loss = running / max(count, 1)
        history["train_loss"].append(train_loss)

        val_loss = float("nan")
        if val_loader is not None:
            model.eval()
            vrun = 0.0
            vcount = 0
            with torch.no_grad():
                for batch in val_loader:
                    img = batch["image"].to(dev)
                    target = batch["heatmaps"].to(dev)
                    vloss = heatmap_loss(model(img), target)
                    vrun += float(vloss.item()) * img.shape[0]
                    vcount += img.shape[0]
            val_loss = vrun / max(vcount, 1)
        history["val_loss"].append(val_loss)
        print(
            f"[train_keypoints] epoch {epoch + 1}/{epochs} "
            f"train_loss={train_loss:.5f} val_loss={val_loss:.5f}"
        )

    config = {
        "num_keypoints": NUM_KEYPOINTS,
        "image_size": IMAGE_SIZE,
        "heatmap_size": HEATMAP_SIZE,
        "epochs": int(epochs),
        "lr": float(lr),
        "batch_size": int(batch_size),
        "augment": bool(augment),
    }
    model_path = os.path.join(out_dir, "model.pt")
    torch.save(
        {"state_dict": model.state_dict(), "config": config}, model_path
    )
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    history["model_path"] = model_path
    history["config"] = config
    history["num_samples"] = n
    return history
