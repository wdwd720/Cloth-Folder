"""Tests for terafold.towel.yolo_runtime (no ultralytics required).

The training-command builder is pure string work and always runs. The inference and
evaluation paths import ultralytics lazily; here we force that import to fail (so the
test is deterministic whether or not ultralytics happens to be installed) and assert
the graceful ``status="unavailable"`` fallback.
"""

from __future__ import annotations

import builtins
import os

import numpy as np

from terafold.towel import dataset as ds
from terafold.towel import yolo_runtime as yr
from terafold.vision.imageio import imwrite


def _block_ultralytics(monkeypatch):
    """Make ``import ultralytics`` raise ImportError, mimicking it being absent."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "ultralytics" or name.startswith("ultralytics."):
            raise ImportError("ultralytics blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def _png(path, w=80, h=60, val=200):
    img = np.full((h, w, 3), 30, np.uint8)
    img[10:h - 10, 10:w - 10] = val
    imwrite(path, img)
    return path


# --------------------------- training command ---------------------------


def test_training_command_strings():
    res = yr.print_towel_training_command("data/yolo_towel_pose_v0")
    assert "yolo pose train" in res["command"]
    assert "towel_pose.yaml" in res["command"]
    assert "model=yolo26n-pose.pt" in res["command"]
    # fallback swaps in the always-available yolo11n weights.
    assert "model=yolo11n-pose.pt" in res["fallback_command"]
    assert "model=yolo26n-pose.pt" not in res["fallback_command"]
    # yaml path is <data>/towel_pose.yaml and a RunPod note is present.
    assert res["yaml"].endswith("towel_pose.yaml")
    assert "data/yolo_towel_pose_v0" in res["yaml"]
    assert isinstance(res["runpod_note"], str) and res["runpod_note"]


def test_training_command_honours_params():
    res = yr.print_towel_training_command(
        "d", model="custom.pt", epochs=7, imgsz=512, batch=4,
        project="runs/x", name="run_a",
    )
    cmd = res["command"]
    assert "model=custom.pt" in cmd and "epochs=7" in cmd and "imgsz=512" in cmd
    assert "batch=4" in cmd and "project=runs/x" in cmd and "name=run_a" in cmd


# --------------------------- inference (unavailable path) ---------------------------


def test_infer_unavailable_without_ultralytics(monkeypatch):
    _block_ultralytics(monkeypatch)
    res = yr.infer_towel_pose("some_image.png", "x.pt")
    assert res["status"] == "unavailable"
    assert "pip install" in res["install"]
    assert "ultralytics" in res["message"]


# --------------------------- evaluation (unavailable path) ---------------------------


def test_evaluate_unavailable_without_ultralytics(monkeypatch, tmp_path):
    _block_ultralytics(monkeypatch)

    # A tiny real dataset (need not be labelled; we stop before any inference runs).
    raw = tmp_path / "raw"
    raw.mkdir()
    _png(str(raw / "a.png"))
    out = str(tmp_path / "towel_real")
    ds.create_towel_real_dataset(str(raw), out)

    res = yr.evaluate_towel_pose(out, "x.pt", str(tmp_path / "eval"))
    assert res["status"] == "unavailable"
    assert "pip install" in res["install"]
    # No report should be written on the unavailable path.
    assert not os.path.exists(os.path.join(str(tmp_path / "eval"), "eval.md"))
