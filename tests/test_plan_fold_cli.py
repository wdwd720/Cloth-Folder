"""Regression: `plan-fold --checkpoint PATH` wires in the learned predictor.

These tests prove the CLI does NOT silently fall back to the classical predictor
when a valid checkpoint is supplied, and that it reports the perception backend.
They mock torch + the learned predictor so they pass with or without torch
installed (no real training needed).
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest
from typer.testing import CliRunner

from terafold.cli import app
from terafold.physics.cloth_state import ClothKeypoints, FoldLine, FoldState
from terafold.vision import infer_keypoints as ik


class _FakeLearned:
    """Stand-in for the torch-backed KeypointPredictor (no torch needed)."""

    name = ik.LEARNED_KIND
    used = False

    def __init__(self, checkpoint, device=None):
        self.checkpoint = checkpoint

    def predict(self, image, direction="right_to_left"):
        _FakeLearned.used = True
        h, w = np.asarray(image).shape[:2]
        kp = ClothKeypoints(
            [0.20 * w, 0.25 * h], [0.78 * w, 0.22 * h],
            [0.80 * w, 0.80 * h], [0.22 * w, 0.82 * h],
        )
        return FoldState(
            keypoints=kp, fold_line=FoldLine.from_corners(kp, direction),
            frame="image", image_shape=(int(h), int(w)),
        )


@pytest.fixture
def fake_learned(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    monkeypatch.setattr(ik, "KeypointPredictor", _FakeLearned)
    _FakeLearned.used = False


def test_resolve_uses_learned_when_checkpoint_valid(tmp_path, fake_learned):
    ckpt = tmp_path / "model.pt"
    ckpt.write_bytes(b"stub")
    predictor, info = ik.resolve_keypoint_predictor(str(ckpt))
    assert info["kind"] == ik.LEARNED_KIND
    assert info["checkpoint"] == str(ckpt)
    assert not isinstance(predictor, ik.FallbackKeypointPredictor)


def test_resolve_falls_back_when_checkpoint_missing(tmp_path):
    predictor, info = ik.resolve_keypoint_predictor(str(tmp_path / "nope.pt"))
    assert info["kind"] == ik.FALLBACK_KIND
    assert "not found" in info["reason"]
    assert isinstance(predictor, ik.FallbackKeypointPredictor)


def test_plan_fold_cli_uses_learned_and_does_not_fall_back(tmp_path, fake_learned):
    ckpt = tmp_path / "model.pt"
    ckpt.write_bytes(b"stub")
    result = CliRunner().invoke(
        app, ["plan-fold", "--camera", "mock", "--checkpoint", str(ckpt)]
    )
    assert result.exit_code == 0, result.output
    # Reports the learned backend + checkpoint path.
    assert f"perception: {ik.LEARNED_KIND}" in result.output
    assert f"checkpoint: {ckpt}" in result.output
    # Did NOT fall back, and the spurious "checkpoint is null" warning is gone.
    assert ik.FALLBACK_KIND not in result.output
    assert "keypoint_checkpoint is null" not in result.output
    # The learned predictor was actually invoked by the planner.
    assert _FakeLearned.used is True


def test_plan_fold_cli_warns_on_invalid_checkpoint(tmp_path):
    missing = tmp_path / "does_not_exist.pt"
    result = CliRunner().invoke(
        app, ["plan-fold", "--camera", "mock", "--checkpoint", str(missing)]
    )
    assert result.exit_code == 0, result.output
    assert f"perception: {ik.FALLBACK_KIND}" in result.output
    assert "WARNING" in result.output  # must warn when it falls back
