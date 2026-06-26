"""Claude Vision labeler — mocked JSON validation (no SDK / network needed)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from terafold.vision.claude_labeler import ClaudeKeypointLabeler, ClaudeLabelError

IMG = np.zeros((200, 300, 3), np.uint8)
VALID = {
    "top_left": [40, 30], "top_right": [260, 35],
    "bottom_left": [45, 170], "bottom_right": [255, 165],
    "grasp_point": [255, 100], "place_point": [45, 100],
    "fold_line": [[150, 30], [150, 170]],
    "confidence": 0.88, "notes": "clear towel",
}


def _labeler(raw):
    return ClaudeKeypointLabeler(responder=lambda im: raw)


def test_valid_label_parses():
    label = _labeler(json.dumps(VALID)).label(IMG)
    assert label.confidence == 0.88
    kp = label.to_keypoints()
    assert kp.corners.shape == (4, 2)
    assert kp.grasp is not None and kp.place is not None


def test_valid_with_code_fence_and_prose():
    raw = "Here are the labels:\n```json\n" + json.dumps(VALID) + "\n```\n"
    label = _labeler(raw).label(IMG)
    assert label.confidence == 0.88


def test_malformed_json_rejected():
    with pytest.raises(ClaudeLabelError):
        _labeler("this is not json").label(IMG)


def test_low_confidence_rejected():
    bad = {**VALID, "confidence": 0.10}
    with pytest.raises(ClaudeLabelError):
        _labeler(json.dumps(bad)).label(IMG)


def test_out_of_bounds_rejected():
    bad = {**VALID, "grasp_point": [9999, 9999]}
    with pytest.raises(ClaudeLabelError):
        _labeler(json.dumps(bad)).label(IMG)


def test_impossible_self_crossing_quad_rejected():
    bad = {**VALID, "top_right": [255, 165], "bottom_right": [260, 35]}  # swap -> self-crossing
    with pytest.raises(ClaudeLabelError):
        _labeler(json.dumps(bad)).label(IMG)


def test_degenerate_identical_corners_rejected():
    bad = {**VALID, "top_right": [40, 30]}  # identical to top_left
    with pytest.raises(ClaudeLabelError):
        _labeler(json.dumps(bad)).label(IMG)


def test_missing_key_rejected():
    bad = {k: v for k, v in VALID.items() if k != "grasp_point"}
    with pytest.raises(ClaudeLabelError):
        _labeler(json.dumps(bad)).label(IMG)
