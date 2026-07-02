"""real-image-ghost-fold integration with the YOLO towel-pose backend (dry-run).

The detector is MOCKED (no ultralytics, no hardware). Asserts the integration:
detect → plan → critic → gated decision, with NO hardware writes and contact LOCKED.
"""

from __future__ import annotations

import numpy as np

import terafold.towel.yolo_runtime as yolo_runtime
from terafold.robot.real_image_ghost import run_real_image_ghost_fold

ROBOT = "physical_7dof_waveshare"


def _good_detection(**over):
    corners = {"tl": [30, 20], "tr": [130, 20], "br": [130, 100], "bl": [30, 100]}
    res = {
        "status": "ok", "bbox": [30, 20, 130, 100], "corners": corners,
        "corner_confidence": {k: 0.9 for k in corners},
        "avg_keypoint_confidence": 0.9, "normalized_corners": corners,
        "predicted_state": "flat_unfolded", "visible": {k: True for k in corners},
        "image": "x.png", "weights": "best.pt",
    }
    res.update(over)
    return res


def _img(tmp_path):
    from terafold.vision.imageio import imwrite

    p = str(tmp_path / "towel.png")
    img = np.full((120, 160, 3), 30, np.uint8)
    img[20:100, 30:130] = 210
    imwrite(p, img)
    return p


def test_ghost_yolo_dryrun_integrates_detector_and_locks_contact(tmp_path, monkeypatch):
    monkeypatch.setattr(yolo_runtime, "infer_towel_pose", lambda *a, **k: _good_detection())
    res = run_real_image_ghost_fold(
        _img(tmp_path), robot=ROBOT, perception_backend="yolo_towel_pose",
        weights="best.pt", log=lambda _m: None,
    )
    assert res["perception_backend"] == "yolo_towel_pose"
    # Detector output flowed into the plan.
    assert res["corners"]["tl"] == [30, 20]
    assert res["confidence"] == 0.9 and res["towel_state"] == "flat_unfolded"
    assert "grasp" in res and "place" in res and res["fold_direction"]
    assert "plan_score" in res and isinstance(res["risk_reasons"], list)
    # A clean towel passes the dry-run gate.
    assert res["status"] == "dry_run" and res["allowed_for_dry_run"] is True
    # NO hardware writes, contact LOCKED — always.
    assert res["motor_commands_sent"] == 0
    assert res["allowed_for_contact"] is False
    assert res["contact_fold_allowed"] is False


def test_ghost_yolo_refuses_low_confidence(tmp_path, monkeypatch):
    monkeypatch.setattr(yolo_runtime, "infer_towel_pose",
                        lambda *a, **k: _good_detection(avg_keypoint_confidence=0.30))
    res = run_real_image_ghost_fold(
        _img(tmp_path), robot=ROBOT, perception_backend="yolo_towel_pose",
        weights="best.pt", min_perception_confidence=0.75, log=lambda _m: None,
    )
    assert res["status"] == "refused"
    assert res["allowed_for_dry_run"] is False
    assert "low_confidence" in res["risk_reasons"] or "confidence" in res["refusal"]
    assert res["motor_commands_sent"] == 0 and res["allowed_for_contact"] is False


def test_ghost_yolo_refuses_bad_view(tmp_path, monkeypatch):
    monkeypatch.setattr(yolo_runtime, "infer_towel_pose",
                        lambda *a, **k: _good_detection(predicted_state="bad_view"))
    res = run_real_image_ghost_fold(
        _img(tmp_path), robot=ROBOT, perception_backend="yolo_towel_pose",
        weights="best.pt", log=lambda _m: None,
    )
    assert res["status"] == "refused"
    assert "bad_view" in res["hard_fail_reasons"]
    assert res["motor_commands_sent"] == 0 and res["allowed_for_contact"] is False


def test_ghost_yolo_unavailable_is_graceful(tmp_path, monkeypatch):
    monkeypatch.setattr(
        yolo_runtime, "infer_towel_pose",
        lambda *a, **k: {"status": "unavailable", "install": "pip install -e '.[yolo]'",
                         "message": "ultralytics not installed"})
    res = run_real_image_ghost_fold(
        _img(tmp_path), robot=ROBOT, perception_backend="yolo_towel_pose",
        weights="best.pt", log=lambda _m: None,
    )
    assert res["status"] == "refused"
    assert res["allowed_for_dry_run"] is False
    assert res["motor_commands_sent"] == 0 and res["allowed_for_contact"] is False


def test_ghost_yolo_real_motion_refused_dryrun_only(tmp_path, monkeypatch):
    # Even with both motion flags, the YOLO backend never executes / never moves.
    monkeypatch.setattr(yolo_runtime, "infer_towel_pose", lambda *a, **k: _good_detection())
    res = run_real_image_ghost_fold(
        _img(tmp_path), robot=ROBOT, perception_backend="yolo_towel_pose", weights="best.pt",
        enable_motion=True, acknowledge=True, log=lambda _m: None,
    )
    assert res["status"] == "refused"
    assert "dry-run" in res["refusal"].lower()
    assert res["motor_commands_sent"] == 0 and res["allowed_for_contact"] is False


def test_classical_backend_unchanged(tmp_path):
    # The default classical path still works and is unaffected by the new options.
    res = run_real_image_ghost_fold(_img(tmp_path), robot=ROBOT, port="/dev/null",
                                    log=lambda _m: None)
    assert res["status"] == "dry_run"
    assert res["contact_fold_allowed"] is False and res["motor_commands_sent"] == 0
