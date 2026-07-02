"""Tests for the numpy-only evaluation metric modules + the run report.

Everything here runs with numpy/pyyaml only — no cv2/torch/matplotlib — so the
core test suite never depends on optional deps.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from terafold.data.episode_schema import write_jsonl
from terafold.eval import calibration_metrics as cal
from terafold.eval import control_metrics as ctl
from terafold.eval import fold_metrics as fold
from terafold.eval import perception_metrics as perc
from terafold.eval.report import eval_run, report_markdown


# --------------------------------------------------------------------------
# Perception metrics
# --------------------------------------------------------------------------


def test_corner_error_zero_when_equal():
    pts = [[0, 0], [10, 0], [10, 10], [0, 10]]
    err = perc.corner_error(pts, pts)
    assert err["mean"] == 0.0
    assert err["max"] == 0.0
    assert err["per_corner"] == [0.0, 0.0, 0.0, 0.0]


def test_corner_error_known_shift():
    gt = [[0, 0], [10, 0], [10, 10], [0, 10]]
    pred = [[3, 4], [10, 0], [10, 10], [0, 10]]  # first corner off by 5 (3-4-5)
    err = perc.corner_error(pred, gt)
    assert err["max"] == pytest.approx(5.0)
    assert err["mean"] == pytest.approx(5.0 / 4)
    assert err["per_corner"][0] == pytest.approx(5.0)


def test_point_and_grasp_place_error():
    assert perc.point_error([1, 1], [1, 1]) == 0.0
    assert perc.point_error([0, 0], [3, 4]) == pytest.approx(5.0)
    assert perc.grasp_error([0, 0], [3, 4]) == pytest.approx(5.0)
    assert perc.place_error([2, 0], [2, 0]) == 0.0


def test_fold_line_error_zero_and_known():
    line = [[0, 0], [10, 0]]  # horizontal
    same = perc.fold_line_error(line, line)
    assert same["angle_deg"] == pytest.approx(0.0)
    assert same["offset"] == pytest.approx(0.0)

    shifted = perc.fold_line_error([[0, 5], [10, 5]], line)  # parallel, offset 5
    assert shifted["angle_deg"] == pytest.approx(0.0)
    assert shifted["offset"] == pytest.approx(5.0)

    rotated = perc.fold_line_error([[0, 0], [0, 10]], line)  # vertical vs horizontal
    assert rotated["angle_deg"] == pytest.approx(90.0)


def test_confidence_pass():
    assert perc.confidence_pass(0.9, 0.7) is True
    assert perc.confidence_pass([0.8, 0.75, 0.71], 0.7) is True
    assert perc.confidence_pass([0.8, 0.5], 0.7) is False
    assert perc.confidence_pass([], 0.7) is False


# --------------------------------------------------------------------------
# Control metrics
# --------------------------------------------------------------------------


def _settling_trace(target=1500, start=1000):
    """A clean trace that overshoots a little then settles on target."""
    return [
        (0.0, start, 400),
        (0.1, 1300, 250),
        (0.2, 1560, 80),   # small overshoot past 1500
        (0.3, 1505, 10),
        (0.4, 1500, 0),    # at rest, on target
        (0.5, 1500, 0),
    ]


def test_settle_error_and_overshoot():
    trace = _settling_trace()
    assert ctl.settle_error(trace, 1500) == pytest.approx(0.0, abs=1e-6)
    assert ctl.overshoot(trace, 1500) == pytest.approx(60.0)  # peaked at 1560
    assert ctl.failure_to_reach(trace, 1500, tol=20) is False


def test_time_to_settle():
    trace = _settling_trace()
    # First time it stays within 20 of 1500 for the rest is at t=0.3 (1505).
    assert ctl.time_to_settle(trace, 1500, tol=20) == pytest.approx(0.3)


def test_failure_to_reach_and_settle_error_nonzero():
    bad = [(0.0, 1000, 300), (0.1, 1200, 100), (0.2, 1250, 0)]  # stalls at 1250
    assert ctl.settle_error(bad, 1500) == pytest.approx(250.0)
    assert ctl.failure_to_reach(bad, 1500, tol=20) is True
    assert math.isinf(ctl.time_to_settle(bad, 1500, tol=20))


def test_readback_noise_at_rest():
    trace = [(0.0, 1000, 300), (0.1, 1499, 0), (0.2, 1501, 0), (0.3, 1500, 0)]
    noise = ctl.readback_noise(trace)
    assert noise == pytest.approx(np.std([1499, 1501, 1500]))


def test_deadband_estimate():
    deltas = [1, 2, 3, 8, 12]
    moved = [False, False, False, True, True]  # largest non-moving delta is 3
    assert ctl.deadband_estimate(deltas, moved) == pytest.approx(3.0)
    assert ctl.deadband_estimate([5, 6], [True, True]) == 0.0


def test_backlash_scalars_and_traces():
    assert ctl.backlash(1510, 1490) == pytest.approx(20.0)
    pos_trace = [(0.0, 1400, 100), (0.1, 1512, 0)]
    neg_trace = [(0.0, 1600, 100), (0.1, 1488, 0)]
    assert ctl.backlash(pos_trace, neg_trace) == pytest.approx(24.0)


# --------------------------------------------------------------------------
# Calibration metrics
# --------------------------------------------------------------------------


def test_homography_rms_identity_is_zero():
    H = np.eye(3)
    pts = [[0, 0], [1, 0], [1, 1], [0, 1], [2, 3]]
    assert cal.homography_rms(H, pts, pts) == pytest.approx(0.0, abs=1e-9)


def test_homography_rms_translation():
    H = np.array([[1, 0, 2.0], [0, 1, -1.0], [0, 0, 1]])
    img = [[0, 0], [1, 0], [1, 1], [0, 1]]
    table = [[2, -1], [3, -1], [3, 0], [2, 0]]  # exactly translated
    assert cal.homography_rms(H, img, table) == pytest.approx(0.0, abs=1e-9)
    # Wrong target -> nonzero RMS.
    assert cal.homography_rms(np.eye(3), img, table) > 0.5


def test_fit_and_held_out_error_small():
    # A known affine/translation map; fit should recover it and generalize.
    img = [[0, 0], [4, 0], [4, 4], [0, 4], [2, 1], [1, 3]]
    table = [[10 + x, 20 + y] for x, y in img]
    H = cal.fit_homography(img, table)
    assert cal.homography_rms(H, img, table) == pytest.approx(0.0, abs=1e-6)
    assert cal.held_out_error(img, table) == pytest.approx(0.0, abs=1e-6)


def test_robot_touch_rms_and_thresholds():
    pred = [[0.10, 0.20], [0.30, 0.40]]
    gt = [[0.10, 0.20], [0.30, 0.40]]
    assert cal.robot_touch_rms(pred, gt) == pytest.approx(0.0)
    rms = cal.robot_touch_rms([[0.0, 0.0]], [[0.03, 0.04]])  # 5 cm
    assert rms == pytest.approx(0.05)
    assert cal.valid_for_hover(rms, threshold=0.08) is True
    assert cal.valid_for_contact(rms, threshold=0.02) is False
    assert cal.valid_for_hover(float("nan"), threshold=1.0) is False


# --------------------------------------------------------------------------
# Fold metrics
# --------------------------------------------------------------------------


def test_crossed_crease_true_and_false():
    fold_line = [[5, 0], [5, 10]]  # vertical crease at x=5
    before = [[0, 0], [0, 10]]     # both left of crease
    after_crossed = [[8, 0], [8, 10]]   # both right -> crossed
    after_same = [[1, 0], [1, 10]]      # still left -> not crossed
    assert fold.crossed_crease(before, after_crossed, fold_line) is True
    assert fold.crossed_crease(before, after_same, fold_line) is False


def test_final_edge_error_and_success():
    after = [[0, 0], [10, 0], [10, 5], [0, 5]]
    target = [[0, 0], [10, 0], [10, 5], [0, 5]]
    assert fold.final_edge_error(after, target) == pytest.approx(0.0)
    assert fold.fold_success(0.0, threshold=0.02) is True

    after2 = [[0, 0.1], [10, 0], [10, 5], [0, 5]]  # one corner off 0.1
    err = fold.final_edge_error(after2, target)
    assert err == pytest.approx(0.1 / 4)
    assert fold.fold_success(err, threshold=0.01) is False


def test_target_side_correct():
    fold_line = [[5, 0], [5, 10]]
    right_corners = [[8, 1], [9, 9]]
    left_corners = [[1, 1], [2, 9]]
    # signed_side sign depends on the line direction; just require the two
    # sides to be opposite and the gate to agree with signed_side.
    side_r = float(fold.signed_side([[8, 5]], fold_line)[0])
    assert fold.target_side_correct(right_corners, fold_line, target_side=side_r) is True
    assert fold.target_side_correct(left_corners, fold_line, target_side=side_r) is False


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def test_eval_run_writes_markdown_with_sections_and_warning(tmp_path):
    """A synthetic motion log with a stalled servo yields a warning line."""
    log_path = tmp_path / "motion.jsonl"
    out_path = tmp_path / "report.md"
    records = [
        {"event": "image_ghost_request",
         "data": {"robot": "physical_7dof_waveshare", "dry_run": False,
                  "enable_motion": True, "acknowledge": True,
                  "perception": "classical_fallback"}},
        {"event": "moved", "data": {"id": 6, "target": 1500}},
        {"event": "readback", "data": {"servo_id": 6, "t": 0.0, "pos": 1000, "speed": 300}},
        {"event": "readback", "data": {"servo_id": 6, "t": 0.1, "pos": 1200, "speed": 120}},
        {"event": "readback", "data": {"servo_id": 6, "t": 0.2, "pos": 1250, "speed": 0}},
        {"event": "refused", "data": {"reason": "protocol_unconfirmed"}},
    ]
    write_jsonl(str(log_path), records)

    metrics = eval_run(str(log_path), out=str(out_path))

    assert out_path.exists()
    md = out_path.read_text()
    for header in ("# TeraFold Evaluation Report", "## Control Metrics",
                   "## Perception", "## Calibration", "## Fold Quality", "## Safety"):
        assert header in md
    # The servo stalled at 1250 vs target 1500 -> failure_to_reach -> warning line.
    assert "WARNING" in md
    assert any("FAILED TO REACH" in w for w in metrics["warnings"])
    assert metrics["control"]["any_failure_to_reach"] is True
    # Perception mode surfaced; safety captured the refusal + real-motion request.
    assert metrics["perception"]["available"] is True
    assert "protocol_unconfirmed" in metrics["safety"]["refusals"]
    assert metrics["safety"]["real_motion_requested"] is True


def test_eval_run_handles_in_memory_list_and_missing_families():
    records = [{"event": "moved", "data": {"id": 6, "target": 1500}}]  # no readbacks
    metrics = eval_run(records)
    assert metrics["control"]["available"] is False
    assert metrics["perception"]["available"] is False
    assert metrics["calibration"]["available"] is False
    assert metrics["fold"]["available"] is False
    # Report still renders every section without crashing.
    md = report_markdown(metrics)
    assert "not available" in md
    assert md.count("##") >= 5


def test_eval_run_clean_trace_no_warning():
    records = [
        {"event": "moved", "data": {"servo_id": 2, "target": 2000}},
        {"event": "readback", "data": {"servo_id": 2, "t": 0.0, "pos": 1800, "speed": 200}},
        {"event": "readback", "data": {"servo_id": 2, "t": 0.1, "pos": 1995, "speed": 5}},
        {"event": "readback", "data": {"servo_id": 2, "t": 0.2, "pos": 2000, "speed": 0}},
    ]
    metrics = eval_run(records)
    assert metrics["control"]["available"] is True
    assert metrics["control"]["any_failure_to_reach"] is False
    assert metrics["warnings"] == [] or all("FAILED" not in w for w in metrics["warnings"])
