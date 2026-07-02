"""Tests for the read-only servo scan + the servo characterization tool.

NONE of these touch real hardware: every backend is a MOCK that records (or
refuses) writes. They pass with only numpy + pyyaml installed (no cv2/torch/
matplotlib): the metric math is pure numpy and plotting is optional.
"""

from __future__ import annotations

import os

import pytest

from terafold.robot.characterization import (backlash, deadband_from_sweep,
                                             readback_stats, repeatability,
                                             run_characterization, settle_metrics)
from terafold.robot.servo_scan import parse_id_range, run_servo_scan

ROBOT = "physical_7dof_waveshare"


class MockBackend:
    """A confirmed read-safe backend that records every write (matches the
    verified-backend interface used across the suite)."""

    def __init__(self, responders=(1, 2, 5, 6), pos=1500, confirmed=True):
        self.responders = set(int(i) for i in responders)
        self.pos = int(pos)
        self.protocol_confirmed = bool(confirmed)
        self.writes = []
        self.closed = False

    def probe_protocol(self, ids=None):
        return {"confirmed": self.protocol_confirmed}

    def ping(self, ids):
        return {int(i): (int(i) in self.responders) for i in ids}

    def read_pos_speed(self, servo_id):
        if int(servo_id) in self.responders:
            return (self.pos + int(servo_id), 0)
        return None

    def read_position(self, servo_id):
        ps = self.read_pos_speed(servo_id)
        return None if ps is None else ps[0]

    def write_position(self, servo_id, target, speed=None, acc=None):
        self.writes.append((int(servo_id), int(target)))
        return True

    def close(self):
        self.closed = True


# --------------------------- parse_id_range ---------------------------


def test_parse_id_range_dash():
    assert parse_id_range("1-30") == list(range(1, 31))


def test_parse_id_range_commas():
    assert parse_id_range("1,2,5,6") == [1, 2, 5, 6]


def test_parse_id_range_mixed_and_dedup():
    assert parse_id_range("1-5,10") == [1, 2, 3, 4, 5, 10]
    # de-dup + sort + whitespace + reversed range tolerated.
    assert parse_id_range(" 3, 1-3 , 6-4 ") == [1, 2, 3, 4, 5, 6]
    assert parse_id_range("") == []


# --------------------------- run_servo_scan ---------------------------


def test_servo_scan_finds_ids_and_computes_missing(tmp_path):
    mb = MockBackend(responders=(1, 2, 5, 6), pos=1500)
    out_dir = str(tmp_path / "scan_run")
    res = run_servo_scan(ROBOT, port="/dev/null", id_min=1, id_max=30,
                         backend=mb, out_dir=out_dir, now="testscan")

    # Found exactly the responders; missing = nominal arm IDs (3,4,7) that didn't answer.
    assert res["found_ids"] == [1, 2, 5, 6]
    assert res["missing_ids"] == [3, 4, 7]

    # Read-only: ZERO servo writes were recorded on the mock.
    assert mb.writes == []
    assert res["commands_sent"] == 0
    assert res["read_only"] is True

    # Per-ID detail carries the readback for responders.
    assert res["per_id"]["1"]["ping"] is True
    assert res["per_id"]["1"]["pos"] == 1501
    assert res["per_id"]["3"]["ping"] is False
    assert res["per_id"]["3"]["pos"] is None

    # Artifacts written.
    assert os.path.exists(res["scan_json"]) and os.path.exists(res["summary_md"])
    assert res["scan_json"] == os.path.join(out_dir, "scan.json")
    summary = open(res["summary_md"]).read()
    assert "Servo Bus Scan" in summary
    assert "3, 4, 7" in summary  # missing IDs listed for physical debugging


def test_servo_scan_missing_when_subset_responds(tmp_path):
    # Only servo 1 answers -> 2,3,4,5,6,7 are all expected-but-missing.
    mb = MockBackend(responders=(1,), pos=2000)
    res = run_servo_scan(ROBOT, port="/dev/null", id_min=1, id_max=10,
                         backend=mb, out_dir=str(tmp_path / "s2"), now="t")
    assert res["found_ids"] == [1]
    assert res["missing_ids"] == [2, 3, 4, 5, 6, 7]
    assert mb.writes == []


# --------------------------- metric functions ---------------------------


def test_readback_stats():
    s = readback_stats([1500, 1501, 1499, 1500, 1500])
    assert s["mean"] == pytest.approx(1500.0)
    assert s["min"] == 1499.0 and s["max"] == 1501.0
    assert s["std"] == pytest.approx(0.6324555, abs=1e-5)
    assert s["n"] == 5
    assert readback_stats([])["n"] == 0


def test_deadband_from_sweep():
    deltas = [2, 4, 6, 8, 10]
    moved = [False, False, True, True, True]
    assert deadband_from_sweep(deltas, moved) == 6.0
    # nothing moves -> None
    assert deadband_from_sweep([1, 2], [False, False]) is None


def test_settle_metrics():
    trace = [1500, 1580, 1610, 1601, 1600, 1600]
    m = settle_metrics(trace, target=1600)
    assert m["settle_error"] == pytest.approx(0.0)
    assert m["overshoot"] == pytest.approx(10.0)   # peak 1610 vs target 1600
    assert m["time_to_settle"] == 3                # settled from index 3 onward
    assert settle_metrics([], 1600)["time_to_settle"] == 0


def test_settle_metrics_downward():
    # Approaching from above with undershoot past the target.
    trace = [2000, 1900, 1890, 1898, 1900]
    m = settle_metrics(trace, target=1900)
    assert m["overshoot"] == pytest.approx(10.0)   # min 1890 is 10 past 1900
    assert m["settle_error"] == pytest.approx(0.0)


def test_backlash():
    assert backlash(1495, 1505) == pytest.approx(10.0)
    # array inputs averaged.
    assert backlash([1498, 1500], [1504, 1506]) == pytest.approx(6.0)


def test_repeatability():
    assert repeatability([1500, 1502, 1498, 1500]) == pytest.approx(2 ** 0.5)
    assert repeatability([]) == 0.0


# --------------------------- run_characterization ---------------------------


def test_characterization_dry_run_moves_nothing():
    mb = MockBackend()
    res = run_characterization(ROBOT, ids=[1, 2], out="unused_dir",
                               dry_run=True, backend=mb, log=lambda _m: None)
    assert res["status"] == "dry_run"
    assert res["moved"] is False
    assert res["commands_sent"] == 0
    assert mb.writes == []                      # no servo writes in dry-run
    assert "plan" in res and res["plan"]["plan_lines"]


def test_characterization_refuses_without_flags(tmp_path):
    mb = MockBackend()
    res = run_characterization(ROBOT, ids=[1], out=str(tmp_path / "o"),
                               dry_run=False, enable_motion=False, acknowledge=False,
                               backend=mb, log=lambda _m: None, sleep_fn=lambda _s: None)
    assert res["status"] == "refused"
    assert "enable-motion" in res["refusal"]
    assert mb.writes == []


def test_characterization_refuses_when_unconfirmed(tmp_path):
    mb = MockBackend(confirmed=False)
    res = run_characterization(ROBOT, ids=[1], out=str(tmp_path / "o"),
                               dry_run=False, enable_motion=True, acknowledge=True,
                               backend=mb, log=lambda _m: None, sleep_fn=lambda _s: None)
    assert res["status"] == "refused"
    assert "protocol not confirmed" in res["refusal"].lower()
    assert mb.writes == []


def test_characterization_real_with_mock_writes_metrics(tmp_path):
    mb = MockBackend(responders=(1, 2), pos=1500)
    out = str(tmp_path / "char")
    res = run_characterization(ROBOT, ids=[1, 2], out=out,
                               dry_run=False, enable_motion=True, acknowledge=True,
                               backend=mb, log=lambda _m: None, sleep_fn=lambda _s: None)
    assert res["status"] == "characterized"
    assert res["moved"] is True
    # one tiny step + return home per servo -> 2 writes each.
    assert res["commands_sent"] == 4
    assert len(mb.writes) == 4
    # step is conservative: within the safe range and bounded by max_sweep (80).
    for sid, target in mb.writes:
        assert 400 <= target <= 3700
    assert mb.closed is False  # caller-supplied backend is not closed

    assert os.path.exists(res["metrics_json"])
    assert os.path.exists(res["summary_md"])
    assert res["metrics_json"] == os.path.join(out, "metrics.json")
    # per-servo CSV traces written.
    assert all(os.path.exists(p) for p in res["traces"])
    assert "1" in res["metrics"] and "readback" in res["metrics"]["1"]
