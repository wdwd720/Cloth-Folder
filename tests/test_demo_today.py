"""The demo-today pipeline: perception -> plan -> safe dry-run -> record."""

from __future__ import annotations

import os

from typer.testing import CliRunner

from terafold.cli import app
from terafold.demo import run_demo_today
from terafold.robot.safety import STOP_FILE

TASK = os.path.abspath("configs/task_fold_towel_half.yaml")


def test_mock_markers_dry_run(tmp_path):
    res = run_demo_today(
        task_path=TASK, camera="mock", robot="mock", mode="markers", dry_run=True,
        out=str(tmp_path / "ep"), overlay_out=str(tmp_path / "ov.png"),
    )
    assert res["status"] == "ok"
    assert res["dry_run"] is True and res["real_motion"] is False
    assert res["perception"] == "markers"
    assert res["num_waypoints"] >= 12
    assert res["executed_actions"] > 0 and res["stopped"] is False
    # episode + overlay artifacts exist
    assert os.path.isdir(os.path.join(res["episode_dir"], "frames"))
    assert os.path.exists(os.path.join(res["episode_dir"], "plan.json"))
    assert os.path.exists(os.path.join(res["episode_dir"], "result.json"))
    assert os.path.exists(res["overlay_path"])


def test_so101_dry_run_without_lerobot(tmp_path):
    # SO-101 dry-run must work even though lerobot is not installed.
    res = run_demo_today(
        task_path=TASK, camera="mock", robot="so101", mode="markers", dry_run=True,
        out=str(tmp_path / "ep"),
    )
    assert res["status"] == "ok"
    assert res["executed_actions"] > 0 and res["stopped"] is False


def test_motion_gate_requires_both_flags(tmp_path):
    # Only one motion flag -> stays dry-run (safe by default).
    res = run_demo_today(
        task_path=TASK, camera="mock", robot="so101", mode="markers",
        dry_run=False, enable_motion=True, acknowledge=False, out=str(tmp_path / "ep"),
    )
    assert res["real_motion"] is False and res["dry_run"] is True


def test_real_motion_refused_without_requirements(tmp_path):
    # Both flags + real robot, but no calibration / verified adapter / dry-run done.
    res = run_demo_today(
        task_path=TASK, camera="mock", robot="so101", mode="markers",
        dry_run=False, enable_motion=True, acknowledge=True, out=str(tmp_path / "ep"),
    )
    assert res["requested_motion"] is True
    assert res["real_motion"] is False  # refused
    assert res["status"] == "real_motion_refused"
    assert res["executed_actions"] == 0  # nothing moved
    assert any("calibrat" in m for m in res["missing"])


def test_camera_fallback_to_mock(tmp_path):
    # No cv2 / no webcam -> falls back to the marker-rendering mock camera.
    res = run_demo_today(
        task_path=TASK, camera="opencv", camera_index=0, robot="mock", mode="markers",
        dry_run=True, out=str(tmp_path / "ep"),
    )
    assert res["status"] == "ok"
    assert res["camera"].startswith("mock")


def test_stop_file_aborts(tmp_path, monkeypatch):
    # A STOP file in the working dir aborts before any motion.
    monkeypatch.chdir(tmp_path)
    (tmp_path / STOP_FILE).write_text("stop")
    res = run_demo_today(
        task_path=TASK, camera="mock", robot="mock", mode="markers", dry_run=True,
        out=str(tmp_path / "ep"),
    )
    assert res["stopped"] is True
    assert "STOP" in (res["stop_reason"] or "")
    assert res["executed_actions"] == 0


def test_cli_demo_today_dry_run(tmp_path):
    result = CliRunner().invoke(
        app,
        ["demo-today", "--camera", "mock", "--robot", "mock", "--mode", "markers",
         "--task", TASK, "--dry-run", "--out", str(tmp_path / "ep")],
    )
    assert result.exit_code == 0, result.output
    assert "dry-run" in result.output.lower()
    assert "episode" in result.output.lower()
