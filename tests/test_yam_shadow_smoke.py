"""MolmoAct2 smoke test (mocked), shadow policy, Rerun fallback, CLI wiring.

None of these tests need torch / transformers / rerun / hardware.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from terafold.cli import app
from terafold.yam import molmoact2_smoke as ms
from terafold.yam import shadow as shadow_mod
from terafold.yam.config import REFERENCE_CONFIG
from terafold.yam.dataset import create_dummy_episode
from terafold.yam.rerun_logger import RERUN_INSTALL_HINT, YamRerunLogger, have_rerun
from terafold.yam.shadow import run_shadow_policy

runner = CliRunner()

YAM_COMMANDS = (
    "yam-check-assets",
    "yam-molmoact2-smoke-test",
    "yam-create-dummy-episode",
    "yam-shadow-policy",
)


class FakeAdapter:
    """Stand-in for a loaded MolmoAct2 model (records calls, returns fixed action)."""

    def __init__(self, action_dim=14):
        self.action_dim = action_dim
        self.calls = []

    def predict(self, images, robot_state=None, task="", max_new_tokens=0):
        self.calls.append({"cameras": sorted(images), "task": task})
        return {"action": [0.1] * self.action_dim, "raw_text": "fake"}

    def describe(self):
        return {"kind": "fake", "model": "fake/Model", "dtype": "fake"}


# -- smoke test ---------------------------------------------------------------


def test_smoke_test_with_mock_adapter(tmp_path):
    out = tmp_path / "smoke" / "actions.json"
    res = ms.run_smoke_test(dtype="bfloat16", out=str(out), mock=True, log=lambda *_: None)
    assert res["status"] == "ok"
    assert res["hardware_commanded"] is False

    actions = json.loads(out.read_text())
    assert actions["hardware_commanded"] is False
    assert len(actions["action"]) == 14  # 2 arms x (6 joints + 1 gripper)

    meta = json.loads((out.parent / "metadata.json").read_text())
    for key in ("model", "dtype", "camera_order", "norm_tag", "action_shape",
                "timestamp", "hardware_commanded"):
        assert key in meta, key
    assert meta["hardware_commanded"] is False
    assert meta["camera_order"] == ["top", "left", "right"]
    assert meta["norm_tag"] == "yam_dual_molmoact2"
    assert meta["action_shape"] == [14]


def test_smoke_test_with_mocked_model_loader(monkeypatch, tmp_path):
    """The mocked-MolmoAct2 path: load_model is swapped, plumbing runs for real."""
    fake = FakeAdapter()
    monkeypatch.setattr(ms, "load_model", lambda *a, **k: fake)
    res = ms.run_smoke_test(out=str(tmp_path / "actions.json"), log=lambda *_: None)
    assert res["status"] == "ok"
    assert fake.calls and fake.calls[0]["cameras"] == ["left", "right", "top"]
    assert fake.calls[0]["task"] == "fold the towel in half neatly"


def test_smoke_test_missing_deps_fails_gracefully(monkeypatch, tmp_path):
    def _boom(*a, **k):
        raise ImportError("\n".join(ms.install_instructions()))

    monkeypatch.setattr(ms, "load_model", _boom)
    res = ms.run_smoke_test(out=str(tmp_path / "actions.json"), log=lambda *_: None)
    assert res["status"] == "deps_missing"
    assert any("pip install" in line for line in res["instructions"])


def test_smoke_test_rejects_bad_dtype(tmp_path):
    res = ms.run_smoke_test(dtype="float8", out=str(tmp_path / "a.json"))
    assert res["status"] == "bad_dtype"


def test_tampered_reference_config_is_refused_not_masked(monkeypatch, tmp_path):
    """A reference config flipped to autonomy=true must be refused on the
    default-config path too, not silently replaced by safe defaults."""
    import terafold.yam.config as yam_config

    bad = tmp_path / "tampered.yaml"
    bad.write_text("autonomous_execution_enabled: true\n")
    monkeypatch.setattr(yam_config, "REFERENCE_CONFIG", bad)

    res = ms.run_smoke_test(out=str(tmp_path / "a.json"), mock=True, log=lambda *_: None)
    assert res["status"] == "bad_config"
    assert "shadow" in res["message"].lower()

    ep = tmp_path / "ep"
    # (episode creation must not depend on the poisoned reference config either)
    from terafold.yam.config import YamDualConfig

    create_dummy_episode(ep, frames=1, config=YamDualConfig())
    res = run_shadow_policy(str(ep), out=str(tmp_path / "s"), mock=True,
                            use_rerun=False, log=lambda *_: None)
    assert res["status"] == "bad_config"


def test_smoke_test_uses_provided_image(tmp_path):
    import numpy as np

    from terafold.vision.imageio import imwrite

    img_path = tmp_path / "top.png"
    imwrite(str(img_path), np.full((32, 32, 3), 200, dtype=np.uint8))
    out = tmp_path / "actions.json"
    res = ms.run_smoke_test(out=str(out), mock=True, top_image=str(img_path),
                            log=lambda *_: None)
    assert res["status"] == "ok"
    doc = json.loads(out.read_text())
    assert doc["image_sources"]["top"] == str(img_path)
    assert doc["image_sources"]["left"] == "dummy"


# -- shadow policy ------------------------------------------------------------


@pytest.fixture()
def episode(tmp_path):
    out = tmp_path / "episode_000001"
    create_dummy_episode(out, frames=3, seed=0)
    return out


def test_shadow_policy_with_injected_adapter(episode, tmp_path):
    fake = FakeAdapter()
    out = tmp_path / "shadow"
    res = run_shadow_policy(str(episode), out=str(out), adapter=fake, use_rerun=False,
                            log=lambda *_: None)
    assert res["status"] == "ok"
    assert res["hardware_commanded"] is False
    assert res["frames_predicted"] == res["frames_compared"] == 3
    # the adapter saw all three cameras every frame
    assert all(c["cameras"] == ["left", "right", "top"] for c in fake.calls)

    preds = json.loads((out / "predicted_actions.json").read_text())
    assert preds["hardware_commanded"] is False
    assert len(preds["predictions"]) == 3

    comp = json.loads((out / "comparison.json").read_text())
    assert comp["frames_compared"] == 3
    assert all(c["comparable"] for c in comp["comparisons"])
    assert all(c["l2"] >= 0 for c in comp["comparisons"])

    meta = json.loads((out / "metadata.json").read_text())
    assert meta["hardware_commanded"] is False
    assert "no hardware commanded" in meta["safety_status"]


def test_shadow_policy_mock_first_frame_only(episode, tmp_path):
    res = run_shadow_policy(str(episode), out=str(tmp_path / "s"), mock=True,
                            first_frame_only=True, use_rerun=False, log=lambda *_: None)
    assert res["status"] == "ok"
    assert res["frames_predicted"] == 1


def test_shadow_policy_zero_frame_episode_predicts_nothing(tmp_path):
    """--first-frame-only on an empty episode must not fabricate a prediction."""
    ep = tmp_path / "empty"
    create_dummy_episode(ep, frames=0)
    res = run_shadow_policy(str(ep), out=str(tmp_path / "s"), mock=True,
                            first_frame_only=True, use_rerun=False, log=lambda *_: None)
    assert res["status"] == "ok"
    assert res["frames_predicted"] == 0


def test_shadow_policy_missing_deps_fails_gracefully(monkeypatch, episode, tmp_path):
    def _boom(*a, **k):
        raise ImportError("\n".join(ms.install_instructions()))

    monkeypatch.setattr(shadow_mod, "load_model", _boom)
    res = run_shadow_policy(str(episode), out=str(tmp_path / "s"), use_rerun=False,
                            log=lambda *_: None)
    assert res["status"] == "deps_missing"
    assert "--mock" in res["hint"]


def test_shadow_policy_bad_episode(tmp_path):
    res = run_shadow_policy(str(tmp_path / "missing"), out=str(tmp_path / "s"),
                            mock=True, log=lambda *_: None)
    assert res["status"] == "bad_episode"
    assert "yam-create-dummy-episode" in res["hint"]


def test_shadow_policy_corrupt_frame_fails_gracefully(episode, tmp_path):
    (episode / "cameras" / "top" / "frame_000001.png").write_bytes(b"not a png")
    res = run_shadow_policy(str(episode), out=str(tmp_path / "s"), mock=True,
                            use_rerun=False, log=lambda *_: None)
    assert res["status"] == "bad_image"
    assert "frame 1" in res["message"]
    assert res["frames_completed"] == 1


def test_smoke_test_malformed_config_is_bad_config(tmp_path):
    bad = tmp_path / "broken.yaml"
    bad.write_text("robot: [unclosed\n  nope: {")
    res = ms.run_smoke_test(out=str(tmp_path / "a.json"), mock=True,
                            config_path=str(bad), log=lambda *_: None)
    assert res["status"] == "bad_config"
    assert "Malformed YAML" in res["message"]


# -- rerun fallback -----------------------------------------------------------


def test_rerun_logger_fallback_never_crashes(tmp_path):
    logger = YamRerunLogger(out_dir=str(tmp_path), enabled=True)
    if not have_rerun():
        assert logger.available is False
        assert "rerun-sdk" in logger.note()
        assert logger.note() == RERUN_INSTALL_HINT
    # With or without rerun installed, logging a frame must not raise.
    logger.log_frame(0, images={"top": None}, robot_state={"state_vector": [0.0, 1.0]},
                     predicted_action=[0.1, 0.2], actual_action=[0.1, 0.3],
                     task="fold", safety_status="shadow")


def test_rerun_logger_disabled_is_silent(tmp_path):
    logger = YamRerunLogger(out_dir=str(tmp_path), enabled=False)
    assert logger.available is False
    logger.log_frame(0)


def test_rerun_safe_swallows_api_drift(tmp_path, capsys):
    """_safe takes thunks, so even attribute errors inside them cannot escape."""
    logger = YamRerunLogger(out_dir=str(tmp_path), enabled=False)
    logger._safe(lambda: (_ for _ in ()).throw(AttributeError("api drift")))
    logger._safe(lambda: 1 / 0)  # only warns once
    out = capsys.readouterr().out
    assert out.count("logging degraded") == 1


# -- CLI wiring ---------------------------------------------------------------


def test_yam_commands_are_registered():
    names = {c.name for c in app.registered_commands}
    for cmd in YAM_COMMANDS:
        assert cmd in names, cmd


def test_cli_help_lists_yam_commands():
    result = runner.invoke(app, ["--help"], env={"COLUMNS": "200"})
    assert result.exit_code == 0
    for cmd in YAM_COMMANDS:
        assert cmd in result.output, cmd


def test_cli_yam_check_assets_runs_without_hardware():
    result = runner.invoke(app, ["yam-check-assets", "--config", str(REFERENCE_CONFIG)])
    assert result.exit_code == 0, result.output
    assert "SUMMARY" in result.output
    assert "yam_dual_molmoact2" in result.output


@pytest.mark.parametrize("content", [
    None,                                   # missing file
    "autonomous_execution_enabled: true\n",  # shadow-mode violation
    "robot: [unclosed\n  nope: {",           # malformed YAML
])
def test_cli_dummy_episode_bad_config_no_traceback(tmp_path, content):
    cfg = tmp_path / "cfg.yaml"
    if content is not None:
        cfg.write_text(content)
    result = runner.invoke(app, ["yam-create-dummy-episode", "--out", str(tmp_path / "ep"),
                                 "--config", str(cfg)])
    assert result.exit_code == 1
    # a handled failure exits via typer.Exit (SystemExit) — never a raw exception
    assert result.exception is None or isinstance(result.exception, SystemExit), (
        result.exception)


def test_cli_dummy_episode_then_mock_shadow(tmp_path):
    ep = tmp_path / "episode_000001"
    result = runner.invoke(app, ["yam-create-dummy-episode", "--out", str(ep),
                                 "--frames", "2"])
    assert result.exit_code == 0, result.output

    result = runner.invoke(app, ["yam-shadow-policy", "--episode", str(ep), "--mock",
                                 "--no-rerun", "--out", str(tmp_path / "shadow")])
    assert result.exit_code == 0, result.output
    assert "hardware_commanded: false" in result.output
