"""YAM config loading + yam-check-assets behavior (no hardware, no heavy deps)."""

from __future__ import annotations

import pytest

from terafold.yam.assets import check_assets
from terafold.yam.config import REFERENCE_CONFIG, load_yam_config
from terafold.yam.safety import ShadowModeViolation


def _by_name(res):
    return {c["name"]: c for c in res["checks"]}


def test_reference_config_loads():
    cfg = load_yam_config()
    assert cfg.robot == "dual_yam_standard"
    assert cfg.arms == ["left_yam", "right_yam"]
    assert cfg.camera_order == ["top", "left", "right"]
    assert cfg.norm_tag == "yam_dual_molmoact2"
    assert cfg.action_mode == "continuous"
    assert cfg.control_mode == "absolute_joint_pose"
    assert cfg.model_name == "allenai/MolmoAct2-BimanualYAM"
    assert cfg.autonomous_execution_enabled is False
    assert cfg.safe_speed_scale == pytest.approx(0.10)
    assert cfg.max_joint_delta_rad is not None
    assert cfg.max_gripper_delta is not None
    assert set(cfg.workspace_bounds) >= {"x", "y", "z"}


def test_reference_config_shapes_are_consistent():
    cfg = load_yam_config(REFERENCE_CONFIG)
    assert cfg.action_dim == len(cfg.arms) * (cfg.joints_per_arm + cfg.grippers_per_arm)
    layout = cfg.state_layout()
    assert len(layout) == cfg.state_dim == cfg.action_dim
    assert layout[0] == "left_yam.joint0"
    assert layout[cfg.per_arm_dim - 1] == "left_yam.gripper"


def test_config_with_autonomy_enabled_is_refused(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("robot: dual_yam_standard\nautonomous_execution_enabled: true\n")
    with pytest.raises(ShadowModeViolation):
        load_yam_config(p)


def test_missing_config_raises_filenotfound(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_yam_config(tmp_path / "nope.yaml")


def test_malformed_yaml_raises_valueerror_per_contract(tmp_path):
    p = tmp_path / "broken.yaml"
    p.write_text("robot: [unclosed\n  nope: {")
    with pytest.raises(ValueError):
        load_yam_config(p)


def test_check_assets_on_reference_config():
    res = check_assets(str(REFERENCE_CONFIG))
    # Off-rig, with nothing cloned/downloaded, this must be WARN — never FAIL.
    assert res["status"] in ("PASS", "WARN")
    checks = _by_name(res)
    assert checks["config"]["status"] == "PASS"
    assert checks["shadow_mode"]["status"] == "PASS"
    assert res["camera_order"] == ["top", "left", "right"]
    assert res["norm_tag"] == "yam_dual_molmoact2"
    assert res["model"] == "allenai/MolmoAct2-BimanualYAM"
    # Unconfigured assets come with exact next-step commands.
    assert "git clone" in checks["i2rt_repo"].get("fix", "")
    assert "huggingface" in checks["molmoact2_checkpoint"].get("fix", "")


def test_check_assets_handles_missing_i2rt_repo_gracefully(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text("paths:\n  i2rt_repo: /nonexistent/i2rt\n")
    res = check_assets(str(p))  # must not raise
    c = _by_name(res)["i2rt_repo"]
    assert c["status"] == "FAIL"  # configured path that does not exist
    assert "git clone" in c["fix"]
    assert res["status"] == "FAIL"


def test_check_assets_finds_existing_paths(tmp_path):
    repo = tmp_path / "i2rt"
    repo.mkdir()
    urdf = tmp_path / "yam.urdf"
    urdf.write_text("<robot/>")
    p = tmp_path / "cfg.yaml"
    p.write_text(f"paths:\n  i2rt_repo: {repo}\n  urdf: {urdf}\n")
    checks = _by_name(check_assets(str(p)))
    assert checks["i2rt_repo"]["status"] == "PASS"
    assert checks["urdf"]["status"] == "PASS"


def test_check_assets_tolerates_shorthand_camera_entries(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text("cameras:\n  top: /dev/video0\n  left: {path: /dev/video1}\n")
    res = check_assets(str(p))  # must not raise on the string-valued entry
    c = _by_name(res)["camera_paths"]
    assert c["status"] == "WARN" and "right" in c["detail"] and "top" not in c["detail"]


def test_check_assets_missing_config_is_fail_not_crash():
    res = check_assets("/nonexistent/definitely/missing.yaml")
    assert res["status"] == "FAIL"
    assert _by_name(res)["config"]["status"] == "FAIL"


def test_check_assets_rejects_autonomy_enabled_config(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text("autonomous_execution_enabled: true\n")
    res = check_assets(str(p))
    assert res["status"] == "FAIL"
    assert "shadow" in _by_name(res)["config"]["detail"].lower()
