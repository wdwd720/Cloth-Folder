"""Isaac scene generator: package layout, script safety, cloud commands, CLI.

No Isaac Sim, GPU, or hardware required anywhere in here.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest
from typer.testing import CliRunner

from terafold.cli import app
from terafold.yam.isaac_scene import (
    ISAAC_IMAGE,
    LIVESTREAM_PORTS,
    generate_isaac_scene,
    isaac_cloud_commands,
)

runner = CliRunner()


@pytest.fixture()
def scene(tmp_path):
    out = tmp_path / "yam_dual_towel_scene"
    res = generate_isaac_scene(str(out))
    assert res["status"] == "ok"
    return out


def test_scene_folder_and_files_exist(scene):
    for name in ("scene_config.json", "build_isaac_scene.py", "README.md"):
        assert (scene / name).is_file(), name


def test_scene_config_contract(scene):
    cfg = json.loads((scene / "scene_config.json").read_text())
    assert cfg["camera_order"] == ["top", "left", "right"]
    assert cfg["arms"] == ["left_yam", "right_yam"]
    assert set(cfg["arm_bases"]) == {"left_yam", "right_yam"}
    assert set(cfg["cameras"]) == {"top", "left", "right"}
    assert cfg["norm_tag"] == "yam_dual_molmoact2"
    assert cfg["simulation_only"] is True
    assert cfg["hardware_commanded"] is False
    assert cfg["table"]["height_m"] > 0
    assert cfg["towel"]["thickness_m"] < 0.05  # thin cloth placeholder


def test_build_script_is_valid_python_with_lazy_simulationapp(scene):
    src = (scene / "build_isaac_scene.py").read_text()
    compile(src, "build_isaac_scene.py", "exec")  # must be syntactically valid
    assert "from isaacsim import SimulationApp" in src
    # lazy: the import statement lives inside build(), not at module top level
    assert src.index("from isaacsim import SimulationApp") > src.index("def build(")
    assert "import isaacsim" not in src.split("def build(")[0]
    assert "cannot import SimulationApp" in src  # graceful explanation exists


def test_build_script_has_no_hardware_imports_or_motion(scene):
    src = (scene / "build_isaac_scene.py").read_text().lower()
    for forbidden in ("import serial", "pyserial", "dynamixel", "write_position",
                      "write_pos_ex", "enable_motion", "i2rt", "/dev/tty"):
        assert forbidden not in src, forbidden
    # module side too
    import inspect

    import terafold.yam.isaac_scene as mod

    mod_src = inspect.getsource(mod).lower()
    for forbidden in ("import serial", "pyserial", "dynamixel", "write_position"):
        assert forbidden not in mod_src, forbidden


def test_build_script_degrades_gracefully_without_isaac(scene):
    """Running the generated script on a laptop must explain, not crash."""
    proc = subprocess.run([sys.executable, str(scene / "build_isaac_scene.py")],
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode == 2, proc.stderr
    assert "Isaac Sim is not installed" in proc.stdout
    assert "yam-print-isaac-cloud-commands" in proc.stdout
    assert "Traceback" not in proc.stderr


def test_generate_scene_rejects_bad_config(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("autonomous_execution_enabled: true\n")
    res = generate_isaac_scene(str(tmp_path / "s"), config_path=str(bad))
    assert res["status"] == "bad_config"


def test_generate_scene_rejects_nonnumeric_and_degenerate_table(tmp_path):
    for table in ("table: {width_m: '1.2 m'}", "table: {depth_m: -0.8}"):
        cfg = tmp_path / "t.yaml"
        cfg.write_text(table + "\n")
        res = generate_isaac_scene(str(tmp_path / "s"), config_path=str(cfg))
        assert res["status"] == "bad_config", table
    assert not (tmp_path / "s").exists()  # no partial output dir left behind


def test_tiny_table_depth_never_mirrors_arm_sides(tmp_path):
    cfg = tmp_path / "t.yaml"
    cfg.write_text("table: {depth_m: 0.08}\n")  # plausible typo for 0.8
    res = generate_isaac_scene(str(tmp_path / "s"), config_path=str(cfg))
    assert res["status"] == "ok"
    scene_cfg = json.loads((tmp_path / "s" / "scene_config.json").read_text())
    left_y = scene_cfg["arm_bases"]["left_yam"][1]
    right_y = scene_cfg["arm_bases"]["right_yam"][1]
    assert left_y < 0 < right_y  # left stays -Y, right stays +Y


def test_wrist_cameras_mirror_and_extra_cameras_get_specs(tmp_path, scene):
    scene_cfg = json.loads((scene / "scene_config.json").read_text())
    left = scene_cfg["cameras"]["left"]
    right = scene_cfg["cameras"]["right"]
    # mirrored across y=0: opposite y positions AND opposite X pitch
    assert left["position"][1] == -right["position"][1]
    assert left["rotation_xyz_deg"][0] == -right["rotation_xyz_deg"][0]

    cfg = tmp_path / "c.yaml"
    cfg.write_text("camera_order: [top, front, left, right]\n")
    res = generate_isaac_scene(str(tmp_path / "s"), config_path=str(cfg))
    assert res["status"] == "ok"
    sc = json.loads((tmp_path / "s" / "scene_config.json").read_text())
    # every camera in camera_order has a spec => build script can never KeyError
    assert set(sc["camera_order"]) <= set(sc["cameras"])


def test_urdf_branch_places_import_and_wait_enables_livestream(scene):
    src = (scene / "build_isaac_scene.py").read_text()
    # URDF import must land on the CURRENT stage and be placed via the
    # returned prim path (a dest_path would silently drop the robot).
    assert "dest_path=" not in src  # the kwarg itself must never be passed
    assert "GetPrimAtPath(str(prim_path))" in src
    # --wait must try to enable the livestream extension and warn when it can't.
    assert "enable_extension" in src
    assert "omni.kit.livestream" in src
    assert "NOT active" in src


def test_scene_readme_uses_absolute_path(scene):
    readme = (scene / "README.md").read_text()
    assert f"./python.sh {scene}/build_isaac_scene.py" in readme  # scene is absolute


def test_cloud_commands_content():
    text = "\n".join(isaac_cloud_commands())
    assert ISAAC_IMAGE == "nvcr.io/nvidia/isaac-sim:6.0.1"
    assert f"docker pull {ISAAC_IMAGE}" in text
    assert "--gpus all" in text
    assert 'ACCEPT_EULA=Y' in text
    assert "--network=host" in text
    assert "/isaac-sim/.cache" in text  # 6.0 cache-mount layout, not the 4.x one
    assert "/root/.cache" not in text
    assert "compatibility_check.sh --no-window" in text  # headless-safe invocation
    assert "runheadless" in text
    assert "build_isaac_scene.py" in text
    assert str(LIVESTREAM_PORTS["tcp"]) in text and str(LIVESTREAM_PORTS["udp"]) in text


def test_cloud_commands_absolute_scene_dir():
    text = "\n".join(isaac_cloud_commands(scene_dir="/abs/path/scene"))
    assert "-v /abs/path/scene:/workspace/yam_scene:rw" in text
    assert "$(pwd)//" not in text
    # relative default still anchors to the caller's cwd
    rel = "\n".join(isaac_cloud_commands())
    assert "-v $(pwd)/sim/yam_dual_towel_scene:/workspace/yam_scene:rw" in rel


def test_cli_build_isaac_scene(tmp_path):
    out = tmp_path / "scene"
    result = runner.invoke(app, ["yam-build-isaac-scene", "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert (out / "build_isaac_scene.py").is_file()
    assert "visual digital twin only" in result.output


def test_cli_print_isaac_cloud_commands():
    result = runner.invoke(app, ["yam-print-isaac-cloud-commands"])
    assert result.exit_code == 0, result.output
    assert "nvcr.io/nvidia/isaac-sim:6.0.1" in result.output
    assert "49100" in result.output and "47998" in result.output


def test_cli_help_lists_isaac_commands():
    result = runner.invoke(app, ["--help"], env={"COLUMNS": "200"})
    assert result.exit_code == 0
    assert "yam-build-isaac-scene" in result.output
    assert "yam-print-isaac-cloud-commands" in result.output
