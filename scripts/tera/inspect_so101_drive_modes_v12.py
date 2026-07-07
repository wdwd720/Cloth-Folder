"""V12 SO-101 drive-mode inspector (feasibility probe + drive audit).

Two jobs:
1. FEASIBILITY: confirm the real SO-101 arm scene builds on Modal (leisaac +
   so101_follower.usd + IsaacLab articulation), which is the prerequisite for
   porting the V11 physics-driven contact fix to the SO-101 fingertip.
2. DRIVE AUDIT: report how the SO-101 is currently driven, so V12 can fix the
   right thing.

Static code facts (verified by reading the repo):
- The SO-101 ARM is driven by articulation position targets
  (`set_joint_position_target` + `write_data_to_sim` + `sim.step`) -> physics
  resolved. That part is fine.
- The V8 fingertip PROXIES (which actually contact the cloth) are moved by
  `set_prim_translation` -> a USD-transform TELEPORT with ~zero swept velocity.
  This is the SAME defect V11 identified for the primitive sphere; it is why the
  SO-101 proxy touched the cloth but did not move it.

This probe builds the scene, dumps arm/jaw info, steps physics, and records the
audit. Diagnostic only; writes JSON to the V12 dir.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from isaaclab.app import AppLauncher

import test_clean_towel_primitive_contact_v8 as v8  # noqa: F401  (ensures deps import)
from clean_towel_v4_common import jaw_positions_from_scene
from so101_clean_towel_scene_utils_v1 import (
    CLEAN_TOWEL_USD_PATH,
    command_run_string,
    create_standalone_so101_clean_towel_scene,
    force_headless_no_cameras,
    size_metrics,
    standalone_robot_scene_info,
    write_json,
)

V12_DIR = Path("/workspace/leisaac/tera_checkpoints/clean_towel_v12")
OUT_JSON = V12_DIR / "so101_drive_modes_v12.json"


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="V12 SO-101 drive-mode inspector.")
    parser.add_argument("--usd_path", type=Path, default=CLEAN_TOWEL_USD_PATH)
    parser.add_argument("--out_dir", type=Path, default=V12_DIR)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    force_headless_no_cameras(args)
    return args


def main() -> None:
    args = parse_args()
    V12_DIR.mkdir(parents=True, exist_ok=True)
    app = AppLauncher(args).app  # noqa: F841

    summary: dict[str, Any] = {
        "status": "V12_SO101_DRIVE_MODES_STARTED",
        "command_run": command_run_string(),
        "diagnostic_only": True,
        "so101_scene_built": False,
        "drive_audit": {
            "arm_drive": "articulation_position_targets (set_joint_position_target + write_data_to_sim + sim.step) -> physics-resolved",
            "v8_proxy_drive": "set_prim_translation (USD transform teleport) -> ~zero swept velocity (defect, same as V11)",
            "v12_fix": "drive fingertip proxies via RigidPrim.set_world_poses (physics kinematic target) so they carry real velocity",
        },
    }
    try:
        scene = create_standalone_so101_clean_towel_scene(args)
        summary["so101_scene_built"] = True
        summary["scene_info"] = standalone_robot_scene_info(scene)

        # step physics a bit and confirm the arm state updates
        for _ in range(10):
            scene.sim.step(render=False)
            dt = scene.sim.get_physics_dt()
            scene.left_arm.update(dt)
            scene.right_arm.update(dt)

        jaws, jaw_info = jaw_positions_from_scene(scene)
        summary["jaw_positions"] = [float(x) for x in jaws]
        summary["jaw_info"] = jaw_info
        summary["left_body_names"] = list(getattr(scene.left_arm.data, "body_names", []) or [])
        summary["left_joint_names"] = list(getattr(scene.left_arm.data, "joint_names", []) or [])

        towel_pts = scene.clean_towel.points_numpy().astype(np.float32)
        summary["towel_num_particles"] = int(towel_pts.shape[0])
        summary["towel_start_metrics"] = size_metrics(towel_pts)

        summary["status"] = "V12_SO101_DRIVE_MODES_DONE"
        write_json(OUT_JSON, summary, print_payload=True)
    except Exception as exc:  # noqa: BLE001
        import traceback

        summary["status"] = "V12_SO101_DRIVE_MODES_FAILED"
        summary["exception_type"] = exc.__class__.__name__
        summary["exception"] = repr(exc)
        summary["traceback_tail"] = traceback.format_exc()[-6000:]
        write_json(OUT_JSON, summary, print_payload=True)
    finally:
        sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == "__main__":
    main()
