"""V12 SO-101 fingertip physics-drive test.

Ports the V11 fix to the SO-101 setting. In the REAL SO-101 scene (leisaac arm +
clean towel), a jaw-scale fingertip PROXY collider is swept along the towel edge
(the SO-101 jaw's intended fold path) and driven two ways:
- teleport            : USD `set_prim_translation` (the V8 SO-101-proxy method)
- kinematic_velocity  : physics `RigidPrim.set_world_poses` (the V11 fix)

This reuses the V11 drive + valid/ballistic classification (via a scene wrapper)
and V8's proxy spawn, so the ONLY thing that changes between trials is how the
SO-101 fingertip proxy is driven. It directly shows whether the V11 mechanism
transfers to the SO-101 fingertip: teleport should fail, physics should move the
cloth.

Note: the SO-101 arm is loaded and present; the fingertip proxy is the contact
element (as established in V8, because the raw jaw geometry under-contacts). The
V6 reach search output is unavailable on Modal, so the proxy follows the
geometry-derived fold path rather than a live-searched arm joint trajectory; this
is stated honestly in the JSON. No particle teleporting counts as success.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from isaaclab.app import AppLauncher

import clean_towel_contact_v11_common as c11
import test_clean_towel_primitive_contact_v8 as v8
from clean_towel_v4_common import jaw_positions_from_scene
from inspect_so101_gripper_collision_v7 import recommended_towel_translation
from so101_clean_towel_scene_utils_v1 import (
    CLEAN_TOWEL_USD_PATH,
    command_run_string,
    create_standalone_so101_clean_towel_scene,
    force_headless_no_cameras,
    initialize_clean_towel_handle_for_sim,
    size_metrics,
    write_json,
)

V12_DIR = Path("/workspace/leisaac/tera_checkpoints/clean_towel_v12")
OUT_JSON = V12_DIR / "so101_fingertip_physics_drive_v12.json"
PROXY_PATH = "/World/V12ProxyFingertip"


@dataclass
class _ProxyScene:
    """Adapts the SO-101 scene so V11's run_drive_trial can drive the proxy."""
    sim: Any
    clean_towel: Any
    primitive_root_path: str
    device: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V12 SO-101 fingertip physics-drive test.")
    parser.add_argument("--usd_path", type=Path, default=CLEAN_TOWEL_USD_PATH)
    parser.add_argument("--out_dir", type=Path, default=V12_DIR)
    parser.add_argument("--edge", type=str, default="right")
    parser.add_argument("--edge_band", type=float, default=0.035)
    parser.add_argument("--settle_initial_steps", type=int, default=20)
    # proxy / path geometry (jaw-scale fingertip)
    parser.add_argument("--primitive_radius", type=float, default=0.02)
    parser.add_argument("--primitive_height", type=float, default=0.012)
    parser.add_argument("--penetration_depth", type=float, default=0.05)
    parser.add_argument("--drag_distance", type=float, default=0.28)
    parser.add_argument("--drag_height_delta", type=float, default=0.0)
    parser.add_argument("--outside_margin", type=float, default=0.05)
    parser.add_argument("--proxy_contact_offset", type=float, default=0.012)
    parser.add_argument("--proxy_rest_offset", type=float, default=0.0)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    force_headless_no_cameras(args)
    return args


def main() -> None:
    args = parse_args()
    V12_DIR.mkdir(parents=True, exist_ok=True)
    app = AppLauncher(args).app  # noqa: F841

    summary: dict[str, Any] = {
        "status": "V12_SO101_FINGERTIP_PHYSICS_DRIVE_STARTED",
        "command_run": command_run_string(),
        "diagnostic_only": True,
        "contact_element": "so101_jaw_scale_fingertip_proxy_in_real_so101_scene",
        "v6_reach_search_available_on_modal": False,
        "trials": [],
    }
    try:
        from isaacsim.core.utils.stage import update_stage

        scene = create_standalone_so101_clean_towel_scene(
            args, clean_usd_path=args.usd_path, towel_translation=recommended_towel_translation()
        )
        scene.step_zero(int(args.settle_initial_steps))

        # spawn a single jaw-scale kinematic fingertip proxy, then re-init sim
        proxy_info = v8.spawn_kinematic_sphere(
            PROXY_PATH, radius=float(args.primitive_radius),
            contact_offset=float(args.proxy_contact_offset), rest_offset=float(args.proxy_rest_offset),
            static_friction=1.6, dynamic_friction=1.2, translation=(1.25, 0.0, 0.30),
        )
        update_stage()
        scene.sim.reset()
        scene.left_arm.update(scene.sim.get_physics_dt())
        scene.right_arm.update(scene.sim.get_physics_dt())
        scene.clean_towel = initialize_clean_towel_handle_for_sim(scene.sim, scene.clean_towel.root_path)
        scene.step_zero(int(args.settle_initial_steps))

        summary["proxy_info"] = proxy_info
        jaws, jaw_info = jaw_positions_from_scene(scene)
        summary["so101_jaw_positions"] = [float(x) for x in jaws]
        summary["so101_jaw_info"] = jaw_info

        start_points = scene.clean_towel.points_numpy().astype(np.float32)
        summary["start_metrics"] = size_metrics(start_points)

        # apply cloth contact params (reuse V8 tuning) via the primitive path
        proxy_scene = _ProxyScene(sim=scene.sim, clean_towel=scene.clean_towel,
                                  primitive_root_path=PROXY_PATH, device=scene.device)
        params = v8.default_primitive_params(args)
        try:
            summary["contact_param_application"] = v8.apply_contact_params(proxy_scene, params)
        except Exception as exc:  # noqa: BLE001
            summary["contact_param_application_error"] = repr(exc)

        edge = str(args.edge)
        radius = float(args.primitive_radius)
        eb = float(args.edge_band)
        configs = [
            ("so101_proxy_teleport", "teleport", 140),
            ("so101_proxy_kinematic_velocity_slow", "kinematic_velocity", 140),
            ("so101_proxy_kinematic_velocity_fast", "kinematic_velocity", 40),
        ]
        for name, mode, nsteps in configs:
            centers = c11.default_path_centers(start_points, args, num_steps=nsteps, edge=edge)
            trial = c11.run_drive_trial(proxy_scene, start_points, centers, edge, eb, radius, drive_mode=mode)
            trial["trial_name"] = name
            summary["trials"].append(trial)

        summary["any_valid_controlled_contact"] = bool(
            any(t.get("valid_controlled_contact") for t in summary["trials"])
        )
        summary["status"] = "V12_SO101_FINGERTIP_PHYSICS_DRIVE_DONE"
        write_json(OUT_JSON, summary, print_payload=True)
    except Exception as exc:  # noqa: BLE001
        import traceback

        summary["status"] = "V12_SO101_FINGERTIP_PHYSICS_DRIVE_FAILED"
        summary["exception_type"] = exc.__class__.__name__
        summary["exception"] = repr(exc)
        summary["traceback_tail"] = traceback.format_exc()[-6000:]
        write_json(OUT_JSON, summary, print_payload=True)
    finally:
        sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == "__main__":
    main()
