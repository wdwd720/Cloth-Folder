"""V11 kinematic-velocity contact test.

Compares, on the SAME clean-towel scene (V8 kinematic sphere):
- baseline_teleport        : V8 drive (USD `set_prim_translation`, ~zero velocity)
- kinematic_velocity_slow  : drive via RigidPrim.set_world_poses (real swept vel)
- kinematic_velocity_fast  : same, fewer steps => higher per-step velocity

Success = actual_touch (<0.01 m) AND edge_displacement > 0.02 m, achieved by a
valid rigid drive (never by writing particle positions). Writes JSON to the V11
dir for the summarizer. Diagnostic only; no policy/training/folding claim.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from isaaclab.app import AppLauncher

import test_clean_towel_primitive_contact_v8 as v8
import clean_towel_contact_v11_common as c11
from so101_clean_towel_scene_utils_v1 import command_run_string, size_metrics, write_json

V11_DIR = Path("/workspace/leisaac/tera_checkpoints/clean_towel_v11")
OUT_JSON = V11_DIR / "kinematic_velocity_v11.json"


def main() -> None:
    args = v8.parse_args()
    args.out_dir = V11_DIR
    V11_DIR.mkdir(parents=True, exist_ok=True)
    app = AppLauncher(args).app  # noqa: F841

    summary: dict[str, Any] = {
        "status": "V11_KINEMATIC_VELOCITY_STARTED",
        "command_run": command_run_string(),
        "diagnostic_only": True,
        "trials": [],
    }
    try:
        scene, scene_info = v8.create_towel_primitive_scene(
            args,
            primitive_radius=float(args.primitive_radius),
            primitive_contact_offset=float(args.primitive_contact_offset),
            primitive_rest_offset=float(args.primitive_rest_offset),
            static_friction=float(args.static_friction),
            dynamic_friction=float(args.dynamic_friction),
        )
        summary["scene_info"] = scene_info
        v8.step_sim(scene, int(args.settle_initial_steps))
        start_points = scene.clean_towel.points_numpy().astype(np.float32)
        summary["start_metrics"] = size_metrics(start_points)
        params = v8.default_primitive_params(args)
        summary["contact_param_application"] = v8.apply_contact_params(scene, params)

        edge = str(getattr(args, "edge", "right"))
        radius = float(args.primitive_radius)
        eb = float(args.edge_band)

        configs = [
            ("baseline_teleport", "teleport", 140),
            ("kinematic_velocity_slow", "kinematic_velocity", 140),
            ("kinematic_velocity_fast", "kinematic_velocity", 40),
        ]
        for name, mode, nsteps in configs:
            centers = c11.default_path_centers(start_points, args, num_steps=nsteps, edge=edge)
            trial = c11.run_drive_trial(scene, start_points, centers, edge, eb, radius, drive_mode=mode)
            trial["trial_name"] = name
            summary["trials"].append(trial)

        summary["any_contact_transfer_success"] = bool(
            any(t.get("contact_transfer_success") for t in summary["trials"])
        )
        summary["best_edge_displacement_m"] = max((t["edge_displacement_m"] for t in summary["trials"]), default=0.0)
        summary["status"] = "V11_KINEMATIC_VELOCITY_DONE"
        write_json(OUT_JSON, summary, print_payload=True)
    except Exception as exc:  # noqa: BLE001
        import traceback
        summary["status"] = "V11_KINEMATIC_VELOCITY_FAILED"
        summary["exception_type"] = exc.__class__.__name__
        summary["exception"] = repr(exc)
        summary["traceback_tail"] = traceback.format_exc()[-4000:]
        write_json(OUT_JSON, summary, print_payload=True)
    finally:
        sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == "__main__":
    main()
