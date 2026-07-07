"""V11 dynamic-rigid contact test.

Builds a scene with a DYNAMIC (non-kinematic) rigid sphere, places it at the
towel edge, and launches it with a real linear velocity (via the RigidPrim view)
into the cloth. Unlike a teleported kinematic body, a dynamic body carries
genuine momentum, so this is the strongest test of whether rigid contact can
transfer motion to the PBD particles.

Trials:
- dynamic_press_drag : moderate inward+downward velocity, gravity on
- dynamic_fast       : higher inward velocity

Success = actual_touch (<0.01 m) AND edge_displacement > 0.02 m via valid rigid
contact (never by writing particle positions). Writes JSON to the V11 dir.
Diagnostic only.
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
OUT_JSON = V11_DIR / "dynamic_rigid_v11.json"


def _velocity(centers: list[np.ndarray], speed: float, downward: float) -> np.ndarray:
    start = np.asarray(centers[0], np.float32)
    end = np.asarray(centers[-1], np.float32)
    horiz = end - start
    horiz[2] = 0.0
    n = float(np.linalg.norm(horiz))
    direction = horiz / n if n > 1e-6 else np.asarray([-1.0, 0.0, 0.0], np.float32)
    vel = direction * float(speed)
    vel[2] = -abs(float(downward))
    return vel.astype(np.float32)


def main() -> None:
    args = v8.parse_args()
    args.out_dir = V11_DIR
    V11_DIR.mkdir(parents=True, exist_ok=True)
    app = AppLauncher(args).app  # noqa: F841

    summary: dict[str, Any] = {
        "status": "V11_DYNAMIC_RIGID_STARTED",
        "command_run": command_run_string(),
        "diagnostic_only": True,
        "trials": [],
    }
    try:
        scene, scene_info = c11.build_scene(
            args, kinematic=False, disable_gravity=False, sphere_mass=0.25,
            sphere_radius=float(args.primitive_radius),
            sphere_contact_offset=float(args.primitive_contact_offset),
            sphere_rest_offset=float(args.primitive_rest_offset),
        )
        summary["scene_info"] = scene_info
        v8.step_sim(scene, int(args.settle_initial_steps))
        start_points = scene.clean_towel.points_numpy().astype(np.float32)
        summary["start_metrics"] = size_metrics(start_points)

        edge = str(getattr(args, "edge", "right"))
        radius = float(args.primitive_radius)
        eb = float(args.edge_band)

        configs = [
            ("dynamic_press_drag", 0.5, 0.4, 100),
            ("dynamic_fast", 1.2, 0.6, 100),
        ]
        for name, speed, downward, nsteps in configs:
            centers = c11.default_path_centers(start_points, args, num_steps=nsteps, edge=edge)
            vel = _velocity(centers, speed, downward)
            trial = c11.run_drive_trial(
                scene, start_points, centers, edge, eb, radius,
                drive_mode="dynamic_velocity", initial_velocity=vel,
            )
            trial["trial_name"] = name
            trial["initial_velocity"] = [float(x) for x in vel]
            summary["trials"].append(trial)

        summary["any_contact_transfer_success"] = bool(
            any(t.get("contact_transfer_success") for t in summary["trials"])
        )
        summary["best_edge_displacement_m"] = max((t["edge_displacement_m"] for t in summary["trials"]), default=0.0)
        summary["status"] = "V11_DYNAMIC_RIGID_DONE"
        write_json(OUT_JSON, summary, print_payload=True)
    except Exception as exc:  # noqa: BLE001
        import traceback
        summary["status"] = "V11_DYNAMIC_RIGID_FAILED"
        summary["exception_type"] = exc.__class__.__name__
        summary["exception"] = repr(exc)
        summary["traceback_tail"] = traceback.format_exc()[-4000:]
        write_json(OUT_JSON, summary, print_payload=True)
    finally:
        sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == "__main__":
    main()
