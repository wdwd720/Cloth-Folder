"""V10 corrected-mass clean-towel contact test.

V9 found the concrete cause of "touch but zero motion": the clean towel is
authored with `physics:mass = 26.91 kg` (2691x too heavy) because
`create_clean_rect_towel_usd_v2.py` sets `MassAPI density = --mass` and PhysX
applies that density PER PARTICLE (2691 particles x 0.01 = 26.91 kg), on top of
very stiff springs (stretch 10000). A 0.10 kg kinematic sphere cannot drag such
a sheet.

V10 tests the fix WITHOUT modifying either proven script:
- It re-authors the towel by running the UNCHANGED create script with
  `--mass = target_total / num_particles` (so the per-particle density yields a
  realistic total mass) and much softer springs.
- It then runs the UNCHANGED V8 contact trial machinery against the corrected
  USD and reports whether the collider now moves the towel.

Success (real rigid contact, not particle teleporting):
  edge_displacement > 0.02 m  AND  best_width < 0.65.
This is a physics diagnostic; no policy/training/folding-success claim is made.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from isaaclab.app import AppLauncher

import test_clean_towel_primitive_contact_v8 as v8
from so101_clean_towel_scene_utils_v1 import command_run_string, size_metrics, write_json

V10_DIR = Path("/workspace/leisaac/tera_checkpoints/clean_towel_v10")
CORRECTED_USD = V10_DIR / "clean_rect_towel_corrected_mass_v10.usda"
V10_SUMMARY = V10_DIR / "contact_corrected_mass_v10.json"

TARGET_TOTAL_MASS_KG = 0.15
EXPECTED_NUM_PARTICLES = 2691  # 69 x 39 grid at 0.01 spacing
SOFT_STRETCH = 300.0
SOFT_BEND = 40.0
SOFT_SHEAR = 60.0
SOFT_DAMPING = 2.0


def _isaac_python() -> str:
    for cand in ("/isaac-sim/python.sh", "/workspace/isaaclab/_isaac_sim/python.sh"):
        if Path(cand).exists():
            return cand
    return sys.executable


def author_corrected_usd(create_script: Path) -> dict[str, Any]:
    """Run the UNCHANGED create script with corrected per-particle mass + soft springs."""
    per_particle_mass = TARGET_TOTAL_MASS_KG / float(EXPECTED_NUM_PARTICLES)
    cmd = [
        _isaac_python(), str(create_script),
        "--usd_path", str(CORRECTED_USD),
        "--out_dir", str(V10_DIR),
        "--mass", f"{per_particle_mass:.8e}",
        "--stretch_stiffness", str(SOFT_STRETCH),
        "--bend_stiffness", str(SOFT_BEND),
        "--shear_stiffness", str(SOFT_SHEAR),
        "--spring_damping", str(SOFT_DAMPING),
    ]
    proc = subprocess.run(
        cmd, cwd=str(create_script.parent.parent.parent), text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=1200,
        env=os.environ.copy(),
    )
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "per_particle_mass_authored": per_particle_mass,
        "target_total_mass_kg": TARGET_TOTAL_MASS_KG,
        "stdout_tail": proc.stdout[-3000:],
        "stderr_tail": proc.stderr[-4000:],
        "usd_written": CORRECTED_USD.exists(),
    }


def read_cloth_mass(stage: Any, mesh_path: str) -> float | None:
    try:
        prim = stage.GetPrimAtPath(mesh_path)
        attr = prim.GetAttribute("physics:mass")
        return float(attr.Get()) if attr and attr.Get() is not None else None
    except Exception:
        return None


def main() -> None:
    # Full trial-param namespace from the V8 parser, then point it at corrected USD.
    args = v8.parse_args()
    args.usd_path = CORRECTED_USD
    args.out_dir = V10_DIR
    V10_DIR.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "status": "CLEAN_TOWEL_CONTACT_CORRECTED_MASS_V10_STARTED",
        "command_run": command_run_string(),
        "diagnostic_only": True,
        "no_policy_no_training_no_folding_success_claim": True,
        "target_total_mass_kg": TARGET_TOTAL_MASS_KG,
        "soft_springs": {"stretch": SOFT_STRETCH, "bend": SOFT_BEND, "shear": SOFT_SHEAR, "damping": SOFT_DAMPING},
    }

    # Locate the unchanged create script (sibling in scripts/tera).
    create_script = Path(__file__).with_name("create_clean_rect_towel_usd_v2.py")
    summary["authoring"] = author_corrected_usd(create_script)
    if not CORRECTED_USD.exists():
        summary["status"] = "CLEAN_TOWEL_CONTACT_CORRECTED_MASS_V10_AUTHORING_FAILED"
        write_json(V10_SUMMARY, summary, print_payload=True)
        sys.stdout.flush(); sys.stderr.flush(); os._exit(0)

    app_launcher = AppLauncher(args)
    _app = app_launcher.app
    try:
        from isaacsim.core.utils.stage import get_current_stage

        scene, scene_info = v8.create_towel_primitive_scene(
            args,
            primitive_radius=float(args.primitive_radius),
            primitive_contact_offset=float(args.primitive_contact_offset),
            primitive_rest_offset=float(args.primitive_rest_offset),
            static_friction=float(args.static_friction),
            dynamic_friction=float(args.dynamic_friction),
        )
        stage = get_current_stage()
        summary["scene_info"] = scene_info
        summary["corrected_cloth_mass_readback_kg"] = read_cloth_mass(stage, scene.clean_towel.mesh_path)

        v8.step_sim(scene, int(args.settle_initial_steps))
        start_points = scene.clean_towel.points_numpy().astype(np.float32)
        summary["start_metrics"] = size_metrics(start_points)

        params = v8.default_primitive_params(args)
        summary["contact_param_application"] = v8.apply_contact_params(scene, params)
        result = v8.run_primitive_contact_trial(
            scene, start_points, params, edge_band=float(args.edge_band), reset_points=True
        )

        summary["result"] = result
        summary["actual_touch_achieved"] = result["actual_touch_achieved"]
        summary["selected_edge_particle_displacement"] = result["selected_edge_particle_displacement"]
        summary["start_width"] = result["start_width"]
        summary["best_width"] = result["best_width"]
        summary["final_width"] = result["final_width"]
        summary["primitive_moves_towel"] = result["primitive_moves_towel"]
        summary["primitive_width_reduction_success"] = result["primitive_width_reduction_success"]
        summary["corrected_mass_enables_contact_transfer"] = bool(
            result["actual_touch_achieved"]
            and result["selected_edge_particle_displacement"] > 0.02
        )
        summary["status"] = "CLEAN_TOWEL_CONTACT_CORRECTED_MASS_V10_DONE"
        write_json(V10_SUMMARY, summary, print_payload=True)
    except Exception as exc:  # noqa: BLE001
        import traceback

        summary["status"] = "CLEAN_TOWEL_CONTACT_CORRECTED_MASS_V10_FAILED"
        summary["exception_type"] = exc.__class__.__name__
        summary["exception"] = repr(exc)
        summary["traceback_tail"] = traceback.format_exc()[-4000:]
        write_json(V10_SUMMARY, summary, print_payload=True)
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
