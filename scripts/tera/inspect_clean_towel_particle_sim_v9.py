"""V9 clean-towel particle-simulation sanity probe.

The V8 result (reproduced on Modal Isaac Sim 4.5.0) showed: kinematic collider
achieves actual touch (nearest 0.0 m) but selected_edge_particle_displacement is
EXACTLY 0.0 and start==best==final width==0.68. That "exactly zero" is the clue:
before blaming collider->particle contact transfer, we must first answer a more
fundamental question:

    Does the clean-towel particle cloth simulate AT ALL under gravity?

This probe reuses V8's `create_towel_primitive_scene` (towel + ground + one
kinematic collider parked far from the towel), then steps the simulation with NO
manipulation and measures whether the cloth particles move under gravity. It also
dumps the PhysX particle-system / cloth parameters that would explain a frozen
cloth (pinned particles, disabled particle system, zero solver iterations, etc.).

Interpretation:
- max particle displacement under gravity > 1e-3 m  => solver IS advancing;
  the contact issue is specifically collider->particle transfer (V10 territory).
- max particle displacement ~ 0.0                   => cloth is FROZEN; that is
  the true root cause under "touch but no motion".

No robot, no policy, no folding claim. Diagnostic only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from isaaclab.app import AppLauncher

import test_clean_towel_primitive_contact_v8 as v8
from so101_clean_towel_scene_utils_v1 import (
    CLEAN_TOWEL_USD_PATH,
    command_run_string,
    force_headless_no_cameras,
    size_metrics,
    write_json,
)

V9_DIR = Path("/workspace/leisaac/tera_checkpoints/clean_towel_v9")
V9_SUMMARY_JSON = V9_DIR / "particle_sim_sanity_v9.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V9 clean towel particle-sim gravity sanity probe.")
    parser.add_argument("--usd_path", type=Path, default=CLEAN_TOWEL_USD_PATH)
    parser.add_argument("--out_dir", type=Path, default=V9_DIR)
    parser.add_argument("--primitive_radius", type=float, default=0.04)
    parser.add_argument("--primitive_contact_offset", type=float, default=0.012)
    parser.add_argument("--primitive_rest_offset", type=float, default=0.0)
    parser.add_argument("--static_friction", type=float, default=1.2)
    parser.add_argument("--dynamic_friction", type=float, default=1.0)
    parser.add_argument("--probe_steps", type=str, default="1,5,10,20,40,80,120")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    force_headless_no_cameras(args)
    return args


def dump_particle_params(stage: Any, particle_system_path: str, mesh_path: str) -> dict[str, Any]:
    """Best-effort dump of PhysX particle-system + cloth attributes."""
    info: dict[str, Any] = {"particle_system_path": particle_system_path, "mesh_path": mesh_path}
    keywords = (
        "particle", "contact", "rest", "offset", "iteration", "collision",
        "enable", "radius", "damping", "stiffness", "stretch", "bend",
        "friction", "adhesion", "mass", "gravity", "solver", "self",
    )

    def _prim_attr_dump(prim_path: str) -> dict[str, Any]:
        out: dict[str, Any] = {}
        try:
            prim = stage.GetPrimAtPath(prim_path)
            if not prim or not prim.IsValid():
                return {"_invalid_prim": prim_path}
            out["_prim_type"] = str(prim.GetTypeName())
            out["_applied_schemas"] = [str(s) for s in prim.GetAppliedSchemas()]
            for attr in prim.GetAttributes():
                name = attr.GetName()
                low = name.lower()
                if not any(k in low for k in keywords):
                    continue
                try:
                    val = attr.Get()
                except Exception:
                    val = None
                if val is None:
                    continue
                try:
                    if hasattr(val, "__len__") and not isinstance(val, str) and len(val) > 8:
                        out[name] = f"<array len={len(val)}>"
                    else:
                        out[name] = val if isinstance(val, (int, float, bool, str)) else str(val)
                except Exception:
                    out[name] = str(val)
        except Exception as exc:  # noqa: BLE001
            out["_error"] = repr(exc)
        return out

    info["particle_system_attrs"] = _prim_attr_dump(particle_system_path)
    info["cloth_mesh_attrs"] = _prim_attr_dump(mesh_path)
    return info


def probe_inverse_masses(clean_towel: Any) -> dict[str, Any]:
    """Pinned particles (inverse mass == 0) can't move -> would explain frozen cloth."""
    out: dict[str, Any] = {"available": False}
    view = getattr(getattr(clean_towel, "cloth", None), "_cloth_prim_view", None)
    if view is None:
        return out
    for meth in ("get_masses", "get_inv_masses", "get_inverse_masses"):
        fn = getattr(view, meth, None)
        if fn is None:
            continue
        try:
            arr = np.asarray(fn().detach().cpu().numpy() if hasattr(fn(), "detach") else fn())
            out.update(
                available=True,
                method=meth,
                shape=list(arr.shape),
                min=float(np.min(arr)),
                max=float(np.max(arr)),
                mean=float(np.mean(arr)),
                num_zero=int(np.sum(arr == 0.0)),
                total=int(arr.size),
            )
            return out
        except Exception as exc:  # noqa: BLE001
            out["_error_" + meth] = repr(exc)
    return out


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    app_launcher = AppLauncher(args)
    _app = app_launcher.app
    summary: dict[str, Any] = {
        "status": "CLEAN_TOWEL_PARTICLE_SIM_V9_STARTED",
        "command_run": command_run_string(),
        "diagnostic_only": True,
        "no_robot_no_policy_no_folding_claim": True,
    }
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

        p0 = scene.clean_towel.points_numpy().astype(np.float32)
        summary["num_particles"] = int(p0.shape[0])
        summary["start_metrics"] = size_metrics(p0)
        summary["start_mean_z"] = float(np.mean(p0[:, 2]))

        summary["particle_params"] = dump_particle_params(
            stage, scene.clean_towel.particle_system_path, scene.clean_towel.mesh_path
        )
        summary["inverse_mass_probe"] = probe_inverse_masses(scene.clean_towel)

        # Free-fall probe: collider stays parked far from towel; step under gravity only.
        probe_steps = sorted({int(s) for s in str(args.probe_steps).split(",") if s.strip()})
        gravity_trace = []
        stepped = 0
        for target in probe_steps:
            v8.step_sim(scene, max(0, target - stepped))
            stepped = target
            pk = scene.clean_towel.points_numpy().astype(np.float32)
            disp = np.linalg.norm(pk - p0, axis=1)
            gravity_trace.append(
                {
                    "step": int(target),
                    "max_particle_disp_m": float(np.max(disp)),
                    "mean_particle_disp_m": float(np.mean(disp)),
                    "mean_z": float(np.mean(pk[:, 2])),
                    "width_x": float(size_metrics(pk)["width_x"]),
                }
            )
        summary["gravity_trace"] = gravity_trace

        final_max_disp = float(gravity_trace[-1]["max_particle_disp_m"]) if gravity_trace else 0.0
        summary["final_max_particle_disp_m"] = final_max_disp
        summary["mean_z_drop_m"] = float(
            summary["start_mean_z"] - gravity_trace[-1]["mean_z"]
        ) if gravity_trace else 0.0
        summary["cloth_simulates_under_gravity"] = bool(final_max_disp > 1.0e-3)
        summary["root_cause_hypothesis"] = (
            "collider_to_particle_contact_transfer"
            if summary["cloth_simulates_under_gravity"]
            else "cloth_particle_solver_frozen_or_pinned"
        )
        summary["status"] = "CLEAN_TOWEL_PARTICLE_SIM_V9_DONE"
        write_json(args.out_dir / V9_SUMMARY_JSON.name, summary, print_payload=True)
    except Exception as exc:  # noqa: BLE001
        import traceback

        summary["status"] = "CLEAN_TOWEL_PARTICLE_SIM_V9_FAILED"
        summary["exception_type"] = exc.__class__.__name__
        summary["exception"] = repr(exc)
        summary["traceback_tail"] = traceback.format_exc()[-4000:]
        write_json(args.out_dir / V9_SUMMARY_JSON.name, summary, print_payload=True)
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
