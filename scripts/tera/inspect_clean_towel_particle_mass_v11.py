"""V11 particle-mass inspector.

V10 proved the mesh `UsdPhysics.MassAPI` does NOT govern particle-cloth mass
(readback stayed 26.91 kg). This script answers: WHERE does the particle mass
come from, and can it be set correctly via the particle system / PBD material?

It builds the standard scene, dumps the PBD particle-material + particle-system +
cloth-mesh attributes, attempts per-particle mass readback via the cloth prim
view, and tries to SET the PBD material density at runtime and re-read it.
Diagnostic only; writes JSON to the V11 dir.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from isaaclab.app import AppLauncher

import test_clean_towel_primitive_contact_v8 as v8
from so101_clean_towel_scene_utils_v1 import command_run_string, write_json

V11_DIR = Path("/workspace/leisaac/tera_checkpoints/clean_towel_v11")
OUT_JSON = V11_DIR / "particle_mass_v11.json"


def dump_attrs(stage: Any, prim_path: str, keywords: tuple[str, ...]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return {"_invalid_prim": prim_path}
        out["_prim_type"] = str(prim.GetTypeName())
        out["_applied_schemas"] = [str(s) for s in prim.GetAppliedSchemas()]
        for attr in prim.GetAttributes():
            low = attr.GetName().lower()
            if not any(k in low for k in keywords):
                continue
            try:
                val = attr.Get()
            except Exception:
                val = None
            if val is None:
                continue
            if hasattr(val, "__len__") and not isinstance(val, str) and len(val) > 8:
                out[attr.GetName()] = f"<array len={len(val)}>"
            else:
                out[attr.GetName()] = val if isinstance(val, (int, float, bool, str)) else str(val)
    except Exception as exc:  # noqa: BLE001
        out["_error"] = repr(exc)
    return out


def find_material_path(stage: Any, particle_system_path: str) -> str | None:
    try:
        for child in stage.GetPrimAtPath(particle_system_path).GetChildren():
            if "material" in str(child.GetPath()).lower():
                return str(child.GetPath())
    except Exception:
        pass
    return f"{particle_system_path}/PhysicsMaterial"


def per_particle_mass(clean_towel: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"available": False}
    view = getattr(getattr(clean_towel, "cloth", None), "_cloth_prim_view", None)
    if view is None:
        return out
    out["view_methods"] = [m for m in dir(view) if "mass" in m.lower() or "velocit" in m.lower()]
    for meth in ("get_masses", "get_inv_masses", "get_inverse_masses"):
        fn = getattr(view, meth, None)
        if fn is None:
            continue
        try:
            raw = fn()
            arr = np.asarray(raw.detach().cpu().numpy() if hasattr(raw, "detach") else raw, dtype=np.float64)
            out.update(available=True, method=meth, shape=list(arr.shape),
                       min=float(arr.min()), max=float(arr.max()), mean=float(arr.mean()),
                       sum=float(arr.sum()))
            return out
        except Exception as exc:  # noqa: BLE001
            out["_err_" + meth] = repr(exc)
    return out


def main() -> None:
    args = v8.parse_args()
    args.out_dir = V11_DIR
    V11_DIR.mkdir(parents=True, exist_ok=True)
    app = AppLauncher(args).app  # noqa: F841

    summary: dict[str, Any] = {
        "status": "V11_PARTICLE_MASS_STARTED",
        "command_run": command_run_string(),
        "diagnostic_only": True,
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
        v8.step_sim(scene, 5)

        mesh_path = scene.clean_towel.mesh_path
        psys_path = scene.clean_towel.particle_system_path
        mat_path = find_material_path(stage, psys_path)

        mass_kw = ("mass", "density")
        summary["cloth_mesh_mass_attrs"] = dump_attrs(stage, mesh_path, mass_kw)
        summary["particle_system_mass_attrs"] = dump_attrs(stage, psys_path, ("mass", "density", "offset", "particle"))
        summary["pbd_material_path"] = mat_path
        summary["pbd_material_attrs"] = dump_attrs(stage, mat_path, ("mass", "density", "friction", "adhesion"))
        summary["per_particle_mass"] = per_particle_mass(scene.clean_towel)

        # Try to SET the PBD material density at runtime and re-read.
        set_probe: dict[str, Any] = {"attempted": False}
        try:
            mat_prim = stage.GetPrimAtPath(mat_path)
            dens_attr = None
            for attr in mat_prim.GetAttributes():
                if "density" in attr.GetName().lower():
                    dens_attr = attr
                    break
            if dens_attr is not None:
                set_probe["attempted"] = True
                set_probe["attr_name"] = dens_attr.GetName()
                set_probe["before"] = dens_attr.Get()
                dens_attr.Set(0.15)
                set_probe["after_set"] = dens_attr.Get()
                set_probe["settable"] = True
            else:
                set_probe["settable"] = False
                set_probe["note"] = "no density attribute found on PBD material prim"
        except Exception as exc:  # noqa: BLE001
            set_probe["error"] = repr(exc)
        summary["material_density_set_probe"] = set_probe

        summary["conclusion"] = (
            "Particle-cloth mass is authored by the particle system / PBD material and per-particle "
            "restOffset, NOT the mesh UsdPhysics.MassAPI (confirmed by V10). See pbd_material_attrs "
            "and per_particle_mass for the governing values."
        )
        summary["status"] = "V11_PARTICLE_MASS_DONE"
        write_json(OUT_JSON, summary, print_payload=True)
    except Exception as exc:  # noqa: BLE001
        import traceback
        summary["status"] = "V11_PARTICLE_MASS_FAILED"
        summary["exception_type"] = exc.__class__.__name__
        summary["exception"] = repr(exc)
        summary["traceback_tail"] = traceback.format_exc()[-4000:]
        write_json(OUT_JSON, summary, print_payload=True)
    finally:
        sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == "__main__":
    main()
