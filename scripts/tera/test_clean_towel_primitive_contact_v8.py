from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from isaaclab.app import AppLauncher

from clean_towel_v5_contact_common import edge_center_displacement, towel_edge_masks
from so101_clean_towel_scene_utils_v1 import (
    CLEAN_TOWEL_PARTICLE_SYSTEM_SUFFIX,
    CLEAN_TOWEL_USD_PATH,
    command_run_string,
    force_headless_no_cameras,
    initialize_clean_towel_handle_for_sim,
    points_to_numpy,
    size_metrics,
    write_json,
)


V8_DIR = Path("/workspace/leisaac/tera_checkpoints/clean_towel_v8_contact_sanity")
PRIMITIVE_SUMMARY_JSON = V8_DIR / "primitive_contact_summary.json"
CONTACT_OFFSETS_SUMMARY_JSON = V8_DIR / "contact_offsets_sweep_summary.json"
SO101_PROXY_SUMMARY_JSON = V8_DIR / "so101_proxy_fingertip_summary.json"
STATUS_JSON = V8_DIR / "STATUS.json"

TOWEL_ROOT_PATH = "/World/Scene/clean_towel"
PRIMITIVE_ROOT_PATH = "/World/PrimitiveFinger"
PRIMITIVE_GEOM_PATH = f"{PRIMITIVE_ROOT_PATH}/geometry/mesh"
DEFAULT_TOWEL_TRANSLATION = (0.0, 0.0, 0.02)
DEFAULT_EDGE = "right"


@dataclass
class TowelPrimitiveScene:
    sim: Any
    clean_towel: Any
    primitive_root_path: str
    primitive_geom_path: str
    device: str


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def finite_float(value: Any) -> tuple[bool, float | None]:
    try:
        number = float(value)
    except Exception:
        return False, None
    return bool(np.isfinite(number)), number


def set_prim_translation(stage: Any, prim_path: str, xyz: np.ndarray | list[float] | tuple[float, float, float]) -> bool:
    from pxr import Gf, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return False
    xform = UsdGeom.Xformable(prim)
    translate_op = None
    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            translate_op = op
            break
    if translate_op is None:
        translate_op = xform.AddTranslateOp()
    value = np.asarray(xyz, dtype=np.float32).reshape(3)
    translate_op.Set(Gf.Vec3d(float(value[0]), float(value[1]), float(value[2])))
    return True


def set_sphere_radius(stage: Any, geom_path: str, radius: float) -> bool:
    prim = stage.GetPrimAtPath(geom_path)
    if not prim or not prim.IsValid():
        return False
    attr = prim.GetAttribute("radius")
    if not attr:
        return False
    attr.Set(float(radius))
    return True


def numeric_attr_set(attr: Any, value: float) -> bool:
    try:
        current = attr.Get()
        if isinstance(current, bool) or current is None:
            return False
        if isinstance(current, (int, float)) or hasattr(current, "real"):
            attr.Set(float(value))
            return True
    except Exception:
        return False
    return False


def set_matching_attrs(prim: Any, patterns: tuple[str, ...], value: float) -> list[dict[str, Any]]:
    applied = []
    try:
        attrs = prim.GetAttributes()
    except Exception:
        attrs = []
    for attr in attrs:
        name = attr.GetName()
        lower = name.lower()
        if not any(pattern in lower for pattern in patterns):
            continue
        old_value = None
        try:
            old_value = attr.Get()
        except Exception:
            old_value = None
        if numeric_attr_set(attr, value):
            applied.append(
                {
                    "prim_path": str(prim.GetPath()),
                    "attr_name": name,
                    "old_value": None if old_value is None else str(old_value),
                    "new_value": float(value),
                }
            )
    return applied


def bind_contact_material(
    stage: Any,
    prim_paths: list[str],
    static_friction: float,
    dynamic_friction: float,
    material_name: str = "V8PrimitiveContactMaterial",
) -> dict[str, Any]:
    result = {
        "requested_static_friction": float(static_friction),
        "requested_dynamic_friction": float(dynamic_friction),
        "material_path": None,
        "bound_prim_paths": [],
        "error": None,
    }
    try:
        from pxr import UsdPhysics, UsdShade

        material_path = (
            f"/World/{material_name}_sf_{str(static_friction).replace('.', '_')}"
            f"_df_{str(dynamic_friction).replace('.', '_')}"
        )
        material = UsdShade.Material.Define(stage, material_path)
        physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        physics_material.CreateStaticFrictionAttr(float(static_friction))
        physics_material.CreateDynamicFrictionAttr(float(dynamic_friction))
        physics_material.CreateRestitutionAttr(0.0)
        for prim_path in prim_paths:
            prim = stage.GetPrimAtPath(prim_path)
            if prim and prim.IsValid():
                UsdShade.MaterialBindingAPI(prim).Bind(
                    material,
                    bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                    materialPurpose="physics",
                )
                result["bound_prim_paths"].append(prim_path)
        result["material_path"] = material_path
    except Exception as exc:
        result["error"] = repr(exc)
    return result


def spawn_kinematic_sphere(
    prim_path: str,
    radius: float,
    contact_offset: float,
    rest_offset: float,
    static_friction: float,
    dynamic_friction: float,
    translation: tuple[float, float, float] = (1.0, 0.0, 0.2),
) -> dict[str, Any]:
    from isaaclab.sim import schemas
    from isaaclab.sim.spawners import materials
    from isaaclab.sim.spawners.shapes import SphereCfg

    cfg = SphereCfg(
        radius=float(radius),
        collision_props=schemas.CollisionPropertiesCfg(
            collision_enabled=True,
            contact_offset=float(contact_offset),
            rest_offset=float(rest_offset),
        ),
        rigid_props=schemas.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            kinematic_enabled=True,
            disable_gravity=True,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=2,
            max_depenetration_velocity=2.0,
        ),
        mass_props=schemas.MassPropertiesCfg(mass=0.10),
        physics_material=materials.RigidBodyMaterialCfg(
            static_friction=float(static_friction),
            dynamic_friction=float(dynamic_friction),
            restitution=0.0,
            friction_combine_mode="max",
            restitution_combine_mode="min",
        ),
    )
    prim = cfg.func(prim_path, cfg, translation=translation)
    return {
        "primitive_root_path": str(prim_path),
        "primitive_geom_path": f"{prim_path}/geometry/mesh",
        "spawned_prim_path": str(prim.GetPath()),
        "radius": float(radius),
        "contact_offset": float(contact_offset),
        "rest_offset": float(rest_offset),
        "static_friction": float(static_friction),
        "dynamic_friction": float(dynamic_friction),
    }


def create_towel_primitive_scene(
    args: argparse.Namespace,
    primitive_radius: float = 0.04,
    primitive_contact_offset: float = 0.01,
    primitive_rest_offset: float = 0.0,
    static_friction: float = 1.2,
    dynamic_friction: float = 1.0,
    towel_translation: tuple[float, float, float] = DEFAULT_TOWEL_TRANSLATION,
) -> tuple[TowelPrimitiveScene, dict[str, Any]]:
    if not args.usd_path.exists():
        raise FileNotFoundError(f"Missing clean towel USD: {args.usd_path}")

    import isaaclab.sim as sim_utils
    from isaaclab.sim import SimulationCfg, SimulationContext
    from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
    from isaacsim.core.utils.stage import add_reference_to_stage, create_new_stage, get_current_stage, update_stage
    from pxr import Gf, UsdGeom

    create_new_stage()
    stage = get_current_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.Xform.Define(stage, "/World/Scene")

    sim = SimulationContext(SimulationCfg(device=args.device, dt=1.0 / 60.0, render_interval=1))
    spawn_ground_plane(prim_path="/World/Ground", cfg=GroundPlaneCfg())
    light_cfg = sim_utils.DomeLightCfg(intensity=1200.0, color=(0.8, 0.8, 0.8))
    light_cfg.func("/World/Light", light_cfg)

    reference_prim = add_reference_to_stage(str(args.usd_path), TOWEL_ROOT_PATH)
    UsdGeom.Xformable(reference_prim).AddTranslateOp().Set(Gf.Vec3d(*towel_translation))

    primitive_info = spawn_kinematic_sphere(
        PRIMITIVE_ROOT_PATH,
        primitive_radius,
        primitive_contact_offset,
        primitive_rest_offset,
        static_friction,
        dynamic_friction,
        translation=(1.0, 0.0, 0.2),
    )
    update_stage()

    sim.reset()
    clean_towel = initialize_clean_towel_handle_for_sim(sim, TOWEL_ROOT_PATH)
    sim.forward()
    scene = TowelPrimitiveScene(
        sim=sim,
        clean_towel=clean_towel,
        primitive_root_path=PRIMITIVE_ROOT_PATH,
        primitive_geom_path=PRIMITIVE_GEOM_PATH,
        device=str(args.device),
    )
    scene_info = {
        "scene_construction": "standalone_clean_towel_plus_ground_plus_kinematic_primitive",
        "clean_towel_usd": str(args.usd_path),
        "towel_root_path": TOWEL_ROOT_PATH,
        "towel_mesh_path": clean_towel.mesh_path,
        "towel_particle_system_path": clean_towel.particle_system_path,
        "ground_prim_path": "/World/Ground",
        "light_prim_path": "/World/Light",
        "primitive_info": primitive_info,
        "device": str(args.device),
        "no_robot_loaded": True,
        "no_cameras": True,
    }
    return scene, scene_info


def step_sim(scene: TowelPrimitiveScene, steps: int = 1) -> None:
    for _ in range(int(steps)):
        scene.sim.step(render=False)


def edge_reference(points: np.ndarray, edge: str, edge_band: float) -> dict[str, Any]:
    pts = points_to_numpy(points)
    masks = towel_edge_masks(pts, edge_band)
    mask = masks[edge]
    edge_pts = pts[mask]
    metrics = size_metrics(pts)
    if edge_pts.size == 0:
        center = np.asarray(metrics["center"], dtype=np.float32)
    else:
        center = edge_pts.mean(axis=0).astype(np.float32)
    return {
        "edge": edge,
        "edge_particle_count": int(edge_pts.shape[0]),
        "edge_center": [float(x) for x in center],
        "towel_metrics": metrics,
    }


def primitive_path_positions(
    start_points: np.ndarray,
    edge: str,
    radius: float,
    primitive_height: float,
    penetration_depth: float,
    drag_distance: float,
    drag_height_delta: float,
    outside_margin: float,
    edge_band: float,
) -> dict[str, Any]:
    pts = points_to_numpy(start_points)
    ref = edge_reference(pts, edge, edge_band)
    metrics = ref["towel_metrics"]
    cmin = np.asarray(metrics["min"], dtype=np.float32)
    cmax = np.asarray(metrics["max"], dtype=np.float32)
    center = np.asarray(metrics["center"], dtype=np.float32)
    edge_center = np.asarray(ref["edge_center"], dtype=np.float32)
    edge_z = float(edge_center[2])
    radius = float(radius)
    outside_margin = float(outside_margin)
    penetration_depth = float(penetration_depth)
    drag_distance = float(drag_distance)
    height_z = edge_z + float(primitive_height)

    if edge == "right":
        outside = np.asarray([cmax[0] + radius + outside_margin, center[1], height_z], dtype=np.float32)
        contact = np.asarray([cmax[0] + radius - penetration_depth, center[1], height_z], dtype=np.float32)
        drag = np.asarray([contact[0] - drag_distance, center[1], height_z + float(drag_height_delta)], dtype=np.float32)
    elif edge == "left":
        outside = np.asarray([cmin[0] - radius - outside_margin, center[1], height_z], dtype=np.float32)
        contact = np.asarray([cmin[0] - radius + penetration_depth, center[1], height_z], dtype=np.float32)
        drag = np.asarray([contact[0] + drag_distance, center[1], height_z + float(drag_height_delta)], dtype=np.float32)
    elif edge == "front":
        outside = np.asarray([center[0], cmin[1] - radius - outside_margin, height_z], dtype=np.float32)
        contact = np.asarray([center[0], cmin[1] - radius + penetration_depth, height_z], dtype=np.float32)
        drag = np.asarray([center[0], contact[1] + drag_distance, height_z + float(drag_height_delta)], dtype=np.float32)
    else:
        outside = np.asarray([center[0], cmax[1] + radius + outside_margin, height_z], dtype=np.float32)
        contact = np.asarray([center[0], cmax[1] + radius - penetration_depth, height_z], dtype=np.float32)
        drag = np.asarray([center[0], contact[1] - drag_distance, height_z + float(drag_height_delta)], dtype=np.float32)
    return {
        "edge_reference": ref,
        "outside_position": [float(x) for x in outside],
        "contact_position": [float(x) for x in contact],
        "drag_final_position": [float(x) for x in drag],
    }


def nearest_sphere_to_particles(
    center: np.ndarray | list[float],
    radius: float,
    particles: np.ndarray,
    edge: str | None = None,
    edge_band: float = 0.035,
) -> dict[str, Any]:
    pts = points_to_numpy(particles)
    indices = np.arange(pts.shape[0])
    if edge is not None:
        mask = towel_edge_masks(pts, edge_band)[edge]
        if np.any(mask):
            indices = indices[mask]
            pts = pts[mask]
    center_arr = np.asarray(center, dtype=np.float32).reshape(1, 3)
    distances = np.linalg.norm(pts - center_arr, axis=1)
    local_idx = int(np.argmin(distances))
    particle_index = int(indices[local_idx])
    center_distance = float(distances[local_idx])
    signed_surface_distance = float(center_distance - float(radius))
    return {
        "nearest_particle_index": particle_index,
        "nearest_particle_position": [float(x) for x in points_to_numpy(particles)[particle_index]],
        "center_distance_m": center_distance,
        "signed_surface_distance_m": signed_surface_distance,
        "surface_distance_m": float(max(0.0, signed_surface_distance)),
        "particles_inside_collider_count": int(np.sum(distances <= float(radius))),
    }


def lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return (1.0 - float(t)) * np.asarray(a, dtype=np.float32) + float(t) * np.asarray(b, dtype=np.float32)


def apply_contact_params(scene: TowelPrimitiveScene, params: dict[str, Any]) -> dict[str, Any]:
    from isaacsim.core.utils.stage import get_current_stage

    stage = get_current_stage()
    applied_attrs = []
    set_radius_ok = set_sphere_radius(stage, scene.primitive_geom_path, float(params["primitive_radius"]))
    for path in (
        scene.clean_towel.root_path,
        scene.clean_towel.mesh_path,
        scene.clean_towel.particle_system_path,
    ):
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            continue
        applied_attrs.extend(
            set_matching_attrs(
                prim,
                ("contactoffset", "contact_offset", "particlecontactoffset", "particle_contact_offset"),
                float(params["cloth_contact_offset"]),
            )
        )
        applied_attrs.extend(
            set_matching_attrs(
                prim,
                ("restoffset", "rest_offset", "solidrestoffset", "solid_rest_offset", "fluidrestoffset", "fluid_rest_offset"),
                float(params["cloth_rest_offset"]),
            )
        )
        applied_attrs.extend(set_matching_attrs(prim, ("radius", "thickness"), float(params["particle_radius"])))

    for path in (scene.primitive_root_path, scene.primitive_geom_path):
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            continue
        applied_attrs.extend(
            set_matching_attrs(
                prim,
                ("contactoffset", "contact_offset"),
                float(params["primitive_contact_offset"]),
            )
        )
        applied_attrs.extend(
            set_matching_attrs(
                prim,
                ("restoffset", "rest_offset"),
                float(params["primitive_rest_offset"]),
            )
        )
    material_result = bind_contact_material(
        stage,
        [scene.primitive_geom_path, scene.clean_towel.mesh_path, scene.clean_towel.particle_system_path],
        float(params["static_friction"]),
        float(params["dynamic_friction"]),
    )
    try:
        scene.sim.forward()
    except Exception:
        pass
    return {
        "set_primitive_radius_ok": bool(set_radius_ok),
        "applied_attrs": applied_attrs,
        "applied_attribute_count": int(len(applied_attrs)),
        "material_result": material_result,
    }


def result_sanity(result: dict[str, Any]) -> dict[str, Any]:
    reasons = []
    for key in (
        "nearest_collider_to_towel_particle_m",
        "selected_edge_particle_displacement",
        "start_width",
        "best_width",
        "final_width",
    ):
        finite, number = finite_float(result.get(key))
        if not finite or number is None:
            reasons.append(f"{key}_not_finite")
            continue
        if key in ("start_width", "best_width", "final_width") and not (0.10 <= number <= 1.25):
            reasons.append(f"{key}_outside_sane_range")
        if key == "selected_edge_particle_displacement" and not (0.0 <= number <= 1.0):
            reasons.append("selected_edge_particle_displacement_outside_sane_range")
    return {"sanity_valid": bool(not reasons), "sanity_invalid_reasons": reasons}


def run_primitive_contact_trial(
    scene: TowelPrimitiveScene,
    start_points: np.ndarray,
    params: dict[str, Any],
    edge_band: float = 0.035,
    reset_points: bool = True,
) -> dict[str, Any]:
    from isaacsim.core.utils.stage import get_current_stage

    stage = get_current_stage()
    edge = str(params.get("edge", DEFAULT_EDGE))
    radius = float(params["primitive_radius"])
    if reset_points:
        scene.clean_towel.set_points(start_points, device=scene.device)
        try:
            scene.sim.forward()
        except Exception:
            pass

    path_info = primitive_path_positions(
        start_points,
        edge=edge,
        radius=radius,
        primitive_height=float(params["primitive_height"]),
        penetration_depth=float(params["penetration_depth"]),
        drag_distance=float(params["drag_distance"]),
        drag_height_delta=float(params.get("drag_height_delta", 0.0)),
        outside_margin=float(params["outside_margin"]),
        edge_band=edge_band,
    )
    outside = np.asarray(path_info["outside_position"], dtype=np.float32)
    contact = np.asarray(path_info["contact_position"], dtype=np.float32)
    drag_final = np.asarray(path_info["drag_final_position"], dtype=np.float32)
    set_prim_translation(stage, scene.primitive_root_path, outside)
    try:
        scene.sim.forward()
    except Exception:
        pass

    settle_before_steps = int(params.get("settle_before_steps", 4))
    for _ in range(settle_before_steps):
        step_sim(scene, 1)

    widths = [float(size_metrics(scene.clean_towel.points_numpy())["width_x"])]
    nearest_values = []
    nearest_edge_values = []
    phase_counts = {
        "approach": int(params["approach_steps"]),
        "hold_contact": int(params["hold_contact_steps"]),
        "drag": int(params["drag_steps"]),
        "settle_after": int(params["settle_after_steps"]),
    }
    centers_by_phase: list[tuple[str, np.ndarray]] = []
    for step in range(phase_counts["approach"]):
        denom = max(1, phase_counts["approach"] - 1)
        centers_by_phase.append(("approach", lerp(outside, contact, step / denom)))
    for _ in range(phase_counts["hold_contact"]):
        centers_by_phase.append(("hold_contact", contact.copy()))
    for step in range(phase_counts["drag"]):
        denom = max(1, phase_counts["drag"] - 1)
        centers_by_phase.append(("drag", lerp(contact, drag_final, step / denom)))
    for _ in range(phase_counts["settle_after"]):
        centers_by_phase.append(("settle_after", drag_final.copy()))

    final_center = outside.copy()
    for _, center in centers_by_phase:
        final_center = center
        set_prim_translation(stage, scene.primitive_root_path, center)
        step_sim(scene, 1)
        points = scene.clean_towel.points_numpy().astype(np.float32)
        widths.append(float(size_metrics(points)["width_x"]))
        nearest = nearest_sphere_to_particles(center, radius, points, edge=None, edge_band=edge_band)
        nearest_edge = nearest_sphere_to_particles(center, radius, points, edge=edge, edge_band=edge_band)
        nearest_values.append(float(nearest["surface_distance_m"]))
        nearest_edge_values.append(float(nearest_edge["surface_distance_m"]))

    final_points = scene.clean_towel.points_numpy().astype(np.float32)
    widths_arr = np.asarray(widths, dtype=np.float32)
    edge_disp = edge_center_displacement(start_points, final_points, edge, edge_band)
    nearest_all = float(min(nearest_values)) if nearest_values else float("inf")
    nearest_edge_all = float(min(nearest_edge_values)) if nearest_edge_values else float("inf")
    result = {
        "trial_id": int(params.get("trial_id", 0)),
        "edge": edge,
        "primitive_shape": "kinematic_sphere",
        "primitive_params": {k: v for k, v in params.items() if not str(k).startswith("_")},
        "path_info": path_info,
        "final_primitive_center": [float(x) for x in final_center],
        "nearest_collider_to_towel_particle_m": nearest_all,
        "nearest_collider_to_selected_edge_particle_m": nearest_edge_all,
        "selected_edge_particle_displacement": float(edge_disp),
        "start_width": float(widths_arr[0]),
        "best_width": float(widths_arr.min()),
        "final_width": float(widths_arr[-1]),
        "actual_touch_achieved": bool(nearest_all < 0.01),
        "primitive_moves_towel": bool(edge_disp > 0.02),
        "primitive_width_reduction_success": bool(float(widths_arr.min()) < 0.65),
        "num_motion_steps": int(len(centers_by_phase)),
    }
    result.update(result_sanity(result))
    return result


def default_primitive_params(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "trial_id": 0,
        "edge": str(args.edge),
        "primitive_radius": float(args.primitive_radius),
        "primitive_contact_offset": float(args.primitive_contact_offset),
        "primitive_rest_offset": float(args.primitive_rest_offset),
        "static_friction": float(args.static_friction),
        "dynamic_friction": float(args.dynamic_friction),
        "cloth_contact_offset": float(args.cloth_contact_offset),
        "cloth_rest_offset": float(args.cloth_rest_offset),
        "particle_radius": float(args.particle_radius),
        "primitive_height": float(args.primitive_height),
        "penetration_depth": float(args.penetration_depth),
        "drag_distance": float(args.drag_distance),
        "drag_height_delta": float(args.drag_height_delta),
        "outside_margin": float(args.outside_margin),
        "approach_steps": int(args.approach_steps),
        "hold_contact_steps": int(args.hold_contact_steps),
        "drag_steps": int(args.drag_steps),
        "settle_after_steps": int(args.settle_after_steps),
        "settle_before_steps": int(args.settle_before_steps),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V8 isolated primitive collider contact sanity test for clean towel.")
    parser.add_argument("--usd_path", type=Path, default=CLEAN_TOWEL_USD_PATH)
    parser.add_argument("--out_dir", type=Path, default=V8_DIR)
    parser.add_argument("--edge", type=str, default=DEFAULT_EDGE, choices=["left", "right", "front", "back"])
    parser.add_argument("--edge_band", type=float, default=0.035)
    parser.add_argument("--settle_initial_steps", type=int, default=30)
    parser.add_argument("--settle_before_steps", type=int, default=4)
    parser.add_argument("--approach_steps", type=int, default=50)
    parser.add_argument("--hold_contact_steps", type=int, default=12)
    parser.add_argument("--drag_steps", type=int, default=80)
    parser.add_argument("--settle_after_steps", type=int, default=20)
    parser.add_argument("--primitive_radius", type=float, default=0.04)
    parser.add_argument("--primitive_height", type=float, default=0.012)
    parser.add_argument("--penetration_depth", type=float, default=0.055)
    parser.add_argument("--drag_distance", type=float, default=0.28)
    parser.add_argument("--drag_height_delta", type=float, default=0.0)
    parser.add_argument("--outside_margin", type=float, default=0.05)
    parser.add_argument("--primitive_contact_offset", type=float, default=0.012)
    parser.add_argument("--primitive_rest_offset", type=float, default=0.0)
    parser.add_argument("--cloth_contact_offset", type=float, default=0.005)
    parser.add_argument("--cloth_rest_offset", type=float, default=0.003)
    parser.add_argument("--particle_radius", type=float, default=0.005)
    parser.add_argument("--static_friction", type=float, default=1.2)
    parser.add_argument("--dynamic_friction", type=float, default=1.0)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    force_headless_no_cameras(args)
    return args


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    app_launcher = AppLauncher(args)
    _simulation_app = app_launcher.app
    try:
        scene, scene_info = create_towel_primitive_scene(
            args,
            primitive_radius=float(args.primitive_radius),
            primitive_contact_offset=float(args.primitive_contact_offset),
            primitive_rest_offset=float(args.primitive_rest_offset),
            static_friction=float(args.static_friction),
            dynamic_friction=float(args.dynamic_friction),
        )
        step_sim(scene, int(args.settle_initial_steps))
        start_points = scene.clean_towel.points_numpy().astype(np.float32)
        params = default_primitive_params(args)
        application = apply_contact_params(scene, params)
        result = run_primitive_contact_trial(scene, start_points, params, edge_band=float(args.edge_band), reset_points=True)
        summary = {
            "status": "CLEAN_TOWEL_PRIMITIVE_CONTACT_V8_DONE",
            "command_run": command_run_string(),
            "scene_info": scene_info,
            "start_towel_metrics": size_metrics(start_points),
            "contact_param_application": application,
            "result": result,
            "nearest_collider_to_towel_particle_m": result["nearest_collider_to_towel_particle_m"],
            "selected_edge_particle_displacement": result["selected_edge_particle_displacement"],
            "start_width": result["start_width"],
            "best_width": result["best_width"],
            "final_width": result["final_width"],
            "actual_touch_achieved": result["actual_touch_achieved"],
            "primitive_moves_towel": result["primitive_moves_towel"],
            "primitive_width_reduction_success": result["primitive_width_reduction_success"],
            "primitive_contact_summary_path": str(args.out_dir / PRIMITIVE_SUMMARY_JSON.name),
            "honest_note": (
                "This is an isolated kinematic primitive collider test. It loads the clean towel v2 USD, ground, "
                "and one primitive collider only; no policy training, SO-101 robot, particle attachment, or folding claim is used."
            ),
        }
        write_json(args.out_dir / PRIMITIVE_SUMMARY_JSON.name, summary, print_payload=True)
    except Exception as exc:
        summary = {
            "status": "CLEAN_TOWEL_PRIMITIVE_CONTACT_V8_FAILED",
            "command_run": command_run_string(),
            "exception_type": exc.__class__.__name__,
            "exception": repr(exc),
            "actual_touch_achieved": False,
            "primitive_moves_towel": False,
            "primitive_width_reduction_success": False,
            "honest_note": "Primitive contact test failed; no contact, motion, folding, or training claim is made.",
        }
        write_json(args.out_dir / PRIMITIVE_SUMMARY_JSON.name, summary, print_payload=True)
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
