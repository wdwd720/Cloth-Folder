"""Shared utilities for V11 contact-drive research.

V11 tests the hypothesis (from V8/V9/V10) that the clean-towel contact blocker is
the COLLIDER DRIVE MECHANISM: V8 moved the kinematic sphere by editing its USD
Xform translate op (`set_prim_translation`), which teleports the body with ~zero
swept velocity, so PhysX imparts no momentum to the resting PBD particles.

This module provides:
- a scene builder mirroring V8's (towel + ground + one sphere) but with a
  configurable kinematic/dynamic + gravity sphere,
- a physics-aware drive (RigidPrim.set_world_poses -> real kinematic velocity)
  with a defensive fallback to the V8 USD-teleport,
- finite-difference particle-velocity measurement near the contact region,
- a generic drive-trial runner returning contact metrics.

All Isaac imports are lazy (inside functions) so the module imports on any host.
Reuses V8 scene constants + metrics. No particle teleporting counts as success.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

import test_clean_towel_primitive_contact_v8 as v8
from clean_towel_v5_contact_common import edge_center_displacement, towel_edge_masks
from so101_clean_towel_scene_utils_v1 import (
    DEFAULT_TOWEL_TRANSLATION,
    initialize_clean_towel_handle_for_sim,
    size_metrics,
)

TOWEL_ROOT_PATH = v8.TOWEL_ROOT_PATH
PRIMITIVE_ROOT_PATH = v8.PRIMITIVE_ROOT_PATH
PRIMITIVE_GEOM_PATH = v8.PRIMITIVE_GEOM_PATH


@dataclass
class V11Scene:
    sim: Any
    clean_towel: Any
    primitive_root_path: str
    device: str


def spawn_sphere(
    prim_path: str,
    radius: float,
    contact_offset: float,
    rest_offset: float,
    static_friction: float,
    dynamic_friction: float,
    kinematic: bool,
    disable_gravity: bool,
    mass: float,
    translation: tuple[float, float, float],
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
            kinematic_enabled=bool(kinematic),
            disable_gravity=bool(disable_gravity),
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=4,
            max_depenetration_velocity=5.0,
        ),
        mass_props=schemas.MassPropertiesCfg(mass=float(mass)),
        physics_material=materials.RigidBodyMaterialCfg(
            static_friction=float(static_friction),
            dynamic_friction=float(dynamic_friction),
            restitution=0.0,
            friction_combine_mode="max",
            restitution_combine_mode="min",
        ),
    )
    prim = cfg.func(prim_path, cfg, translation=translation)
    return {"spawned_prim_path": str(prim.GetPath()), "kinematic": bool(kinematic),
            "disable_gravity": bool(disable_gravity), "mass": float(mass), "radius": float(radius)}


def build_scene(
    args: Any,
    kinematic: bool,
    disable_gravity: bool,
    sphere_mass: float = 0.10,
    sphere_radius: float = 0.04,
    sphere_contact_offset: float = 0.012,
    sphere_rest_offset: float = 0.0,
    static_friction: float = 1.2,
    dynamic_friction: float = 1.0,
    sphere_translation: tuple[float, float, float] = (1.0, 0.0, 0.2),
    towel_translation: tuple[float, float, float] = DEFAULT_TOWEL_TRANSLATION,
) -> tuple[V11Scene, dict[str, Any]]:
    import isaaclab.sim as sim_utils
    from isaaclab.sim import SimulationCfg, SimulationContext
    from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
    from isaacsim.core.utils.stage import (
        add_reference_to_stage, create_new_stage, get_current_stage, update_stage,
    )
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

    sphere_info = spawn_sphere(
        PRIMITIVE_ROOT_PATH, sphere_radius, sphere_contact_offset, sphere_rest_offset,
        static_friction, dynamic_friction, kinematic, disable_gravity, sphere_mass, sphere_translation,
    )
    update_stage()
    sim.reset()
    clean_towel = initialize_clean_towel_handle_for_sim(sim, TOWEL_ROOT_PATH)
    sim.forward()
    scene = V11Scene(sim=sim, clean_towel=clean_towel, primitive_root_path=PRIMITIVE_ROOT_PATH,
                     device=str(args.device))
    info = {"sphere_info": sphere_info, "towel_mesh_path": clean_towel.mesh_path,
            "particle_system_path": clean_towel.particle_system_path,
            "kinematic": bool(kinematic), "disable_gravity": bool(disable_gravity)}
    return scene, info


def make_rigid_view(prim_path: str) -> tuple[Any, list[str]]:
    errs: list[str] = []
    for cls_name in ("RigidPrim", "SingleRigidPrim"):
        try:
            mod = __import__("isaacsim.core.prims", fromlist=[cls_name])
            cls = getattr(mod, cls_name)
        except Exception as exc:  # noqa: BLE001
            errs.append(f"import {cls_name}: {exc!r}")
            continue
        for kwargs in ({"prim_paths_expr": prim_path}, {"prim_path": prim_path}):
            try:
                view = cls(**kwargs)
                try:
                    view.initialize()
                except Exception:
                    pass
                return view, errs
            except Exception as exc:  # noqa: BLE001
                errs.append(f"{cls_name}({list(kwargs)[0]}): {exc!r}")
    return None, errs


def _to_view_tensor(xyz: np.ndarray, device: str):
    import torch
    return torch.tensor([[float(xyz[0]), float(xyz[1]), float(xyz[2])]], dtype=torch.float32, device=device)


def drive_pose(view: Any, stage: Any, prim_path: str, xyz: np.ndarray, device: str) -> str:
    """Move the collider to xyz. Prefer physics-aware set_world_poses (gives the
    kinematic body a real swept velocity). Fall back to the V8 USD teleport."""
    if view is not None:
        try:
            view.set_world_poses(positions=_to_view_tensor(xyz, device))
            return "physics_set_world_poses"
        except Exception:
            try:
                view.set_world_poses(_to_view_tensor(xyz, device), None)
                return "physics_set_world_poses_positional"
            except Exception:
                pass
    v8.set_prim_translation(stage, prim_path, xyz)
    return "usd_set_prim_translation"


def get_view_position(view: Any) -> np.ndarray | None:
    if view is None:
        return None
    fn = getattr(view, "get_world_poses", None)
    if fn is None:
        return None
    try:
        out = fn()
        pos = out[0] if isinstance(out, (tuple, list)) else out
        arr = pos.detach().cpu().numpy() if hasattr(pos, "detach") else np.asarray(pos)
        arr = np.asarray(arr, dtype=np.float32).reshape(-1, 3)
        return arr[0]
    except Exception:
        return None


def set_linear_velocity(view: Any, lin: np.ndarray, device: str) -> bool:
    import torch
    if view is None:
        return False
    vel6 = torch.tensor([[float(lin[0]), float(lin[1]), float(lin[2]), 0.0, 0.0, 0.0]],
                        dtype=torch.float32, device=device)
    for meth, arg in (("set_velocities", vel6), ("set_linear_velocities", vel6[:, :3])):
        fn = getattr(view, meth, None)
        if fn is None:
            continue
        try:
            fn(arg)
            return True
        except Exception:
            continue
    return False


def run_drive_trial(
    scene: V11Scene,
    start_points: np.ndarray,
    path_centers: list[np.ndarray],
    edge: str,
    edge_band: float,
    radius: float,
    drive_mode: str,  # "teleport" | "kinematic_velocity" | "dynamic_velocity"
    initial_velocity: np.ndarray | None = None,
) -> dict[str, Any]:
    from isaacsim.core.utils.stage import get_current_stage

    stage = get_current_stage()
    device = scene.device
    dt = float(scene.sim.get_physics_dt())

    view, view_errs = (None, [])
    if drive_mode in ("kinematic_velocity", "dynamic_velocity"):
        view, view_errs = make_rigid_view(scene.primitive_root_path)

    # reset cloth to a known start
    scene.clean_towel.set_points(start_points, device=device)
    try:
        scene.sim.forward()
    except Exception:
        pass

    if drive_mode == "dynamic_velocity":
        # place the dynamic sphere at the path start, then launch it with velocity
        drive_pose(view, stage, scene.primitive_root_path, np.asarray(path_centers[0], np.float32), device)
        try:
            scene.sim.forward()
        except Exception:
            pass
        if view is not None and initial_velocity is not None:
            set_linear_velocity(view, initial_velocity, device)

    widths = [float(size_metrics(scene.clean_towel.points_numpy())["width_x"])]
    nearest_vals: list[float] = []
    particle_speed_peak = 0.0
    drive_methods: set[str] = set()
    prev_points = scene.clean_towel.points_numpy().astype(np.float32)

    for center in path_centers:
        if drive_mode in ("teleport", "kinematic_velocity"):
            method = drive_pose(view, stage, scene.primitive_root_path, np.asarray(center, np.float32), device)
            drive_methods.add(method)
            measure_center = np.asarray(center, np.float32)
        else:
            # dynamic mode: sphere moves under its own velocity/gravity; measure at its real pose
            actual = get_view_position(view)
            measure_center = actual if actual is not None else np.asarray(center, np.float32)
        scene.sim.step(render=False)
        pts = scene.clean_towel.points_numpy().astype(np.float32)
        # finite-difference particle speed near the collider xy
        fd = np.linalg.norm(pts - prev_points, axis=1) / max(dt, 1e-6)
        d_xy = np.linalg.norm(pts[:, :2] - measure_center[:2], axis=1)
        near = d_xy < (radius + 0.05)
        if np.any(near):
            particle_speed_peak = max(particle_speed_peak, float(np.max(fd[near])))
        prev_points = pts
        widths.append(float(size_metrics(pts)["width_x"]))
        nearest_vals.append(float(v8.nearest_sphere_to_particles(
            measure_center, radius, pts, edge=None, edge_band=edge_band)["surface_distance_m"]))

    final_points = scene.clean_towel.points_numpy().astype(np.float32)
    widths_arr = np.asarray(widths, np.float32)
    edge_disp = float(edge_center_displacement(start_points, final_points, edge, edge_band))
    nearest_all = float(min(nearest_vals)) if nearest_vals else float("inf")
    max_particle_disp = float(np.max(np.linalg.norm(final_points - start_points, axis=1)))
    width_before = float(widths_arr[0])
    width_after = float(widths_arr[-1])
    best_width = float(widths_arr.min())
    width_reduction = width_before - best_width

    actual_touch = bool(nearest_all < 0.01)
    naive_threshold_pass = bool(actual_touch and edge_disp > 0.02)
    # Reject spurious ballistic results: a real fold-scale drag keeps particles
    # bounded (towel is ~0.68 m wide) and REDUCES width; a ballistic knock flings
    # particles far (edge_disp >> towel size) with particles pinned at the particle
    # system max_velocity (5.0 m/s) and no width reduction.
    spurious_ballistic = bool(
        edge_disp > 0.5 or max_particle_disp > 0.5 or particle_speed_peak >= 4.9
    )
    valid_controlled_contact = bool(
        actual_touch
        and 0.02 < edge_disp <= 0.5
        and width_reduction > 0.005
        and not spurious_ballistic
    )
    return {
        "drive_mode": drive_mode,
        "drive_methods_used": sorted(drive_methods),
        "rigid_view_available": view is not None,
        "rigid_view_errors": view_errs[:4],
        "actual_touch_distance_m": nearest_all,
        "actual_touch": actual_touch,
        "edge_displacement_m": edge_disp,
        "max_particle_displacement_m": max_particle_disp,
        "particle_velocity_peak": particle_speed_peak,
        "width_before": width_before,
        "width_after": width_after,
        "best_width": best_width,
        "width_reduction_m": width_reduction,
        "num_steps": len(path_centers),
        "naive_threshold_pass": naive_threshold_pass,
        "spurious_ballistic": spurious_ballistic,
        "valid_controlled_contact": valid_controlled_contact,
        # headline success = valid controlled contact ONLY (ballistic flings excluded)
        "contact_transfer_success": valid_controlled_contact,
    }


def default_path_centers(start_points: np.ndarray, args: Any, num_steps: int, edge: str = "right") -> list[np.ndarray]:
    """Approach->contact->drag path over the given edge (reuses V8 geometry)."""
    path_info = v8.primitive_path_positions(
        start_points, edge=edge, radius=float(args.primitive_radius),
        primitive_height=float(args.primitive_height), penetration_depth=float(args.penetration_depth),
        drag_distance=float(args.drag_distance), drag_height_delta=float(getattr(args, "drag_height_delta", 0.0)),
        outside_margin=float(args.outside_margin), edge_band=float(args.edge_band),
    )
    outside = np.asarray(path_info["outside_position"], np.float32)
    contact = np.asarray(path_info["contact_position"], np.float32)
    drag_final = np.asarray(path_info["drag_final_position"], np.float32)
    n_app = max(2, num_steps // 3)
    n_drag = max(2, num_steps - n_app)
    centers = [v8.lerp(outside, contact, i / max(1, n_app - 1)) for i in range(n_app)]
    centers += [v8.lerp(contact, drag_final, i / max(1, n_drag - 1)) for i in range(n_drag)]
    return centers
