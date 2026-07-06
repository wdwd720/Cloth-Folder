from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from isaaclab.app import AppLauncher

from clean_towel_v4_common import reset_scene_to_points
from clean_towel_v5_contact_common import edge_center_displacement
from inspect_so101_gripper_collision_v7 import (
    EDGE_PINCH_DRAG_SUMMARY_JSON,
    candidate_body_poses,
    clipped_action,
    load_json_if_exists,
    recommended_towel_center,
    recommended_towel_translation,
    v6_recommended_candidate,
)
from so101_clean_towel_scene_utils_v1 import (
    CLEAN_TOWEL_USD_PATH,
    TASK_ID,
    command_run_string,
    create_standalone_so101_clean_towel_scene,
    force_headless_no_cameras,
    initialize_clean_towel_handle_for_sim,
    size_metrics,
    standalone_robot_scene_info,
    write_json,
)
from test_clean_towel_primitive_contact_v8 import (
    CONTACT_OFFSETS_SUMMARY_JSON,
    PRIMITIVE_SUMMARY_JSON,
    SO101_PROXY_SUMMARY_JSON,
    STATUS_JSON,
    V8_DIR,
    nearest_sphere_to_particles,
    set_prim_translation,
    spawn_kinematic_sphere,
)
from test_so101_edge_pinch_drag_v7 import action_with_gripper, make_drag_action, make_pose_action, mapping_values


def selected_motion_params(arm: str) -> dict[str, Any]:
    pure = load_json_if_exists(EDGE_PINCH_DRAG_SUMMARY_JSON) or {}
    if pure.get("verified_best_result") and pure["verified_best_result"].get("params"):
        params = dict(pure["verified_best_result"]["params"])
    elif pure.get("best_result") and pure["best_result"].get("params"):
        params = dict(pure["best_result"]["params"])
    else:
        params = {
            "trial_id": 0,
            "approach_height_delta": 0.08,
            "jaw_depth_delta": -0.18,
            "close_duration": 24,
            "drag_distance": 0.32,
            "drag_height_delta": -0.04,
            "open_value": 0.9,
            "close_value": 0.0,
        }
    values = mapping_values(arm)
    if values["mapping_known"]:
        params["open_value"] = float(values["open_value"])
        params["close_value"] = float(values["close_value"])
    return params


def selected_proxy_body_poses(scene: Any, arm: str, max_bodies: int) -> list[dict[str, Any]]:
    poses = candidate_body_poses(scene, arm)
    preferred = [
        pose
        for pose in poses
        if pose.get("position") is not None and ("jaw" in pose["body_name"].lower() or "finger" in pose["body_name"].lower())
    ]
    if len(preferred) < int(max_bodies):
        for pose in poses:
            if pose.get("position") is None:
                continue
            if pose in preferred:
                continue
            if "gripper" in pose["body_name"].lower() or "tool" in pose["body_name"].lower() or "end" in pose["body_name"].lower():
                preferred.append(pose)
            if len(preferred) >= int(max_bodies):
                break
    if not preferred:
        preferred = [pose for pose in poses if pose.get("position") is not None]
    return preferred[: int(max_bodies)]


def spawn_proxy_spheres(
    count: int,
    radius: float,
    contact_offset: float,
    rest_offset: float,
    static_friction: float,
    dynamic_friction: float,
) -> list[dict[str, Any]]:
    proxies = []
    for index in range(int(count)):
        prim_path = f"/World/V8ProxyFingertip_{index}"
        info = spawn_kinematic_sphere(
            prim_path,
            radius=float(radius),
            contact_offset=float(contact_offset),
            rest_offset=float(rest_offset),
            static_friction=float(static_friction),
            dynamic_friction=float(dynamic_friction),
            translation=(1.25, 0.0, 0.30 + 0.05 * index),
        )
        proxies.append(info)
    return proxies


def set_proxy_positions(
    scene: Any,
    arm: str,
    proxies: list[dict[str, Any]],
    offset: np.ndarray,
) -> list[dict[str, Any]]:
    from isaacsim.core.utils.stage import get_current_stage

    stage = get_current_stage()
    poses = selected_proxy_body_poses(scene, arm, max_bodies=len(proxies))
    centers = []
    for index, proxy in enumerate(proxies):
        pose = poses[min(index, len(poses) - 1)] if poses else None
        if pose is None or pose.get("position") is None:
            center = np.asarray([1.25, 0.0, 0.30 + 0.05 * index], dtype=np.float32)
            body_name = None
            body_index = None
        else:
            center = np.asarray(pose["position"], dtype=np.float32).reshape(3) + offset
            body_name = pose.get("body_name")
            body_index = pose.get("body_index")
        set_prim_translation(stage, proxy["primitive_root_path"], center)
        centers.append(
            {
                "proxy_index": int(index),
                "proxy_root_path": proxy["primitive_root_path"],
                "follows_body_name": body_name,
                "follows_body_index": body_index,
                "center": [float(x) for x in center],
            }
        )
    return centers


def nearest_proxy_to_particles(
    centers: list[dict[str, Any]],
    radius: float,
    points: np.ndarray,
    edge: str,
    edge_band: float,
) -> dict[str, Any]:
    best = None
    for center_info in centers:
        nearest = nearest_sphere_to_particles(center_info["center"], radius, points, edge=None, edge_band=edge_band)
        nearest_edge = nearest_sphere_to_particles(center_info["center"], radius, points, edge=edge, edge_band=edge_band)
        item = {
            **center_info,
            "nearest_particle_index": nearest["nearest_particle_index"],
            "nearest_particle_position": nearest["nearest_particle_position"],
            "center_distance_m": nearest["center_distance_m"],
            "signed_surface_distance_m": nearest["signed_surface_distance_m"],
            "surface_distance_m": nearest["surface_distance_m"],
            "particles_inside_collider_count": nearest["particles_inside_collider_count"],
            "selected_edge_surface_distance_m": nearest_edge["surface_distance_m"],
        }
        if best is None or item["surface_distance_m"] < best["surface_distance_m"]:
            best = item
    return best or {
        "surface_distance_m": float("inf"),
        "selected_edge_surface_distance_m": float("inf"),
        "particles_inside_collider_count": 0,
    }


def apply_action_step_with_proxies(
    scene: Any,
    action: np.ndarray,
    arm: str,
    proxies: list[dict[str, Any]],
    proxy_offset: np.ndarray,
) -> list[dict[str, Any]]:
    action = clipped_action(np.asarray(action, dtype=np.float32)).reshape(1, 12)
    left = torch.tensor(action[:, :6], dtype=torch.float32, device=scene.device)
    right = torch.tensor(action[:, 6:], dtype=torch.float32, device=scene.device)
    scene.left_arm.set_joint_position_target(left)
    scene.right_arm.set_joint_position_target(right)
    scene.left_arm.write_data_to_sim()
    scene.right_arm.write_data_to_sim()
    centers_before_step = set_proxy_positions(scene, arm, proxies, proxy_offset)
    scene.sim.step(render=False)
    dt = scene.sim.get_physics_dt()
    scene.left_arm.update(dt)
    scene.right_arm.update(dt)
    set_proxy_positions(scene, arm, proxies, proxy_offset)
    return centers_before_step


def run_proxy_trial(
    scene: Any,
    start_points: np.ndarray,
    candidate: dict[str, Any],
    motion_params: dict[str, Any],
    proxies: list[dict[str, Any]],
    proxy_radius: float,
    proxy_offset: np.ndarray,
    approach_steps: int,
    drag_steps: int,
    edge_band: float,
) -> dict[str, Any]:
    arm = str(candidate["arm"])
    edge = str(candidate["edge"])
    base_action = clipped_action(np.asarray(candidate["action_vector"], dtype=np.float32))
    approach_action = make_pose_action(
        base_action,
        arm,
        edge,
        float(motion_params["approach_height_delta"]),
        float(motion_params["jaw_depth_delta"]),
        float(motion_params["open_value"]),
    )
    close_action = action_with_gripper(approach_action, arm, float(motion_params["close_value"]))
    drag_action = make_drag_action(
        close_action,
        arm,
        edge,
        float(motion_params["drag_distance"]),
        float(motion_params["drag_height_delta"]),
        float(motion_params["close_value"]),
    )

    reset_scene_to_points(scene, start_points)
    set_proxy_positions(scene, arm, proxies, proxy_offset)
    try:
        scene.sim.forward()
    except Exception:
        pass

    widths = [float(size_metrics(start_points)["width_x"])]
    nearest_values = []
    nearest_edge_values = []
    inside_counts = []
    last_nearest = None
    for phase_name, steps, action in (
        ("approach_open", int(approach_steps), approach_action),
        ("close_gripper", int(motion_params["close_duration"]), close_action),
        ("drag_inward", int(drag_steps), drag_action),
    ):
        for _ in range(int(steps)):
            centers = apply_action_step_with_proxies(scene, action, arm, proxies, proxy_offset)
            points = scene.clean_towel.points_numpy().astype(np.float32)
            widths.append(float(size_metrics(points)["width_x"]))
            nearest = nearest_proxy_to_particles(centers, proxy_radius, points, edge=edge, edge_band=edge_band)
            nearest["phase"] = phase_name
            nearest_values.append(float(nearest["surface_distance_m"]))
            nearest_edge_values.append(float(nearest["selected_edge_surface_distance_m"]))
            inside_counts.append(int(nearest.get("particles_inside_collider_count", 0)))
            last_nearest = nearest

    final_points = scene.clean_towel.points_numpy().astype(np.float32)
    widths_arr = np.asarray(widths, dtype=np.float32)
    edge_disp = edge_center_displacement(start_points, final_points, edge, edge_band)
    nearest_all = float(min(nearest_values)) if nearest_values else float("inf")
    nearest_edge_all = float(min(nearest_edge_values)) if nearest_edge_values else float("inf")
    result = {
        "arm": arm,
        "edge": edge,
        "motion_params": motion_params,
        "approach_action": [float(x) for x in approach_action],
        "close_action": [float(x) for x in close_action],
        "drag_action": [float(x) for x in drag_action],
        "proxy_radius": float(proxy_radius),
        "proxy_offset_xyz": [float(x) for x in proxy_offset],
        "proxy_paths": [proxy["primitive_root_path"] for proxy in proxies],
        "nearest_proxy_to_towel_particle_m": nearest_all,
        "nearest_proxy_to_selected_edge_particle_m": nearest_edge_all,
        "max_particles_inside_proxy_count": int(max(inside_counts)) if inside_counts else 0,
        "last_nearest_proxy": last_nearest,
        "selected_edge_particle_displacement": float(edge_disp),
        "start_width": float(widths_arr[0]),
        "best_width": float(widths_arr.min()),
        "final_width": float(widths_arr[-1]),
        "proxy_actual_touch_achieved": bool(nearest_all < 0.01),
        "proxy_moves_towel": bool(edge_disp > 0.02),
        "proxy_width_reduction_success": bool(float(widths_arr.min()) < 0.65),
        "particle_space_attachment_used": False,
    }
    return result


def proxy_success(summary: dict[str, Any]) -> bool:
    return bool(
        summary.get("proxy_actual_touch_achieved", False)
        and summary.get("proxy_moves_towel", False)
        and summary.get("proxy_width_reduction_success", False)
    )


def build_status(proxy_summary: dict[str, Any]) -> dict[str, Any]:
    primitive = load_json_if_exists(PRIMITIVE_SUMMARY_JSON) or {}
    offsets = load_json_if_exists(CONTACT_OFFSETS_SUMMARY_JSON) or {}
    primitive_baseline_success = bool(
        primitive.get("actual_touch_achieved", False)
        and primitive.get("primitive_moves_towel", False)
        and primitive.get("primitive_width_reduction_success", False)
    )
    offset_success = bool(offsets.get("physically_plausible_setting_found", False))
    clean_towel_can_move = bool(primitive_baseline_success or offset_success)
    contact_params_required = bool((not primitive_baseline_success) and offset_success)
    proxy_helped = proxy_success(proxy_summary)
    proxy_touched = bool(proxy_summary.get("proxy_actual_touch_achieved", False))
    proxy_moved = bool(proxy_summary.get("proxy_moves_towel", False))

    if not clean_towel_can_move:
        bottleneck = "clean_towel_cloth_collision_setup_or_primitive_contact_parameters"
    elif proxy_helped:
        bottleneck = "original_so101_jaw_geometry_or_collision_extent; proxy_fingertip_geometry_can_move_towel"
    elif not proxy_touched:
        bottleneck = "so101_jaw_pose_path_or_proxy_following_did_not_reach_towel_particles"
    elif not proxy_moved:
        bottleneck = "so101_drag_path_or_effective_fingertip_contact_not_transferring_motion"
    else:
        bottleneck = "contact_moves_some_towel_but_width_reduction_metric_not_met"

    return {
        "status": "CLEAN_TOWEL_V8_CONTACT_SANITY_DONE",
        "clean_towel_can_be_moved_by_primitive_collider": clean_towel_can_move,
        "baseline_primitive_contact_success": primitive_baseline_success,
        "contact_parameters_were_required": contact_params_required,
        "contact_offsets_physically_plausible_setting_found": offset_success,
        "best_contact_setting": offsets.get("best_contact_setting"),
        "so101_proxy_fingertip_colliders_helped": proxy_helped,
        "proxy_actual_touch_achieved": proxy_touched,
        "proxy_moves_towel": proxy_moved,
        "proxy_width_reduction_success": bool(proxy_summary.get("proxy_width_reduction_success", False)),
        "bottleneck_assessment": bottleneck,
        "ready_for_v9_policy_training": bool(clean_towel_can_move and proxy_helped),
        "ready_for_v9_policy_training_scope": (
            "proxy-fingertip contact policy only; original SO-101 jaw true-contact folding is not proven"
            if clean_towel_can_move and proxy_helped
            else "not ready"
        ),
        "true_robot_contact_folding_evidence_exists": False,
        "true_robot_contact_folding_evidence_basis": (
            "V8 uses primitive/proxy colliders for isolated contact sanity and does not prove autonomous folding "
            "with the original SO-101 jaw geometry."
        ),
        "do_not_train_in_v8": True,
        "paths": {
            "primitive_contact_summary": str(V8_DIR / PRIMITIVE_SUMMARY_JSON.name),
            "contact_offsets_sweep_summary": str(V8_DIR / CONTACT_OFFSETS_SUMMARY_JSON.name),
            "so101_proxy_fingertip_summary": str(V8_DIR / SO101_PROXY_SUMMARY_JSON.name),
            "status_json": str(V8_DIR / STATUS_JSON.name),
        },
        "honest_note": "V8 is isolated contact sanity only. No policy training or autonomous folding claim is made.",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V8 SO-101 proxy fingertip collider contact sanity test.")
    parser.add_argument("--task", type=str, default=TASK_ID)
    parser.add_argument("--usd_path", type=Path, default=CLEAN_TOWEL_USD_PATH)
    parser.add_argument("--out_dir", type=Path, default=V8_DIR)
    parser.add_argument("--edge_band", type=float, default=0.035)
    parser.add_argument("--approach_steps", type=int, default=80)
    parser.add_argument("--drag_steps", type=int, default=80)
    parser.add_argument("--settle_initial_steps", type=int, default=20)
    parser.add_argument("--proxy_count", type=int, default=2)
    parser.add_argument("--proxy_radius", type=float, default=0.035)
    parser.add_argument("--proxy_contact_offset", type=float, default=0.012)
    parser.add_argument("--proxy_rest_offset", type=float, default=0.0)
    parser.add_argument("--proxy_static_friction", type=float, default=1.6)
    parser.add_argument("--proxy_dynamic_friction", type=float, default=1.2)
    parser.add_argument("--proxy_x_offset", type=float, default=0.0)
    parser.add_argument("--proxy_y_offset", type=float, default=0.0)
    parser.add_argument("--proxy_z_offset", type=float, default=0.0)
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
        candidate = v6_recommended_candidate()
        if not candidate or not candidate.get("action_vector"):
            raise RuntimeError("Missing V6 recommended reachable candidate.")
        scene = create_standalone_so101_clean_towel_scene(
            args,
            clean_usd_path=args.usd_path,
            towel_translation=recommended_towel_translation(),
        )
        scene.step_zero(int(args.settle_initial_steps))
        proxies = spawn_proxy_spheres(
            count=int(args.proxy_count),
            radius=float(args.proxy_radius),
            contact_offset=float(args.proxy_contact_offset),
            rest_offset=float(args.proxy_rest_offset),
            static_friction=float(args.proxy_static_friction),
            dynamic_friction=float(args.proxy_dynamic_friction),
        )
        from isaacsim.core.utils.stage import update_stage

        update_stage()
        scene.sim.reset()
        scene.left_arm.update(scene.sim.get_physics_dt())
        scene.right_arm.update(scene.sim.get_physics_dt())
        scene.clean_towel = initialize_clean_towel_handle_for_sim(scene.sim, scene.clean_towel.root_path)
        scene.step_zero(int(args.settle_initial_steps))
        start_points = scene.clean_towel.points_numpy().astype(np.float32)
        motion_params = selected_motion_params(str(candidate["arm"]))
        proxy_offset = np.asarray(
            [float(args.proxy_x_offset), float(args.proxy_y_offset), float(args.proxy_z_offset)],
            dtype=np.float32,
        )
        result = run_proxy_trial(
            scene,
            start_points,
            candidate,
            motion_params,
            proxies,
            proxy_radius=float(args.proxy_radius),
            proxy_offset=proxy_offset,
            approach_steps=int(args.approach_steps),
            drag_steps=int(args.drag_steps),
            edge_band=float(args.edge_band),
        )
        summary = {
            "status": "SO101_PROXY_FINGERTIP_V8_DONE",
            "command_run": command_run_string(),
            "v6_recommended_towel_center": recommended_towel_center(),
            "towel_translation_used": [float(x) for x in recommended_towel_translation()],
            "v6_recommended_candidate": candidate,
            "scene_info": standalone_robot_scene_info(scene),
            "start_towel_metrics": size_metrics(start_points),
            "proxy_colliders": proxies,
            "result": result,
            "nearest_proxy_to_towel_particle_m": result["nearest_proxy_to_towel_particle_m"],
            "selected_edge_particle_displacement": result["selected_edge_particle_displacement"],
            "start_width": result["start_width"],
            "best_width": result["best_width"],
            "final_width": result["final_width"],
            "proxy_actual_touch_achieved": result["proxy_actual_touch_achieved"],
            "proxy_moves_towel": result["proxy_moves_towel"],
            "proxy_width_reduction_success": result["proxy_width_reduction_success"],
            "particle_space_attachment_used": False,
            "so101_proxy_fingertip_summary_path": str(args.out_dir / SO101_PROXY_SUMMARY_JSON.name),
            "honest_note": (
                "Proxy fingertip colliders follow SO-101 jaw/gripper body poses and never directly move towel particles. "
                "This is proxy-geometry contact sanity, not true original-jaw contact evidence or policy training."
            ),
        }
        status = build_status(summary)
        write_json(args.out_dir / SO101_PROXY_SUMMARY_JSON.name, summary, print_payload=False)
        write_json(args.out_dir / STATUS_JSON.name, status, print_payload=False)
        print(json.dumps({"so101_proxy_fingertip_summary": summary, "v8_status": status}, indent=2), flush=True)
    except Exception as exc:
        summary = {
            "status": "SO101_PROXY_FINGERTIP_V8_FAILED",
            "command_run": command_run_string(),
            "exception_type": exc.__class__.__name__,
            "exception": repr(exc),
            "proxy_actual_touch_achieved": False,
            "proxy_moves_towel": False,
            "proxy_width_reduction_success": False,
            "particle_space_attachment_used": False,
            "honest_note": "Proxy fingertip test failed; no contact, folding, or training claim is made.",
        }
        status = build_status(summary)
        write_json(args.out_dir / SO101_PROXY_SUMMARY_JSON.name, summary, print_payload=False)
        write_json(args.out_dir / STATUS_JSON.name, status, print_payload=False)
        print(json.dumps({"so101_proxy_fingertip_summary": summary, "v8_status": status}, indent=2), flush=True)
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
