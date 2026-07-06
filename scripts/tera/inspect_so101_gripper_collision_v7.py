from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from isaaclab.app import AppLauncher

from clean_towel_v4_common import apply_action_step
from clean_towel_v5_contact_common import towel_edge_masks
from so101_clean_towel_scene_utils_v1 import (
    CLEAN_TOWEL_USD_PATH,
    TASK_ID,
    command_run_string,
    create_standalone_so101_clean_towel_scene,
    force_headless_no_cameras,
    points_to_numpy,
    size_metrics,
    standalone_robot_scene_info,
    write_json,
)


V7_DIR = Path("/workspace/leisaac/tera_checkpoints/clean_towel_v7_contact_mechanics")
V6_DIR = Path("/workspace/leisaac/tera_checkpoints/clean_towel_v6_reach")
V6_STATUS_JSON = V6_DIR / "STATUS.json"
V6_PLACEMENT_JSON = V6_DIR / "towel_placement_search_summary.json"
V6_REACH_JSON = V6_DIR / "reach_search_results.json"

GRIPPER_COLLISION_JSON = V7_DIR / "gripper_collision_inspection_summary.json"
GRIPPER_MAPPING_JSON = V7_DIR / "gripper_open_close_mapping_summary.json"
EDGE_PINCH_DRAG_SUMMARY_JSON = V7_DIR / "edge_pinch_drag_sweep_summary.json"
EDGE_PINCH_DRAG_RESULTS_JSON = V7_DIR / "edge_pinch_drag_sweep_results.json"
CONTACT_PARAMETER_SUMMARY_JSON = V7_DIR / "contact_parameter_sweep_summary.json"
CONTACT_PARAMETER_RESULTS_JSON = V7_DIR / "contact_parameter_sweep_results.json"
LOCAL_ATTACHMENT_JSON = V7_DIR / "local_gripper_attachment_summary.json"
STATUS_JSON = V7_DIR / "STATUS.json"

ARM_NAMES = ["left", "right"]
EDGE_NAMES = ["left", "right", "front", "back"]
DEFAULT_RECOMMENDED_TOWEL_CENTER = [0.20217525959014893, -0.5512138605117798, 0.0030000002589076757]
DEFAULT_STANDALONE_TOWEL_Z_TRANSLATION = 0.02
GRIPPER_BODY_KEYWORDS = ("jaw", "finger", "gripper", "tool", "end")


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def recommended_towel_center() -> list[float]:
    placement = load_json_if_exists(V6_PLACEMENT_JSON)
    status = load_json_if_exists(V6_STATUS_JSON)
    for payload in (placement, status):
        if payload and payload.get("recommended_towel_center"):
            return [float(x) for x in payload["recommended_towel_center"]]
    return [float(x) for x in DEFAULT_RECOMMENDED_TOWEL_CENTER]


def recommended_towel_translation() -> tuple[float, float, float]:
    center = recommended_towel_center()
    return (float(center[0]), float(center[1]), float(DEFAULT_STANDALONE_TOWEL_Z_TRANSLATION))


def v6_recommended_candidate() -> dict[str, Any] | None:
    placement = load_json_if_exists(V6_PLACEMENT_JSON)
    if placement and placement.get("recommended_candidate"):
        candidate = dict(placement["recommended_candidate"])
        candidate["selection_source"] = "v6_recommended_placement"
        return candidate
    reach = load_json_if_exists(V6_REACH_JSON)
    if reach and reach.get("global_best"):
        candidate = dict(reach["global_best"])
        candidate["selection_source"] = "v6_reach_global_best"
        return candidate
    return None


def arm_object(scene: Any, arm: str) -> Any:
    return scene.left_arm if arm == "left" else scene.right_arm


def active_slice(arm: str) -> slice:
    return slice(0, 6) if arm == "left" else slice(6, 12)


def gripper_index(arm: str) -> int:
    return 5 if arm == "left" else 11


def tensor_to_numpy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        if hasattr(value, "detach") and callable(value.detach):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=np.float32)
    except Exception:
        return None


def body_names_for_arm(arm_obj: Any) -> list[str]:
    try:
        return [str(x) for x in (getattr(arm_obj.data, "body_names", []) or [])]
    except Exception:
        return []


def candidate_body_indices(arm_obj: Any) -> list[int]:
    names = body_names_for_arm(arm_obj)
    indices = [
        idx
        for idx, name in enumerate(names)
        if any(keyword in name.lower() for keyword in GRIPPER_BODY_KEYWORDS)
    ]
    if not indices and names:
        indices = [len(names) - 1]
    return indices


def body_pose(arm_obj: Any, body_index: int) -> dict[str, Any]:
    names = body_names_for_arm(arm_obj)
    pos = tensor_to_numpy(getattr(getattr(arm_obj, "data", None), "body_pos_w", None))
    quat = tensor_to_numpy(getattr(getattr(arm_obj, "data", None), "body_quat_w", None))
    out = {
        "body_index": int(body_index),
        "body_name": names[body_index] if body_index < len(names) else f"body_{body_index}",
        "position": None,
        "orientation_xyzw": None,
    }
    if pos is not None and pos.size >= 3:
        flat = pos.reshape(-1, 3)
        if body_index < flat.shape[0] and np.isfinite(flat[body_index]).all():
            out["position"] = [float(x) for x in flat[body_index]]
    if quat is not None and quat.size >= 4:
        flat_q = quat.reshape(-1, 4)
        if body_index < flat_q.shape[0] and np.isfinite(flat_q[body_index]).all():
            out["orientation_xyzw"] = [float(x) for x in flat_q[body_index]]
    return out


def candidate_body_poses(scene: Any, arm: str) -> list[dict[str, Any]]:
    arm_obj = arm_object(scene, arm)
    return [body_pose(arm_obj, idx) for idx in candidate_body_indices(arm_obj)]


def nearest_body_to_particles(body_poses: list[dict[str, Any]], particles: np.ndarray) -> dict[str, Any]:
    pts = points_to_numpy(particles)
    best = None
    for pose in body_poses:
        if pose.get("position") is None:
            continue
        pos = np.asarray(pose["position"], dtype=np.float32).reshape(1, 3)
        distances = np.linalg.norm(pts - pos, axis=1)
        index = int(np.argmin(distances))
        item = {
            "body_name": pose["body_name"],
            "body_index": int(pose["body_index"]),
            "nearest_particle_index": index,
            "nearest_particle_position": [float(x) for x in pts[index]],
            "distance_m": float(distances[index]),
        }
        if best is None or item["distance_m"] < best["distance_m"]:
            best = item
    return best or {
        "body_name": None,
        "body_index": None,
        "nearest_particle_index": None,
        "nearest_particle_position": None,
        "distance_m": float("inf"),
    }


def nearest_body_to_edge_particles(
    body_poses: list[dict[str, Any]],
    particles: np.ndarray,
    edge: str,
    edge_band: float,
) -> dict[str, Any]:
    pts = points_to_numpy(particles)
    masks = towel_edge_masks(pts, edge_band)
    edge_pts = pts[masks[edge]]
    if edge_pts.size == 0:
        return {
            "body_name": None,
            "body_index": None,
            "nearest_particle_index": None,
            "nearest_particle_position": None,
            "distance_m": float("inf"),
        }
    best = None
    edge_indices = np.nonzero(masks[edge])[0]
    for pose in body_poses:
        if pose.get("position") is None:
            continue
        pos = np.asarray(pose["position"], dtype=np.float32).reshape(1, 3)
        distances = np.linalg.norm(edge_pts - pos, axis=1)
        local_index = int(np.argmin(distances))
        particle_index = int(edge_indices[local_index])
        item = {
            "body_name": pose["body_name"],
            "body_index": int(pose["body_index"]),
            "nearest_particle_index": particle_index,
            "nearest_particle_position": [float(x) for x in pts[particle_index]],
            "distance_m": float(distances[local_index]),
        }
        if best is None or item["distance_m"] < best["distance_m"]:
            best = item
    return best or {
        "body_name": None,
        "body_index": None,
        "nearest_particle_index": None,
        "nearest_particle_position": None,
        "distance_m": float("inf"),
    }


def clipped_action(action: np.ndarray) -> np.ndarray:
    out = np.asarray(action, dtype=np.float32).reshape(12).copy()
    out[:5] = np.clip(out[:5], -1.85, 1.85)
    out[6:11] = np.clip(out[6:11], -1.85, 1.85)
    out[5] = float(np.clip(out[5], -0.05, 0.90))
    out[11] = float(np.clip(out[11], -0.05, 0.90))
    return out


def apply_action_for_steps(scene: Any, action: np.ndarray, steps: int) -> None:
    action = clipped_action(action)
    for _ in range(int(steps)):
        apply_action_step(scene, action)


def all_gripper_body_poses(scene: Any) -> dict[str, list[dict[str, Any]]]:
    return {arm: candidate_body_poses(scene, arm) for arm in ARM_NAMES}


def find_matching_body_prim_paths(stage: Any, robot_root: str, body_name: str) -> list[str]:
    root = stage.GetPrimAtPath(robot_root)
    if not root or not root.IsValid():
        return []
    target = body_name.lower()
    matches = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith(robot_root + "/"):
            continue
        if prim.GetName().lower() == target:
            matches.append(path)
    return matches


def attr_value_json(attr: Any) -> Any:
    try:
        value = attr.Get()
    except Exception:
        return None
    if hasattr(value, "real"):
        try:
            return float(value)
        except Exception:
            pass
    if isinstance(value, (int, float, bool, str)):
        return value
    if value is None:
        return None
    try:
        return [float(x) for x in value]
    except Exception:
        return str(value)


def interesting_physics_attrs(prim: Any) -> dict[str, Any]:
    out = {}
    keywords = (
        "collision",
        "contact",
        "friction",
        "restitution",
        "offset",
        "radius",
        "thickness",
        "material",
        "rest",
    )
    try:
        attrs = prim.GetAttributes()
    except Exception:
        attrs = []
    for attr in attrs:
        name = attr.GetName()
        if any(keyword in name.lower() for keyword in keywords):
            out[name] = attr_value_json(attr)
    return out


def collision_prim_info(stage: Any, prim_path: str) -> list[dict[str, Any]]:
    from pxr import UsdPhysics

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return []
    infos = []
    for child in stage.Traverse():
        path = str(child.GetPath())
        if path != prim_path and not path.startswith(prim_path + "/"):
            continue
        has_collision_api = bool(child.HasAPI(UsdPhysics.CollisionAPI))
        type_name = child.GetTypeName()
        name_lower = child.GetName().lower()
        type_is_collision_like = type_name in {"Mesh", "Cube", "Sphere", "Capsule", "Cylinder"}
        name_is_collision_like = "collis" in name_lower or "collider" in name_lower
        if not (has_collision_api or type_is_collision_like or name_is_collision_like):
            continue
        enabled = None
        if has_collision_api:
            try:
                enabled = UsdPhysics.CollisionAPI(child).GetCollisionEnabledAttr().Get()
            except Exception:
                enabled = None
        material_binding = None
        try:
            from pxr import UsdShade

            material, _ = UsdShade.MaterialBindingAPI(child).ComputeBoundMaterial()
            if material:
                material_binding = str(material.GetPrim().GetPath())
        except Exception:
            material_binding = None
        infos.append(
            {
                "path": path,
                "type_name": str(type_name),
                "has_usdphysics_collision_api": has_collision_api,
                "collision_enabled": enabled,
                "bound_material_path": material_binding,
                "physics_attrs": interesting_physics_attrs(child),
            }
        )
    return infos


def gripper_collision_report(scene: Any) -> dict[str, Any]:
    from isaacsim.core.utils.stage import get_current_stage

    stage = get_current_stage()
    report = {}
    for arm in ARM_NAMES:
        robot_root = scene.left_prim_path if arm == "left" else scene.right_prim_path
        report[arm] = []
        arm_obj = arm_object(scene, arm)
        for idx in candidate_body_indices(arm_obj):
            name = body_names_for_arm(arm_obj)[idx]
            prim_paths = find_matching_body_prim_paths(stage, robot_root, name)
            collision_infos = []
            for prim_path in prim_paths:
                collision_infos.extend(collision_prim_info(stage, prim_path))
            report[arm].append(
                {
                    "body_index": int(idx),
                    "body_name": name,
                    "body_prim_paths": prim_paths,
                    "collision_prims": collision_infos,
                    "collision_geometry_found": bool(
                        any(item.get("has_usdphysics_collision_api") for item in collision_infos)
                    ),
                    "collision_like_geometry_count": int(len(collision_infos)),
                }
            )
    return report


def towel_physics_report(scene: Any) -> dict[str, Any]:
    from isaacsim.core.utils.stage import get_current_stage

    stage = get_current_stage()
    paths = [scene.clean_towel.root_path, scene.clean_towel.mesh_path, scene.clean_towel.particle_system_path]
    out = {}
    for path in paths:
        prim = stage.GetPrimAtPath(path)
        out[path] = {
            "valid": bool(prim and prim.IsValid()),
            "type_name": None if not prim or not prim.IsValid() else str(prim.GetTypeName()),
            "physics_attrs": {} if not prim or not prim.IsValid() else interesting_physics_attrs(prim),
        }
    return out


def pose_distance_report(scene: Any, selected_edge: str, edge_band: float) -> dict[str, Any]:
    points = scene.clean_towel.points_numpy().astype(np.float32)
    all_poses = all_gripper_body_poses(scene)
    per_arm = {}
    all_pose_list = []
    for arm, poses in all_poses.items():
        all_pose_list.extend([{**pose, "arm": arm} for pose in poses])
        per_arm[arm] = {
            "body_poses": poses,
            "nearest_gripper_body_to_any_towel_particle": nearest_body_to_particles(poses, points),
            "nearest_gripper_body_to_selected_edge_particles": nearest_body_to_edge_particles(
                poses, points, selected_edge, edge_band
            ),
        }
    nearest_any = nearest_body_to_particles(all_pose_list, points)
    nearest_edge = nearest_body_to_edge_particles(all_pose_list, points, selected_edge, edge_band)
    return {
        "selected_edge": selected_edge,
        "edge_band": float(edge_band),
        "per_arm": per_arm,
        "nearest_gripper_body_to_any_towel_particle": nearest_any,
        "nearest_gripper_body_to_selected_edge_particles": nearest_edge,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect SO-101 gripper collision and nearest-particle contact geometry.")
    parser.add_argument("--task", type=str, default=TASK_ID)
    parser.add_argument("--usd_path", type=Path, default=CLEAN_TOWEL_USD_PATH)
    parser.add_argument("--out_dir", type=Path, default=V7_DIR)
    parser.add_argument("--settle_steps", type=int, default=20)
    parser.add_argument("--action_steps", type=int, default=30)
    parser.add_argument("--edge_band", type=float, default=0.035)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    force_headless_no_cameras(args)
    return args


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    _ = simulation_app
    try:
        candidate = v6_recommended_candidate()
        selected_edge = str(candidate.get("edge", "right")) if candidate else "right"
        scene = create_standalone_so101_clean_towel_scene(
            args,
            clean_usd_path=args.usd_path,
            towel_translation=recommended_towel_translation(),
        )
        scene.step_zero(args.settle_steps)
        neutral_report = pose_distance_report(scene, selected_edge, args.edge_band)
        collision_report = gripper_collision_report(scene)
        towel_report = towel_physics_report(scene)

        action = None
        if candidate and candidate.get("action_vector"):
            action = clipped_action(np.asarray(candidate["action_vector"], dtype=np.float32))
            apply_action_for_steps(scene, action, args.action_steps)
        reachable_report = pose_distance_report(scene, selected_edge, args.edge_band)
        jaw_collision_bodies_found = bool(
            any(
                body.get("collision_geometry_found")
                for arm_items in collision_report.values()
                for body in arm_items
                if any(keyword in body["body_name"].lower() for keyword in ("jaw", "finger", "gripper"))
            )
        )
        finger_jaw_names = {
            arm: [
                item["body_name"]
                for item in collision_report[arm]
                if any(keyword in item["body_name"].lower() for keyword in ("jaw", "finger", "gripper"))
            ]
            for arm in ARM_NAMES
        }
        nearest = reachable_report["nearest_gripper_body_to_any_towel_particle"]["distance_m"]
        summary = {
            "status": "SO101_GRIPPER_COLLISION_INSPECTION_V7_DONE",
            "command_run": command_run_string(),
            "v6_recommended_towel_center": recommended_towel_center(),
            "towel_translation_used": [float(x) for x in recommended_towel_translation()],
            "v6_recommended_candidate": candidate,
            "left_arm_body_names": body_names_for_arm(scene.left_arm),
            "right_arm_body_names": body_names_for_arm(scene.right_arm),
            "candidate_gripper_jaw_finger_bodies": {
                arm: [body["body_name"] for body in collision_report[arm]] for arm in ARM_NAMES
            },
            "finger_jaw_body_names_identified": finger_jaw_names,
            "jaw_collision_bodies_found": jaw_collision_bodies_found,
            "collision_report": collision_report,
            "towel_physics_report": towel_report,
            "neutral_pose_distance_report": neutral_report,
            "v6_reachable_action_pose_distance_report": reachable_report,
            "nearest_jaw_to_towel_particle_m": float(nearest),
            "nearest_jaw_to_selected_edge_particle_m": float(
                reachable_report["nearest_gripper_body_to_selected_edge_particles"]["distance_m"]
            ),
            "towel_metrics": size_metrics(scene.clean_towel.points_numpy()),
            "scene_info": standalone_robot_scene_info(scene),
            "gripper_collision_inspection_summary_path": str(args.out_dir / GRIPPER_COLLISION_JSON.name),
            "honest_note": (
                "This inspects runtime collision and geometry only. Body-origin distance is a proxy; "
                "it does not prove contact, towel motion, folding, or train any policy."
            ),
        }
        write_json(args.out_dir / GRIPPER_COLLISION_JSON.name, summary, print_payload=True)
    except Exception as exc:
        summary = {
            "status": "SO101_GRIPPER_COLLISION_INSPECTION_V7_FAILED",
            "command_run": command_run_string(),
            "exception_type": exc.__class__.__name__,
            "exception": repr(exc),
            "jaw_collision_bodies_found": False,
            "nearest_jaw_to_towel_particle_m": None,
            "honest_note": "Inspection failed; no contact, towel motion, folding, or training claim is made.",
        }
        write_json(args.out_dir / GRIPPER_COLLISION_JSON.name, summary, print_payload=True)
    finally:
        import os
        import sys

        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
