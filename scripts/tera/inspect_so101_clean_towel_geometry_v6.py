from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from isaaclab.app import AppLauncher

from clean_towel_v5_contact_common import edge_summary
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


V6_DIR = Path("/workspace/leisaac/tera_checkpoints/clean_towel_v6_reach")
GEOMETRY_JSON = V6_DIR / "geometry_summary.json"
REACH_JSON = V6_DIR / "reach_search_results.json"
REACH_NPZ = V6_DIR / "reach_search_results.npz"
PLACEMENT_JSON = V6_DIR / "towel_placement_search_summary.json"
CONTACT_JSON = V6_DIR / "contact_after_reach_fix_summary.json"
STATUS_JSON = V6_DIR / "STATUS.json"

EDGE_NAMES = ["left", "right", "front", "back"]
ARM_NAMES = ["left", "right"]
DEFAULT_STANDALONE_TOWEL_TRANSLATION = (0.0, 0.0, 0.02)


def tensor_to_numpy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        if hasattr(value, "detach") and callable(value.detach):
            value = value.detach().cpu().numpy()
        arr = np.asarray(value, dtype=np.float32)
        return arr
    except Exception:
        return None


def finite_float_list(value: Any) -> list[float] | None:
    arr = tensor_to_numpy(value)
    if arr is None:
        return None
    flat = arr.reshape(-1)
    if flat.size < 3 or not np.isfinite(flat[:3]).all():
        return None
    return [float(x) for x in flat[:3]]


def body_names_for_arm(arm: Any) -> list[str]:
    try:
        return [str(x) for x in (getattr(arm.data, "body_names", []) or [])]
    except Exception:
        return []


def joint_names_for_arm(arm: Any) -> list[str]:
    try:
        return [str(x) for x in (getattr(arm.data, "joint_names", []) or [])]
    except Exception:
        return []


def base_position_for_arm(arm: Any) -> dict[str, Any]:
    for attr in ("root_pos_w", "root_state_w"):
        arr = tensor_to_numpy(getattr(getattr(arm, "data", None), attr, None))
        if arr is None or arr.size < 3:
            continue
        if attr == "root_state_w":
            arr = arr.reshape(-1, arr.shape[-1])[:, :3]
        pos = arr.reshape(-1, 3)[0]
        if np.isfinite(pos).all():
            return {"available": True, "source": attr, "position": [float(x) for x in pos]}

    arr = tensor_to_numpy(getattr(getattr(arm, "data", None), "body_pos_w", None))
    if arr is not None and arr.size >= 3:
        pos = arr.reshape(-1, 3)[0]
        if np.isfinite(pos).all():
            return {"available": True, "source": "body_pos_w[0]", "position": [float(x) for x in pos]}

    return {"available": False, "source": None, "position": None}


def jaw_info_for_arm(arm: Any) -> dict[str, Any]:
    names = body_names_for_arm(arm)
    lower = [name.lower() for name in names]
    preferred = []
    for idx, name in enumerate(lower):
        if name == "jaw" or name.endswith("/jaw"):
            preferred.append(idx)
    candidates = preferred or [
        idx
        for idx, name in enumerate(lower)
        if "jaw" in name or "gripper" in name or "tool" in name or "end" in name
    ]
    index = int(candidates[-1]) if candidates else max(0, len(names) - 1)
    arr = tensor_to_numpy(getattr(getattr(arm, "data", None), "body_pos_w", None))
    if arr is None or arr.size < 3:
        return {
            "available": False,
            "body_index": None,
            "body_name": "unavailable",
            "position": None,
            "selection_rule": "body_pos_w_unavailable",
        }
    arr = arr.reshape(-1, 3)
    if index >= arr.shape[0]:
        index = max(0, arr.shape[0] - 1)
    pos = arr[index]
    available = bool(np.isfinite(pos).all())
    body_name = names[index] if index < len(names) else f"body_{index}"
    return {
        "available": available,
        "body_index": int(index),
        "body_name": body_name,
        "position": [float(x) for x in pos] if available else None,
        "selection_rule": "last_matching_jaw_gripper_tool_end_body",
    }


def jaw_positions_by_arm(scene: Any) -> dict[str, np.ndarray]:
    return {
        "left": np.asarray(jaw_info_for_arm(scene.left_arm)["position"], dtype=np.float32),
        "right": np.asarray(jaw_info_for_arm(scene.right_arm)["position"], dtype=np.float32),
    }


def edge_centers_from_metrics(metrics: dict[str, Any]) -> dict[str, list[float]]:
    cmin = np.asarray(metrics["min"], dtype=np.float32)
    cmax = np.asarray(metrics["max"], dtype=np.float32)
    center = np.asarray(metrics["center"], dtype=np.float32)
    return {
        "left": [float(cmin[0]), float(center[1]), float(center[2])],
        "right": [float(cmax[0]), float(center[1]), float(center[2])],
        "front": [float(center[0]), float(cmin[1]), float(center[2])],
        "back": [float(center[0]), float(cmax[1]), float(center[2])],
    }


def edge_centers_array(edge_centers: dict[str, Any]) -> np.ndarray:
    return np.asarray([edge_centers[name] for name in EDGE_NAMES], dtype=np.float32)


def distance_to_edge_centers(jaw_position: Any, edge_centers: dict[str, Any]) -> dict[str, float]:
    jaw = np.asarray(jaw_position, dtype=np.float32).reshape(3)
    distances = {}
    for name in EDGE_NAMES:
        center = np.asarray(edge_centers[name], dtype=np.float32).reshape(3)
        distances[name] = float(np.linalg.norm(jaw - center))
    return distances


def geometry_payload(scene: Any, settle_steps: int, command: str) -> dict[str, Any]:
    points = scene.clean_towel.points_numpy().astype(np.float32)
    metrics = size_metrics(points)
    exact_edge_centers = edge_centers_from_metrics(metrics)
    left_jaw = jaw_info_for_arm(scene.left_arm)
    right_jaw = jaw_info_for_arm(scene.right_arm)
    jaw_edge_distances = {
        "left": distance_to_edge_centers(left_jaw["position"], exact_edge_centers) if left_jaw["position"] else {},
        "right": distance_to_edge_centers(right_jaw["position"], exact_edge_centers) if right_jaw["position"] else {},
    }
    all_distances = [
        distance
        for by_edge in jaw_edge_distances.values()
        for distance in by_edge.values()
        if np.isfinite(float(distance))
    ]
    nearest_distance = min(all_distances) if all_distances else float("inf")
    return {
        "status": "SO101_CLEAN_TOWEL_GEOMETRY_V6_DONE",
        "command_run": command,
        "settle_steps": int(settle_steps),
        "clean_towel_root_path": scene.clean_towel.root_path,
        "clean_towel_mesh_path": scene.clean_towel.mesh_path,
        "clean_towel_particle_system_path": scene.clean_towel.particle_system_path,
        "left_robot_prim_path": scene.left_prim_path,
        "right_robot_prim_path": scene.right_prim_path,
        "left_arm_body_names": body_names_for_arm(scene.left_arm),
        "right_arm_body_names": body_names_for_arm(scene.right_arm),
        "left_arm_joint_names": joint_names_for_arm(scene.left_arm),
        "right_arm_joint_names": joint_names_for_arm(scene.right_arm),
        "left_base_position": base_position_for_arm(scene.left_arm),
        "right_base_position": base_position_for_arm(scene.right_arm),
        "left_jaw_neutral": left_jaw,
        "right_jaw_neutral": right_jaw,
        "towel_metrics": metrics,
        "towel_exact_edge_centers": exact_edge_centers,
        "towel_particle_edge_band_summary": edge_summary(points, band=0.035),
        "jaw_to_exact_edge_center_distances_m": jaw_edge_distances,
        "nearest_jaw_to_exact_edge_center_m": float(nearest_distance),
        "scene_info": standalone_robot_scene_info(scene),
        "geometry_summary_json_path": str(GEOMETRY_JSON),
        "honest_note": (
            "This is a geometry diagnostic only. Distances are from neutral SO-101 jaw body positions "
            "to exact clean towel bounding-box edge centers; no contact, folding, or training is claimed."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect SO-101 + clean towel geometry for V6 reach diagnostics.")
    parser.add_argument("--task", type=str, default=TASK_ID)
    parser.add_argument("--usd_path", type=Path, default=CLEAN_TOWEL_USD_PATH)
    parser.add_argument("--out_dir", type=Path, default=V6_DIR)
    parser.add_argument("--settle_steps", type=int, default=20)
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
        scene = create_standalone_so101_clean_towel_scene(args, clean_usd_path=args.usd_path)
        scene.step_zero(args.settle_steps)
        summary = geometry_payload(scene, args.settle_steps, command_run_string())
        write_json(args.out_dir / GEOMETRY_JSON.name, summary, print_payload=True)
    except Exception as exc:
        summary = {
            "status": "SO101_CLEAN_TOWEL_GEOMETRY_V6_FAILED",
            "command_run": command_run_string(),
            "exception_type": exc.__class__.__name__,
            "exception": repr(exc),
            "geometry_summary_json_path": str(args.out_dir / GEOMETRY_JSON.name),
            "honest_note": "Geometry inspection failed; no reach, contact, folding, or training claim is made.",
        }
        write_json(args.out_dir / GEOMETRY_JSON.name, summary, print_payload=True)
    finally:
        import os
        import sys

        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
