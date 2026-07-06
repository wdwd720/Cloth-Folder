from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from isaaclab.app import AppLauncher

from clean_towel_v4_common import apply_action_step, reset_scene_to_points
from clean_towel_v5_contact_common import CONTACT_SWEEP_JSON
from inspect_so101_clean_towel_geometry_v6 import (
    ARM_NAMES,
    EDGE_NAMES,
    REACH_JSON,
    REACH_NPZ,
    V6_DIR,
    distance_to_edge_centers,
    edge_centers_array,
    edge_centers_from_metrics,
    jaw_info_for_arm,
    tensor_to_numpy,
)
from so101_clean_towel_scene_utils_v1 import (
    CLEAN_TOWEL_USD_PATH,
    TASK_ID,
    command_run_string,
    create_standalone_so101_clean_towel_scene,
    force_headless_no_cameras,
    size_metrics,
    standalone_robot_scene_info,
    write_json,
)


FALLBACK_JOINT_RANGES = np.asarray(
    [
        [-1.60, 1.60],
        [-1.45, 1.25],
        [-1.45, 1.80],
        [-1.65, 1.65],
        [-1.65, 1.65],
        [0.50, 0.80],
    ],
    dtype=np.float32,
)


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def arm_object(scene: Any, arm: str) -> Any:
    return scene.left_arm if arm == "left" else scene.right_arm


def active_slice(arm: str) -> slice:
    return slice(0, 6) if arm == "left" else slice(6, 12)


def default_action() -> np.ndarray:
    return np.zeros(12, dtype=np.float32)


def sanitize_joint_ranges(ranges: np.ndarray | None) -> tuple[np.ndarray, str]:
    fallback = FALLBACK_JOINT_RANGES.copy()
    if ranges is None:
        return fallback, "fallback_ranges"
    arr = np.asarray(ranges, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.shape == (2, 6):
        arr = arr.T
    if arr.shape != (6, 2):
        return fallback, f"fallback_ranges_invalid_shape_{arr.shape}"

    out = fallback.copy()
    source = "articulation_joint_limits"
    for idx in range(6):
        low = float(arr[idx, 0])
        high = float(arr[idx, 1])
        if not np.isfinite(low) or not np.isfinite(high) or high <= low or high - low > 8.0:
            source = f"{source}_partially_sanitized"
            continue
        out[idx] = [low, high]

    # Reach candidates should keep the gripper open enough to avoid measuring a closed fingertip pose.
    gripper_low = max(float(out[5, 0]), 0.45)
    gripper_high = min(float(out[5, 1]), 0.85)
    if gripper_high <= gripper_low:
        gripper_low, gripper_high = float(fallback[5, 0]), float(fallback[5, 1])
    out[5] = [gripper_low, gripper_high]
    return out.astype(np.float32), source


def joint_ranges_for_arm(arm: Any) -> tuple[np.ndarray, str]:
    data = getattr(arm, "data", None)
    for attr in ("soft_joint_pos_limits", "joint_pos_limits", "joint_limits"):
        arr = tensor_to_numpy(getattr(data, attr, None))
        ranges, source = sanitize_joint_ranges(arr)
        if not source.startswith("fallback_ranges"):
            return ranges, f"{attr}:{source}"
    return FALLBACK_JOINT_RANGES.copy(), "fallback_ranges_no_articulation_limits"


def clip_active_to_ranges(active: np.ndarray, ranges: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(active, dtype=np.float32), ranges[:, 0], ranges[:, 1]).astype(np.float32)


def load_v5_warm_actions(arm: str, ranges: np.ndarray) -> list[np.ndarray]:
    sweep = load_json_if_exists(CONTACT_SWEEP_JSON)
    if not sweep:
        return []
    candidates = []
    for key in ("best_candidate",):
        if sweep.get(key):
            candidates.append(sweep[key])
    candidates.extend(sweep.get("top_candidates", [])[:12])
    out = []
    sl = active_slice(arm)
    for candidate in candidates:
        for segment in candidate.get("action_sequence", []):
            action = np.asarray(segment.get("action", []), dtype=np.float32)
            if action.shape != (12,):
                continue
            active = clip_active_to_ranges(action[sl], ranges)
            active[5] = max(active[5], float(ranges[5, 0]))
            full = default_action()
            full[sl] = active
            out.append(full)
    return out


def synthetic_warm_actions(arm: str, ranges: np.ndarray) -> list[np.ndarray]:
    from clean_towel_v4_common import synthetic_action

    out = []
    sl = active_slice(arm)
    for progress in np.linspace(0.0, 1.0, 9, dtype=np.float32):
        for phase_id in (0, 1, 2):
            action = synthetic_action(float(progress), int(phase_id))
            full = default_action()
            full[sl] = clip_active_to_ranges(action[sl], ranges)
            full[sl][5] = max(full[sl][5], float(ranges[5, 0]))
            out.append(full.astype(np.float32))
    neutral = default_action()
    neutral[sl] = clip_active_to_ranges(np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, ranges[5, 1]], dtype=np.float32), ranges)
    out.append(neutral)
    return out


def latin_hypercube(rng: np.random.Generator, count: int, dim: int) -> np.ndarray:
    samples = np.empty((int(count), int(dim)), dtype=np.float32)
    base = np.arange(int(count), dtype=np.float32)
    for dim_idx in range(int(dim)):
        column = (base + rng.random(int(count), dtype=np.float32)) / float(max(1, int(count)))
        rng.shuffle(column)
        samples[:, dim_idx] = column
    return samples


def generate_candidate_actions(arm: str, count: int, ranges: np.ndarray, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    count = max(300, int(count))
    rng = np.random.default_rng(int(seed) + (0 if arm == "left" else 1009))
    sl = active_slice(arm)
    warm = synthetic_warm_actions(arm, ranges) + load_v5_warm_actions(arm, ranges)
    unique = []
    seen = set()
    for action in warm:
        key = tuple(np.round(action, 4).tolist())
        if key in seen:
            continue
        seen.add(key)
        unique.append(action.astype(np.float32))
    random_needed = max(0, count - len(unique))
    lhs = latin_hypercube(rng, random_needed, 6) if random_needed else np.zeros((0, 6), dtype=np.float32)
    random_actions = []
    if random_needed:
        active = ranges[:, 0] + lhs * (ranges[:, 1] - ranges[:, 0])
        gripper_values = np.linspace(float(ranges[5, 0]), float(ranges[5, 1]), 3, dtype=np.float32)
        rng.shuffle(gripper_values)
        active[:, 5] = gripper_values[np.arange(random_needed) % len(gripper_values)]
        for row in active:
            full = default_action()
            full[sl] = clip_active_to_ranges(row, ranges)
            random_actions.append(full.astype(np.float32))
    actions = np.asarray((unique + random_actions)[:count], dtype=np.float32)
    metadata = {
        "requested_candidate_count": int(count),
        "warm_candidate_count": int(min(len(unique), count)),
        "random_latin_hypercube_candidate_count": int(max(0, len(actions) - min(len(unique), count))),
        "seed": int(seed),
    }
    return actions, metadata


def measure_active_jaw(scene: Any, arm: str) -> tuple[np.ndarray, dict[str, Any]]:
    info = jaw_info_for_arm(arm_object(scene, arm))
    if info.get("position") is None:
        return np.asarray([np.nan, np.nan, np.nan], dtype=np.float32), info
    return np.asarray(info["position"], dtype=np.float32).reshape(3), info


def candidate_result(
    arm: str,
    candidate_id: int,
    action: np.ndarray,
    jaw_position: np.ndarray,
    edge_centers: dict[str, Any],
) -> dict[str, Any]:
    distances = distance_to_edge_centers(jaw_position, edge_centers)
    best_edge = min(distances, key=distances.get)
    return {
        "arm": arm,
        "candidate_id": int(candidate_id),
        "action_vector": [float(x) for x in np.asarray(action, dtype=np.float32).reshape(12)],
        "jaw_position": [float(x) for x in jaw_position],
        "distances_to_edge_centers_m": distances,
        "best_edge": best_edge,
        "best_distance_m": float(distances[best_edge]),
    }


def run_arm_search(
    scene: Any,
    arm: str,
    start_points: np.ndarray,
    edge_centers: dict[str, Any],
    candidate_count: int,
    settle_steps: int,
    seed: int,
) -> dict[str, Any]:
    ranges, range_source = joint_ranges_for_arm(arm_object(scene, arm))
    actions, action_metadata = generate_candidate_actions(arm, candidate_count, ranges, seed)
    num_candidates = int(actions.shape[0])
    jaw_positions = np.full((num_candidates, 3), np.nan, dtype=np.float32)
    distances = np.full((num_candidates, len(EDGE_NAMES)), np.inf, dtype=np.float32)
    per_candidate = []
    best_by_edge: dict[str, dict[str, Any] | None] = {edge: None for edge in EDGE_NAMES}

    for idx, action in enumerate(actions):
        reset_scene_to_points(scene, start_points)
        for _ in range(int(settle_steps)):
            apply_action_step(scene, action)
        jaw, jaw_info = measure_active_jaw(scene, arm)
        jaw_positions[idx] = jaw
        if not np.isfinite(jaw).all():
            result = {
                "arm": arm,
                "candidate_id": int(idx),
                "action_vector": [float(x) for x in action],
                "jaw_position": None,
                "distances_to_edge_centers_m": {edge: float("inf") for edge in EDGE_NAMES},
                "best_edge": None,
                "best_distance_m": float("inf"),
                "jaw_info": jaw_info,
            }
        else:
            result = candidate_result(arm, idx, action, jaw, edge_centers)
        per_candidate.append(result)
        for edge_idx, edge in enumerate(EDGE_NAMES):
            distance = float(result["distances_to_edge_centers_m"][edge])
            distances[idx, edge_idx] = distance
            current = best_by_edge[edge]
            if current is None or distance < float(current["distance_m"]):
                best_by_edge[edge] = {
                    "arm": arm,
                    "edge": edge,
                    "candidate_id": int(idx),
                    "distance_m": distance,
                    "jaw_position": None if result["jaw_position"] is None else list(result["jaw_position"]),
                    "action_vector": list(result["action_vector"]),
                    "joint_ranges_source": range_source,
                }

    finite_min = float(np.nanmin(distances)) if np.isfinite(distances).any() else float("inf")
    top_candidates = sorted(per_candidate, key=lambda item: float(item["best_distance_m"]))[:20]
    return {
        "arm": arm,
        "joint_ranges": ranges,
        "joint_ranges_source": range_source,
        "actions": actions,
        "jaw_positions": jaw_positions,
        "distances": distances,
        "best_by_edge": best_by_edge,
        "top_candidates": top_candidates,
        "num_candidates": num_candidates,
        "min_distance_m": finite_min,
        "action_generation": action_metadata,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search SO-101 action space for clean towel edge reachability.")
    parser.add_argument("--task", type=str, default=TASK_ID)
    parser.add_argument("--usd_path", type=Path, default=CLEAN_TOWEL_USD_PATH)
    parser.add_argument("--out_dir", type=Path, default=V6_DIR)
    parser.add_argument("--num_candidates_per_arm", type=int, default=640)
    parser.add_argument("--settle_steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2606)
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
    started = time.monotonic()
    try:
        scene = create_standalone_so101_clean_towel_scene(args, clean_usd_path=args.usd_path)
        scene.step_zero(20)
        start_points = scene.clean_towel.points_numpy().astype(np.float32)
        start_metrics = size_metrics(start_points)
        edge_centers = edge_centers_from_metrics(start_metrics)

        arm_results = {}
        for arm in ARM_NAMES:
            arm_results[arm] = run_arm_search(
                scene=scene,
                arm=arm,
                start_points=start_points,
                edge_centers=edge_centers,
                candidate_count=args.num_candidates_per_arm,
                settle_steps=args.settle_steps,
                seed=args.seed,
            )

        best_by_arm_edge = {arm: arm_results[arm]["best_by_edge"] for arm in ARM_NAMES}
        all_best = []
        for arm in ARM_NAMES:
            for edge in EDGE_NAMES:
                item = best_by_arm_edge[arm][edge]
                if item is not None:
                    all_best.append(item)
        global_best = min(all_best, key=lambda item: float(item["distance_m"])) if all_best else None
        best_distance = float(global_best["distance_m"]) if global_best else float("inf")
        reachable = bool(best_distance < 0.05)
        strong_reachable = bool(best_distance < 0.025)

        npz_path = args.out_dir / REACH_NPZ.name
        np.savez_compressed(
            npz_path,
            edge_names=np.asarray(EDGE_NAMES),
            arm_names=np.asarray(ARM_NAMES),
            edge_centers=edge_centers_array(edge_centers).astype(np.float32),
            start_points=start_points.astype(np.float32),
            left_actions=arm_results["left"]["actions"].astype(np.float32),
            right_actions=arm_results["right"]["actions"].astype(np.float32),
            left_jaw_positions=arm_results["left"]["jaw_positions"].astype(np.float32),
            right_jaw_positions=arm_results["right"]["jaw_positions"].astype(np.float32),
            left_distances=arm_results["left"]["distances"].astype(np.float32),
            right_distances=arm_results["right"]["distances"].astype(np.float32),
            left_joint_ranges=arm_results["left"]["joint_ranges"].astype(np.float32),
            right_joint_ranges=arm_results["right"]["joint_ranges"].astype(np.float32),
        )

        summary = {
            "status": "SO101_CLEAN_TOWEL_REACH_SEARCH_V6_DONE",
            "command_run": command_run_string(),
            "num_candidates_per_arm": int(args.num_candidates_per_arm),
            "actual_num_candidates": {arm: int(arm_results[arm]["num_candidates"]) for arm in ARM_NAMES},
            "settle_steps_per_candidate": int(args.settle_steps),
            "seed": int(args.seed),
            "elapsed_seconds": float(time.monotonic() - started),
            "towel_metrics": start_metrics,
            "towel_exact_edge_centers": edge_centers,
            "scene_info": standalone_robot_scene_info(scene),
            "joint_range_sources": {arm: arm_results[arm]["joint_ranges_source"] for arm in ARM_NAMES},
            "action_generation": {arm: arm_results[arm]["action_generation"] for arm in ARM_NAMES},
            "best_by_arm_and_edge": best_by_arm_edge,
            "global_best": global_best,
            "best_jaw_to_edge_center_distance_m": best_distance,
            "reachable_edge_found": reachable,
            "strong_reachable_edge_found": strong_reachable,
            "top_candidates_by_arm": {arm: arm_results[arm]["top_candidates"] for arm in ARM_NAMES},
            "reach_search_results_json_path": str(args.out_dir / REACH_JSON.name),
            "reach_search_results_npz_path": str(npz_path),
            "honest_note": (
                "This is a reachability search over SO-101 joint targets only. "
                "It measures jaw body distance to clean towel edge centers; it does not claim contact, towel motion, folding, or training."
            ),
        }
        write_json(args.out_dir / REACH_JSON.name, summary, print_payload=True)
    except Exception as exc:
        summary = {
            "status": "SO101_CLEAN_TOWEL_REACH_SEARCH_V6_FAILED",
            "command_run": command_run_string(),
            "exception_type": exc.__class__.__name__,
            "exception": repr(exc),
            "elapsed_seconds": float(time.monotonic() - started),
            "reachable_edge_found": False,
            "strong_reachable_edge_found": False,
            "honest_note": "Reach search failed; no contact, folding, or training claim is made.",
        }
        write_json(args.out_dir / REACH_JSON.name, summary, print_payload=True)
    finally:
        import os
        import sys

        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
