from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from so101_clean_towel_scene_utils_v1 import command_run_string, write_json


V6_DIR = Path("/workspace/leisaac/tera_checkpoints/clean_towel_v6_reach")
REACH_JSON = V6_DIR / "reach_search_results.json"
REACH_NPZ = V6_DIR / "reach_search_results.npz"
PLACEMENT_JSON = V6_DIR / "towel_placement_search_summary.json"
EDGE_NAMES = ["left", "right", "front", "back"]
ARM_NAMES = ["left", "right"]


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def edge_centers_for_towel_center(center: np.ndarray, size: np.ndarray) -> dict[str, np.ndarray]:
    cx, cy, cz = [float(x) for x in center]
    sx, sy, _ = [float(x) for x in size]
    return {
        "left": np.asarray([cx - 0.5 * sx, cy, cz], dtype=np.float32),
        "right": np.asarray([cx + 0.5 * sx, cy, cz], dtype=np.float32),
        "front": np.asarray([cx, cy - 0.5 * sy, cz], dtype=np.float32),
        "back": np.asarray([cx, cy + 0.5 * sy, cz], dtype=np.float32),
    }


def candidate_arrays(data: Any, arm: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    actions = data[f"{arm}_actions"].astype(np.float32)
    jaws = data[f"{arm}_jaw_positions"].astype(np.float32)
    distances = data[f"{arm}_distances"].astype(np.float32)
    return actions, jaws, distances


def finite_rows(points: np.ndarray) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float32)
    return arr[np.isfinite(arr).all(axis=1)]


def grid_values(low: float, high: float, step: float) -> np.ndarray:
    count = int(np.floor((float(high) - float(low)) / float(step))) + 1
    count = max(1, count)
    return (float(low) + float(step) * np.arange(count, dtype=np.float32)).astype(np.float32)


def placement_grid(
    current_center: np.ndarray,
    towel_size: np.ndarray,
    all_jaws: np.ndarray,
    step: float,
    margin: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    finite = finite_rows(all_jaws)
    half_x = 0.5 * float(towel_size[0])
    half_y = 0.5 * float(towel_size[1])
    if finite.size:
        qmin = np.percentile(finite[:, :2], 2.0, axis=0)
        qmax = np.percentile(finite[:, :2], 98.0, axis=0)
        x_low = min(float(current_center[0]) - 0.80, float(qmin[0]) - half_x - float(margin))
        x_high = max(float(current_center[0]) + 0.80, float(qmax[0]) + half_x + float(margin))
        y_low = min(float(current_center[1]) - 0.80, float(qmin[1]) - half_y - float(margin))
        y_high = max(float(current_center[1]) + 0.45, float(qmax[1]) + half_y + float(margin))
    else:
        x_low, x_high = float(current_center[0]) - 0.80, float(current_center[0]) + 0.80
        y_low, y_high = float(current_center[1]) - 0.80, float(current_center[1]) + 0.45

    x_low, x_high = max(x_low, -1.60), min(x_high, 1.60)
    y_low, y_high = max(y_low, -1.60), min(y_high, 1.20)
    xs = grid_values(x_low, x_high, step)
    ys = grid_values(y_low, y_high, step)
    centers = np.asarray([[x, y, float(current_center[2])] for x in xs for y in ys], dtype=np.float32)
    meta = {
        "grid_step_m": float(step),
        "grid_margin_m": float(margin),
        "x_range": [float(xs[0]), float(xs[-1]), int(xs.shape[0])],
        "y_range": [float(ys[0]), float(ys[-1]), int(ys.shape[0])],
        "num_placements": int(centers.shape[0]),
        "jaw_workspace_xy_percentile_bounds": None
        if not finite.size
        else {
            "p02": [float(x) for x in np.percentile(finite[:, :2], 2.0, axis=0)],
            "p98": [float(x) for x in np.percentile(finite[:, :2], 98.0, axis=0)],
            "finite_jaw_samples": int(finite.shape[0]),
        },
    }
    return centers, meta


def evaluate_placement(
    center: np.ndarray,
    towel_size: np.ndarray,
    actions_by_arm: dict[str, np.ndarray],
    jaws_by_arm: dict[str, np.ndarray],
) -> dict[str, Any]:
    edge_centers = edge_centers_for_towel_center(center, towel_size)
    best = None
    best_by_arm_edge = {}
    for arm in ARM_NAMES:
        best_by_arm_edge[arm] = {}
        jaws = finite_rows(jaws_by_arm[arm])
        if jaws.size == 0:
            for edge in EDGE_NAMES:
                best_by_arm_edge[arm][edge] = None
            continue
        finite_indices = np.nonzero(np.isfinite(jaws_by_arm[arm]).all(axis=1))[0]
        for edge in EDGE_NAMES:
            distances = np.linalg.norm(jaws - edge_centers[edge].reshape(1, 3), axis=1)
            local_idx = int(np.argmin(distances))
            candidate_id = int(finite_indices[local_idx])
            item = {
                "arm": arm,
                "edge": edge,
                "candidate_id": candidate_id,
                "distance_m": float(distances[local_idx]),
                "jaw_position": [float(x) for x in jaws[local_idx]],
                "edge_center": [float(x) for x in edge_centers[edge]],
                "action_vector": [float(x) for x in actions_by_arm[arm][candidate_id]],
            }
            best_by_arm_edge[arm][edge] = item
            if best is None or float(item["distance_m"]) < float(best["distance_m"]):
                best = item
    return {
        "towel_center": [float(x) for x in center],
        "best": best,
        "best_by_arm_edge": best_by_arm_edge,
        "towel_exact_edge_centers": {edge: [float(x) for x in value] for edge, value in edge_centers.items()},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search clean towel placements that put an edge inside the SO-101 jaw workspace.")
    parser.add_argument("--out_dir", type=Path, default=V6_DIR)
    parser.add_argument("--grid_step", type=float, default=0.05)
    parser.add_argument("--grid_margin", type=float, default=0.20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    reach_json_path = args.out_dir / REACH_JSON.name
    reach_npz_path = args.out_dir / REACH_NPZ.name
    try:
        reach = load_json_if_exists(reach_json_path)
        if not reach:
            raise FileNotFoundError(f"Missing reach summary: {reach_json_path}")
        if not reach_npz_path.exists():
            raise FileNotFoundError(f"Missing reach NPZ: {reach_npz_path}")
        data = np.load(reach_npz_path)
        metrics = reach["towel_metrics"]
        current_center = np.asarray(metrics["center"], dtype=np.float32)
        towel_size = np.asarray(metrics["size"], dtype=np.float32)
        actions_by_arm = {}
        jaws_by_arm = {}
        for arm in ARM_NAMES:
            actions, jaws, _ = candidate_arrays(data, arm)
            actions_by_arm[arm] = actions
            jaws_by_arm[arm] = jaws

        all_jaws = np.concatenate([jaws_by_arm[arm] for arm in ARM_NAMES], axis=0)
        centers, grid_meta = placement_grid(current_center, towel_size, all_jaws, args.grid_step, args.grid_margin)
        placement_results = []
        for center in centers:
            placement_results.append(evaluate_placement(center, towel_size, actions_by_arm, jaws_by_arm))
        placement_results.sort(key=lambda item: float("inf") if item["best"] is None else float(item["best"]["distance_m"]))
        best_placement = placement_results[0] if placement_results else None
        best_distance = float("inf") if best_placement is None or best_placement["best"] is None else float(best_placement["best"]["distance_m"])
        current_reachable = bool(reach.get("reachable_edge_found", False))
        recommended_exists = bool(best_distance < 0.05)
        recommended_center = best_placement["towel_center"] if recommended_exists and best_placement else None
        move_delta = None
        move_instruction = None
        if recommended_center is not None:
            delta = np.asarray(recommended_center, dtype=np.float32) - current_center
            move_delta = [float(delta[0]), float(delta[1]), float(delta[2])]
            if not current_reachable:
                move_instruction = (
                    f"Move towel center by dx={delta[0]:.3f} m, dy={delta[1]:.3f} m "
                    f"from current center [{current_center[0]:.3f}, {current_center[1]:.3f}, {current_center[2]:.3f}]."
                )

        summary = {
            "status": "SO101_CLEAN_TOWEL_PLACEMENT_SEARCH_V6_DONE",
            "command_run": command_run_string(),
            "method": "uses V6 reach-search jaw workspace samples; no model training and no particle assist",
            "current_towel_center": [float(x) for x in current_center],
            "current_towel_size": [float(x) for x in towel_size],
            "current_placement_reachable_from_reach_search": current_reachable,
            "current_best_jaw_to_edge_center_distance_m": reach.get("best_jaw_to_edge_center_distance_m"),
            "grid": grid_meta,
            "recommended_towel_center": recommended_center,
            "recommended_towel_move_delta_m": move_delta,
            "recommended_towel_move_instruction": move_instruction,
            "recommended_candidate": None if not best_placement else best_placement["best"],
            "recommended_towel_exact_edge_centers": None if not best_placement else best_placement["towel_exact_edge_centers"],
            "recommended_reachable_edge_found": recommended_exists,
            "recommended_strong_reachable_edge_found": bool(best_distance < 0.025),
            "best_recommended_jaw_to_edge_center_distance_m": best_distance,
            "top_placements": placement_results[:20],
            "placement_search_summary_json_path": str(args.out_dir / PLACEMENT_JSON.name),
            "honest_note": (
                "The placement search rigidly shifts the clean towel bounding box in x/y and preserves its rectangular size and z height. "
                "It only predicts reachability from already measured SO-101 jaw workspace samples; it does not claim contact or folding."
            ),
        }
        write_json(args.out_dir / PLACEMENT_JSON.name, summary, print_payload=True)
    except Exception as exc:
        summary = {
            "status": "SO101_CLEAN_TOWEL_PLACEMENT_SEARCH_V6_FAILED",
            "command_run": command_run_string(),
            "exception_type": exc.__class__.__name__,
            "exception": repr(exc),
            "recommended_towel_center": None,
            "recommended_reachable_edge_found": False,
            "honest_note": "Placement search failed; no reach, contact, folding, or training claim is made.",
        }
        write_json(args.out_dir / PLACEMENT_JSON.name, summary, print_payload=True)


if __name__ == "__main__":
    main()
