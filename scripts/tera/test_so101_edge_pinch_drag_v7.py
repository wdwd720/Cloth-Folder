from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
from isaaclab.app import AppLauncher

from clean_towel_v4_common import apply_action_step, reset_scene_to_points
from clean_towel_v5_contact_common import edge_center_displacement, towel_edge_masks
from inspect_so101_gripper_collision_v7 import (
    EDGE_PINCH_DRAG_RESULTS_JSON,
    EDGE_PINCH_DRAG_SUMMARY_JSON,
    GRIPPER_MAPPING_JSON,
    V7_DIR,
    active_slice,
    candidate_body_poses,
    clipped_action,
    gripper_index,
    load_json_if_exists,
    nearest_body_to_particles,
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
    points_to_numpy,
    size_metrics,
    standalone_robot_scene_info,
    write_json,
)


def inward_unit_for_edge(edge: str) -> np.ndarray:
    if edge == "left":
        return np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    if edge == "right":
        return np.asarray([-1.0, 0.0, 0.0], dtype=np.float32)
    if edge == "front":
        return np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    return np.asarray([0.0, -1.0, 0.0], dtype=np.float32)


def drag_seed_delta(edge: str, arm: str, magnitude: float) -> tuple[float, float, float]:
    if edge == "left":
        sign = 1.0 if arm == "left" else -1.0
        return sign * magnitude, 0.0, -0.10 * magnitude
    if edge == "right":
        sign = -1.0 if arm == "right" else 1.0
        return sign * magnitude, 0.0, -0.10 * magnitude
    if edge == "front":
        return 0.0, 0.25 * magnitude, -0.05 * magnitude
    return 0.0, -0.25 * magnitude, -0.05 * magnitude


def mapping_values(arm: str) -> dict[str, Any]:
    mapping = load_json_if_exists(GRIPPER_MAPPING_JSON)
    if not mapping:
        return {
            "mapping_known": False,
            "open_value": 0.65,
            "close_value": 0.02,
            "source": "fallback_v6_values_missing_mapping_summary",
        }
    selected = (mapping.get("by_arm") or {}).get(arm, {}).get("selected_mapping") or {}
    known = bool(selected.get("gripper_open_close_mapping_known", False))
    return {
        "mapping_known": known,
        "open_value": float(selected.get("open_value", mapping.get("open_value", 0.65))),
        "close_value": float(selected.get("close_value", mapping.get("close_value", 0.02))),
        "source": "gripper_open_close_mapping_summary" if known else "fallback_mapping_not_known",
    }


def open_close_pairs(arm: str) -> list[dict[str, float]]:
    values = mapping_values(arm)
    open_value = float(values["open_value"])
    close_value = float(values["close_value"])
    pairs = [
        {"open_value": open_value, "close_value": close_value},
        {"open_value": open_value, "close_value": 0.02},
    ]
    unique = []
    seen = set()
    for pair in pairs:
        key = (round(pair["open_value"], 4), round(pair["close_value"], 4))
        if key in seen:
            continue
        seen.add(key)
        unique.append(pair)
    return unique


def action_with_gripper(action: np.ndarray, arm: str, value: float) -> np.ndarray:
    out = clipped_action(action)
    out[gripper_index(arm)] = float(value)
    return clipped_action(out)


def make_pose_action(
    base_action: np.ndarray,
    arm: str,
    edge: str,
    approach_height_delta: float,
    jaw_depth_delta: float,
    gripper_value: float,
) -> np.ndarray:
    action = action_with_gripper(base_action, arm, gripper_value)
    sl = active_slice(arm)
    dpan, _, dwrist = drag_seed_delta(edge, arm, jaw_depth_delta)
    action[sl.start + 0] += dpan
    action[sl.start + 1] += float(approach_height_delta)
    action[sl.start + 2] += -0.35 * float(approach_height_delta)
    action[sl.start + 3] += float(dwrist) - 0.50 * float(approach_height_delta)
    return clipped_action(action)


def make_drag_action(
    close_action: np.ndarray,
    arm: str,
    edge: str,
    drag_distance: float,
    drag_height_delta: float,
    close_value: float,
) -> np.ndarray:
    action = action_with_gripper(close_action, arm, close_value)
    sl = active_slice(arm)
    dpan, dlift, dwrist = drag_seed_delta(edge, arm, drag_distance)
    action[sl.start + 0] += dpan
    action[sl.start + 1] += dlift + float(drag_height_delta)
    action[sl.start + 3] += dwrist - 0.50 * float(drag_height_delta)
    return clipped_action(action)


def nearest_gripper_to_particles(scene: Any, arm: str, particles: np.ndarray) -> dict[str, Any]:
    return nearest_body_to_particles(candidate_body_poses(scene, arm), particles)


def run_trial(
    scene: Any,
    start_points: np.ndarray,
    candidate: dict[str, Any],
    params: dict[str, Any],
    approach_steps: int,
    drag_steps: int,
    edge_band: float,
    capture_trace: bool = False,
) -> dict[str, Any]:
    arm = str(candidate["arm"])
    edge = str(candidate["edge"])
    base_action = clipped_action(np.asarray(candidate["action_vector"], dtype=np.float32))
    approach_action = make_pose_action(
        base_action,
        arm,
        edge,
        params["approach_height_delta"],
        params["jaw_depth_delta"],
        params["open_value"],
    )
    close_action = action_with_gripper(approach_action, arm, params["close_value"])
    drag_action = make_drag_action(
        close_action,
        arm,
        edge,
        params["drag_distance"],
        params["drag_height_delta"],
        params["close_value"],
    )
    reset_scene_to_points(scene, start_points)
    widths = [float(size_metrics(start_points)["width_x"])]
    nearest_values = []
    points_trace = [start_points.copy()] if capture_trace else None
    jaw_trace = []
    for phase_name, steps, action in (
        ("approach_open", approach_steps, approach_action),
        ("close_gripper", int(params["close_duration"]), close_action),
        ("drag_inward", drag_steps, drag_action),
    ):
        for _ in range(int(steps)):
            apply_action_step(scene, action)
            points = scene.clean_towel.points_numpy().astype(np.float32)
            widths.append(float(size_metrics(points)["width_x"]))
            nearest = nearest_gripper_to_particles(scene, arm, points)
            nearest_values.append(float(nearest["distance_m"]))
            if capture_trace:
                points_trace.append(points.copy())
                jaw_trace.append(nearest)
    final_points = scene.clean_towel.points_numpy().astype(np.float32)
    edge_disp = edge_center_displacement(start_points, final_points, edge, edge_band)
    nearest_all = float(min(nearest_values)) if nearest_values else float("inf")
    widths_arr = np.asarray(widths, dtype=np.float32)
    result = {
        "trial_id": int(params["trial_id"]),
        "arm": arm,
        "edge": edge,
        "params": params,
        "approach_action": [float(x) for x in approach_action],
        "close_action": [float(x) for x in close_action],
        "drag_action": [float(x) for x in drag_action],
        "nearest_jaw_to_towel_particle_m": nearest_all,
        "selected_edge_particle_displacement": float(edge_disp),
        "start_width": float(widths_arr[0]),
        "best_width": float(widths_arr.min()),
        "final_width": float(widths_arr[-1]),
        "actual_touch_achieved": bool(nearest_all < 0.015),
        "contact_moves_towel": bool(edge_disp > 0.02),
        "contact_width_reduction_success": bool(float(widths_arr.min()) < 0.65),
        "strong_contact_width_reduction_success": bool(float(widths_arr.min()) < 0.60),
    }
    if capture_trace:
        result["_capture"] = {
            "points_trace": np.asarray(points_trace, dtype=np.float32),
            "widths": widths_arr,
        }
    return result


def result_score(result: dict[str, Any]) -> float:
    return (
        8.0 * float(result.get("selected_edge_particle_displacement", 0.0))
        + 5.0 * max(0.0, 0.68 - float(result.get("best_width", 0.68)))
        - 0.30 * float(result.get("nearest_jaw_to_towel_particle_m", 99.0))
    )


def unique_results_for_fresh_replay(results: list[dict[str, Any]], max_replays: int = 3) -> list[dict[str, Any]]:
    if not results:
        return []
    selected = [
        max(results, key=result_score),
        min(results, key=lambda item: float(item["nearest_jaw_to_towel_particle_m"])),
        max(results, key=lambda item: float(item["selected_edge_particle_displacement"])),
        min(results, key=lambda item: float(item["best_width"])),
    ]
    selected.extend(sorted(results, key=result_score, reverse=True)[: int(max_replays)])
    unique = []
    seen = set()
    for item in selected:
        trial_id = int(item["trial_id"])
        if trial_id in seen:
            continue
        seen.add(trial_id)
        unique.append(item)
        if len(unique) >= int(max_replays):
            break
    return unique


def fresh_replay_candidates(
    args: argparse.Namespace,
    candidate: dict[str, Any],
    exploratory_results: list[dict[str, Any]],
    approach_steps: int,
    drag_steps: int,
    edge_band: float,
) -> list[dict[str, Any]]:
    fresh_results = []
    for source in unique_results_for_fresh_replay(exploratory_results):
        fresh_scene = create_standalone_so101_clean_towel_scene(
            args,
            clean_usd_path=args.usd_path,
            towel_translation=recommended_towel_translation(),
        )
        fresh_scene.step_zero(20)
        fresh_start = fresh_scene.clean_towel.points_numpy().astype(np.float32)
        replay = run_trial(
            fresh_scene,
            fresh_start,
            candidate,
            source["params"],
            approach_steps=approach_steps,
            drag_steps=drag_steps,
            edge_band=edge_band,
            capture_trace=False,
        )
        replay["fresh_scene_replay_of_trial_id"] = int(source["trial_id"])
        replay["exploratory_result"] = source
        fresh_results.append(replay)
    return fresh_results


def build_trial_params(arm: str, max_trials: int) -> list[dict[str, Any]]:
    pairs = open_close_pairs(arm)
    product = list(
        itertools.product(
            [-0.08, 0.0, 0.08],
            [-0.18, -0.06, 0.06, 0.18],
            [8, 24],
            [0.12, 0.32, 0.55],
            [-0.04, 0.04],
            pairs,
        )
    )
    if len(product) > int(max_trials):
        indices = np.linspace(0, len(product) - 1, int(max_trials), dtype=np.int32)
        product = [product[int(idx)] for idx in indices]
    params = []
    for trial_id, (height, depth, close_duration, drag_distance, drag_height, pair) in enumerate(product):
        params.append(
            {
                "trial_id": int(trial_id),
                "approach_height_delta": float(height),
                "jaw_depth_delta": float(depth),
                "close_duration": int(close_duration),
                "drag_distance": float(drag_distance),
                "drag_height_delta": float(drag_height),
                "open_value": float(pair["open_value"]),
                "close_value": float(pair["close_value"]),
            }
        )
    return params


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep SO-101 pure edge pinch/drag contact mechanics.")
    parser.add_argument("--task", type=str, default=TASK_ID)
    parser.add_argument("--usd_path", type=Path, default=CLEAN_TOWEL_USD_PATH)
    parser.add_argument("--out_dir", type=Path, default=V7_DIR)
    parser.add_argument("--max_trials", type=int, default=144)
    parser.add_argument("--approach_steps", type=int, default=14)
    parser.add_argument("--drag_steps", type=int, default=26)
    parser.add_argument("--edge_band", type=float, default=0.035)
    parser.add_argument("--verify_existing_results_only", action="store_true")
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
        if not candidate or not candidate.get("action_vector"):
            summary = {
                "status": "SO101_EDGE_PINCH_DRAG_SWEEP_V7_SKIPPED",
                "command_run": command_run_string(),
                "reason": "missing_v6_recommended_candidate",
                "actual_touch_achieved": False,
                "contact_moves_towel": False,
                "contact_width_reduction_success": False,
                "strong_contact_width_reduction_success": False,
                "honest_note": "No pure contact sweep was run because the V6 reachable candidate is missing.",
            }
            write_json(args.out_dir / EDGE_PINCH_DRAG_SUMMARY_JSON.name, summary, print_payload=True)
            write_json(args.out_dir / EDGE_PINCH_DRAG_RESULTS_JSON.name, {"results": []}, print_payload=False)
            return

        if args.verify_existing_results_only:
            existing = load_json_if_exists(args.out_dir / EDGE_PINCH_DRAG_RESULTS_JSON.name) or {}
            exploratory_results = existing.get("results", [])
            if not exploratory_results:
                raise RuntimeError("Missing exploratory edge pinch/drag results for verification.")
            scene = create_standalone_so101_clean_towel_scene(
                args,
                clean_usd_path=args.usd_path,
                towel_translation=recommended_towel_translation(),
            )
            scene.step_zero(20)
            start_points = scene.clean_towel.points_numpy().astype(np.float32)
            source = unique_results_for_fresh_replay(exploratory_results, max_replays=1)[0]
            fresh_replay = run_trial(
                scene,
                start_points,
                candidate,
                source["params"],
                approach_steps=args.approach_steps,
                drag_steps=args.drag_steps,
                edge_band=args.edge_band,
                capture_trace=True,
            )
            capture = fresh_replay.pop("_capture")
            fresh_replay["fresh_scene_replay_of_trial_id"] = int(source["trial_id"])
            fresh_replay["exploratory_result"] = source
            capture_path = args.out_dir / "edge_pinch_drag_verified_replay_capture_v7.npz"
            np.savez_compressed(
                capture_path,
                points_trace=capture["points_trace"].astype(np.float32),
                widths=capture["widths"].astype(np.float32),
                verified_params=np.asarray([json.dumps(source["params"])], dtype=object),
            )
            exploratory_best = max(exploratory_results, key=result_score)
            exploratory_nearest = min(
                exploratory_results, key=lambda item: float(item["nearest_jaw_to_towel_particle_m"])
            )
            exploratory_move = max(
                exploratory_results, key=lambda item: float(item["selected_edge_particle_displacement"])
            )
            exploratory_width = min(exploratory_results, key=lambda item: float(item["best_width"]))
            verified_basis = [fresh_replay]
            summary = {
                "status": "SO101_EDGE_PINCH_DRAG_SWEEP_V7_DONE",
                "command_run": command_run_string(),
                "mode": "verify_existing_results_only",
                "v6_recommended_towel_center": recommended_towel_center(),
                "towel_translation_used": [float(x) for x in recommended_towel_translation()],
                "v6_recommended_candidate": candidate,
                "mapping_values": mapping_values(str(candidate["arm"])),
                "num_trials": int(len(exploratory_results)),
                "approach_steps": int(args.approach_steps),
                "drag_steps": int(args.drag_steps),
                "edge_band": float(args.edge_band),
                "start_towel_metrics": size_metrics(start_points),
                "best_result": exploratory_best,
                "nearest_touch_best_result": exploratory_nearest,
                "edge_displacement_best_result": exploratory_move,
                "width_best_result": exploratory_width,
                "fresh_scene_replay_results": verified_basis,
                "verified_best_result": fresh_replay,
                "verified_nearest_touch_best_result": fresh_replay,
                "verified_edge_displacement_best_result": fresh_replay,
                "verified_width_best_result": fresh_replay,
                "best_pure_contact_edge_displacement": float(fresh_replay["selected_edge_particle_displacement"]),
                "best_pure_contact_width": float(fresh_replay["best_width"]),
                "actual_touch_achieved": bool(fresh_replay["actual_touch_achieved"]),
                "contact_moves_towel": bool(fresh_replay["contact_moves_towel"]),
                "contact_width_reduction_success": bool(fresh_replay["contact_width_reduction_success"]),
                "strong_contact_width_reduction_success": bool(fresh_replay["strong_contact_width_reduction_success"]),
                "success_metrics_basis": "single_fresh_process_replay_of_top_exploratory_candidate",
                "results_json_path": str(args.out_dir / EDGE_PINCH_DRAG_RESULTS_JSON.name),
                "best_capture_path": str(capture_path),
                "scene_info": standalone_robot_scene_info(scene),
                "honest_note": (
                    "The broad sweep results are exploratory. Success flags in this summary come from a fresh-process replay "
                    "of the selected candidate to avoid stale cloth state artifacts. No particle attachment or training is used."
                ),
            }
            existing["fresh_scene_replay_results"] = verified_basis
            write_json(args.out_dir / EDGE_PINCH_DRAG_RESULTS_JSON.name, existing, print_payload=False)
            write_json(args.out_dir / EDGE_PINCH_DRAG_SUMMARY_JSON.name, summary, print_payload=True)
            return

        scene = create_standalone_so101_clean_towel_scene(
            args,
            clean_usd_path=args.usd_path,
            towel_translation=recommended_towel_translation(),
        )
        scene.step_zero(20)
        start_points = scene.clean_towel.points_numpy().astype(np.float32)
        params = build_trial_params(str(candidate["arm"]), args.max_trials)
        results = []
        for item in params:
            result = run_trial(
                scene,
                start_points,
                candidate,
                item,
                approach_steps=args.approach_steps,
                drag_steps=args.drag_steps,
                edge_band=args.edge_band,
                capture_trace=False,
            )
            results.append(result)
        best = max(results, key=result_score) if results else None
        nearest_best = min(results, key=lambda item: float(item["nearest_jaw_to_towel_particle_m"])) if results else None
        move_best = max(results, key=lambda item: float(item["selected_edge_particle_displacement"])) if results else None
        width_best = min(results, key=lambda item: float(item["best_width"])) if results else None
        fresh_replays = []
        verified_best = max(fresh_replays, key=result_score) if fresh_replays else None
        verified_nearest_best = (
            min(fresh_replays, key=lambda item: float(item["nearest_jaw_to_towel_particle_m"])) if fresh_replays else None
        )
        verified_move_best = (
            max(fresh_replays, key=lambda item: float(item["selected_edge_particle_displacement"])) if fresh_replays else None
        )
        verified_width_best = min(fresh_replays, key=lambda item: float(item["best_width"])) if fresh_replays else None

        capture_path = args.out_dir / "edge_pinch_drag_best_capture_v7.npz"
        capture_source = verified_best or best
        if capture_source is not None:
            best_capture = run_trial(
                scene,
                start_points,
                candidate,
                capture_source["params"],
                approach_steps=args.approach_steps,
                drag_steps=args.drag_steps,
                edge_band=args.edge_band,
                capture_trace=True,
            )
            capture = best_capture.pop("_capture")
            np.savez_compressed(
                capture_path,
                points_trace=capture["points_trace"].astype(np.float32),
                widths=capture["widths"].astype(np.float32),
                best_params=np.asarray([json.dumps(capture_source["params"])], dtype=object),
            )
        else:
            capture_path = None

        verified_basis = fresh_replays if fresh_replays else results
        actual_touch = bool(any(item["actual_touch_achieved"] for item in verified_basis))
        moved = bool(any(item["contact_moves_towel"] for item in verified_basis))
        width_success = bool(any(item["contact_width_reduction_success"] for item in verified_basis))
        strong_width = bool(any(item["strong_contact_width_reduction_success"] for item in verified_basis))
        results_payload = {
            "status": "SO101_EDGE_PINCH_DRAG_SWEEP_RESULTS_V7_DONE",
            "command_run": command_run_string(),
            "results": results,
            "fresh_scene_replay_results": fresh_replays,
        }
        write_json(args.out_dir / EDGE_PINCH_DRAG_RESULTS_JSON.name, results_payload, print_payload=False)
        summary = {
            "status": "SO101_EDGE_PINCH_DRAG_SWEEP_V7_DONE",
            "command_run": command_run_string(),
            "v6_recommended_towel_center": recommended_towel_center(),
            "towel_translation_used": [float(x) for x in recommended_towel_translation()],
            "v6_recommended_candidate": candidate,
            "mapping_values": mapping_values(str(candidate["arm"])),
            "num_trials": int(len(results)),
            "approach_steps": int(args.approach_steps),
            "drag_steps": int(args.drag_steps),
            "edge_band": float(args.edge_band),
            "start_towel_metrics": size_metrics(start_points),
            "best_result": best,
            "nearest_touch_best_result": nearest_best,
            "edge_displacement_best_result": move_best,
            "width_best_result": width_best,
            "fresh_scene_replay_results": fresh_replays,
            "verified_best_result": verified_best,
            "verified_nearest_touch_best_result": verified_nearest_best,
            "verified_edge_displacement_best_result": verified_move_best,
            "verified_width_best_result": verified_width_best,
            "best_pure_contact_edge_displacement": None
            if verified_move_best is None
            else float(verified_move_best["selected_edge_particle_displacement"]),
            "best_pure_contact_width": None if verified_width_best is None else float(verified_width_best["best_width"]),
            "actual_touch_achieved": actual_touch,
            "contact_moves_towel": moved,
            "contact_width_reduction_success": width_success,
            "strong_contact_width_reduction_success": strong_width,
            "success_metrics_basis": "fresh_scene_replay_results",
            "results_json_path": str(args.out_dir / EDGE_PINCH_DRAG_RESULTS_JSON.name),
            "best_capture_path": None if capture_path is None else str(capture_path),
            "scene_info": standalone_robot_scene_info(scene),
            "honest_note": (
                "This sweep uses only SO-101 joint targets and physical simulation. "
                "No particle-space attachment or policy training is used; folding is not claimed unless width metrics pass."
            ),
        }
        write_json(args.out_dir / EDGE_PINCH_DRAG_SUMMARY_JSON.name, summary, print_payload=True)
    except Exception as exc:
        summary = {
            "status": "SO101_EDGE_PINCH_DRAG_SWEEP_V7_FAILED",
            "command_run": command_run_string(),
            "exception_type": exc.__class__.__name__,
            "exception": repr(exc),
            "actual_touch_achieved": False,
            "contact_moves_towel": False,
            "contact_width_reduction_success": False,
            "strong_contact_width_reduction_success": False,
            "honest_note": "Pure contact sweep failed; no contact, towel motion, folding, or training claim is made.",
        }
        write_json(args.out_dir / EDGE_PINCH_DRAG_SUMMARY_JSON.name, summary, print_payload=True)
        write_json(args.out_dir / EDGE_PINCH_DRAG_RESULTS_JSON.name, {"results": []}, print_payload=False)
    finally:
        import os
        import sys

        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
