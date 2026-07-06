from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from isaaclab.app import AppLauncher

from clean_towel_v4_common import apply_action_step, reset_scene_to_points
from clean_towel_v5_contact_common import (
    CONTACT_SWEEP_JSON,
    edge_center_displacement,
    towel_edge_masks,
)
from inspect_so101_clean_towel_geometry_v6 import (
    CONTACT_JSON,
    DEFAULT_STANDALONE_TOWEL_TRANSLATION,
    EDGE_NAMES,
    PLACEMENT_JSON,
    REACH_JSON,
    STATUS_JSON,
    V6_DIR,
    distance_to_edge_centers,
    edge_centers_from_metrics,
    jaw_info_for_arm,
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


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def arm_object(scene: Any, arm: str) -> Any:
    return scene.left_arm if arm == "left" else scene.right_arm


def active_slice(arm: str) -> slice:
    return slice(0, 6) if arm == "left" else slice(6, 12)


def gripper_index(arm: str) -> int:
    return 5 if arm == "left" else 11


def active_jaw_position(scene: Any, arm: str) -> np.ndarray:
    info = jaw_info_for_arm(arm_object(scene, arm))
    if info.get("position") is None:
        return np.asarray([np.nan, np.nan, np.nan], dtype=np.float32)
    return np.asarray(info["position"], dtype=np.float32)


def selected_candidate(reach: dict[str, Any] | None, placement: dict[str, Any] | None) -> dict[str, Any] | None:
    if reach and reach.get("reachable_edge_found") and reach.get("global_best"):
        item = dict(reach["global_best"])
        item["selection_source"] = "current_towel_reach_search"
        item["selected_towel_center"] = reach.get("towel_metrics", {}).get("center")
        return item
    if placement and placement.get("recommended_reachable_edge_found") and placement.get("recommended_candidate"):
        item = dict(placement["recommended_candidate"])
        item["selection_source"] = "recommended_towel_placement_search"
        item["selected_towel_center"] = placement.get("recommended_towel_center")
        return item
    return None


def towel_translation_for_candidate(candidate: dict[str, Any], reach: dict[str, Any] | None) -> tuple[float, float, float]:
    if candidate.get("selection_source") != "recommended_towel_placement_search":
        return DEFAULT_STANDALONE_TOWEL_TRANSLATION
    current_center = np.asarray((reach or {}).get("towel_metrics", {}).get("center", [0.0, 0.0, 0.0]), dtype=np.float32)
    selected_center = np.asarray(candidate.get("selected_towel_center", current_center), dtype=np.float32)
    delta = selected_center - current_center
    return (float(delta[0]), float(delta[1]), float(DEFAULT_STANDALONE_TOWEL_TRANSLATION[2]))


def inward_unit_for_edge(edge: str) -> np.ndarray:
    if edge == "left":
        return np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    if edge == "right":
        return np.asarray([-1.0, 0.0, 0.0], dtype=np.float32)
    if edge == "front":
        return np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    return np.asarray([0.0, -1.0, 0.0], dtype=np.float32)


def clipped_action(action: np.ndarray) -> np.ndarray:
    out = np.asarray(action, dtype=np.float32).reshape(12).copy()
    out[:5] = np.clip(out[:5], -1.85, 1.85)
    out[6:11] = np.clip(out[6:11], -1.85, 1.85)
    out[5] = float(np.clip(out[5], -0.05, 0.90))
    out[11] = float(np.clip(out[11], -0.05, 0.90))
    return out


def drag_seed_delta(edge: str, arm: str, magnitude: float) -> tuple[float, float, float]:
    sign = 1.0
    if edge == "left":
        sign = 1.0 if arm == "left" else -1.0
        return sign * magnitude, 0.0, -0.10 * magnitude
    if edge == "right":
        sign = -1.0 if arm == "right" else 1.0
        return sign * magnitude, 0.0, -0.10 * magnitude
    if edge == "front":
        return 0.0, 0.25 * magnitude, -0.05 * magnitude
    return 0.0, -0.25 * magnitude, -0.05 * magnitude


def drag_action_candidates(base_action: np.ndarray, arm: str, edge: str, close_gripper: float) -> list[np.ndarray]:
    sl = active_slice(arm)
    grip = gripper_index(arm)
    base = clipped_action(base_action)
    base[grip] = close_gripper
    candidates = [base.copy()]

    for magnitude in (0.12, 0.24, 0.38, 0.55, 0.72):
        dp, dl, dw = drag_seed_delta(edge, arm, magnitude)
        action = base.copy()
        action[sl.start + 0] += dp
        action[sl.start + 1] += dl
        action[sl.start + 3] += dw
        candidates.append(clipped_action(action))

    for joint_offset in range(4):
        for delta in (-0.30, -0.18, -0.09, 0.09, 0.18, 0.30):
            action = base.copy()
            action[sl.start + joint_offset] += delta
            candidates.append(clipped_action(action))

    rng = np.random.default_rng(6060 if arm == "left" else 7070)
    local_scale = np.asarray([0.22, 0.18, 0.18, 0.16, 0.08, 0.0], dtype=np.float32)
    for _ in range(24):
        action = base.copy()
        action[sl] += rng.normal(0.0, local_scale).astype(np.float32)
        action[grip] = close_gripper
        candidates.append(clipped_action(action))

    unique = []
    seen = set()
    for action in candidates:
        key = tuple(np.round(action, 4).tolist())
        if key in seen:
            continue
        seen.add(key)
        unique.append(action)
    return unique


def nearest_jaw_to_edge_particles(jaw_positions: np.ndarray, edge_points: np.ndarray) -> float:
    jaws = np.asarray(jaw_positions, dtype=np.float32).reshape(-1, 3)
    pts = np.asarray(edge_points, dtype=np.float32).reshape(-1, 3)
    if jaws.size == 0 or pts.size == 0:
        return float("inf")
    nearest = float("inf")
    for jaw in jaws:
        if not np.isfinite(jaw).all():
            continue
        nearest = min(nearest, float(np.linalg.norm(pts - jaw.reshape(1, 3), axis=1).min()))
    return nearest


def choose_drag_action(
    scene: Any,
    start_points: np.ndarray,
    arm: str,
    edge: str,
    approach_action: np.ndarray,
    close_action: np.ndarray,
    edge_points: np.ndarray,
    candidates: list[np.ndarray],
    approach_steps: int,
    close_steps: int,
    settle_steps: int,
) -> dict[str, Any]:
    inward = inward_unit_for_edge(edge)
    scored = []
    for idx, action in enumerate(candidates):
        reset_scene_to_points(scene, start_points)
        for _ in range(int(approach_steps)):
            apply_action_step(scene, approach_action)
        for _ in range(int(close_steps)):
            apply_action_step(scene, close_action)
        before = active_jaw_position(scene, arm)
        for _ in range(int(settle_steps)):
            apply_action_step(scene, action)
        after = active_jaw_position(scene, arm)
        displacement = after - before
        inward_motion = float(np.dot(displacement, inward)) if np.isfinite(displacement).all() else -float("inf")
        lateral = displacement - inward_motion * inward if np.isfinite(displacement).all() else np.asarray([np.inf, np.inf, np.inf])
        nearest_after = nearest_jaw_to_edge_particles(after.reshape(1, 3), edge_points)
        score = inward_motion - 0.25 * float(np.linalg.norm(lateral)) - 0.15 * max(0.0, nearest_after - 0.08)
        scored.append(
            {
                "candidate_index": int(idx),
                "score": float(score),
                "inward_jaw_motion_m": float(inward_motion),
                "nearest_after_drag_candidate_m": float(nearest_after),
                "before_jaw": [float(x) for x in before],
                "after_jaw": [float(x) for x in after],
                "action_vector": [float(x) for x in action],
            }
        )
    scored.sort(key=lambda item: float(item["score"]), reverse=True)
    return {
        "selected": scored[0] if scored else None,
        "top_drag_candidates": scored[:10],
        "num_drag_candidates": int(len(candidates)),
    }


def run_sequence(
    scene: Any,
    start_points: np.ndarray,
    arm: str,
    segments: list[dict[str, Any]],
) -> dict[str, Any]:
    widths = [float(size_metrics(start_points)["width_x"])]
    points_trace = [start_points.astype(np.float32).copy()]
    jaw_trace = [active_jaw_position(scene, arm)]
    segment_steps = []
    step_index = 0
    for segment in segments:
        start_step = step_index
        action = np.asarray(segment["action"], dtype=np.float32)
        for _ in range(int(segment["steps"])):
            apply_action_step(scene, action)
            points = scene.clean_towel.points_numpy().astype(np.float32)
            widths.append(float(size_metrics(points)["width_x"]))
            points_trace.append(points.copy())
            jaw_trace.append(active_jaw_position(scene, arm))
            step_index += 1
        segment_steps.append({"name": segment["name"], "start_step": int(start_step), "end_step": int(step_index)})
    return {
        "widths": np.asarray(widths, dtype=np.float32),
        "points_trace": np.asarray(points_trace, dtype=np.float32),
        "jaw_trace": np.asarray(jaw_trace, dtype=np.float32),
        "segment_steps": segment_steps,
    }


def compact_best_distances(reach: dict[str, Any] | None) -> dict[str, Any]:
    if not reach:
        return {}
    out = {}
    for arm, by_edge in (reach.get("best_by_arm_and_edge") or {}).items():
        out[arm] = {}
        for edge, item in by_edge.items():
            out[arm][edge] = None if item is None else float(item["distance_m"])
    return out


def build_status(
    reach: dict[str, Any] | None,
    placement: dict[str, Any] | None,
    contact: dict[str, Any],
) -> dict[str, Any]:
    v5_sweep = load_json_if_exists(CONTACT_SWEEP_JSON)
    v5_best = None if not v5_sweep else v5_sweep.get("best_candidate")
    v5_nearest = None if not v5_best else v5_best.get("nearest_jaw_to_selected_edge_m")
    v5_invalidated = bool(v5_nearest is not None and float(v5_nearest) >= 0.05)
    near_contact = bool(contact.get("near_contact_achieved", False))
    moved = bool(contact.get("contact_moves_towel", False))
    width_success = bool(contact.get("contact_width_reduction_success", False))
    ready_for_v7 = bool(near_contact and moved)
    return {
        "status": "CLEAN_TOWEL_V6_REACH_CONTACT_DONE",
        "v5_invalidated_by_jaw_distance": v5_invalidated,
        "v5_best_nearest_jaw_to_selected_edge_m": v5_nearest,
        "v5_invalidated_threshold_m": 0.05,
        "current_towel_placement_reachable": bool(reach and reach.get("reachable_edge_found", False)),
        "current_towel_placement_strong_reachable": bool(reach and reach.get("strong_reachable_edge_found", False)),
        "current_best_jaw_to_edge_center_distance_m": None
        if not reach
        else reach.get("best_jaw_to_edge_center_distance_m"),
        "best_jaw_to_edge_center_distances_by_arm_edge_m": compact_best_distances(reach),
        "recommended_towel_placement_needed": bool(reach and not reach.get("reachable_edge_found", False)),
        "recommended_towel_center": None if not placement else placement.get("recommended_towel_center"),
        "recommended_towel_move_delta_m": None if not placement else placement.get("recommended_towel_move_delta_m"),
        "recommended_best_jaw_to_edge_center_distance_m": None
        if not placement
        else placement.get("best_recommended_jaw_to_edge_center_distance_m"),
        "contact_after_reach_fix_status": contact.get("status"),
        "actual_contact_moved_towel": moved,
        "near_contact_achieved": near_contact,
        "nearest_jaw_to_selected_edge_m": contact.get("nearest_jaw_to_selected_edge_m"),
        "selected_edge_particle_displacement": contact.get("selected_edge_particle_displacement"),
        "contact_width_reduction_success": width_success,
        "strong_contact_width_reduction_success": bool(contact.get("strong_contact_width_reduction_success", False)),
        "ready_for_v7_contact_policy_training": ready_for_v7,
        "ready_for_v7_contact_policy_training_basis": (
            "true only when near_contact_achieved and contact_moves_towel are both true; no autonomous folding is claimed from V6"
        ),
        "do_not_train_in_v6": True,
        "paths": {
            "geometry_summary": str(V6_DIR / "geometry_summary.json"),
            "reach_search_results": str(V6_DIR / REACH_JSON.name),
            "reach_search_npz": str(V6_DIR / "reach_search_results.npz"),
            "placement_search_summary": str(V6_DIR / PLACEMENT_JSON.name),
            "contact_after_reach_fix_summary": str(V6_DIR / CONTACT_JSON.name),
            "status_json": str(V6_DIR / STATUS_JSON.name),
        },
        "honest_note": (
            "V6 is geometry, reachability, towel-placement, and contact validation only. "
            "Do not claim contact unless nearest-jaw and towel-motion metrics pass, and do not claim autonomous folding unless width metrics pass."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test SO-101 clean towel contact after V6 reach or placement fix.")
    parser.add_argument("--task", type=str, default=TASK_ID)
    parser.add_argument("--usd_path", type=Path, default=CLEAN_TOWEL_USD_PATH)
    parser.add_argument("--out_dir", type=Path, default=V6_DIR)
    parser.add_argument("--edge_band", type=float, default=0.035)
    parser.add_argument("--approach_steps", type=int, default=24)
    parser.add_argument("--close_steps", type=int, default=18)
    parser.add_argument("--drag_steps", type=int, default=36)
    parser.add_argument("--drag_candidate_settle_steps", type=int, default=10)
    parser.add_argument("--open_gripper_min", type=float, default=0.55)
    parser.add_argument("--close_gripper", type=float, default=0.02)
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
    contact_summary: dict[str, Any]
    try:
        reach = load_json_if_exists(args.out_dir / REACH_JSON.name)
        placement = load_json_if_exists(args.out_dir / PLACEMENT_JSON.name)
        candidate = selected_candidate(reach, placement)
        if candidate is None:
            contact_summary = {
                "status": "SO101_CLEAN_TOWEL_CONTACT_AFTER_REACH_FIX_V6_SKIPPED",
                "command_run": command_run_string(),
                "reason": "no_current_or_recommended_reachable_edge_candidate",
                "near_contact_achieved": False,
                "contact_moves_towel": False,
                "contact_width_reduction_success": False,
                "strong_contact_width_reduction_success": False,
                "honest_note": "No contact test was run because V6 did not find a reachable current or recommended towel edge.",
            }
            status = build_status(reach, placement, contact_summary)
            write_json(args.out_dir / CONTACT_JSON.name, contact_summary, print_payload=False)
            write_json(args.out_dir / STATUS_JSON.name, status, print_payload=False)
            print(json.dumps({"contact_summary": contact_summary, "v6_status": status}, indent=2), flush=True)
            return

        towel_translation = towel_translation_for_candidate(candidate, reach)
        scene = create_standalone_so101_clean_towel_scene(
            args,
            clean_usd_path=args.usd_path,
            towel_translation=towel_translation,
        )
        scene.step_zero(20)
        start_points = scene.clean_towel.points_numpy().astype(np.float32)
        start_metrics = size_metrics(start_points)
        exact_edge_centers = edge_centers_from_metrics(start_metrics)
        arm = str(candidate["arm"])
        edge = str(candidate["edge"])
        action = clipped_action(np.asarray(candidate["action_vector"], dtype=np.float32))
        action[gripper_index(arm)] = max(float(action[gripper_index(arm)]), float(args.open_gripper_min))
        close_action = action.copy()
        close_action[gripper_index(arm)] = float(args.close_gripper)
        edge_mask = towel_edge_masks(start_points, args.edge_band)[edge]
        edge_points = start_points[edge_mask]
        drag_candidates = drag_action_candidates(action, arm, edge, close_gripper=float(args.close_gripper))
        drag_selection = choose_drag_action(
            scene=scene,
            start_points=start_points,
            arm=arm,
            edge=edge,
            approach_action=action,
            close_action=close_action,
            edge_points=edge_points,
            candidates=drag_candidates,
            approach_steps=max(4, args.approach_steps // 2),
            close_steps=max(4, args.close_steps // 2),
            settle_steps=args.drag_candidate_settle_steps,
        )
        selected_drag = drag_selection["selected"]
        drag_action = close_action.copy() if selected_drag is None else np.asarray(selected_drag["action_vector"], dtype=np.float32)

        reset_scene_to_points(scene, start_points)
        trace = run_sequence(
            scene,
            start_points,
            arm,
            [
                {"name": "approach_open", "steps": int(args.approach_steps), "action": action.tolist()},
                {"name": "close_gripper", "steps": int(args.close_steps), "action": close_action.tolist()},
                {"name": "drag_inward", "steps": int(args.drag_steps), "action": drag_action.tolist()},
            ],
        )
        final_points = trace["points_trace"][-1]
        widths = trace["widths"]
        start_width = float(size_metrics(start_points)["width_x"])
        best_width = float(widths.min())
        final_width = float(widths[-1])
        selected_edge_displacement = edge_center_displacement(start_points, final_points, edge, args.edge_band)
        nearest = nearest_jaw_to_edge_particles(trace["jaw_trace"], edge_points)
        final_jaw = trace["jaw_trace"][-1]
        exact_center_distances = distance_to_edge_centers(final_jaw, exact_edge_centers) if np.isfinite(final_jaw).all() else {}
        capture_path = args.out_dir / "contact_after_reach_fix_capture_v6.npz"
        np.savez_compressed(
            capture_path,
            points_trace=trace["points_trace"].astype(np.float32),
            widths=widths.astype(np.float32),
            jaw_trace=trace["jaw_trace"].astype(np.float32),
            approach_action=action.astype(np.float32),
            close_action=close_action.astype(np.float32),
            drag_action=drag_action.astype(np.float32),
            start_points=start_points.astype(np.float32),
            final_points=final_points.astype(np.float32),
        )
        contact_summary = {
            "status": "SO101_CLEAN_TOWEL_CONTACT_AFTER_REACH_FIX_V6_DONE",
            "command_run": command_run_string(),
            "selected_candidate": candidate,
            "selected_arm": arm,
            "selected_edge": edge,
            "selection_source": candidate.get("selection_source"),
            "towel_translation_used": [float(x) for x in towel_translation],
            "start_towel_metrics": start_metrics,
            "start_towel_exact_edge_centers": exact_edge_centers,
            "selected_edge_particle_count": int(edge_points.shape[0]),
            "approach_action": [float(x) for x in action],
            "close_action": [float(x) for x in close_action],
            "drag_action": [float(x) for x in drag_action],
            "drag_selection": drag_selection,
            "segment_steps": trace["segment_steps"],
            "capture_path": str(capture_path),
            "start_width": start_width,
            "best_width": best_width,
            "final_width": final_width,
            "selected_edge_particle_displacement": float(selected_edge_displacement),
            "nearest_jaw_to_selected_edge_m": float(nearest),
            "final_active_jaw_to_exact_edge_center_distances_m": exact_center_distances,
            "near_contact_achieved": bool(nearest < 0.05),
            "contact_moves_towel": bool(selected_edge_displacement > 0.02),
            "contact_width_reduction_success": bool(best_width < 0.65),
            "strong_contact_width_reduction_success": bool(best_width < 0.60),
            "scene_info": standalone_robot_scene_info(scene),
            "contact_after_reach_fix_summary_json_path": str(args.out_dir / CONTACT_JSON.name),
            "status_json_path": str(args.out_dir / STATUS_JSON.name),
            "honest_note": (
                "This test uses only SO-101 joint targets and physical simulation after the V6 reach/placement selection. "
                "It does not train a model and does not claim autonomous folding unless width thresholds pass."
            ),
        }
        status = build_status(reach, placement, contact_summary)
        write_json(args.out_dir / CONTACT_JSON.name, contact_summary, print_payload=False)
        write_json(args.out_dir / STATUS_JSON.name, status, print_payload=False)
        print(json.dumps({"contact_summary": contact_summary, "v6_status": status}, indent=2), flush=True)
    except Exception as exc:
        reach = load_json_if_exists(args.out_dir / REACH_JSON.name)
        placement = load_json_if_exists(args.out_dir / PLACEMENT_JSON.name)
        contact_summary = {
            "status": "SO101_CLEAN_TOWEL_CONTACT_AFTER_REACH_FIX_V6_FAILED",
            "command_run": command_run_string(),
            "exception_type": exc.__class__.__name__,
            "exception": repr(exc),
            "near_contact_achieved": False,
            "contact_moves_towel": False,
            "contact_width_reduction_success": False,
            "strong_contact_width_reduction_success": False,
            "honest_note": "Contact-after-reach-fix failed; no contact, towel motion, folding, or training claim is made.",
        }
        status = build_status(reach, placement, contact_summary)
        write_json(args.out_dir / CONTACT_JSON.name, contact_summary, print_payload=False)
        write_json(args.out_dir / STATUS_JSON.name, status, print_payload=False)
        print(json.dumps({"contact_summary": contact_summary, "v6_status": status}, indent=2), flush=True)
    finally:
        import os
        import sys

        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
