from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from isaaclab.app import AppLauncher

from clean_towel_v4_common import apply_action_step, reset_scene_to_points
from clean_towel_v5_contact_common import edge_center_displacement
from inspect_so101_gripper_collision_v7 import (
    CONTACT_PARAMETER_SUMMARY_JSON,
    EDGE_PINCH_DRAG_SUMMARY_JSON,
    GRIPPER_COLLISION_JSON,
    GRIPPER_MAPPING_JSON,
    LOCAL_ATTACHMENT_JSON,
    STATUS_JSON,
    V7_DIR,
    candidate_body_poses,
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
    points_to_numpy,
    size_metrics,
    standalone_robot_scene_info,
    write_json,
)
from test_so101_edge_pinch_drag_v7 import (
    action_with_gripper,
    make_drag_action,
    make_pose_action,
    mapping_values,
)


def selected_motion_params(arm: str) -> dict[str, Any]:
    pure = load_json_if_exists(EDGE_PINCH_DRAG_SUMMARY_JSON)
    if pure and pure.get("verified_best_result") and pure["verified_best_result"].get("params"):
        params = dict(pure["verified_best_result"]["params"])
    elif pure and pure.get("best_result") and pure["best_result"].get("params"):
        params = dict(pure["best_result"]["params"])
    else:
        params = {
            "trial_id": 0,
            "approach_height_delta": 0.0,
            "jaw_depth_delta": 0.06,
            "close_duration": 24,
            "drag_distance": 0.32,
            "drag_height_delta": -0.04,
            "open_value": 0.65,
            "close_value": 0.02,
        }
    values = mapping_values(arm)
    if values["mapping_known"]:
        params["open_value"] = float(values["open_value"])
        params["close_value"] = float(values["close_value"])
    return params


def active_contact_position(scene: Any, arm: str) -> tuple[np.ndarray, str]:
    poses = candidate_body_poses(scene, arm)
    preferred = [pose for pose in poses if "jaw" in pose["body_name"].lower() or "finger" in pose["body_name"].lower()]
    candidates = preferred or poses
    for pose in candidates:
        if pose.get("position") is not None:
            return np.asarray(pose["position"], dtype=np.float32), str(pose["body_name"])
    return np.asarray([np.nan, np.nan, np.nan], dtype=np.float32), "unavailable"


def local_attachment_trial(
    scene: Any,
    start_points: np.ndarray,
    candidate: dict[str, Any],
    motion_params: dict[str, Any],
    radius: float,
    strength: float,
    approach_steps: int,
    drag_steps: int,
    edge_band: float,
) -> dict[str, Any]:
    arm = str(candidate["arm"])
    edge = str(candidate["edge"])
    base_action = np.asarray(candidate["action_vector"], dtype=np.float32)
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
    widths = [float(size_metrics(start_points)["width_x"])]
    for _ in range(int(approach_steps)):
        apply_action_step(scene, approach_action)
        widths.append(float(size_metrics(scene.clean_towel.points_numpy())["width_x"]))
    for _ in range(int(motion_params["close_duration"])):
        apply_action_step(scene, close_action)
        widths.append(float(size_metrics(scene.clean_towel.points_numpy())["width_x"]))

    close_points = scene.clean_towel.points_numpy().astype(np.float32)
    jaw_pos, jaw_body = active_contact_position(scene, arm)
    if np.isfinite(jaw_pos).all():
        distances = np.linalg.norm(close_points - jaw_pos.reshape(1, 3), axis=1)
        attach_mask = distances <= float(radius)
    else:
        distances = np.full(close_points.shape[0], np.inf, dtype=np.float32)
        attach_mask = np.zeros(close_points.shape[0], dtype=bool)
    offsets = close_points[attach_mask] - jaw_pos.reshape(1, 3) if np.any(attach_mask) else np.zeros((0, 3), dtype=np.float32)

    for _ in range(int(drag_steps)):
        apply_action_step(scene, drag_action)
        current = scene.clean_towel.points_numpy().astype(np.float32)
        if np.any(attach_mask) and float(strength) > 0.0:
            jaw_now, _ = active_contact_position(scene, arm)
            if np.isfinite(jaw_now).all():
                target = jaw_now.reshape(1, 3) + offsets
                current[attach_mask] = (1.0 - float(strength)) * current[attach_mask] + float(strength) * target
                scene.clean_towel.set_points(current, device=scene.device)
                try:
                    scene.sim.forward()
                except Exception:
                    pass
        widths.append(float(size_metrics(scene.clean_towel.points_numpy())["width_x"]))

    final_points = scene.clean_towel.points_numpy().astype(np.float32)
    widths_arr = np.asarray(widths, dtype=np.float32)
    edge_disp = edge_center_displacement(start_points, final_points, edge, edge_band)
    return {
        "radius": float(radius),
        "strength": float(strength),
        "jaw_body_used_for_attachment": jaw_body,
        "number_of_particles_attached": int(attach_mask.sum()),
        "attached_particle_fraction": float(attach_mask.sum() / max(1, start_points.shape[0])),
        "nearest_particle_at_close_m": float(np.min(distances)) if distances.size else float("inf"),
        "start_width": float(widths_arr[0]),
        "best_width": float(widths_arr.min()),
        "final_width": float(widths_arr[-1]),
        "selected_edge_particle_displacement": float(edge_disp),
        "local_attachment_moves_towel": bool(edge_disp > 0.02),
        "local_attachment_width_under_0_50": bool(float(widths_arr.min()) < 0.50),
        "attachment_type": "local_gripper_attachment_assisted_not_true_contact",
    }


def local_result_sanity(result: dict[str, Any]) -> dict[str, Any]:
    reasons = []
    numeric_values = {}
    for key in (
        "nearest_particle_at_close_m",
        "selected_edge_particle_displacement",
        "start_width",
        "best_width",
        "final_width",
        "attached_particle_fraction",
    ):
        try:
            number = float(result.get(key))
        except Exception:
            number = float("nan")
        numeric_values[key] = number
        if not np.isfinite(number):
            reasons.append(f"{key}_not_finite")

    start_width = numeric_values["start_width"]
    best_width = numeric_values["best_width"]
    final_width = numeric_values["final_width"]
    displacement = numeric_values["selected_edge_particle_displacement"]
    attached_fraction = numeric_values["attached_particle_fraction"]

    if np.isfinite(start_width) and not (0.45 <= start_width <= 0.85):
        reasons.append("start_width_outside_expected_clean_towel_range")
    if np.isfinite(best_width) and not (0.10 <= best_width <= 1.20):
        reasons.append("best_width_outside_sane_range")
    if np.isfinite(final_width) and not (0.10 <= final_width <= 1.20):
        reasons.append("final_width_outside_sane_range")
    if np.isfinite(displacement) and not (0.0 <= displacement <= 1.0):
        reasons.append("edge_displacement_outside_sane_range")
    if np.isfinite(attached_fraction) and attached_fraction > 0.05:
        reasons.append("attachment_not_local_enough")

    number_attached = int(result.get("number_of_particles_attached", 0))
    eligible_attachment = bool(number_attached > 0 and not reasons)
    return {
        "sanity_valid": bool(not reasons),
        "sanity_invalid_reasons": reasons,
        "eligible_local_attachment_result": eligible_attachment,
    }


def local_score(result: dict[str, Any]) -> float:
    locality_penalty = 0.25 * float(result.get("attached_particle_fraction", 1.0))
    return (
        8.0 * float(result.get("selected_edge_particle_displacement", 0.0))
        + 5.0 * max(0.0, 0.68 - float(result.get("best_width", 0.68)))
        - locality_penalty
    )


def build_status(local_summary: dict[str, Any]) -> dict[str, Any]:
    collision = load_json_if_exists(GRIPPER_COLLISION_JSON) or {}
    mapping = load_json_if_exists(GRIPPER_MAPPING_JSON) or {}
    pure = load_json_if_exists(EDGE_PINCH_DRAG_SUMMARY_JSON) or {}
    params = load_json_if_exists(CONTACT_PARAMETER_SUMMARY_JSON) or {}

    pure_actual_touch = bool(pure.get("actual_touch_achieved", False))
    pure_moved = bool(pure.get("contact_moves_towel", False))
    pure_width_success = bool(pure.get("contact_width_reduction_success", False))
    param_actual_touch = bool(params.get("parameter_sweep_actual_touch_achieved", False))
    param_moved = bool(params.get("parameter_sweep_moves_towel", False))
    param_width_success = bool(params.get("parameter_sweep_width_reduction_success", False))
    param_touch_moved = bool(params.get("parameter_sweep_touch_confirmed_moves_towel", False))
    param_touch_width_success = bool(params.get("parameter_sweep_touch_confirmed_width_reduction_success", False))
    param_contact_evidence_success = bool(params.get("parameter_sweep_contact_evidence_success", False))
    true_robot_contact_folding = bool(
        (pure_actual_touch and pure_moved and pure_width_success)
        or (param_actual_touch and param_touch_moved and param_touch_width_success)
    )
    local_required = bool(not true_robot_contact_folding and local_summary.get("local_attachment_improves_over_pure_contact", False))

    return {
        "status": "CLEAN_TOWEL_V7_CONTACT_MECHANICS_DONE",
        "jaw_collision_bodies_exist": bool(collision.get("jaw_collision_bodies_found", False)),
        "finger_jaw_body_names_identified": collision.get("finger_jaw_body_names_identified"),
        "gripper_open_close_mapping_works": bool(mapping.get("gripper_open_close_mapping_known", False)),
        "jaw_gap_changes_with_action": bool(mapping.get("jaw_gap_changes_with_action", False)),
        "open_value": mapping.get("open_value"),
        "close_value": mapping.get("close_value"),
        "actual_touch_was_achieved": pure_actual_touch,
        "pure_contact_moved_towel": pure_moved,
        "best_pure_contact_edge_displacement": pure.get("best_pure_contact_edge_displacement"),
        "best_pure_contact_width": pure.get("best_pure_contact_width"),
        "pure_contact_width_reduction_success": pure_width_success,
        "contact_parameter_sweep_helped": param_contact_evidence_success,
        "parameter_sweep_moves_towel": param_moved,
        "parameter_sweep_width_reduction_success": param_width_success,
        "parameter_sweep_actual_touch_achieved": param_actual_touch,
        "parameter_sweep_touch_confirmed_moves_towel": param_touch_moved,
        "parameter_sweep_touch_confirmed_width_reduction_success": param_touch_width_success,
        "parameter_sweep_contact_evidence_success": param_contact_evidence_success,
        "parameter_sweep_invalid_result_count": params.get("invalid_result_count"),
        "parameter_sweep_note": (
            "Raw width/displacement changes from contact parameter sweeps are not counted as robot-contact evidence "
            "unless the nearest-jaw metric also confirms actual touch."
        ),
        "best_contact_params": params.get("best_contact_params"),
        "best_touch_confirmed_contact_params": params.get("best_touch_confirmed_contact_params"),
        "local_attachment_was_run": bool(local_summary.get("status") == "SO101_LOCAL_GRIPPER_ATTACHMENT_V7_DONE"),
        "local_attachment_required": local_required,
        "minimum_local_attachment_strength_for_best_width_under_0_50": local_summary.get(
            "minimum_local_attachment_strength_for_best_width_under_0_50"
        ),
        "minimum_local_attachment_radius_for_best_width_under_0_50": local_summary.get(
            "minimum_local_attachment_radius_for_best_width_under_0_50"
        ),
        "true_robot_contact_folding_evidence_exists": true_robot_contact_folding,
        "ready_for_v8_policy_training": true_robot_contact_folding,
        "ready_for_v8_policy_training_basis": (
            "true only if pure robot contact or runtime contact-parameter pure contact both move towel and reduce width; "
            "local attachment evidence alone is not true robot-contact folding evidence"
        ),
        "do_not_train_in_v7": True,
        "paths": {
            "gripper_collision_inspection": str(V7_DIR / GRIPPER_COLLISION_JSON.name),
            "gripper_open_close_mapping": str(V7_DIR / GRIPPER_MAPPING_JSON.name),
            "edge_pinch_drag_sweep_summary": str(V7_DIR / EDGE_PINCH_DRAG_SUMMARY_JSON.name),
            "edge_pinch_drag_sweep_results": str(V7_DIR / "edge_pinch_drag_sweep_results.json"),
            "contact_parameter_sweep_summary": str(V7_DIR / CONTACT_PARAMETER_SUMMARY_JSON.name),
            "contact_parameter_sweep_results": str(V7_DIR / "contact_parameter_sweep_results.json"),
            "local_gripper_attachment_summary": str(V7_DIR / LOCAL_ATTACHMENT_JSON.name),
            "status_json": str(V7_DIR / STATUS_JSON.name),
        },
        "honest_note": (
            "V7 is contact-mechanics debugging only. Local gripper attachment is labeled assisted and does not count as true contact."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare local gripper attachment against pure SO-101 contact.")
    parser.add_argument("--task", type=str, default=TASK_ID)
    parser.add_argument("--usd_path", type=Path, default=CLEAN_TOWEL_USD_PATH)
    parser.add_argument("--out_dir", type=Path, default=V7_DIR)
    parser.add_argument("--radii", type=float, nargs="*", default=[0.01, 0.02, 0.03, 0.04])
    parser.add_argument("--strengths", type=float, nargs="*", default=[0.1, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--approach_steps", type=int, default=14)
    parser.add_argument("--drag_steps", type=int, default=26)
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
        if not candidate or not candidate.get("action_vector"):
            raise RuntimeError("Missing V6 recommended candidate.")
        motion_params = selected_motion_params(str(candidate["arm"]))
        scene = create_standalone_so101_clean_towel_scene(
            args,
            clean_usd_path=args.usd_path,
            towel_translation=recommended_towel_translation(),
        )
        scene.step_zero(20)
        start_points = scene.clean_towel.points_numpy().astype(np.float32)
        pure_summary = load_json_if_exists(EDGE_PINCH_DRAG_SUMMARY_JSON) or {}
        pure_best = pure_summary.get("verified_best_result") or pure_summary.get("best_result")
        results = []
        for radius in args.radii:
            for strength in args.strengths:
                result = local_attachment_trial(
                    scene,
                    start_points,
                    candidate,
                    motion_params,
                    radius=float(radius),
                    strength=float(strength),
                    approach_steps=args.approach_steps,
                    drag_steps=args.drag_steps,
                    edge_band=args.edge_band,
                )
                result.update(local_result_sanity(result))
                results.append(result)
        sanity_valid_results = [item for item in results if item.get("sanity_valid")]
        eligible_results = [item for item in results if item.get("eligible_local_attachment_result")]
        excluded_results = [item for item in results if not item.get("sanity_valid") or item.get("number_of_particles_attached", 0) <= 0]
        best = max(eligible_results, key=local_score) if eligible_results else None
        best_sane_result = max(sanity_valid_results, key=local_score) if sanity_valid_results else None
        successes = [item for item in eligible_results if item["best_width"] < 0.50]
        minimum_strength = None
        minimum_radius = None
        if successes:
            minimum_strength = float(min(item["strength"] for item in successes))
            minimum_radius = float(min(item["radius"] for item in successes if item["strength"] == minimum_strength))
        pure_disp = 0.0 if not pure_best else float(pure_best.get("selected_edge_particle_displacement", 0.0))
        pure_width = 0.68 if not pure_best else float(pure_best.get("best_width", 0.68))
        local_improves = bool(
            best
            and (
                float(best["selected_edge_particle_displacement"]) > pure_disp + 0.005
                or float(best["best_width"]) < pure_width - 0.01
            )
        )
        max_attached_fraction = float(max((item["attached_particle_fraction"] for item in results), default=0.0))
        eligible_max_attached_fraction = float(max((item["attached_particle_fraction"] for item in eligible_results), default=0.0))
        summary = {
            "status": "SO101_LOCAL_GRIPPER_ATTACHMENT_V7_DONE",
            "command_run": command_run_string(),
            "v6_recommended_towel_center": recommended_towel_center(),
            "towel_translation_used": [float(x) for x in recommended_towel_translation()],
            "v6_recommended_candidate": candidate,
            "motion_params": motion_params,
            "pure_contact_best_result": pure_best,
            "radii_tested": [float(x) for x in args.radii],
            "strengths_tested": [float(x) for x in args.strengths],
            "num_trials": int(len(results)),
            "total_particles": int(start_points.shape[0]),
            "max_attached_particle_fraction": max_attached_fraction,
            "eligible_local_attachment_trial_count": int(len(eligible_results)),
            "excluded_local_attachment_trial_count": int(len(excluded_results)),
            "excluded_local_attachment_trials": [
                {
                    "radius": item.get("radius"),
                    "strength": item.get("strength"),
                    "number_of_particles_attached": item.get("number_of_particles_attached"),
                    "sanity_invalid_reasons": item.get("sanity_invalid_reasons"),
                    "selected_edge_particle_displacement": item.get("selected_edge_particle_displacement"),
                    "best_width": item.get("best_width"),
                    "final_width": item.get("final_width"),
                }
                for item in excluded_results
            ],
            "eligible_max_attached_particle_fraction": eligible_max_attached_fraction,
            "best_result": best,
            "best_sane_result_including_zero_attachment": best_sane_result,
            "results": results,
            "minimum_local_attachment_strength_for_best_width_under_0_50": minimum_strength,
            "minimum_local_attachment_radius_for_best_width_under_0_50": minimum_radius,
            "local_attachment_improves_over_pure_contact": local_improves,
            "attachment_label": "gripper-attachment assisted, not true robot contact",
            "scene_info": standalone_robot_scene_info(scene),
            "local_gripper_attachment_summary_path": str(args.out_dir / LOCAL_ATTACHMENT_JSON.name),
            "status_json_path": str(args.out_dir / STATUS_JSON.name),
            "honest_note": (
                "This directly constrains only particles within a small radius of the jaw/finger contact region. "
                "It never attaches the whole towel and is not true contact evidence."
            ),
        }
        status = build_status(summary)
        write_json(args.out_dir / LOCAL_ATTACHMENT_JSON.name, summary, print_payload=False)
        write_json(args.out_dir / STATUS_JSON.name, status, print_payload=False)
        print(json.dumps({"local_gripper_attachment_summary": summary, "v7_status": status}, indent=2), flush=True)
    except Exception as exc:
        summary = {
            "status": "SO101_LOCAL_GRIPPER_ATTACHMENT_V7_FAILED",
            "command_run": command_run_string(),
            "exception_type": exc.__class__.__name__,
            "exception": repr(exc),
            "minimum_local_attachment_strength_for_best_width_under_0_50": None,
            "local_attachment_improves_over_pure_contact": False,
            "honest_note": "Local attachment comparison failed; no contact, folding, or training claim is made.",
        }
        status = build_status(summary)
        write_json(args.out_dir / LOCAL_ATTACHMENT_JSON.name, summary, print_payload=False)
        write_json(args.out_dir / STATUS_JSON.name, status, print_payload=False)
        print(json.dumps({"local_gripper_attachment_summary": summary, "v7_status": status}, indent=2), flush=True)
    finally:
        import os
        import sys

        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
