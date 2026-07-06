from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from isaaclab.app import AppLauncher

from clean_towel_v4_common import apply_action_step, reset_scene_to_points
from inspect_so101_gripper_collision_v7 import (
    ARM_NAMES,
    GRIPPER_MAPPING_JSON,
    V7_DIR,
    active_slice,
    body_names_for_arm,
    candidate_body_poses,
    clipped_action,
    gripper_index,
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
    standalone_robot_scene_info,
    write_json,
)


def proxy_gap_from_poses(poses: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [pose for pose in poses if pose.get("position") is not None]
    if len(valid) < 2:
        return {
            "gap_m": 0.0,
            "gap_source": "insufficient_candidate_bodies",
            "body_a": None,
            "body_b": None,
        }
    jaw_like = [pose for pose in valid if "jaw" in pose["body_name"].lower() or "finger" in pose["body_name"].lower()]
    gripper_like = [pose for pose in valid if "gripper" in pose["body_name"].lower()]
    pairs = []
    if jaw_like and gripper_like:
        for a in jaw_like:
            for b in gripper_like:
                pairs.append((a, b, "jaw_or_finger_to_gripper_body_origin"))
    else:
        for i, a in enumerate(valid):
            for b in valid[i + 1 :]:
                pairs.append((a, b, "candidate_body_origin_pair"))
    def quat_angle(a: dict[str, Any], b: dict[str, Any]) -> float | None:
        qa = a.get("orientation_xyzw")
        qb = b.get("orientation_xyzw")
        if qa is None or qb is None:
            return None
        qa_arr = np.asarray(qa, dtype=np.float32)
        qb_arr = np.asarray(qb, dtype=np.float32)
        qa_norm = float(np.linalg.norm(qa_arr))
        qb_norm = float(np.linalg.norm(qb_arr))
        if qa_norm < 1.0e-6 or qb_norm < 1.0e-6:
            return None
        qa_arr = qa_arr / qa_norm
        qb_arr = qb_arr / qb_norm
        dot = float(np.clip(abs(float(np.dot(qa_arr, qb_arr))), 0.0, 1.0))
        return float(2.0 * np.arccos(dot))

    best = None
    for a, b, source in pairs:
        pa = np.asarray(a["position"], dtype=np.float32)
        pb = np.asarray(b["position"], dtype=np.float32)
        gap = float(np.linalg.norm(pa - pb))
        angle = quat_angle(a, b)
        effective_proxy = gap if angle is None else gap + 0.03 * float(angle)
        item = {
            "gap_m": gap,
            "jaw_opening_rotation_proxy_rad": angle,
            "effective_gap_proxy_m": float(effective_proxy),
            "gap_source": source,
            "body_a": a["body_name"],
            "body_b": b["body_name"],
        }
        if best is None or effective_proxy > best["effective_gap_proxy_m"]:
            best = item
    return best or {
        "gap_m": 0.0,
        "jaw_opening_rotation_proxy_rad": None,
        "effective_gap_proxy_m": 0.0,
        "gap_source": "no_valid_pair",
        "body_a": None,
        "body_b": None,
    }


def base_action_for_arm(arm: str, context: str, candidate: dict[str, Any] | None) -> np.ndarray:
    action = np.zeros(12, dtype=np.float32)
    if context == "v6_reachable_action" and candidate and candidate.get("action_vector"):
        action = clipped_action(np.asarray(candidate["action_vector"], dtype=np.float32))
        if candidate.get("arm") != arm:
            # Mirror the active six-dimensional pose into the requested arm only as a stable
            # non-neutral posture for gap probing; the gripper index is overwritten during sweep.
            source_sl = active_slice(str(candidate.get("arm", arm)))
            target_sl = active_slice(arm)
            action = np.zeros(12, dtype=np.float32)
            action[target_sl] = np.asarray(candidate["action_vector"], dtype=np.float32)[source_sl]
    return clipped_action(action)


def measure_gap_for_value(
    scene: Any,
    start_points: np.ndarray,
    arm: str,
    base_action: np.ndarray,
    value: float,
    settle_steps: int,
) -> dict[str, Any]:
    reset_scene_to_points(scene, start_points)
    action = clipped_action(base_action)
    action[gripper_index(arm)] = float(value)
    for _ in range(int(settle_steps)):
        apply_action_step(scene, action)
    poses = candidate_body_poses(scene, arm)
    gap = proxy_gap_from_poses(poses)
    return {
        "gripper_action_value": float(value),
        "action_vector": [float(x) for x in action],
        "candidate_body_poses": poses,
        **gap,
    }


def summarize_arm_mapping(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {
            "gripper_open_close_mapping_known": False,
            "jaw_gap_changes_with_action": False,
            "open_value": None,
            "close_value": None,
            "gap_range_m": 0.0,
        }
    gaps = np.asarray([float(item["gap_m"]) for item in results], dtype=np.float32)
    proxy_gaps = np.asarray([float(item.get("effective_gap_proxy_m", item["gap_m"])) for item in results], dtype=np.float32)
    rotations = np.asarray(
        [
            0.0
            if item.get("jaw_opening_rotation_proxy_rad") is None
            else float(item["jaw_opening_rotation_proxy_rad"])
            for item in results
        ],
        dtype=np.float32,
    )
    values = np.asarray([float(item["gripper_action_value"]) for item in results], dtype=np.float32)
    min_idx = int(np.argmin(gaps))
    max_idx = int(np.argmax(gaps))
    gap_range = float(gaps[max_idx] - gaps[min_idx])
    proxy_min_idx = int(np.argmin(proxy_gaps))
    proxy_max_idx = int(np.argmax(proxy_gaps))
    proxy_range = float(proxy_gaps[proxy_max_idx] - proxy_gaps[proxy_min_idx])
    rotation_range = float(rotations.max() - rotations.min()) if rotations.size else 0.0
    body_gap_changes = bool(gap_range > 0.001)
    proxy_changes = bool(proxy_range > 0.001 or rotation_range > 0.02)
    changes = bool(body_gap_changes or proxy_changes)
    corr = 0.0
    signal = gaps if body_gap_changes else proxy_gaps
    if len(values) > 1 and float(np.std(values)) > 1.0e-6 and float(np.std(signal)) > 1.0e-6:
        corr = float(np.corrcoef(values, signal)[0, 1])
    open_idx = max_idx if body_gap_changes else proxy_max_idx
    close_idx = min_idx if body_gap_changes else proxy_min_idx
    return {
        "gripper_open_close_mapping_known": changes,
        "jaw_gap_changes_with_action": changes,
        "body_origin_gap_changes_with_action": body_gap_changes,
        "opening_proxy_changes_with_action": proxy_changes,
        "opening_proxy_source": "body_origin_gap" if body_gap_changes else "jaw_to_gripper_relative_rotation",
        "open_value": float(values[open_idx]),
        "close_value": float(values[close_idx]),
        "open_gap_m": float(gaps[open_idx]),
        "close_gap_m": float(gaps[close_idx]),
        "open_effective_gap_proxy_m": float(proxy_gaps[open_idx]),
        "close_effective_gap_proxy_m": float(proxy_gaps[close_idx]),
        "open_rotation_proxy_rad": float(rotations[open_idx]),
        "close_rotation_proxy_rad": float(rotations[close_idx]),
        "gap_range_m": gap_range,
        "effective_gap_proxy_range_m": proxy_range,
        "rotation_proxy_range_rad": rotation_range,
        "gap_increases_with_larger_action": bool(corr > 0.25),
        "gap_value_correlation": corr,
        "open_result": results[open_idx],
        "close_result": results[close_idx],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep SO-101 gripper action values and measure open/close mapping.")
    parser.add_argument("--task", type=str, default=TASK_ID)
    parser.add_argument("--usd_path", type=Path, default=CLEAN_TOWEL_USD_PATH)
    parser.add_argument("--out_dir", type=Path, default=V7_DIR)
    parser.add_argument("--settle_steps", type=int, default=18)
    parser.add_argument(
        "--values",
        type=float,
        nargs="*",
        default=[-0.05, 0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90],
    )
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
        scene = create_standalone_so101_clean_towel_scene(
            args,
            clean_usd_path=args.usd_path,
            towel_translation=recommended_towel_translation(),
        )
        scene.step_zero(20)
        start_points = scene.clean_towel.points_numpy().astype(np.float32)
        contexts = ["neutral_non_gripper_pose", "v6_reachable_action"]
        by_arm = {}
        for arm in ARM_NAMES:
            arm_contexts = {}
            for context in contexts:
                base_action = base_action_for_arm(arm, context, candidate)
                results = [
                    measure_gap_for_value(scene, start_points, arm, base_action, value, args.settle_steps)
                    for value in args.values
                ]
                arm_contexts[context] = {
                    "base_action": [float(x) for x in base_action],
                    "sweep_results": results,
                    "mapping": summarize_arm_mapping(results),
                }
            best_context_name = max(
                contexts,
                key=lambda name: float(arm_contexts[name]["mapping"].get("effective_gap_proxy_range_m", 0.0)),
            )
            by_arm[arm] = {
                "body_names": body_names_for_arm(scene.left_arm if arm == "left" else scene.right_arm),
                "candidate_contexts": arm_contexts,
                "selected_context": best_context_name,
                "selected_mapping": arm_contexts[best_context_name]["mapping"],
            }

        global_known = bool(all(by_arm[arm]["selected_mapping"]["gripper_open_close_mapping_known"] for arm in ARM_NAMES))
        summary = {
            "status": "SO101_GRIPPER_OPEN_CLOSE_MAPPING_V7_DONE",
            "command_run": command_run_string(),
            "v6_recommended_towel_center": recommended_towel_center(),
            "towel_translation_used": [float(x) for x in recommended_towel_translation()],
            "gripper_values_tested": [float(x) for x in args.values],
            "settle_steps_per_value": int(args.settle_steps),
            "v6_recommended_candidate": candidate,
            "by_arm": by_arm,
            "gripper_open_close_mapping_known": global_known,
            "jaw_gap_changes_with_action": bool(
                any(by_arm[arm]["selected_mapping"]["jaw_gap_changes_with_action"] for arm in ARM_NAMES)
            ),
            "open_value": None if not global_known else float(by_arm["right"]["selected_mapping"]["open_value"]),
            "close_value": None if not global_known else float(by_arm["right"]["selected_mapping"]["close_value"]),
            "scene_info": standalone_robot_scene_info(scene),
            "gripper_open_close_mapping_summary_path": str(args.out_dir / GRIPPER_MAPPING_JSON.name),
            "honest_note": (
                "This uses body-origin gap as the gripper opening proxy. It only identifies actuation direction; "
                "it does not claim contact, towel motion, folding, or train any policy."
            ),
        }
        write_json(args.out_dir / GRIPPER_MAPPING_JSON.name, summary, print_payload=True)
    except Exception as exc:
        summary = {
            "status": "SO101_GRIPPER_OPEN_CLOSE_MAPPING_V7_FAILED",
            "command_run": command_run_string(),
            "exception_type": exc.__class__.__name__,
            "exception": repr(exc),
            "gripper_open_close_mapping_known": False,
            "jaw_gap_changes_with_action": False,
            "open_value": None,
            "close_value": None,
            "honest_note": "Mapping failed; no contact, towel motion, folding, or training claim is made.",
        }
        write_json(args.out_dir / GRIPPER_MAPPING_JSON.name, summary, print_payload=True)
    finally:
        import os
        import sys

        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
