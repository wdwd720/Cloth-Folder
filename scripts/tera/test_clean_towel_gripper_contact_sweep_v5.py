from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher

from clean_towel_v5_contact_common import (
    CONTACT_SWEEP_JSON,
    CONTACT_SWEEP_NPZ,
    V5_DIR,
    build_contact_candidates,
    contact_score,
    edge_summary,
    jaw_positions,
    reset_scene_to_points,
    run_action_sequence,
    summarize_contact_trial,
    write_v5_status,
    V5_COMPILE_COMMAND,
    command_for_script,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep SO-101 gripper contact candidates against clean towel edges.")
    parser.add_argument("--task", type=str, default=TASK_ID)
    parser.add_argument("--usd_path", type=Path, default=CLEAN_TOWEL_USD_PATH)
    parser.add_argument("--out_dir", type=Path, default=V5_DIR)
    parser.add_argument("--max_candidates", type=int, default=96)
    parser.add_argument("--settle_steps", type=int, default=20)
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
    scene = None
    try:
        scene = create_standalone_so101_clean_towel_scene(args, clean_usd_path=args.usd_path)
        scene.step_zero(args.settle_steps)
        start_points = scene.clean_towel.points_numpy().astype(np.float32)
        start_metrics = size_metrics(start_points)
        candidates = build_contact_candidates(args.max_candidates)
        results = []
        metric_arrays = {
            "candidate_id": [],
            "best_width": [],
            "final_width": [],
            "width_reduction": [],
            "edge_displacement": [],
            "nearest_distance": [],
        }
        for candidate in candidates:
            reset_scene_to_points(scene, start_points)
            trace = run_action_sequence(scene, candidate["action_sequence"])
            active_jaw_index = 0 if candidate["arm"] == "left" else 1
            result = summarize_contact_trial(candidate, start_points, trace, active_jaw_index, args.edge_band)
            results.append(result)
            metric_arrays["candidate_id"].append(int(result["candidate_id"]))
            metric_arrays["best_width"].append(float(result["best_width"]))
            metric_arrays["final_width"].append(float(result["final_width"]))
            metric_arrays["width_reduction"].append(float(result["width_reduction"]))
            metric_arrays["edge_displacement"].append(float(result["selected_edge_particle_displacement"]))
            metric_arrays["nearest_distance"].append(float(result["nearest_jaw_to_selected_edge_m"]))
        best_candidate = max(results, key=contact_score) if results else None
        contact_moves_towel = bool(any(item["contact_moves_towel"] for item in results))
        contact_fold_candidate = bool(any(item["contact_fold_candidate"] for item in results))
        jaws, jaw_info = jaw_positions(scene)
        np.savez_compressed(
            CONTACT_SWEEP_NPZ,
            **{key: np.asarray(value, dtype=np.float32 if key != "candidate_id" else np.int32) for key, value in metric_arrays.items()},
            start_points=start_points,
            final_jaw_positions=jaws.astype(np.float32),
        )
        summary = {
            "status": "CLEAN_TOWEL_GRIPPER_CONTACT_SWEEP_V5_DONE",
            "command_run": command_run_string(),
            "num_candidates": int(len(results)),
            "settle_steps": int(args.settle_steps),
            "edge_band": float(args.edge_band),
            "start_metrics": start_metrics,
            "edge_summary": edge_summary(start_points, args.edge_band),
            "scene_info": standalone_robot_scene_info(scene),
            "jaw_info": jaw_info,
            "contact_moves_towel": contact_moves_towel,
            "contact_fold_candidate": contact_fold_candidate,
            "best_candidate": best_candidate,
            "top_candidates": sorted(results, key=contact_score, reverse=True)[:10],
            "results_json_path": str(CONTACT_SWEEP_JSON),
            "results_npz_path": str(CONTACT_SWEEP_NPZ),
            "honest_note": "This sweep uses only SO-101 joint targets and physical contact. No towel particles are attached or directly assisted.",
        }
        write_json(CONTACT_SWEEP_JSON, summary, print_payload=True)
        write_v5_status([V5_COMPILE_COMMAND, command_for_script("scripts/tera/test_clean_towel_gripper_contact_sweep_v5.py")])
    except Exception as exc:
        summary = {
            "status": "CLEAN_TOWEL_GRIPPER_CONTACT_SWEEP_V5_FAILED",
            "command_run": command_run_string(),
            "exception_type": exc.__class__.__name__,
            "exception": repr(exc),
            "honest_note": "No contact result is claimed because the sweep failed.",
        }
        write_json(CONTACT_SWEEP_JSON, summary, print_payload=True)
        write_v5_status([V5_COMPILE_COMMAND, command_for_script("scripts/tera/test_clean_towel_gripper_contact_sweep_v5.py")])
    finally:
        import os
        import sys

        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
