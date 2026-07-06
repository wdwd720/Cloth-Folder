from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher

from clean_towel_v5_contact_common import (
    CONTACT_REPLAY_SUMMARY,
    CONTACT_SWEEP_JSON,
    V5_COMPILE_COMMAND,
    V5_DIR,
    command_for_script,
    generate_rerun_points,
    load_json_if_exists,
    reset_scene_to_points,
    run_action_sequence,
    write_v5_status,
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
    parser = argparse.ArgumentParser(description="Replay the best pure-contact SO-101 gripper candidate.")
    parser.add_argument("--task", type=str, default=TASK_ID)
    parser.add_argument("--usd_path", type=Path, default=CLEAN_TOWEL_USD_PATH)
    parser.add_argument("--out_dir", type=Path, default=V5_DIR)
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
        sweep = load_json_if_exists(CONTACT_SWEEP_JSON)
        if not sweep or not sweep.get("best_candidate"):
            summary = {
                "status": "CLEAN_TOWEL_ROBOT_CONTACT_BEST_V5_SKIPPED",
                "command_run": command_run_string(),
                "reason": "missing_best_candidate_from_contact_sweep",
                "honest_note": "No pure-contact replay was run because the sweep did not produce a candidate.",
            }
            write_json(CONTACT_REPLAY_SUMMARY, summary, print_payload=True)
            write_v5_status([V5_COMPILE_COMMAND, command_for_script("scripts/tera/run_clean_towel_robot_contact_fold_best_v5.py")])
            return
        candidate = sweep["best_candidate"]
        scene = create_standalone_so101_clean_towel_scene(args, clean_usd_path=args.usd_path)
        start_points = scene.clean_towel.points_numpy().astype(np.float32)
        reset_scene_to_points(scene, start_points)
        trace = run_action_sequence(scene, candidate["action_sequence"])
        points_trace = np.concatenate([start_points[None, ...], trace["points_trace"]], axis=0)
        widths = np.concatenate([[float(size_metrics(start_points)["width_x"])], trace["widths"]]).astype(np.float32)
        best_step = int(np.argmin(widths))
        best_width = float(widths[best_step])
        final_width = float(widths[-1])
        capture_path = args.out_dir / "robot_contact_fold_best_capture_v5.npz"
        np.savez_compressed(
            capture_path,
            points_trace=points_trace.astype(np.float32),
            widths=widths,
            jaw_trace=trace["jaw_trace"].astype(np.float32),
            best_step=np.asarray([best_step], dtype=np.int32),
        )
        rrd_path = args.out_dir / "robot_contact_fold_best_actual_state_v5.rrd"
        rerun = generate_rerun_points(
            rrd_path,
            "clean_towel_v5_robot_contact_best",
            [
                ("start", points_trace[0]),
                (f"best_step_{best_step:03d}", points_trace[best_step]),
                ("final", points_trace[-1]),
            ],
        )
        summary = {
            "status": "CLEAN_TOWEL_ROBOT_CONTACT_BEST_V5_DONE",
            "command_run": command_run_string(),
            "candidate": candidate,
            "capture_path": str(capture_path),
            "particle_space_assist_used": False,
            "gripper_attachment_used": False,
            "start_width": float(widths[0]),
            "best_width": best_width,
            "best_step": best_step,
            "final_width": final_width,
            "robot_contact_width_reduction_success": bool(best_width < 0.62),
            "strong_robot_contact_success": bool(best_width < 0.55),
            "scene_info": standalone_robot_scene_info(scene),
            **rerun,
            "honest_note": "This replay uses only physical SO-101 robot contact. No particle attachment or towel particle assist is applied.",
        }
        write_json(CONTACT_REPLAY_SUMMARY, summary, print_payload=True)
        write_v5_status(
            [
                V5_COMPILE_COMMAND,
                command_for_script("scripts/tera/test_clean_towel_gripper_contact_sweep_v5.py"),
                command_run_string(),
            ]
        )
    except Exception as exc:
        summary = {
            "status": "CLEAN_TOWEL_ROBOT_CONTACT_BEST_V5_FAILED",
            "command_run": command_run_string(),
            "exception_type": exc.__class__.__name__,
            "exception": repr(exc),
            "honest_note": "No pure-contact replay success is claimed because replay failed.",
        }
        write_json(CONTACT_REPLAY_SUMMARY, summary, print_payload=True)
        write_v5_status([V5_COMPILE_COMMAND, command_for_script("scripts/tera/run_clean_towel_robot_contact_fold_best_v5.py")])
    finally:
        import os
        import sys

        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
