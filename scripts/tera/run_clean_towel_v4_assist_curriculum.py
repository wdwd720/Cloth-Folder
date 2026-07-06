from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher

from clean_towel_v4_common import (
    CURRICULUM_SUMMARY_PATH,
    POLICY_PATH,
    V4_COMPILE_COMMAND,
    V4_DIR,
    command_for_script,
    command_run_string,
    generate_rerun_points,
    load_policy_bundle,
    reset_scene_to_points,
    run_policy_episode,
    write_v4_status,
)
from so101_clean_towel_scene_utils_v1 import (
    CLEAN_TOWEL_USD_PATH,
    TASK_ID,
    create_standalone_so101_clean_towel_scene,
    force_headless_no_cameras,
    standalone_robot_scene_info,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate V4 policy with particle-assist curriculum ratios.")
    parser.add_argument("--task", type=str, default=TASK_ID)
    parser.add_argument("--usd_path", type=Path, default=CLEAN_TOWEL_USD_PATH)
    parser.add_argument("--policy_path", type=Path, default=POLICY_PATH)
    parser.add_argument("--out_dir", type=Path, default=V4_DIR)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--assist_ratios", type=float, nargs="*", default=[1.0, 0.75, 0.50, 0.25, 0.0])
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
        bundle = load_policy_bundle(args.policy_path)
        scene = create_standalone_so101_clean_towel_scene(args, clean_usd_path=args.usd_path)
        scene_info = standalone_robot_scene_info(scene)
        initial_points = scene.clean_towel.points_numpy().astype(np.float32)
        results = []
        capture = {}
        rerun_frames = []
        for ratio in args.assist_ratios:
            reset_scene_to_points(scene, initial_points)
            episode = run_policy_episode(scene, bundle, steps=args.steps, assist_ratio=float(ratio))
            ratio_key = str(float(ratio)).replace(".", "_")
            capture[f"points_trace_ratio_{ratio_key}"] = episode["points_trace"]
            capture[f"widths_ratio_{ratio_key}"] = episode["widths"]
            capture[f"actions_ratio_{ratio_key}"] = episode["actions"]
            best_step = int(episode["best_step"])
            rerun_frames.append((f"ratio_{ratio_key}_start", episode["points_trace"][0]))
            rerun_frames.append((f"ratio_{ratio_key}_best", episode["points_trace"][best_step]))
            rerun_frames.append((f"ratio_{ratio_key}_final", episode["points_trace"][-1]))
            results.append(
                {
                    "assist_ratio": float(ratio),
                    "start_width": episode["start_width"],
                    "best_width": episode["best_width"],
                    "best_step": best_step,
                    "final_width": episode["final_width"],
                    "best_under_0_60": bool(episode["best_width"] < 0.60),
                    "best_under_0_50": bool(episode["best_width"] < 0.50),
                    "final_under_0_55": bool(episode["final_width"] < 0.55),
                }
            )
        capture_path = args.out_dir / "assist_curriculum_capture_v4.npz"
        np.savez_compressed(capture_path, **capture)
        rrd_path = args.out_dir / "assist_curriculum_actual_state_v4.rrd"
        rerun = generate_rerun_points(rrd_path, "clean_towel_v4_assist_curriculum", rerun_frames)
        minimum_assist = None
        for item in sorted(results, key=lambda x: x["assist_ratio"]):
            if item["best_under_0_50"]:
                minimum_assist = item["assist_ratio"]
                break
        summary = {
            "status": "CLEAN_TOWEL_V4_ASSIST_CURRICULUM_DONE",
            "command_run": command_run_string(),
            "policy_path": str(args.policy_path),
            "capture_path": str(capture_path),
            "assist_ratios": [float(x) for x in args.assist_ratios],
            "results": results,
            "minimum_assist_ratio_with_best_width_under_0_50": minimum_assist,
            "scene_info": scene_info,
            **rerun,
            "honest_note": (
                "Assist ratios greater than 0 directly blend clean towel particles toward the assisted fold primitive. "
                "The 0.0 row is the pure no-assist policy result."
            ),
        }
        write_json(CURRICULUM_SUMMARY_PATH, summary, print_payload=True)
        commands = [
            V4_COMPILE_COMMAND,
            command_for_script("scripts/tera/train_clean_towel_policy_v4.py"),
            command_for_script("scripts/tera/eval_clean_towel_policy_v4_no_assist.py"),
            command_for_script("scripts/tera/eval_clean_towel_policy_v4_with_stop.py"),
            command_run_string(),
        ]
        write_v4_status(commands)
    except Exception as exc:
        summary = {
            "status": "CLEAN_TOWEL_V4_ASSIST_CURRICULUM_FAILED",
            "command_run": command_run_string(),
            "exception_type": exc.__class__.__name__,
            "exception": repr(exc),
            "honest_note": "Curriculum did not complete; no assist requirement conclusion is claimed.",
        }
        write_json(CURRICULUM_SUMMARY_PATH, summary, print_payload=True)
        write_v4_status([V4_COMPILE_COMMAND, command_run_string()])
    finally:
        import os
        import sys

        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
