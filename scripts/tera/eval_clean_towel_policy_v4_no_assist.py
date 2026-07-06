from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher

from clean_towel_v4_common import (
    NO_ASSIST_SUMMARY_PATH,
    POLICY_PATH,
    V4_DIR,
    command_run_string,
    generate_rerun_points,
    load_policy_bundle,
    run_policy_episode,
    success_flags,
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
    parser = argparse.ArgumentParser(description="Evaluate V4 policy with no particle-space assist.")
    parser.add_argument("--task", type=str, default=TASK_ID)
    parser.add_argument("--usd_path", type=Path, default=CLEAN_TOWEL_USD_PATH)
    parser.add_argument("--policy_path", type=Path, default=POLICY_PATH)
    parser.add_argument("--out_dir", type=Path, default=V4_DIR)
    parser.add_argument("--steps", type=int, default=100)
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
        episode = run_policy_episode(scene, bundle, steps=args.steps, assist_ratio=0.0)
        points = episode["points_trace"]
        capture_path = args.out_dir / "eval_no_assist_capture_v4.npz"
        np.savez_compressed(
            capture_path,
            points_trace=points,
            actions=episode["actions"],
            widths=episode["widths"],
            best_step=np.asarray([episode["best_step"]], dtype=np.int32),
        )
        rrd_path = args.out_dir / "eval_no_assist_actual_state_v4.rrd"
        rerun = generate_rerun_points(
            rrd_path,
            "clean_towel_v4_no_assist",
            [
                ("start", points[0]),
                (f"best_step_{episode['best_step']:03d}", points[episode["best_step"]]),
                ("final", points[-1]),
            ],
        )
        flags = success_flags(episode["best_width"], episode["final_width"])
        summary = {
            "status": "CLEAN_TOWEL_POLICY_V4_NO_ASSIST_EVAL_DONE",
            "command_run": command_run_string(),
            "policy_path": str(args.policy_path),
            "capture_path": str(capture_path),
            "particle_space_assist_used": False,
            "autonomous_robot_action_only": True,
            "start_width": episode["start_width"],
            "best_width": episode["best_width"],
            "best_step": int(episode["best_step"]),
            "final_width": episode["final_width"],
            **flags,
            "start_metrics": episode["start_metrics"],
            "best_metrics": episode["best_metrics"],
            "final_metrics": episode["final_metrics"],
            "width_trace": episode["widths"].tolist(),
            "jaw_info": episode["jaw_info"],
            "scene_info": scene_info,
            **rerun,
            "honest_note": "This evaluation uses only policy-predicted SO-101 joint targets. No towel particles are assisted or directly edited.",
        }
        write_json(NO_ASSIST_SUMMARY_PATH, summary, print_payload=True)
    except Exception as exc:
        summary = {
            "status": "CLEAN_TOWEL_POLICY_V4_NO_ASSIST_EVAL_FAILED",
            "command_run": command_run_string(),
            "exception_type": exc.__class__.__name__,
            "exception": repr(exc),
            "honest_note": "No autonomous result is claimed because evaluation failed.",
        }
        write_json(NO_ASSIST_SUMMARY_PATH, summary, print_payload=True)
    finally:
        import os
        import sys

        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
