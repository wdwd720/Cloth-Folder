from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher

from clean_towel_v5_contact_common import (
    ATTACHMENT_SUMMARY,
    V5_COMPILE_COMMAND,
    V5_DIR,
    command_for_script,
    generate_rerun_points,
    reset_scene_to_points,
    run_attachment_episode,
    write_v5_status,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run gripper-attachment assisted clean towel folding curriculum.")
    parser.add_argument("--task", type=str, default=TASK_ID)
    parser.add_argument("--usd_path", type=Path, default=CLEAN_TOWEL_USD_PATH)
    parser.add_argument("--out_dir", type=Path, default=V5_DIR)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--attachment_band_x", type=float, default=0.18)
    parser.add_argument("--attachment_strengths", type=float, nargs="*", default=[1.0, 0.75, 0.5, 0.25, 0.1, 0.0])
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
        start_points = scene.clean_towel.points_numpy().astype(np.float32)
        results = []
        capture = {}
        rerun_frames = []
        for strength in args.attachment_strengths:
            reset_scene_to_points(scene, start_points)
            episode = run_attachment_episode(
                scene,
                start_points,
                attachment_strength=float(strength),
                steps=args.steps,
                band_x=args.attachment_band_x,
            )
            key = str(float(strength)).replace(".", "_")
            capture[f"points_trace_strength_{key}"] = episode["points_trace"]
            capture[f"widths_strength_{key}"] = episode["widths"]
            capture[f"actions_strength_{key}"] = episode["actions"]
            best_step = int(episode["best_step"])
            rerun_frames.append((f"strength_{key}_start", episode["points_trace"][0]))
            rerun_frames.append((f"strength_{key}_best", episode["points_trace"][best_step]))
            rerun_frames.append((f"strength_{key}_final", episode["points_trace"][-1]))
            results.append(
                {
                    "attachment_strength": float(strength),
                    "start_width": episode["start_width"],
                    "best_width": episode["best_width"],
                    "best_step": best_step,
                    "final_width": episode["final_width"],
                    "attached_particle_count": episode["attached_particle_count"],
                    "attached_particle_fraction": episode["attached_particle_fraction"],
                    "attachment_band_x": episode["attachment_band_x"],
                    "best_under_0_60": bool(episode["best_width"] < 0.60),
                    "best_under_0_50": bool(episode["best_width"] < 0.50),
                    "final_under_0_55": bool(episode["final_width"] < 0.55),
                }
            )
        capture_path = args.out_dir / "gripper_attachment_curriculum_capture_v5.npz"
        np.savez_compressed(capture_path, **capture)
        rrd_path = args.out_dir / "gripper_attachment_curriculum_actual_state_v5.rrd"
        rerun = generate_rerun_points(rrd_path, "clean_towel_v5_gripper_attachment", rerun_frames)
        useful = [item["attachment_strength"] for item in results if item["best_under_0_50"]]
        min_strength = None if not useful else float(min(useful))
        summary = {
            "status": "CLEAN_TOWEL_GRIPPER_ATTACHMENT_CURRICULUM_V5_DONE",
            "command_run": command_run_string(),
            "capture_path": str(capture_path),
            "attachment_is_assisted": True,
            "particle_space_assist_used": True,
            "attachment_model": "local_left_edge_band_particles_follow_assisted_fold_gripper_path",
            "attachment_strengths": [float(x) for x in args.attachment_strengths],
            "results": results,
            "minimum_attachment_strength_for_best_width_under_0_50": min_strength,
            "scene_info": standalone_robot_scene_info(scene),
            **rerun,
            "honest_note": (
                "This is gripper-attachment assisted manipulation. It directly constrains a local left-edge particle band; "
                "it is not evidence of pure physical robot-contact grasping."
            ),
        }
        write_json(ATTACHMENT_SUMMARY, summary, print_payload=True)
        write_v5_status(
            [
                V5_COMPILE_COMMAND,
                command_for_script("scripts/tera/test_clean_towel_gripper_contact_sweep_v5.py"),
                command_for_script("scripts/tera/run_clean_towel_robot_contact_fold_best_v5.py"),
                command_run_string(),
            ]
        )
    except Exception as exc:
        summary = {
            "status": "CLEAN_TOWEL_GRIPPER_ATTACHMENT_CURRICULUM_V5_FAILED",
            "command_run": command_run_string(),
            "exception_type": exc.__class__.__name__,
            "exception": repr(exc),
            "honest_note": "No gripper-attachment conclusion is claimed because the curriculum failed.",
        }
        write_json(ATTACHMENT_SUMMARY, summary, print_payload=True)
        write_v5_status([V5_COMPILE_COMMAND, command_for_script("scripts/tera/run_clean_towel_gripper_attachment_fold_v5.py")])
    finally:
        import os
        import sys

        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
