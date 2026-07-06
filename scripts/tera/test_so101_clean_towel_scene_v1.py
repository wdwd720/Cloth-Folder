from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher

from so101_clean_towel_scene_utils_v1 import (
    CLEAN_TOWEL_USD_PATH,
    OUT_DIR,
    TASK_ID,
    command_run_string,
    create_standalone_so101_clean_towel_scene,
    force_headless_no_cameras,
    looks_rectangular,
    size_metrics,
    standalone_robot_scene_info,
    write_json,
)


DEFAULT_SUMMARY_PATH = OUT_DIR / "test_so101_clean_towel_scene_v1_summary.json"
DEFAULT_CAPTURE_PATH = OUT_DIR / "test_so101_clean_towel_scene_v1_capture.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test SO-101 dual arms with clean rectangular towel v2.")
    parser.add_argument("--task", type=str, default=TASK_ID)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--usd_path", type=Path, default=CLEAN_TOWEL_USD_PATH)
    parser.add_argument("--out_dir", type=Path, default=OUT_DIR)
    parser.add_argument("--width_x", type=float, default=0.68)
    parser.add_argument("--height_y", type=float, default=0.38)
    parser.add_argument("--tolerance", type=float, default=0.03)
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
        scene = create_standalone_so101_clean_towel_scene(
            args,
            clean_usd_path=args.usd_path,
        )
        clean_towel = scene.clean_towel

        scene_info = standalone_robot_scene_info(scene)
        reset_points = clean_towel.points_numpy()
        reset_metrics = size_metrics(reset_points)
        clean_towel_exists = bool(reset_metrics["num_particles"] > 0)

        scene.step_zero(args.steps)

        after_points = clean_towel.points_numpy()
        after_metrics = size_metrics(after_points)
        rectangular = looks_rectangular(after_metrics, args.width_x, args.height_y, args.tolerance)
        success = bool(
            scene_info["left_arm_exists"]
            and scene_info["right_arm_exists"]
            and scene_info["action_shape_is_12"]
            and clean_towel_exists
            and rectangular
        )

        capture_path = args.out_dir / DEFAULT_CAPTURE_PATH.name
        np.savez_compressed(
            capture_path,
            reset_points=reset_points.astype(np.float32),
            after_100_points=after_points.astype(np.float32),
            target_width_x=np.asarray([args.width_x], dtype=np.float32),
            target_height_y=np.asarray([args.height_y], dtype=np.float32),
        )

        summary = {
            "status": "SO101_CLEAN_TOWEL_SCENE_V1_TEST_OK" if success else "SO101_CLEAN_TOWEL_SCENE_V1_TEST_FAILED",
            "command_run": command_run_string(),
            "task": args.task,
            "device": str(args.device),
            "clean_towel_usd_path": str(args.usd_path),
            "capture_path": str(capture_path),
            "clean_towel_root_path": clean_towel.root_path,
            "clean_towel_mesh_path": clean_towel.mesh_path,
            "clean_towel_particle_system_path": clean_towel.particle_system_path,
            "ground_prim_path": scene.ground_prim_path,
            "left_robot_prim_path": scene.left_prim_path,
            "right_robot_prim_path": scene.right_prim_path,
            "physics_scene_path": clean_towel.physics_scene_path,
            "original_garment_moved_out_of_workspace": None,
            "num_steps": int(args.steps),
            "left_arm_exists": bool(scene_info["left_arm_exists"]),
            "right_arm_exists": bool(scene_info["right_arm_exists"]),
            "action_shape": scene_info["action_shape"],
            "action_dim": int(scene_info["action_dim"]),
            "action_shape_is_12": bool(scene_info["action_shape_is_12"]),
            "left_right_so101_control_exists": bool(scene_info["left_right_so101_control_exists"]),
            "clean_towel_exists": clean_towel_exists,
            "num_particles": int(reset_metrics["num_particles"]),
            "reset_metrics": reset_metrics,
            "after_100_zero_steps_metrics": after_metrics,
            "target_width_x": float(args.width_x),
            "target_height_y": float(args.height_y),
            "width_error_after_100": abs(float(after_metrics["width_x"]) - float(args.width_x)),
            "height_error_after_100": abs(float(after_metrics["height_y"]) - float(args.height_y)),
            "looks_rectangular_by_metric": bool(rectangular),
            "scene_info": scene_info,
            "honest_note": (
                "This standalone reversible path imports the SO-101 left/right arm USDs and the verified clean towel v2 USD into one Isaac scene. "
                "It does not modify or depend on the original FoldCloth task files."
            ),
        }
        write_json(args.out_dir / DEFAULT_SUMMARY_PATH.name, summary, print_payload=True)

    except Exception as exc:
        summary = {
            "status": "SO101_CLEAN_TOWEL_SCENE_V1_TEST_FAILED",
            "command_run": command_run_string(),
            "task": args.task,
            "device": str(getattr(args, "device", "unknown")),
            "clean_towel_usd_path": str(args.usd_path),
            "exception_type": exc.__class__.__name__,
            "exception": repr(exc),
            "honest_note": "The smoke test did not complete; no scene success is claimed.",
        }
        write_json(args.out_dir / DEFAULT_SUMMARY_PATH.name, summary, print_payload=True)
    finally:
        import os
        import sys

        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
