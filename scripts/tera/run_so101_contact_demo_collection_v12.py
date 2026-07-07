"""V12 SO-101 contact demo collection.

Re-runs the winning physics-driven SO-101 fingertip-proxy fold in the real SO-101
scene and records per-step demonstration data: compact state, action, and contact
metrics (towel width before/after, edge displacement, particle velocity, touch).
Saves an npz + manifest to the V12 demos dir (Modal Volume only, never git).

HONESTY: the fold is produced by a physics-driven fingertip PROXY, and the joint
`action` labels are the SAME synthetic SO-101 joint targets used in v0/v1 (there
is no live SO-101 reach/IK on Modal). Therefore these are contact-VALIDATED fold
demos with SYNTHETIC joint labels, NOT true robot-joint-action contact demos:
    robot_contact_data_ready = false
so downstream Training v2 will honestly skip real robot-contact training.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from isaaclab.app import AppLauncher

import clean_towel_contact_v11_common as c11
import test_clean_towel_primitive_contact_v8 as v8
from clean_towel_v4_common import (
    compact_state,
    jaw_positions_from_scene,
    phase_id_for_step,
    progress_for_step,
    synthetic_action,
)
from clean_towel_v5_contact_common import edge_center_displacement
from inspect_so101_gripper_collision_v7 import recommended_towel_translation
from so101_clean_towel_scene_utils_v1 import (
    CLEAN_TOWEL_USD_PATH,
    command_run_string,
    create_standalone_so101_clean_towel_scene,
    force_headless_no_cameras,
    initialize_clean_towel_handle_for_sim,
    size_metrics,
    write_json,
)
from test_so101_fingertip_physics_drive_v12 import _ProxyScene, PROXY_PATH

V12_DIR = Path("/workspace/leisaac/tera_checkpoints/clean_towel_v12")
DEMO_DIR = V12_DIR / "demos"
DEMO_NPZ = DEMO_DIR / "so101_contact_demos_v12.npz"
DEMO_MANIFEST = DEMO_DIR / "so101_contact_demos_manifest_v12.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V12 SO-101 contact demo collection.")
    parser.add_argument("--usd_path", type=Path, default=CLEAN_TOWEL_USD_PATH)
    parser.add_argument("--out_dir", type=Path, default=V12_DIR)
    parser.add_argument("--edge", type=str, default="right")
    parser.add_argument("--edge_band", type=float, default=0.035)
    parser.add_argument("--settle_initial_steps", type=int, default=20)
    parser.add_argument("--num_episodes", type=int, default=3)
    parser.add_argument("--fold_steps", type=int, default=100)
    parser.add_argument("--primitive_radius", type=float, default=0.02)
    parser.add_argument("--primitive_height", type=float, default=0.012)
    parser.add_argument("--penetration_depth", type=float, default=0.05)
    parser.add_argument("--drag_distance", type=float, default=0.28)
    parser.add_argument("--drag_height_delta", type=float, default=0.0)
    parser.add_argument("--outside_margin", type=float, default=0.05)
    parser.add_argument("--proxy_contact_offset", type=float, default=0.012)
    parser.add_argument("--proxy_rest_offset", type=float, default=0.0)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    force_headless_no_cameras(args)
    return args


def main() -> None:
    args = parse_args()
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    app = AppLauncher(args).app  # noqa: F841

    manifest: dict[str, Any] = {
        "status": "V12_SO101_DEMO_COLLECTION_STARTED",
        "command_run": command_run_string(),
        "demo_type": "so101_scene_physics_driven_fingertip_proxy_fold",
        "joint_action_labels": "synthetic_so101_joint_targets (same as v0/v1)",
        "true_robot_joint_action_contact": False,
        "robot_contact_data_ready": False,
        "episodes": [],
    }
    try:
        from isaacsim.core.utils.stage import get_current_stage, update_stage

        scene = create_standalone_so101_clean_towel_scene(
            args, clean_usd_path=args.usd_path, towel_translation=recommended_towel_translation()
        )
        scene.step_zero(int(args.settle_initial_steps))
        v8.spawn_kinematic_sphere(
            PROXY_PATH, radius=float(args.primitive_radius),
            contact_offset=float(args.proxy_contact_offset), rest_offset=float(args.proxy_rest_offset),
            static_friction=1.6, dynamic_friction=1.2, translation=(1.25, 0.0, 0.30),
        )
        update_stage()
        scene.sim.reset()
        scene.left_arm.update(scene.sim.get_physics_dt())
        scene.right_arm.update(scene.sim.get_physics_dt())
        scene.clean_towel = initialize_clean_towel_handle_for_sim(scene.sim, scene.clean_towel.root_path)
        scene.step_zero(int(args.settle_initial_steps))

        proxy_scene = _ProxyScene(sim=scene.sim, clean_towel=scene.clean_towel,
                                  primitive_root_path=PROXY_PATH, device=scene.device)
        try:
            v8.apply_contact_params(proxy_scene, v8.default_primitive_params(args))
        except Exception:
            pass

        edge = str(args.edge)
        stage = get_current_stage()
        view, _ = c11.make_rigid_view(PROXY_PATH)

        all_states, all_actions = [], []
        any_valid = False
        for ep in range(int(args.num_episodes)):
            start_points = scene.clean_towel.points_numpy().astype(np.float32)
            centers = c11.default_path_centers(start_points, args, num_steps=int(args.fold_steps), edge=edge)
            widths = [float(size_metrics(start_points)["width_x"])]
            ep_states, ep_actions = [], []
            for step, center in enumerate(centers):
                c11.drive_pose(view, stage, PROXY_PATH, np.asarray(center, np.float32), scene.device)
                scene.sim.step(render=False)
                pts = scene.clean_towel.points_numpy().astype(np.float32)
                progress = progress_for_step(step, int(args.fold_steps))
                phase = phase_id_for_step(step, int(args.fold_steps))
                jaws, _ = jaw_positions_from_scene(scene)
                state, _ = compact_state(pts, progress, phase, jaws)
                action = synthetic_action(progress, phase)  # SYNTHETIC labels (honest)
                ep_states.append(state.astype(np.float32))
                ep_actions.append(action.astype(np.float32))
                widths.append(float(size_metrics(pts)["width_x"]))
            final_points = scene.clean_towel.points_numpy().astype(np.float32)
            warr = np.asarray(widths, np.float32)
            edge_disp = float(edge_center_displacement(start_points, final_points, edge, float(args.edge_band)))
            wred = float(warr[0] - warr.min())
            valid = bool(edge_disp > 0.02 and 0.02 < edge_disp <= 0.5 and wred > 0.005)
            any_valid = any_valid or valid
            manifest["episodes"].append({
                "episode": ep, "num_steps": len(centers),
                "width_before": float(warr[0]), "width_after": float(warr[-1]),
                "best_width": float(warr.min()), "width_reduction_m": wred,
                "edge_displacement_m": edge_disp, "valid_controlled_contact": valid,
            })
            all_states.extend(ep_states)
            all_actions.extend(ep_actions)

        states = np.asarray(all_states, np.float32)
        actions = np.asarray(all_actions, np.float32)
        np.savez_compressed(
            DEMO_NPZ, states=states, actions=actions,
            demo_type="so101_scene_physics_driven_fingertip_proxy_fold",
            joint_action_labels="synthetic", true_robot_joint_action_contact=False,
        )
        manifest.update(
            num_episodes=int(args.num_episodes),
            total_samples=int(states.shape[0]),
            state_dim=int(states.shape[1]) if states.ndim == 2 else 0,
            action_dim=int(actions.shape[1]) if actions.ndim == 2 else 0,
            any_episode_valid_contact=bool(any_valid),
            dataset_path=str(DEMO_NPZ),
            # true robot-contact data requires REAL joint-action labels -> not met here
            robot_contact_data_ready=False,
            status="V12_SO101_DEMO_COLLECTION_DONE",
        )
        write_json(DEMO_MANIFEST, manifest, print_payload=True)
    except Exception as exc:  # noqa: BLE001
        import traceback
        manifest["status"] = "V12_SO101_DEMO_COLLECTION_FAILED"
        manifest["exception_type"] = exc.__class__.__name__
        manifest["exception"] = repr(exc)
        manifest["traceback_tail"] = traceback.format_exc()[-6000:]
        write_json(DEMO_MANIFEST, manifest, print_payload=True)
    finally:
        sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == "__main__":
    main()
