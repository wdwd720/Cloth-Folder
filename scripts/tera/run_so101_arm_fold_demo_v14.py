"""V14 REAL SO-101 robot-contact demo collection (articulation-driven fold).

ONLY meaningful if a V14 config produced a VALID arm-driven fold on the normal
towel. Replays that config's recorded REAL joint-target trajectory in a fresh
scene for several episodes (with small perturbations), recording per step:
  - state:  compact_state(cloth metrics + REAL jaw positions)
  - action: the REAL 12-D commanded joint position target (NOT synthetic)
  - contact metrics: edge displacement, width, touch, particle velocity

robot_contact_data_ready = true ONLY if the source fold was VALID AND at least
one replayed episode reproduces a valid arm-driven contact. Demos are saved to
the Modal volume only (npz + manifest); never committed to git.

Geometry (shape/radius/length/axis/offset, and --pinch for the two-patch gripper
config) is passed explicitly so the exact winning contact geometry is rebuilt.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

import isaaclab_rl_stubs_v12  # noqa: F401,E402

from isaaclab.app import AppLauncher

from clean_towel_v4_common import (
    apply_action_step,
    compact_state,
    jaw_positions_from_scene,
    phase_id_for_step,
    progress_for_step,
    reset_scene_to_points,
)
from clean_towel_v5_contact_common import edge_center_displacement
from so101_clean_towel_scene_utils_v1 import (
    CLEAN_TOWEL_USD_PATH,
    command_run_string,
    create_standalone_so101_clean_towel_scene,
    force_headless_no_cameras,
    size_metrics,
    write_json,
)
from so101_edge_capture_common_v14 import (
    DEFAULT_RIGHT_BASE,
    EDGE_BAND,
    STABLE_TOWEL_TRANSLATION,
    attach_patch,
    classify_fold,
    patch_center,
)

V14_DIR = Path("/workspace/leisaac/tera_checkpoints/clean_towel_v14")
DEMO_DIR = V14_DIR / "demos"
DEMO_NPZ = DEMO_DIR / "so101_real_contact_demos_v14.npz"
DEMO_MANIFEST = DEMO_DIR / "so101_real_contact_demos_manifest_v14.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V14 real SO-101 articulation contact demo collection.")
    p.add_argument("--usd_path", type=Path, default=CLEAN_TOWEL_USD_PATH)
    p.add_argument("--actions_npz", type=Path, required=True, help="best config recorded actions npz")
    p.add_argument("--config_name", type=str, default="best")
    p.add_argument("--right_base", type=float, nargs=3, default=list(DEFAULT_RIGHT_BASE))
    p.add_argument("--patch_shape", type=str, default="capsule", choices=["sphere", "cylinder", "capsule"])
    p.add_argument("--patch_radius", type=float, default=0.03)
    p.add_argument("--patch_length", type=float, default=0.34)
    p.add_argument("--patch_axis", type=str, default="Y", choices=["X", "Y", "Z"])
    p.add_argument("--patch_offset", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    p.add_argument("--patch_contact_offset", type=float, default=0.02)
    p.add_argument("--patch_static_friction", type=float, default=2.0)
    p.add_argument("--patch_dynamic_friction", type=float, default=1.6)
    p.add_argument("--pinch", action="store_true", help="two-patch gripper config (jaw + gripper links)")
    p.add_argument("--num_episodes", type=int, default=12)
    p.add_argument("--settle_steps", type=int, default=20)
    AppLauncher.add_app_launcher_args(p)
    args = p.parse_args()
    force_headless_no_cameras(args)
    return args


def main() -> None:
    args = parse_args()
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    app = AppLauncher(args).app  # noqa: F841
    patch_offset = tuple(float(x) for x in args.patch_offset)

    manifest: dict[str, Any] = {
        "status": "V14_SO101_REAL_CONTACT_DEMO_STARTED",
        "command_run": command_run_string(),
        "demo_type": "so101_real_articulation_driven_fold_v14",
        "action_is_real_joint_targets": True,
        "joint_action_labels": "real_commanded_so101_joint_position_targets",
        "config_name": args.config_name,
        "robot_contact_data_ready": False,
        "episodes": [],
    }
    try:
        if not args.actions_npz.exists():
            raise FileNotFoundError(f"missing recorded actions npz: {args.actions_npz}")
        data = np.load(args.actions_npz, allow_pickle=True)
        traj = np.asarray(data["actions"], np.float32)
        edge = str(data["edge"]) if "edge" in data else "right"
        source_valid = bool(data["valid"]) if "valid" in data else False
        manifest["source_fold_valid"] = source_valid
        manifest["trajectory_steps"] = int(traj.shape[0])
        manifest["reach_edge"] = edge
        if not source_valid:
            manifest["status"] = "V14_SO101_REAL_CONTACT_DEMO_SKIPPED"
            manifest["skip_reason"] = "source V14 fold was not a VALID arm-driven contact; no valid demos to record."
            write_json(DEMO_MANIFEST, manifest, print_payload=True)
            sys.stdout.flush(); sys.stderr.flush(); os._exit(0)

        attach_info: dict[str, Any] = {}

        def _attach():
            attach_info["jaw"] = attach_patch(
                "/World/Right_Robot", name=f"v14_demo_jaw_{args.config_name}", shape=args.patch_shape,
                radius=float(args.patch_radius), length=float(args.patch_length), axis=args.patch_axis,
                body_match="jaw", local_offset=patch_offset, contact_offset=float(args.patch_contact_offset),
                static_friction=float(args.patch_static_friction), dynamic_friction=float(args.patch_dynamic_friction),
                mat_path_str="/World/Materials/v14_demo_mat")
            if args.pinch:
                attach_info["gripper"] = attach_patch(
                    "/World/Right_Robot", name=f"v14_demo_grip_{args.config_name}", shape=args.patch_shape,
                    radius=float(args.patch_radius), length=float(args.patch_length), axis=args.patch_axis,
                    body_match="gripper", local_offset=(0.0, 0.0, 0.0),
                    static_friction=float(args.patch_static_friction), dynamic_friction=float(args.patch_dynamic_friction),
                    mat_path_str="/World/Materials/v14_demo_mat")

        scene = create_standalone_so101_clean_towel_scene(
            args, clean_usd_path=args.usd_path, towel_translation=STABLE_TOWEL_TRANSLATION,
            right_base=tuple(args.right_base), pre_reset_spawn=_attach)
        manifest["patch_attach"] = attach_info
        scene.step_zero(int(args.settle_steps))
        start_points = scene.clean_towel.points_numpy().astype(np.float32)
        fold_steps = int(traj.shape[0])

        def center_fn(sc: Any) -> np.ndarray:
            return patch_center(sc, patch_offset, "jaw")

        rng = np.random.default_rng(1414)
        all_states, all_actions = [], []
        any_valid = False
        for ep in range(int(args.num_episodes)):
            reset_scene_to_points(scene, start_points)
            # small action-trajectory perturbation for demo variety (still real joint targets)
            jitter = rng.normal(0.0, 0.01, size=(12,)).astype(np.float32) if ep > 0 else np.zeros(12, np.float32)
            jitter[:6] = 0.0  # keep left arm exactly at default
            prev = start_points.copy()
            dt = float(scene.sim.get_physics_dt())
            widths = [float(size_metrics(start_points)["width_x"])]
            vpeak, nearest_min = 0.0, float("inf")
            ep_states, ep_actions = [], []
            for step in range(fold_steps):
                action = (traj[step] + jitter).astype(np.float32)
                apply_action_step(scene, action)
                pts = scene.clean_towel.points_numpy().astype(np.float32)
                jaws, _ = jaw_positions_from_scene(scene)
                progress = progress_for_step(step, fold_steps)
                phase = phase_id_for_step(step, fold_steps)
                state, _ = compact_state(pts, progress, phase, jaws)
                ep_states.append(state.astype(np.float32))
                ep_actions.append(action)
                c = center_fn(scene)
                fd = np.linalg.norm(pts - prev, axis=1) / max(dt, 1e-6)
                near = np.linalg.norm(pts[:, :2] - c[:2].reshape(1, 2), axis=1) < (float(args.patch_radius) + 0.06)
                if np.any(near):
                    vpeak = max(vpeak, float(np.max(fd[near])))
                nearest_min = min(nearest_min, float(np.min(np.linalg.norm(pts - c.reshape(1, 3), axis=1))) - float(args.patch_radius))
                widths.append(float(size_metrics(pts)["width_x"]))
                prev = pts
            final_pts = scene.clean_towel.points_numpy().astype(np.float32)
            warr = np.asarray(widths, np.float32)
            m = {
                "actual_touch_distance_m": float(max(0.0, nearest_min)),
                "edge_displacement_m": float(edge_center_displacement(start_points, final_pts, edge, EDGE_BAND)),
                "max_particle_displacement_m": float(np.max(np.linalg.norm(final_pts - start_points, axis=1))),
                "particle_velocity_peak": float(vpeak),
                "width_reduction_m": float(warr[0] - warr.min()),
            }
            cls = classify_fold(m)
            valid = bool(cls["valid_controlled_contact"])
            any_valid = any_valid or valid
            manifest["episodes"].append({
                "episode": ep, "num_steps": fold_steps, "perturbed": bool(ep > 0),
                "width_before": float(warr[0]), "width_after": float(warr[-1]), "best_width": float(warr.min()),
                "width_reduction_m": m["width_reduction_m"], "edge_displacement_m": m["edge_displacement_m"],
                "actual_touch_distance_m": m["actual_touch_distance_m"], "particle_velocity_peak": vpeak,
                "valid_controlled_contact": valid,
            })
            all_states.extend(ep_states)
            all_actions.extend(ep_actions)

        states = np.asarray(all_states, np.float32)
        actions = np.asarray(all_actions, np.float32)
        ready = bool(source_valid and any_valid)
        np.savez_compressed(
            DEMO_NPZ, states=states, actions=actions,
            demo_type="so101_real_articulation_driven_fold_v14",
            action_is_real_joint_targets=True, robot_contact_data_ready=ready, edge=edge)
        num_valid_success = sum(1 for e in manifest["episodes"] if e["valid_controlled_contact"])
        manifest.update(
            num_episodes=int(args.num_episodes), total_samples=int(states.shape[0]),
            state_dim=int(states.shape[1]) if states.ndim == 2 else 0,
            action_dim=int(actions.shape[1]) if actions.ndim == 2 else 0,
            num_valid_success=int(num_valid_success),
            num_valid_failure=int(int(args.num_episodes) - num_valid_success),
            any_episode_valid_contact=bool(any_valid),
            dataset_path=str(DEMO_NPZ),
            robot_contact_data_ready=ready,
            status="V14_SO101_REAL_CONTACT_DEMO_DONE",
        )
        write_json(DEMO_MANIFEST, manifest, print_payload=True)
    except Exception as exc:  # noqa: BLE001
        import traceback
        manifest["status"] = "V14_SO101_REAL_CONTACT_DEMO_FAILED"
        manifest["exception_type"] = exc.__class__.__name__
        manifest["exception"] = repr(exc)
        manifest["traceback_tail"] = traceback.format_exc()[-6000:]
        write_json(DEMO_MANIFEST, manifest, print_payload=True)
    finally:
        sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == "__main__":
    main()
