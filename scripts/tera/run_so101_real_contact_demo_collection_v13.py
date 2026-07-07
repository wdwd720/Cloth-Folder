"""V13 REAL SO-101 robot-contact demo collection (articulation-driven).

Records demonstrations of the REAL SO-101 arm folding the towel via
physics-resolved joint position targets and a jaw-attached contact patch. Unlike
V12 (proxy fold with SYNTHETIC joint labels), here the `action` labels are the
ACTUAL commanded joint position targets — real robot-joint-action contact demos.

It replays the validated fold action trajectory
(`so101_articulation_fold_actions_v13.npz`, written by the V13 fold) in a fresh
scene for a few episodes, recording per step:
  - state:  compact_state(cloth metrics + REAL jaw positions)
  - action: the REAL 12-D commanded joint position target (NOT synthetic)
  - contact metrics: cumulative edge displacement, width, touch, particle velocity

robot_contact_data_ready = true ONLY if the fold was a VALID controlled contact
AND at least one replayed episode reproduces valid arm-driven contact
(touch < 0.01, edge_disp > 0.02, width reduction, not ballistic).
Saved to the Modal volume only (npz + manifest); never committed to git.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

# RL import stubs BEFORE any IsaacLab/SimulationApp import (import-only shim).
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
from test_so101_articulation_contact_fold_v13 import (
    DEFAULT_RIGHT_BASE,
    STABLE_TOWEL_TRANSLATION,
    attach_jaw_contact_patch,
    right_jaw_xyz,
)

V13_DIR = Path("/workspace/leisaac/tera_checkpoints/clean_towel_v13")
FOLD_JSON = V13_DIR / "so101_articulation_contact_fold_v13.json"
ACTIONS_NPZ = V13_DIR / "so101_articulation_fold_actions_v13.npz"
DEMO_DIR = V13_DIR / "demos"
DEMO_NPZ = DEMO_DIR / "so101_real_contact_demos_v13.npz"
DEMO_MANIFEST = DEMO_DIR / "so101_real_contact_demos_manifest_v13.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V13 real SO-101 articulation contact demo collection.")
    parser.add_argument("--usd_path", type=Path, default=CLEAN_TOWEL_USD_PATH)
    parser.add_argument("--right_base", type=float, nargs=3, default=list(DEFAULT_RIGHT_BASE))
    parser.add_argument("--patch_radius", type=float, default=0.06)
    parser.add_argument("--patch_contact_offset", type=float, default=0.02)
    parser.add_argument("--patch_rest_offset", type=float, default=0.0)
    parser.add_argument("--patch_static_friction", type=float, default=1.8)
    parser.add_argument("--patch_dynamic_friction", type=float, default=1.4)
    parser.add_argument("--settle_steps", type=int, default=20)
    parser.add_argument("--num_episodes", type=int, default=3)
    parser.add_argument("--edge_band", type=float, default=0.035)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    force_headless_no_cameras(args)
    return args


def main() -> None:
    args = parse_args()
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    app = AppLauncher(args).app  # noqa: F841

    manifest: dict[str, Any] = {
        "status": "V13_SO101_REAL_CONTACT_DEMO_STARTED",
        "command_run": command_run_string(),
        "demo_type": "so101_real_articulation_driven_fold",
        "action_is_real_joint_targets": True,
        "joint_action_labels": "real_commanded_so101_joint_position_targets",
        "robot_contact_data_ready": False,
        "episodes": [],
    }
    try:
        fold = json.loads(FOLD_JSON.read_text()) if FOLD_JSON.exists() else {}
        fold_solved = bool(fold.get("articulation_contact_fold_solved"))
        manifest["fold_solved"] = fold_solved
        if not fold_solved:
            manifest["status"] = "V13_SO101_REAL_CONTACT_DEMO_SKIPPED"
            manifest["skip_reason"] = "V13 articulation fold not solved; no valid arm-driven contact to demonstrate."
            write_json(DEMO_MANIFEST, manifest, print_payload=True)
            sys.stdout.flush(); sys.stderr.flush(); os._exit(0)
        if not ACTIONS_NPZ.exists():
            raise FileNotFoundError(f"missing validated action trajectory: {ACTIONS_NPZ}")
        data = np.load(ACTIONS_NPZ, allow_pickle=True)
        traj = np.asarray(data["actions"], np.float32)  # (T, 12) real joint targets
        edge = str(data["edge"]) if "edge" in data else "right"
        manifest["trajectory_steps"] = int(traj.shape[0])
        manifest["reach_edge"] = edge

        attach_info: dict[str, Any] = {}

        def _attach():
            attach_info.update(attach_jaw_contact_patch(
                "/World/Right_Robot", radius=float(args.patch_radius),
                contact_offset=float(args.patch_contact_offset), rest_offset=float(args.patch_rest_offset),
                static_friction=float(args.patch_static_friction), dynamic_friction=float(args.patch_dynamic_friction),
            ))

        scene = create_standalone_so101_clean_towel_scene(
            args, clean_usd_path=args.usd_path, towel_translation=STABLE_TOWEL_TRANSLATION,
            right_base=tuple(args.right_base), pre_reset_spawn=_attach,
        )
        manifest["patch_attach"] = attach_info
        scene.step_zero(int(args.settle_steps))
        start_points = scene.clean_towel.points_numpy().astype(np.float32)
        fold_steps = int(traj.shape[0])

        all_states, all_actions = [], []
        any_valid = False
        for ep in range(int(args.num_episodes)):
            reset_scene_to_points(scene, start_points)
            prev = start_points.copy()
            dt = float(scene.sim.get_physics_dt())
            widths = [float(size_metrics(start_points)["width_x"])]
            vpeak = 0.0
            nearest_min = float("inf")
            ep_states, ep_actions = [], []
            for step in range(fold_steps):
                action = traj[step].astype(np.float32)  # REAL commanded joint targets
                apply_action_step(scene, action)
                pts = scene.clean_towel.points_numpy().astype(np.float32)
                jaws, _ = jaw_positions_from_scene(scene)
                progress = progress_for_step(step, fold_steps)
                phase = phase_id_for_step(step, fold_steps)
                state, _ = compact_state(pts, progress, phase, jaws)
                ep_states.append(state.astype(np.float32))
                ep_actions.append(action)  # real joint-action label
                jaw = right_jaw_xyz(scene)
                fd = np.linalg.norm(pts - prev, axis=1) / max(dt, 1e-6)
                near = np.linalg.norm(pts[:, :2] - jaw[:2].reshape(1, 2), axis=1) < (float(args.patch_radius) + 0.05)
                if np.any(near):
                    vpeak = max(vpeak, float(np.max(fd[near])))
                nearest_min = min(nearest_min, float(np.min(np.linalg.norm(pts - jaw.reshape(1, 3), axis=1))) - float(args.patch_radius))
                widths.append(float(size_metrics(pts)["width_x"]))
                prev = pts
            final_pts = scene.clean_towel.points_numpy().astype(np.float32)
            warr = np.asarray(widths, np.float32)
            edge_disp = float(edge_center_displacement(start_points, final_pts, edge, float(args.edge_band)))
            wred = float(warr[0] - warr.min())
            touch = float(max(0.0, nearest_min))
            max_disp = float(np.max(np.linalg.norm(final_pts - start_points, axis=1)))
            valid = bool(touch < 0.01 and 0.02 < edge_disp <= 0.5 and wred > 0.005 and not (max_disp > 0.5 or vpeak >= 4.9))
            any_valid = any_valid or valid
            manifest["episodes"].append({
                "episode": ep, "num_steps": fold_steps,
                "width_before": float(warr[0]), "width_after": float(warr[-1]), "best_width": float(warr.min()),
                "width_reduction_m": wred, "edge_displacement_m": edge_disp,
                "actual_touch_distance_m": touch, "particle_velocity_peak": vpeak,
                "valid_controlled_contact": valid,
            })
            all_states.extend(ep_states)
            all_actions.extend(ep_actions)

        states = np.asarray(all_states, np.float32)
        actions = np.asarray(all_actions, np.float32)
        ready = bool(fold_solved and any_valid)
        np.savez_compressed(
            DEMO_NPZ, states=states, actions=actions,
            demo_type="so101_real_articulation_driven_fold",
            action_is_real_joint_targets=True, robot_contact_data_ready=ready,
        )
        manifest.update(
            num_episodes=int(args.num_episodes), total_samples=int(states.shape[0]),
            state_dim=int(states.shape[1]) if states.ndim == 2 else 0,
            action_dim=int(actions.shape[1]) if actions.ndim == 2 else 0,
            any_episode_valid_contact=bool(any_valid),
            dataset_path=str(DEMO_NPZ),
            robot_contact_data_ready=ready,
            status="V13_SO101_REAL_CONTACT_DEMO_DONE",
        )
        write_json(DEMO_MANIFEST, manifest, print_payload=True)
    except Exception as exc:  # noqa: BLE001
        import traceback
        manifest["status"] = "V13_SO101_REAL_CONTACT_DEMO_FAILED"
        manifest["exception_type"] = exc.__class__.__name__
        manifest["exception"] = repr(exc)
        manifest["traceback_tail"] = traceback.format_exc()[-6000:]
        write_json(DEMO_MANIFEST, manifest, print_payload=True)
    finally:
        sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == "__main__":
    main()
