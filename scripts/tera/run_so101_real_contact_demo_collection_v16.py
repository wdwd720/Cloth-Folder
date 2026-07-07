"""V16 state-reactive REAL SO-101 robot-contact demo collection.

Re-runs the PROVEN V14 fold MECHANISM (jaw-mounted disk patch + physics-resolved
articulation, gentle press + slow inward drag, built from a local joint->jaw
Jacobian at the V13 reach pose) — but instead of replaying ONE fixed trajectory,
it PARAMETRICALLY samples many diverse real articulation-driven folds, randomizing:

  * towel x/y start offset          (shift the settled cloth before the episode)
  * edge target offset              (y-component of the inward drag endpoint)
  * press depth                     (how far the disk presses the edge band down)
  * drag distance                   (how far inward the edge is dragged)
  * drag speed                      (# physics steps for the drag segment)
  * small contact patch offset       (lateral joint-space shift of the contact pose)

Every step records:
  * state:  compact_state_reactive(...) — NO progress/phase clock (V16 schema)
  * action: the REAL 12-D commanded SO-101 joint position target
  * contact metrics used to CLASSIFY the episode as valid / ballistic / no-fold

Only VALID controlled-contact episodes go into the training dataset; ballistic
("fake") and no-fold episodes are labeled and EXCLUDED from the success demos
(some aggressive/fast configs are sampled on purpose as controlled failures).

`robot_contact_data_ready = true` ONLY if >= MIN_VALID episodes are valid
arm-driven folds. Saved to the Modal volume only (npz + manifest); never git.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

import isaaclab_rl_stubs_v12  # noqa: F401,E402

from isaaclab.app import AppLauncher

from clean_towel_v4_common import apply_action_step, reset_scene_to_points
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
    default_action12,
    drive_to,
    estimate_jaw_jacobian,
    patch_center,
    solve_joint_delta,
)
from so101_state_reactive_common_v16 import (
    STATE_REACTIVE_SCHEMA_VERSION,
    compact_state_reactive,
    joint_positions_from_scene,
)
from clean_towel_v4_common import jaw_positions_from_scene

V16_DIR = Path("/workspace/leisaac/tera_checkpoints/clean_towel_v16")
DEMO_DIR = V16_DIR / "demos"
DEMO_NPZ = DEMO_DIR / "so101_real_contact_demos_v16.npz"
DEMO_MANIFEST = DEMO_DIR / "so101_real_contact_demos_manifest_v16.json"

MIN_VALID = 20  # goal: minimum 20 valid demos


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V16 state-reactive SO-101 contact demo collection.")
    p.add_argument("--usd_path", type=Path, default=CLEAN_TOWEL_USD_PATH)
    p.add_argument("--reach_json", type=Path,
                   default=Path("/workspace/leisaac/tera_checkpoints/clean_towel_v13/so101_articulation_reach_v13.json"))
    p.add_argument("--config_name", type=str, default="disk_z_r14")
    p.add_argument("--right_base", type=float, nargs=3, default=list(DEFAULT_RIGHT_BASE))
    # winning V14 geometry defaults (disk_z_r14)
    p.add_argument("--patch_shape", type=str, default="cylinder", choices=["sphere", "cylinder", "capsule"])
    p.add_argument("--patch_radius", type=float, default=0.14)
    p.add_argument("--patch_length", type=float, default=0.03)
    p.add_argument("--patch_axis", type=str, default="Z", choices=["X", "Y", "Z"])
    p.add_argument("--patch_offset", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    p.add_argument("--patch_contact_offset", type=float, default=0.04)
    p.add_argument("--patch_static_friction", type=float, default=2.0)
    p.add_argument("--patch_dynamic_friction", type=float, default=1.6)
    p.add_argument("--num_demos", type=int, default=30)
    p.add_argument("--num_failures", type=int, default=6, help="deliberate controlled-failure (fast/aggressive) samples")
    p.add_argument("--settle_steps", type=int, default=20)
    p.add_argument("--time_budget_s", type=float, default=2700.0)
    p.add_argument("--seed", type=int, default=1616)
    AppLauncher.add_app_launcher_args(p)
    args = p.parse_args()
    force_headless_no_cameras(args)
    return args


def _sample_plans(rng: np.random.Generator, n_success: int, n_failure: int) -> list[dict[str, Any]]:
    """Sample randomized (press/drag/speed/offset) demo configs. success-region
    configs stay in the V14 gentle-press + slow-drag sweet spot; failure configs
    are intentionally fast/aggressive (labeled, kept out of the success dataset)."""
    plans: list[dict[str, Any]] = []
    for _ in range(int(n_success)):
        plans.append({
            "kind": "success",
            "press_depth": float(rng.uniform(0.022, 0.045)),
            "drag_dist": float(rng.uniform(0.14, 0.22)),
            "drag_steps": int(rng.integers(140, 171)),
            "edge_y_off": float(rng.uniform(-0.025, 0.025)),
            "lateral": float(rng.uniform(-0.015, 0.015)),
            "towel_dx": float(rng.uniform(-0.020, 0.020)),
            "towel_dy": float(rng.uniform(-0.020, 0.020)),
        })
    for _ in range(int(n_failure)):
        plans.append({
            "kind": "failure_candidate",
            "press_depth": float(rng.uniform(0.050, 0.075)),
            "drag_dist": float(rng.uniform(0.24, 0.32)),
            "drag_steps": int(rng.integers(70, 105)),
            "edge_y_off": float(rng.uniform(-0.03, 0.03)),
            "lateral": float(rng.uniform(-0.02, 0.02)),
            "towel_dx": float(rng.uniform(-0.020, 0.020)),
            "towel_dy": float(rng.uniform(-0.020, 0.020)),
        })
    return plans


def _run_episode(scene: Any, start_points: np.ndarray, contact_action: np.ndarray, base: np.ndarray,
                 J: np.ndarray, inward_sign: float, cfg: dict[str, Any], edge: str,
                 patch_offset, patch_radius: float, center_fn) -> tuple[list, list, dict]:
    """Execute default->contact->press->drag via real articulation targets, recording
    the V16 state-reactive obs + real action each physics step. Returns
    (states, actions, metrics)."""
    reset_scene_to_points(scene, start_points)

    # build the segment endpoints from the Jacobian (same construction as V14 search)
    contact = contact_action.copy()
    contact[:6] = base[:6]
    if abs(cfg["lateral"]) > 1e-6:  # small lateral contact-patch offset (joint space)
        contact = (contact + solve_joint_delta(J, np.array([0.0, cfg["lateral"], 0.0], np.float32))).astype(np.float32)
        contact[:6] = base[:6]
    press_delta = solve_joint_delta(J, np.array([0.0, 0.0, -float(cfg["press_depth"])], np.float32))
    press = (contact + press_delta).astype(np.float32)
    dp = np.array([inward_sign * float(cfg["drag_dist"]), float(cfg["edge_y_off"]), -0.005], np.float32)
    drag = (press + solve_joint_delta(J, dp)).astype(np.float32)

    segs = [
        (base.astype(np.float32), contact, 36),
        (contact, press, 40),
        (press, drag, int(cfg["drag_steps"])),
    ]

    dt = float(scene.sim.get_physics_dt())
    prev_action = base.astype(np.float32).copy()
    jaws0, _ = jaw_positions_from_scene(scene)
    prev_jaw = np.asarray(jaws0, np.float32).reshape(6)[3:6].copy()

    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    widths = [float(size_metrics(start_points)["width_x"])]
    prev_pts = start_points.copy()
    vpeak, nearest_min = 0.0, float("inf")

    for a0, a1, nsteps in segs:
        for k in range(int(nsteps)):
            a = (a0 + (a1 - a0) * (float(k + 1) / float(nsteps))).astype(np.float32)
            # --- observe BEFORE applying (matches closed-loop eval semantics) ---
            cur_pts = scene.clean_towel.points_numpy().astype(np.float32)
            joints, _ = joint_positions_from_scene(scene)
            jaws, _ = jaw_positions_from_scene(scene)
            right_jaw = np.asarray(jaws, np.float32).reshape(6)[3:6]
            jaw_vel = ((right_jaw - prev_jaw) / max(dt, 1e-6)).astype(np.float32)
            state, _ = compact_state_reactive(cur_pts, joints, jaws, jaw_vel, prev_action, edge=edge)
            states.append(state.astype(np.float32))
            actions.append(a.copy())
            # --- step physics ---
            apply_action_step(scene, a)
            pts = scene.clean_towel.points_numpy().astype(np.float32)
            c = center_fn(scene)
            fd = np.linalg.norm(pts - prev_pts, axis=1) / max(dt, 1e-6)
            near = np.linalg.norm(pts[:, :2] - c[:2].reshape(1, 2), axis=1) < (float(patch_radius) + 0.06)
            if np.any(near):
                vpeak = max(vpeak, float(np.max(fd[near])))
            nearest_min = min(nearest_min, float(np.min(np.linalg.norm(pts - c.reshape(1, 3), axis=1))) - float(patch_radius))
            widths.append(float(size_metrics(pts)["width_x"]))
            prev_pts = pts
            prev_action = a
            prev_jaw = right_jaw.copy()

    final_pts = scene.clean_towel.points_numpy().astype(np.float32)
    warr = np.asarray(widths, np.float32)
    metrics = {
        "actual_touch_distance_m": float(max(0.0, nearest_min)),
        "edge_displacement_m": float(edge_center_displacement(start_points, final_pts, edge, EDGE_BAND)),
        "max_particle_displacement_m": float(np.max(np.linalg.norm(final_pts - start_points, axis=1))),
        "particle_velocity_peak": float(vpeak),
        "width_before": float(warr[0]),
        "width_after": float(warr[-1]),
        "best_width": float(warr.min()),
        "width_reduction_m": float(warr[0] - warr.min()),
        "num_steps": int(sum(n for _, _, n in segs)),
    }
    return states, actions, metrics


def main() -> None:
    args = parse_args()
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    app = AppLauncher(args).app  # noqa: F841
    patch_offset = tuple(float(x) for x in args.patch_offset)
    started = time.monotonic()

    manifest: dict[str, Any] = {
        "status": "V16_SO101_STATE_REACTIVE_DEMO_STARTED",
        "command_run": command_run_string(),
        "demo_type": "so101_real_articulation_driven_fold_v16_state_reactive",
        "action_is_real_joint_targets": True,
        "joint_action_labels": "real_commanded_so101_joint_position_targets",
        "observation_schema_version": STATE_REACTIVE_SCHEMA_VERSION,
        "no_phase_clock": True,
        "config_name": args.config_name,
        "robot_contact_data_ready": False,
        "episodes": [],
    }
    try:
        if not args.reach_json.exists():
            raise FileNotFoundError(f"missing V13 reach solution: {args.reach_json}")
        reach = json.loads(args.reach_json.read_text())
        best = reach.get("global_best")
        if not best or not reach.get("reachable_edge_found"):
            raise RuntimeError("V13 reach solution has no reachable edge")
        contact_action = np.asarray(best["action_vector"], dtype=np.float32).reshape(12)
        edge = str(best["edge"])
        manifest["reach_edge"] = edge
        inward_sign = -1.0 if edge == "right" else (1.0 if edge == "left" else 0.0)

        attach_info: dict[str, Any] = {}

        def _attach():
            attach_info.update(attach_patch(
                "/World/Right_Robot", name=f"v16_demo_{args.config_name}", shape=args.patch_shape,
                radius=float(args.patch_radius), length=float(args.patch_length), axis=args.patch_axis,
                body_match="jaw", local_offset=patch_offset, contact_offset=float(args.patch_contact_offset),
                static_friction=float(args.patch_static_friction), dynamic_friction=float(args.patch_dynamic_friction),
                mat_path_str="/World/Materials/v16_demo_mat"))

        scene = create_standalone_so101_clean_towel_scene(
            args, clean_usd_path=args.usd_path, towel_translation=STABLE_TOWEL_TRANSLATION,
            right_base=tuple(args.right_base), pre_reset_spawn=_attach)
        manifest["patch_attach"] = attach_info
        if not attach_info.get("attached"):
            raise RuntimeError(f"patch attach failed: {attach_info}")
        scene.step_zero(int(args.settle_steps))
        start_points = scene.clean_towel.points_numpy().astype(np.float32)
        base = default_action12(scene)
        contact_action[:6] = base[:6]
        # capture the exact V16 feature-name list (so the CPU validate/summarize
        # stages can assert "no clock" without importing torch)
        _, feature_names = compact_state_reactive(
            start_points, np.zeros(12, np.float32), np.zeros(6, np.float32),
            np.zeros(3, np.float32), base, edge=edge)
        manifest["state_feature_names"] = list(feature_names)

        def center_fn(sc: Any) -> np.ndarray:
            return patch_center(sc, patch_offset, "jaw")

        # --- no-contact baseline (arm at default; disk must be inert) ------------
        reset_scene_to_points(scene, start_points)
        drive_to(scene, base, 120)
        base_pts = scene.clean_towel.points_numpy().astype(np.float32)
        baseline_edge = float(edge_center_displacement(start_points, base_pts, edge, EDGE_BAND))
        manifest["baseline_no_contact_edge_disp_m"] = baseline_edge

        # --- local joint->jaw Jacobian at the reach pose (V14 construction) ------
        _, J = estimate_jaw_jacobian(scene, contact_action, start_points, center_fn)
        manifest["jacobian_cols_djaw_dq"] = J.tolist()

        rng = np.random.default_rng(int(args.seed))
        n_success = max(0, int(args.num_demos) - int(args.num_failures))
        plans = _sample_plans(rng, n_success, int(args.num_failures))

        all_states, all_actions = [], []
        num_valid, num_ballistic, num_nofold = 0, 0, 0
        for i, cfg in enumerate(plans):
            if time.monotonic() - started > float(args.time_budget_s):
                manifest["time_budget_reached_at_demo"] = i
                break
            off = np.array([cfg["towel_dx"], cfg["towel_dy"], 0.0], np.float32)
            ep_start = (start_points + off.reshape(1, 3)).astype(np.float32)
            states, actions, m = _run_episode(
                scene, ep_start, contact_action, base, J, inward_sign, cfg, edge,
                patch_offset, float(args.patch_radius), center_fn)
            cls = classify_fold(m, baseline_edge=baseline_edge)
            valid = bool(cls["valid_controlled_contact"])
            ballistic = bool(cls["spurious_ballistic"])
            if valid:
                num_valid += 1
                all_states.extend(states)
                all_actions.extend(actions)
            elif ballistic:
                num_ballistic += 1
            else:
                num_nofold += 1
            manifest["episodes"].append({
                "episode": i, "kind": cfg["kind"], **{k: cfg[k] for k in (
                    "press_depth", "drag_dist", "drag_steps", "edge_y_off", "lateral", "towel_dx", "towel_dy")},
                "num_steps": m["num_steps"], "width_before": m["width_before"], "width_after": m["width_after"],
                "best_width": m["best_width"], "width_reduction_m": m["width_reduction_m"],
                "edge_displacement_m": m["edge_displacement_m"], "actual_touch_distance_m": m["actual_touch_distance_m"],
                "particle_velocity_peak": m["particle_velocity_peak"],
                "max_particle_displacement_m": m["max_particle_displacement_m"],
                "valid_controlled_contact": valid, "spurious_ballistic": ballistic,
                "included_in_success_dataset": valid,
            })

        states_arr = np.asarray(all_states, np.float32) if all_states else np.zeros((0, 72), np.float32)
        actions_arr = np.asarray(all_actions, np.float32) if all_actions else np.zeros((0, 12), np.float32)
        ready = bool(num_valid >= MIN_VALID)
        np.savez_compressed(
            DEMO_NPZ, states=states_arr, actions=actions_arr,
            demo_type="so101_real_articulation_driven_fold_v16_state_reactive",
            observation_schema_version=STATE_REACTIVE_SCHEMA_VERSION,
            no_phase_clock=True, action_is_real_joint_targets=True,
            robot_contact_data_ready=ready, edge=edge,
            feature_names=np.asarray(feature_names, dtype=object))
        manifest.update(
            num_demos_attempted=int(len(manifest["episodes"])),
            num_valid_success=int(num_valid),
            num_ballistic=int(num_ballistic),
            num_nofold=int(num_nofold),
            num_episodes=int(num_valid),  # episodes that contribute to the training set
            total_samples=int(states_arr.shape[0]),
            state_dim=int(states_arr.shape[1]) if states_arr.ndim == 2 else 0,
            action_dim=int(actions_arr.shape[1]) if actions_arr.ndim == 2 else 0,
            dataset_path=str(DEMO_NPZ),
            robot_contact_data_ready=ready,
            min_valid_required=int(MIN_VALID),
            elapsed_s=float(time.monotonic() - started),
            status="V16_SO101_STATE_REACTIVE_DEMO_DONE",
        )
        write_json(DEMO_MANIFEST, manifest, print_payload=True)
    except Exception as exc:  # noqa: BLE001
        import traceback
        manifest["status"] = "V16_SO101_STATE_REACTIVE_DEMO_FAILED"
        manifest["exception_type"] = exc.__class__.__name__
        manifest["exception"] = repr(exc)
        manifest["traceback_tail"] = traceback.format_exc()[-6000:]
        write_json(DEMO_MANIFEST, manifest, print_payload=True)
    finally:
        sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == "__main__":
    main()
