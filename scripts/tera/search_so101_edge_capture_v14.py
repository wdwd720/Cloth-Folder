"""V14 SO-101 edge-capture search (real articulation-driven fold attempt).

Builds ONE scene with ONE robot-attached contact patch (geometry from args) and
searches MOTION plans (press+drag, outside-hook) with a cloth-aware objective, to
turn the V13 arm-contact MECHANISM into a real fold (edge drag + width reduction).

Why a Jacobian: there is no IK on this Modal image, so to construct principled
press-DOWN / hook-OUTSIDE / drag-INWARD motions we probe a local joint->jaw
Jacobian at the reach pose and solve small joint deltas for the desired Cartesian
moves. Every candidate is then VERIFIED by the DIRECT cloth metrics (edge
displacement, width reduction, particle velocity), never by the Jacobian estimate.

All motion is physics-resolved articulation (set_joint_position_target ->
write_data_to_sim -> sim.step); the patch is rigidly attached to the jaw link
(NOT a free proxy, NOT a teleport, NOT particle writes). Ballistic flings are
excluded.

Per-config output: <out_dir>/edge_capture_<config_name>.json  (Modal copies to
/artifacts/contact_v14/). global_best actions are saved to an npz for the demo.
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

# RL import stubs BEFORE any IsaacLab/SimulationApp import (import-only shim).
import isaaclab_rl_stubs_v12  # noqa: F401,E402

from isaaclab.app import AppLauncher

from clean_towel_v4_common import reset_scene_to_points
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
    RIGHT_JOINTS,
    STABLE_TOWEL_TRANSLATION,
    CapturePlan,
    attach_patch,
    classify_fold,
    default_action12,
    drive_to,
    estimate_jaw_jacobian,
    link_pose,
    patch_center,
    run_capture_episode,
    score_metrics,
    solve_joint_delta,
)

V14_DIR = Path("/workspace/leisaac/tera_checkpoints/clean_towel_v14")
V13_REACH_JSON = Path("/workspace/leisaac/tera_checkpoints/clean_towel_v13/so101_articulation_reach_v13.json")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V14 SO-101 edge-capture cloth-aware search.")
    p.add_argument("--usd_path", type=Path, default=CLEAN_TOWEL_USD_PATH)
    p.add_argument("--reach_json", type=Path, default=V13_REACH_JSON)
    p.add_argument("--out_dir", type=Path, default=V14_DIR)
    p.add_argument("--config_name", type=str, default="sphere_r05_ctrl")
    p.add_argument("--right_base", type=float, nargs=3, default=list(DEFAULT_RIGHT_BASE))
    # patch geometry (build-time; robot-side contact geometry, fully allowed)
    p.add_argument("--patch_shape", type=str, default="sphere", choices=["sphere", "cylinder", "capsule"])
    p.add_argument("--patch_radius", type=float, default=0.05)
    p.add_argument("--patch_length", type=float, default=0.0)
    p.add_argument("--patch_axis", type=str, default="Y", choices=["X", "Y", "Z"])
    p.add_argument("--patch_offset", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    p.add_argument("--patch_contact_offset", type=float, default=0.02)
    p.add_argument("--patch_rest_offset", type=float, default=0.0)
    p.add_argument("--patch_static_friction", type=float, default=2.0)
    p.add_argument("--patch_dynamic_friction", type=float, default=1.6)
    p.add_argument("--soft_cloth", action="store_true", help="label only; the Modal app passes a soft --usd_path")
    p.add_argument("--settle_steps", type=int, default=20)
    AppLauncher.add_app_launcher_args(p)
    args = p.parse_args()
    force_headless_no_cameras(args)
    return args


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / f"edge_capture_{args.config_name}.json"
    app = AppLauncher(args).app  # noqa: F841
    started = time.monotonic()

    patch_offset = tuple(float(x) for x in args.patch_offset)
    summary: dict[str, Any] = {
        "status": "V14_EDGE_CAPTURE_STARTED",
        "config_name": args.config_name,
        "command_run": command_run_string(),
        "soft_cloth_variant": bool(args.soft_cloth),
        "usd_path": str(args.usd_path),
        "right_base": list(args.right_base),
        "drive_mode": "physics_resolved_articulation (set_joint_position_target + write_data_to_sim + sim.step)",
        "contact_element": f"jaw-attached {args.patch_shape} patch r={args.patch_radius} len={args.patch_length} axis={args.patch_axis} (moves with articulation; NOT a free proxy)",
        "patch": {
            "shape": args.patch_shape, "radius": args.patch_radius, "length": args.patch_length,
            "axis": args.patch_axis, "offset": list(patch_offset),
            "static_friction": args.patch_static_friction, "dynamic_friction": args.patch_dynamic_friction,
        },
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
        summary["reach_edge"] = edge
        summary["reach_best_distance_m"] = float(best["distance_m"])

        attach_info: dict[str, Any] = {}

        def _attach():
            attach_info.update(attach_patch(
                "/World/Right_Robot", name=f"v14_patch_{args.config_name}", shape=args.patch_shape,
                radius=float(args.patch_radius), length=float(args.patch_length), axis=args.patch_axis,
                body_match="jaw", local_offset=patch_offset,
                contact_offset=float(args.patch_contact_offset), rest_offset=float(args.patch_rest_offset),
                static_friction=float(args.patch_static_friction), dynamic_friction=float(args.patch_dynamic_friction),
            ))

        scene = create_standalone_so101_clean_towel_scene(
            args, clean_usd_path=args.usd_path, towel_translation=STABLE_TOWEL_TRANSLATION,
            right_base=tuple(args.right_base), pre_reset_spawn=_attach,
        )
        summary["patch_attach"] = attach_info
        if not attach_info.get("attached"):
            raise RuntimeError(f"patch attach failed: {attach_info}")
        scene.step_zero(int(args.settle_steps))
        start_points = scene.clean_towel.points_numpy().astype(np.float32)
        summary["start_metrics"] = size_metrics(start_points)
        base = default_action12(scene)
        contact_action[:6] = base[:6]  # left arm stays at default; only right arm acts

        def center_fn(sc: Any) -> np.ndarray:
            return patch_center(sc, patch_offset, "jaw")

        # --- jaw-orientation diagnostic (tells us how a bar patch is oriented) ---
        reset_scene_to_points(scene, start_points)
        drive_to(scene, contact_action, 30)
        jaw_pos, jaw_quat = link_pose(scene.right_arm, "jaw")
        from so101_edge_capture_common_v14 import _quat_rotate
        summary["contact_pose_jaw_xyz"] = [float(x) for x in jaw_pos]
        summary["contact_pose_patch_xyz"] = [float(x) for x in center_fn(scene)]
        summary["contact_pose_jaw_quat_wxyz"] = [float(x) for x in jaw_quat]
        summary["jaw_local_axes_in_world"] = {
            ax: [float(v) for v in _quat_rotate(jaw_quat, e)]
            for ax, e in zip("XYZ", np.eye(3, dtype=np.float32))
        }

        # --- NO-CONTACT baseline (arm held at default; cloth must be inert) ------
        reset_scene_to_points(scene, start_points)
        drive_to(scene, base, 120)
        base_pts = scene.clean_towel.points_numpy().astype(np.float32)
        from clean_towel_v5_contact_common import edge_center_displacement
        baseline_edge = float(edge_center_displacement(start_points, base_pts, edge, EDGE_BAND))
        summary["baseline_no_contact"] = {
            "edge_displacement_m": baseline_edge,
            "width_before": float(size_metrics(start_points)["width_x"]),
            "width_after": float(size_metrics(base_pts)["width_x"]),
        }

        # --- local joint->jaw Jacobian at the contact pose ----------------------
        jaw0, J = estimate_jaw_jacobian(scene, contact_action, start_points, center_fn)
        summary["jacobian_jaw0_xyz"] = [float(x) for x in jaw0]
        summary["jacobian_cols_djaw_dq"] = J.tolist()

        # inward = toward towel centre in x (edge 'right' => -x); down = -z
        inward_sign = -1.0 if edge == "right" else (1.0 if edge == "left" else 0.0)

        # --- build candidate motion plans --------------------------------------
        # SLOW drags: a fast drag flings the stiff PBD sheet (ballistic). Gentle
        # presses: an aggressive press of a wide flat patch over-constrains the
        # PBD solve and explodes it. So sweep gentle press depths + slow drags.
        plans: list[CapturePlan] = []
        # A. press + drag inward (wide patch presses the edge band, then buckle-folds it)
        for press_depth in (0.015, 0.03, 0.05):
            press_delta = solve_joint_delta(J, np.array([0.0, 0.0, -press_depth], np.float32))
            for drag_dist in (0.08, 0.16, 0.26):
                dp = np.array([inward_sign * drag_dist, 0.0, -0.005], np.float32)
                drag_delta = solve_joint_delta(J, dp)
                plans.append(CapturePlan(
                    name=f"pressdrag_p{press_depth:.3f}_d{drag_dist:.2f}",
                    contact_action=contact_action.copy(), press_delta=press_delta, drag_delta=drag_delta,
                    steps={"contact": 36, "press": 40, "drag": 160}))
        # B. outside-hook: approach beyond the edge & low, then sweep inward through it
        for margin in (0.05, 0.09):
            approach_delta = solve_joint_delta(J, np.array([-inward_sign * margin, 0.0, -0.025], np.float32))
            approach = (contact_action + approach_delta).astype(np.float32); approach[:6] = base[:6]
            press_delta = solve_joint_delta(J, np.array([0.0, 0.0, -0.025], np.float32))
            for drag_dist in (0.14, 0.24):
                dp = np.array([inward_sign * drag_dist, 0.0, 0.0], np.float32)
                drag_delta = solve_joint_delta(J, dp)
                plans.append(CapturePlan(
                    name=f"outsidehook_m{margin:.2f}_d{drag_dist:.2f}",
                    contact_action=contact_action.copy(), approach_action=approach,
                    press_delta=press_delta, drag_delta=drag_delta,
                    steps={"approach": 44, "contact": 30, "press": 30, "drag": 150}))

        # --- run the sweep (cloth-aware) ---------------------------------------
        trials = []
        best = None
        for plan in plans:
            m, _ = run_capture_episode(
                scene, start_points, plan, edge, patch_offset=patch_offset,
                patch_radius=float(args.patch_radius), left_default=base, center_fn=center_fn)
            cls = classify_fold(m, baseline_edge=baseline_edge)
            score = score_metrics(m, baseline_edge=baseline_edge)
            row = {"name": plan.name, "score": float(score), **{k: m[k] for k in (
                "edge_displacement_m", "width_reduction_m", "actual_touch_distance_m",
                "particle_velocity_peak", "width_before", "width_after", "best_width",
                "near_particle_span_y_m", "max_particle_displacement_m")}, **cls}
            trials.append(row)
            if best is None or score > best["score"]:
                best = {"plan": plan, "row": row, "score": float(score)}
        trials.sort(key=lambda r: -r["score"])
        summary["num_plans"] = len(plans)
        summary["all_trials"] = trials[:24]

        # --- re-run the winner WITH recording -> demo trajectory ---------------
        bm, best_actions = run_capture_episode(
            scene, start_points, best["plan"], edge, patch_offset=patch_offset,
            patch_radius=float(args.patch_radius), left_default=base, center_fn=center_fn, record=True)
        cls = classify_fold(bm, baseline_edge=baseline_edge)
        summary["baseline_edge_disp_m"] = float(baseline_edge)
        summary["best_trial"] = {"name": best["plan"].name, **bm, **cls}
        summary["arm_driven_fold_solved"] = bool(cls["valid_controlled_contact"])
        summary["velocity_above_baseline"] = bool(float(bm["particle_velocity_peak"]) > 0.05 >= abs(baseline_edge))
        summary["status"] = "V14_EDGE_CAPTURE_DONE"
        summary["elapsed_s"] = float(time.monotonic() - started)
        np.savez_compressed(
            args.out_dir / f"edge_capture_{args.config_name}_actions.npz",
            actions=np.asarray(best_actions, np.float32), edge=edge,
            right_base=np.asarray(args.right_base, np.float32),
            config_name=args.config_name, valid=bool(cls["valid_controlled_contact"]))
        write_json(out_json, summary, print_payload=True)
    except Exception as exc:  # noqa: BLE001
        import traceback
        summary["status"] = "V14_EDGE_CAPTURE_FAILED"
        summary["exception_type"] = exc.__class__.__name__
        summary["exception"] = repr(exc)
        summary["traceback_tail"] = traceback.format_exc()[-6000:]
        write_json(out_json, summary, print_payload=True)
    finally:
        sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == "__main__":
    main()
