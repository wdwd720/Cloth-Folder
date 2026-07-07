"""V14 Experiment 3 — SO-101 gripper PINCH-lift-drag (real articulation).

Distinct mechanism from the single press/drag patch: attach TWO contact patches,
one to the `jaw` link (the moving finger) and one to the `gripper` link (the fixed
finger). Closing the REAL SO-101 gripper joint (action index 11) brings them
together and traps the towel edge between them — a genuine robot-driven pinch, not
a free proxy. Then lift slightly and drag inward to fold.

Everything is physics-resolved articulation; the patches move only as their links
move. Cloth metrics (edge displacement, width, velocity) are measured directly.

Output: <out_dir>/pinch_lift_drag_<config_name>.json (+ best-actions npz).
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

from clean_towel_v4_common import reset_scene_to_points
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
    CapturePlan,
    attach_patch,
    classify_fold,
    default_action12,
    drive_to,
    estimate_jaw_jacobian,
    patch_center,
    run_capture_episode,
    score_metrics,
    solve_joint_delta,
)

V14_DIR = Path("/workspace/leisaac/tera_checkpoints/clean_towel_v14")
V13_REACH_JSON = Path("/workspace/leisaac/tera_checkpoints/clean_towel_v13/so101_articulation_reach_v13.json")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V14 SO-101 gripper pinch-lift-drag.")
    p.add_argument("--usd_path", type=Path, default=CLEAN_TOWEL_USD_PATH)
    p.add_argument("--reach_json", type=Path, default=V13_REACH_JSON)
    p.add_argument("--out_dir", type=Path, default=V14_DIR)
    p.add_argument("--config_name", type=str, default="pinch_default")
    p.add_argument("--right_base", type=float, nargs=3, default=list(DEFAULT_RIGHT_BASE))
    # small SPHERES on jaw + gripper: reliable center-based touch metric, clean
    # baseline (no big collider to bulldoze the cloth at rest).
    p.add_argument("--patch_shape", type=str, default="sphere", choices=["sphere", "cylinder", "capsule"])
    p.add_argument("--patch_radius", type=float, default=0.028)
    p.add_argument("--patch_length", type=float, default=0.0)
    p.add_argument("--patch_axis", type=str, default="Y", choices=["X", "Y", "Z"])
    p.add_argument("--patch_static_friction", type=float, default=2.0)
    p.add_argument("--patch_dynamic_friction", type=float, default=1.6)
    p.add_argument("--soft_cloth", action="store_true")
    p.add_argument("--settle_steps", type=int, default=20)
    AppLauncher.add_app_launcher_args(p)
    args = p.parse_args()
    force_headless_no_cameras(args)
    return args


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / f"pinch_lift_drag_{args.config_name}.json"
    app = AppLauncher(args).app  # noqa: F841
    started = time.monotonic()

    summary: dict[str, Any] = {
        "status": "V14_PINCH_LIFT_DRAG_STARTED",
        "config_name": args.config_name,
        "command_run": command_run_string(),
        "experiment": "gripper_pinch_lift_drag",
        "soft_cloth_variant": bool(args.soft_cloth),
        "drive_mode": "physics_resolved_articulation (real gripper joint closes two link-attached patches)",
        "contact_element": "two patches on jaw + gripper links; real gripper joint pinches the edge (NOT a free proxy)",
        "right_base": list(args.right_base),
    }
    try:
        reach = json.loads(args.reach_json.read_text())
        best0 = reach.get("global_best")
        if not best0 or not reach.get("reachable_edge_found"):
            raise RuntimeError("V13 reach solution has no reachable edge")
        contact_action = np.asarray(best0["action_vector"], dtype=np.float32).reshape(12)
        edge = str(best0["edge"])
        summary["reach_edge"] = edge

        attach_info: dict[str, Any] = {}

        def _attach():
            attach_info["jaw"] = attach_patch(
                "/World/Right_Robot", name=f"v14_pinch_jaw_{args.config_name}", shape=args.patch_shape,
                radius=float(args.patch_radius), length=float(args.patch_length), axis=args.patch_axis,
                body_match="jaw", local_offset=(0.0, 0.0, 0.0),
                static_friction=float(args.patch_static_friction), dynamic_friction=float(args.patch_dynamic_friction),
                mat_path_str="/World/Materials/v14_pinch_mat")
            attach_info["gripper"] = attach_patch(
                "/World/Right_Robot", name=f"v14_pinch_grip_{args.config_name}", shape=args.patch_shape,
                radius=float(args.patch_radius), length=float(args.patch_length), axis=args.patch_axis,
                body_match="gripper", local_offset=(0.0, 0.0, 0.0),
                static_friction=float(args.patch_static_friction), dynamic_friction=float(args.patch_dynamic_friction),
                mat_path_str="/World/Materials/v14_pinch_mat")

        scene = create_standalone_so101_clean_towel_scene(
            args, clean_usd_path=args.usd_path, towel_translation=STABLE_TOWEL_TRANSLATION,
            right_base=tuple(args.right_base), pre_reset_spawn=_attach)
        summary["patch_attach"] = attach_info
        if not (attach_info.get("jaw", {}).get("attached") and attach_info.get("gripper", {}).get("attached")):
            raise RuntimeError(f"pinch patch attach failed: {attach_info}")
        scene.step_zero(int(args.settle_steps))
        start_points = scene.clean_towel.points_numpy().astype(np.float32)
        summary["start_metrics"] = size_metrics(start_points)
        base = default_action12(scene)
        contact_action[:6] = base[:6]

        def center_fn(sc: Any) -> np.ndarray:
            return patch_center(sc, (0.0, 0.0, 0.0), "jaw")

        jaw0, J = estimate_jaw_jacobian(scene, contact_action, start_points, center_fn)
        summary["jacobian_jaw0_xyz"] = [float(x) for x in jaw0]
        inward_sign = -1.0 if edge == "right" else (1.0 if edge == "left" else 0.0)

        # NO-CONTACT baseline
        reset_scene_to_points(scene, start_points)
        drive_to(scene, base, 120)
        base_pts = scene.clean_towel.points_numpy().astype(np.float32)
        baseline_edge = float(edge_center_displacement(start_points, base_pts, edge, EDGE_BAND))
        summary["baseline_no_contact_edge_disp_m"] = baseline_edge

        press_delta = solve_joint_delta(J, np.array([0.0, 0.0, -0.03], np.float32))
        lift_delta = solve_joint_delta(J, np.array([0.0, 0.0, 0.05], np.float32))

        # gripper index 11; sweep both close directions (semantics unknown a priori)
        gripper_limits = (0.45, 0.85)
        plans: list[CapturePlan] = []
        for g_close in (0.50, 0.62, 0.80):
            for drag_dist in (0.14, 0.22, 0.30):
                drag_delta = solve_joint_delta(J, np.array([inward_sign * drag_dist, 0.0, 0.0], np.float32))
                ca = contact_action.copy(); ca[11] = float(gripper_limits[1] if g_close < 0.65 else gripper_limits[0])
                plans.append(CapturePlan(
                    name=f"gclose{g_close:.2f}_drag{drag_dist:.2f}", contact_action=ca,
                    press_delta=press_delta, gripper_close=float(g_close), lift_delta=lift_delta, drag_delta=drag_delta,
                    steps={"contact": 28, "press": 20, "pinch": 18, "lift": 16, "drag": 90}))

        trials, best = [], None
        for plan in plans:
            m, _ = run_capture_episode(
                scene, start_points, plan, edge, patch_radius=float(args.patch_radius),
                left_default=base, center_fn=center_fn)
            cls = classify_fold(m, baseline_edge=baseline_edge)
            score = score_metrics(m, baseline_edge=baseline_edge)
            row = {"name": plan.name, "score": float(score), **{k: m[k] for k in (
                "edge_displacement_m", "width_reduction_m", "actual_touch_distance_m",
                "particle_velocity_peak", "best_width", "near_particle_span_y_m")}, **cls}
            trials.append(row)
            if best is None or score > best["score"]:
                best = {"plan": plan, "row": row, "score": float(score)}
        trials.sort(key=lambda r: -r["score"])
        summary["num_plans"] = len(plans)
        summary["all_trials"] = trials

        bm, best_actions = run_capture_episode(
            scene, start_points, best["plan"], edge, patch_radius=float(args.patch_radius),
            left_default=base, center_fn=center_fn, record=True)
        cls = classify_fold(bm, baseline_edge=baseline_edge)
        summary["best_trial"] = {"name": best["plan"].name, **bm, **cls}
        summary["arm_driven_fold_solved"] = bool(cls["valid_controlled_contact"])
        summary["status"] = "V14_PINCH_LIFT_DRAG_DONE"
        summary["elapsed_s"] = float(time.monotonic() - started)
        np.savez_compressed(
            args.out_dir / f"pinch_lift_drag_{args.config_name}_actions.npz",
            actions=np.asarray(best_actions, np.float32), edge=edge,
            right_base=np.asarray(args.right_base, np.float32),
            config_name=args.config_name, valid=bool(cls["valid_controlled_contact"]))
        write_json(out_json, summary, print_payload=True)
    except Exception as exc:  # noqa: BLE001
        import traceback
        summary["status"] = "V14_PINCH_LIFT_DRAG_FAILED"
        summary["exception_type"] = exc.__class__.__name__
        summary["exception"] = repr(exc)
        summary["traceback_tail"] = traceback.format_exc()[-6000:]
        write_json(out_json, summary, print_payload=True)
    finally:
        sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == "__main__":
    main()
