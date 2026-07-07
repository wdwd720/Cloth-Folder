"""V14 Experiment 2 (focused) — SO-101 outside-hook drag, fine parameter sweep.

The broad search (search_so101_edge_capture_v14.py) covers outside-hook as one
mode across several patch geometries. This focused runner sweeps the outside-hook
parameters at finer resolution for a SINGLE chosen geometry — approach margin
(how far beyond the edge to start), approach depth (how low), and inward drag
distance — to squeeze a real fold out of the most promising geometry.

Motion (all physics-resolved articulation, jaw-attached patch): default -> reach
a pre-edge pose just OUTSIDE the edge and low -> sweep inward through the edge lip
-> continue inward (drag). Cloth metrics measured directly; ballistic excluded.

Output: <out_dir>/outside_hook_<config_name>.json (+ best-actions npz).
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
    p = argparse.ArgumentParser(description="V14 SO-101 outside-hook drag (focused sweep).")
    p.add_argument("--usd_path", type=Path, default=CLEAN_TOWEL_USD_PATH)
    p.add_argument("--reach_json", type=Path, default=V13_REACH_JSON)
    p.add_argument("--out_dir", type=Path, default=V14_DIR)
    p.add_argument("--config_name", type=str, default="hook_capsule")
    p.add_argument("--right_base", type=float, nargs=3, default=list(DEFAULT_RIGHT_BASE))
    p.add_argument("--patch_shape", type=str, default="capsule", choices=["sphere", "cylinder", "capsule"])
    p.add_argument("--patch_radius", type=float, default=0.03)
    p.add_argument("--patch_length", type=float, default=0.34)
    p.add_argument("--patch_axis", type=str, default="Y", choices=["X", "Y", "Z"])
    p.add_argument("--patch_offset", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    p.add_argument("--patch_static_friction", type=float, default=2.2)
    p.add_argument("--patch_dynamic_friction", type=float, default=1.8)
    p.add_argument("--soft_cloth", action="store_true")
    p.add_argument("--settle_steps", type=int, default=20)
    AppLauncher.add_app_launcher_args(p)
    args = p.parse_args()
    force_headless_no_cameras(args)
    return args


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / f"outside_hook_{args.config_name}.json"
    app = AppLauncher(args).app  # noqa: F841
    started = time.monotonic()
    patch_offset = tuple(float(x) for x in args.patch_offset)

    summary: dict[str, Any] = {
        "status": "V14_OUTSIDE_HOOK_STARTED",
        "config_name": args.config_name,
        "command_run": command_run_string(),
        "experiment": "outside_hook_drag_focused",
        "soft_cloth_variant": bool(args.soft_cloth),
        "drive_mode": "physics_resolved_articulation (set_joint_position_target + write_data_to_sim + sim.step)",
        "contact_element": f"jaw-attached {args.patch_shape} patch (NOT a free proxy)",
        "right_base": list(args.right_base),
        "patch": {"shape": args.patch_shape, "radius": args.patch_radius, "length": args.patch_length,
                  "axis": args.patch_axis, "offset": list(patch_offset)},
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
            attach_info.update(attach_patch(
                "/World/Right_Robot", name=f"v14_hook_{args.config_name}", shape=args.patch_shape,
                radius=float(args.patch_radius), length=float(args.patch_length), axis=args.patch_axis,
                body_match="jaw", local_offset=patch_offset,
                static_friction=float(args.patch_static_friction), dynamic_friction=float(args.patch_dynamic_friction)))

        scene = create_standalone_so101_clean_towel_scene(
            args, clean_usd_path=args.usd_path, towel_translation=STABLE_TOWEL_TRANSLATION,
            right_base=tuple(args.right_base), pre_reset_spawn=_attach)
        summary["patch_attach"] = attach_info
        if not attach_info.get("attached"):
            raise RuntimeError(f"patch attach failed: {attach_info}")
        scene.step_zero(int(args.settle_steps))
        start_points = scene.clean_towel.points_numpy().astype(np.float32)
        summary["start_metrics"] = size_metrics(start_points)
        base = default_action12(scene)
        contact_action[:6] = base[:6]

        def center_fn(sc: Any) -> np.ndarray:
            return patch_center(sc, patch_offset, "jaw")

        jaw0, J = estimate_jaw_jacobian(scene, contact_action, start_points, center_fn)
        summary["jacobian_jaw0_xyz"] = [float(x) for x in jaw0]
        inward_sign = -1.0 if edge == "right" else (1.0 if edge == "left" else 0.0)

        reset_scene_to_points(scene, start_points)
        drive_to(scene, base, 120)
        base_pts = scene.clean_towel.points_numpy().astype(np.float32)
        baseline_edge = float(edge_center_displacement(start_points, base_pts, edge, EDGE_BAND))
        summary["baseline_no_contact_edge_disp_m"] = baseline_edge

        # fine outside-hook sweep
        plans: list[CapturePlan] = []
        for margin in (0.03, 0.06, 0.10):
            for depth in (0.02, 0.04):
                approach_delta = solve_joint_delta(J, np.array([-inward_sign * margin, 0.0, -depth], np.float32))
                approach = (contact_action + approach_delta).astype(np.float32); approach[:6] = base[:6]
                press_delta = solve_joint_delta(J, np.array([0.0, 0.0, -depth], np.float32))
                for drag_dist in (0.16, 0.24, 0.32):
                    drag_delta = solve_joint_delta(J, np.array([inward_sign * drag_dist, 0.0, 0.0], np.float32))
                    plans.append(CapturePlan(
                        name=f"m{margin:.2f}_z{depth:.2f}_d{drag_dist:.2f}",
                        contact_action=contact_action.copy(), approach_action=approach,
                        press_delta=press_delta, drag_delta=drag_delta,
                        steps={"approach": 34, "contact": 24, "press": 18, "drag": 100}))

        trials, best = [], None
        for plan in plans:
            m, _ = run_capture_episode(
                scene, start_points, plan, edge, patch_offset=patch_offset,
                patch_radius=float(args.patch_radius), left_default=base, center_fn=center_fn)
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
        summary["all_trials"] = trials[:24]

        bm, best_actions = run_capture_episode(
            scene, start_points, best["plan"], edge, patch_offset=patch_offset,
            patch_radius=float(args.patch_radius), left_default=base, center_fn=center_fn, record=True)
        cls = classify_fold(bm, baseline_edge=baseline_edge)
        summary["best_trial"] = {"name": best["plan"].name, **bm, **cls}
        summary["arm_driven_fold_solved"] = bool(cls["valid_controlled_contact"])
        summary["status"] = "V14_OUTSIDE_HOOK_DONE"
        summary["elapsed_s"] = float(time.monotonic() - started)
        np.savez_compressed(
            args.out_dir / f"outside_hook_{args.config_name}_actions.npz",
            actions=np.asarray(best_actions, np.float32), edge=edge,
            right_base=np.asarray(args.right_base, np.float32),
            config_name=args.config_name, valid=bool(cls["valid_controlled_contact"]))
        write_json(out_json, summary, print_payload=True)
    except Exception as exc:  # noqa: BLE001
        import traceback
        summary["status"] = "V14_OUTSIDE_HOOK_FAILED"
        summary["exception_type"] = exc.__class__.__name__
        summary["exception"] = repr(exc)
        summary["traceback_tail"] = traceback.format_exc()[-6000:]
        write_json(out_json, summary, print_payload=True)
    finally:
        sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == "__main__":
    main()
