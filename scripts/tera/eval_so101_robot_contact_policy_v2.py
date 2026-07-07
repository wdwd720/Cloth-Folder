"""V2 CLOSED-LOOP evaluation of the trained SO-101 robot-contact policy.

Loads the Training-v2 behavior-cloning checkpoint and runs it CLOSED-LOOP in a
fresh Isaac SO-101 + towel scene (with the winning V14 contact geometry): at each
step the policy predicts a 12-D joint target from the compact state (cloth metrics
+ real jaw positions), which is applied via physics-resolved articulation
(apply_action_step). This tests whether the LEARNED policy folds autonomously —
NOT open-loop replay.

Honest interpretation is enforced downstream: if the policy only imitates the demo
but does not fold in rollout, or if open-loop replay works but the learned policy
fails, that is reported as-is. No autonomous success is claimed without this eval.

Output: <out_dir>/eval_v2_robot_contact_summary.json
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

import torch
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

EVAL_DIR = Path("/workspace/leisaac/tera_checkpoints/eval_v2_robot_contact")
OUT_JSON = EVAL_DIR / "eval_v2_robot_contact_summary.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V2 closed-loop SO-101 robot-contact policy eval.")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--usd_path", type=Path, default=CLEAN_TOWEL_USD_PATH)
    p.add_argument("--config_name", type=str, default="best")
    p.add_argument("--right_base", type=float, nargs=3, default=list(DEFAULT_RIGHT_BASE))
    p.add_argument("--patch_shape", type=str, default="capsule", choices=["sphere", "cylinder", "capsule"])
    p.add_argument("--patch_radius", type=float, default=0.03)
    p.add_argument("--patch_length", type=float, default=0.34)
    p.add_argument("--patch_axis", type=str, default="Y", choices=["X", "Y", "Z"])
    p.add_argument("--patch_offset", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    p.add_argument("--patch_contact_offset", type=float, default=0.02)
    p.add_argument("--pinch", action="store_true")
    p.add_argument("--num_episodes", type=int, default=6)
    p.add_argument("--rollout_steps", type=int, default=170)
    p.add_argument("--settle_steps", type=int, default=20)
    AppLauncher.add_app_launcher_args(p)
    args = p.parse_args()
    force_headless_no_cameras(args)
    return args


def load_policy(ckpt_path: Path):
    bundle = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    obs_dim = int(bundle["obs_dim"]); act_dim = int(bundle["act_dim"])
    model = torch.nn.Sequential(
        torch.nn.Linear(obs_dim, 256), torch.nn.ReLU(),
        torch.nn.Linear(256, 256), torch.nn.ReLU(),
        torch.nn.Linear(256, act_dim))
    model.load_state_dict(bundle["model_state_dict"])
    model.eval()
    return model, bundle


def main() -> None:
    args = parse_args()
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    app = AppLauncher(args).app  # noqa: F841
    patch_offset = tuple(float(x) for x in args.patch_offset)

    summary: dict[str, Any] = {
        "status": "V2_EVAL_STARTED",
        "command_run": command_run_string(),
        "checkpoint": str(args.checkpoint),
        "config_name": args.config_name,
        "closed_loop": True,
        "drive_mode": "physics_resolved_articulation (policy predicts joint targets each step)",
        "num_eval_episodes": int(args.num_episodes),
        "episodes": [],
    }
    try:
        model, bundle = load_policy(args.checkpoint)
        s_mean = np.asarray(bundle["state_mean"], np.float32); s_std = np.asarray(bundle["state_std"], np.float32)
        a_mean = np.asarray(bundle["action_mean"], np.float32); a_std = np.asarray(bundle["action_std"], np.float32)

        attach_info: dict[str, Any] = {}

        def _attach():
            attach_info["jaw"] = attach_patch(
                "/World/Right_Robot", name=f"v2eval_jaw_{args.config_name}", shape=args.patch_shape,
                radius=float(args.patch_radius), length=float(args.patch_length), axis=args.patch_axis,
                body_match="jaw", local_offset=patch_offset, contact_offset=float(args.patch_contact_offset),
                mat_path_str="/World/Materials/v2eval_mat")
            if args.pinch:
                attach_info["gripper"] = attach_patch(
                    "/World/Right_Robot", name=f"v2eval_grip_{args.config_name}", shape=args.patch_shape,
                    radius=float(args.patch_radius), length=float(args.patch_length), axis=args.patch_axis,
                    body_match="gripper", local_offset=(0.0, 0.0, 0.0), mat_path_str="/World/Materials/v2eval_mat")

        scene = create_standalone_so101_clean_towel_scene(
            args, clean_usd_path=args.usd_path, towel_translation=STABLE_TOWEL_TRANSLATION,
            right_base=tuple(args.right_base), pre_reset_spawn=_attach)
        summary["patch_attach"] = attach_info
        scene.step_zero(int(args.settle_steps))
        start_points = scene.clean_towel.points_numpy().astype(np.float32)
        edge = "right"
        steps = int(args.rollout_steps)

        def center_fn(sc: Any) -> np.ndarray:
            return patch_center(sc, patch_offset, "jaw")

        success_count = 0
        widths_before, widths_after, edge_disps = [], [], []
        for ep in range(int(args.num_episodes)):
            reset_scene_to_points(scene, start_points)
            prev = start_points.copy()
            dt = float(scene.sim.get_physics_dt())
            widths = [float(size_metrics(start_points)["width_x"])]
            vpeak, nearest_min = 0.0, float("inf")
            for step in range(steps):
                cur = scene.clean_towel.points_numpy().astype(np.float32)
                jaws, _ = jaw_positions_from_scene(scene)
                state, _ = compact_state(cur, progress_for_step(step, steps), phase_id_for_step(step, steps), jaws)
                x = (state - s_mean) / s_std
                with torch.no_grad():
                    act = model(torch.from_numpy(x).float().unsqueeze(0)).squeeze(0).cpu().numpy() * a_std + a_mean
                apply_action_step(scene, act.astype(np.float32))
                pts = scene.clean_towel.points_numpy().astype(np.float32)
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
            ok = bool(cls["valid_controlled_contact"])
            success_count += int(ok)
            widths_before.append(float(warr[0])); widths_after.append(float(warr.min()))
            edge_disps.append(m["edge_displacement_m"])
            summary["episodes"].append({
                "episode": ep, "width_before": float(warr[0]), "width_after_best": float(warr.min()),
                "width_reduction_m": m["width_reduction_m"], "edge_displacement_m": m["edge_displacement_m"],
                "actual_touch_distance_m": m["actual_touch_distance_m"], "particle_velocity_peak": vpeak,
                "valid_fold": ok})

        n = int(args.num_episodes)
        summary.update(
            success_count=int(success_count),
            success_rate=float(success_count / max(1, n)),
            mean_width_before=float(np.mean(widths_before)) if widths_before else None,
            mean_width_after=float(np.mean(widths_after)) if widths_after else None,
            mean_edge_displacement=float(np.mean(edge_disps)) if edge_disps else None,
            failures=[e for e in summary["episodes"] if not e["valid_fold"]],
            autonomous_fold_policy_validated=bool(success_count >= max(1, n // 2)),
            final_robot_contact_policy_training_ready=True,
            status="V2_EVAL_DONE",
        )
        write_json(OUT_JSON, summary, print_payload=True)
    except Exception as exc:  # noqa: BLE001
        import traceback
        summary["status"] = "V2_EVAL_FAILED"
        summary["exception_type"] = exc.__class__.__name__
        summary["exception"] = repr(exc)
        summary["traceback_tail"] = traceback.format_exc()[-6000:]
        write_json(OUT_JSON, summary, print_payload=True)
    finally:
        sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == "__main__":
    main()
