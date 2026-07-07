"""V3 CLOSED-LOOP eval of the STATE-REACTIVE SO-101 fold policy (NO phase clock).

Loads the Training-v3 checkpoint and runs it CLOSED-LOOP in a fresh Isaac SO-101 +
towel scene with the winning V14 disk contact geometry. At each physics step the
policy predicts a 12-D joint target from the V16 STATE-REACTIVE observation
(`compact_state_reactive`: real joint angles + jaw pose/velocity + previous action
+ cloth metrics + edge features) — with NO progress/phase clock — applied via
physics-resolved articulation.

Because the observation carries no timeline, a fold in rollout is evidence of
state-reactive control, not trajectory imitation. Ballistic rollouts are REJECTED
(not counted as clean success) and their rate is reported. Starts are randomized
(small towel x/y offset) so success is not a single-pose artifact.

Output: <out_dir>/eval_v3_state_reactive_summary.json
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
from clean_towel_v4_common import apply_action_step, jaw_positions_from_scene, reset_scene_to_points
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
    patch_center,
)
from so101_state_reactive_common_v16 import compact_state_reactive, joint_positions_from_scene

EVAL_DIR = Path("/workspace/leisaac/tera_checkpoints/eval_v3_state_reactive")
OUT_JSON = EVAL_DIR / "eval_v3_state_reactive_summary.json"

BALLISTIC_VPEAK = 4.9      # particle-system velocity cap => ballistic fling
BALLISTIC_MAXDISP = 0.5    # runaway particle displacement


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V3 closed-loop state-reactive SO-101 policy eval.")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--usd_path", type=Path, default=CLEAN_TOWEL_USD_PATH)
    p.add_argument("--config_name", type=str, default="disk_z_r14")
    p.add_argument("--right_base", type=float, nargs=3, default=list(DEFAULT_RIGHT_BASE))
    p.add_argument("--patch_shape", type=str, default="cylinder", choices=["sphere", "cylinder", "capsule"])
    p.add_argument("--patch_radius", type=float, default=0.14)
    p.add_argument("--patch_length", type=float, default=0.03)
    p.add_argument("--patch_axis", type=str, default="Z", choices=["X", "Y", "Z"])
    p.add_argument("--patch_offset", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    p.add_argument("--patch_contact_offset", type=float, default=0.04)
    p.add_argument("--num_episodes", type=int, default=20)
    p.add_argument("--rollout_steps", type=int, default=230)
    p.add_argument("--settle_steps", type=int, default=20)
    p.add_argument("--start_jitter", type=float, default=0.02, help="max towel x/y start offset (m)")
    p.add_argument("--action_smooth_beta", type=float, default=0.0,
                   help="optional EMA action filter (0=off, raw policy). Headline uses raw.")
    p.add_argument("--seed", type=int, default=3030)
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
        "status": "V3_EVAL_STARTED",
        "command_run": command_run_string(),
        "checkpoint": str(args.checkpoint),
        "config_name": args.config_name,
        "closed_loop": True,
        "no_phase_clock": True,
        "drive_mode": "physics_resolved_articulation (state-reactive policy predicts joint targets each step)",
        "action_smooth_beta": float(args.action_smooth_beta),
        "num_eval_episodes": int(args.num_episodes),
        "episodes": [],
    }
    try:
        model, bundle = load_policy(args.checkpoint)
        if not bool(bundle.get("no_phase_clock", False)):
            raise RuntimeError("checkpoint is not a no_phase_clock policy; refusing to eval as state-reactive")
        s_mean = np.asarray(bundle["state_mean"], np.float32); s_std = np.asarray(bundle["state_std"], np.float32)
        a_mean = np.asarray(bundle["action_mean"], np.float32); a_std = np.asarray(bundle["action_std"], np.float32)

        attach_info: dict[str, Any] = {}

        def _attach():
            attach_info.update(attach_patch(
                "/World/Right_Robot", name=f"v3eval_{args.config_name}", shape=args.patch_shape,
                radius=float(args.patch_radius), length=float(args.patch_length), axis=args.patch_axis,
                body_match="jaw", local_offset=patch_offset, contact_offset=float(args.patch_contact_offset),
                mat_path_str="/World/Materials/v3eval_mat"))

        scene = create_standalone_so101_clean_towel_scene(
            args, clean_usd_path=args.usd_path, towel_translation=STABLE_TOWEL_TRANSLATION,
            right_base=tuple(args.right_base), pre_reset_spawn=_attach)
        summary["patch_attach"] = attach_info
        scene.step_zero(int(args.settle_steps))
        base_start_points = scene.clean_towel.points_numpy().astype(np.float32)
        base = default_action12(scene)
        edge = "right"
        steps = int(args.rollout_steps)

        def center_fn(sc: Any) -> np.ndarray:
            return patch_center(sc, patch_offset, "jaw")

        # no-contact baseline for the classifier (disk must be inert at rest)
        reset_scene_to_points(scene, base_start_points)
        from so101_edge_capture_common_v14 import drive_to
        drive_to(scene, base, 120)
        base_pts = scene.clean_towel.points_numpy().astype(np.float32)
        baseline_edge = float(edge_center_displacement(base_start_points, base_pts, edge, EDGE_BAND))
        summary["baseline_no_contact_edge_disp_m"] = baseline_edge

        rng = np.random.default_rng(int(args.seed))
        n = int(args.num_episodes)
        clean_success, ballistic_count = 0, 0
        widths_before, widths_after, edge_disps = [], [], []
        beta = float(args.action_smooth_beta)

        for ep in range(n):
            off = rng.uniform(-float(args.start_jitter), float(args.start_jitter), size=2).astype(np.float32)
            start_points = base_start_points.copy()
            start_points[:, 0] += off[0]; start_points[:, 1] += off[1]
            reset_scene_to_points(scene, start_points)

            dt = float(scene.sim.get_physics_dt())
            prev_action = base.astype(np.float32).copy()
            filt_action = base.astype(np.float32).copy()
            jaws0, _ = jaw_positions_from_scene(scene)
            prev_jaw = np.asarray(jaws0, np.float32).reshape(6)[3:6].copy()
            prev_pts = start_points.copy()
            widths = [float(size_metrics(start_points)["width_x"])]
            vpeak, nearest_min = 0.0, float("inf")

            for step in range(steps):
                cur = scene.clean_towel.points_numpy().astype(np.float32)
                joints, _ = joint_positions_from_scene(scene)
                jaws, _ = jaw_positions_from_scene(scene)
                right_jaw = np.asarray(jaws, np.float32).reshape(6)[3:6]
                jaw_vel = ((right_jaw - prev_jaw) / max(dt, 1e-6)).astype(np.float32)
                state, _ = compact_state_reactive(cur, joints, jaws, jaw_vel, prev_action, edge=edge)
                x = (state - s_mean) / s_std
                with torch.no_grad():
                    act = model(torch.from_numpy(x).float().unsqueeze(0)).squeeze(0).cpu().numpy() * a_std + a_mean
                act = act.astype(np.float32)
                if beta > 0.0:  # optional EMA safety filter (headline run keeps beta=0)
                    filt_action = (beta * filt_action + (1.0 - beta) * act).astype(np.float32)
                    applied = filt_action
                else:
                    applied = act
                apply_action_step(scene, applied)
                pts = scene.clean_towel.points_numpy().astype(np.float32)
                c = center_fn(scene)
                fd = np.linalg.norm(pts - prev_pts, axis=1) / max(dt, 1e-6)
                near = np.linalg.norm(pts[:, :2] - c[:2].reshape(1, 2), axis=1) < (float(args.patch_radius) + 0.06)
                if np.any(near):
                    vpeak = max(vpeak, float(np.max(fd[near])))
                nearest_min = min(nearest_min, float(np.min(np.linalg.norm(pts - c.reshape(1, 3), axis=1))) - float(args.patch_radius))
                widths.append(float(size_metrics(pts)["width_x"]))
                prev_pts = pts
                prev_action = applied
                prev_jaw = right_jaw.copy()

            final_pts = scene.clean_towel.points_numpy().astype(np.float32)
            warr = np.asarray(widths, np.float32)
            m = {
                "actual_touch_distance_m": float(max(0.0, nearest_min)),
                "edge_displacement_m": float(edge_center_displacement(start_points, final_pts, edge, EDGE_BAND)),
                "max_particle_displacement_m": float(np.max(np.linalg.norm(final_pts - start_points, axis=1))),
                "particle_velocity_peak": float(vpeak),
                "width_reduction_m": float(warr[0] - warr.min()),
            }
            cls = classify_fold(m, baseline_edge=baseline_edge)
            ballistic = bool(cls["spurious_ballistic"] or vpeak >= BALLISTIC_VPEAK
                             or m["max_particle_displacement_m"] > BALLISTIC_MAXDISP)
            clean = bool(cls["valid_controlled_contact"] and not ballistic)
            clean_success += int(clean)
            ballistic_count += int(ballistic)
            widths_before.append(float(warr[0])); widths_after.append(float(warr.min()))
            # only aggregate edge displacement over NON-ballistic episodes: a ballistic
            # rollout can blow particles to huge finite coordinates, which would poison
            # a naive mean. Ballistic episodes are separately counted in ballistic_rate.
            if not ballistic:
                edge_disps.append(m["edge_displacement_m"])
            summary["episodes"].append({
                "episode": ep, "towel_dx": float(off[0]), "towel_dy": float(off[1]),
                "width_before": float(warr[0]), "width_after_best": float(warr.min()),
                "width_reduction_m": m["width_reduction_m"], "edge_displacement_m": m["edge_displacement_m"],
                "particle_velocity_peak": vpeak, "max_particle_displacement_m": m["max_particle_displacement_m"],
                "valid_controlled_contact": bool(cls["valid_controlled_contact"]),
                "ballistic": ballistic, "clean_success": clean,
            })

        mean_after = float(np.mean(widths_after)) if widths_after else None
        clean_rate = float(clean_success / max(1, n))
        ballistic_rate = float(ballistic_count / max(1, n))
        summary.update(
            num_clean_success=int(clean_success),
            num_ballistic=int(ballistic_count),
            success_rate_clean_non_ballistic=clean_rate,
            ballistic_rate=ballistic_rate,
            mean_width_before=float(np.mean(widths_before)) if widths_before else None,
            mean_width_after=mean_after,
            mean_edge_displacement_clean=float(np.mean(edge_disps)) if edge_disps else None,
            num_reduce_width=int(sum(1 for wb, wa in zip(widths_before, widths_after) if wa < wb - 1e-4)),
            # honest verdicts vs the goal's acceptance criteria
            meets_clean_success_target_0_8=bool(clean_rate >= 0.8),
            beats_v2_clean_0_5=bool(clean_rate > 0.5),
            mean_width_after_below_0_55=bool(mean_after is not None and mean_after < 0.55),
            ballistic_rate_below_0_2=bool(ballistic_rate < 0.2),
            no_phase_clock_policy_works=bool(clean_rate > 0.5 and ballistic_rate < 0.2),
            autonomous_so101_fold_policy_robust=bool(clean_rate >= 0.8 and ballistic_rate < 0.2
                                                     and mean_after is not None and mean_after < 0.55),
            status="V3_EVAL_DONE",
        )
        write_json(OUT_JSON, summary, print_payload=True)
    except Exception as exc:  # noqa: BLE001
        import traceback
        summary["status"] = "V3_EVAL_FAILED"
        summary["exception_type"] = exc.__class__.__name__
        summary["exception"] = repr(exc)
        summary["traceback_tail"] = traceback.format_exc()[-6000:]
        write_json(OUT_JSON, summary, print_payload=True)
    finally:
        sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == "__main__":
    main()
