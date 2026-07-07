"""V4 MULTI-SEED closed-loop eval of the DEEP-SLOW state-reactive policy (no clock).

Loads the Training-v4 checkpoint and evaluates it CLOSED-LOOP across MULTIPLE seeds
and MULTIPLE EMA action-filter betas in ONE SimulationApp (one Isaac launch per
container — respects the proven per-container isolation model; loops internally by
resetting the scene). Reports the DISTRIBUTION (per seed, per beta), not one run,
because the GPU physics is non-deterministic.

For each (beta, seed) config it runs `episodes_per_config` randomized-start rollouts
with the V16 state-reactive observation (NO progress/phase clock), physics-resolved
articulation, the winning V14 disk geometry, and ballistic rejection. Aggregates
clean-success / ballistic / width_after per beta across seeds, picks the best beta,
and writes the honest verdict.

Output: /workspace/.../eval_v4_deep_slow/eval_v4_deep_slow_summary.json
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
    drive_to,
    patch_center,
)
from so101_state_reactive_common_v16 import compact_state_reactive, joint_positions_from_scene

EVAL_DIR = Path("/workspace/leisaac/tera_checkpoints/eval_v4_deep_slow")
OUT_JSON = EVAL_DIR / "eval_v4_deep_slow_summary.json"

BALLISTIC_VPEAK = 4.9
BALLISTIC_MAXDISP = 0.5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V4 multi-seed state-reactive SO-101 policy eval.")
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
    p.add_argument("--seeds", type=int, nargs="+", default=[3030, 4040, 5050])
    p.add_argument("--betas", type=float, nargs="+", default=[0.0, 0.3, 0.4, 0.6])
    p.add_argument("--episodes_per_config", type=int, default=10)
    p.add_argument("--rollout_steps", type=int, default=240)
    p.add_argument("--settle_steps", type=int, default=20)
    p.add_argument("--start_jitter", type=float, default=0.02)
    p.add_argument("--time_budget_s", type=float, default=3000.0)
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


def _rollout(scene, model, norm, base, start_points, patch_radius, center_fn, edge,
             steps: int, beta: float) -> dict[str, Any]:
    s_mean, s_std, a_mean, a_std = norm
    reset_scene_to_points(scene, start_points)
    dt = float(scene.sim.get_physics_dt())
    prev_action = base.astype(np.float32).copy()
    filt_action = base.astype(np.float32).copy()
    jaws0, _ = jaw_positions_from_scene(scene)
    prev_jaw = np.asarray(jaws0, np.float32).reshape(6)[3:6].copy()
    prev_pts = start_points.copy()
    widths = [float(size_metrics(start_points)["width_x"])]
    vpeak, nearest_min = 0.0, float("inf")
    for _ in range(int(steps)):
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
        if beta > 0.0:
            filt_action = (beta * filt_action + (1.0 - beta) * act).astype(np.float32)
            applied = filt_action
        else:
            applied = act
        apply_action_step(scene, applied)
        pts = scene.clean_towel.points_numpy().astype(np.float32)
        c = center_fn(scene)
        fd = np.linalg.norm(pts - prev_pts, axis=1) / max(dt, 1e-6)
        near = np.linalg.norm(pts[:, :2] - c[:2].reshape(1, 2), axis=1) < (float(patch_radius) + 0.06)
        if np.any(near):
            vpeak = max(vpeak, float(np.max(fd[near])))
        nearest_min = min(nearest_min, float(np.min(np.linalg.norm(pts - c.reshape(1, 3), axis=1))) - float(patch_radius))
        widths.append(float(size_metrics(pts)["width_x"]))
        prev_pts = pts
        prev_action = applied
        prev_jaw = right_jaw.copy()
    final_pts = scene.clean_towel.points_numpy().astype(np.float32)
    warr = np.asarray(widths, np.float32)
    m = {
        "edge_displacement_m": float(edge_center_displacement(start_points, final_pts, edge, EDGE_BAND)),
        "max_particle_displacement_m": float(np.max(np.linalg.norm(final_pts - start_points, axis=1))),
        "particle_velocity_peak": float(vpeak),
        "actual_touch_distance_m": float(max(0.0, nearest_min)),
        "width_reduction_m": float(warr[0] - warr.min()),
        "width_before": float(warr[0]), "width_after_best": float(warr.min()),
    }
    return m


def main() -> None:
    args = parse_args()
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    app = AppLauncher(args).app  # noqa: F841
    patch_offset = tuple(float(x) for x in args.patch_offset)
    started = time.monotonic()

    summary: dict[str, Any] = {
        "status": "V4_MULTISEED_EVAL_STARTED",
        "command_run": command_run_string(),
        "checkpoint": str(args.checkpoint),
        "closed_loop": True,
        "no_phase_clock": True,
        "seeds": list(args.seeds),
        "betas": list(args.betas),
        "episodes_per_config": int(args.episodes_per_config),
        "configs": [],
    }
    try:
        model, bundle = load_policy(args.checkpoint)
        if not bool(bundle.get("no_phase_clock", False)):
            raise RuntimeError("checkpoint is not a no_phase_clock policy")
        norm = (np.asarray(bundle["state_mean"], np.float32), np.asarray(bundle["state_std"], np.float32),
                np.asarray(bundle["action_mean"], np.float32), np.asarray(bundle["action_std"], np.float32))

        attach_info: dict[str, Any] = {}

        def _attach():
            attach_info.update(attach_patch(
                "/World/Right_Robot", name=f"v4eval_{args.config_name}", shape=args.patch_shape,
                radius=float(args.patch_radius), length=float(args.patch_length), axis=args.patch_axis,
                body_match="jaw", local_offset=patch_offset, contact_offset=float(args.patch_contact_offset),
                mat_path_str="/World/Materials/v4eval_mat"))

        scene = create_standalone_so101_clean_towel_scene(
            args, clean_usd_path=args.usd_path, towel_translation=STABLE_TOWEL_TRANSLATION,
            right_base=tuple(args.right_base), pre_reset_spawn=_attach)
        summary["patch_attach"] = attach_info
        scene.step_zero(int(args.settle_steps))
        base_start_points = scene.clean_towel.points_numpy().astype(np.float32)
        base = default_action12(scene)
        edge = "right"

        def center_fn(sc: Any) -> np.ndarray:
            return patch_center(sc, patch_offset, "jaw")

        reset_scene_to_points(scene, base_start_points)
        drive_to(scene, base, 120)
        base_pts = scene.clean_towel.points_numpy().astype(np.float32)
        baseline_edge = float(edge_center_displacement(base_start_points, base_pts, edge, EDGE_BAND))
        summary["baseline_no_contact_edge_disp_m"] = baseline_edge

        # beta-major order: each beta completes across all seeds before the next,
        # so a time-budget truncation still leaves earlier betas fully measured.
        for beta in args.betas:
            for seed in args.seeds:
                if time.monotonic() - started > float(args.time_budget_s):
                    summary["time_budget_truncated"] = True
                    break
                rng = np.random.default_rng(int(seed))
                clean_c, ball_c, reduce_c = 0, 0, 0
                w_afters, edge_clean = [], []
                for ep in range(int(args.episodes_per_config)):
                    off = rng.uniform(-float(args.start_jitter), float(args.start_jitter), size=2).astype(np.float32)
                    sp = base_start_points.copy()
                    sp[:, 0] += off[0]; sp[:, 1] += off[1]
                    m = _rollout(scene, model, norm, base, sp, float(args.patch_radius), center_fn, edge,
                                 int(args.rollout_steps), float(beta))
                    cls = classify_fold(m, baseline_edge=baseline_edge)
                    ballistic = bool(cls["spurious_ballistic"] or m["particle_velocity_peak"] >= BALLISTIC_VPEAK
                                     or m["max_particle_displacement_m"] > BALLISTIC_MAXDISP)
                    clean = bool(cls["valid_controlled_contact"] and not ballistic)
                    clean_c += int(clean); ball_c += int(ballistic)
                    reduce_c += int(m["width_after_best"] < m["width_before"] - 1e-4)
                    w_afters.append(m["width_after_best"])
                    if not ballistic:
                        edge_clean.append(m["edge_displacement_m"])
                nconf = int(args.episodes_per_config)
                summary["configs"].append({
                    "beta": float(beta), "seed": int(seed), "episodes": nconf,
                    "clean_success_rate": float(clean_c / max(1, nconf)),
                    "ballistic_rate": float(ball_c / max(1, nconf)),
                    "num_reduce_width": int(reduce_c),
                    "mean_width_after": float(np.mean(w_afters)) if w_afters else None,
                    "min_width_after": float(min(w_afters)) if w_afters else None,
                    "mean_edge_displacement_clean": float(np.mean(edge_clean)) if edge_clean else None,
                })
            else:
                continue
            break  # broke out of inner loop due to time budget

        # ---- aggregate per beta across seeds ----
        by_beta = {}
        for beta in args.betas:
            rows = [c for c in summary["configs"] if c["beta"] == beta]
            if not rows:
                continue
            clean = [r["clean_success_rate"] for r in rows]
            ball = [r["ballistic_rate"] for r in rows]
            wa = [r["mean_width_after"] for r in rows if r["mean_width_after"] is not None]
            by_beta[f"{beta:.2f}"] = {
                "seeds": [r["seed"] for r in rows],
                "clean_success_rate_by_seed": clean,
                "ballistic_rate_by_seed": ball,
                "mean_clean_success_rate": float(np.mean(clean)),
                "min_clean_success_rate": float(np.min(clean)),
                "mean_ballistic_rate": float(np.mean(ball)),
                "max_ballistic_rate": float(np.max(ball)),
                "mean_width_after": float(np.mean(wa)) if wa else None,
            }
        summary["by_beta"] = by_beta

        # best beta: meets targets if possible (clean>=0.8, ballistic<0.15, width<0.55),
        # else maximize a robustness score = mean_clean - 2*mean_ballistic - max(0, width-0.55)
        def score(b):
            v = by_beta[b]
            w = v["mean_width_after"] if v["mean_width_after"] is not None else 1.0
            meets = (v["mean_clean_success_rate"] >= 0.8 and v["max_ballistic_rate"] < 0.15 and w < 0.55)
            return (1 if meets else 0, v["mean_clean_success_rate"] - 2.0 * v["mean_ballistic_rate"] - max(0.0, w - 0.55))
        best_beta = max(by_beta, key=score) if by_beta else None
        summary["best_beta"] = best_beta
        bb = by_beta.get(best_beta, {}) if best_beta is not None else {}
        summary.update(
            best_beta_mean_clean_success_rate=bb.get("mean_clean_success_rate"),
            best_beta_mean_ballistic_rate=bb.get("mean_ballistic_rate"),
            best_beta_max_ballistic_rate=bb.get("max_ballistic_rate"),
            best_beta_mean_width_after=bb.get("mean_width_after"),
            meets_clean_0_8=bool(bb.get("mean_clean_success_rate", 0) >= 0.8),
            meets_ballistic_below_0_15=bool(bb.get("max_ballistic_rate", 1.0) < 0.15),
            meets_width_below_0_55=bool((bb.get("mean_width_after") or 1.0) < 0.55),
            robust_autonomous_so101_folding_validated=bool(
                bb.get("mean_clean_success_rate", 0) >= 0.8 and bb.get("max_ballistic_rate", 1.0) < 0.15
                and (bb.get("mean_width_after") or 1.0) < 0.55),
            status="V4_MULTISEED_EVAL_DONE",
        )
        write_json(OUT_JSON, summary, print_payload=True)
    except Exception as exc:  # noqa: BLE001
        import traceback
        summary["status"] = "V4_MULTISEED_EVAL_FAILED"
        summary["exception_type"] = exc.__class__.__name__
        summary["exception"] = repr(exc)
        summary["traceback_tail"] = traceback.format_exc()[-6000:]
        write_json(OUT_JSON, summary, print_payload=True)
    finally:
        sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == "__main__":
    main()
