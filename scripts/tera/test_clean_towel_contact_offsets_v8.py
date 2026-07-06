from __future__ import annotations

import argparse
import itertools
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from isaaclab.app import AppLauncher

from so101_clean_towel_scene_utils_v1 import CLEAN_TOWEL_USD_PATH, command_run_string, force_headless_no_cameras, size_metrics, write_json
from test_clean_towel_primitive_contact_v8 import (
    CONTACT_OFFSETS_SUMMARY_JSON,
    V8_DIR,
    apply_contact_params,
    create_towel_primitive_scene,
    default_primitive_params,
    result_sanity,
    run_primitive_contact_trial,
    step_sim,
)


def physically_plausible_success(result: dict[str, Any]) -> bool:
    return bool(
        result.get("sanity_valid", False)
        and result.get("actual_touch_achieved", False)
        and result.get("primitive_moves_towel", False)
    )


def result_score(result: dict[str, Any]) -> float:
    touch_bonus = 1.0 if result.get("actual_touch_achieved") else 0.0
    plausible_bonus = 2.0 if physically_plausible_success(result) else 0.0
    return (
        plausible_bonus
        + touch_bonus
        + 8.0 * float(result.get("selected_edge_particle_displacement", 0.0))
        + 5.0 * max(0.0, 0.68 - float(result.get("best_width", 0.68)))
        - 0.20 * float(result.get("nearest_collider_to_towel_particle_m", 99.0))
    )


def build_contact_grid(base: dict[str, Any], max_trials: int) -> list[dict[str, Any]]:
    grid = list(
        itertools.product(
            [0.004, 0.010, 0.020, 0.035],
            [0.0, 0.002, 0.005],
            [0.004, 0.006, 0.010],
            [(0.8, 0.6), (1.6, 1.2), (3.0, 2.4)],
            [0.025, 0.040, 0.060],
            [0.002, 0.012, 0.026],
            [0.025, 0.055, 0.085],
        )
    )
    if len(grid) > int(max_trials):
        indices = np.linspace(0, len(grid) - 1, int(max_trials), dtype=np.int32)
        grid = [grid[int(idx)] for idx in indices]
    out = []
    for trial_id, (contact_offset, rest_offset, particle_radius, friction, primitive_radius, height, penetration) in enumerate(grid):
        params = dict(base)
        params.update(
            {
                "trial_id": int(trial_id),
                "cloth_contact_offset": float(contact_offset),
                "cloth_rest_offset": float(rest_offset),
                "particle_radius": float(particle_radius),
                "primitive_contact_offset": float(contact_offset),
                "primitive_rest_offset": float(rest_offset),
                "static_friction": float(friction[0]),
                "dynamic_friction": float(friction[1]),
                "primitive_radius": float(primitive_radius),
                "primitive_height": float(height),
                "penetration_depth": float(penetration),
            }
        )
        out.append(params)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V8 clean towel primitive contact-offset sweep.")
    parser.add_argument("--usd_path", type=Path, default=CLEAN_TOWEL_USD_PATH)
    parser.add_argument("--out_dir", type=Path, default=V8_DIR)
    parser.add_argument("--edge", type=str, default="right", choices=["left", "right", "front", "back"])
    parser.add_argument("--edge_band", type=float, default=0.035)
    parser.add_argument("--max_trials", type=int, default=48)
    parser.add_argument("--settle_initial_steps", type=int, default=30)
    parser.add_argument("--settle_before_steps", type=int, default=4)
    parser.add_argument("--approach_steps", type=int, default=40)
    parser.add_argument("--hold_contact_steps", type=int, default=10)
    parser.add_argument("--drag_steps", type=int, default=60)
    parser.add_argument("--settle_after_steps", type=int, default=14)
    parser.add_argument("--primitive_radius", type=float, default=0.04)
    parser.add_argument("--primitive_height", type=float, default=0.012)
    parser.add_argument("--penetration_depth", type=float, default=0.055)
    parser.add_argument("--drag_distance", type=float, default=0.28)
    parser.add_argument("--drag_height_delta", type=float, default=0.0)
    parser.add_argument("--outside_margin", type=float, default=0.05)
    parser.add_argument("--primitive_contact_offset", type=float, default=0.012)
    parser.add_argument("--primitive_rest_offset", type=float, default=0.0)
    parser.add_argument("--cloth_contact_offset", type=float, default=0.005)
    parser.add_argument("--cloth_rest_offset", type=float, default=0.003)
    parser.add_argument("--particle_radius", type=float, default=0.005)
    parser.add_argument("--static_friction", type=float, default=1.2)
    parser.add_argument("--dynamic_friction", type=float, default=1.0)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    force_headless_no_cameras(args)
    return args


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    app_launcher = AppLauncher(args)
    _simulation_app = app_launcher.app
    try:
        scene, scene_info = create_towel_primitive_scene(
            args,
            primitive_radius=float(args.primitive_radius),
            primitive_contact_offset=float(args.primitive_contact_offset),
            primitive_rest_offset=float(args.primitive_rest_offset),
            static_friction=float(args.static_friction),
            dynamic_friction=float(args.dynamic_friction),
        )
        step_sim(scene, int(args.settle_initial_steps))
        start_points = scene.clean_towel.points_numpy().astype(np.float32)
        base = default_primitive_params(args)
        results = []
        for params in build_contact_grid(base, int(args.max_trials)):
            applied = apply_contact_params(scene, params)
            result = run_primitive_contact_trial(
                scene,
                start_points,
                params,
                edge_band=float(args.edge_band),
                reset_points=True,
            )
            result["contact_param_application"] = applied
            result["physically_plausible_success"] = physically_plausible_success(result)
            results.append(result)

        valid_results = [item for item in results if item.get("sanity_valid")]
        plausible_successes = [item for item in valid_results if physically_plausible_success(item)]
        touch_results = [item for item in valid_results if item.get("actual_touch_achieved")]
        best = max(valid_results, key=result_score) if valid_results else None
        best_plausible = max(plausible_successes, key=result_score) if plausible_successes else None
        best_touch = max(touch_results, key=result_score) if touch_results else None
        move_best = (
            max(valid_results, key=lambda item: float(item.get("selected_edge_particle_displacement", 0.0)))
            if valid_results
            else None
        )
        width_best = min(valid_results, key=lambda item: float(item.get("best_width", 99.0))) if valid_results else None
        physically_plausible_setting_found = bool(best_plausible is not None)
        summary = {
            "status": "CLEAN_TOWEL_CONTACT_OFFSETS_SWEEP_V8_DONE",
            "command_run": command_run_string(),
            "scene_info": scene_info,
            "start_towel_metrics": size_metrics(start_points),
            "num_trials": int(len(results)),
            "num_sanity_valid_trials": int(len(valid_results)),
            "num_actual_touch_trials": int(len(touch_results)),
            "physically_plausible_setting_found": physically_plausible_setting_found,
            "primitive_moves_towel_found": bool(
                any(item.get("primitive_moves_towel", False) and item.get("sanity_valid", False) for item in results)
            ),
            "primitive_width_reduction_success_found": bool(
                any(item.get("primitive_width_reduction_success", False) and item.get("sanity_valid", False) for item in results)
            ),
            "best_result": best,
            "best_touch_result": best_touch,
            "best_plausible_result": best_plausible,
            "edge_displacement_best_result": move_best,
            "width_best_result": width_best,
            "best_contact_setting": None if best_plausible is None else best_plausible.get("primitive_params"),
            "best_contact_setting_basis": (
                "requires sanity_valid, actual_touch_achieved, and primitive_moves_towel"
            ),
            "results": results,
            "contact_offsets_sweep_summary_path": str(args.out_dir / CONTACT_OFFSETS_SUMMARY_JSON.name),
            "honest_note": (
                "This sweep uses a kinematic primitive collider and runtime-accessible USD/PhysX contact settings. "
                "It does not use SO-101, particle attachment, cameras, policy training, or autonomous folding."
            ),
        }
        write_json(args.out_dir / CONTACT_OFFSETS_SUMMARY_JSON.name, summary, print_payload=True)
    except Exception as exc:
        summary = {
            "status": "CLEAN_TOWEL_CONTACT_OFFSETS_SWEEP_V8_FAILED",
            "command_run": command_run_string(),
            "exception_type": exc.__class__.__name__,
            "exception": repr(exc),
            "physically_plausible_setting_found": False,
            "best_contact_setting": None,
            "honest_note": "Contact offset sweep failed; no contact, towel motion, folding, or training claim is made.",
        }
        write_json(args.out_dir / CONTACT_OFFSETS_SUMMARY_JSON.name, summary, print_payload=True)
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
