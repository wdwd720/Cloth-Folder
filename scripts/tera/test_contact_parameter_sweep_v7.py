from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
from isaaclab.app import AppLauncher

from inspect_so101_gripper_collision_v7 import (
    CONTACT_PARAMETER_RESULTS_JSON,
    CONTACT_PARAMETER_SUMMARY_JSON,
    EDGE_PINCH_DRAG_SUMMARY_JSON,
    V7_DIR,
    collision_prim_info,
    find_matching_body_prim_paths,
    load_json_if_exists,
    recommended_towel_center,
    recommended_towel_translation,
    v6_recommended_candidate,
)
from so101_clean_towel_scene_utils_v1 import (
    CLEAN_TOWEL_USD_PATH,
    TASK_ID,
    command_run_string,
    create_standalone_so101_clean_towel_scene,
    force_headless_no_cameras,
    standalone_robot_scene_info,
    write_json,
)
from test_so101_edge_pinch_drag_v7 import mapping_values, result_score, run_trial


def finite_float(value: Any) -> tuple[bool, float | None]:
    try:
        number = float(value)
    except Exception:
        return False, None
    return bool(np.isfinite(number)), number


def contact_result_sanity(result: dict[str, Any]) -> dict[str, Any]:
    reasons = []
    values = {}
    for key in (
        "nearest_jaw_to_towel_particle_m",
        "selected_edge_particle_displacement",
        "start_width",
        "best_width",
        "final_width",
    ):
        finite, number = finite_float(result.get(key))
        if not finite or number is None:
            reasons.append(f"{key}_not_finite")
            values[key] = None
        else:
            values[key] = number

    start_width = values.get("start_width")
    best_width = values.get("best_width")
    final_width = values.get("final_width")
    displacement = values.get("selected_edge_particle_displacement")

    if start_width is not None and not (0.45 <= start_width <= 0.85):
        reasons.append("start_width_outside_expected_clean_towel_range")
    if best_width is not None and not (0.10 <= best_width <= 1.20):
        reasons.append("best_width_outside_sane_range")
    if final_width is not None and not (0.10 <= final_width <= 1.20):
        reasons.append("final_width_outside_sane_range")
    if displacement is not None and not (0.0 <= displacement <= 1.0):
        reasons.append("edge_displacement_outside_sane_range")

    actual_touch = bool(result.get("actual_touch_achieved", False))
    return {
        "sanity_valid": bool(not reasons),
        "sanity_invalid_reasons": reasons,
        "actual_touch_confirmed": actual_touch,
    }


def should_skip_parameter_sweep() -> tuple[bool, str | None]:
    pure = load_json_if_exists(EDGE_PINCH_DRAG_SUMMARY_JSON)
    if pure and pure.get("contact_moves_towel"):
        return True, "pure_contact_already_moved_towel"
    return False, None


def numeric_attr_set(attr: Any, value: float) -> bool:
    try:
        current = attr.Get()
        if isinstance(current, bool):
            return False
        if current is None:
            return False
        if isinstance(current, (int, float)) or hasattr(current, "real"):
            attr.Set(float(value))
            return True
    except Exception:
        return False
    return False


def set_matching_attrs(prim: Any, patterns: tuple[str, ...], value: float) -> list[dict[str, Any]]:
    applied = []
    try:
        attrs = prim.GetAttributes()
    except Exception:
        attrs = []
    for attr in attrs:
        name = attr.GetName()
        lower = name.lower()
        if not any(pattern in lower for pattern in patterns):
            continue
        old_value = None
        try:
            old_value = attr.Get()
        except Exception:
            old_value = None
        if numeric_attr_set(attr, value):
            applied.append(
                {
                    "prim_path": str(prim.GetPath()),
                    "attr_name": name,
                    "old_value": None if old_value is None else str(old_value),
                    "new_value": float(value),
                }
            )
    return applied


def candidate_collision_paths(scene: Any, stage: Any) -> list[str]:
    paths = []
    for arm, root in (("left", scene.left_prim_path), ("right", scene.right_prim_path)):
        arm_obj = scene.left_arm if arm == "left" else scene.right_arm
        names = list(getattr(arm_obj.data, "body_names", []) or [])
        for name in names:
            if not any(keyword in name.lower() for keyword in ("jaw", "finger", "gripper")):
                continue
            for body_path in find_matching_body_prim_paths(stage, root, name):
                for info in collision_prim_info(stage, body_path):
                    if info.get("has_usdphysics_collision_api"):
                        paths.append(info["path"])
    return sorted(set(paths))


def bind_contact_material(stage: Any, prim_paths: list[str], static_friction: float, dynamic_friction: float) -> dict[str, Any]:
    result = {
        "requested_static_friction": float(static_friction),
        "requested_dynamic_friction": float(dynamic_friction),
        "material_path": None,
        "bound_prim_paths": [],
        "error": None,
    }
    try:
        from pxr import Sdf, UsdPhysics, UsdShade

        material_path = f"/World/V7ContactMaterial_sf_{str(static_friction).replace('.', '_')}_df_{str(dynamic_friction).replace('.', '_')}"
        material = UsdShade.Material.Define(stage, material_path)
        physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        physics_material.CreateStaticFrictionAttr(float(static_friction))
        physics_material.CreateDynamicFrictionAttr(float(dynamic_friction))
        physics_material.CreateRestitutionAttr(0.0)
        for prim_path in prim_paths:
            prim = stage.GetPrimAtPath(prim_path)
            if prim and prim.IsValid():
                UsdShade.MaterialBindingAPI(prim).Bind(material, bindingStrength=UsdShade.Tokens.strongerThanDescendants)
                result["bound_prim_paths"].append(prim_path)
        result["material_path"] = material_path
        _ = Sdf
    except Exception as exc:
        result["error"] = repr(exc)
    return result


def apply_contact_params(scene: Any, params: dict[str, Any]) -> dict[str, Any]:
    from isaacsim.core.utils.stage import get_current_stage

    stage = get_current_stage()
    applied_attrs = []
    towel_paths = [scene.clean_towel.root_path, scene.clean_towel.mesh_path, scene.clean_towel.particle_system_path]
    for path in towel_paths:
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            continue
        if params["cloth_contact_offset"] is not None:
            applied_attrs.extend(
                set_matching_attrs(
                    prim,
                    ("contactoffset", "contact_offset", "particlecontactoffset", "particle_contact_offset"),
                    float(params["cloth_contact_offset"]),
                )
            )
        if params["rest_offset"] is not None:
            applied_attrs.extend(
                set_matching_attrs(
                    prim,
                    ("restoffset", "rest_offset", "solidrestoffset", "solid_rest_offset"),
                    float(params["rest_offset"]),
                )
            )
        if params["particle_radius"] is not None:
            applied_attrs.extend(set_matching_attrs(prim, ("radius", "thickness"), float(params["particle_radius"])))

    collision_paths = candidate_collision_paths(scene, stage)
    for prim_path in collision_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            continue
        if params["jaw_contact_offset"] is not None:
            applied_attrs.extend(
                set_matching_attrs(
                    prim,
                    ("contactoffset", "contact_offset", "restoffset", "rest_offset"),
                    float(params["jaw_contact_offset"]),
                )
            )
    material_result = bind_contact_material(
        stage,
        collision_paths + [scene.clean_towel.mesh_path],
        static_friction=float(params["static_friction"]),
        dynamic_friction=float(params["dynamic_friction"]),
    )
    try:
        scene.sim.forward()
    except Exception:
        pass
    return {
        "applied_attrs": applied_attrs,
        "applied_attribute_count": int(len(applied_attrs)),
        "collision_paths_for_material": collision_paths,
        "material_result": material_result,
    }


def best_pure_params() -> dict[str, Any]:
    pure = load_json_if_exists(EDGE_PINCH_DRAG_SUMMARY_JSON)
    if pure and pure.get("verified_best_result") and pure["verified_best_result"].get("params"):
        return pure["verified_best_result"]["params"]
    if pure and pure.get("best_result") and pure["best_result"].get("params"):
        return pure["best_result"]["params"]
    return {
        "trial_id": 0,
        "approach_height_delta": 0.0,
        "jaw_depth_delta": 0.06,
        "close_duration": 24,
        "drag_distance": 0.32,
        "drag_height_delta": -0.04,
        "open_value": 0.65,
        "close_value": 0.02,
    }


def build_param_grid(max_trials: int) -> list[dict[str, Any]]:
    product = list(
        itertools.product(
            [None, 0.006, 0.015],
            [None, 0.0, 0.004],
            [None, 0.006],
            [None, 0.006, 0.015],
            [(0.8, 0.6), (1.8, 1.4), (3.0, 2.4)],
        )
    )
    if len(product) > int(max_trials):
        indices = np.linspace(0, len(product) - 1, int(max_trials), dtype=np.int32)
        product = [product[int(idx)] for idx in indices]
    out = []
    for idx, (cloth_contact, rest, radius, jaw_contact, friction_pair) in enumerate(product):
        out.append(
            {
                "param_trial_id": int(idx),
                "cloth_contact_offset": None if cloth_contact is None else float(cloth_contact),
                "rest_offset": None if rest is None else float(rest),
                "particle_radius": None if radius is None else float(radius),
                "jaw_contact_offset": None if jaw_contact is None else float(jaw_contact),
                "static_friction": float(friction_pair[0]),
                "dynamic_friction": float(friction_pair[1]),
                "soft_fingertip_proxy_radius": None,
                "soft_fingertip_proxy_note": "not used; this sweep only uses accessible USD/PhysX material and offset attributes",
            }
        )
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep accessible contact parameters for SO-101 clean towel contact.")
    parser.add_argument("--task", type=str, default=TASK_ID)
    parser.add_argument("--usd_path", type=Path, default=CLEAN_TOWEL_USD_PATH)
    parser.add_argument("--out_dir", type=Path, default=V7_DIR)
    parser.add_argument("--max_param_trials", type=int, default=36)
    parser.add_argument("--approach_steps", type=int, default=14)
    parser.add_argument("--drag_steps", type=int, default=26)
    parser.add_argument("--edge_band", type=float, default=0.035)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    force_headless_no_cameras(args)
    return args


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    _ = simulation_app
    try:
        skip, reason = should_skip_parameter_sweep()
        if skip:
            summary = {
                "status": "SO101_CONTACT_PARAMETER_SWEEP_V7_SKIPPED",
                "command_run": command_run_string(),
                "reason": reason,
                "parameter_sweep_moves_towel": False,
                "parameter_sweep_width_reduction_success": False,
                "best_contact_params": None,
                "honest_note": "Parameter sweep skipped because pure contact already satisfied the motion criterion.",
            }
            write_json(args.out_dir / CONTACT_PARAMETER_SUMMARY_JSON.name, summary, print_payload=True)
            write_json(args.out_dir / CONTACT_PARAMETER_RESULTS_JSON.name, {"results": []}, print_payload=False)
            return

        candidate = v6_recommended_candidate()
        if not candidate or not candidate.get("action_vector"):
            raise RuntimeError("Missing V6 recommended candidate.")
        base_params = best_pure_params()
        values = mapping_values(str(candidate["arm"]))
        if values["mapping_known"]:
            base_params["open_value"] = float(values["open_value"])
            base_params["close_value"] = float(values["close_value"])

        scene = create_standalone_so101_clean_towel_scene(
            args,
            clean_usd_path=args.usd_path,
            towel_translation=recommended_towel_translation(),
        )
        scene.step_zero(20)
        start_points = scene.clean_towel.points_numpy().astype(np.float32)
        results = []
        for params in build_param_grid(args.max_param_trials):
            applied = apply_contact_params(scene, params)
            trial_params = dict(base_params)
            trial_params["trial_id"] = int(params["param_trial_id"])
            result = run_trial(
                scene,
                start_points,
                candidate,
                trial_params,
                approach_steps=args.approach_steps,
                drag_steps=args.drag_steps,
                edge_band=args.edge_band,
                capture_trace=False,
            )
            result["contact_params"] = params
            result["contact_param_application"] = applied
            result.update(contact_result_sanity(result))
            results.append(result)

        valid_results = [item for item in results if item.get("sanity_valid")]
        touch_valid_results = [item for item in valid_results if item.get("actual_touch_confirmed")]
        invalid_results = [item for item in results if not item.get("sanity_valid")]

        best = max(valid_results, key=result_score) if valid_results else None
        move_best = (
            max(valid_results, key=lambda item: float(item["selected_edge_particle_displacement"]))
            if valid_results
            else None
        )
        width_best = min(valid_results, key=lambda item: float(item["best_width"])) if valid_results else None
        touch_best = max(touch_valid_results, key=result_score) if touch_valid_results else None
        moved = bool(any(float(item["selected_edge_particle_displacement"]) > 0.02 for item in valid_results))
        width_success = bool(any(float(item["best_width"]) < 0.65 for item in valid_results))
        touch_moved = bool(any(float(item["selected_edge_particle_displacement"]) > 0.02 for item in touch_valid_results))
        touch_width_success = bool(any(float(item["best_width"]) < 0.65 for item in touch_valid_results))
        actual_touch_any = bool(touch_valid_results)
        contact_evidence_success = bool(actual_touch_any and touch_moved and touch_width_success)
        results_payload = {
            "status": "SO101_CONTACT_PARAMETER_SWEEP_RESULTS_V7_DONE",
            "command_run": command_run_string(),
            "results": results,
            "invalid_results_excluded_from_success_metrics": invalid_results,
        }
        write_json(args.out_dir / CONTACT_PARAMETER_RESULTS_JSON.name, results_payload, print_payload=False)
        summary = {
            "status": "SO101_CONTACT_PARAMETER_SWEEP_V7_DONE",
            "command_run": command_run_string(),
            "v6_recommended_towel_center": recommended_towel_center(),
            "towel_translation_used": [float(x) for x in recommended_towel_translation()],
            "v6_recommended_candidate": candidate,
            "base_motion_params": base_params,
            "num_param_trials": int(len(results)),
            "num_sanity_valid_param_trials": int(len(valid_results)),
            "invalid_result_count": int(len(invalid_results)),
            "invalid_results_excluded_from_success_metrics": [
                {
                    "param_trial_id": item.get("contact_params", {}).get("param_trial_id"),
                    "sanity_invalid_reasons": item.get("sanity_invalid_reasons"),
                    "selected_edge_particle_displacement": item.get("selected_edge_particle_displacement"),
                    "best_width": item.get("best_width"),
                    "final_width": item.get("final_width"),
                }
                for item in invalid_results
            ],
            "best_result": best,
            "edge_displacement_best_result": move_best,
            "width_best_result": width_best,
            "touch_confirmed_best_result": touch_best,
            "parameter_sweep_moves_towel": moved,
            "parameter_sweep_width_reduction_success": width_success,
            "parameter_sweep_actual_touch_achieved": actual_touch_any,
            "parameter_sweep_touch_confirmed_moves_towel": touch_moved,
            "parameter_sweep_touch_confirmed_width_reduction_success": touch_width_success,
            "parameter_sweep_contact_evidence_success": contact_evidence_success,
            "best_contact_params": None if best is None else best.get("contact_params"),
            "best_contact_params_basis": "best_sanity_valid_result_not_necessarily_touch_confirmed",
            "best_touch_confirmed_contact_params": None if touch_best is None else touch_best.get("contact_params"),
            "best_parameter_sweep_edge_displacement": None
            if move_best is None
            else float(move_best["selected_edge_particle_displacement"]),
            "best_parameter_sweep_width": None if width_best is None else float(width_best["best_width"]),
            "results_json_path": str(args.out_dir / CONTACT_PARAMETER_RESULTS_JSON.name),
            "scene_info": standalone_robot_scene_info(scene),
            "honest_note": (
                "This sweep applies only runtime-accessible contact offset/friction/material settings. "
                "It excludes non-finite or physically unreasonable cloth explosions from success metrics. "
                "It does not use particle-space attachment or policy training."
            ),
        }
        write_json(args.out_dir / CONTACT_PARAMETER_SUMMARY_JSON.name, summary, print_payload=True)
    except Exception as exc:
        summary = {
            "status": "SO101_CONTACT_PARAMETER_SWEEP_V7_FAILED",
            "command_run": command_run_string(),
            "exception_type": exc.__class__.__name__,
            "exception": repr(exc),
            "parameter_sweep_moves_towel": False,
            "parameter_sweep_width_reduction_success": False,
            "best_contact_params": None,
            "honest_note": "Contact parameter sweep failed; no contact, towel motion, folding, or training claim is made.",
        }
        write_json(args.out_dir / CONTACT_PARAMETER_SUMMARY_JSON.name, summary, print_payload=True)
        write_json(args.out_dir / CONTACT_PARAMETER_RESULTS_JSON.name, {"results": []}, print_payload=False)
    finally:
        import os
        import sys

        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
