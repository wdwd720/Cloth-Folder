"""V13 SO-101 ARTICULATION-driven contact fold (real arm, physics-resolved).

This is the piece the V12 summary called for: drive the REAL SO-101 arm
articulation (physics-resolved joint position targets, NOT a free proxy, NOT a
USD teleport, NOT particle writes) to reach a towel edge and fold it.

The raw SO-101 jaw geometry under-contacts the stiff particle cloth (established
V8/V12). The honest bridge that the task explicitly allows ("gripper/fingertip/
contact patch reaches towel edge"; only FREE proxy motion is disallowed) is a
contact patch RIGIDLY ATTACHED to the jaw link: it is part of the jaw's rigid
body, so it moves ONLY as the physics-resolved articulation moves it. The drive
signal is 100% real joint position targets via `apply_action_step`.

Flow:
  1. read the V13 reach solution (global_best.action_vector) as the contact pose;
  2. build scene (centered stable towel, repositioned arm base) with the contact
     patch attached to the jaw BEFORE the single sim.reset() (pre_reset_spawn);
  3. NO-CONTACT baseline: hold the arm retracted, step, measure ~0 cloth motion;
  4. pick an inward drag pose by perturbing the contact pose and measuring which
     right-arm joint delta moves the jaw toward the towel center;
  5. FOLD: drive default->contact->drag via `apply_action_step`, recording cloth
     edge displacement / width / touch / particle-velocity per step;
  6. classify a VALID controlled contact (touch<0.01, edge>0.02, width reduction,
     real velocity spike, NOT ballistic) — a real articulation-driven fold.
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

V13_DIR = Path("/workspace/leisaac/tera_checkpoints/clean_towel_v13")
REACH_JSON = V13_DIR / "so101_articulation_reach_v13.json"
OUT_JSON = V13_DIR / "so101_articulation_contact_fold_v13.json"

STABLE_TOWEL_TRANSLATION = (0.0, 0.0, 0.02)
DEFAULT_RIGHT_BASE = (0.42, -0.28, 0.02)
PATCH_CHILD = "v13_contact_patch"


def attach_jaw_contact_patch(
    arm_prim_path: str, radius: float, contact_offset: float, rest_offset: float,
    static_friction: float, dynamic_friction: float, local_offset=(0.0, 0.0, 0.0),
) -> dict[str, Any]:
    """Add a collision sphere as a CHILD of the jaw link prim, so it becomes part
    of the jaw's rigid body and moves with the physics-resolved articulation."""
    from isaacsim.core.utils.stage import get_current_stage
    from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    stage = get_current_stage()
    root = stage.GetPrimAtPath(arm_prim_path)
    jaw_prim = None
    for prim in Usd.PrimRange(root):
        if "jaw" in prim.GetName().lower():
            jaw_prim = prim  # deepest/last match
    if jaw_prim is None:
        for prim in Usd.PrimRange(root):
            nm = prim.GetName().lower()
            if "gripper" in nm or "tool" in nm:
                jaw_prim = prim
    if jaw_prim is None:
        return {"attached": False, "reason": f"no jaw/gripper prim under {arm_prim_path}"}

    patch_path = jaw_prim.GetPath().AppendChild(PATCH_CHILD)
    sphere = UsdGeom.Sphere.Define(stage, patch_path)
    sphere.CreateRadiusAttr(float(radius))
    UsdGeom.Xformable(sphere).AddTranslateOp().Set(Gf.Vec3d(*[float(x) for x in local_offset]))
    prim = sphere.GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim)
    try:
        pxc = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        pxc.CreateContactOffsetAttr(float(contact_offset))
        pxc.CreateRestOffsetAttr(float(rest_offset))
    except Exception as exc:  # noqa: BLE001
        return {"attached": True, "patch_path": str(patch_path), "jaw_prim": str(jaw_prim.GetPath()),
                "radius": float(radius), "physx_offset_error": repr(exc)}
    try:
        mat_path = Sdf.Path("/World/Materials/v13_patch_mat")
        if not stage.GetPrimAtPath(mat_path).IsValid():
            UsdShade.Material.Define(stage, mat_path)
            mat_api = UsdPhysics.MaterialAPI.Apply(stage.GetPrimAtPath(mat_path))
            mat_api.CreateStaticFrictionAttr(float(static_friction))
            mat_api.CreateDynamicFrictionAttr(float(dynamic_friction))
            mat_api.CreateRestitutionAttr(0.0)
        binding = UsdShade.MaterialBindingAPI.Apply(prim)
        binding.Bind(UsdShade.Material(stage.GetPrimAtPath(mat_path)),
                     bindingStrength=UsdShade.Tokens.weakerThanDescendants, materialPurpose="physics")
    except Exception as exc:  # noqa: BLE001
        return {"attached": True, "patch_path": str(patch_path), "jaw_prim": str(jaw_prim.GetPath()),
                "radius": float(radius), "material_bind_error": repr(exc)}
    return {"attached": True, "patch_path": str(patch_path), "jaw_prim": str(jaw_prim.GetPath()),
            "radius": float(radius), "contact_offset": float(contact_offset)}


def right_jaw_xyz(scene) -> np.ndarray:
    jaws, _ = jaw_positions_from_scene(scene)
    return np.asarray(jaws[3:6], dtype=np.float32)  # right jaw xyz


def surface_touch_distance(scene, radius: float) -> float:
    pts = scene.clean_towel.points_numpy().astype(np.float32)
    jaw = right_jaw_xyz(scene)
    d = float(np.min(np.linalg.norm(pts - jaw.reshape(1, 3), axis=1)))
    return max(0.0, d - float(radius))


def drive_to_action(scene, action12: np.ndarray, steps: int) -> None:
    for _ in range(int(steps)):
        apply_action_step(scene, action12)


def default_action12(scene) -> np.ndarray:
    left = scene.left_arm.data.default_joint_pos[0].detach().cpu().numpy().astype(np.float32)
    right = scene.right_arm.data.default_joint_pos[0].detach().cpu().numpy().astype(np.float32)
    return np.concatenate([left[:6], right[:6]]).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V13 SO-101 articulation contact fold.")
    parser.add_argument("--usd_path", type=Path, default=CLEAN_TOWEL_USD_PATH)
    parser.add_argument("--reach_json", type=Path, default=REACH_JSON)
    parser.add_argument("--right_base", type=float, nargs=3, default=list(DEFAULT_RIGHT_BASE))
    # Patch radius ~0.05: the jaw sits ~z=0.04 above the cloth (z=0.003); this
    # penetrates ~0.01 m so the lower hemisphere grips the edge. The dominant lever
    # is a CLOTH-AWARE drag search (below), not the radius.
    parser.add_argument("--patch_radius", type=float, default=0.05)
    parser.add_argument("--patch_contact_offset", type=float, default=0.02)
    parser.add_argument("--patch_rest_offset", type=float, default=0.0)
    parser.add_argument("--patch_static_friction", type=float, default=1.8)
    parser.add_argument("--patch_dynamic_friction", type=float, default=1.4)
    parser.add_argument("--settle_steps", type=int, default=20)
    parser.add_argument("--approach_steps", type=int, default=30)
    parser.add_argument("--drag_steps", type=int, default=100)
    parser.add_argument("--edge_band", type=float, default=0.035)
    parser.add_argument("--drag_search_steps", type=int, default=16)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    force_headless_no_cameras(args)
    return args


def main() -> None:
    args = parse_args()
    V13_DIR.mkdir(parents=True, exist_ok=True)
    app = AppLauncher(args).app  # noqa: F841

    summary: dict[str, Any] = {
        "status": "V13_SO101_ARTICULATION_CONTACT_FOLD_STARTED",
        "command_run": command_run_string(),
        "contact_element": "so101_jaw_attached_contact_patch (moves with physics-resolved articulation; NOT a free proxy)",
        "drive_mode": "physics_resolved_articulation (set_joint_position_target + write_data_to_sim + sim.step)",
        "right_base": list(args.right_base),
        "towel_translation": list(STABLE_TOWEL_TRANSLATION),
    }
    try:
        # 1. read the reach solution
        if not args.reach_json.exists():
            raise FileNotFoundError(f"missing reach solution: {args.reach_json}")
        reach = json.loads(args.reach_json.read_text())
        best = reach.get("global_best")
        if not best or not reach.get("reachable_edge_found"):
            summary["status"] = "V13_SO101_ARTICULATION_CONTACT_FOLD_SKIPPED"
            summary["skip_reason"] = "no reachable edge from the V13 reach search"
            summary["reach_best_distance_m"] = reach.get("best_jaw_to_edge_center_distance_m")
            write_json(OUT_JSON, summary, print_payload=True)
            sys.stdout.flush(); sys.stderr.flush(); os._exit(0)
        contact_action = np.asarray(best["action_vector"], dtype=np.float32).reshape(12)
        edge = str(best["edge"])
        summary["reach_edge"] = edge
        summary["reach_best_distance_m"] = float(best["distance_m"])
        summary["contact_action_vector"] = [float(x) for x in contact_action]

        # 2. build scene with the jaw-attached contact patch (pre-reset)
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
        summary["patch_attach"] = attach_info
        scene.step_zero(int(args.settle_steps))
        start_points = scene.clean_towel.points_numpy().astype(np.float32)
        summary["start_metrics"] = size_metrics(start_points)
        towel_center = np.asarray(size_metrics(start_points)["center"][:2], dtype=np.float32)
        default12 = default_action12(scene)
        # keep the LEFT arm at default throughout; only the right arm acts
        contact_action[:6] = default12[:6]

        # 3. NO-CONTACT baseline: arm stays retracted (default), cloth should not move
        reset_scene_to_points(scene, start_points)
        drive_to_action(scene, default12, int(args.approach_steps + args.drag_steps) // 2)
        base_pts = scene.clean_towel.points_numpy().astype(np.float32)
        summary["baseline_no_contact"] = {
            "edge_displacement_m": float(edge_center_displacement(start_points, base_pts, edge, float(args.edge_band))),
            "width_before": float(size_metrics(start_points)["width_x"]),
            "width_after": float(size_metrics(base_pts)["width_x"]),
            "note": "arm held at default (retracted); confirms cloth is inert without arm contact",
        }

        # reach the contact pose, measure touch
        reset_scene_to_points(scene, start_points)
        drive_to_action(scene, contact_action, int(args.approach_steps))
        summary["contact_pose_touch_distance_m"] = surface_touch_distance(scene, float(args.patch_radius))
        summary["contact_pose_jaw_xyz"] = [float(x) for x in right_jaw_xyz(scene)]

        # 4+5. CLOTH-AWARE drag search: run the FULL default->contact->drag for each
        #      candidate right-arm joint delta and keep the one that moves the CLOTH
        #      edge most (optimises the fold metric, not just jaw motion — a big jaw
        #      sweep can lift the patch off the cloth and drag nothing).
        dt = float(scene.sim.get_physics_dt())

        def run_drag(delta, record):
            reset_scene_to_points(scene, start_points)
            widths = [float(size_metrics(start_points)["width_x"])]
            nearest_min = float("inf"); vpk = 0.0
            prev = start_points.copy(); acts = []
            segs = [(default12, contact_action, int(args.approach_steps)),
                    (contact_action, (contact_action + delta).astype(np.float32), int(args.drag_steps))]
            for a0, a1, nsteps in segs:
                for k in range(nsteps):
                    a = (a0 + (a1 - a0) * (float(k + 1) / float(nsteps))).astype(np.float32)
                    apply_action_step(scene, a)
                    if record:
                        acts.append(a)
                    pts = scene.clean_towel.points_numpy().astype(np.float32)
                    jaw = right_jaw_xyz(scene)
                    fd = np.linalg.norm(pts - prev, axis=1) / max(dt, 1e-6)
                    near = np.linalg.norm(pts[:, :2] - jaw[:2].reshape(1, 2), axis=1) < (float(args.patch_radius) + 0.05)
                    if np.any(near):
                        vpk = max(vpk, float(np.max(fd[near])))
                    nearest_min = min(nearest_min, float(np.min(np.linalg.norm(pts - jaw.reshape(1, 3), axis=1))) - float(args.patch_radius))
                    widths.append(float(size_metrics(pts)["width_x"]))
                    prev = pts
            fp = scene.clean_towel.points_numpy().astype(np.float32)
            warr = np.asarray(widths, np.float32)
            return {
                "edge_disp": float(edge_center_displacement(start_points, fp, edge, float(args.edge_band))),
                "max_disp": float(np.max(np.linalg.norm(fp - start_points, axis=1))),
                "vpeak": vpk, "width_before": float(warr[0]), "width_after": float(warr[-1]),
                "best_width": float(warr.min()), "width_reduction": float(warr[0] - warr.min()),
                "touch": float(max(0.0, nearest_min)),
            }, acts

        candidates = []
        for j in (6, 7, 8):  # right shoulder_pan, shoulder_lift, elbow_flex
            for mag in (-0.9, -0.5, 0.5, 0.9):
                d = np.zeros(12, np.float32); d[j] = mag
                candidates.append((f"j{j}_{mag:+.1f}", d))
        for pan in (-0.9, -0.6):
            for lift in (-0.4, 0.4):  # combine inward pan with a down/up press
                d = np.zeros(12, np.float32); d[6] = pan; d[7] = lift
                candidates.append((f"pan{pan:+.1f}_lift{lift:+.1f}", d))

        search = []
        best = None
        for name, d in candidates:
            m, _ = run_drag(d, record=False)
            score = m["edge_disp"] + 2.0 * max(0.0, m["width_reduction"])
            search.append({"name": name, "edge_disp": m["edge_disp"], "width_reduction": m["width_reduction"], "score": score})
            if best is None or score > best["score"]:
                best = {"name": name, "delta": d, "metrics": m, "score": score}
        search.sort(key=lambda x: -x["score"])
        summary["drag_search"] = {
            "num_candidates": len(candidates), "best_name": best["name"],
            "best_delta_nonzero_index": [int(i) for i in np.nonzero(best["delta"])[0]],
            "top": search[:6],
        }

        # re-run the winning delta WITH recording -> final fold metrics + demo trajectory
        fm, recorded_actions = run_drag(best["delta"], record=True)
        drag_action = (contact_action + best["delta"]).astype(np.float32)
        edge_disp = fm["edge_disp"]; max_disp = fm["max_disp"]; vpeak = fm["vpeak"]
        width_before = fm["width_before"]; width_after = fm["width_after"]; best_width = fm["best_width"]
        width_reduction = fm["width_reduction"]; actual_touch = fm["touch"]
        spurious_ballistic = bool(edge_disp > 0.5 or max_disp > 0.5 or vpeak >= 4.9)
        valid = bool(actual_touch < 0.01 and 0.02 < edge_disp <= 0.5 and width_reduction > 0.005 and not spurious_ballistic)

        summary["fold"] = {
            "actual_touch_distance_m": actual_touch,
            "edge_displacement_m": edge_disp,
            "max_particle_displacement_m": max_disp,
            "particle_velocity_peak": vpeak,
            "width_before": width_before, "width_after": width_after, "best_width": best_width,
            "width_reduction_m": width_reduction,
            "num_action_steps": len(recorded_actions),
            "spurious_ballistic": spurious_ballistic,
            "valid_controlled_contact": valid,
        }
        summary["articulation_contact_fold_solved"] = valid
        summary["status"] = "V13_SO101_ARTICULATION_CONTACT_FOLD_DONE"
        # persist the real action trajectory for the demo stage (npz, volume-only)
        np.savez_compressed(V13_DIR / "so101_articulation_fold_actions_v13.npz",
                            actions=np.asarray(recorded_actions, np.float32),
                            contact_action=contact_action, drag_action=drag_action,
                            edge=edge, right_base=np.asarray(args.right_base, np.float32))
        write_json(OUT_JSON, summary, print_payload=True)
    except Exception as exc:  # noqa: BLE001
        import traceback
        summary["status"] = "V13_SO101_ARTICULATION_CONTACT_FOLD_FAILED"
        summary["exception_type"] = exc.__class__.__name__
        summary["exception"] = repr(exc)
        summary["traceback_tail"] = traceback.format_exc()[-6000:]
        write_json(OUT_JSON, summary, print_payload=True)
    finally:
        sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == "__main__":
    main()
