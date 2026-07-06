from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from clean_towel_v4_common import (
    apply_action_step,
    folded_positions_np,
    generate_rerun_points,
    phase_id_for_step,
    progress_for_step,
    reset_scene_to_points,
    synthetic_action,
)
from so101_clean_towel_scene_utils_v1 import PYTHON_SH, points_to_numpy, size_metrics, write_json


V5_DIR = Path("/workspace/leisaac/tera_checkpoints/clean_towel_v5_contact")
CONTACT_SWEEP_JSON = V5_DIR / "gripper_contact_sweep_results.json"
CONTACT_SWEEP_NPZ = V5_DIR / "gripper_contact_sweep_results.npz"
CONTACT_REPLAY_SUMMARY = V5_DIR / "robot_contact_fold_best_summary.json"
ATTACHMENT_SUMMARY = V5_DIR / "gripper_attachment_curriculum_summary.json"
STATUS_PATH = V5_DIR / "STATUS.json"

V5_COMPILE_COMMAND = (
    f"{PYTHON_SH} -m py_compile "
    "scripts/tera/clean_towel_v5_contact_common.py "
    "scripts/tera/test_clean_towel_gripper_contact_sweep_v5.py "
    "scripts/tera/run_clean_towel_robot_contact_fold_best_v5.py "
    "scripts/tera/run_clean_towel_gripper_attachment_fold_v5.py"
)


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def towel_edge_masks(points: np.ndarray, band: float = 0.035) -> dict[str, np.ndarray]:
    pts = points_to_numpy(points)
    cmin = pts.min(axis=0)
    cmax = pts.max(axis=0)
    return {
        "left": pts[:, 0] <= cmin[0] + band,
        "right": pts[:, 0] >= cmax[0] - band,
        "front": pts[:, 1] <= cmin[1] + band,
        "back": pts[:, 1] >= cmax[1] - band,
    }


def edge_summary(points: np.ndarray, band: float = 0.035) -> dict[str, Any]:
    pts = points_to_numpy(points)
    masks = towel_edge_masks(pts, band)
    out = {}
    for name, mask in masks.items():
        edge = pts[mask]
        out[name] = {
            "count": int(edge.shape[0]),
            "center": [float(x) for x in edge.mean(axis=0)],
            "min": [float(x) for x in edge.min(axis=0)],
            "max": [float(x) for x in edge.max(axis=0)],
        }
    return out


def jaw_positions(scene: Any) -> tuple[np.ndarray, dict[str, Any]]:
    def one(arm: Any) -> tuple[np.ndarray, str]:
        try:
            names = list(getattr(arm.data, "body_names", []) or [])
            lower = [x.lower() for x in names]
            candidates = [i for i, name in enumerate(lower) if "jaw" in name or "gripper" in name or "tool" in name]
            idx = candidates[-1] if candidates else max(0, len(names) - 1)
            return arm.data.body_pos_w[0, idx].detach().cpu().numpy().astype(np.float32), names[idx]
        except Exception:
            return np.zeros(3, dtype=np.float32), "unavailable"

    left, left_name = one(scene.left_arm)
    right, right_name = one(scene.right_arm)
    return np.stack([left, right]).astype(np.float32), {
        "left_body_name": left_name,
        "right_body_name": right_name,
        "jaw_positions_available": bool(left_name != "unavailable" and right_name != "unavailable"),
    }


def nearest_distance_to_edge(jaw: np.ndarray, points: np.ndarray, edge_name: str, band: float = 0.035) -> float:
    pts = points_to_numpy(points)
    mask = towel_edge_masks(pts, band)[edge_name]
    edge = pts[mask]
    if edge.size == 0:
        return float("inf")
    d = np.linalg.norm(edge - np.asarray(jaw, dtype=np.float32).reshape(1, 3), axis=1)
    return float(d.min())


def edge_center_displacement(before: np.ndarray, after: np.ndarray, edge_name: str, band: float = 0.035) -> float:
    before_pts = points_to_numpy(before)
    after_pts = points_to_numpy(after)
    mask = towel_edge_masks(before_pts, band)[edge_name]
    if not np.any(mask):
        return 0.0
    return float(np.linalg.norm(after_pts[mask].mean(axis=0) - before_pts[mask].mean(axis=0)))


def default_action() -> np.ndarray:
    return np.zeros(12, dtype=np.float32)


def make_candidate_action(arm: str, shoulder_pan: float, shoulder_lift: float, elbow: float, wrist: float, gripper: float) -> np.ndarray:
    action = default_action()
    offset = 0 if arm == "left" else 6
    action[offset : offset + 6] = np.asarray(
        [shoulder_pan, shoulder_lift, elbow, wrist, 0.0, gripper],
        dtype=np.float32,
    )
    return action


def drag_delta_for(edge_name: str, arm: str, magnitude: float) -> tuple[float, float, float]:
    sign = 1.0
    if edge_name == "left":
        sign = 1.0 if arm == "left" else -1.0
        return sign * magnitude, 0.0, -0.10 * magnitude
    if edge_name == "right":
        sign = -1.0 if arm == "right" else 1.0
        return sign * magnitude, 0.0, -0.10 * magnitude
    if edge_name == "front":
        return 0.0, 0.25 * magnitude, -0.05 * magnitude
    return 0.0, -0.25 * magnitude, -0.05 * magnitude


def build_contact_candidates(max_candidates: int = 96) -> list[dict[str, Any]]:
    candidates = []
    height_cfgs = [
        {"approach_height": 0.010, "shoulder_lift": -0.72, "elbow": 1.12, "wrist": 0.18},
        {"approach_height": 0.030, "shoulder_lift": -0.52, "elbow": 0.95, "wrist": 0.02},
        {"approach_height": 0.060, "shoulder_lift": -0.30, "elbow": 0.78, "wrist": -0.18},
    ]
    edge_pan = {
        "left": {"left": [0.15, 0.45, 0.75], "right": [-0.75, -0.45]},
        "right": {"left": [0.45, 0.75], "right": [-0.15, -0.45, -0.75]},
        "front": {"left": [0.0, 0.35], "right": [0.0, -0.35]},
        "back": {"left": [0.25, 0.55], "right": [-0.25, -0.55]},
    }
    gripper_pairs = [(0.75, 0.05), (0.50, 0.20)]
    drag_magnitudes = [0.35, 0.65]
    candidate_id = 0
    for edge_name in ["left", "right", "front", "back"]:
        for arm in ["left", "right"]:
            for pan in edge_pan[edge_name][arm]:
                for cfg in height_cfgs:
                    for open_value, close_value in gripper_pairs:
                        for drag_mag in drag_magnitudes:
                            dp, dl, dw = drag_delta_for(edge_name, arm, drag_mag)
                            approach = make_candidate_action(
                                arm,
                                pan,
                                cfg["shoulder_lift"],
                                cfg["elbow"],
                                cfg["wrist"],
                                open_value,
                            )
                            close = make_candidate_action(
                                arm,
                                pan,
                                cfg["shoulder_lift"],
                                cfg["elbow"],
                                cfg["wrist"],
                                close_value,
                            )
                            drag = make_candidate_action(
                                arm,
                                pan + dp,
                                cfg["shoulder_lift"] + dl,
                                cfg["elbow"],
                                cfg["wrist"] + dw,
                                close_value,
                            )
                            candidates.append(
                                {
                                    "candidate_id": candidate_id,
                                    "arm": arm,
                                    "edge": edge_name,
                                    "approach_height_cmd": cfg["approach_height"],
                                    "shoulder_pan": pan,
                                    "shoulder_lift": cfg["shoulder_lift"],
                                    "elbow": cfg["elbow"],
                                    "wrist": cfg["wrist"],
                                    "gripper_open": open_value,
                                    "gripper_close": close_value,
                                    "drag_magnitude": drag_mag,
                                    "action_sequence": [
                                        {"name": "approach_open", "repeats": 8, "action": approach.tolist()},
                                        {"name": "close", "repeats": 8, "action": close.tolist()},
                                        {"name": "drag_inward", "repeats": 16, "action": drag.tolist()},
                                    ],
                                }
                            )
                            candidate_id += 1
    return candidates[: int(max_candidates)]


def run_action_sequence(scene: Any, action_sequence: list[dict[str, Any]]) -> dict[str, Any]:
    widths = []
    points_trace = []
    jaw_trace = []
    for segment in action_sequence:
        action = np.asarray(segment["action"], dtype=np.float32)
        for _ in range(int(segment["repeats"])):
            apply_action_step(scene, action)
            pts = scene.clean_towel.points_numpy().astype(np.float32)
            jaws, _ = jaw_positions(scene)
            widths.append(float(size_metrics(pts)["width_x"]))
            points_trace.append(pts)
            jaw_trace.append(jaws)
    return {
        "widths": np.asarray(widths, dtype=np.float32),
        "points_trace": np.asarray(points_trace, dtype=np.float32),
        "jaw_trace": np.asarray(jaw_trace, dtype=np.float32),
    }


def summarize_contact_trial(
    candidate: dict[str, Any],
    start_points: np.ndarray,
    trace: dict[str, Any],
    active_jaw_index: int,
    edge_band: float = 0.035,
) -> dict[str, Any]:
    final_points = trace["points_trace"][-1]
    widths = trace["widths"]
    start_width = float(size_metrics(start_points)["width_x"])
    best_width = float(widths.min()) if len(widths) else start_width
    final_width = float(widths[-1]) if len(widths) else start_width
    edge_disp = edge_center_displacement(start_points, final_points, candidate["edge"], edge_band)
    edge_mask = towel_edge_masks(start_points, edge_band)[candidate["edge"]]
    edge_points = start_points[edge_mask]
    nearest = float("inf")
    if edge_points.size and len(trace["jaw_trace"]):
        active_jaws = trace["jaw_trace"][:, active_jaw_index, :]
        for jaw in active_jaws:
            nearest = min(nearest, float(np.linalg.norm(edge_points - jaw.reshape(1, 3), axis=1).min()))
    width_reduction = start_width - best_width
    return {
        **candidate,
        "start_width": start_width,
        "best_width": best_width,
        "final_width": final_width,
        "width_reduction": float(width_reduction),
        "selected_edge_particle_displacement": float(edge_disp),
        "nearest_jaw_to_selected_edge_m": float(nearest),
        "contact_moves_towel": bool(edge_disp > 0.03),
        "contact_fold_candidate": bool(width_reduction > 0.05),
    }


def contact_score(result: dict[str, Any]) -> float:
    return (
        10.0 * float(result.get("width_reduction", 0.0))
        + 2.0 * float(result.get("selected_edge_particle_displacement", 0.0))
        - 0.1 * float(result.get("nearest_jaw_to_selected_edge_m", 99.0))
    )


def attachment_mask(points: np.ndarray, band_x: float = 0.18) -> np.ndarray:
    pts = points_to_numpy(points)
    min_x = float(pts[:, 0].min())
    return pts[:, 0] <= min_x + float(band_x)


def run_attachment_episode(
    scene: Any,
    start_points: np.ndarray,
    attachment_strength: float,
    steps: int = 100,
    fold_steps: int = 80,
    band_x: float = 0.18,
) -> dict[str, Any]:
    strength = float(np.clip(attachment_strength, 0.0, 1.0))
    mask = attachment_mask(start_points, band_x)
    points_trace = [start_points.copy()]
    widths = [float(size_metrics(start_points)["width_x"])]
    actions = []
    for step in range(1, int(steps) + 1):
        progress = progress_for_step(step, fold_steps)
        target = folded_positions_np(start_points, progress)
        current = scene.clean_towel.points_numpy().astype(np.float32)
        if strength > 0.0:
            current[mask] = (1.0 - strength) * current[mask] + strength * target[mask]
            scene.clean_towel.set_points(current, device=scene.device)
        phase_id = phase_id_for_step(step, fold_steps)
        action = synthetic_action(progress, phase_id)
        actions.append(action)
        apply_action_step(scene, action)
        after = scene.clean_towel.points_numpy().astype(np.float32)
        points_trace.append(after.copy())
        widths.append(float(size_metrics(after)["width_x"]))
    widths_arr = np.asarray(widths, dtype=np.float32)
    best_step = int(np.argmin(widths_arr))
    return {
        "points_trace": np.asarray(points_trace, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "widths": widths_arr,
        "start_width": float(widths_arr[0]),
        "best_width": float(widths_arr[best_step]),
        "best_step": best_step,
        "final_width": float(widths_arr[-1]),
        "attached_particle_count": int(mask.sum()),
        "attached_particle_fraction": float(mask.sum() / len(mask)),
        "attachment_band_x": float(band_x),
        "attachment_strength": strength,
    }


def write_v5_status(commands_run: list[str]) -> dict[str, Any]:
    sweep = load_json_if_exists(CONTACT_SWEEP_JSON)
    replay = load_json_if_exists(CONTACT_REPLAY_SUMMARY)
    attach = load_json_if_exists(ATTACHMENT_SUMMARY)
    pure_moved = bool(sweep and sweep.get("contact_moves_towel"))
    pure_fold = bool(sweep and sweep.get("contact_fold_candidate"))
    replay_success = bool(replay and replay.get("robot_contact_width_reduction_success"))
    best_pure_width = None
    if replay and "best_width" in replay:
        best_pure_width = float(replay["best_width"])
    elif sweep and sweep.get("best_candidate"):
        best_pure_width = float(sweep["best_candidate"]["best_width"])
    min_attach = None if not attach else attach.get("minimum_attachment_strength_for_best_width_under_0_50")
    attachment_required = bool(not replay_success and min_attach is not None)
    status = {
        "status": "CLEAN_TOWEL_V5_CONTACT_DONE" if sweep and attach else "CLEAN_TOWEL_V5_CONTACT_INCOMPLETE",
        "pure_robot_contact_moved_towel": pure_moved,
        "pure_robot_contact_fold_candidate": pure_fold,
        "pure_robot_contact_replay_success": replay_success,
        "best_pure_contact_width": best_pure_width,
        "gripper_attachment_required": attachment_required,
        "minimum_useful_attachment_strength": min_attach,
        "ready_for_v5_policy_training": bool(replay_success or min_attach is not None),
        "has_true_robot_contact_folding_evidence": bool(replay_success),
        "has_gripper_attachment_assisted_model": bool(min_attach is not None),
        "commands_run": commands_run,
        "paths": {
            "contact_sweep_json": str(CONTACT_SWEEP_JSON),
            "contact_sweep_npz": str(CONTACT_SWEEP_NPZ),
            "contact_replay_summary": str(CONTACT_REPLAY_SUMMARY),
            "attachment_summary": str(ATTACHMENT_SUMMARY),
            "status_json": str(STATUS_PATH),
        },
        "honest_note": (
            "Pure robot-contact success is only claimed from the no-attachment replay. "
            "The gripper attachment curriculum directly constrains a local towel edge band and is assisted manipulation."
        ),
    }
    write_json(STATUS_PATH, status, print_payload=False)
    return status


def command_for_script(script: str) -> str:
    return " ".join([PYTHON_SH, script])
