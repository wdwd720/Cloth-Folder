from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from so101_clean_towel_scene_utils_v1 import (
    CLEAN_TOWEL_GRID_PATH,
    CLEAN_TOWEL_USD_PATH,
    PYTHON_SH,
    TASK_ID,
    command_run_string,
    points_to_numpy,
    size_metrics,
    write_json,
)


V1_DIR = Path("/workspace/leisaac/tera_checkpoints/so101_clean_towel_v1")
V4_DIR = Path("/workspace/leisaac/tera_checkpoints/clean_towel_v4")
V1_ASSISTED_CAPTURE = V1_DIR / "assisted_fold_capture_v1.npz"
V1_ASSISTED_TRACE = V1_DIR / "assisted_fold_metrics_trace_v1.json"
V1_ASSISTED_SUMMARY = V1_DIR / "assisted_fold_summary.json"

POLICY_PATH = V4_DIR / "clean_towel_policy_v4.pt"
DATASET_PATH = V4_DIR / "clean_towel_policy_v4_dataset.npz"
TRAIN_SUMMARY_PATH = V4_DIR / "train_clean_towel_policy_v4_summary.json"
NO_ASSIST_SUMMARY_PATH = V4_DIR / "eval_no_assist_summary.json"
WITH_STOP_SUMMARY_PATH = V4_DIR / "eval_with_stop_summary.json"
CURRICULUM_SUMMARY_PATH = V4_DIR / "assist_curriculum_summary.json"
STATUS_PATH = V4_DIR / "STATUS.json"

V4_COMPILE_COMMAND = (
    f"{PYTHON_SH} -m py_compile "
    "scripts/tera/clean_towel_v4_common.py "
    "scripts/tera/train_clean_towel_policy_v4.py "
    "scripts/tera/eval_clean_towel_policy_v4_no_assist.py "
    "scripts/tera/eval_clean_towel_policy_v4_with_stop.py "
    "scripts/tera/run_clean_towel_v4_assist_curriculum.py"
)

PHASE_NAMES = ["approach", "fold", "settle"]
ACTION_NAMES = [
    "left_shoulder_pan",
    "left_shoulder_lift",
    "left_elbow_flex",
    "left_wrist_flex",
    "left_wrist_roll",
    "left_gripper",
    "right_shoulder_pan",
    "right_shoulder_lift",
    "right_elbow_flex",
    "right_wrist_flex",
    "right_wrist_roll",
    "right_gripper",
]


class PolicyMLP(nn.Module):
    def __init__(self, state_dim: int, action_dim: int = 12, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def phase_id_for_step(step: int, fold_steps: int = 80) -> int:
    if step < max(1, int(0.20 * fold_steps)):
        return 0
    if step <= fold_steps:
        return 1
    return 2


def progress_for_step(step: int, fold_steps: int = 80) -> float:
    return float(np.clip(step / float(max(1, fold_steps)), 0.0, 1.0))


def phase_one_hot(phase_id: int) -> np.ndarray:
    phase = np.zeros(len(PHASE_NAMES), dtype=np.float32)
    phase[int(np.clip(phase_id, 0, len(PHASE_NAMES) - 1))] = 1.0
    return phase


def corner_and_edge_features(points: np.ndarray) -> tuple[np.ndarray, list[str]]:
    pts = points_to_numpy(points)
    cmin = pts.min(axis=0)
    cmax = pts.max(axis=0)
    corner_targets = [
        [cmin[0], cmin[1], cmin[2]],
        [cmin[0], cmax[1], cmin[2]],
        [cmax[0], cmin[1], cmin[2]],
        [cmax[0], cmax[1], cmin[2]],
    ]
    corner_names = ["minx_miny", "minx_maxy", "maxx_miny", "maxx_maxy"]
    corners = []
    for target in corner_targets:
        target_arr = np.asarray(target, dtype=np.float32)
        idx = np.argmin(np.sum((pts[:, :2] - target_arr[:2]) ** 2, axis=1))
        corners.append(pts[idx])

    x = pts[:, 0]
    y = pts[:, 1]
    eps_x = max(1.0e-5, 0.05 * float(cmax[0] - cmin[0]))
    eps_y = max(1.0e-5, 0.05 * float(cmax[1] - cmin[1]))
    masks = [
        x <= cmin[0] + eps_x,
        x >= cmax[0] - eps_x,
        y <= cmin[1] + eps_y,
        y >= cmax[1] - eps_y,
    ]
    edge_names = ["left_edge", "right_edge", "bottom_edge", "top_edge"]
    edges = []
    for mask in masks:
        edges.append(pts[mask].mean(axis=0) if np.any(mask) else np.zeros(3, dtype=np.float32))

    values = np.asarray(corners + edges, dtype=np.float32).reshape(-1)
    names = []
    for name in corner_names + edge_names:
        names.extend([f"{name}_x", f"{name}_y", f"{name}_z"])
    return values, names


def compact_state(
    points: np.ndarray,
    progress: float,
    phase_id: int,
    jaw_positions: np.ndarray | None = None,
) -> tuple[np.ndarray, list[str]]:
    metrics = size_metrics(points)
    if jaw_positions is None:
        jaw_positions = np.zeros(6, dtype=np.float32)
    jaw_positions = np.asarray(jaw_positions, dtype=np.float32).reshape(6)
    corner_edge, corner_edge_names = corner_and_edge_features(points)
    state_parts = [
        np.asarray([progress], dtype=np.float32),
        phase_one_hot(phase_id),
        np.asarray(metrics["min"] + metrics["max"] + metrics["center"] + metrics["size"], dtype=np.float32),
        jaw_positions,
        corner_edge,
    ]
    names = (
        ["progress"]
        + [f"phase_{name}" for name in PHASE_NAMES]
        + ["min_x", "min_y", "min_z", "max_x", "max_y", "max_z", "center_x", "center_y", "center_z", "size_x", "size_y", "size_z"]
        + ["left_jaw_x", "left_jaw_y", "left_jaw_z", "right_jaw_x", "right_jaw_y", "right_jaw_z"]
        + corner_edge_names
    )
    return np.concatenate(state_parts).astype(np.float32), names


def planned_jaw_positions(points: np.ndarray, progress: float) -> np.ndarray:
    metrics = size_metrics(points)
    cmin = np.asarray(metrics["min"], dtype=np.float32)
    cmax = np.asarray(metrics["max"], dtype=np.float32)
    center = np.asarray(metrics["center"], dtype=np.float32)
    width = max(float(metrics["width_x"]), 1.0e-6)
    lift = 0.05 * math.sin(math.pi * min(max(progress, 0.0), 1.0))
    left = np.asarray([cmin[0] + width * (0.18 + 0.62 * progress), cmin[1] - 0.08, center[2] + 0.045 + lift], dtype=np.float32)
    right = np.asarray([cmax[0] - width * 0.12, cmin[1] - 0.08, center[2] + 0.04], dtype=np.float32)
    return np.concatenate([left, right]).astype(np.float32)


def jaw_positions_from_scene(scene: Any) -> tuple[np.ndarray, dict[str, Any]]:
    def _one_arm(arm: Any) -> tuple[np.ndarray, str]:
        try:
            names = list(getattr(arm.data, "body_names", []) or [])
            lower = [name.lower() for name in names]
            candidates = [
                i
                for i, name in enumerate(lower)
                if "jaw" in name or "gripper" in name or "end" in name or "tool" in name
            ]
            idx = candidates[-1] if candidates else max(0, len(names) - 1)
            pos = arm.data.body_pos_w[0, idx].detach().cpu().numpy().astype(np.float32)
            return pos, names[idx] if names else "unknown"
        except Exception:
            return np.zeros(3, dtype=np.float32), "unavailable"

    left, left_name = _one_arm(scene.left_arm)
    right, right_name = _one_arm(scene.right_arm)
    values = np.concatenate([left, right]).astype(np.float32)
    return values, {
        "jaw_positions_available": bool(left_name != "unavailable" and right_name != "unavailable"),
        "left_body_name": left_name,
        "right_body_name": right_name,
    }


def folded_positions_np(
    start_points: np.ndarray,
    alpha: float,
    final_layer_gap: float = 0.012,
    arc_lift: float = 0.08,
) -> np.ndarray:
    pts = np.asarray(start_points, dtype=np.float32).copy()
    alpha = float(np.clip(alpha, 0.0, 1.0))
    center_x = 0.5 * (float(start_points[:, 0].min()) + float(start_points[:, 0].max()))
    left_mask = start_points[:, 0] < center_x
    dx = center_x - start_points[left_mask, 0]
    max_dx = max(float(dx.max()) if dx.size else 1.0, 1.0e-6)
    theta = math.pi * alpha
    pts[left_mask, 0] = center_x - dx * math.cos(theta)
    pts[left_mask, 2] = start_points[left_mask, 2] + float(arc_lift) * math.sin(theta) * (dx / max_dx) + float(final_layer_gap) * alpha
    return pts.astype(np.float32)


def synthetic_action(progress: float, phase_id: int) -> np.ndarray:
    p = float(np.clip(progress, 0.0, 1.0))
    smooth = 0.5 - 0.5 * math.cos(math.pi * p)
    lift = math.sin(math.pi * p)
    settle = 1.0 if phase_id == 2 else 0.0
    left = np.asarray(
        [
            -0.45 + 0.92 * smooth,
            -0.70 + 0.24 * lift - 0.12 * settle,
            1.05 - 0.30 * smooth,
            0.18 + 0.58 * lift - 0.24 * smooth,
            0.20 * math.sin(2.0 * math.pi * p),
            0.36 - 0.16 * settle,
        ],
        dtype=np.float32,
    )
    right = np.asarray(
        [
            0.28 - 0.18 * smooth,
            -0.62 + 0.12 * lift - 0.08 * settle,
            0.92 - 0.08 * smooth,
            0.08 + 0.18 * lift,
            -0.08 * math.sin(math.pi * p),
            0.36 - 0.16 * settle,
        ],
        dtype=np.float32,
    )
    return np.clip(np.concatenate([left, right]), -1.6, 1.6).astype(np.float32)


def load_start_points_for_training() -> tuple[np.ndarray, str]:
    if V1_ASSISTED_CAPTURE.exists():
        data = np.load(V1_ASSISTED_CAPTURE)
        return data["start_points"].astype(np.float32), str(V1_ASSISTED_CAPTURE)
    if CLEAN_TOWEL_GRID_PATH.exists():
        data = np.load(CLEAN_TOWEL_GRID_PATH)
        return data["points"].astype(np.float32), str(CLEAN_TOWEL_GRID_PATH)
    xs = np.linspace(-0.34, 0.34, 69, dtype=np.float32)
    ys = np.linspace(-0.19, 0.19, 39, dtype=np.float32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    points = np.stack([xx.reshape(-1), yy.reshape(-1), np.zeros(xx.size, dtype=np.float32)], axis=1)
    return points, "generated_rect_grid_fallback"


def build_training_dataset(fold_steps: int = 80, settle_steps: int = 20, augment_repeats: int = 3) -> dict[str, Any]:
    start_points, source = load_start_points_for_training()
    states = []
    actions = []
    progresses = []
    phase_ids = []
    widths = []
    state_feature_names = None
    for repeat in range(max(1, augment_repeats)):
        jitter_scale = 0.0 if repeat == 0 else 0.0015 * repeat
        for step in range(fold_steps + settle_steps + 1):
            progress = progress_for_step(step, fold_steps)
            phase_id = phase_id_for_step(step, fold_steps)
            points = folded_positions_np(start_points, progress)
            if jitter_scale:
                rng = np.random.default_rng(1234 + repeat * 1000 + step)
                points = points + rng.normal(0.0, jitter_scale, size=points.shape).astype(np.float32)
            jaw = planned_jaw_positions(points, progress)
            state, names = compact_state(points, progress, phase_id, jaw)
            state_feature_names = names
            states.append(state)
            actions.append(synthetic_action(progress, phase_id))
            progresses.append(progress)
            phase_ids.append(phase_id)
            widths.append(float(size_metrics(points)["width_x"]))
    return {
        "states": np.asarray(states, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "progress": np.asarray(progresses, dtype=np.float32),
        "phase_ids": np.asarray(phase_ids, dtype=np.int32),
        "widths": np.asarray(widths, dtype=np.float32),
        "feature_names": np.asarray(state_feature_names or [], dtype=object),
        "action_names": np.asarray(ACTION_NAMES, dtype=object),
        "source_points": source,
        "action_label_source": "synthetic_open_loop_so101_joint_targets_fit_to_particle_space_assist_progress",
        "contains_robot_contact_demonstrations": False,
    }


def save_dataset(path: Path, dataset: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **dataset)


def load_policy_bundle(policy_path: Path = POLICY_PATH) -> dict[str, Any]:
    bundle = torch.load(policy_path, map_location="cpu", weights_only=False)
    model = PolicyMLP(int(bundle["state_dim"]), int(bundle["action_dim"]), int(bundle["hidden_dim"]))
    model.load_state_dict(bundle["model_state_dict"])
    model.eval()
    bundle["model"] = model
    return bundle


def policy_action(bundle: dict[str, Any], state: np.ndarray) -> np.ndarray:
    model: PolicyMLP = bundle["model"]
    state_mean = np.asarray(bundle["state_mean"], dtype=np.float32)
    state_std = np.asarray(bundle["state_std"], dtype=np.float32)
    action_mean = np.asarray(bundle["action_mean"], dtype=np.float32)
    action_std = np.asarray(bundle["action_std"], dtype=np.float32)
    x = (np.asarray(state, dtype=np.float32) - state_mean) / state_std
    with torch.no_grad():
        pred = model(torch.from_numpy(x).float().unsqueeze(0)).squeeze(0).cpu().numpy()
    return (pred * action_std + action_mean).astype(np.float32)


def apply_action_step(scene: Any, action: np.ndarray) -> None:
    action = np.asarray(action, dtype=np.float32).reshape(1, 12)
    left = torch.tensor(action[:, :6], dtype=torch.float32, device=scene.device)
    right = torch.tensor(action[:, 6:], dtype=torch.float32, device=scene.device)
    scene.left_arm.set_joint_position_target(left)
    scene.right_arm.set_joint_position_target(right)
    scene.left_arm.write_data_to_sim()
    scene.right_arm.write_data_to_sim()
    scene.sim.step(render=False)
    dt = scene.sim.get_physics_dt()
    scene.left_arm.update(dt)
    scene.right_arm.update(dt)


def reset_scene_to_points(scene: Any, points: np.ndarray) -> None:
    scene.clean_towel.set_points(points, device=scene.device)
    try:
        scene.left_arm.write_joint_state_to_sim(scene.left_arm.data.default_joint_pos, scene.left_arm.data.default_joint_vel)
        scene.right_arm.write_joint_state_to_sim(scene.right_arm.data.default_joint_pos, scene.right_arm.data.default_joint_vel)
    except Exception:
        pass
    scene.sim.forward()
    dt = scene.sim.get_physics_dt()
    scene.left_arm.update(dt)
    scene.right_arm.update(dt)


def run_policy_episode(
    scene: Any,
    policy_bundle: dict[str, Any],
    steps: int = 100,
    assist_ratio: float = 0.0,
    fold_steps: int = 80,
    settle_steps: int = 20,
) -> dict[str, Any]:
    assist_ratio = float(np.clip(assist_ratio, 0.0, 1.0))
    start_points = scene.clean_towel.points_numpy().astype(np.float32)
    points_trace = [start_points.copy()]
    widths = [float(size_metrics(start_points)["width_x"])]
    actions = []
    metrics_by_step = [{"step": 0, "metrics": size_metrics(start_points)}]
    jaw_info = None

    for step in range(1, int(steps) + 1):
        current = scene.clean_towel.points_numpy().astype(np.float32)
        progress = progress_for_step(step - 1, fold_steps)
        phase_id = phase_id_for_step(step - 1, fold_steps)
        jaws, jaw_info = jaw_positions_from_scene(scene)
        state, _ = compact_state(current, progress, phase_id, jaws)
        action = policy_action(policy_bundle, state)
        if assist_ratio > 0.0:
            target = folded_positions_np(start_points, progress_for_step(step, fold_steps))
            assisted = (1.0 - assist_ratio) * current + assist_ratio * target
            scene.clean_towel.set_points(assisted.astype(np.float32), device=scene.device)
        apply_action_step(scene, action)
        after = scene.clean_towel.points_numpy().astype(np.float32)
        metrics = size_metrics(after)
        actions.append(action)
        points_trace.append(after.copy())
        widths.append(float(metrics["width_x"]))
        metrics_by_step.append({"step": step, "metrics": metrics})

    widths_arr = np.asarray(widths, dtype=np.float32)
    best_step = int(np.argmin(widths_arr))
    return {
        "points_trace": np.asarray(points_trace, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "widths": widths_arr,
        "metrics_by_step": metrics_by_step,
        "start_width": float(widths_arr[0]),
        "best_width": float(widths_arr[best_step]),
        "best_step": best_step,
        "final_width": float(widths_arr[-1]),
        "start_metrics": metrics_by_step[0]["metrics"],
        "best_metrics": metrics_by_step[best_step]["metrics"],
        "final_metrics": metrics_by_step[-1]["metrics"],
        "assist_ratio": assist_ratio,
        "jaw_info": jaw_info or {"jaw_positions_available": False},
    }


def generate_rerun_points(rrd_path: Path, label: str, frames: list[tuple[str, np.ndarray]]) -> dict[str, Any]:
    try:
        import rerun as rr

        rr.init(label, spawn=False)
        rr.save(str(rrd_path))
        for name, points in frames:
            pts = np.asarray(points, dtype=np.float32)
            colors = np.tile(np.array([[34, 128, 210]], dtype=np.uint8), (len(pts), 1))
            rr.log(f"world/{name}", rr.Points3D(pts, radii=0.003, colors=colors))
        return {"rerun_status": "rerun_ok", "rrd_path": str(rrd_path), "rerun_error": None}
    except Exception as exc:
        return {"rerun_status": "rerun_failed", "rrd_path": str(rrd_path), "rerun_error": repr(exc)}


def success_flags(best_width: float, final_width: float) -> dict[str, bool]:
    return {
        "no_assist_width_reduction_success": bool(best_width < 0.60),
        "strong_success": bool(best_width < 0.50),
        "final_success": bool(final_width < 0.55),
    }


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_v4_status(commands_run: list[str]) -> dict[str, Any]:
    train = load_json_if_exists(TRAIN_SUMMARY_PATH)
    no_assist = load_json_if_exists(NO_ASSIST_SUMMARY_PATH)
    with_stop = load_json_if_exists(WITH_STOP_SUMMARY_PATH)
    curriculum = load_json_if_exists(CURRICULUM_SUMMARY_PATH)
    train_ok = bool(train and train.get("status") == "CLEAN_TOWEL_POLICY_V4_TRAIN_OK")
    no_assist_ok = bool(no_assist and no_assist.get("status") == "CLEAN_TOWEL_POLICY_V4_NO_ASSIST_EVAL_DONE")
    stop_ok = bool(with_stop and with_stop.get("status") == "CLEAN_TOWEL_POLICY_V4_WITH_STOP_EVAL_DONE")
    curr_ok = bool(curriculum and curriculum.get("status") == "CLEAN_TOWEL_V4_ASSIST_CURRICULUM_DONE")
    pure_success = bool(no_assist and no_assist.get("no_assist_width_reduction_success"))
    strong_success = bool(no_assist and no_assist.get("strong_success"))
    curriculum_results = [] if not curriculum else curriculum.get("results", [])
    assist_success_ratios = [
        float(item["assist_ratio"])
        for item in curriculum_results
        if item.get("best_width", 999.0) < 0.50
    ]
    status = {
        "status": "CLEAN_TOWEL_V4_PIPELINE_DONE" if all([train_ok, no_assist_ok, stop_ok, curr_ok]) else "CLEAN_TOWEL_V4_PIPELINE_INCOMPLETE",
        "train_status": None if train is None else train.get("status"),
        "no_assist_eval_status": None if no_assist is None else no_assist.get("status"),
        "with_stop_eval_status": None if with_stop is None else with_stop.get("status"),
        "assist_curriculum_status": None if curriculum is None else curriculum.get("status"),
        "ready_for_more_training": bool(train_ok and no_assist_ok and curr_ok),
        "has_true_robot_contact_folding_evidence": pure_success,
        "has_strong_true_robot_contact_folding_evidence": strong_success,
        "still_requires_assist_for_strong_fold": bool(not strong_success),
        "minimum_assist_ratio_with_best_width_under_0_50": None if not assist_success_ratios else float(min(assist_success_ratios)),
        "v4_policy_basis": "supervised_policy_fit_to_synthetic_joint_targets_from_particle_space_assisted_fold_progress",
        "contains_robot_contact_demonstrations": False if train is None else bool(train.get("contains_robot_contact_demonstrations", False)),
        "commands_run": commands_run,
        "paths": {
            "policy": str(POLICY_PATH),
            "dataset": str(DATASET_PATH),
            "train_summary": str(TRAIN_SUMMARY_PATH),
            "no_assist_summary": str(NO_ASSIST_SUMMARY_PATH),
            "with_stop_summary": str(WITH_STOP_SUMMARY_PATH),
            "curriculum_summary": str(CURRICULUM_SUMMARY_PATH),
            "status_json": str(STATUS_PATH),
        },
        "honest_note": (
            "Autonomous success is only claimed if no-assist evaluation proves it. "
            "The V4 training labels are synthetic and derived from the particle-space assisted fold, not robot-contact demonstrations."
        ),
    }
    write_json(STATUS_PATH, status, print_payload=False)
    return status


def command_for_script(script_path: str) -> str:
    return " ".join([PYTHON_SH, script_path])
