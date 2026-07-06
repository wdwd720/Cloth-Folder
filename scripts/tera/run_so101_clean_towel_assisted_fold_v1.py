from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher

from so101_clean_towel_scene_utils_v1 import (
    CLEAN_TOWEL_GRID_PATH,
    CLEAN_TOWEL_USD_PATH,
    COMPILE_COMMAND,
    OUT_DIR,
    TASK_ID,
    command_run_string,
    create_standalone_so101_clean_towel_scene,
    force_headless_no_cameras,
    load_json_if_exists,
    size_metrics,
    standalone_robot_scene_info,
    status_payload,
    write_json,
)


DEFAULT_SUMMARY_PATH = OUT_DIR / "assisted_fold_summary.json"
DEFAULT_CAPTURE_PATH = OUT_DIR / "assisted_fold_capture_v1.npz"
DEFAULT_RRD_PATH = OUT_DIR / "assisted_fold_actual_state_v1.rrd"
DEFAULT_OBJ_PATH = OUT_DIR / "assisted_fold_final_surface_v1.obj"
DEFAULT_METRICS_TRACE_PATH = OUT_DIR / "assisted_fold_metrics_trace_v1.json"
DEFAULT_STATUS_PATH = OUT_DIR / "STATUS.json"
SMOKE_SUMMARY_PATH = OUT_DIR / "test_so101_clean_towel_scene_v1_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an assisted fold primitive in the SO-101 + clean towel scene.")
    parser.add_argument("--task", type=str, default=TASK_ID)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--usd_path", type=Path, default=CLEAN_TOWEL_USD_PATH)
    parser.add_argument("--out_dir", type=Path, default=OUT_DIR)
    parser.add_argument("--width_success_threshold", type=float, default=0.50)
    parser.add_argument("--fold_steps", type=int, default=80)
    parser.add_argument("--settle_steps", type=int, default=20)
    parser.add_argument("--final_layer_gap", type=float, default=0.012)
    parser.add_argument("--arc_lift", type=float, default=0.08)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    force_headless_no_cameras(args)
    return args


def folded_positions(start_points, alpha: float, final_layer_gap: float, arc_lift: float):
    import torch

    points = start_points.clone()
    min_x = torch.min(start_points[:, 0])
    max_x = torch.max(start_points[:, 0])
    center_x = (min_x + max_x) * 0.5
    left_mask = start_points[:, 0] < center_x
    dx = center_x - start_points[left_mask, 0]
    max_dx = torch.clamp(torch.max(dx), min=1.0e-6)
    theta = torch.tensor(np.pi * alpha, dtype=start_points.dtype, device=start_points.device)
    points[left_mask, 0] = center_x - dx * torch.cos(theta)
    points[left_mask, 2] = (
        start_points[left_mask, 2]
        + float(arc_lift) * torch.sin(theta) * (dx / max_dx)
        + float(final_layer_gap) * alpha
    )
    return points


def write_obj(path: Path, points: np.ndarray, face_indices: np.ndarray | None) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# SO-101 clean towel assisted fold final surface\n")
        for p in points:
            f.write(f"v {p[0]:.8f} {p[1]:.8f} {p[2]:.8f}\n")
        if face_indices is None:
            return 0
        triangles = face_indices.reshape(-1, 3)
        for tri in triangles:
            f.write(f"f {tri[0] + 1} {tri[1] + 1} {tri[2] + 1}\n")
    return int(len(face_indices) // 3)


def generate_rerun(rrd_path: Path, frames: list[tuple[int, np.ndarray]]) -> dict[str, object]:
    try:
        import rerun as rr

        rr.init("so101_clean_towel_assisted_fold_v1", spawn=False)
        rr.save(str(rrd_path))
        for step, points in frames:
            colors = np.tile(np.array([[31, 119, 180]], dtype=np.uint8), (len(points), 1))
            rr.log(
                f"world/clean_towel_particles/step_{int(step):03d}",
                rr.Points3D(points.astype(np.float32), radii=0.003, colors=colors),
            )
        return {"rerun_status": "rerun_ok", "rrd_path": str(rrd_path), "rerun_error": None}
    except Exception as exc:
        return {"rerun_status": "rerun_failed", "rrd_path": str(rrd_path), "rerun_error": repr(exc)}


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    scene = None

    try:
        scene = create_standalone_so101_clean_towel_scene(
            args,
            clean_usd_path=args.usd_path,
        )
        clean_towel = scene.clean_towel
        scene_info = standalone_robot_scene_info(scene)

        start_tensor = clean_towel.points_tensor()[0].clone()
        start_points = start_tensor.detach().cpu().numpy().astype(np.float32)
        start_metrics = size_metrics(start_points)
        widths = [float(start_metrics["width_x"])]
        metrics_by_step = [{"step": 0, "metrics": start_metrics}]
        best_width = float(start_metrics["width_x"])
        best_step = 0
        best_points = start_points.copy()

        for step in range(1, int(args.fold_steps) + 1):
            alpha = step / float(args.fold_steps)
            target = folded_positions(
                start_tensor,
                alpha=alpha,
                final_layer_gap=float(args.final_layer_gap),
                arc_lift=float(args.arc_lift),
            )
            clean_towel.set_points(target, device=scene.device)
            scene.step_zero(1)
            current_points = clean_towel.points_numpy()
            metrics = size_metrics(current_points)
            widths.append(float(metrics["width_x"]))
            metrics_by_step.append({"step": step, "metrics": metrics})
            if float(metrics["width_x"]) < best_width:
                best_width = float(metrics["width_x"])
                best_step = step
                best_points = current_points.copy()

        for settle_i in range(int(args.settle_steps)):
            scene.step_zero(1)
            current_points = clean_towel.points_numpy()
            metrics = size_metrics(current_points)
            step = int(args.fold_steps) + settle_i + 1
            widths.append(float(metrics["width_x"]))
            metrics_by_step.append({"step": step, "metrics": metrics})
            if float(metrics["width_x"]) < best_width:
                best_width = float(metrics["width_x"])
                best_step = step
                best_points = current_points.copy()

        final_points = clean_towel.points_numpy()
        final_metrics = size_metrics(final_points)
        final_width = float(final_metrics["width_x"])
        fold_success = bool(best_width < float(args.width_success_threshold))

        capture_path = args.out_dir / DEFAULT_CAPTURE_PATH.name
        metrics_trace_path = args.out_dir / DEFAULT_METRICS_TRACE_PATH.name
        np.savez_compressed(
            capture_path,
            start_points=start_points,
            best_points=best_points.astype(np.float32),
            final_points=final_points.astype(np.float32),
            widths=np.asarray(widths, dtype=np.float32),
            best_step=np.asarray([best_step], dtype=np.int32),
            width_success_threshold=np.asarray([args.width_success_threshold], dtype=np.float32),
        )
        write_json(metrics_trace_path, {"metrics_by_step": metrics_by_step}, print_payload=False)

        face_indices = None
        if CLEAN_TOWEL_GRID_PATH.exists():
            grid = np.load(CLEAN_TOWEL_GRID_PATH)
            if "face_indices" in grid and "points" in grid and len(grid["points"]) == len(final_points):
                face_indices = grid["face_indices"].astype(np.int32)
        obj_path = args.out_dir / DEFAULT_OBJ_PATH.name
        num_triangles = write_obj(obj_path, final_points, face_indices)

        rrd_path = args.out_dir / DEFAULT_RRD_PATH.name
        mid_step = max(1, int(args.fold_steps) // 2)
        mid_points = folded_positions(
            start_tensor,
            alpha=mid_step / float(args.fold_steps),
            final_layer_gap=float(args.final_layer_gap),
            arc_lift=float(args.arc_lift),
        ).detach().cpu().numpy().astype(np.float32)
        rerun_info = generate_rerun(
            rrd_path,
            frames=[
                (0, start_points),
                (mid_step, mid_points),
                (best_step, best_points.astype(np.float32)),
                (int(args.fold_steps) + int(args.settle_steps), final_points.astype(np.float32)),
            ],
        )

        summary = {
            "status": "SO101_CLEAN_TOWEL_ASSISTED_FOLD_V1_OK" if fold_success else "SO101_CLEAN_TOWEL_ASSISTED_FOLD_V1_FAILED",
            "command_run": command_run_string(),
            "task": args.task,
            "device": str(args.device),
            "clean_towel_usd_path": str(args.usd_path),
            "clean_towel_root_path": clean_towel.root_path,
            "ground_prim_path": scene.ground_prim_path,
            "left_robot_prim_path": scene.left_prim_path,
            "right_robot_prim_path": scene.right_prim_path,
            "original_garment_moved_out_of_workspace": None,
            "capture_path": str(capture_path),
            "metrics_trace_path": str(metrics_trace_path),
            "obj_path": str(obj_path),
            "num_triangles_in_obj": int(num_triangles),
            "fold_steps": int(args.fold_steps),
            "settle_steps": int(args.settle_steps),
            "start_width": float(start_metrics["width_x"]),
            "best_width": float(best_width),
            "best_step": int(best_step),
            "final_width": float(final_width),
            "width_success_threshold": float(args.width_success_threshold),
            "fold_width_reduction_success": bool(fold_success),
            "start_metrics": start_metrics,
            "best_metrics": size_metrics(best_points),
            "final_metrics": final_metrics,
            "width_trace": widths,
            "scene_info": scene_info,
            **rerun_info,
            "honest_note": (
                "This is an assisted particle-space fold primitive applied to the real clean towel v2 particle state "
                "inside a standalone SO-101 scene. It is not an autonomous robot-contact policy."
            ),
        }
        write_json(args.out_dir / DEFAULT_SUMMARY_PATH.name, summary, print_payload=True)

        smoke_summary = load_json_if_exists(args.out_dir / SMOKE_SUMMARY_PATH.name)
        commands_run = [COMPILE_COMMAND]
        if smoke_summary and smoke_summary.get("command_run"):
            commands_run.append(str(smoke_summary["command_run"]))
        commands_run.append(command_run_string())
        status = status_payload(smoke_summary, summary, commands_run)
        write_json(args.out_dir / DEFAULT_STATUS_PATH.name, status, print_payload=False)

    except Exception as exc:
        summary = {
            "status": "SO101_CLEAN_TOWEL_ASSISTED_FOLD_V1_FAILED",
            "command_run": command_run_string(),
            "task": args.task,
            "device": str(getattr(args, "device", "unknown")),
            "clean_towel_usd_path": str(args.usd_path),
            "exception_type": exc.__class__.__name__,
            "exception": repr(exc),
            "honest_note": "The assisted fold did not complete; no fold success is claimed.",
        }
        write_json(args.out_dir / DEFAULT_SUMMARY_PATH.name, summary, print_payload=True)
        smoke_summary = load_json_if_exists(args.out_dir / SMOKE_SUMMARY_PATH.name)
        commands_run = [COMPILE_COMMAND]
        if smoke_summary and smoke_summary.get("command_run"):
            commands_run.append(str(smoke_summary["command_run"]))
        commands_run.append(command_run_string())
        status = status_payload(smoke_summary, summary, commands_run)
        write_json(args.out_dir / DEFAULT_STATUS_PATH.name, status, print_payload=False)
    finally:
        import os
        import sys

        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
