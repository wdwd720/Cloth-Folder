"""V13 SO-101 articulation reach search (real arm, physics-resolved joint targets).

Goal: find joint position targets that bring a REAL SO-101 arm's jaw/gripper to a
clean-towel edge, as the seed pose for the V13 articulation-driven fold. This is
the "run V6 reach search on Modal" step the V12 summary called for — ported onto
the proven V12-isolated environment (RL stubs + h5py + single-reset build order).

Design decisions (honest):
- Towel is kept at the CENTERED stable rest (0,0,0.02) proven not to crash the GPU
  particle+articulation solve (the V6 arm-relative center interpenetrates the arms
  and crashes; see test_so101_fingertip_physics_drive_v12).
- The default ±0.75 m arm bases cannot reach a centered towel, so ONE arm base is
  repositioned near a towel edge (``--right_base``). This is a scene-layout choice,
  not a contact trick: the arm is still driven only by physics-resolved joint
  position targets (set_joint_position_target -> write_data_to_sim -> sim.step).
- Reuses the V6 candidate search (`run_arm_search`): warm-start + Latin-hypercube
  joint actions, each applied via `apply_action_step`, measuring the jaw body pose
  vs the towel edge centers. NO contact/fold is claimed here — reach only.

Output (container-local; the Modal app copies it to /artifacts/contact_v13):
  so101_articulation_reach_v13.json  — global_best.action_vector is the seed pose.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# RL import stubs BEFORE any IsaacLab/SimulationApp import (import-only shim;
# RL_STUBS_FOR_ISAACLAB_IMPORT_ONLY; never used for training).
import isaaclab_rl_stubs_v12  # noqa: F401,E402

from isaaclab.app import AppLauncher

from inspect_so101_clean_towel_geometry_v6 import edge_centers_from_metrics
from search_so101_towel_edge_reach_v6 import run_arm_search
from so101_clean_towel_scene_utils_v1 import (
    CLEAN_TOWEL_USD_PATH,
    command_run_string,
    create_standalone_so101_clean_towel_scene,
    force_headless_no_cameras,
    size_metrics,
    write_json,
)

V13_DIR = Path("/workspace/leisaac/tera_checkpoints/clean_towel_v13")
OUT_JSON = V13_DIR / "so101_articulation_reach_v13.json"

# Centered stable towel (proven no-crash) and a right-arm base repositioned near
# the towel so an edge is reachable by the ~0.35 m SO-101 arm.
STABLE_TOWEL_TRANSLATION = (0.0, 0.0, 0.02)
DEFAULT_RIGHT_BASE = (0.42, -0.28, 0.02)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V13 SO-101 articulation reach search.")
    parser.add_argument("--usd_path", type=Path, default=CLEAN_TOWEL_USD_PATH)
    parser.add_argument("--out_dir", type=Path, default=V13_DIR)
    parser.add_argument("--arm", type=str, default="right", choices=["left", "right", "both"])
    parser.add_argument("--num_candidates_per_arm", type=int, default=640)
    parser.add_argument("--settle_steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=1313)
    parser.add_argument("--right_base", type=float, nargs=3, default=list(DEFAULT_RIGHT_BASE))
    parser.add_argument("--left_base", type=float, nargs=3, default=None)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    force_headless_no_cameras(args)
    return args


def main() -> None:
    args = parse_args()
    V13_DIR.mkdir(parents=True, exist_ok=True)
    app = AppLauncher(args).app  # noqa: F841
    started = time.monotonic()

    arms = ["left", "right"] if args.arm == "both" else [args.arm]
    summary: dict[str, Any] = {
        "status": "V13_SO101_ARTICULATION_REACH_STARTED",
        "command_run": command_run_string(),
        "diagnostic_only": True,
        "towel_translation": list(STABLE_TOWEL_TRANSLATION),
        "right_base": list(args.right_base),
        "left_base": None if args.left_base is None else list(args.left_base),
        "arms_searched": arms,
        "drive_mode": "physics_resolved_articulation (set_joint_position_target + write_data_to_sim + sim.step)",
    }
    try:
        scene = create_standalone_so101_clean_towel_scene(
            args,
            clean_usd_path=args.usd_path,
            towel_translation=STABLE_TOWEL_TRANSLATION,
            right_base=tuple(args.right_base),
            left_base=None if args.left_base is None else tuple(args.left_base),
        )
        scene.step_zero(20)
        start_points = scene.clean_towel.points_numpy().astype(np.float32)
        start_metrics = size_metrics(start_points)
        edge_centers = edge_centers_from_metrics(start_metrics)
        summary["towel_start_metrics"] = start_metrics
        summary["towel_edge_centers"] = {k: [float(x) for x in v] for k, v in edge_centers.items()}

        arm_results = {}
        best_global = None
        for arm in arms:
            res = run_arm_search(
                scene=scene, arm=arm, start_points=start_points, edge_centers=edge_centers,
                candidate_count=int(args.num_candidates_per_arm), settle_steps=int(args.settle_steps),
                seed=int(args.seed),
            )
            # keep only JSON-serializable pieces (drop raw np arrays)
            arm_results[arm] = {
                "min_distance_m": float(res["min_distance_m"]),
                "best_by_edge": res["best_by_edge"],
                "top_candidates": res["top_candidates"][:8],
                "num_candidates": int(res["num_candidates"]),
                "joint_ranges_source": res["joint_ranges_source"],
            }
            for edge, item in res["best_by_edge"].items():
                if item is None:
                    continue
                if best_global is None or float(item["distance_m"]) < float(best_global["distance_m"]):
                    best_global = item

        summary["arm_results"] = arm_results
        summary["global_best"] = best_global
        best_dist = float(best_global["distance_m"]) if best_global else float("inf")
        summary["best_jaw_to_edge_center_distance_m"] = best_dist
        summary["reachable_edge_found"] = bool(best_dist < 0.05)
        summary["strong_reachable_edge_found"] = bool(best_dist < 0.025)
        summary["elapsed_s"] = float(time.monotonic() - started)
        summary["status"] = "V13_SO101_ARTICULATION_REACH_DONE"
        write_json(OUT_JSON, summary, print_payload=True)
    except Exception as exc:  # noqa: BLE001
        import traceback

        summary["status"] = "V13_SO101_ARTICULATION_REACH_FAILED"
        summary["exception_type"] = exc.__class__.__name__
        summary["exception"] = repr(exc)
        summary["traceback_tail"] = traceback.format_exc()[-6000:]
        write_json(OUT_JSON, summary, print_payload=True)
    finally:
        sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == "__main__":
    main()
