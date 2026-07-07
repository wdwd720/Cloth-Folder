"""V16 demo dataset summary (CPU, numpy-only; NO torch/Isaac imports).

Reads the V16 demo manifest (+ npz + validation JSON if present) from the Modal
volume and writes the single required dataset-verdict artifact:
  /artifacts/contact_v16/contact_v16_demo_dataset_summary.json

Reports the honest V16 dataset verdict: how many diverse real articulation-driven
folds were collected, how many were valid vs ballistic vs no-fold, the fold
statistics over the VALID demos, confirmation that the observation carries no
phase/progress clock, and the exact next step.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_DIR = Path("/artifacts/contact_v16")
MIN_VALID = 20


def _mean(xs: list[float]) -> float | None:
    return float(np.mean(xs)) if xs else None


def main() -> None:
    p = argparse.ArgumentParser(description="Summarize the V16 state-reactive demo dataset.")
    p.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    args = p.parse_args()

    manifest_path = args.dir / "demos" / "so101_real_contact_demos_manifest_v16.json"
    npz_path = args.dir / "demos" / "so101_real_contact_demos_v16.npz"
    validation_path = args.dir / "demos" / "contact_v16_demo_validation.json"
    out = args.dir / "contact_v16_demo_dataset_summary.json"

    summary: dict[str, Any] = {"status": "V16_DEMO_DATASET_SUMMARY_STARTED", "dir": str(args.dir)}
    try:
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        validation = json.loads(validation_path.read_text()) if validation_path.exists() else {}
        episodes = manifest.get("episodes", [])
        valid_eps = [e for e in episodes if e.get("valid_controlled_contact")]
        ballistic_eps = [e for e in episodes if e.get("spurious_ballistic")]

        state_dim, action_dim, total_samples = 0, 0, 0
        if npz_path.exists():
            data = np.load(npz_path, allow_pickle=True)
            st = np.asarray(data["states"], np.float32)
            ac = np.asarray(data["actions"], np.float32)
            total_samples = int(st.shape[0])
            state_dim = int(st.shape[1]) if st.ndim == 2 else 0
            action_dim = int(ac.shape[1]) if ac.ndim == 2 else 0

        num_valid = int(manifest.get("num_valid_success", len(valid_eps)))
        ready = bool(num_valid >= MIN_VALID)
        summary.update(
            demo_type=manifest.get("demo_type"),
            observation_schema_version=manifest.get("observation_schema_version"),
            no_phase_clock=bool(manifest.get("no_phase_clock", True)),
            config_name=manifest.get("config_name"),
            reach_edge=manifest.get("reach_edge"),
            baseline_no_contact_edge_disp_m=manifest.get("baseline_no_contact_edge_disp_m"),
            num_demos_attempted=int(manifest.get("num_demos_attempted", len(episodes))),
            num_valid_success=num_valid,
            num_ballistic=int(manifest.get("num_ballistic", len(ballistic_eps))),
            num_nofold=int(manifest.get("num_nofold", 0)),
            min_valid_required=MIN_VALID,
            total_samples=total_samples,
            state_dim=state_dim,
            action_dim=action_dim,
            action_is_real_joint_targets=bool(manifest.get("action_is_real_joint_targets", False)),
            # fold statistics over the VALID demos only
            valid_mean_width_before=_mean([e["width_before"] for e in valid_eps]),
            valid_mean_width_after=_mean([e["best_width"] for e in valid_eps]),
            valid_mean_width_reduction_m=_mean([e["width_reduction_m"] for e in valid_eps]),
            valid_mean_edge_displacement_m=_mean([e["edge_displacement_m"] for e in valid_eps]),
            valid_mean_particle_velocity_peak=_mean([e["particle_velocity_peak"] for e in valid_eps]),
            # diversity ranges actually realized across the valid demos
            press_depth_range=[_mean([e["press_depth"] for e in valid_eps])] if valid_eps else None,
            demo_validation_valid_dataset=bool(validation.get("valid_dataset", False)),
            demo_validation_checks=validation.get("checks", {}),
            robot_contact_data_ready=ready,
            v16_dataset_ready_for_training_v3=bool(ready and validation.get("valid_dataset", False)),
        )
        if valid_eps:
            summary["diversity_realized"] = {
                "press_depth": [float(min(e["press_depth"] for e in valid_eps)), float(max(e["press_depth"] for e in valid_eps))],
                "drag_dist": [float(min(e["drag_dist"] for e in valid_eps)), float(max(e["drag_dist"] for e in valid_eps))],
                "drag_steps": [int(min(e["drag_steps"] for e in valid_eps)), int(max(e["drag_steps"] for e in valid_eps))],
                "towel_dx": [float(min(e["towel_dx"] for e in valid_eps)), float(max(e["towel_dx"] for e in valid_eps))],
                "towel_dy": [float(min(e["towel_dy"] for e in valid_eps)), float(max(e["towel_dy"] for e in valid_eps))],
            }
        summary["exact_next_step"] = (
            "Run Training v3 (state-reactive, no_phase_clock, action-smoothing/effort penalty) on this dataset."
            if summary["v16_dataset_ready_for_training_v3"] else
            f"Collect more valid demos: have {num_valid}/{MIN_VALID} valid; re-run V16 demo collection."
        )
        summary["status"] = "V16_DEMO_DATASET_SUMMARY_DONE"
    except Exception as exc:  # noqa: BLE001
        import traceback
        summary["status"] = "V16_DEMO_DATASET_SUMMARY_FAILED"
        summary["error"] = repr(exc)
        summary["traceback_tail"] = traceback.format_exc()[-4000:]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("demo_validation_checks",)}, indent=2))


if __name__ == "__main__":
    main()
