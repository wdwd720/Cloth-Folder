"""V17 deep-slow demo dataset summary (CPU, numpy-only; NO torch/Isaac imports).

Writes the single required dataset-verdict artifact:
  /artifacts/contact_v17/contact_v17_demo_dataset_summary.json

Reports whether the deep-slow demos are DEEPER than V16 (mean width_after below the
V16 demo mean of 0.564, target < 0.55) while staying non-ballistic, plus the honest
counts and the exact next step.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_DIR = Path("/artifacts/contact_v17")
MIN_VALID = 20
V16_DEMO_MEAN_WIDTH_AFTER = 0.564


def _mean(xs: list[float]) -> float | None:
    return float(np.mean(xs)) if xs else None


def main() -> None:
    p = argparse.ArgumentParser(description="Summarize the V17 deep-slow demo dataset.")
    p.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    args = p.parse_args()

    manifest_path = args.dir / "demos" / "so101_deep_slow_demos_manifest_v17.json"
    npz_path = args.dir / "demos" / "so101_deep_slow_demos_v17.npz"
    validation_path = args.dir / "demos" / "contact_v17_demo_validation.json"
    out = args.dir / "contact_v17_demo_dataset_summary.json"

    summary: dict[str, Any] = {"status": "V17_DEMO_DATASET_SUMMARY_STARTED", "dir": str(args.dir)}
    try:
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        validation = json.loads(validation_path.read_text()) if validation_path.exists() else {}
        episodes = manifest.get("episodes", [])
        valid_eps = [e for e in episodes if e.get("valid_controlled_contact")]

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
        mean_after = _mean([e["best_width"] for e in valid_eps])
        summary.update(
            demo_type=manifest.get("demo_type"),
            observation_schema_version=manifest.get("observation_schema_version"),
            no_phase_clock=bool(manifest.get("no_phase_clock", True)),
            config_name=manifest.get("config_name"),
            deep_slow_ranges=manifest.get("deep_slow_ranges"),
            baseline_no_contact_edge_disp_m=manifest.get("baseline_no_contact_edge_disp_m"),
            num_demos_attempted=int(manifest.get("num_demos_attempted", len(episodes))),
            num_valid_success=num_valid,
            num_ballistic=int(manifest.get("num_ballistic", 0)),
            num_nofold=int(manifest.get("num_nofold", 0)),
            min_valid_required=MIN_VALID,
            total_samples=total_samples,
            state_dim=state_dim,
            action_dim=action_dim,
            action_is_real_joint_targets=bool(manifest.get("action_is_real_joint_targets", False)),
            valid_mean_width_before=_mean([e["width_before"] for e in valid_eps]),
            valid_mean_width_after=mean_after,
            valid_min_width_after=float(min(e["best_width"] for e in valid_eps)) if valid_eps else None,
            valid_num_below_0_55=int(sum(1 for e in valid_eps if e["best_width"] < 0.55)),
            valid_mean_width_reduction_m=_mean([e["width_reduction_m"] for e in valid_eps]),
            valid_mean_edge_displacement_m=_mean([e["edge_displacement_m"] for e in valid_eps]),
            valid_mean_particle_velocity_peak=_mean([e["particle_velocity_peak"] for e in valid_eps]),
            v16_demo_mean_width_after=V16_DEMO_MEAN_WIDTH_AFTER,
            beats_v16_demo_mean=bool(mean_after is not None and mean_after < V16_DEMO_MEAN_WIDTH_AFTER),
            valid_mean_width_after_below_0_55=bool(mean_after is not None and mean_after < 0.55),
            demo_validation_valid_dataset=bool(validation.get("valid_dataset", False)),
            robot_contact_data_ready=ready,
            v17_dataset_ready_for_training_v4=bool(ready and validation.get("valid_dataset", False)),
        )
        if valid_eps:
            summary["diversity_realized"] = {
                "press_depth": [float(min(e["press_depth"] for e in valid_eps)), float(max(e["press_depth"] for e in valid_eps))],
                "drag_dist": [float(min(e["drag_dist"] for e in valid_eps)), float(max(e["drag_dist"] for e in valid_eps))],
                "drag_steps": [int(min(e["drag_steps"] for e in valid_eps)), int(max(e["drag_steps"] for e in valid_eps))],
            }
        if not summary["v17_dataset_ready_for_training_v4"]:
            summary["exact_next_step"] = f"Have {num_valid}/{MIN_VALID} valid; re-run V17 demo collection (adjust ranges)."
        elif summary["beats_v16_demo_mean"]:
            summary["exact_next_step"] = "Run Training v4 on the deeper V17 demos, then multi-seed Eval v4."
        else:
            summary["exact_next_step"] = ("V17 valid but NOT deeper than V16 (mean_width_after >= 0.564); "
                                          "widen drag_dist toward 0.28 / lengthen drag before Training v4.")
        summary["status"] = "V17_DEMO_DATASET_SUMMARY_DONE"
    except Exception as exc:  # noqa: BLE001
        import traceback
        summary["status"] = "V17_DEMO_DATASET_SUMMARY_FAILED"
        summary["error"] = repr(exc)
        summary["traceback_tail"] = traceback.format_exc()[-4000:]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
