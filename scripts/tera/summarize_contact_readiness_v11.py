"""V11 contact-readiness summarizer.

Reads the per-experiment V11 JSONs (kinematic-velocity, dynamic-rigid,
particle-mass) and writes a single contact_v11_summary.json with the required
fields. Pure file aggregation; no Isaac Sim launch.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

V11_DIR = Path("/workspace/leisaac/tera_checkpoints/clean_towel_v11")
OUT_JSON = V11_DIR / "contact_v11_summary.json"

SOURCES = [
    "kinematic_velocity_v11.json",
    "dynamic_rigid_v11.json",
    "particle_mass_v11.json",
]


def load(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def main() -> None:
    loaded = {name: load(V11_DIR / name) for name in SOURCES}

    trials: list[dict[str, Any]] = []
    for name in ("kinematic_velocity_v11.json", "dynamic_rigid_v11.json"):
        doc = loaded.get(name) or {}
        for t in doc.get("trials", []):
            t = dict(t)
            t["_source"] = name
            trials.append(t)

    touched = [t for t in trials if t.get("actual_touch")]
    valid = [t for t in trials if t.get("valid_controlled_contact")]
    ballistic = [t for t in trials if t.get("spurious_ballistic")]

    def _wred(t):
        return float(t.get("width_reduction_m", 0.0) or 0.0)

    # best trial = best VALID controlled contact (by width reduction); ballistic
    # flings are explicitly excluded from "best" and from the solved decision.
    best_trial = max(valid, key=_wred) if valid else (
        max([t for t in touched if not t.get("spurious_ballistic")], key=_wred, default=None)
    )
    solved = len(valid) > 0

    summary: dict[str, Any] = {
        "status": "V11_CONTACT_READINESS_SUMMARY",
        "sources_present": {k: (v is not None) for k, v in loaded.items()},
        "num_trials": len(trials),
        "num_trials_touching": len(touched),
        "num_trials_valid_controlled_contact": len(valid),
        "num_trials_spurious_ballistic": len(ballistic),
        "spurious_ballistic_trials": [
            {"trial_name": t.get("trial_name"), "drive_mode": t.get("drive_mode"),
             "edge_displacement_m": t.get("edge_displacement_m"),
             "width_reduction_m": t.get("width_reduction_m"),
             "particle_velocity_peak": t.get("particle_velocity_peak")}
            for t in ballistic
        ],
        "contact_transfer_solved": bool(solved),
        "best_trial": None if best_trial is None else {
            "trial_name": best_trial.get("trial_name"),
            "source": best_trial.get("_source"),
            "drive_mode": best_trial.get("drive_mode"),
            "drive_methods_used": best_trial.get("drive_methods_used"),
            "rigid_view_available": best_trial.get("rigid_view_available"),
            "actual_touch": best_trial.get("actual_touch"),
            "actual_touch_distance_m": best_trial.get("actual_touch_distance_m"),
            "edge_displacement_m": best_trial.get("edge_displacement_m"),
            "max_particle_displacement_m": best_trial.get("max_particle_displacement_m"),
            "particle_velocity_peak": best_trial.get("particle_velocity_peak"),
            "width_before": best_trial.get("width_before"),
            "width_after": best_trial.get("width_after"),
            "width_reduction_m": best_trial.get("width_reduction_m"),
            "valid_controlled_contact": best_trial.get("valid_controlled_contact"),
        },
        # top-level required fields (from the best VALID trial)
        "actual_touch": bool(best_trial.get("actual_touch")) if best_trial else False,
        "edge_displacement_m": (float(best_trial.get("edge_displacement_m", 0.0)) if best_trial else 0.0),
        "width_before": (best_trial.get("width_before") if best_trial else None),
        "width_after": (best_trial.get("width_after") if best_trial else None),
        "particle_velocity_peak": (best_trial.get("particle_velocity_peak") if best_trial else 0.0),
        "final_robot_contact_policy_training_ready": bool(solved),
        "remaining_blocker": (
            None if solved else
            "No VALID controlled rigid contact transfer yet (a physics-driven collider that reduces "
            "towel width without ballistic fling). Contact transfer is NOT solved; final robot-contact "
            "policy training remains disallowed."
        ),
        "particle_mass_findings": (loaded.get("particle_mass_v11.json") or {}).get("conclusion"),
        "honest_note": (
            "Success = a VALID controlled contact: actual_touch (<0.01 m), edge_displacement in "
            "(0.02, 0.5] m, real width reduction (>0.005 m), and particles NOT pinned at the velocity "
            "cap. Ballistic dynamic-sphere flings (huge displacement, no width reduction, velocity at "
            "cap) are explicitly EXCLUDED. Directly writing particle positions does not count and is "
            "not used."
        ),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    sys.exit(0)


if __name__ == "__main__":
    main()
