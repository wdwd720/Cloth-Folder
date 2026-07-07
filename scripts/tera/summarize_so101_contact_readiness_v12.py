"""V12 SO-101 contact-readiness summarizer.

Aggregates the SO-101 fingertip physics-drive trials (and the demo manifest if
present) into contact_v12_summary.json with the required fields. Pure file
aggregation; no Isaac launch.

Honesty model:
- so101_contact_transfer_solved  = a VALID controlled contact (physics-driven
  fingertip proxy that reduces towel width, not a teleport and not a ballistic
  fling) occurred in the real SO-101 scene.
- robot_contact_data_ready        = true ONLY if demos with REAL robot-joint-
  action labels + valid contact exist. Proxy-driven folds with synthetic joint
  labels do NOT qualify.
- final_robot_contact_policy_training_ready = so101_contact_transfer_solved AND
  robot_contact_data_ready.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

V12_DIR = Path("/workspace/leisaac/tera_checkpoints/clean_towel_v12")
DRIVE_JSON = V12_DIR / "so101_fingertip_physics_drive_v12.json"
DEMO_MANIFEST = V12_DIR / "demos" / "so101_contact_demos_manifest_v12.json"
OUT_JSON = V12_DIR / "contact_v12_summary.json"


def load(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def main() -> None:
    drive = load(DRIVE_JSON) or {}
    demos = load(DEMO_MANIFEST)
    trials = list(drive.get("trials", []))

    def _wred(t):
        return float(t.get("width_reduction_m", 0.0) or 0.0)

    valid = [t for t in trials if t.get("valid_controlled_contact")]
    ballistic = [t for t in trials if t.get("spurious_ballistic")]
    teleport = [t for t in trials if t.get("drive_mode") == "teleport"]
    best = max(valid, key=_wred) if valid else (
        max([t for t in trials if not t.get("spurious_ballistic")], key=_wred, default=None)
    )
    solved = len(valid) > 0

    robot_contact_data_ready = bool(demos and demos.get("robot_contact_data_ready"))

    summary: dict[str, Any] = {
        "status": "V12_SO101_CONTACT_READINESS_SUMMARY",
        "sources_present": {"drive": bool(drive), "demos": demos is not None},
        "contact_element": drive.get("contact_element"),
        "num_trials": len(trials),
        "num_valid_controlled_contact": len(valid),
        "num_spurious_ballistic": len(ballistic),
        "teleport_baseline_moved_towel": bool(any(t.get("edge_displacement_m", 0.0) > 0.02 for t in teleport)),
        "so101_contact_transfer_solved": bool(solved),
        "best_trial": None if best is None else {
            "trial_name": best.get("trial_name"),
            "drive_mode": best.get("drive_mode"),
            "drive_methods_used": best.get("drive_methods_used"),
            "valid_controlled_contact": best.get("valid_controlled_contact"),
            "spurious_ballistic": best.get("spurious_ballistic"),
        },
        # required top-level fields
        "width_before": (best.get("width_before") if best else None),
        "width_after": (best.get("width_after") if best else None),
        "edge_displacement_m": (float(best.get("edge_displacement_m", 0.0)) if best else 0.0),
        "actual_touch_distance_m": (best.get("actual_touch_distance_m") if best else None),
        "particle_velocity_peak": (best.get("particle_velocity_peak") if best else 0.0),
        "valid_controlled_contact": bool(best.get("valid_controlled_contact")) if best else False,
        "ballistic_or_teleport_excluded": {
            "num_spurious_ballistic_excluded": len(ballistic),
            "teleport_baseline_included_for_comparison": len(teleport),
        },
        "robot_contact_data_ready": robot_contact_data_ready,
        "final_robot_contact_policy_training_ready": bool(solved and robot_contact_data_ready),
        "remaining_blocker": None,
        "honest_note": (
            "so101_contact_transfer_solved means a physics-driven SO-101 fingertip proxy moved+reduced the "
            "towel in the real SO-101 scene (teleport excluded, ballistic flings excluded). robot_contact_data_ready "
            "requires demos with REAL robot-joint-action labels; proxy folds with synthetic joint labels do not qualify."
        ),
    }
    if not solved:
        summary["remaining_blocker"] = (
            "No valid controlled SO-101 fingertip contact (physics-driven proxy did not reduce towel width)."
        )
    elif not robot_contact_data_ready:
        summary["remaining_blocker"] = (
            "SO-101 contact MECHANISM solved (physics-driven fingertip proxy moves cloth), but true robot-joint-action "
            "demonstrations are not available: the SO-101 reach solution (V6) is not on Modal, so the fold is proxy-driven "
            "with synthetic joint labels. Next: run V6 reach search on Modal + drive the arm articulation to fold, then "
            "record real (state, joint_action) contact demos."
        )

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    sys.exit(0)


if __name__ == "__main__":
    main()
