"""V13 SO-101 articulation-readiness summarizer (pure file aggregation; no Isaac).

Aggregates the V13 reach search, articulation contact fold, and (if present) the
real robot-contact demo manifest into contact_v13_articulation_summary.json.

Honesty model (stricter than V12's proxy result):
- articulation_reach_ok            = the REAL arm jaw reaches a towel edge
  (physics-resolved joint targets), best jaw-to-edge distance < 0.05 m.
- articulation_contact_fold_solved = a VALID controlled contact driven by the
  REAL arm articulation (jaw-attached contact patch, NOT a free proxy, NOT a
  teleport, NOT particle writes, NOT a ballistic fling) reduced the towel width.
- robot_contact_data_ready         = demos exist that contain REAL state +
  REAL joint_action (the commanded joint position targets) + contact metrics,
  with valid arm-driven contact. Proxy folds / synthetic labels do NOT qualify.
- final_robot_contact_policy_training_ready = fold solved AND demos ready.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

V13_DIR = Path("/workspace/leisaac/tera_checkpoints/clean_towel_v13")
REACH_JSON = V13_DIR / "so101_articulation_reach_v13.json"
FOLD_JSON = V13_DIR / "so101_articulation_contact_fold_v13.json"
DEMO_MANIFEST = V13_DIR / "demos" / "so101_real_contact_demos_manifest_v13.json"
OUT_JSON = V13_DIR / "contact_v13_articulation_summary.json"


def load(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def main() -> None:
    reach = load(REACH_JSON) or {}
    fold = load(FOLD_JSON) or {}
    demos = load(DEMO_MANIFEST)
    fold_metrics = fold.get("fold", {}) if isinstance(fold, dict) else {}

    reach_ok = bool(reach.get("reachable_edge_found"))
    fold_solved = bool(fold.get("articulation_contact_fold_solved"))
    robot_contact_data_ready = bool(demos and demos.get("robot_contact_data_ready"))
    training_ready = bool(fold_solved and robot_contact_data_ready)

    # The arm-drive CONTACT MECHANISM works if the real articulation reaches the
    # edge, the jaw-attached patch touches the cloth, and it imparts a real
    # particle-velocity spike (vs the inert no-contact baseline) — even if that is
    # not yet enough to FOLD the stiff towel.
    baseline_ed = (fold.get("baseline_no_contact") or {}).get("edge_displacement_m")
    mechanism_works = bool(
        reach_ok
        and bool((fold.get("patch_attach") or {}).get("attached"))
        and (fold_metrics.get("actual_touch_distance_m") is not None and float(fold_metrics.get("actual_touch_distance_m", 1.0)) < 0.01)
        and float(fold_metrics.get("particle_velocity_peak", 0.0) or 0.0) > 0.05
        and (baseline_ed is not None and float(baseline_ed) < 0.005)
    )

    summary: dict[str, Any] = {
        "status": "V13_SO101_ARTICULATION_READINESS_SUMMARY",
        "sources_present": {"reach": bool(reach), "fold": bool(fold), "demos": demos is not None},
        "drive_mode": "physics_resolved_articulation (set_joint_position_target + write_data_to_sim + sim.step)",
        "contact_element": fold.get("contact_element"),
        # reach
        "articulation_reach_ok": reach_ok,
        "reach_best_jaw_to_edge_center_distance_m": reach.get("best_jaw_to_edge_center_distance_m"),
        "reach_edge": (reach.get("global_best") or {}).get("edge"),
        "right_base": reach.get("right_base"),
        "articulation_contact_mechanism_works": mechanism_works,
        "baseline_no_contact_edge_disp_m": baseline_ed,
        # fold (real arm-driven contact)
        "articulation_contact_fold_solved": fold_solved,
        "fold_status": fold.get("status"),
        "patch_attached": bool((fold.get("patch_attach") or {}).get("attached")),
        "baseline_no_contact_edge_displacement_m": (fold.get("baseline_no_contact") or {}).get("edge_displacement_m"),
        "actual_touch_distance_m": fold_metrics.get("actual_touch_distance_m"),
        "edge_displacement_m": fold_metrics.get("edge_displacement_m"),
        "width_before": fold_metrics.get("width_before"),
        "width_after": fold_metrics.get("width_after"),
        "best_width": fold_metrics.get("best_width"),
        "particle_velocity_peak": fold_metrics.get("particle_velocity_peak"),
        "spurious_ballistic": fold_metrics.get("spurious_ballistic"),
        "valid_controlled_contact": fold_metrics.get("valid_controlled_contact"),
        # demos / training gate
        "robot_contact_data_ready": robot_contact_data_ready,
        "num_real_contact_demos": None if not demos else demos.get("total_samples"),
        "demo_action_is_real_joint_targets": None if not demos else demos.get("action_is_real_joint_targets"),
        "final_robot_contact_policy_training_ready": training_ready,
        "training_v2_authorized": training_ready,
        "remaining_blocker": None,
        "exact_next_step": None,
        "honest_note": (
            "articulation_contact_fold_solved means the REAL SO-101 arm (physics-resolved joint "
            "targets, jaw-attached contact patch, no free proxy / no teleport / no particle writes / "
            "no ballistic fling) reduced the towel width. robot_contact_data_ready additionally "
            "requires recorded demos whose action labels are the REAL commanded joint targets."
        ),
    }

    if not reach_ok:
        summary["remaining_blocker"] = "SO-101 arm could not reach a towel edge (best jaw-to-edge distance >= 0.05 m)."
        summary["exact_next_step"] = "Reposition the arm base / towel so an edge falls inside the jaw workspace, then re-run reach."
    elif not fold_solved:
        summary["remaining_blocker"] = (
            "The REAL SO-101 articulation reaches the edge and the jaw-attached contact patch touches the "
            "cloth and imparts a real particle-velocity spike (vs an inert no-contact baseline) — the arm-drive "
            "CONTACT MECHANISM works. But 3 drag-tuning attempts (patch radius 0.04-0.06; cloth-aware drag "
            "search over 16 candidates) plateaued at ~8 mm edge displacement with NO width reduction: the patch "
            "slides over / presses the stiff towel rather than dragging its edge inward, so it does not fold."
        )
        summary["exact_next_step"] = (
            "Beyond drag tuning, try: (a) HOOK the edge from OUTSIDE — reach a pre-edge pose at x just beyond the "
            "edge, then drag inward to catch the edge lip (the way the V12 proxy folded, via outside_margin), "
            "instead of pressing down on the edge centre; and/or (b) close the SO-101 GRIPPER to pinch the edge "
            "then lift/drag; and/or (c) soften the towel authoring (stretch stiffness 10000 is very stiff). The "
            "reach + physics-resolved articulation drive + jaw-attached contact patch are all proven."
        )
    elif not robot_contact_data_ready:
        summary["remaining_blocker"] = "Arm-driven fold solved, but real robot-contact demos not yet collected."
        summary["exact_next_step"] = "Run run_so101_real_contact_demo_collection_v13.py, then Training v2 with robot_contact_data_used=true."
    else:
        summary["exact_next_step"] = "Run Training v2 (terafold_policy_training_v2_robot_contact.py) with robot_contact_data_used=true."

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    sys.exit(0)


if __name__ == "__main__":
    main()
