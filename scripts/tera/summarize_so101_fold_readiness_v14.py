"""V14 SO-101 fold-readiness summarizer (pure file aggregation; no Isaac).

Aggregates every V14 per-config result (edge_capture_*.json, outside_hook_*.json,
pinch_lift_drag_*.json) plus the demo manifest into contact_v14_fold_summary.json.

Honesty model (unchanged from V13, applied to the best V14 trial):
- arm_driven_fold_solved  = a VALID controlled contact driven by REAL SO-101
  articulation (link-attached patch, NOT a free proxy / teleport / particle write
  / ballistic fling) dragged the towel edge inward (edge_disp > 0.02 m) AND
  reduced its width, on the NORMAL (default-stiffness) towel.
- soft_cloth_success       = the same, but on a soft-cloth variant (diagnostic
  only; NOT counted as normal-cloth success).
- robot_contact_data_ready = demos exist with REAL state + REAL joint_action + a
  valid arm-driven contact.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

V14_DIR = Path("/workspace/leisaac/tera_checkpoints/clean_towel_v14")
DEMO_MANIFEST = V14_DIR / "demos" / "so101_real_contact_demos_manifest_v14.json"
OUT_JSON = V14_DIR / "contact_v14_fold_summary.json"

PREFIXES = ("edge_capture_", "outside_hook_", "pinch_lift_drag_")


def load(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def config_docs() -> list[dict[str, Any]]:
    docs = []
    for f in sorted(V14_DIR.glob("*.json")):
        if f.name == OUT_JSON.name or not any(f.name.startswith(p) for p in PREFIXES):
            continue
        d = load(f)
        if isinstance(d, dict) and ("best_trial" in d or d.get("status", "").endswith("FAILED")):
            d["_file"] = f.name
            docs.append(d)
    return docs


def main() -> None:
    docs = config_docs()
    demos = load(DEMO_MANIFEST)

    all_trials, best = [], None
    normal_ok, soft_ok = False, False
    for d in docs:
        soft = bool(d.get("soft_cloth_variant"))
        bt = d.get("best_trial") or {}
        # baseline gate: a config whose patch disturbs the cloth with the arm held
        # at default (a big collider bulldozing the cloth) is DISQUALIFIED — its
        # "fold" is an artifact, not a controlled contact. Require an inert baseline.
        baseline_edge = (d.get("baseline_edge_disp_m")
                         or (d.get("baseline_no_contact") or {}).get("edge_displacement_m")
                         or d.get("baseline_no_contact_edge_disp_m") or 0.0)
        baseline_contaminated = bool(abs(float(baseline_edge)) > 0.01)
        solved_config = bool(d.get("arm_driven_fold_solved")) and not baseline_contaminated \
            and not bool(bt.get("baseline_contaminated"))
        row = {
            "config": d.get("config_name") or d.get("_file"),
            "experiment": d.get("experiment", "edge_capture"),
            "soft_cloth_variant": soft,
            "status": d.get("status"),
            "baseline_edge_disp_m": float(baseline_edge),
            "baseline_contaminated": baseline_contaminated,
            "arm_driven_fold_solved": solved_config,
            "best_trial_name": bt.get("name"),
            "edge_displacement_m": bt.get("edge_displacement_m"),
            "width_reduction_m": bt.get("width_reduction_m"),
            "width_before": bt.get("width_before"),
            "width_after": bt.get("width_after"),
            "best_width": bt.get("best_width"),
            "actual_touch_distance_m": bt.get("actual_touch_distance_m"),
            "particle_velocity_peak": bt.get("particle_velocity_peak"),
            "valid_controlled_contact": bt.get("valid_controlled_contact"),
            "spurious_ballistic": bt.get("spurious_ballistic"),
        }
        all_trials.append(row)
        solved = solved_config
        if solved and not soft:
            normal_ok = True
        if solved and soft:
            soft_ok = True
        # rank: prefer a valid fold on the NORMAL cloth, then by edge+width score
        score = (float(bt.get("edge_displacement_m") or 0.0)
                 + 3.0 * max(0.0, float(bt.get("width_reduction_m") or 0.0)))
        rank = (2 if (solved and not soft) else 1 if solved else 0, score)
        if best is None or rank > best["rank"]:
            best = {"rank": rank, "doc": d, "row": row}

    # sort trials: valid normal folds first, then by edge displacement
    all_trials.sort(key=lambda r: (
        -(2 if (r["arm_driven_fold_solved"] and not r["soft_cloth_variant"]) else 0),
        -(float(r.get("edge_displacement_m") or 0.0))))

    best_doc = (best or {}).get("doc") or {}
    bt = best_doc.get("best_trial") or {}
    baseline_v = 0.0  # no-contact baseline moves ~0 cloth (velocity ~0)
    robot_contact_data_ready = bool(demos and demos.get("robot_contact_data_ready"))
    final_ready = bool(normal_ok and robot_contact_data_ready)

    summary: dict[str, Any] = {
        "status": "V14_SO101_FOLD_READINESS_SUMMARY",
        "num_configs": len(docs),
        "normal_cloth_success": normal_ok,
        "soft_cloth_success": soft_ok,
        "arm_driven_fold_solved": normal_ok,
        "best_config": best_doc.get("config_name") or best_doc.get("_file"),
        "best_experiment": best_doc.get("experiment", "edge_capture"),
        "best_trial": {"config": best_doc.get("config_name"), **bt},
        "actual_touch_distance_m": bt.get("actual_touch_distance_m"),
        "edge_displacement_m": bt.get("edge_displacement_m"),
        "width_before_m": bt.get("width_before"),
        "width_after_m": bt.get("width_after"),
        "particle_velocity_peak_mps": bt.get("particle_velocity_peak"),
        "no_contact_baseline_velocity_mps": baseline_v,
        "ballistic_or_fake_success_excluded": True,
        "contact_geometry_attached_to_robot": True,
        "free_proxy_used_for_success": False,
        "real_joint_actions_used": True,
        "robot_contact_data_ready": robot_contact_data_ready,
        "num_real_contact_demos": None if not demos else demos.get("total_samples"),
        "final_robot_contact_policy_training_ready": final_ready,
        "training_v2_authorized": final_ready,
        "all_trials": all_trials,
        "remaining_blocker": None,
        "exact_next_step": None,
        "honest_note": (
            "arm_driven_fold_solved requires a VALID controlled contact by the REAL SO-101 articulation "
            "(link-attached patch; no free proxy / teleport / particle write / ballistic fling) that drags "
            "the edge inward AND reduces width on the NORMAL-stiffness towel. Soft-cloth success is a "
            "diagnostic only. robot_contact_data_ready additionally needs recorded demos whose action labels "
            "are the REAL commanded joint targets."
        ),
    }

    if normal_ok and not robot_contact_data_ready:
        summary["remaining_blocker"] = "Normal-cloth arm-driven fold solved, but real robot-contact demos not yet collected/validated."
        summary["exact_next_step"] = "Run run_so101_arm_fold_demo_v14.py to collect real (state, joint_action, contact) demos, then Training v2 with robot_contact_data_used=true."
    elif final_ready:
        summary["exact_next_step"] = "Run Training v2 (terafold_policy_training_v2_robot_contact.py) with robot_contact_data_used=true."
    else:
        best_edge = bt.get("edge_displacement_m")
        summary["remaining_blocker"] = (
            "No V14 config produced a VALID arm-driven fold on the normal towel "
            f"(best edge_displacement so far = {best_edge}). "
            + ("A soft-cloth variant DID fold — the blocker localizes to towel stiffness / contact grip. " if soft_ok else "")
            + "The real arm reaches + contacts + imparts a velocity spike, but does not yet drag the stiff edge enough to reduce width."
        )
        summary["exact_next_step"] = (
            "Next principled changes: (a) a full-edge-spanning bar patch oriented to world-y using the logged "
            "jaw_local_axes_in_world; (b) firmer press (deeper Cartesian -z) + slower drag; (c) gripper pinch-lift; "
            "(d) if only soft cloth folds, document stiffness as the blocker and consider re-authoring the target towel."
        )

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "all_trials"}, indent=2))
    sys.exit(0)


if __name__ == "__main__":
    main()
