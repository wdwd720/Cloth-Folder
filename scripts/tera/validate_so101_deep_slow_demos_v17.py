"""V17 deep-slow demo dataset validator (CPU, numpy-only; NO torch/Isaac imports).

Same honesty invariants as the V16 validator (no phase/progress clock; real joint
actions; state_dim==72; only valid non-ballistic episodes contribute; ready flag
consistent with >= MIN_VALID) applied to the V17 deep-slow dataset.

Writes <dir>/contact_v17_demo_validation.json and prints a status dict.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_NPZ = Path("/artifacts/contact_v17/demos/so101_deep_slow_demos_v17.npz")
DEFAULT_MANIFEST = Path("/artifacts/contact_v17/demos/so101_deep_slow_demos_manifest_v17.json")
CLOCK_TOKENS = ("progress", "phase", "timestep", "time_step", "clock", "normalized_step")
EXPECTED_STATE_DIM = 72
MIN_VALID = 20


def main() -> None:
    p = argparse.ArgumentParser(description="Validate V17 deep-slow state-reactive demo dataset.")
    p.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    out = args.out or (args.npz.parent / "contact_v17_demo_validation.json")
    result: dict[str, Any] = {
        "status": "V17_DEMO_VALIDATION_STARTED",
        "npz_path": str(args.npz),
        "checks": {},
        "valid_dataset": False,
    }
    try:
        manifest = json.loads(args.manifest.read_text()) if args.manifest.exists() else {}
        if not args.npz.exists():
            result["status"] = "V17_DEMO_VALIDATION_FAILED"
            result["error"] = f"missing npz: {args.npz}"
            _write(out, result)
            return
        data = np.load(args.npz, allow_pickle=True)
        states = np.asarray(data["states"], np.float32)
        actions = np.asarray(data["actions"], np.float32)
        feature_names = [str(x) for x in list(data["feature_names"])] if "feature_names" in data else \
            [str(x) for x in manifest.get("state_feature_names", [])]
        no_clock_flag = bool(data["no_phase_clock"]) if "no_phase_clock" in data else bool(manifest.get("no_phase_clock"))
        real_actions = bool(data["action_is_real_joint_targets"]) if "action_is_real_joint_targets" in data \
            else bool(manifest.get("action_is_real_joint_targets"))
        ready_flag = bool(data["robot_contact_data_ready"]) if "robot_contact_data_ready" in data \
            else bool(manifest.get("robot_contact_data_ready"))

        clock_hits = [n for n in feature_names for t in CLOCK_TOKENS if t in n.lower()]
        num_valid = int(manifest.get("num_valid_success", 0))

        checks = {
            "npz_loads": True,
            "states_2d": bool(states.ndim == 2),
            "actions_2d": bool(actions.ndim == 2),
            "same_num_samples": bool(states.shape[0] == actions.shape[0]),
            "action_dim_is_12": bool(actions.shape[1] == 12) if actions.ndim == 2 else False,
            "state_dim_is_72": bool(states.shape[1] == EXPECTED_STATE_DIM) if states.ndim == 2 else False,
            "no_phase_clock_flag": no_clock_flag,
            "no_clock_feature_names": bool(len(clock_hits) == 0),
            "feature_names_present": bool(len(feature_names) > 0),
            "feature_name_count_matches_state_dim":
                bool(states.ndim == 2 and len(feature_names) == states.shape[1]) if feature_names else False,
            "action_is_real_joint_targets": real_actions,
            "has_samples": bool(states.shape[0] > 0),
            "min_valid_met": bool(num_valid >= MIN_VALID),
            "ready_flag_consistent": bool(ready_flag == (num_valid >= MIN_VALID)),
        }
        result["checks"] = checks
        result["clock_feature_hits"] = clock_hits
        result["num_samples"] = int(states.shape[0])
        result["state_dim"] = int(states.shape[1]) if states.ndim == 2 else 0
        result["action_dim"] = int(actions.shape[1]) if actions.ndim == 2 else 0
        result["num_valid_success"] = num_valid
        result["num_ballistic"] = int(manifest.get("num_ballistic", 0))
        result["num_nofold"] = int(manifest.get("num_nofold", 0))
        result["valid_mean_width_after"] = manifest.get("valid_mean_width_after")
        result["beats_v16_demo_mean"] = bool(manifest.get("beats_v16_demo_mean", False))
        result["robot_contact_data_ready"] = ready_flag
        core = ["npz_loads", "states_2d", "actions_2d", "same_num_samples", "action_dim_is_12",
                "state_dim_is_72", "no_phase_clock_flag", "no_clock_feature_names",
                "action_is_real_joint_targets", "has_samples", "min_valid_met", "ready_flag_consistent"]
        result["valid_dataset"] = bool(all(checks[k] for k in core))
        result["status"] = "V17_DEMO_VALIDATION_DONE"
    except Exception as exc:  # noqa: BLE001
        import traceback
        result["status"] = "V17_DEMO_VALIDATION_FAILED"
        result["error"] = repr(exc)
        result["traceback_tail"] = traceback.format_exc()[-4000:]
    _write(out, result)


def _write(out: Path, result: dict) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
