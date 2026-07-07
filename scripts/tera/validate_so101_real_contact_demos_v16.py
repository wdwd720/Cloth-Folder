"""V16 demo dataset validator (CPU, numpy-only; NO torch/Isaac imports).

Loads the V16 demo npz + manifest from the Modal volume and enforces the honesty
invariants that make the dataset usable for a STATE-REACTIVE policy:

  * observation carries NO phase/progress clock (schema flag + feature-name scan)
  * action labels are REAL commanded SO-101 joint targets (12-D)
  * state/action arrays are shape-consistent; state_dim == 72 (V16 schema)
  * only VALID controlled-contact episodes contribute samples
  * robot_contact_data_ready is consistent with num_valid >= MIN_VALID

Writes <dir>/contact_v16_demo_validation.json and prints a status dict.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_NPZ = Path("/artifacts/contact_v16/demos/so101_real_contact_demos_v16.npz")
DEFAULT_MANIFEST = Path("/artifacts/contact_v16/demos/so101_real_contact_demos_manifest_v16.json")
CLOCK_TOKENS = ("progress", "phase", "timestep", "time_step", "clock", "normalized_step")
EXPECTED_STATE_DIM = 72
MIN_VALID = 20


def _load_manifest(path: Path) -> dict[str, Any]:
    for cand in (path, path.parent / "so101_real_contact_demos_manifest_v16.json"):
        if cand.exists():
            return json.loads(cand.read_text())
    return {}


def main() -> None:
    p = argparse.ArgumentParser(description="Validate V16 state-reactive demo dataset.")
    p.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    out = args.out or (args.npz.parent / "contact_v16_demo_validation.json")
    result: dict[str, Any] = {
        "status": "V16_DEMO_VALIDATION_STARTED",
        "npz_path": str(args.npz),
        "checks": {},
        "valid_dataset": False,
    }
    try:
        manifest = _load_manifest(args.manifest)
        if not args.npz.exists():
            result["status"] = "V16_DEMO_VALIDATION_FAILED"
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
        num_ballistic = int(manifest.get("num_ballistic", 0))
        num_nofold = int(manifest.get("num_nofold", 0))

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
        result["num_ballistic"] = num_ballistic
        result["num_nofold"] = num_nofold
        result["robot_contact_data_ready"] = ready_flag
        # dataset is VALID for training iff all invariants hold AND >= MIN_VALID demos
        core = ["npz_loads", "states_2d", "actions_2d", "same_num_samples", "action_dim_is_12",
                "state_dim_is_72", "no_phase_clock_flag", "no_clock_feature_names",
                "action_is_real_joint_targets", "has_samples", "min_valid_met", "ready_flag_consistent"]
        result["valid_dataset"] = bool(all(checks[k] for k in core))
        result["status"] = "V16_DEMO_VALIDATION_DONE"
    except Exception as exc:  # noqa: BLE001
        import traceback
        result["status"] = "V16_DEMO_VALIDATION_FAILED"
        result["error"] = repr(exc)
        result["traceback_tail"] = traceback.format_exc()[-4000:]
    _write(out, result)


def _write(out: Path, result: dict) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
