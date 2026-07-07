"""V16 state-reactive observation schema (NO progress/phase clock).

The V14/Training-v2/eval-v2 observation (`compact_state`) begins with an explicit
`progress` scalar + a `phase` one-hot — a function of the scripted trajectory
step. A behavior-cloning policy fed that clock can imitate the *timeline* of the
demo without ever reacting to the cloth/robot state, which is exactly the V14 eval
caveat (trajectory imitation + ballistic over-drive).

V16 replaces the clock with the REAL robot + cloth state so the policy MUST be
state-reactive. The observation is assembled ONLY from allowed signals:

  * joint positions           — the actual SO-101 joint angles of both arms; the
                                arm advances monotonically through the fold, so the
                                fold "phase" is INFERABLE from the true joint config
                                (state), never handed to the policy as a clock.
  * jaw / end-effector poses   — world position of each arm's jaw (contact patch pose)
  * jaw velocity              — recent finite-difference velocity of the acting jaw
                                (a "recent velocity/contact feature")
  * previous action           — the last commanded 12-D joint target (action history;
                                lets the policy do smooth closed-loop control and
                                enables a smoothness/velocity penalty in training)
  * towel size metrics         — min/max/center/size (width & edge extent in sim)
  * corner + edge features     — 4 corners + 4 edge centers (cloth shape)
  * relative jaw-to-edge       — acting jaw minus the target-edge center

Explicitly DISALLOWED and absent here: a normalized timestep, and any phase/progress
variable that tells the policy where it is in the scripted trajectory.

This module is import-safe under `py_compile` (no Isaac imports at module scope).
Both the V16 demo collection AND the v3 eval import `compact_state_reactive` so the
train-time and rollout-time observation vectors are byte-for-byte identical.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Reuse pure-geometry helpers (NO clock inside them).
from clean_towel_v4_common import ACTION_NAMES, corner_and_edge_features
from clean_towel_v5_contact_common import towel_edge_masks
from so101_clean_towel_scene_utils_v1 import size_metrics

STATE_REACTIVE_SCHEMA_VERSION = "v16_state_reactive_no_phase_clock"
TARGET_EDGE = "right"  # the V14 fold acts on the towel's right edge
EDGE_BAND = 0.035


def joint_positions_from_scene(scene: Any) -> tuple[np.ndarray, dict[str, Any]]:
    """Actual current SO-101 joint angles: concat(left[6], right[6]) = 12-D.

    These are the true articulation state (physics-resolved), NOT commanded
    targets and NOT a clock. Falls back to the commanded default if joint_pos is
    unavailable, and is right-padded/truncated to 6 per arm defensively.
    """

    def _one(arm: Any) -> tuple[np.ndarray, bool]:
        for attr in ("joint_pos", "default_joint_pos"):
            try:
                q = np.asarray(getattr(arm.data, attr)[0].detach().cpu().numpy(), dtype=np.float32).reshape(-1)
                out = np.zeros(6, dtype=np.float32)
                out[: min(6, q.shape[0])] = q[:6]
                return out, bool(attr == "joint_pos")
            except Exception:  # noqa: BLE001
                continue
        return np.zeros(6, dtype=np.float32), False

    left, l_ok = _one(scene.left_arm)
    right, r_ok = _one(scene.right_arm)
    return np.concatenate([left, right]).astype(np.float32), {"joint_pos_available": bool(l_ok and r_ok)}


def target_edge_center(points: np.ndarray, edge: str = TARGET_EDGE, band: float = EDGE_BAND) -> np.ndarray:
    """World-frame center of the target towel edge (mean of edge-band particles)."""
    pts = np.asarray(points, dtype=np.float32)
    mask = towel_edge_masks(pts, band).get(edge)
    if mask is None or not np.any(mask):
        return np.zeros(3, dtype=np.float32)
    return pts[mask].mean(axis=0).astype(np.float32)


def compact_state_reactive(
    points: np.ndarray,
    joint_positions: np.ndarray,
    jaw_positions: np.ndarray,
    jaw_velocity: np.ndarray,
    prev_action: np.ndarray,
    edge: str = TARGET_EDGE,
    band: float = EDGE_BAND,
) -> tuple[np.ndarray, list[str]]:
    """Assemble the V16 state-reactive observation. Returns (state_vec, names).

    Args:
      points:          (N,3) cloth particle positions this step.
      joint_positions: (12,) actual joint angles (left[6]+right[6]).
      jaw_positions:   (6,) world jaw pos (left xyz + right xyz).
      jaw_velocity:    (3,) recent finite-diff world velocity of the acting jaw.
      prev_action:     (12,) last commanded joint targets (action history).
      edge:            target edge for the relative jaw-to-edge feature.
    """
    joint_positions = np.asarray(joint_positions, dtype=np.float32).reshape(-1)[:12]
    joint_positions = np.pad(joint_positions, (0, max(0, 12 - joint_positions.shape[0])))[:12]
    jaw_positions = np.asarray(jaw_positions, dtype=np.float32).reshape(-1)[:6]
    jaw_positions = np.pad(jaw_positions, (0, max(0, 6 - jaw_positions.shape[0])))[:6]
    jaw_velocity = np.asarray(jaw_velocity, dtype=np.float32).reshape(-1)[:3]
    jaw_velocity = np.pad(jaw_velocity, (0, max(0, 3 - jaw_velocity.shape[0])))[:3]
    prev_action = np.asarray(prev_action, dtype=np.float32).reshape(-1)[:12]
    prev_action = np.pad(prev_action, (0, max(0, 12 - prev_action.shape[0])))[:12]

    metrics = size_metrics(points)
    cloth_metrics = np.asarray(
        metrics["min"] + metrics["max"] + metrics["center"] + metrics["size"], dtype=np.float32
    )
    corner_edge, corner_edge_names = corner_and_edge_features(points)

    right_jaw = jaw_positions[3:6]
    edge_center = target_edge_center(points, edge, band)
    rel_jaw_edge = (right_jaw - edge_center).astype(np.float32)

    state_parts = [
        joint_positions,   # 12
        jaw_positions,     # 6
        jaw_velocity,      # 3
        prev_action,       # 12
        cloth_metrics,     # 12
        corner_edge,       # 24
        rel_jaw_edge,      # 3
    ]
    names = (
        [f"jointpos_{n}" for n in ACTION_NAMES]
        + ["left_jaw_x", "left_jaw_y", "left_jaw_z", "right_jaw_x", "right_jaw_y", "right_jaw_z"]
        + ["right_jaw_vx", "right_jaw_vy", "right_jaw_vz"]
        + [f"prevact_{n}" for n in ACTION_NAMES]
        + ["min_x", "min_y", "min_z", "max_x", "max_y", "max_z", "center_x", "center_y", "center_z", "size_x", "size_y", "size_z"]
        + corner_edge_names
        + ["rel_jaw_edge_x", "rel_jaw_edge_y", "rel_jaw_edge_z"]
    )
    state = np.concatenate(state_parts).astype(np.float32)
    assert state.shape[0] == len(names), (state.shape[0], len(names))
    return state, names


def state_reactive_dim() -> int:
    """Fixed length of the V16 observation (12+6+3+12+12+24+3)."""
    return 72
