"""Train the learned residual correction from recorded demonstrations.

The residual learns the *systematic* gap between what the geometric planner
proposed (stored in each episode's ``plan.json``) and what was actually
executed (the ``actions.jsonl`` end-effector poses, e.g. from a teleoperated or
human-corrected demo). Targets are simple additive deltas:

    target = executed - geometric   (grasp/place xy, lift/arc/release heights)

plus per-episode ``confidence`` (1.0 for any recorded demo) and
``predicted_success`` (1.0 if the episode succeeded, else 0.0).

Torch is required to *train*; it is imported lazily so this module imports under
numpy alone. If no episodes yield usable targets, a clear error is raised.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from terafold.planning.residual_model import (
    RESIDUAL_FEATURE_NAMES,
    RESIDUAL_OUTPUT_NAMES,
    ResidualMLP,
    build_residual_features,
)

__all__ = ["train_residual", "build_residual_dataset"]

# Trajectory phases (see terafold.planning.trajectory.FOLD_PHASES) that mark
# where the grasp and place actually happen during execution.
_GRASP_PHASES = ("close", "descend", "pregrasp")
_PLACE_PHASES = ("release", "place")
_LIFT_PHASES = ("lift", "tension_lift")
_ARC_PHASES = ("arc",)


def _xyz_at_phase(
    waypoints: np.ndarray, phases: List[str], wanted: Tuple[str, ...]
) -> Optional[np.ndarray]:
    """Return the first waypoint xyz whose phase is in ``wanted``."""
    for i, ph in enumerate(phases):
        if ph in wanted and i < len(waypoints):
            return np.asarray(waypoints[i], dtype=np.float64).reshape(-1)[:3]
    return None


def _max_z_at_phases(
    waypoints: np.ndarray, phases: List[str], wanted: Tuple[str, ...]
) -> Optional[float]:
    zs = [
        float(waypoints[i][2])
        for i, ph in enumerate(phases)
        if ph in wanted and i < len(waypoints)
    ]
    return max(zs) if zs else None


def _executed_waypoints(actions: List[dict]) -> np.ndarray:
    """Extract executed end-effector xyz from recorded actions."""
    pts = []
    for a in actions:
        pose = a.get("target_ee_pose")
        if pose is None:
            continue
        pose = np.asarray(pose, dtype=np.float64).reshape(-1)
        if pose.size >= 3:
            pts.append(pose[:3])
    return np.asarray(pts, dtype=np.float64) if pts else np.zeros((0, 3))


def _episode_target(
    plan_dict: dict, actions: List[dict], success: Optional[bool]
) -> Optional[Tuple[Dict[str, float], np.ndarray]]:
    """Build ``(plan_features, target_vector)`` for one episode, or None.

    The target vector follows :data:`RESIDUAL_OUTPUT_NAMES`. ``confidence`` and
    ``predicted_success`` are stored as *probabilities* (0/1); the trainer maps
    them to logits internally.
    """
    features = plan_dict.get("features", {})
    if not features:
        return None

    geom_grasp = np.array(
        [features.get("grasp_x", 0.0), features.get("grasp_y", 0.0)], dtype=np.float64
    )
    geom_place = np.array(
        [features.get("place_x", 0.0), features.get("place_y", 0.0)], dtype=np.float64
    )
    geom_lift = float(features.get("lift_height_m", 0.0))
    geom_arc = float(features.get("arc_height_m", 0.0))
    geom_release = float(features.get("release_height_m", 0.0))

    traj = plan_dict.get("trajectory", {})
    phases = list(traj.get("phases", []))
    planned_wp = np.asarray(traj.get("waypoints", []), dtype=np.float64)
    exec_wp = _executed_waypoints(actions)

    if exec_wp.shape[0] == 0 or planned_wp.shape[0] == 0 or not phases:
        return None

    # Align executed waypoints to planned phases by index (the recorder writes
    # one action per trajectory waypoint). Trim to the common length.
    n = min(len(phases), planned_wp.shape[0], exec_wp.shape[0])
    if n == 0:
        return None
    phases = phases[:n]
    exec_wp = exec_wp[:n]
    planned_wp = planned_wp[:n]

    # The residual is defined relative to the PLANNED trajectory (which already
    # includes slip compensation), NOT the raw pre-slip feature points. Using
    # the planned waypoints as the geometric baseline guarantees the target is
    # ~0 for a demo that executed the plan unchanged — otherwise slip
    # compensation would be double-counted (planner adds it, residual adds it
    # again at inference). Fall back to the feature values if a phase waypoint
    # is missing.
    grasp_geom_wp = _xyz_at_phase(planned_wp, phases, _GRASP_PHASES)
    place_geom_wp = _xyz_at_phase(planned_wp, phases, _PLACE_PHASES)
    lift_geom_wp = _max_z_at_phases(planned_wp, phases, _LIFT_PHASES)
    arc_geom_wp = _max_z_at_phases(planned_wp, phases, _ARC_PHASES)
    grasp_base = grasp_geom_wp[:2] if grasp_geom_wp is not None else geom_grasp
    place_base = place_geom_wp[:2] if place_geom_wp is not None else geom_place
    lift_base = float(lift_geom_wp) if lift_geom_wp is not None else geom_lift
    arc_base = float(arc_geom_wp) if arc_geom_wp is not None else geom_arc
    release_base = float(place_geom_wp[2]) if place_geom_wp is not None else geom_release

    grasp_exec = _xyz_at_phase(exec_wp, phases, _GRASP_PHASES)
    place_exec = _xyz_at_phase(exec_wp, phases, _PLACE_PHASES)
    lift_exec = _max_z_at_phases(exec_wp, phases, _LIFT_PHASES)
    arc_exec = _max_z_at_phases(exec_wp, phases, _ARC_PHASES)
    release_exec = _xyz_at_phase(exec_wp, phases, _PLACE_PHASES)

    if grasp_exec is None and place_exec is None:
        return None

    grasp_dxy = (grasp_exec[:2] - grasp_base) if grasp_exec is not None else np.zeros(2)
    place_dxy = (place_exec[:2] - place_base) if place_exec is not None else np.zeros(2)
    lift_delta = (lift_exec - lift_base) if lift_exec is not None else 0.0
    arc_delta = (arc_exec - arc_base) if arc_exec is not None else 0.0
    release_delta = (
        (float(release_exec[2]) - release_base) if release_exec is not None else 0.0
    )

    conf = 1.0
    succ = 1.0 if success else (0.0 if success is False else 0.5)

    target = np.array(
        [
            grasp_dxy[0],
            grasp_dxy[1],
            place_dxy[0],
            place_dxy[1],
            lift_delta,
            arc_delta,
            release_delta,
            conf,
            succ,
        ],
        dtype=np.float64,
    )
    return features, target


def build_residual_dataset(
    episodes_dir: str, task: Any
) -> Tuple[np.ndarray, np.ndarray]:
    """Scan recorded episodes and build ``(X, Y)`` arrays.

    ``X`` is ``(M, len(RESIDUAL_FEATURE_NAMES))`` and ``Y`` is
    ``(M, len(RESIDUAL_OUTPUT_NAMES))``. Raises if nothing usable is found.
    """
    # Lazy import: the episode schema is numpy-only, but keep the dependency
    # local so this module's import surface stays minimal.
    from terafold.data.episode_schema import list_episode_dirs, read_json, read_jsonl

    episode_paths = list_episode_dirs(episodes_dir)
    if not episode_paths:
        raise FileNotFoundError(
            f"No episode directories found under {episodes_dir!r}. Record "
            "demonstrations first (terafold.data.recorder.record_demo)."
        )

    feats: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    for ep in episode_paths:
        if not os.path.exists(ep.plan) or not os.path.exists(ep.actions):
            continue
        try:
            plan_dict = read_json(ep.plan)
            actions = list(read_jsonl(ep.actions))
        except (OSError, json.JSONDecodeError):
            continue

        success: Optional[bool] = None
        if os.path.exists(ep.result):
            try:
                success = read_json(ep.result).get("success")
            except (OSError, json.JSONDecodeError):
                success = None

        out = _episode_target(plan_dict, actions, success)
        if out is None:
            continue
        features, target = out
        feats.append(build_residual_features(features, task))
        targets.append(target)

    if not feats:
        raise ValueError(
            f"Found episodes under {episodes_dir!r} but none yielded usable residual "
            "targets. Each episode needs a plan.json (with features + trajectory "
            "phases) and an actions.jsonl with executed target_ee_pose entries. "
            "Record teleoperated/corrected demonstrations before training."
        )

    return np.asarray(feats, dtype=np.float64), np.asarray(targets, dtype=np.float64)


def _resolve_task(task: Any):
    """Return a FoldTask, defaulting to the package default config if None."""
    if task is not None:
        return task
    from terafold.config.schema import TaskConfig
    from terafold.planning.fold_task import FoldTask

    return FoldTask.from_config(TaskConfig())


def train_residual(
    episodes_dir: str,
    out_dir: str,
    epochs: int = 100,
    lr: float = 1e-3,
    device: Optional[str] = None,
    task: Any = None,
    batch_size: int = 32,
    hidden: Tuple[int, ...] = (64, 64),
    seed: int = 0,
) -> Dict[str, Any]:
    """Train a :class:`ResidualMLP` from recorded episodes.

    Saves ``<out_dir>/model.pt`` (state_dict + config) and
    ``<out_dir>/config.json``. Requires torch; raises a clear ``ImportError``
    with install guidance if it is missing, and a clear error if no usable
    training targets are found.
    """
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required to train the residual model. Install it with "
            "`pip install torch` (or `pip install -e '.[torch]'`)."
        ) from exc

    task = _resolve_task(task)
    X_np, Y_np = build_residual_dataset(episodes_dir, task)

    dev = torch.device(device) if device else torch.device("cpu")
    torch.manual_seed(seed)

    X = torch.as_tensor(X_np, dtype=torch.float32, device=dev)
    # Split targets: first 7 are regression deltas, last 2 are probabilities ->
    # supervised as logits via BCEWithLogits (so they pair with the raw head).
    Y_reg = torch.as_tensor(Y_np[:, :7], dtype=torch.float32, device=dev)
    Y_prob = torch.as_tensor(Y_np[:, 7:9], dtype=torch.float32, device=dev)

    in_dim = X.shape[1]
    out_dim = len(RESIDUAL_OUTPUT_NAMES)
    model = ResidualMLP(in_dim=in_dim, hidden=hidden, out_dim=out_dim).to(dev)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()

    n = X.shape[0]
    bs = max(1, min(batch_size, n))
    rng = np.random.default_rng(seed)
    history: List[float] = []

    model.train()
    for _ in range(int(epochs)):
        perm = rng.permutation(n)
        epoch_loss = 0.0
        nb = 0
        for start in range(0, n, bs):
            idx = perm[start : start + bs]
            ti = torch.as_tensor(idx, dtype=torch.long, device=dev)
            pred = model(X.index_select(0, ti))
            loss = mse(pred[:, :7], Y_reg.index_select(0, ti)) + bce(
                pred[:, 7:9], Y_prob.index_select(0, ti)
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item())
            nb += 1
        history.append(epoch_loss / max(1, nb))

    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "model.pt")
    config = {
        "in_dim": int(in_dim),
        "hidden": list(hidden),
        "out_dim": int(out_dim),
        "feature_names": RESIDUAL_FEATURE_NAMES,
        "output_names": RESIDUAL_OUTPUT_NAMES,
        "num_samples": int(n),
        "epochs": int(epochs),
        "lr": float(lr),
    }
    torch.save({"state_dict": model.state_dict(), "config": config}, model_path)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    return {
        "model_path": model_path,
        "config": config,
        "num_samples": int(n),
        "final_loss": history[-1] if history else None,
        "history": history,
    }
