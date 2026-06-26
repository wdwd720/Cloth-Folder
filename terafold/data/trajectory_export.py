"""Export a fold trajectory to CSV/JSON for manual / vendor-software testing.

If we can't command the arm yet, we can still export the planned waypoints so a
human can replay or compare them in the vendor's software, or use them later to
implement the protocol. Columns: waypoint, x, y, z, gripper, speed, timestamp, phase.
"""

from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, List

import numpy as np

__all__ = ["export_trajectory", "rows_from_trajectory_dict", "write_rows", "load_trajectory_dict"]

_FIELDS = ["waypoint", "x", "y", "z", "gripper", "speed", "timestamp", "phase"]


def rows_from_trajectory_dict(traj: Dict[str, Any]) -> List[dict]:
    """Build canonical export rows from a ``FoldTrajectory.to_dict()`` payload."""
    wp = np.asarray(traj.get("waypoints", []), dtype=np.float64)
    grip = np.asarray(traj.get("gripper", []), dtype=np.float64)
    times = np.asarray(traj.get("times", []), dtype=np.float64)
    phases = list(traj.get("phases", []))
    rows: List[dict] = []
    for i in range(wp.shape[0]):
        speed = 0.0
        if i > 0:
            dt = float(times[i] - times[i - 1]) if i < len(times) else 0.0
            if dt > 1e-9:
                speed = float(np.linalg.norm(wp[i] - wp[i - 1]) / dt)
        rows.append(
            {
                "waypoint": i,
                "x": float(wp[i, 0]), "y": float(wp[i, 1]), "z": float(wp[i, 2]),
                "gripper": float(grip[i]) if i < len(grip) else 1.0,
                "speed": speed,
                "timestamp": float(times[i]) if i < len(times) else 0.0,
                "phase": phases[i] if i < len(phases) else "",
            }
        )
    return rows


def _normalize_row(row: dict, index: int, t_accum: float) -> dict:
    """Accept either a canonical row or an adapter-log row ({ee_pose, duration})."""
    if "x" in row and "y" in row and "z" in row:
        return {
            "waypoint": row.get("waypoint", index),
            "x": float(row["x"]), "y": float(row["y"]), "z": float(row["z"]),
            "gripper": float(row.get("gripper", 1.0)),
            "speed": float(row.get("speed", 0.0)),
            "timestamp": float(row.get("timestamp", t_accum)),
            "phase": row.get("phase", ""),
        }
    pose = np.asarray(row.get("ee_pose", [0, 0, 0]), dtype=np.float64).reshape(-1)
    return {
        "waypoint": index,
        "x": float(pose[0]), "y": float(pose[1]), "z": float(pose[2]),
        "gripper": float(row.get("gripper", 1.0)),
        "speed": float(row.get("speed", 0.0)),
        "timestamp": t_accum,
        "phase": row.get("phase", ""),
    }


def write_rows(rows: List[dict], out: str, fmt: str = "csv") -> str:
    """Write rows (canonical or adapter-log shaped) to CSV or JSON. Returns the path."""
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    t_accum = 0.0
    norm: List[dict] = []
    for i, r in enumerate(rows):
        nr = _normalize_row(r, i, t_accum)
        norm.append(nr)
        t_accum = nr["timestamp"] + float(r.get("duration", 0.0))
    if fmt.lower() == "json":
        with open(out, "w") as f:
            json.dump({"trajectory": norm, "fields": _FIELDS}, f, indent=2)
    else:
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_FIELDS)
            writer.writeheader()
            for nr in norm:
                writer.writerow(nr)
    return out


def load_trajectory_dict(plan_json_path: str) -> Dict[str, Any]:
    """Find the trajectory dict inside a saved plan / demo-result JSON."""
    with open(plan_json_path) as f:
        data = json.load(f)
    for getter in (
        lambda d: d.get("trajectory"),
        lambda d: (d.get("plan") or {}).get("trajectory"),
        lambda d: (d.get("fold_plan") or {}).get("trajectory"),
    ):
        traj = getter(data)
        if isinstance(traj, dict) and "waypoints" in traj:
            return traj
    raise ValueError(
        f"No trajectory found in {plan_json_path}. Expected a FoldPlan JSON or a "
        "demo result with a 'plan.trajectory'."
    )


def export_trajectory(plan_json: str, out: str, fmt: str = "csv") -> Dict[str, Any]:
    """Read a plan/result JSON and export its trajectory to ``out`` (CSV by default)."""
    traj = load_trajectory_dict(plan_json)
    rows = rows_from_trajectory_dict(traj)
    path = write_rows(rows, out, fmt=fmt)
    return {"status": "ok", "rows": len(rows), "out": path, "format": fmt}
