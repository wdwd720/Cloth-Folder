"""Robot<-table rigid transform calibration (Umeyama / Kabsch, pure numpy).

The table homography places perception in the *table* frame; to actually hover
the arm over a table point we also need the rigid transform from table
coordinates into the *robot* base frame. This module:

* :func:`umeyama` — the least-squares rigid (optionally similarity) alignment of
  two matched point sets, pure numpy.
* :func:`robot_touch_calibration` — a SAFE, dry-run-by-default operator workflow
  for collecting table<->robot touch correspondences. It never drives the arm
  autonomously: the operator hand-guides a torque-off arm to known table points
  and the tip pose is *read*. Real mode is gated by the two-flag motion check AND
  a confirmed protocol; it still only reads.
* :func:`fit_robot_table_transform` — fit and persist the robot<-table transform
  from collected touch points, with hover/contact validity flags.

No serial protocol is invented anywhere here; nothing in this module sends a
servo write.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from terafold.calibration.table_homography import _dump_yaml
from terafold.data.episode_schema import read_json, write_json
from terafold.robot.safety import SafetyError, require_motion_enabled

__all__ = [
    "umeyama",
    "robot_touch_calibration",
    "fit_robot_table_transform",
]


# ----------------------------------------------------------------------
# Rigid alignment
# ----------------------------------------------------------------------


def _pad3(a: Any) -> np.ndarray:
    """Coerce points to ``(N, 3)``, padding a missing z column with zeros."""
    arr = np.asarray(a, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"points must be 2D (N, 2|3), got shape {arr.shape}")
    if arr.shape[1] == 2:
        arr = np.concatenate([arr, np.zeros((arr.shape[0], 1))], axis=1)
    elif arr.shape[1] != 3:
        raise ValueError(f"points must be Nx2 or Nx3, got {arr.shape}")
    return arr


def umeyama(
    src: Any, dst: Any, with_scale: bool = False
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Least-squares rigid/similarity alignment mapping ``src`` -> ``dst``.

    Solves for ``R, t, s`` minimizing ``sum_i || dst_i - (s * R @ src_i + t) ||^2``
    using the Umeyama/Kabsch closed form (SVD), with a reflection guard so ``R``
    is always a proper rotation. ``src``/``dst`` are ``Nx3`` or ``Nx2`` (2D is
    padded with ``z = 0``).

    Returns
    -------
    (R, t, s):
        ``R`` is ``3x3``, ``t`` is ``(3,)``, ``s`` is a float (``1.0`` when
        ``with_scale`` is False).
    """
    s_pts = _pad3(src)
    d_pts = _pad3(dst)
    if s_pts.shape != d_pts.shape:
        raise ValueError(f"src {s_pts.shape} and dst {d_pts.shape} must match")
    n, dim = s_pts.shape
    if n < 1:
        raise ValueError("need >= 1 point pair")

    mu_src = s_pts.mean(axis=0)
    mu_dst = d_pts.mean(axis=0)
    src_c = s_pts - mu_src
    dst_c = d_pts - mu_dst

    cov = (dst_c.T @ src_c) / n  # (dim, dim)
    U, D, Vt = np.linalg.svd(cov)

    S = np.eye(dim)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1.0

    R = U @ S @ Vt
    if with_scale:
        var_src = (src_c ** 2).sum() / n
        s = float((D * np.diag(S)).sum() / var_src) if var_src > 1e-18 else 1.0
    else:
        s = 1.0
    t = mu_dst - s * (R @ mu_src)
    return R, t, float(s)


# ----------------------------------------------------------------------
# Operator touch-point collection (SAFE: never moves hardware autonomously)
# ----------------------------------------------------------------------


def _normalize_points(points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    norm: List[Dict[str, Any]] = []
    for p in points:
        txy = [float(v) for v in p["table_xy"]]
        rxyz = [float(v) for v in p["robot_xyz"]]
        if len(txy) != 2:
            raise ValueError("each table_xy must have 2 components")
        if len(rxyz) != 3:
            raise ValueError("each robot_xyz must have 3 components")
        norm.append({"table_xy": txy, "robot_xyz": rxyz})
    return norm


def robot_touch_calibration(
    robot: str,
    joint_map: Any = None,
    out: Optional[str] = None,
    dry_run: bool = True,
    enable_motion: bool = False,
    acknowledge: bool = False,
    backend: Any = None,
    points: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Plan / record manual table<->robot touch correspondences (dry-run default).

    This is a MANUAL, torque-off procedure: the operator hand-guides the arm to
    known table markers and the tip pose is read. The robot is never driven
    autonomously and no servo write is ever sent.

    Behaviour
    ---------
    * ``points`` given: persist the recorded pairs to ``out`` (JSON) and return.
    * default (``dry_run=True``): return a step-by-step plan; touches nothing.
    * ``dry_run=False`` (real mode): require BOTH motion flags AND a confirmed
      protocol; refuse with a clear dict otherwise. Even when armed it only reads.
    """
    robot_name = robot
    dof: Optional[int] = None
    try:
        from terafold.robot.arm_config import load_arm_config

        cfg = load_arm_config(robot)
        robot_name = cfg.robot_name
        dof = int(cfg.dof)
    except Exception:
        cfg = None

    # 1) If pre-recorded pairs are supplied, just persist them.
    if points is not None:
        recorded = _normalize_points(points)
        payload = {
            "robot": robot_name,
            "num_points": len(recorded),
            "points": recorded,
            "note": "operator-recorded table<->robot touch points (manual, torque-off).",
        }
        saved: Optional[str] = None
        if out:
            write_json(out, payload)
            saved = out
        return {
            "status": "saved",
            "moves_hardware": False,
            "robot": robot_name,
            "num_points": len(recorded),
            "points": recorded,
            "out": saved,
        }

    procedure = [
        "Place >=3 (ideally >=4) markers at table points whose table (x, y) you know.",
        "Put the arm in a torque-off / hand-guided mode (NO autonomous motion).",
        "For each marker: by HAND move the tool tip to touch it and hold steady.",
        "Read the robot tip (x, y, z) for that pose and pair it with the marker (x, y).",
        "Spread the markers across the workspace for a well-conditioned fit.",
        "Save the pairs, then run fit-robot-table-transform to solve robot<-table.",
    ]
    base: Dict[str, Any] = {
        "robot": robot_name,
        "dof": dof,
        "moves_hardware": False,
        "joint_map_present": joint_map is not None,
        "procedure": procedure,
        "output_format": {
            "points": [{"table_xy": [0.0, 0.0], "robot_xyz": [0.0, 0.0, 0.0]}]
        },
        "next": "fit_robot_table_transform(touch_points_path, out)",
    }

    # 2) Dry-run plan (default).
    if dry_run:
        base["status"] = "dry_run"
        base["note"] = (
            "DRY-RUN plan only: nothing was sent to hardware. This is a MANUAL, "
            "torque-off touch calibration; the arm is never driven autonomously."
        )
        return base

    # 3) Real mode: two-flag gate AND confirmed protocol. Refuse clearly otherwise.
    try:
        require_motion_enabled(enable_motion, acknowledge)
    except SafetyError as exc:
        return {**base, "status": "refused", "allowed": False, "reason": str(exc)}

    if not bool(getattr(backend, "protocol_confirmed", False)):
        return {
            **base,
            "status": "refused",
            "allowed": False,
            "reason": (
                "protocol not confirmed; a read-only probe must confirm the servo "
                "protocol before any armed procedure (run robot-status --probe)."
            ),
        }

    base["status"] = "armed"
    base["allowed"] = True
    base["note"] = (
        "ARMED manual touch calibration: even armed, this only READS tip poses "
        "while you hand-guide the torque-off arm. No servo writes are sent."
    )
    return base


# ----------------------------------------------------------------------
# Fit + persist the transform
# ----------------------------------------------------------------------


def fit_robot_table_transform(
    touch_points_path: str,
    out: str,
    valid_threshold_m: float = 0.005,
    timestamp: float = 0.0,
) -> Dict[str, Any]:
    """Fit and persist the robot<-table rigid transform from touch points.

    Reads ``{"points": [{"table_xy": [x, y], "robot_xyz": [x, y, z]}, ...]}``
    (``>= 3`` points required; raises :class:`ValueError` otherwise), fits the
    transform with :func:`umeyama`, and writes a YAML artifact carrying ``R``,
    ``t``, the reprojection errors, and hover/contact validity flags.

    Apply the result as ``robot = scale * R @ [x, y, 0] + t``.
    """
    data = read_json(touch_points_path)
    points = data.get("points") if isinstance(data, dict) else data
    if not points or len(points) < 3:
        n = 0 if not points else len(points)
        raise ValueError(f"need >= 3 touch points to fit a transform, got {n}")

    table = np.array(
        [[float(p["table_xy"][0]), float(p["table_xy"][1])] for p in points],
        dtype=np.float64,
    )
    robot = np.array(
        [
            [float(p["robot_xyz"][0]), float(p["robot_xyz"][1]), float(p["robot_xyz"][2])]
            for p in points
        ],
        dtype=np.float64,
    )

    R, t, s = umeyama(table, robot, with_scale=False)
    pred = s * (_pad3(table) @ R.T) + t  # robot = s*R@src + t (row-vector form)
    err = np.sqrt(((pred - robot) ** 2).sum(axis=1))
    rms = float(np.sqrt((err ** 2).mean()))
    max_err = float(err.max())
    thr = float(valid_threshold_m)

    record: Dict[str, Any] = {
        "type": "robot_table_transform",
        "R": [[float(v) for v in row] for row in R],
        "t": [float(v) for v in t],
        "scale": float(s),
        "num_points": int(len(points)),
        "rms_error_m": rms,
        "max_error_m": max_err,
        "valid_threshold_m": thr,
        "valid_for_hover": bool(rms <= thr),
        "valid_for_contact": bool(rms <= thr * 0.6),
        "timestamp": float(timestamp),
        "note": "robot<-table rigid transform (Umeyama). Apply: robot = scale*R@[x,y,0]+t.",
    }
    _dump_yaml(out, record)
    result = dict(record)
    result["out"] = out
    return result
