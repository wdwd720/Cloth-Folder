"""Simulation preview of the image-driven, air-only GHOST fold.

SIMULATION ONLY — this never opens a serial port and never sends a motor command.
It loads a fold plan (a :class:`~terafold.planning.fold_plan.FoldPlan` / demo
result JSON), lifts the planned Cartesian trajectory into the air via
:func:`terafold.robot.real_motion.ghost_fold_core`, and renders a simple TOP-DOWN
2D preview (towel rectangle, grasp/place, fold arrow, table bounds) so an operator
can eyeball the fold *before* any hardware is ever involved.

Hard invariants this module advertises in every result:

* ``contact_disabled`` is **always** ``True`` — the preview is air-only, the
  gripper never closes hard, and contact folding stays LOCKED.
* ``missing_kinematics`` is **always** ``True`` — the custom arm has no validated
  kinematics, so the Cartesian waypoints are a preview, not an executable path.
* ``touches_table`` is ``False`` — every waypoint is lifted to at least the
  clearance above the table plane (``table_plane_z = 0.0``).

``matplotlib`` is the only optional dependency and is **lazy-imported**: if it is
missing the PNG is skipped (``rendered=False`` with a clear note) but the metadata
JSON is always written and the metadata dict is always returned.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional, Tuple

from terafold.data.episode_schema import write_json
from terafold.robot.joint_map import JointMap, load_joint_map
from terafold.robot.real_motion import _load_plan, ghost_fold_core

__all__ = ["run_sim_real_ghost_fold"]

# The simulated table plane (top-down preview lives in this z=0 plane).
TABLE_PLANE_Z = 0.0

# Default output location when the caller passes no ``out`` (discoverable by ops).
_DEFAULT_OUT = "runs/sim_ghost_preview/sim_real_ghost_fold.png"

# Corner keys, in draw order, that make up the towel rectangle.
_CORNER_KEYS = ("top_left", "top_right", "bottom_right", "bottom_left")

# Recognized image-derived fold directions (mirrors real_motion._DIR_SWEEP).
_KNOWN_DIRECTIONS = ("right_to_left", "left_to_right", "top_to_bottom", "bottom_to_top")


def _noop(_m: str) -> None:
    pass


# ----------------------------------------------------------------------
# Plan parsing (tolerant — works on FoldPlan dicts, demo results, and the
# minimal ``{"trajectory": {...}}`` shape used by tests).
# ----------------------------------------------------------------------


def _resolve_plan(plan_json: Any) -> Dict[str, Any]:
    """Return the inner plan dict from a path or an already-loaded dict."""
    if isinstance(plan_json, dict):
        d = plan_json
        if "trajectory" in d:
            return d
        for k in ("plan", "fold_plan"):
            if isinstance(d.get(k), dict):
                return d[k]
        return d
    return _load_plan(str(plan_json))


def _coerce_xy(v: Any) -> Optional[List[float]]:
    """Best-effort ``[x, y]`` from a sequence; ``None`` if it cannot be read."""
    try:
        return [float(v[0]), float(v[1])]
    except (TypeError, ValueError, IndexError):
        return None


def _trajectory(plan: Dict[str, Any]) -> Tuple[List[Any], Optional[List[Any]], Optional[List[Any]]]:
    """Pull ``(waypoints, gripper, phases)`` out of the plan's trajectory dict."""
    traj = plan.get("trajectory") if isinstance(plan, dict) else None
    if not isinstance(traj, dict):
        return [], None, None
    return (traj.get("waypoints") or [], traj.get("gripper"), traj.get("phases"))


def _find_keypoints(plan: Dict[str, Any]) -> Dict[str, Any]:
    """Locate a keypoints dict across the known plan shapes."""
    if not isinstance(plan, dict):
        return {}
    kp = plan.get("keypoints")
    if isinstance(kp, dict):
        return kp
    for sub in ("fold_state", "image_fold_state"):
        s = plan.get(sub)
        if isinstance(s, dict) and isinstance(s.get("keypoints"), dict):
            return s["keypoints"]
    return {}


def _corners(plan: Dict[str, Any]) -> Optional[List[List[float]]]:
    """The four towel corners (table frame) if all are present, else ``None``."""
    kp = _find_keypoints(plan)
    if not isinstance(kp, dict):
        return None
    pts: List[List[float]] = []
    for k in _CORNER_KEYS:
        xy = _coerce_xy(kp.get(k))
        if xy is None:
            return None
        pts.append(xy)
    return pts


def _grasp_place(plan: Dict[str, Any]) -> Tuple[Optional[List[float]], Optional[List[float]]]:
    """Grasp/place points from the grasp_place block, top-level, or keypoints."""
    grasp = place = None
    gp = plan.get("grasp_place") if isinstance(plan, dict) else None
    if isinstance(gp, dict):
        grasp = _coerce_xy(gp.get("grasp"))
        place = _coerce_xy(gp.get("place"))
    if isinstance(plan, dict):
        grasp = grasp or _coerce_xy(plan.get("grasp"))
        place = place or _coerce_xy(plan.get("place"))
    kp = _find_keypoints(plan)
    if isinstance(kp, dict):
        grasp = grasp or _coerce_xy(kp.get("grasp"))
        place = place or _coerce_xy(kp.get("place"))
    return grasp, place


def _plan_direction(plan: Dict[str, Any]) -> Optional[str]:
    """Image-derived fold direction from metadata, top-level, or the task name."""
    if not isinstance(plan, dict):
        return None
    md = plan.get("metadata")
    if isinstance(md, dict) and md.get("direction"):
        return str(md["direction"])
    if plan.get("direction"):
        return str(plan["direction"])
    traj = plan.get("trajectory")
    if isinstance(traj, dict):
        tmd = traj.get("metadata")
        if isinstance(tmd, dict) and tmd.get("direction"):
            return str(tmd["direction"])
    name = str(plan.get("task") or plan.get("task_name") or "")
    for d in _KNOWN_DIRECTIONS:
        if d in name:
            return d
    return None


def _calibrated(plan: Dict[str, Any]) -> bool:
    """True only if the plan clearly carries a positive ``calibrated`` flag."""
    if not isinstance(plan, dict):
        return False
    if bool(plan.get("calibrated")):
        return True
    ts = plan.get("table_space")
    if isinstance(ts, dict) and bool(ts.get("calibrated")):
        return True
    return False


def _normalize_path(pts: List[List[float]]) -> List[List[float]]:
    """Min-max normalize a 2D path into ``[0, 1]^2`` within its own bbox."""
    if not pts:
        return []
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    dx = (xmax - xmin) or 1.0
    dy = (ymax - ymin) or 1.0
    return [[round((x - xmin) / dx, 4), round((y - ymin) / dy, 4)] for x, y in pts]


def _table_bounds(points: List[List[float]], margin: float = 0.05) -> Optional[List[float]]:
    """An ``[xmin, ymin, xmax, ymax]`` bbox around all points, with margin."""
    pts = [p for p in points if p is not None]
    if not pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return [round(min(xs) - margin, 4), round(min(ys) - margin, 4),
            round(max(xs) + margin, 4), round(max(ys) + margin, 4)]


# ----------------------------------------------------------------------
# Output path helpers
# ----------------------------------------------------------------------


def _png_for(out: str) -> str:
    """The PNG target: ``out`` itself if it ends in .png, else swap the ext."""
    base, ext = os.path.splitext(out)
    return out if ext.lower() == ".png" else base + ".png"


def _ensure_parent(path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)


# ----------------------------------------------------------------------
# Rendering (matplotlib is OPTIONAL and lazy)
# ----------------------------------------------------------------------


def _render_preview(
    png_path: str,
    *,
    corners: Optional[List[List[float]]],
    grasp: Optional[List[float]],
    place: Optional[List[float]],
    fold_path_xy: List[List[float]],
    direction: Optional[str],
    table_bounds: Optional[List[float]],
) -> Tuple[bool, str]:
    """Render a top-down 2D preview PNG. Returns ``(rendered, note)``.

    If matplotlib is not importable the PNG is skipped gracefully and the metadata
    JSON (written by the caller) still fully describes the preview.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless / no display required
        import matplotlib.pyplot as plt
    except Exception as exc:  # ImportError or backend failure
        return False, (
            "matplotlib is not available (" + type(exc).__name__ + "): skipped the "
            "2D PNG preview. The metadata JSON was still written. Install "
            "matplotlib (`pip install matplotlib`) to get the rendered image."
        )

    _ensure_parent(png_path)
    fig, ax = plt.subplots(figsize=(6.0, 6.0))

    if table_bounds:
        xmin, ymin, xmax, ymax = table_bounds
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, linestyle=":", edgecolor="0.6",
                                   label="table bounds"))

    if corners:
        poly = list(corners) + [corners[0]]
        ax.plot([p[0] for p in poly], [p[1] for p in poly],
                "-", color="#1f77b4", linewidth=2.0, label="towel")
        ax.fill([p[0] for p in corners], [p[1] for p in corners],
                color="#1f77b4", alpha=0.10)

    if fold_path_xy:
        ax.plot([p[0] for p in fold_path_xy], [p[1] for p in fold_path_xy],
                "-", color="0.5", linewidth=1.0, alpha=0.8, label="ghost path (air)")

    if grasp and place:
        ax.annotate("", xy=(place[0], place[1]), xytext=(grasp[0], grasp[1]),
                    arrowprops=dict(arrowstyle="->", color="#d62728", linewidth=2.0))
    if grasp:
        ax.plot([grasp[0]], [grasp[1]], "o", color="#2ca02c", markersize=10, label="grasp")
    if place:
        ax.plot([place[0]], [place[1]], "X", color="#d62728", markersize=11, label="place")

    ax.set_aspect("equal", adjustable="datalim")
    ax.invert_yaxis()  # top-down image convention (origin top-left)
    ax.set_xlabel("table x")
    ax.set_ylabel("table y")
    ax.set_title("Ghost fold preview (SIM, air-only, no contact)\n"
                 f"direction={direction or 'unknown'}  contact=DISABLED")
    ax.legend(loc="best", fontsize=8, framealpha=0.7)
    fig.tight_layout()
    fig.savefig(png_path, dpi=110)
    plt.close(fig)
    return True, "rendered top-down 2D ghost-fold preview"


# ----------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------


def _resolve_joint_map(joint_map: Any) -> Optional[JointMap]:
    """Accept a :class:`JointMap`, a robot name, a YAML path, or ``None``."""
    if joint_map is None:
        return None
    if isinstance(joint_map, JointMap):
        return joint_map
    return load_joint_map(str(joint_map))


def run_sim_real_ghost_fold(
    plan_json: Any,
    joint_map: Any = None,
    out: Optional[str] = None,
    height_clearance_m: float = 0.10,
    fold_direction: Optional[str] = None,
    on_log: Callable[[str], None] = print,
) -> Dict[str, Any]:
    """Render a SIM top-down preview of the image-driven air-only ghost fold.

    Parameters
    ----------
    plan_json:
        Path to a fold-plan / demo-result JSON, or an already-loaded plan dict.
    joint_map:
        A :class:`JointMap`, a robot name, a joint-map YAML path, or ``None``.
        Used only to report the active servo ids — never to move anything.
    out:
        Destination for the PNG (``.mp4`` is accepted and re-pointed to ``.png``).
        The metadata JSON is written to ``out + ".meta.json"``.
    height_clearance_m:
        Minimum height above the table plane every waypoint is lifted to.
    fold_direction:
        Override the image-derived fold direction (defaults to the plan's).
    on_log:
        Line logger (defaults to :func:`print`).

    Returns
    -------
    dict
        The metadata dict, including ``status``, ``contact_disabled`` (always
        ``True``), ``missing_calibration``, ``missing_kinematics`` (always
        ``True``), ``num_waypoints``, ``min_z``, ``touches_table`` (``False``),
        ``fold_direction``, ``rendered``, ``meta_json`` and ``warnings``.
    """
    log = on_log or _noop

    plan = _resolve_plan(plan_json)
    waypoints, gripper, phases = _trajectory(plan)

    # Air-only ghost: lift every waypoint >= clearance above the table plane.
    ghost = ghost_fold_core(waypoints, gripper, phases,
                            height_clearance_m=float(height_clearance_m),
                            table_z=TABLE_PLANE_Z)

    direction = fold_direction or _plan_direction(plan)
    corners = _corners(plan)
    grasp, place = _grasp_place(plan)
    fold_path_xy = [[float(w["xyz"][0]), float(w["xyz"][1])] for w in ghost["waypoints"]]
    normalized_fold_path = _normalize_path(fold_path_xy)

    jm = _resolve_joint_map(joint_map)
    active_servo_ids = sorted(jm.mapped_ids()) if jm is not None else []

    missing_calibration = not _calibrated(plan)
    missing_kinematics = True  # custom arm: no validated kinematics, EVER here.

    # ---- warnings (always non-empty: kinematics + sim note are unconditional) ----
    warnings: List[str] = [
        "No validated kinematics for this custom arm: the Cartesian waypoints are a "
        "PREVIEW only and cannot be executed as real Cartesian motion.",
        "SIMULATION ONLY: no serial port is opened and no motor command is ever sent.",
    ]
    if missing_calibration:
        warnings.append(
            "No table calibration in the plan: the image->table mapping is ASSUMED "
            "and must not be trusted for real motion (sim preview only).")
    if jm is None:
        warnings.append(
            "No joint map provided: active servo ids are unknown. Run `map-servo-joints` "
            "before attempting any (still gated) joint-space ghost fold.")
    elif not active_servo_ids:
        warnings.append(
            "Joint map has no mapped servo ids: active servo ids are unknown.")
    if not fold_path_xy:
        warnings.append(
            "Plan trajectory had no waypoints: nothing to preview beyond the metadata.")
    if direction is None:
        warnings.append(
            "Could not derive the fold direction from the plan; preview shows the raw "
            "ghost path without a labeled direction.")

    out_path = out or _DEFAULT_OUT
    png_path = _png_for(out_path)
    meta_json = out_path + ".meta.json"

    bounds = _table_bounds(list(fold_path_xy) + (corners or []) +
                           [p for p in (grasp, place) if p is not None])

    rendered, render_note = _render_preview(
        png_path,
        corners=corners, grasp=grasp, place=place,
        fold_path_xy=fold_path_xy, direction=direction, table_bounds=bounds,
    )
    if not rendered:
        warnings.append(render_note)

    metadata: Dict[str, Any] = {
        "status": "ok" if fold_path_xy else "empty_trajectory",
        "simulation_only": True,
        "contact_disabled": True,        # ALWAYS — air-only ghost preview.
        "contact_fold_locked": True,
        "table_plane_z": TABLE_PLANE_Z,
        "height_clearance_m": float(height_clearance_m),
        "fold_direction": direction,
        "missing_calibration": bool(missing_calibration),
        "missing_calibration_warning": bool(missing_calibration),
        "missing_kinematics": bool(missing_kinematics),
        "missing_kinematics_warning": bool(missing_kinematics),
        "num_waypoints": int(ghost["num_waypoints"]),
        "min_z": ghost["min_z"],
        "touches_table": bool(ghost["touches_table"]),  # always False (lifted)
        "corners": corners,
        "grasp": grasp,
        "place": place,
        "normalized_fold_path": normalized_fold_path,
        "ghost_waypoints": ghost["waypoints"],
        "table_bounds": bounds,
        "joint_map_present": jm is not None,
        "active_servo_ids": active_servo_ids,
        "rendered": bool(rendered),
        "render_note": render_note,
        "png": png_path if rendered else None,
        "meta_json": meta_json,
        "warnings": warnings,
        "note": ("SIM preview of the air-only image-driven ghost fold. Contact folding "
                 "is DISABLED; the gripper never closes hard and every waypoint stays "
                 ">= clearance above the table. No kinematics, no hardware."),
    }

    write_json(meta_json, metadata)

    log(f"[sim] ghost-fold preview: direction={direction or 'unknown'} "
        f"waypoints={metadata['num_waypoints']} min_z={metadata['min_z']}m "
        f"contact_disabled=True touches_table={metadata['touches_table']}")
    if rendered:
        log(f"[sim] wrote preview PNG -> {png_path}")
    else:
        log(f"[sim] {render_note}")
    log(f"[sim] wrote metadata -> {meta_json}")
    for w in warnings:
        log(f"[sim][warn] {w}")

    return metadata
