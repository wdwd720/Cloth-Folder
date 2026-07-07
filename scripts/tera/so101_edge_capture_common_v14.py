"""V14 SO-101 edge-capture common utilities (real articulation-driven fold).

V13 established: the REAL SO-101 arm (physics-resolved joint targets) reaches the
towel edge and a jaw-attached contact patch touches the cloth + imparts a real
particle-velocity spike, but PRESSING on the edge centre with a small sphere and
dragging inward only slides over the stiff sheet (edge ~8 mm, no width reduction).

The jaw at the V13 contact pose sits ~48 mm ABOVE the flat cloth (cloth rests on
the ground at z~0.003), so a small sphere merely grazes the top. And width_x is
set by the towel corners (y = +/-0.19) — a point contact at y~0 cannot reduce it.

V14 changes the ROBOT-SIDE contact, all still driven by real articulation and a
patch rigidly attached to a robot link (NOT a free proxy, NOT a teleport, NOT
particle writes):
  * wider contact geometry (big sphere / flat-bottom cylinder disk / capsule bar
    spanning the whole edge incl. corners) so the whole edge moves together;
  * a real PRESS (command the jaw below the cloth so the position drive pushes
    down, building normal force → friction grip);
  * an OUTSIDE-HOOK approach (start just beyond the edge at low z, sweep inward
    through the edge lip) instead of pressing on the edge centre;
  * optional GRIPPER PINCH (a second patch on the `gripper` link; closing the
    real gripper joint traps the edge between the two patches) then lift + drag.

This module is import-safe under py_compile (Isaac imports are all inside funcs).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from clean_towel_v4_common import apply_action_step, reset_scene_to_points
from clean_towel_v5_contact_common import edge_center_displacement, towel_edge_masks
from so101_clean_towel_scene_utils_v1 import size_metrics

EDGE_BAND = 0.035
STABLE_TOWEL_TRANSLATION = (0.0, 0.0, 0.02)
DEFAULT_RIGHT_BASE = (0.42, -0.28, 0.02)
RIGHT_JOINTS = (6, 7, 8, 9)  # right shoulder_pan, shoulder_lift, elbow_flex, wrist_flex (translate the jaw)


# --------------------------------------------------------------------------- #
# Patch attachment (generalized V13 attach_jaw_contact_patch)
# --------------------------------------------------------------------------- #
def find_body_prim(arm_prim_path: str, body_match: str):
    """Return the deepest prim under ``arm_prim_path`` whose name contains
    ``body_match`` (e.g. 'jaw' or 'gripper')."""
    from isaacsim.core.utils.stage import get_current_stage
    from pxr import Usd

    stage = get_current_stage()
    root = stage.GetPrimAtPath(arm_prim_path)
    match = None
    for prim in Usd.PrimRange(root):
        if body_match in prim.GetName().lower():
            match = prim  # deepest/last match
    return match


def attach_patch(
    arm_prim_path: str,
    *,
    name: str,
    shape: str = "sphere",
    radius: float = 0.05,
    length: float = 0.0,
    axis: str = "Y",
    body_match: str = "jaw",
    local_offset=(0.0, 0.0, 0.0),
    contact_offset: float = 0.02,
    rest_offset: float = 0.0,
    static_friction: float = 2.0,
    dynamic_friction: float = 1.6,
    mat_path_str: str | None = None,
) -> dict[str, Any]:
    """Attach a collision ``shape`` as a CHILD of a robot link prim, so it is part
    of that link's rigid body and moves ONLY as the physics-resolved articulation
    moves it (V13-proven honest bridge; the raw jaw under-contacts the stiff cloth).

    shape: 'sphere' | 'cylinder' (flat-bottom disk, vertical axis) | 'capsule'
    axis:  capsule/cylinder long axis in the link's LOCAL frame ('X'|'Y'|'Z').
    """
    from isaacsim.core.utils.stage import get_current_stage
    from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics, UsdShade

    stage = get_current_stage()
    link_prim = find_body_prim(arm_prim_path, body_match)
    if link_prim is None:
        return {"attached": False, "reason": f"no '{body_match}' prim under {arm_prim_path}"}

    patch_path = link_prim.GetPath().AppendChild(name)
    axis_tok = {"X": UsdGeom.Tokens.x, "Y": UsdGeom.Tokens.y, "Z": UsdGeom.Tokens.z}.get(axis.upper(), UsdGeom.Tokens.y)
    extent = {"shape": shape, "radius": float(radius), "length": float(length), "axis": axis.upper()}
    if shape == "sphere":
        geom = UsdGeom.Sphere.Define(stage, patch_path)
        geom.CreateRadiusAttr(float(radius))
    elif shape == "cylinder":
        geom = UsdGeom.Cylinder.Define(stage, patch_path)
        geom.CreateRadiusAttr(float(radius))
        geom.CreateHeightAttr(float(max(length, 0.01)))
        geom.CreateAxisAttr(axis_tok)
    elif shape == "capsule":
        geom = UsdGeom.Capsule.Define(stage, patch_path)
        geom.CreateRadiusAttr(float(radius))
        geom.CreateHeightAttr(float(max(length, 0.01)))  # cylinder segment length (excl. caps)
        geom.CreateAxisAttr(axis_tok)
    else:
        return {"attached": False, "reason": f"unknown shape {shape}"}

    UsdGeom.Xformable(geom).AddTranslateOp().Set(Gf.Vec3d(*[float(x) for x in local_offset]))
    prim = geom.GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim)
    try:
        pxc = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        pxc.CreateContactOffsetAttr(float(contact_offset))
        pxc.CreateRestOffsetAttr(float(rest_offset))
    except Exception as exc:  # noqa: BLE001
        return {"attached": True, "patch_path": str(patch_path), "link_prim": str(link_prim.GetPath()),
                "extent": extent, "physx_offset_error": repr(exc)}
    try:
        mat_path = Sdf.Path(mat_path_str or f"/World/Materials/{name}_mat")
        if not stage.GetPrimAtPath(mat_path).IsValid():
            UsdShade.Material.Define(stage, mat_path)
            mat_api = UsdPhysics.MaterialAPI.Apply(stage.GetPrimAtPath(mat_path))
            mat_api.CreateStaticFrictionAttr(float(static_friction))
            mat_api.CreateDynamicFrictionAttr(float(dynamic_friction))
            mat_api.CreateRestitutionAttr(0.0)
        binding = UsdShade.MaterialBindingAPI.Apply(prim)
        binding.Bind(UsdShade.Material(stage.GetPrimAtPath(mat_path)),
                     bindingStrength=UsdShade.Tokens.weakerThanDescendants, materialPurpose="physics")
    except Exception as exc:  # noqa: BLE001
        return {"attached": True, "patch_path": str(patch_path), "link_prim": str(link_prim.GetPath()),
                "extent": extent, "material_bind_error": repr(exc)}
    return {"attached": True, "patch_path": str(patch_path), "link_prim": str(link_prim.GetPath()),
            "extent": extent, "local_offset": [float(x) for x in local_offset],
            "contact_offset": float(contact_offset), "static_friction": float(static_friction)}


# --------------------------------------------------------------------------- #
# Pose / measurement helpers
# --------------------------------------------------------------------------- #
def _quat_rotate(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    w, x, y, z = [float(c) for c in q_wxyz]
    r = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float32)
    return (r @ np.asarray(v, dtype=np.float32).reshape(3)).astype(np.float32)


def _link_index(arm: Any, body_match: str) -> int:
    names = [str(n).lower() for n in (getattr(arm.data, "body_names", []) or [])]
    cands = [i for i, n in enumerate(names) if body_match in n]
    return cands[-1] if cands else max(0, len(names) - 1)


def link_pose(arm: Any, body_match: str = "jaw") -> tuple[np.ndarray, np.ndarray]:
    idx = _link_index(arm, body_match)
    pos = arm.data.body_pos_w[0, idx].detach().cpu().numpy().astype(np.float32)
    try:
        quat = arm.data.body_quat_w[0, idx].detach().cpu().numpy().astype(np.float32)
    except Exception:  # noqa: BLE001
        quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return pos, quat


def patch_center(scene: Any, local_offset=(0.0, 0.0, 0.0), body_match: str = "jaw") -> np.ndarray:
    """World-frame centre of a link-attached patch = link_pos + R(link_quat)@offset."""
    pos, quat = link_pose(scene.right_arm, body_match)
    return (pos + _quat_rotate(quat, np.asarray(local_offset, dtype=np.float32))).astype(np.float32)


def default_action12(scene: Any) -> np.ndarray:
    left = scene.left_arm.data.default_joint_pos[0].detach().cpu().numpy().astype(np.float32)
    right = scene.right_arm.data.default_joint_pos[0].detach().cpu().numpy().astype(np.float32)
    return np.concatenate([left[:6], right[:6]]).astype(np.float32)


def drive_to(scene: Any, action12: np.ndarray, steps: int) -> None:
    for _ in range(int(steps)):
        apply_action_step(scene, action12)


def solve_joint_delta(J: np.ndarray, dp: np.ndarray, clip: float = 1.3) -> np.ndarray:
    """Least-squares RIGHT_JOINTS delta for a desired Cartesian jaw move ``dp``,
    padded to a full 12-vector and clipped. The Jacobian is only used to CONSTRUCT
    candidate motions; success is judged by direct cloth metrics, never by ``J``."""
    dq, *_ = np.linalg.lstsq(np.asarray(J, dtype=np.float32), np.asarray(dp, dtype=np.float32).reshape(3), rcond=None)
    delta = np.zeros(12, dtype=np.float32)
    for col, j in enumerate(RIGHT_JOINTS):
        delta[j] = float(np.clip(dq[col], -clip, clip))
    return delta


def estimate_jaw_jacobian(
    scene: Any, contact_action: np.ndarray, start_points: np.ndarray,
    center_fn: Callable[[Any], np.ndarray], *, eps: float = 0.14, steps: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """Probe a local joint->jaw Jacobian (3 x len(RIGHT_JOINTS)) at the reach pose
    by finite differences. Returns (jaw0, J)."""
    reset_scene_to_points(scene, start_points)
    drive_to(scene, contact_action, steps)
    jaw0 = center_fn(scene)
    J = np.zeros((3, len(RIGHT_JOINTS)), dtype=np.float32)
    for col, j in enumerate(RIGHT_JOINTS):
        a = np.asarray(contact_action, dtype=np.float32).copy(); a[j] += eps
        reset_scene_to_points(scene, start_points)
        drive_to(scene, a, steps)
        J[:, col] = (center_fn(scene) - jaw0) / eps
    return jaw0, J


# --------------------------------------------------------------------------- #
# Capture-episode plan + executor
# --------------------------------------------------------------------------- #
@dataclass
class CapturePlan:
    name: str
    contact_action: np.ndarray                    # reach pose (jaw near edge)
    approach_action: np.ndarray | None = None     # optional OUTSIDE pre-edge pose
    press_delta: np.ndarray | None = None         # added at contact (press down/in)
    drag_delta: np.ndarray | None = None          # added to sweep inward (fold)
    lift_delta: np.ndarray | None = None          # added after pinch (lift edge)
    gripper_close: float | None = None            # right gripper (idx 11) target for pinch
    steps: dict = field(default_factory=lambda: {
        "approach": 30, "contact": 30, "press": 20, "pinch": 16, "lift": 16, "drag": 90})


def _zeros12() -> np.ndarray:
    return np.zeros(12, dtype=np.float32)


def run_capture_episode(
    scene: Any,
    start_points: np.ndarray,
    plan: CapturePlan,
    edge: str,
    *,
    patch_offset=(0.0, 0.0, 0.0),
    patch_radius: float = 0.05,
    edge_band: float = EDGE_BAND,
    left_default: np.ndarray | None = None,
    record: bool = False,
    center_fn: Callable[[Any], np.ndarray] | None = None,
) -> tuple[dict[str, Any], list[np.ndarray]]:
    """Execute default -> [approach] -> contact -> [press] -> [pinch] -> [lift] ->
    drag, all via real articulation targets, recording cloth metrics each physics
    step. Returns (metrics, recorded_actions). Cloth measurements (edge disp,
    width, velocity) are DIRECT and cannot be faked by the touch proxy."""
    reset_scene_to_points(scene, start_points)
    if center_fn is None:
        def center_fn(sc: Any) -> np.ndarray:
            return patch_center(sc, patch_offset, "jaw")

    contact = np.asarray(plan.contact_action, dtype=np.float32).copy()
    if left_default is not None:
        contact[:6] = left_default[:6]
    approach = None if plan.approach_action is None else np.asarray(plan.approach_action, dtype=np.float32).copy()
    if approach is not None and left_default is not None:
        approach[:6] = left_default[:6]

    press = contact + (plan.press_delta if plan.press_delta is not None else _zeros12())
    # pinch closes the right gripper (idx 11) at the pressed pose
    pinch = press.copy()
    if plan.gripper_close is not None:
        pinch[11] = float(plan.gripper_close)
    lift = pinch + (plan.lift_delta if plan.lift_delta is not None else _zeros12())
    drag = lift + (plan.drag_delta if plan.drag_delta is not None else _zeros12())

    st = plan.steps
    base = default_action12(scene) if left_default is None else left_default
    segs: list[tuple[np.ndarray, np.ndarray, int]] = []
    prev = base
    if approach is not None and int(st.get("approach", 0)) > 0:
        segs.append((prev, approach, int(st["approach"])))
        prev = approach
    segs.append((prev, contact, int(st.get("contact", 30))))
    prev = contact
    if plan.press_delta is not None and int(st.get("press", 0)) > 0:
        segs.append((prev, press, int(st["press"])))
        prev = press
    if plan.gripper_close is not None and int(st.get("pinch", 0)) > 0:
        segs.append((prev, pinch, int(st["pinch"])))
        prev = pinch
    if plan.lift_delta is not None and int(st.get("lift", 0)) > 0:
        segs.append((prev, lift, int(st["lift"])))
        prev = lift
    if plan.drag_delta is not None and int(st.get("drag", 0)) > 0:
        segs.append((prev, drag, int(st["drag"])))
        prev = drag

    dt = float(scene.sim.get_physics_dt())
    widths = [float(size_metrics(start_points)["width_x"])]
    heights = [float(size_metrics(start_points)["height_y"])]
    prev_pts = start_points.copy()
    vpeak = 0.0
    nearest_min = float("inf")
    contact_touch = None
    near_span_y = 0.0
    acts: list[np.ndarray] = []

    for a0, a1, nsteps in segs:
        for k in range(nsteps):
            a = (a0 + (a1 - a0) * (float(k + 1) / float(nsteps))).astype(np.float32)
            apply_action_step(scene, a)
            if record:
                acts.append(a.copy())
            pts = scene.clean_towel.points_numpy().astype(np.float32)
            c = center_fn(scene)
            d3 = np.linalg.norm(pts - c.reshape(1, 3), axis=1)
            surf = float(np.min(d3)) - float(patch_radius)
            nearest_min = min(nearest_min, surf)
            near = np.linalg.norm(pts[:, :2] - c[:2].reshape(1, 2), axis=1) < (float(patch_radius) + 0.06)
            if np.any(near):
                fd = np.linalg.norm(pts - prev_pts, axis=1) / max(dt, 1e-6)
                vpeak = max(vpeak, float(np.max(fd[near])))
                ys = pts[near, 1]
                near_span_y = max(near_span_y, float(ys.max() - ys.min()))
            widths.append(float(size_metrics(pts)["width_x"]))
            heights.append(float(size_metrics(pts)["height_y"]))
            prev_pts = pts
        if a1 is contact or (prev is contact and contact_touch is None):
            contact_touch = float(max(0.0, nearest_min))

    final_pts = scene.clean_towel.points_numpy().astype(np.float32)
    warr = np.asarray(widths, dtype=np.float32)
    edge_disp = float(edge_center_displacement(start_points, final_pts, edge, edge_band))
    max_disp = float(np.max(np.linalg.norm(final_pts - start_points, axis=1)))
    metrics = {
        "actual_touch_distance_m": float(max(0.0, nearest_min)),
        "contact_pose_touch_distance_m": contact_touch,
        "edge_displacement_m": edge_disp,
        "max_particle_displacement_m": max_disp,
        "particle_velocity_peak": float(vpeak),
        "width_before": float(warr[0]),
        "width_after": float(warr[-1]),
        "best_width": float(warr.min()),
        "width_reduction_m": float(warr[0] - warr.min()),
        "near_particle_span_y_m": float(near_span_y),
        "num_steps": int(sum(n for _, _, n in segs)),
    }
    return metrics, acts


def classify_fold(m: dict[str, Any], *, min_edge: float = 0.02, min_width_red: float = 0.005,
                  baseline_edge: float = 0.0, max_baseline: float = 0.01) -> dict[str, Any]:
    """Fold VALID iff genuine contact drags the edge inward with a width reduction,
    excluding (a) ballistic flings (huge displacement / velocity at the 5.0 m/s
    cap) AND (b) BASELINE-CONTAMINATED configs — where the patch disturbs the cloth
    even with the arm held at default (a big collider attached to the jaw bulldozes
    the cloth, so the 'fold' is an artifact, not a controlled contact). A real fold
    requires the no-contact baseline to be ~inert."""
    edge_disp = float(m.get("edge_displacement_m", 0.0))
    max_disp = float(m.get("max_particle_displacement_m", 0.0))
    vpeak = float(m.get("particle_velocity_peak", 0.0))
    width_red = float(m.get("width_reduction_m", 0.0))
    touch = float(m.get("actual_touch_distance_m", 1.0))
    spurious = bool(max_disp > 0.5 or edge_disp > 0.5 or vpeak >= 4.9)
    baseline_contaminated = bool(abs(float(baseline_edge)) > max_baseline)
    valid = bool(touch < 0.01 and min_edge < edge_disp <= 0.5 and width_red > min_width_red
                 and not spurious and not baseline_contaminated)
    return {"spurious_ballistic": spurious, "baseline_contaminated": baseline_contaminated,
            "valid_controlled_contact": valid, "edge_ok": bool(min_edge < edge_disp <= 0.5),
            "width_ok": bool(width_red > min_width_red), "touch_ok": bool(touch < 0.01)}


def score_metrics(m: dict[str, Any], baseline_edge: float = 0.0) -> float:
    """Cloth-aware search objective: reward edge drag + width reduction, penalize
    ballistic / no-touch / baseline-contaminated. Never rewards spurious flings or
    a config whose patch bulldozes the cloth at rest."""
    cls = classify_fold(m, baseline_edge=baseline_edge)
    if cls["spurious_ballistic"] or cls["baseline_contaminated"]:
        return -1.0
    return float(m.get("edge_displacement_m", 0.0)) + 3.0 * max(0.0, float(m.get("width_reduction_m", 0.0)))
