"""Simulate a fold: a virtual SO-101-style arm / end-effector follows the plan.

SIMULATION ONLY — not calibrated to real hardware. This never sends motor
commands and never touches a robot adapter; it only visualizes the trajectory
TeraFold already planned.

Renderers:
  * 2D top-down (numpy + imageio) — the always-works path. Shows the table,
    towel, crease, grasp/place markers, and the end-effector following the
    planned waypoints (opening/closing the gripper). ``cloth-proxy`` folds the
    moving half across the crease as the gripper crosses it.
  * MuJoCo (optional, ``pip install -e ".[sim]"``) — a real 3D scene with a
    table, towel proxy, markers, an end-effector, and (for ``arm-ik``) an
    approximate 6-DOF arm welded to the end-effector so it follows the path.

Levels:
  * ``ee-only``    (default) — end-effector sphere follows the path; no arm IK.
  * ``arm-ik``     — approximate SO-101 arm follows via MuJoCo (requires mujoco).
  * ``cloth-proxy``— the towel proxy visually folds along the crease.
"""

from __future__ import annotations

import importlib.util
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

SIM_BANNER = "SIMULATION ONLY — not calibrated to real hardware."
LEVELS = ("ee-only", "arm-ik", "cloth-proxy", "so101-real")
MUJOCO_LEVELS = ("arm-ik", "so101-real")  # levels that require a 3D MuJoCo arm
ALL_VIEWS = ("iso", "top", "side", "gripper")
SLOWMO_FACTOR = 2.5
SIM_INSTALL_HINT = 'python3 -m pip install -e ".[sim]"'

__all__ = ["run_sim_fold", "load_plan", "parse_trajectory", "LEVELS", "SIM_BANNER"]


def have_mujoco() -> bool:
    return importlib.util.find_spec("mujoco") is not None


def _have_imageio() -> bool:
    return importlib.util.find_spec("imageio") is not None


# --------------------------------------------------------------------------
# Plan loading / trajectory parsing
# --------------------------------------------------------------------------


def load_plan(plan_json: str) -> Dict[str, Any]:
    """Load a FoldPlan / demo-result JSON and return the plan dict (unwrapped)."""
    with open(plan_json) as f:
        data = json.load(f)
    if isinstance(data, dict):
        if "trajectory" in data and "fold_state" in data:
            return data
        for key in ("plan", "fold_plan"):
            if isinstance(data.get(key), dict):
                return data[key]
    return data


def parse_trajectory(plan: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], Dict[str, Any]]:
    """Extract ``(waypoints[N,3], gripper[N], times[N], phases, scene)`` from a plan."""
    traj = plan.get("trajectory")
    if not isinstance(traj, dict) or "waypoints" not in traj:
        raise ValueError("plan JSON has no trajectory.waypoints — is this a TeraFold plan?")
    waypoints = np.asarray(traj["waypoints"], dtype=np.float64)
    if waypoints.ndim != 2 or waypoints.shape[1] != 3:
        raise ValueError(f"trajectory waypoints must be (N, 3), got {waypoints.shape}")
    n = waypoints.shape[0]
    gripper = np.asarray(traj.get("gripper", [1.0] * n), dtype=np.float64)
    times = np.asarray(traj.get("times", list(range(n))), dtype=np.float64)
    phases = list(traj.get("phases", [""] * n))

    fold_state = plan.get("fold_state", {}) or {}
    direction = (
        (plan.get("metadata", {}) or {}).get("direction")
        or (fold_state.get("metadata", {}) or {}).get("direction")
        or "right_to_left"
    )
    corners = _corners_from_plan(fold_state, waypoints)
    fl = plan.get("fold_line", {}) or {}
    gp = plan.get("grasp_place", {}) or {}
    scene = {
        "corners": corners,  # (4,2) TL,TR,BR,BL in table xy
        "crease_a": np.asarray(fl.get("a", corners.mean(0)), dtype=np.float64)[:2],
        "crease_b": np.asarray(fl.get("b", corners.mean(0)), dtype=np.float64)[:2],
        "grasp": np.asarray(gp.get("grasp", waypoints[0, :2]), dtype=np.float64)[:2],
        "place": np.asarray(gp.get("place", waypoints[-1, :2]), dtype=np.float64)[:2],
        "direction": direction,
    }
    return waypoints, gripper, times, phases, scene


def _corners_from_plan(fold_state: dict, waypoints: np.ndarray) -> np.ndarray:
    kp = fold_state.get("keypoints")
    if isinstance(kp, dict) and kp.get("top_left") is not None:
        return np.array([kp["top_left"], kp["top_right"], kp["bottom_right"], kp["bottom_left"]],
                        dtype=np.float64)[:, :2]
    # Fall back to the trajectory's xy bounding box.
    xy = waypoints[:, :2]
    lo, hi = xy.min(0), xy.max(0)
    return np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])


# --------------------------------------------------------------------------
# Frame timeline (shared by both renderers)
# --------------------------------------------------------------------------


def _timeline(waypoints, gripper, times, phases, fps, max_seconds):
    duration = float(times[-1] - times[0]) if len(times) > 1 else 1.0
    t_frames = max(2, int(round(fps * min(max(duration, 0.2), max_seconds))))
    sample_t = np.linspace(times[0], times[-1], t_frames)
    ee = np.stack([np.interp(sample_t, times, waypoints[:, k]) for k in range(3)], axis=1)
    idx = [max(0, int(np.searchsorted(times, t, side="right")) - 1) for t in sample_t]
    grip = np.array([gripper[i] for i in idx])
    ph = [phases[i] if i < len(phases) else "" for i in idx]
    return ee, grip, ph, duration, t_frames


def _fold_fraction(ee_xy, grip, grasp, place):
    """Per-frame fold fraction in [0,1]: progress grasp->place while gripper closed."""
    travel = np.asarray(place) - np.asarray(grasp)
    denom = float(travel @ travel) or 1.0
    raw = np.clip(((ee_xy - grasp) @ travel) / denom, 0.0, 1.0)
    out = np.zeros(len(ee_xy))
    held = 0.0
    started = False
    for i in range(len(ee_xy)):
        if grip[i] < 0.5:  # gripper closed -> grasping
            held = max(held, float(raw[i]))
            started = True
        out[i] = held if started else 0.0
    return out


# --------------------------------------------------------------------------
# 2D raster renderer (always works)
# --------------------------------------------------------------------------


def _disk(img, c, r, color, fill=True, thickness=2.0):
    h, w = img.shape[:2]
    cx, cy = float(c[0]), float(c[1])
    x0, x1 = max(0, int(cx - r - 1)), min(w, int(cx + r + 2))
    y0, y1 = max(0, int(cy - r - 1)), min(h, int(cy + r + 2))
    if x1 <= x0 or y1 <= y0:
        return
    ys, xs = np.mgrid[y0:y1, x0:x1]
    d = np.hypot(xs - cx, ys - cy)
    mask = d <= r if fill else (d <= r) & (d >= r - thickness)
    img[y0:y1, x0:x1][mask] = np.asarray(color, np.uint8)


def _seg(img, p0, p1, color, thickness=2.0):
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    n = int(max(2, np.hypot(*(p1 - p0))))
    for t in np.linspace(0, 1, n):
        _disk(img, p0 + (p1 - p0) * t, thickness / 2.0, color, fill=True)


def _dashed(img, p0, p1, color, thickness=2.0, dash=10):
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    length = float(np.hypot(*(p1 - p0)))
    n = max(2, int(length / dash))
    for i in range(0, n, 2):
        a = p0 + (p1 - p0) * (i / n)
        b = p0 + (p1 - p0) * (min(i + 1, n) / n)
        _seg(img, a, b, color, thickness)


def _point_in_poly(gx, gy, poly):
    n = len(poly)
    inside = np.zeros(gx.shape, bool)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        cond = ((yi > gy) != (yj > gy)) & (gx < (xj - xi) * (gy - yi) / ((yj - yi) + 1e-12) + xi)
        inside ^= cond
        j = i
    return inside


def _fill_poly(img, poly, color, alpha=0.55):
    poly = np.asarray(poly, float)
    h, w = img.shape[:2]
    x0, y0 = np.floor(poly.min(0)).astype(int)
    x1, y1 = np.ceil(poly.max(0)).astype(int)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1 + 1), min(h, y1 + 1)
    if x1 <= x0 or y1 <= y0:
        return
    gx, gy = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
    inside = _point_in_poly(gx, gy, poly)
    region = img[y0:y1, x0:x1].astype(float)
    col = np.asarray(color, float)
    region[inside] = (1 - alpha) * region[inside] + alpha * col
    img[y0:y1, x0:x1] = np.clip(region, 0, 255).astype(np.uint8)


def _make_mapper(points, w, h, margin_frac=0.14):
    p = np.asarray(points, float)
    lo, hi = p.min(0), p.max(0)
    dx, dy = max(hi[0] - lo[0], 1e-3), max(hi[1] - lo[1], 1e-3)
    m = margin_frac * min(w, h)
    scale = min((w - 2 * m) / dx, (h - 2 * m) / dy)
    cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
    return lambda xy: np.array([w / 2 + (xy[0] - cx) * scale, h / 2 + (xy[1] - cy) * scale]), scale


def _banner(img, text):
    try:
        from PIL import Image, ImageDraw

        pil = Image.fromarray(img)
        dr = ImageDraw.Draw(pil)
        dr.rectangle([0, 0, img.shape[1], 24], fill=(170, 30, 30))
        dr.text((8, 6), text, fill=(255, 255, 255))
        img[:] = np.asarray(pil)
    except Exception:
        img[:24] = (170, 30, 30)  # red banner without text


def _overlay(img, top_text, bottom_text=None):
    """Top safety banner + optional bottom phase subtitle (PIL if available)."""
    h, w = img.shape[:2]
    try:
        from PIL import Image, ImageDraw

        pil = Image.fromarray(np.ascontiguousarray(img[:, :, :3]))
        dr = ImageDraw.Draw(pil)
        dr.rectangle([0, 0, w, 24], fill=(170, 30, 30))
        dr.text((8, 6), top_text, fill=(255, 255, 255))
        if bottom_text:
            dr.rectangle([0, h - 24, w, h], fill=(20, 22, 28))
            dr.text((8, h - 18), bottom_text, fill=(240, 240, 120))
        img[:, :, :3] = np.asarray(pil)
    except Exception:
        img[:24] = (170, 30, 30)
        if bottom_text:
            img[h - 24:] = (20, 22, 28)


def _moving_half(corners, direction, f):
    """Return the moving-half quad folded by fraction ``f`` (corners reflect across crease)."""
    tl, tr, br, bl = corners
    mt = 0.5 * (tl + tr)
    mb = 0.5 * (br + bl)
    ml = 0.5 * (bl + tl)
    mr = 0.5 * (tr + br)
    crease = {
        "right_to_left": (mt, mb, tr, br),    # crease verts mt,mb ; moving outer tr,br
        "left_to_right": (mt, mb, tl, bl),
        "top_to_bottom": (ml, mr, tl, tr),
        "bottom_to_top": (ml, mr, bl, br),
    }.get(direction, (mt, mb, tr, br))
    c0, c1, o0, o1 = crease
    from terafold.math.geometry import reflect_point_across_line

    d = c1 - c0
    o0f = (1 - f) * o0 + f * reflect_point_across_line(o0, c0, d)
    o1f = (1 - f) * o1 + f * reflect_point_across_line(o1, c0, d)
    return np.array([c0, o0f, o1f, c1])


def _render_2d(level, waypoints, gripper, times, phases, scene, w, h, fps, max_seconds, log) -> List[np.ndarray]:
    ee, grip, ph, duration, n_frames = _timeline(waypoints, gripper, times, phases, fps, max_seconds)
    corners = scene["corners"]
    all_pts = np.vstack([corners, ee[:, :2], scene["grasp"], scene["place"],
                         scene["crease_a"], scene["crease_b"]])
    to_px, scale = _make_mapper(all_pts, w, h)
    fold = _fold_fraction(ee[:, :2], grip, scene["grasp"], scene["place"]) if level == "cloth-proxy" \
        else np.zeros(n_frames)

    static_half = _moving_half(corners, scene["direction"], 0.0)  # the moving half at f=0 (flat)
    # The stationary half = full towel minus moving half; just fill the whole towel then overlay.

    frames: List[np.ndarray] = []
    trail: List[np.ndarray] = []
    for i in range(n_frames):
        img = np.zeros((h, w, 3), np.uint8)
        img[:] = (26, 28, 36)  # backdrop
        # Table panel.
        tcorners = np.array([to_px(c) for c in _expand(corners, 1.6)])
        _fill_poly(img, tcorners, (62, 66, 74), alpha=1.0)
        # Towel: stationary half + (folding) moving half.
        if level == "cloth-proxy":
            stat = _stationary_half(corners, scene["direction"])
            _fill_poly(img, [to_px(c) for c in stat], (70, 130, 130), alpha=0.9)
            mov = _moving_half(corners, scene["direction"], float(fold[i]))
            _fill_poly(img, [to_px(c) for c in mov], (110, 175, 170), alpha=0.85)
        else:
            _fill_poly(img, [to_px(c) for c in corners], (80, 145, 145), alpha=0.85)
        # Crease (dashed yellow).
        _dashed(img, to_px(scene["crease_a"]), to_px(scene["crease_b"]), (230, 210, 60), 2.0)
        # Grasp (orange ring) + place (cyan ring).
        _disk(img, to_px(scene["grasp"]), 9, (245, 150, 30), fill=False, thickness=3)
        _disk(img, to_px(scene["place"]), 9, (40, 210, 210), fill=False, thickness=3)
        # EE trail.
        cur = to_px(ee[i, :2])
        trail.append(cur)
        for k in range(1, len(trail)):
            _seg(img, trail[k - 1], trail[k], (200, 200, 210), 1.4)
        # EE marker: closed = filled red, open = green ring. Radius hints height z.
        r = 7 + 26 * float(max(ee[i, 2], 0.0))
        if grip[i] < 0.5:
            _disk(img, cur, r, (235, 60, 50), fill=True)
        else:
            _disk(img, cur, r, (60, 210, 90), fill=False, thickness=3)
        _overlay(img, f"{SIM_BANNER}  [{level}]", f"phase: {ph[i]}")
        frames.append(img)
    log(f"   2D renderer: {n_frames} frames, ~{n_frames / fps:.1f}s @ {fps}fps")
    return frames


def _expand(corners, factor):
    c = np.asarray(corners, float)
    ctr = c.mean(0)
    return ctr + (c - ctr) * factor


def _stationary_half(corners, direction):
    tl, tr, br, bl = corners
    mt = 0.5 * (tl + tr)
    mb = 0.5 * (br + bl)
    ml = 0.5 * (bl + tl)
    mr = 0.5 * (tr + br)
    return {
        "right_to_left": np.array([tl, mt, mb, bl]),
        "left_to_right": np.array([mt, tr, br, mb]),
        "top_to_bottom": np.array([ml, mr, br, bl]),
        "bottom_to_top": np.array([tl, tr, mr, ml]),
    }.get(direction, np.array([tl, mt, mb, bl]))


# --------------------------------------------------------------------------
# Video writing
# --------------------------------------------------------------------------


def _write_video(frames: List[np.ndarray], out: str, fps: int) -> Dict[str, Any]:
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    if not _have_imageio():
        frames_dir = os.path.splitext(out)[0] + "_frames"
        os.makedirs(frames_dir, exist_ok=True)
        from terafold.vision.imageio import imwrite

        for i, fr in enumerate(frames):
            imwrite(os.path.join(frames_dir, f"frame_{i:04d}.png"), fr)
        return {"video": None, "frames_dir": frames_dir,
                "note": "imageio not installed; wrote PNG frames instead of a video."}
    import imageio.v2 as iio

    ext = os.path.splitext(out)[1].lower()
    if ext == ".gif":
        iio.mimsave(out, frames, fps=fps)
    else:
        writer = iio.get_writer(out, fps=fps, macro_block_size=1)
        for fr in frames:
            writer.append_data(fr)
        writer.close()
    return {"video": out}


# --------------------------------------------------------------------------
# MuJoCo renderer (optional, best-effort)
# --------------------------------------------------------------------------

# View presets: (azimuth deg, elevation deg). Distance is auto-fit per scene.
VIEW_PRESETS = {
    "iso": (45.0, -32.0),
    "top": (90.0, -89.0),
    "side": (90.0, -10.0),
    "follow-ee": (45.0, -28.0),
    "gripper": (35.0, -22.0),  # close-up that tracks the gripper
}
_FOLLOW_VIEWS = ("follow-ee", "gripper")


def _scene_bounds(waypoints, scene) -> Dict[str, np.ndarray]:
    """Bounds over the whole trajectory + towel + markers (for camera auto-fit)."""
    flat = [waypoints[:, :3]]
    for c in scene["corners"]:
        flat.append(np.array([[c[0], c[1], 0.0]]))
    for key in ("grasp", "place", "crease_a", "crease_b"):
        p = scene[key]
        flat.append(np.array([[p[0], p[1], 0.0]]))
    pts = np.vstack(flat)
    lo, hi = pts.min(0), pts.max(0)
    center = 0.5 * (lo + hi)
    radius = float(np.linalg.norm(hi - center))  # half the 3D diagonal
    return {"lo": lo, "hi": hi, "center": center, "radius": max(radius, 0.06)}


def _make_camera(view, bounds, fovy):
    import mujoco

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = bounds["center"]
    az, el = VIEW_PRESETS.get(view, VIEW_PRESETS["iso"])
    import math

    # Distance so the scene's diagonal fits the vertical FOV, with margin.
    fit = bounds["radius"] / max(math.tan(math.radians(fovy) / 2.0), 1e-3)
    dist = fit * 1.4 + 0.15
    if view == "follow-ee":
        dist = max(bounds["radius"] * 2.2, 0.30)
    elif view == "gripper":
        dist = max(bounds["radius"] * 1.1, 0.20)  # tight close-up
    cam.azimuth, cam.elevation, cam.distance = az, el, dist
    return cam, {"azimuth": az, "elevation": el, "distance": dist, "fovy": fovy,
                 "lookat": bounds["center"].tolist()}


def _cam_world_pos(cam_info):
    import math

    az = math.radians(cam_info["azimuth"])
    el = math.radians(cam_info["elevation"])
    fwd = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    return np.asarray(cam_info["lookat"]) - cam_info["distance"] * fwd


def _framing_warning(view, cam_info, bounds) -> Optional[str]:
    import math

    visible_r = cam_info["distance"] * math.tan(math.radians(cam_info["fovy"]) / 2.0)
    if bounds["radius"] > visible_r * 1.02:
        cp = _cam_world_pos(cam_info)
        return (
            f"[warn] trajectory may exceed the camera frame (view={view}): "
            f"traj_radius={bounds['radius']:.3f} > visible_radius={visible_r:.3f}. "
            f"camera_pos={np.round(cp, 3).tolist()}, target={np.round(cam_info['lookat'], 3).tolist()}, "
            f"bounds lo={np.round(bounds['lo'], 3).tolist()} hi={np.round(bounds['hi'], 3).tolist()}. "
            "Try --view top or a larger frame."
        )
    return None


def _add_trail(scene_obj, pts, max_pts=60, rgba=(0.95, 0.95, 1.0, 0.75)):
    """Append small spheres tracing the EE path to the render scene."""
    import mujoco

    if len(pts) == 0:
        return
    step = max(1, len(pts) // max_pts)
    mat = np.eye(3).flatten()
    size = np.array([0.004, 0.0, 0.0])
    col = np.array(rgba, dtype=np.float32)
    for p in pts[::step]:
        if scene_obj.ngeom >= scene_obj.maxgeom:
            break
        g = scene_obj.geoms[scene_obj.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, size,
                            np.ascontiguousarray(p, dtype=np.float64), mat, col)
        scene_obj.ngeom += 1


def _add_waypoint_dots(scene_obj, waypoints, rgba=(0.85, 0.35, 0.95, 0.9)):
    """Append a small sphere at each planned waypoint."""
    import mujoco

    mat = np.eye(3).flatten()
    size = np.array([0.006, 0.0, 0.0])
    col = np.array(rgba, dtype=np.float32)
    for p in waypoints[:, :3]:
        if scene_obj.ngeom >= scene_obj.maxgeom:
            break
        g = scene_obj.geoms[scene_obj.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, size,
                            np.ascontiguousarray(p, dtype=np.float64), mat, col)
        scene_obj.ngeom += 1


def _setup_so101(model, data, level):
    """For so101-real: set a sensible home pose and return gripper actuator ids."""
    import mujoco

    from terafold.sim import so101

    grip_acts = []
    if level != "so101-real":
        return grip_acts, 0.0
    for jname, deg in so101.SO101_SPEC["home_deg"].items():
        try:
            data.qpos[model.joint(jname).qposadr[0]] = np.radians(deg)
        except Exception:
            pass
    mujoco.mj_forward(model, data)
    for aname in so101.SO101_GRIPPER_ACTUATORS:
        try:
            grip_acts.append(model.actuator(aname).id)
        except Exception:
            pass
    return grip_acts, so101.SO101_SPEC["finger_travel"]


def _render_mujoco(level, waypoints, gripper, times, phases, scene, w, h, fps, max_seconds, log,
                   view: str = "iso", waypoint_dots: bool = False,
                   debug_xml_path: Optional[str] = None) -> List[np.ndarray]:
    import mujoco  # noqa: F401

    ee, grip, ph, duration, n_frames = _timeline(waypoints, gripper, times, phases, fps, max_seconds)
    fold = _fold_fraction(ee[:, :2], grip, scene["grasp"], scene["place"]) if level == "cloth-proxy" \
        else np.zeros(n_frames)
    xml = _build_mjcf(level, scene, ee, w, h)
    try:
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=h, width=w)
        mocap_id = model.body("ee").mocapid[0]
        hinge_qadr = None
        if level == "cloth-proxy":
            try:
                hinge_qadr = model.joint("fold_hinge").qposadr[0]
            except Exception:
                hinge_qadr = None
        grip_acts, grip_close = _setup_so101(model, data, level)
        substeps = 60 if level == "so101-real" else 40

        # Auto-fit a free camera to the whole trajectory + table.
        bounds = _scene_bounds(waypoints, scene)
        fovy = float(model.vis.global_.fovy)
        cam, cam_info = _make_camera(view, bounds, fovy)
        # Follow/gripper views are intentional close-ups that track the EE, so a
        # "whole trajectory doesn't fit" warning would be expected noise there.
        if view not in _FOLLOW_VIEWS:
            warn = _framing_warning(view, cam_info, bounds)
            if warn:
                log(warn)
        log(f"   camera: view={view} az={cam_info['azimuth']} el={cam_info['elevation']} "
            f"dist={cam_info['distance']:.2f} target={np.round(cam_info['lookat'], 3).tolist()}")

        frames: List[np.ndarray] = []
        for i in range(n_frames):
            data.mocap_pos[mocap_id] = ee[i]
            if hinge_qadr is not None:
                data.qpos[hinge_qadr] = float(fold[i]) * np.pi
            # Gripper open(1)/close(0) from the trajectory -> finger close ctrl.
            for aid in grip_acts:
                data.ctrl[aid] = (1.0 - float(grip[i])) * grip_close
            if level in ("arm-ik", "so101-real"):
                for _ in range(substeps):  # weld constraint pulls the arm to the EE
                    mujoco.mj_step(model, data)
            else:
                mujoco.mj_forward(model, data)
            if view in _FOLLOW_VIEWS:
                cam.lookat[:] = ee[i]
            renderer.update_scene(data, camera=cam)
            _add_trail(renderer.scene, ee[: i + 1])  # EE trail overlay
            if waypoint_dots:
                _add_waypoint_dots(renderer.scene, waypoints)
            img = renderer.render().copy()
            _overlay(img, f"{SIM_BANNER}  [{level}]", f"phase: {ph[i]}")
            frames.append(img)
        renderer.close()
    except Exception as exc:
        # Req 6: on any MuJoCo failure, dump the exact MJCF and surface a snippet.
        path = debug_xml_path
        if path:
            try:
                os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
                with open(path, "w") as f:
                    f.write(xml)
            except Exception:
                pass
        snippet = "\n".join(xml.strip().splitlines()[:12])
        raise RuntimeError(
            f"{type(exc).__name__}: {exc}"
            + (f"\nMJCF written to: {path}" if path else "")
            + f"\n--- MJCF (head) ---\n{snippet}"
        ) from exc
    log(f"   MuJoCo renderer: {n_frames} frames @ {fps}fps (level={level})")
    return frames


def make_link_inertial(mass: float = 0.05, inertia: float = 1e-4) -> str:
    """Inertial tag giving a moving body positive mass + diagonal inertia.

    Every body that has a joint (a *moving* body) must have positive mass and
    inertia, or MuJoCo errors with "mass and inertia of moving bodies must be
    larger than mjMINVAL". Applying this to every arm/gripper/finger body — even
    ones that would otherwise be massless — guarantees a valid model.
    """
    mass = max(float(mass), 1e-4)
    inertia = max(float(inertia), 1e-6)
    return (
        f'<inertial pos="0 0 0" mass="{mass:.5f}" '
        f'diaginertia="{inertia:.6f} {inertia:.6f} {inertia:.6f}"/>'
    )


# 6-DOF hinge axes for the approximate SO-101 arm.
_ARM_AXES = ("0 0 1", "0 1 0", "0 1 0", "0 1 0", "1 0 0", "0 0 1")


def _arm_chain(span: float, bx: float, by: float) -> str:
    """Approximate SO-101 6-DOF arm: arm0..arm5 + gripper + finger.

    Every body carries an explicit inertial (via :func:`make_link_inertial`) AND
    a positive-size geom, so no moving body has zero mass/inertia.
    """
    link = max(0.06, 0.16 * span)
    # Innermost first: gripper + finger (both massive).
    body = (
        f'<body name="gripper" pos="{link:.4f} 0 0">'
        f'{make_link_inertial(0.03, 5e-5)}'
        f'<geom type="box" size="0.020 0.014 0.008" rgba="0.20 0.20 0.25 1"/>'
        f'<site name="tip" pos="0 0 0" size="0.005"/>'
        f'<body name="finger" pos="0.020 0 0">'
        f'{make_link_inertial(0.01, 1e-5)}'
        f'<geom type="box" size="0.006 0.002 0.012" rgba="0.15 0.15 0.18 1"/>'
        f'</body>'
        f'</body>'
    )
    # Wrap arm5 .. arm0 around it (arm0 is anchored to the world at the base).
    for idx in reversed(range(6)):
        pos = f"{bx:.4f} {by:.4f} 0.02" if idx == 0 else f"{link:.4f} 0 0"
        radius = max(0.008, 0.013 - 0.0008 * idx)
        body = (
            f'<body name="arm{idx}" pos="{pos}">'
            f'<joint name="j{idx}" type="hinge" axis="{_ARM_AXES[idx]}"/>'
            f'{make_link_inertial()}'
            f'<geom type="capsule" fromto="0 0 0 {link:.4f} 0 0" size="{radius:.4f}" '
            f'rgba="0.55 0.55 0.60 1"/>'
            f'{body}'
            f'</body>'
        )
    return body


def _build_mjcf(level, scene, ee, w, h) -> str:
    corners = scene["corners"]
    ctr = corners.mean(0)
    cx, cy = float(ctr[0]), float(ctr[1])
    span = float(np.linalg.norm(corners.max(0) - corners.min(0))) + 1e-3
    gx, gy = map(float, scene["grasp"])
    px, py = map(float, scene["place"])
    z0 = float(ee[0, 2])
    cam_z = 0.6 + span
    arm_xml = ""
    weld_xml = ""
    defaults_xml = ""
    actuator_xml = ""
    extra_materials = ""
    ee_geom = '<geom name="ee_sphere" type="sphere" size="0.016" rgba="0.95 0.2 0.18 1"/>'
    if level == "arm-ik":
        arm_xml = _arm_chain(span, cx - 0.5 * span, cy)
        # Weld the gripper to the EE mocap so the arm follows the planned path.
        weld_xml = '<equality><weld body1="gripper" body2="ee" solref="0.02 1"/></equality>'
        defaults_xml = '<default><joint damping="3" armature="0.05" limited="false"/></default>'
    elif level == "so101-real":
        from terafold.sim import so101

        base_off = max(0.55 * span, 0.20)
        arm = so101.build_so101_arm((cx, cy - base_off), base_z=0.0)
        arm_xml = arm["body_xml"]
        weld_xml = arm["weld_xml"]
        actuator_xml = arm["actuator_xml"]
        extra_materials = arm["materials_xml"]
        defaults_xml = ('<default><joint damping="5" armature="0.1"/>'
                        '<position kp="30"/></default>')
        # The real gripper renders at the EE; keep only a tiny weld-target marker.
        ee_geom = '<geom name="ee_sphere" type="sphere" size="0.006" rgba="0.95 0.3 0.3 0.6"/>'

    cloth_moving = ""
    if level == "cloth-proxy":
        cmid = 0.5 * (scene["crease_a"] + scene["crease_b"])
        cdir = scene["crease_b"] - scene["crease_a"]
        cdir = cdir / (np.linalg.norm(cdir) + 1e-9)
        cloth_moving = f"""
    <body name="moving_half" pos="{float(cmid[0])} {float(cmid[1])} 0.002">
      <joint name="fold_hinge" type="hinge" axis="{float(cdir[0])} {float(cdir[1])} 0" pos="0 0 0" limited="false"/>
      <geom type="box" pos="{float(ctr[0]-cmid[0])} {float(ctr[1]-cmid[1])} 0" size="{0.25*span} {0.5*span} 0.002" material="towel_moving"/>
    </body>"""
        static_geom = f'<geom name="towel" type="box" pos="{cx} {cy} 0" size="{0.5*span} {0.5*span} 0.002" material="towel"/>'
    else:
        static_geom = f'<geom name="towel" type="box" pos="{cx} {cy} 0.002" size="{0.5*span} {0.5*span} 0.0015" material="towel"/>'

    ca, cb = scene["crease_a"], scene["crease_b"]
    cl = max(0.5 * span, 0.05)  # frame-axis length
    # Debug objects: fold crease line + an RGB frame triad at the towel centre.
    crease_geom = (
        f'<geom name="crease" type="capsule" '
        f'fromto="{float(ca[0])} {float(ca[1])} 0.004 {float(cb[0])} {float(cb[1])} 0.004" '
        f'size="0.004" rgba="0.95 0.85 0.20 1"/>'
    )
    axes_geom = (
        f'<geom type="capsule" fromto="{cx} {cy} 0.003 {cx + cl} {cy} 0.003" size="0.0025" rgba="0.9 0.25 0.25 1"/>'
        f'<geom type="capsule" fromto="{cx} {cy} 0.003 {cx} {cy + cl} 0.003" size="0.0025" rgba="0.25 0.8 0.25 1"/>'
        f'<geom type="capsule" fromto="{cx} {cy} 0.003 {cx} {cy} {0.003 + cl} " size="0.0025" rgba="0.35 0.45 0.95 1"/>'
    )
    return f"""
<mujoco model="terafold_fold_sim">
  <compiler angle="degree" autolimits="true"/>
  <option timestep="0.002" gravity="0 0 0" integrator="implicitfast"/>
  <visual>
    <global offwidth="{w}" offheight="{h}"/>
    <headlight ambient="0.55 0.55 0.55" diffuse="0.7 0.7 0.7" specular="0.2 0.2 0.2"/>
    <rgba haze="0.18 0.19 0.22 1"/>
    <map shadowclip="2"/>
  </visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.34 0.36 0.42" rgb2="0.12 0.13 0.16" width="64" height="64"/>
    <texture name="tabletex" type="2d" builtin="checker" rgb1="0.50 0.51 0.55" rgb2="0.42 0.43 0.47" width="300" height="300"/>
    <material name="table" texture="tabletex" texrepeat="6 6" specular="0.2" shininess="0.1"/>
    <material name="towel" rgba="0.20 0.55 0.62 1" specular="0.1" shininess="0.05"/>
    <material name="towel_moving" rgba="0.30 0.66 0.70 1" specular="0.1" shininess="0.05"/>
    {extra_materials}
  </asset>
  {defaults_xml}
  <worldbody>
    <light name="key" pos="{cx + span} {cy - span} {cam_z}" dir="-1 1 -2" diffuse="0.7 0.7 0.7" specular="0.3 0.3 0.3" directional="true"/>
    <light name="fill" pos="{cx - span} {cy + span} {cam_z}" dir="1 -1 -2" diffuse="0.4 0.4 0.4" directional="true"/>
    <geom name="table" type="box" pos="{cx} {cy} -0.01" size="{1.0*span} {1.0*span} 0.01" material="table"/>
    {static_geom}
    {cloth_moving}
    {crease_geom}
    {axes_geom}
    <geom name="grasp" type="sphere" pos="{gx} {gy} 0.008" size="0.012" rgba="1 0.55 0.1 1"/>
    <geom name="place" type="sphere" pos="{px} {py} 0.008" size="0.012" rgba="0.1 0.8 0.85 1"/>
    <body name="ee" mocap="true" pos="{float(ee[0,0])} {float(ee[0,1])} {z0}">
      {ee_geom}
    </body>{arm_xml}
  </worldbody>
  {actuator_xml}
  {weld_xml}
</mujoco>"""


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------


def _meta_path(out: str) -> str:
    return os.path.splitext(out)[0] + ".meta.json"


def _missing_mujoco(plan_json, out, level, n_waypoints) -> Dict[str, Any]:
    return {
        "status": "missing_dependency",
        "level": level,
        "out": out,
        "install_command": SIM_INSTALL_HINT,
        "message": (
            f"--level {level} needs MuJoCo, which is not installed. Install the sim "
            f"extra:  {SIM_INSTALL_HINT}   (or use --level ee-only / cloth-proxy, "
            "which work without MuJoCo)."
        ),
        "num_waypoints": int(n_waypoints),
        "control": "none (simulation only)",
        "motor_commands_sent": 0,
        "simulation_only": True,
    }


def _view_out(out: str, view: str) -> str:
    base, ext = os.path.splitext(out)
    return f"{base}_{view}{ext}"


def _save_frames(frames: List[np.ndarray], out_path: str) -> str:
    frames_dir = os.path.splitext(out_path)[0] + "_frames"
    os.makedirs(frames_dir, exist_ok=True)
    from terafold.vision.imageio import imwrite

    for i, fr in enumerate(frames):
        imwrite(os.path.join(frames_dir, f"frame_{i:04d}.png"), np.ascontiguousarray(fr[:, :, :3]))
    return frames_dir


def run_sim_fold(
    plan_json: str,
    out: Optional[str] = None,
    level: str = "ee-only",
    fps: int = 20,
    max_seconds: float = 10.0,
    width: int = 640,
    height: int = 480,
    use_mujoco: bool = False,
    view: str = "iso",
    all_views: bool = False,
    save_frames: bool = False,
    slowmo: bool = False,
    waypoint_dots: Optional[bool] = None,
    on_log=print,
) -> Dict[str, Any]:
    """Render a fold simulation video (or videos) from a TeraFold plan JSON.

    Returns metadata. ``level`` so101-real / arm-ik need MuJoCo. ``all_views``
    renders iso/top/side/gripper to ``<out>_<view>.<ext>``. ``slowmo`` lengthens
    the playback; ``save_frames`` also exports PNG frames.
    """
    log = on_log or (lambda _m: None)
    level = (level or "ee-only").lower()
    if level not in LEVELS:
        raise ValueError(f"unknown level {level!r}; use one of {LEVELS}")

    plan = load_plan(plan_json)
    waypoints, gripper, times, phases, scene = parse_trajectory(plan)
    out = out or f"runs/sim/fold_{level.replace('-', '_')}.mp4"
    log(SIM_BANNER)

    requires_mujoco = level in MUJOCO_LEVELS
    want_mujoco = use_mujoco or requires_mujoco
    use_mj = False
    if want_mujoco:
        if not have_mujoco():
            if requires_mujoco:
                return _missing_mujoco(plan_json, out, level, len(waypoints))
            log(f"[warn] MuJoCo not installed; using the 2D renderer. {SIM_INSTALL_HINT}")
        else:
            use_mj = True

    # so101-real: report which SO-101 assets were found / used.
    asset_report = None
    if level == "so101-real":
        from terafold.sim import so101

        asset_report = so101.find_so101_assets()
        log(f"   SO-101 assets: {'official ' + str(asset_report['type']) + ' @ ' + str(asset_report['path']) if asset_report['found'] else 'approximation (no official assets found)'}")

    eff_max = max_seconds * (SLOWMO_FACTOR if slowmo else 1.0)
    if waypoint_dots is None:
        waypoint_dots = level == "so101-real"  # default-on for the realistic arm

    if all_views and use_mj:
        views = list(ALL_VIEWS)
    else:
        if all_views and not use_mj:
            log("[note] --all-views applies to MuJoCo renders; producing one 2D video.")
        views = [view]

    outputs: Dict[str, str] = {}
    frames_dirs: Dict[str, str] = {}
    renderer = "2d"
    render_error = None
    n_frames = 0

    for v in views:
        out_v = _view_out(out, v) if (all_views and use_mj) else out
        frames: Optional[List[np.ndarray]] = None
        if use_mj:
            debug_xml = os.path.splitext(out_v)[0] + ".mjcf.xml"
            try:
                frames = _render_mujoco(level, waypoints, gripper, times, phases, scene,
                                        width, height, fps, eff_max, log, view=v,
                                        waypoint_dots=waypoint_dots, debug_xml_path=debug_xml)
                renderer = "mujoco"
            except Exception as exc:
                render_error = str(exc)
                if requires_mujoco:
                    return {
                        "status": "error", "level": level, "view": v, "out": out_v,
                        "error": f"MuJoCo render failed: {exc}",
                        "xml_debug": debug_xml if os.path.exists(debug_xml) else None,
                        "hint": "Inspect the MJCF above, or use --level ee-only (2D, always works).",
                        "num_waypoints": int(len(waypoints)),
                        "control": "none (simulation only)", "motor_commands_sent": 0,
                        "simulation_only": True,
                    }
                log(f"[warn] MuJoCo render failed ({exc}); falling back to the 2D renderer.")
        if frames is None:
            frames = _render_2d(level, waypoints, gripper, times, phases, scene,
                                width, height, fps, eff_max, log)
            renderer = "2d"
        n_frames = len(frames)
        video = _write_video(frames, out_v, fps)
        outputs[v] = video.get("video") or video.get("frames_dir")
        if save_frames:
            frames_dirs[v] = _save_frames(frames, out_v)

    primary_view = views[0]
    primary_out = outputs[primary_view]
    meta = {
        "status": "ok",
        "level": level,
        "view": primary_view,
        "views": outputs,
        "all_views": bool(all_views and use_mj),
        "renderer": renderer,
        "out": primary_out,
        "video": primary_out,
        "frames": n_frames,
        "fps": fps,
        "slowmo": bool(slowmo),
        "waypoint_dots": bool(waypoint_dots),
        "duration_s": round(n_frames / fps, 2),
        "num_waypoints": int(len(waypoints)),
        "fold_direction": scene["direction"],
        "control": "none (simulation only)",
        "motor_commands_sent": 0,
        "simulation_only": True,
        "note": SIM_BANNER,
        "source_plan": plan_json,
    }
    if asset_report is not None:
        meta["so101_assets"] = asset_report
    if frames_dirs:
        meta["frames_dirs"] = frames_dirs
    if render_error:
        meta["mujoco_error"] = render_error
    meta_path = _meta_path(out)
    os.makedirs(os.path.dirname(os.path.abspath(meta_path)) or ".", exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    meta["meta_json"] = meta_path
    return meta
