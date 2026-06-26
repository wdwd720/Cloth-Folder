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
LEVELS = ("ee-only", "arm-ik", "cloth-proxy")
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
        _banner(img, f"{SIM_BANNER}   [{level}]  phase={ph[i]}")
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


def _render_mujoco(level, waypoints, gripper, times, phases, scene, w, h, fps, max_seconds, log) -> List[np.ndarray]:
    import mujoco  # noqa: F401

    ee, grip, ph, duration, n_frames = _timeline(waypoints, gripper, times, phases, fps, max_seconds)
    fold = _fold_fraction(ee[:, :2], grip, scene["grasp"], scene["place"]) if level == "cloth-proxy" \
        else np.zeros(n_frames)
    xml = _build_mjcf(level, scene, ee, w, h)
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

    frames: List[np.ndarray] = []
    for i in range(n_frames):
        data.mocap_pos[mocap_id] = ee[i]
        if hinge_qadr is not None:
            data.qpos[hinge_qadr] = float(fold[i]) * np.pi
        if level == "arm-ik":
            for _ in range(20):  # let the weld constraint pull the arm to the EE
                mujoco.mj_step(model, data)
        else:
            mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera="topcam")
        frames.append(renderer.render().copy())
    log(f"   MuJoCo renderer: {n_frames} frames @ {fps}fps (level={level})")
    return frames


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
    if level == "arm-ik":
        bx = cx - 0.5 * span
        arm_xml = f"""
    <body name="arm0" pos="{bx} {cy} 0">
      <geom type="cylinder" size="0.02 0.04" rgba="0.3 0.3 0.35 1"/>
      <body name="arm1" pos="0 0 0.04"><joint type="hinge" axis="0 0 1"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.12" size="0.012" rgba="0.55 0.55 0.6 1"/>
        <body name="arm2" pos="0 0 0.12"><joint type="hinge" axis="0 1 0"/>
          <geom type="capsule" fromto="0 0 0 {0.4*span} 0 0" size="0.011" rgba="0.55 0.55 0.6 1"/>
          <body name="arm3" pos="{0.4*span} 0 0"><joint type="hinge" axis="0 1 0"/>
            <geom type="capsule" fromto="0 0 0 {0.4*span} 0 0" size="0.010" rgba="0.55 0.55 0.6 1"/>
            <body name="arm4" pos="{0.4*span} 0 0"><joint type="hinge" axis="1 0 0"/>
              <joint type="hinge" axis="0 1 0"/>
              <body name="arm_tip" pos="0.03 0 0">
                <joint type="hinge" axis="0 0 1"/>
                <geom type="sphere" size="0.012" rgba="0.2 0.2 0.25 1"/>
                <site name="tip" pos="0 0 0" size="0.005"/>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>"""
        weld_xml = '<equality><weld body1="arm_tip" body2="ee"/></equality>'

    cloth_moving = ""
    if level == "cloth-proxy":
        cmid = 0.5 * (scene["crease_a"] + scene["crease_b"])
        cdir = scene["crease_b"] - scene["crease_a"]
        cdir = cdir / (np.linalg.norm(cdir) + 1e-9)
        cloth_moving = f"""
    <body name="moving_half" pos="{float(cmid[0])} {float(cmid[1])} 0.002">
      <joint name="fold_hinge" type="hinge" axis="{float(cdir[0])} {float(cdir[1])} 0" pos="0 0 0" limited="false"/>
      <geom type="box" pos="{float(ctr[0]-cmid[0])} {float(ctr[1]-cmid[1])} 0" size="{0.25*span} {0.5*span} 0.002" rgba="0.43 0.69 0.67 1"/>
    </body>"""
        static_geom = f'<geom name="towel" type="box" pos="{cx} {cy} 0" size="{0.5*span} {0.5*span} 0.002" rgba="0.28 0.55 0.55 1"/>'
    else:
        static_geom = f'<geom name="towel" type="box" pos="{cx} {cy} 0.002" size="{0.5*span} {0.5*span} 0.003" rgba="0.32 0.57 0.57 1"/>'

    return f"""
<mujoco model="terafold_fold_sim">
  <option gravity="0 0 0" integrator="implicitfast"/>
  <visual><global offwidth="{w}" offheight="{h}"/></visual>
  <worldbody>
    <light pos="{cx} {cy} {cam_z}" dir="0 0 -1" diffuse="0.9 0.9 0.9"/>
    <camera name="topcam" pos="{cx} {cy - 0.25 * span} {cam_z}" xyaxes="1 0 0 0 0.7 0.72"/>
    <geom name="table" type="box" pos="{cx} {cy} -0.01" size="{0.9*span} {0.9*span} 0.01" rgba="0.5 0.5 0.55 1"/>
    {static_geom}
    {cloth_moving}
    <geom name="grasp" type="sphere" pos="{gx} {gy} 0.006" size="0.01" rgba="1 0.55 0.1 1"/>
    <geom name="place" type="sphere" pos="{px} {py} 0.006" size="0.01" rgba="0.1 0.8 0.8 1"/>
    <body name="ee" mocap="true" pos="{float(ee[0,0])} {float(ee[0,1])} {z0}">
      <geom name="ee_sphere" type="sphere" size="0.013" rgba="0.9 0.2 0.18 1"/>
    </body>{arm_xml}
  </worldbody>
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


def run_sim_fold(
    plan_json: str,
    out: Optional[str] = None,
    level: str = "ee-only",
    fps: int = 20,
    max_seconds: float = 10.0,
    width: int = 640,
    height: int = 480,
    use_mujoco: bool = False,
    on_log=print,
) -> Dict[str, Any]:
    """Render a fold simulation video from a TeraFold plan JSON. Returns metadata."""
    log = on_log or (lambda _m: None)
    level = (level or "ee-only").lower()
    if level not in LEVELS:
        raise ValueError(f"unknown level {level!r}; use one of {LEVELS}")

    plan = load_plan(plan_json)
    waypoints, gripper, times, phases, scene = parse_trajectory(plan)
    out = out or f"runs/sim/fold_{level.replace('-', '_')}.mp4"
    log(SIM_BANNER)

    want_mujoco = use_mujoco or level == "arm-ik"
    renderer = None
    frames: Optional[List[np.ndarray]] = None
    render_error = None

    if want_mujoco:
        if not have_mujoco():
            if level == "arm-ik":
                return _missing_mujoco(plan_json, out, level, len(waypoints))
            log(f"[warn] MuJoCo not installed; using the 2D renderer. {SIM_INSTALL_HINT}")
        else:
            try:
                frames = _render_mujoco(level, waypoints, gripper, times, phases, scene,
                                        width, height, fps, max_seconds, log)
                renderer = "mujoco"
            except Exception as exc:
                render_error = str(exc)
                if level == "arm-ik":
                    return {
                        "status": "error", "level": level, "out": out,
                        "error": f"MuJoCo render failed: {exc}",
                        "hint": "Try --level ee-only (2D, always works).",
                        "num_waypoints": int(len(waypoints)),
                        "control": "none (simulation only)", "motor_commands_sent": 0,
                        "simulation_only": True,
                    }
                log(f"[warn] MuJoCo render failed ({exc}); falling back to the 2D renderer.")

    if frames is None:
        frames = _render_2d(level, waypoints, gripper, times, phases, scene,
                            width, height, fps, max_seconds, log)
        renderer = renderer or "2d"

    video = _write_video(frames, out, fps)
    meta = {
        "status": "ok",
        "level": level,
        "renderer": renderer,
        "out": video.get("video") or video.get("frames_dir"),
        "video": video.get("video"),
        "frames": len(frames),
        "fps": fps,
        "duration_s": round(len(frames) / fps, 2),
        "num_waypoints": int(len(waypoints)),
        "fold_direction": scene["direction"],
        "control": "none (simulation only)",
        "motor_commands_sent": 0,
        "simulation_only": True,
        "note": SIM_BANNER,
        "source_plan": plan_json,
    }
    if render_error:
        meta["mujoco_error"] = render_error
    meta_path = _meta_path(out)
    os.makedirs(os.path.dirname(os.path.abspath(meta_path)) or ".", exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    meta["meta_json"] = meta_path
    return meta
