"""SO-101 arm: official-asset discovery + a realistic kinematic approximation.

We first look for *official* SO-101 assets (URDF / MJCF / meshes) shipped by the
repo or installed packages (LeRobot, MuJoCo Menagerie, robot_descriptions). If
none are found, :func:`build_so101_arm` emits a much closer visual/kinematic
**approximation** of the SO-101 (a.k.a. SO-ARM101) low-cost 5-DOF arm with a
parallel gripper — measured-proportional link lengths, realistic joint limits,
and distinct materials.

SIMULATION ONLY — not calibrated to real hardware. This module emits MJCF
geometry; it never drives a real robot.
"""

from __future__ import annotations

import glob
import importlib.util
import os
from typing import Any, Dict, List, Optional

# Approximate SO-101 / SO-ARM101 geometry (metres) and joint limits (degrees).
# Proportional to the published open-hardware design; not a calibrated model.
SO101_SPEC: Dict[str, Any] = {
    "base_height": 0.055,
    "l_upper": 0.116,    # shoulder -> elbow
    "l_fore": 0.135,     # elbow -> wrist
    "l_wrist": 0.060,    # wrist -> gripper mount
    "gripper_len": 0.055,
    "finger_travel": 0.012,   # parallel-jaw inward travel (open->closed)
    "joint_limits_deg": {
        "j_base": (-110.0, 110.0),
        "j_shoulder": (-100.0, 100.0),
        "j_elbow": (-110.0, 110.0),
        "j_wristpitch": (-100.0, 100.0),
        "j_wristroll": (-180.0, 180.0),
    },
    # A sensible "ready" pose (degrees) so the IK weld starts near a good config.
    "home_deg": {"j_shoulder": -55.0, "j_elbow": 95.0, "j_wristpitch": -40.0},
    "dof": 5,
}

# Body names of every moving link (used by the renderer + tests).
SO101_BODIES = (
    "arm_j1", "arm_j2", "arm_j3", "arm_j4", "arm_j5",
    "gripper_base", "finger_left", "finger_right",
)
SO101_GRIPPER_ACTUATORS = ("a_grip_l", "a_grip_r")
SO101_WELD_BODY = "gripper_base"

_ASSET_KEYWORDS = ("so101", "so-101", "so_101", "so100", "so-100", "soarm", "so_arm", "so-arm")


def find_so101_assets() -> Dict[str, Any]:
    """Search for official SO-101 URDF/MJCF/mesh assets.

    Returns a report dict: ``{found, type, path, meshes, searched, note}``. Honors
    ``TERAFOLD_SO101_MJCF`` / ``TERAFOLD_SO101_URDF`` env overrides first.
    """
    searched: List[str] = []

    for env, kind in (("TERAFOLD_SO101_MJCF", "mjcf"), ("TERAFOLD_SO101_URDF", "urdf")):
        p = os.environ.get(env)
        searched.append(f"${env}")
        if p and os.path.exists(p):
            return {"found": True, "type": kind, "path": p, "meshes": _sibling_meshes(p),
                    "searched": searched, "note": f"using {env}"}

    # Repo-local asset dirs.
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(here, "..", ".."))
    for d in (os.path.join(here, "assets"), os.path.join(repo_root, "assets"),
              os.path.join(repo_root, "assets", "so101")):
        searched.append(d)
        hit = _scan_dir_for_so101(d)
        if hit:
            return {**hit, "searched": searched, "note": "repo asset dir"}

    # Installed packages that may ship SO-101 sim assets.
    for pkg in ("lerobot", "mujoco_menagerie", "robot_descriptions"):
        spec = importlib.util.find_spec(pkg)
        searched.append(pkg)
        if spec is None or not spec.origin:
            continue
        hit = _scan_dir_for_so101(os.path.dirname(spec.origin))
        if hit:
            return {**hit, "searched": searched, "note": f"package {pkg}"}

    return {
        "found": False, "type": None, "path": None, "meshes": [], "searched": searched,
        "note": "No official SO-101 URDF/MJCF/mesh assets found; using TeraFold's "
                "SO-101 approximation. Provide one via $TERAFOLD_SO101_MJCF to override.",
    }


def _sibling_meshes(path: str) -> List[str]:
    d = os.path.dirname(path)
    out: List[str] = []
    for ext in ("*.stl", "*.STL", "*.obj", "*.dae"):
        out += glob.glob(os.path.join(d, "**", ext), recursive=True)
    return out[:200]


def _scan_dir_for_so101(directory: str) -> Optional[Dict[str, Any]]:
    if not directory or not os.path.isdir(directory):
        return None
    for ext, kind in (("*.xml", "mjcf"), ("*.mjcf", "mjcf"), ("*.urdf", "urdf")):
        for p in glob.glob(os.path.join(directory, "**", ext), recursive=True):
            if any(k in p.lower() for k in _ASSET_KEYWORDS):
                return {"found": True, "type": kind, "path": p, "meshes": _sibling_meshes(p)}
    return None


def so101_materials_mjcf() -> str:
    """Distinct materials for the SO-101 links / gripper / accents."""
    return (
        '<material name="so101_link" rgba="0.86 0.87 0.90 1" specular="0.5" shininess="0.35"/>'
        '<material name="so101_accent" rgba="0.90 0.45 0.12 1" specular="0.4" shininess="0.3"/>'
        '<material name="so101_gripper" rgba="0.16 0.17 0.20 1" specular="0.6" shininess="0.4"/>'
    )


def _inertial(mass: float, inertia: float = 1e-4) -> str:
    mass = max(float(mass), 1e-4)
    inertia = max(float(inertia), 1e-6)
    return (f'<inertial pos="0 0 0" mass="{mass:.5f}" '
            f'diaginertia="{inertia:.6f} {inertia:.6f} {inertia:.6f}"/>')


def build_so101_arm(base_xy, base_z: float = 0.0) -> Dict[str, Any]:
    """Build the SO-101 approximation MJCF.

    Returns ``{body_xml, actuator_xml, weld_xml, materials_xml, home_qpos_deg,
    gripper_actuators, gripper_close, weld_body, bodies}``. ``home_qpos_deg`` maps
    joint name -> degrees; the renderer sets these as the initial pose (converting
    to radians). ``gripper_close`` is the finger ctrl value for a fully closed jaw.
    """
    s = SO101_SPEC
    bx, by = float(base_xy[0]), float(base_xy[1])
    lim = s["joint_limits_deg"]
    lup, lfore, lwrist = s["l_upper"], s["l_fore"], s["l_wrist"]
    gmax = s["finger_travel"]

    def rng(j):
        lo, hi = lim[j]
        return f'range="{lo} {hi}"'

    body_xml = (
        f'<body name="so101_base" pos="{bx} {by} {base_z}">'
        f'<geom type="cylinder" size="0.036 0.012" material="so101_link"/>'
        f'<geom type="box" pos="0 0 0.032" size="0.030 0.030 0.022" material="so101_link"/>'
        # j1: base yaw
        f'<body name="arm_j1" pos="0 0 {s["base_height"]}">'
        f'<joint name="j_base" type="hinge" axis="0 0 1" {rng("j_base")}/>{_inertial(0.08)}'
        f'<geom type="box" size="0.027 0.020 0.030" material="so101_accent"/>'
        # j2: shoulder pitch
        f'<body name="arm_j2" pos="0 0 0.030">'
        f'<joint name="j_shoulder" type="hinge" axis="0 1 0" {rng("j_shoulder")}/>{_inertial(0.09)}'
        f'<geom type="capsule" fromto="0 0 0 0 0 {lup:.4f}" size="0.016" material="so101_link"/>'
        f'<geom type="box" pos="0 0 {lup:.4f}" size="0.020 0.017 0.018" material="so101_accent"/>'
        # j3: elbow
        f'<body name="arm_j3" pos="0 0 {lup:.4f}">'
        f'<joint name="j_elbow" type="hinge" axis="0 1 0" {rng("j_elbow")}/>{_inertial(0.07)}'
        f'<geom type="capsule" fromto="0 0 0 0 0 {lfore:.4f}" size="0.014" material="so101_link"/>'
        f'<geom type="box" pos="0 0 {lfore:.4f}" size="0.017 0.015 0.016" material="so101_accent"/>'
        # j4: wrist pitch
        f'<body name="arm_j4" pos="0 0 {lfore:.4f}">'
        f'<joint name="j_wristpitch" type="hinge" axis="0 1 0" {rng("j_wristpitch")}/>{_inertial(0.04)}'
        f'<geom type="capsule" fromto="0 0 0 0 0 {lwrist:.4f}" size="0.012" material="so101_link"/>'
        # j5: wrist roll
        f'<body name="arm_j5" pos="0 0 {lwrist:.4f}">'
        f'<joint name="j_wristroll" type="hinge" axis="0 0 1" {rng("j_wristroll")}/>{_inertial(0.03)}'
        f'<geom type="cylinder" size="0.015 0.010" material="so101_accent"/>'
        # gripper palm
        f'<body name="gripper_base" pos="0 0 0.018">{_inertial(0.03)}'
        f'<geom type="box" size="0.024 0.016 0.012" material="so101_gripper"/>'
        # two parallel fingers (slide inward to close)
        f'<body name="finger_left" pos="0.013 0 0.012">'
        f'<joint name="grip_left" type="slide" axis="-1 0 0" range="0 {gmax}"/>{_inertial(0.008)}'
        f'<geom type="box" pos="0 0 0.016" size="0.004 0.012 0.020" material="so101_gripper"/>'
        f'</body>'
        f'<body name="finger_right" pos="-0.013 0 0.012">'
        f'<joint name="grip_right" type="slide" axis="1 0 0" range="0 {gmax}"/>{_inertial(0.008)}'
        f'<geom type="box" pos="0 0 0.016" size="0.004 0.012 0.020" material="so101_gripper"/>'
        f'</body>'
        f'<site name="ee_site" pos="0 0 0.030" size="0.005" rgba="0.95 0.2 0.2 1"/>'
        f'</body></body></body></body></body></body></body>'
    )

    actuator_xml = (
        '<actuator>'
        f'<position name="a_grip_l" joint="grip_left" kp="30" ctrlrange="0 {gmax}"/>'
        f'<position name="a_grip_r" joint="grip_right" kp="30" ctrlrange="0 {gmax}"/>'
        '</actuator>'
    )
    weld_xml = ('<equality><weld body1="gripper_base" body2="ee" '
                'torquescale="0" solref="0.01 1"/></equality>')

    return {
        "body_xml": body_xml,
        "actuator_xml": actuator_xml,
        "weld_xml": weld_xml,
        "materials_xml": so101_materials_mjcf(),
        "home_qpos_deg": dict(s["home_deg"]),
        "gripper_actuators": list(SO101_GRIPPER_ACTUATORS),
        "gripper_close": float(gmax),
        "weld_body": SO101_WELD_BODY,
        "bodies": list(SO101_BODIES),
    }
