"""Generate the Isaac Sim digital-twin scene package for the dual-YAM rig.

``yam-build-isaac-scene`` writes a self-contained folder::

    <out>/
      scene_config.json      # everything dynamic (dims, poses, paths, cameras)
      build_isaac_scene.py   # standalone Isaac Sim script (no terafold imports)
      README.md              # how to run it locally / in the container

The build script is deliberately a *constant template*: all rig-specific values
travel via ``scene_config.json``, so regenerating after a config change never
rewrites code. It lazy-imports ``SimulationApp`` and explains the container
workflow if Isaac Sim is absent — this module itself never imports Isaac,
never talks to hardware, and works on a laptop with core deps only.

This is a VISUAL digital twin only: table, towel placeholder, YAM placeholders
(or URDF/USD import when configured), three cameras (top/left/right), lights.
No cloth physics tuning, no policy training, no robot I/O.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from terafold.yam.config import load_yam_config

__all__ = [
    "ISAAC_IMAGE",
    "LIVESTREAM_PORTS",
    "generate_isaac_scene",
    "isaac_cloud_commands",
]

ISAAC_IMAGE = "nvcr.io/nvidia/isaac-sim:6.0.1"

#: WebRTC livestream ports the pod/firewall must expose.
LIVESTREAM_PORTS = {"tcp": 49100, "udp": 47998}


# ---------------------------------------------------------------------------
# The standalone build script (constant template — config travels via JSON)
# ---------------------------------------------------------------------------

_BUILD_SCRIPT = '''#!/usr/bin/env python3
"""Build the dual-YAM digital-twin scene in Isaac Sim.

Standalone script: run it with Isaac Sim's bundled interpreter
(``./python.sh build_isaac_scene.py``) or any python inside the Isaac Sim
container. Reads scene_config.json from its own directory.

VISUAL DIGITAL TWIN ONLY — no hardware I/O of any kind exists in this script,
no robot SDK is imported, and nothing here can command a real YAM.
"""

import argparse
import json
import os
import sys

SCENE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCENE_DIR, "scene_config.json")

NO_ISAAC_MSG = """\\
Isaac Sim is not installed in this Python environment (cannot import SimulationApp).

This script must run inside Isaac Sim's interpreter. Easiest path (cloud GPU):

  terafold yam-print-isaac-cloud-commands     # exact docker pull/run commands

or on a machine with Isaac Sim installed:

  cd <isaac-sim install dir> && ./python.sh %s

Nothing was built. (This is expected on a laptop — the scene generator itself
already wrote scene_config.json; only the USD build needs Isaac Sim.)
""" % os.path.abspath(__file__)


def load_scene_config():
    with open(CONFIG_PATH) as fh:
        return json.load(fh)


def build(cfg, headless=True, wait=False, warmup_frames=60):
    # Lazy import: only executed when the script actually runs under Isaac Sim.
    try:
        from isaacsim import SimulationApp  # Isaac Sim >= 4.0 / 5.x / 6.x
    except ImportError:
        try:
            from omni.isaac.kit import SimulationApp  # older Isaac Sim
        except ImportError:
            print(NO_ISAAC_MSG)
            return 2

    sim_app = SimulationApp({"headless": headless})
    # Everything below may only be imported after SimulationApp exists.
    import omni.usd
    from pxr import Gf, UsdGeom, UsdLux, UsdPhysics

    ctx = omni.usd.get_context()
    ctx.new_stage()
    stage = ctx.get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    notes = []

    def xform_box(path, size_xyz, pos_xyz, color, rot_z_deg=0.0):
        cube = UsdGeom.Cube.Define(stage, path)
        cube.GetSizeAttr().Set(1.0)
        xf = UsdGeom.Xformable(cube.GetPrim())
        xf.AddTranslateOp().Set(Gf.Vec3d(*pos_xyz))
        if rot_z_deg:
            xf.AddRotateZOp().Set(rot_z_deg)
        xf.AddScaleOp().Set(Gf.Vec3f(*[s for s in size_xyz]))
        cube.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])
        return cube

    def xform_cylinder(path, radius, height, pos_xyz, color):
        cyl = UsdGeom.Cylinder.Define(stage, path)
        cyl.GetRadiusAttr().Set(radius)
        cyl.GetHeightAttr().Set(height)
        cyl.GetAxisAttr().Set("Z")
        xf = UsdGeom.Xformable(cyl.GetPrim())
        xf.AddTranslateOp().Set(Gf.Vec3d(*pos_xyz))
        cyl.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])
        return cyl

    # -- physics scene + ground (a "world" even without isaacsim.core) -------
    try:
        UsdPhysics.Scene.Define(stage, "/World/physicsScene")
    except Exception as e:
        notes.append("physics scene skipped: %s" % e)
    xform_box("/World/ground", (4.0, 4.0, 0.02), (0.0, 0.0, -0.01), (0.35, 0.35, 0.38))

    # -- table ---------------------------------------------------------------
    t = cfg["table"]
    top_z = t["height_m"]
    xform_box("/World/table/top", (t["width_m"], t["depth_m"], 0.04),
              (0.0, 0.0, top_z - 0.02), (0.55, 0.42, 0.30))
    for i, (sx, sy) in enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)]):
        xform_box("/World/table/leg_%d" % i, (0.05, 0.05, top_z - 0.04),
                  (sx * (t["width_m"] / 2 - 0.05), sy * (t["depth_m"] / 2 - 0.05),
                   (top_z - 0.04) / 2), (0.45, 0.34, 0.25))

    # -- towel placeholder (thin, cloth-colored; real deformable comes later) --
    tw = cfg["towel"]
    xform_box("/World/towel", (tw["width_m"], tw["depth_m"], tw["thickness_m"]),
              (tw["center_xy"][0], tw["center_xy"][1], top_z + tw["thickness_m"] / 2),
              tuple(tw["color_rgb"]), rot_z_deg=tw.get("yaw_deg", 0.0))

    # -- arms: real asset if configured, placeholder towers otherwise ---------
    def add_arm(name, base_pos):
        root = "/World/arms/%s" % name
        usd_path = (cfg["paths"].get("usd") or "").strip()
        urdf_path = (cfg["paths"].get("urdf") or "").strip()
        if usd_path and os.path.exists(usd_path):
            prim = stage.DefinePrim(root, "Xform")
            prim.GetReferences().AddReference(usd_path)
            UsdGeom.Xformable(prim).AddTranslateOp().Set(Gf.Vec3d(*base_pos))
            notes.append("%s: referenced USD %s" % (name, usd_path))
            return
        if urdf_path and os.path.exists(urdf_path):
            try:
                import omni.kit.commands
                # No dest_path: import onto the CURRENT stage (a dest_path would
                # write a separate USD file that never appears in this scene).
                ok, prim_path = omni.kit.commands.execute(
                    "URDFParseAndImportFile", urdf_path=urdf_path)
                prim = stage.GetPrimAtPath(str(prim_path)) if (ok and prim_path) else None
                if prim is not None and prim.IsValid():
                    try:
                        UsdGeom.XformCommonAPI(prim).SetTranslate(Gf.Vec3d(*base_pos))
                        notes.append("%s: imported URDF at %s" % (name, prim_path))
                    except Exception as e:
                        notes.append("%s: imported URDF at %s but could not place it "
                                     "(%s: %s) — move it manually" %
                                     (name, prim_path, type(e).__name__, e))
                    return
                notes.append("%s: URDF import returned %s (%s) -> placeholder"
                             % (name, ok, prim_path))
            except Exception as e:
                notes.append("%s: URDF import failed (%s: %s) -> placeholder"
                             % (name, type(e).__name__, e))
        # placeholder: base cylinder + two stacked links, YAM-ish proportions
        bx, by, bz = base_pos
        xform_cylinder(root + "/base", 0.05, 0.10, (bx, by, bz + 0.05), (0.15, 0.15, 0.17))
        xform_box(root + "/link1", (0.06, 0.06, 0.30), (bx, by, bz + 0.25), (0.85, 0.55, 0.10))
        xform_box(root + "/link2", (0.05, 0.05, 0.25),
                  (bx + 0.10, by, bz + 0.45), (0.85, 0.55, 0.10))
        notes.append("%s: placeholder (configure paths.urdf or paths.usd for the real YAM)"
                     % name)

    arms = cfg["arm_bases"]
    for arm_name, pos in arms.items():
        add_arm(arm_name, tuple(pos))

    # -- cameras (order matters: top, left, right) -----------------------------
    def add_camera(name, pos, rot_xyz_deg):
        cam = UsdGeom.Camera.Define(stage, "/World/cameras/%s" % name)
        cam.GetFocalLengthAttr().Set(18.0)
        xf = UsdGeom.Xformable(cam.GetPrim())
        xf.AddTranslateOp().Set(Gf.Vec3d(*pos))
        xf.AddRotateXYZOp().Set(Gf.Vec3f(*rot_xyz_deg))

    for cam_name in cfg["camera_order"]:
        spec = cfg["cameras"].get(cam_name)
        if spec is None:
            notes.append("camera %r has no spec in scene_config.json; skipped" % cam_name)
            continue
        add_camera(cam_name, tuple(spec["position"]), tuple(spec["rotation_xyz_deg"]))

    # -- lights ----------------------------------------------------------------
    dome = UsdLux.DomeLight.Define(stage, "/World/lights/dome")
    dome.GetIntensityAttr().Set(600.0)
    sun = UsdLux.DistantLight.Define(stage, "/World/lights/sun")
    sun.GetIntensityAttr().Set(2500.0)
    UsdGeom.Xformable(sun.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-50.0, 20.0, 0.0))

    # -- save + warm up ---------------------------------------------------------
    usd_out = os.path.join(SCENE_DIR, cfg["usd_filename"])
    saved = False
    try:
        saved = bool(ctx.save_as_stage(usd_out))
    except Exception:
        pass
    if not saved:
        try:
            saved = stage.GetRootLayer().Export(usd_out)
        except Exception as e:
            notes.append("USD save failed: %s: %s" % (type(e).__name__, e))

    for _ in range(max(0, warmup_frames)):
        sim_app.update()

    print("--- yam digital twin scene ---")
    print("cameras (order): %s" % cfg["camera_order"])
    print("usd: %s" % (usd_out if saved else "NOT SAVED"))
    for n in notes:
        print("note: %s" % n)
    print("simulation only: no hardware was (or can be) commanded by this script.")

    if wait:
        # A bare headless SimulationApp does NOT stream: the livestream
        # extension must be enabled explicitly (see the official
        # standalone_examples livestream.py).
        streaming = False
        try:
            sim_app.set_setting("/app/window/drawMouse", True)
            try:
                from isaacsim.core.experimental.utils.app import enable_extension  # 6.x
            except ImportError:
                from isaacsim.core.utils.extensions import enable_extension  # 4.x/5.x
            for ext in ("omni.kit.livestream.app", "omni.kit.livestream.webrtc"):
                try:
                    if enable_extension(ext) is not False:
                        streaming = True
                        print("livestream extension enabled: %s" % ext)
                        break
                except Exception:
                    continue
        except Exception as e:
            print("could not enable livestream (%s: %s)" % (type(e).__name__, e))
        if not streaming:
            print("WebRTC livestream is NOT active — to view the scene, quit and "
                  "run ./runheadless.sh -v instead, then open the saved USD.")
        else:
            print("connect the Isaac Sim WebRTC Streaming Client now "
                  "(TCP 49100 / UDP 47998).")
        print("running until Ctrl-C ...")
        try:
            while sim_app.is_running():
                sim_app.update()
        except KeyboardInterrupt:
            pass

    sim_app.close()
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-headless", action="store_true",
                    help="Open the full UI (default: headless).")
    ap.add_argument("--wait", action="store_true",
                    help="Keep the app running after building (for livestream viewing).")
    ap.add_argument("--warmup-frames", type=int, default=60)
    args = ap.parse_args()
    cfg = load_scene_config()
    return build(cfg, headless=not args.no_headless, wait=args.wait,
                 warmup_frames=args.warmup_frames)


if __name__ == "__main__":
    sys.exit(main())
'''


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def _scene_config(cfg, config_path: str) -> Dict[str, Any]:
    t = cfg.table or {}
    table = {"width_m": float(t.get("width_m", 1.2)),
             "depth_m": float(t.get("depth_m", 0.8)),
             "height_m": float(t.get("height_m", 0.75))}
    bad_dims = [k for k, v in table.items() if not v > 0]
    if bad_dims:
        raise ValueError(f"table dimensions must be positive, got "
                         f"{ {k: table[k] for k in bad_dims} } — fix the config")
    top_z = table["height_m"]
    # Bimanual mount: bases on the table surface, mirrored across x.
    # Clamped >= 0.05 so a mis-measured tiny depth can never mirror the layout.
    arm_y = round(max(0.05, min(0.35, table["depth_m"] / 2 - 0.05)), 3)
    arm_bases = {}
    for i, arm in enumerate(cfg.arms):
        side = -1.0 if i == 0 else 1.0
        arm_bases[arm] = [-table["width_m"] / 2 + 0.12, side * arm_y, top_z]
    # USD cameras look down local -Z: pitching X by +55 aims toward +Y, so the
    # -Y (left) camera pitches +55 and the +Y (right) camera pitches -55.
    cameras = {
        "top": {"position": [0.0, 0.0, top_z + 0.85],
                "rotation_xyz_deg": [0.0, 0.0, 0.0],
                "role": "overhead"},
        "left": {"position": [0.15, -arm_y, top_z + 0.35],
                 "rotation_xyz_deg": [55.0, 0.0, 25.0],
                 "role": "left wrist placeholder (rigid until URDF attach)"},
        "right": {"position": [0.15, arm_y, top_z + 0.35],
                  "rotation_xyz_deg": [-55.0, 0.0, -25.0],
                  "role": "right wrist placeholder (rigid until URDF attach)"},
    }
    # Any extra/renamed cameras in camera_order still get a (generic) spec so
    # the build script never KeyErrors on the GPU pod.
    for extra in cfg.camera_order:
        if extra not in cameras:
            cameras[extra] = {"position": [-table["width_m"] / 2 - 0.3, 0.0, top_z + 0.5],
                              "rotation_xyz_deg": [0.0, -60.0, 0.0],
                              "role": f"generic placeholder for unrecognized camera "
                                      f"{extra!r} — tune manually"}
    return {
        "scene": "yam_dual_towel_scene",
        "robot": cfg.robot,
        "arms": cfg.arms,
        "arm_bases": arm_bases,
        "camera_order": cfg.camera_order,
        "cameras": cameras,
        "norm_tag": cfg.norm_tag,
        "table": table,
        "towel": {"width_m": 0.40, "depth_m": 0.30, "thickness_m": 0.006,
                  "center_xy": [0.10, 0.0], "yaw_deg": 0.0,
                  "color_rgb": [0.16, 0.42, 0.80],
                  "note": "rigid placeholder; deformable cloth comes in a later sprint"},
        "workspace_bounds": cfg.workspace_bounds,
        "paths": {"urdf": (cfg.paths or {}).get("urdf"),
                  "usd": (cfg.paths or {}).get("usd")},
        "usd_filename": "yam_dual_towel_scene.usd",
        "simulation_only": True,
        "hardware_commanded": False,
        "generated_from": str(config_path),
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }


def _scene_readme(out_dir: str) -> str:
    out_dir = str(Path(out_dir).resolve())  # instructions must survive a `cd`
    return f"""# YAM dual-arm digital twin scene (visual only)

Generated by `terafold yam-build-isaac-scene`. Contents:

- `scene_config.json` — table/towel/arm/camera layout derived from
  `configs/yam_dual_reference.yaml`. Regenerate rather than hand-editing.
- `build_isaac_scene.py` — standalone Isaac Sim script (needs Isaac Sim's
  Python; it explains itself and exits cleanly anywhere else).

## Run it

With Isaac Sim installed locally:

    cd <isaac-sim install dir>
    ./python.sh {out_dir}/build_isaac_scene.py

On a cloud GPU, get the exact container commands with:

    terafold yam-print-isaac-cloud-commands

Add `--wait` to keep the app alive for WebRTC livestream viewing.
Output USD: `yam_dual_towel_scene.usd` (in this directory).

## Safety

Visual digital twin only. This scene contains no controllers, imports no robot
SDK, and cannot command real YAM hardware. Do not use it to justify autonomous
motion on the rig.
"""


def generate_isaac_scene(out_dir: str, config_path: Optional[str] = None) -> Dict[str, Any]:
    """Write scene_config.json + build_isaac_scene.py + README.md into ``out_dir``.

    Pure file generation — needs neither Isaac Sim nor hardware. Statuses:
    ``ok`` | ``bad_config``.
    """
    from terafold.yam.safety import ShadowModeViolation

    try:
        cfg = load_yam_config(config_path)
        scene_cfg = _scene_config(cfg, config_path or str(cfg.source))
    except ShadowModeViolation as e:
        return {"status": "bad_config", "message": str(e)}
    except (FileNotFoundError, ValueError, TypeError) as e:
        return {"status": "bad_config", "message": str(e)}

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    (out / "scene_config.json").write_text(json.dumps(scene_cfg, indent=2))
    (out / "build_isaac_scene.py").write_text(_BUILD_SCRIPT)
    (out / "README.md").write_text(_scene_readme(str(out)))

    return {"status": "ok", "out": str(out),
            "files": ["scene_config.json", "build_isaac_scene.py", "README.md"],
            "usd_filename": scene_cfg["usd_filename"],
            "camera_order": scene_cfg["camera_order"],
            "arms": scene_cfg["arms"],
            "simulation_only": True}


# ---------------------------------------------------------------------------
# Cloud / container commands
# ---------------------------------------------------------------------------


def isaac_cloud_commands(scene_dir: str = "sim/yam_dual_towel_scene",
                         image: str = ISAAC_IMAGE) -> List[str]:
    """Exact commands to run the scene build in the Isaac Sim container.

    Cache-mount layout matches the Isaac Sim 6.0 container docs (the 6.0 image
    runs as uid 1234 and keeps caches under /isaac-sim/.cache — the 4.x
    /root/.cache layout no longer persists anything).
    """
    cache = "~/docker/isaac-sim"
    scene_mount_src = (scene_dir if os.path.isabs(os.path.expanduser(scene_dir))
                       else f"$(pwd)/{scene_dir}")
    run = (
        f"docker run --name isaac-sim --entrypoint bash -it --gpus all "
        f'-u 1234:1234 -e "ACCEPT_EULA=Y" -e "PRIVACY_CONSENT=Y" --network=host \\\n'
        f"  -v {cache}/cache/main:/isaac-sim/.cache:rw \\\n"
        f"  -v {cache}/cache/computecache:/isaac-sim/.nv/ComputeCache:rw \\\n"
        f"  -v {cache}/logs:/isaac-sim/.nvidia-omniverse/logs:rw \\\n"
        f"  -v {cache}/config:/isaac-sim/.nvidia-omniverse/config:rw \\\n"
        f"  -v {cache}/data:/isaac-sim/.local/share/ov/data:rw \\\n"
        f"  -v {cache}/pkg:/isaac-sim/.local/share/ov/pkg:rw \\\n"
        f"  -v ~/.cache/ov/hub:/var/cache/hub:rw \\\n"
        f"  -v {scene_mount_src}:/workspace/yam_scene:rw \\\n"
        f"  {image}"
    )
    return [
        "# 0. One-time host prep (the 6.0 container runs as uid 1234, so the",
        "#    mounted dirs must be writable by it):",
        f"mkdir -p {cache}/cache/main {cache}/cache/computecache {cache}/logs "
        f"{cache}/config {cache}/data {cache}/pkg ~/.cache/ov/hub",
        f"chmod -R a+rw {cache} {scene_mount_src.replace('$(pwd)/', '')}",
        "",
        "# 1. On the cloud GPU host (NVIDIA driver + nvidia-container-toolkit required):",
        f"docker pull {image}",
        "",
        "# 2. Start the container (bash entrypoint; host networking for livestream;",
        f"#    livestream needs TCP {LIVESTREAM_PORTS['tcp']} and UDP "
        f"{LIVESTREAM_PORTS['udp']} reachable):",
        run,
        "",
        "# 3. Inside the container — compatibility check first (headless flags):",
        "./isaac-sim.compatibility_check.sh --no-window --/app/quitAfter=10",
        "",
        "# 4. Build the scene (writes yam_dual_towel_scene.usd next to the script):",
        "./python.sh /workspace/yam_scene/build_isaac_scene.py",
        "",
        "# 5. Headless WebRTC livestream (then connect the Isaac Sim WebRTC",
        "#    Streaming Client to the pod's public IP):",
        "./runheadless.sh -v",
        "",
        "# 5b. Or build and keep the app alive, attempting to enable the WebRTC",
        "#     livestream extension from the script (watch the console: if it says",
        "#     livestream is NOT active, use ./runheadless.sh -v from step 5):",
        "./python.sh /workspace/yam_scene/build_isaac_scene.py --wait",
        "",
        "# docs: docs/YAM_ISAAC_SIM.md (ports, WebRTC client, troubleshooting)",
    ]
