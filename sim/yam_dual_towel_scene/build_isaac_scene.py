#!/usr/bin/env python3
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

NO_ISAAC_MSG = """\
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
