import argparse
import json
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher


OUT_DIR = Path("/workspace/leisaac/tera_checkpoints/clean_towel_v2")
DEFAULT_USD_PATH = OUT_DIR / "clean_rect_towel_usd_v2.usda"
DEFAULT_CAPTURE_PATH = OUT_DIR / "clean_rect_towel_usd_v2_capture.npz"
DEFAULT_SUMMARY_PATH = OUT_DIR / "test_clean_rect_towel_usd_v2_summary.json"
DEFAULT_STATUS_PATH = OUT_DIR / "STATUS.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test the clean rectangular towel cloth USD in Isaac.")
    parser.add_argument("--usd_path", type=Path, default=DEFAULT_USD_PATH)
    parser.add_argument("--out_dir", type=Path, default=OUT_DIR)
    parser.add_argument("--width_x", type=float, default=0.68)
    parser.add_argument("--height_y", type=float, default=0.38)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--tolerance", type=float, default=0.03)
    parser.add_argument("--gravity", type=float, default=-9.81)
    parser.add_argument("--spawn_z", type=float, default=0.02)
    parser.add_argument("--ground_z", type=float, default=0.0)
    parser.add_argument("--disable_ground", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if hasattr(args, "headless"):
        args.headless = True
    return args


def size_metrics(points: np.ndarray) -> dict[str, object]:
    finite = points[np.isfinite(points).all(axis=1)]
    if finite.size == 0:
        raise RuntimeError("No finite cloth points were returned.")
    cmin = finite.min(axis=0)
    cmax = finite.max(axis=0)
    size = cmax - cmin
    return {
        "min": [float(x) for x in cmin],
        "max": [float(x) for x in cmax],
        "center": [float(x) for x in ((cmin + cmax) / 2.0)],
        "size": [float(x) for x in size],
        "width_x": float(size[0]),
        "height_y": float(size[1]),
        "thickness_z": float(size[2]),
        "num_finite": int(finite.shape[0]),
    }


def write_summary(summary: dict[str, object], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in (out_dir / DEFAULT_SUMMARY_PATH.name, out_dir / DEFAULT_STATUS_PATH.name):
        with path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    try:
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleClothPrim, SingleParticleSystem
        from isaacsim.core.utils.stage import add_reference_to_stage, create_new_stage, get_current_stage, update_stage
        from pxr import Gf, UsdGeom

        if not args.usd_path.exists():
            summary = {
                "status": "CLEAN_RECT_TOWEL_USD_V2_TEST_FAILED",
                "reason": "missing_usd",
                "usd_path": str(args.usd_path),
            }
            write_summary(summary, args.out_dir)
            return

        World.clear_instance()
        create_new_stage()
        stage = get_current_stage()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        root = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(root.GetPrim())
        UsdGeom.Xform.Define(stage, "/World/Scene")

        world = World(stage_units_in_meters=1.0, backend="torch", device=args.device)
        world.get_physics_context().set_gravity(float(args.gravity))
        if not args.disable_ground:
            world.scene.add_default_ground_plane(z_position=float(args.ground_z))

        reference_prim = add_reference_to_stage(str(args.usd_path), "/World/Scene/cloth")
        UsdGeom.Xformable(reference_prim).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, float(args.spawn_z)))
        update_stage()

        mesh_path = "/World/Scene/cloth/cloth/sim_cloth/sim_cloth"
        particle_system_path = "/World/Scene/cloth/cloth/ParticleSystem"
        particle_system = SingleParticleSystem(
            prim_path=particle_system_path,
            simulation_owner=world.get_physics_context().prim_path,
        )
        particle_system.set_simulation_owner(world.get_physics_context().prim_path)
        cloth = SingleClothPrim(prim_path=mesh_path, particle_system=particle_system)
        world.scene.add(cloth)
        world.reset(soft=False)
        update_stage()

        reset_points = cloth._cloth_prim_view.get_world_positions()[0].detach().cpu().numpy().astype(np.float32)
        reset_metrics = size_metrics(reset_points)

        for _ in range(args.steps):
            world.step(render=False)

        after_points = cloth._cloth_prim_view.get_world_positions()[0].detach().cpu().numpy().astype(np.float32)
        after_metrics = size_metrics(after_points)

        capture_path = args.out_dir / DEFAULT_CAPTURE_PATH.name
        np.savez_compressed(
            capture_path,
            reset_points=reset_points,
            after_100_points=after_points,
            target_width_x=np.asarray([args.width_x], dtype=np.float32),
            target_height_y=np.asarray([args.height_y], dtype=np.float32),
        )

        width_error = abs(after_metrics["width_x"] - args.width_x)
        height_error = abs(after_metrics["height_y"] - args.height_y)
        looks_rectangular = width_error < args.tolerance and height_error < args.tolerance
        status = "CLEAN_RECT_TOWEL_USD_V2_TEST_OK" if looks_rectangular else "CLEAN_RECT_TOWEL_USD_V2_TEST_FAILED"
        summary = {
            "status": status,
            "usd_path": str(args.usd_path),
            "capture_path": str(capture_path),
            "referenced_prim_path": reference_prim.GetPath().pathString,
            "mesh_prim_path": mesh_path,
            "particle_system_prim_path": particle_system_path,
            "device": str(args.device),
            "gravity_z": float(args.gravity),
            "spawn_z": float(args.spawn_z),
            "ground_enabled": not bool(args.disable_ground),
            "ground_z": None if args.disable_ground else float(args.ground_z),
            "num_steps": int(args.steps),
            "num_particles": int(after_points.shape[0]),
            "reset_metrics": reset_metrics,
            "after_100_zero_steps_metrics": after_metrics,
            "target_width_x": float(args.width_x),
            "target_height_y": float(args.height_y),
            "width_error_after_100": float(width_error),
            "height_error_after_100": float(height_error),
            "looks_rectangular_by_metric": bool(looks_rectangular),
            "honest_note": "This loads the v2 clean rectangular towel USD into an Isaac stage and steps physics with no actions.",
        }
        write_summary(summary, args.out_dir)
        world.stop()
        World.clear_instance()

    except Exception as exc:
        summary = {
            "status": "CLEAN_RECT_TOWEL_USD_V2_TEST_FAILED",
            "usd_path": str(args.usd_path),
            "device": str(getattr(args, "device", "unknown")),
            "exception_type": exc.__class__.__name__,
            "exception": repr(exc),
            "honest_note": "The smoke test could not complete; no rectangular success metric is claimed.",
        }
        write_summary(summary, args.out_dir)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
