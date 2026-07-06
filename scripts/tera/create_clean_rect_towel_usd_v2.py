import argparse
import json
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher


OUT_DIR = Path("/workspace/leisaac/tera_checkpoints/clean_towel_v2")
DEFAULT_USD_PATH = OUT_DIR / "clean_rect_towel_usd_v2.usda"
DEFAULT_GRID_PATH = OUT_DIR / "clean_rect_towel_grid_v2.npz"
DEFAULT_OBJ_PATH = OUT_DIR / "clean_rect_towel_grid_mesh_v2.obj"
DEFAULT_SUMMARY_PATH = OUT_DIR / "create_clean_rect_towel_usd_v2_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a clean rectangular PhysX towel cloth USD.")
    parser.add_argument("--out_dir", type=Path, default=OUT_DIR)
    parser.add_argument("--usd_path", type=Path, default=DEFAULT_USD_PATH)
    parser.add_argument("--width_x", type=float, default=0.68)
    parser.add_argument("--height_y", type=float, default=0.38)
    parser.add_argument("--spacing", type=float, default=0.01)
    parser.add_argument("--z", type=float, default=0.0)
    parser.add_argument("--stretch_stiffness", type=float, default=10000.0)
    parser.add_argument("--bend_stiffness", type=float, default=7500.0)
    parser.add_argument("--shear_stiffness", type=float, default=1500.0)
    parser.add_argument("--spring_damping", type=float, default=10.0)
    parser.add_argument("--particle_contact_offset", type=float, default=0.005)
    parser.add_argument("--mass", type=float, default=0.01)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if hasattr(args, "headless"):
        args.headless = True
    return args


def build_grid(width_x: float, height_y: float, spacing: float, z: float) -> dict[str, np.ndarray | int | float]:
    nx = int(round(width_x / spacing)) + 1
    ny = int(round(height_y / spacing)) + 1
    if nx < 2 or ny < 2:
        raise ValueError("Grid must have at least 2 vertices in each dimension.")

    xs = np.linspace(-width_x / 2.0, width_x / 2.0, nx, dtype=np.float32)
    ys = np.linspace(-height_y / 2.0, height_y / 2.0, ny, dtype=np.float32)
    gx, gy = np.meshgrid(xs, ys, indexing="xy")
    points = np.column_stack(
        [
            gx.reshape(-1),
            gy.reshape(-1),
            np.full(nx * ny, z, dtype=np.float32),
        ]
    ).astype(np.float32)
    normals = np.tile(np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32), (nx * ny, 1))
    velocities = np.zeros_like(points, dtype=np.float32)

    face_counts: list[int] = []
    face_indices: list[int] = []
    for y in range(ny - 1):
        for x in range(nx - 1):
            a = y * nx + x
            b = a + 1
            c = (y + 1) * nx + x
            d = c + 1
            face_counts.extend([3, 3])
            face_indices.extend([a, b, d, a, d, c])

    return {
        "nx": nx,
        "ny": ny,
        "points": points,
        "normals": normals,
        "velocities": velocities,
        "face_counts": np.asarray(face_counts, dtype=np.int32),
        "face_indices": np.asarray(face_indices, dtype=np.int32),
    }


def add_spring(
    springs: list[tuple[int, int]],
    lengths: list[float],
    stiffnesses: list[float],
    dampings: list[float],
    points: np.ndarray,
    a: int,
    b: int,
    stiffness: float,
    damping: float,
) -> None:
    springs.append((a, b))
    lengths.append(float(np.linalg.norm(points[a] - points[b])))
    stiffnesses.append(float(stiffness))
    dampings.append(float(damping))


def build_springs(
    points: np.ndarray,
    nx: int,
    ny: int,
    stretch_stiffness: float,
    bend_stiffness: float,
    shear_stiffness: float,
    damping: float,
) -> dict[str, np.ndarray]:
    springs: list[tuple[int, int]] = []
    lengths: list[float] = []
    stiffnesses: list[float] = []
    dampings: list[float] = []

    for y in range(ny):
        for x in range(nx):
            i = y * nx + x
            if x > 0:
                add_spring(springs, lengths, stiffnesses, dampings, points, i, i - 1, stretch_stiffness, damping)
            if y > 0:
                add_spring(springs, lengths, stiffnesses, dampings, points, i, i - nx, stretch_stiffness, damping)
            if x > 0 and y > 0:
                add_spring(springs, lengths, stiffnesses, dampings, points, i, i - nx - 1, shear_stiffness, damping)
            if x < nx - 1 and y > 0:
                add_spring(springs, lengths, stiffnesses, dampings, points, i, i - nx + 1, shear_stiffness, damping)
            if x > 1:
                add_spring(springs, lengths, stiffnesses, dampings, points, i, i - 2, bend_stiffness, damping)
            if y > 1:
                add_spring(springs, lengths, stiffnesses, dampings, points, i, i - 2 * nx, bend_stiffness, damping)

    return {
        "spring_indices": np.asarray(springs, dtype=np.int32),
        "spring_rest_lengths": np.asarray(lengths, dtype=np.float32),
        "spring_stiffnesses": np.asarray(stiffnesses, dtype=np.float32),
        "spring_dampings": np.asarray(dampings, dtype=np.float32),
    }


def write_obj(path: Path, points: np.ndarray, face_indices: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# Clean rectangular towel cloth grid v2\n")
        for p in points:
            f.write(f"v {p[0]:.8f} {p[1]:.8f} {p[2]:.8f}\n")
        for i in range(0, len(face_indices), 3):
            a, b, c = face_indices[i : i + 3]
            f.write(f"f {a + 1} {b + 1} {c + 1}\n")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.usd_path.parent.mkdir(parents=True, exist_ok=True)

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    from omni.physx.scripts import particleUtils, physicsUtils
    from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

    grid = build_grid(args.width_x, args.height_y, args.spacing, args.z)
    springs = build_springs(
        grid["points"],
        int(grid["nx"]),
        int(grid["ny"]),
        args.stretch_stiffness,
        args.bend_stiffness,
        args.shear_stiffness,
        args.spring_damping,
    )

    stage = Usd.Stage.CreateNew(str(args.usd_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Root")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.Xform.Define(stage, "/Root/cloth")
    UsdGeom.Xform.Define(stage, "/Root/cloth/sim_cloth")

    particle_system_path = Sdf.Path("/Root/cloth/ParticleSystem")
    particle_system = particleUtils.add_physx_particle_system(
        stage,
        particle_system_path,
        particle_system_enabled=True,
        particle_contact_offset=args.particle_contact_offset,
        solid_rest_offset=args.particle_contact_offset * 0.6,
        fluid_rest_offset=args.particle_contact_offset * 0.6,
        contact_offset=args.particle_contact_offset,
        rest_offset=args.particle_contact_offset * 0.6,
        solver_position_iterations=16,
        max_velocity=5.0,
        non_particle_collision_enabled=True,
    )
    material_path = Sdf.Path("/Root/cloth/ParticleSystem/PhysicsMaterial")
    particleUtils.add_pbd_particle_material(
        stage,
        material_path,
        friction=0.72,
        particle_friction_scale=0.6,
        damping=0.0001,
        adhesion=0.0005,
        particle_adhesion_scale=0.2,
        adhesion_offset_scale=0.0,
        gravity_scale=1.0,
        density=0.0,
        cfl_coefficient=1.0,
    )
    physicsUtils.add_physics_material_to_prim(stage, particle_system.GetPrim(), material_path)

    mesh_path = Sdf.Path("/Root/cloth/sim_cloth/sim_cloth")
    mesh = UsdGeom.Mesh.Define(stage, mesh_path)
    mesh.CreateDoubleSidedAttr().Set(True)
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    mesh.CreatePointsAttr().Set(Vt.Vec3fArray.FromNumpy(grid["points"]))
    mesh.CreateNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(grid["normals"]))
    mesh.CreateVelocitiesAttr().Set(Vt.Vec3fArray.FromNumpy(grid["velocities"]))
    mesh.CreateFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(grid["face_counts"]))
    mesh.CreateFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(grid["face_indices"]))
    mesh.CreateExtentAttr().Set(
        [
            Gf.Vec3f(-args.width_x / 2.0, -args.height_y / 2.0, args.z),
            Gf.Vec3f(args.width_x / 2.0, args.height_y / 2.0, args.z),
        ]
    )

    cloth_api = PhysxSchema.PhysxParticleClothAPI.Apply(mesh.GetPrim())
    particle_api = PhysxSchema.PhysxParticleAPI.Apply(mesh.GetPrim())
    cloth_api.GetRestPointsAttr().Set(Vt.Vec3fArray.FromNumpy(grid["points"]))
    cloth_api.CreateSpringIndicesAttr().Set(Vt.Vec2iArray.FromNumpy(springs["spring_indices"]))
    cloth_api.CreateSpringStiffnessesAttr().Set(Vt.FloatArray.FromNumpy(springs["spring_stiffnesses"]))
    cloth_api.CreateSpringDampingsAttr().Set(Vt.FloatArray.FromNumpy(springs["spring_dampings"]))
    cloth_api.CreateSpringRestLengthsAttr().Set(Vt.FloatArray.FromNumpy(springs["spring_rest_lengths"]))
    cloth_api.CreateSelfCollisionFilterAttr().Set(True)
    cloth_api.CreatePressureAttr().Set(0.0)
    particle_api.CreateParticleSystemRel().SetTargets([particle_system_path])
    particle_api.CreateParticleEnabledAttr().Set(True)
    particle_api.CreateSelfCollisionAttr().Set(True)
    particle_api.CreateParticleGroupAttr().Set(0)

    auto_api = PhysxSchema.PhysxAutoParticleClothAPI.Apply(mesh.GetPrim())
    auto_api.CreateDisableMeshWeldingAttr().Set(True)
    auto_api.CreateSpringStretchStiffnessAttr().Set(args.stretch_stiffness)
    auto_api.CreateSpringBendStiffnessAttr().Set(args.bend_stiffness)
    auto_api.CreateSpringShearStiffnessAttr().Set(args.shear_stiffness)
    auto_api.CreateSpringDampingAttr().Set(args.spring_damping)

    mass_api = UsdPhysics.MassAPI.Apply(mesh.GetPrim())
    mass_api.CreateMassAttr().Set(args.mass)
    mass_api.CreateDensityAttr().Set(args.mass)

    looks = UsdGeom.Scope.Define(stage, "/Root/cloth/sim_cloth/Looks")
    material = UsdShade.Material.Define(stage, looks.GetPath().AppendChild("TowelBlue"))
    shader = UsdShade.Shader.Define(stage, material.GetPath().AppendChild("Shader"))
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.18, 0.45, 0.80))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)

    stage.GetRootLayer().Save()

    npz_path = args.out_dir / DEFAULT_GRID_PATH.name
    obj_path = args.out_dir / DEFAULT_OBJ_PATH.name
    np.savez_compressed(
        npz_path,
        points=grid["points"],
        normals=grid["normals"],
        velocities=grid["velocities"],
        face_counts=grid["face_counts"],
        face_indices=grid["face_indices"],
        spring_indices=springs["spring_indices"],
        spring_rest_lengths=springs["spring_rest_lengths"],
        spring_stiffnesses=springs["spring_stiffnesses"],
        spring_dampings=springs["spring_dampings"],
        width_x=np.asarray([args.width_x], dtype=np.float32),
        height_y=np.asarray([args.height_y], dtype=np.float32),
        nx=np.asarray([grid["nx"]], dtype=np.int32),
        ny=np.asarray([grid["ny"]], dtype=np.int32),
    )
    write_obj(obj_path, grid["points"], grid["face_indices"])

    bounds_min = grid["points"].min(axis=0)
    bounds_max = grid["points"].max(axis=0)
    summary = {
        "status": "CLEAN_RECT_TOWEL_USD_V2_CREATED",
        "usd_path": str(args.usd_path),
        "grid_npz_path": str(npz_path),
        "grid_obj_path": str(obj_path),
        "mesh_prim_path": "/Root/cloth/sim_cloth/sim_cloth",
        "particle_system_prim_path": "/Root/cloth/ParticleSystem",
        "num_points": int(grid["points"].shape[0]),
        "num_triangles": int(len(grid["face_indices"]) // 3),
        "num_springs": int(springs["spring_indices"].shape[0]),
        "grid_nx": int(grid["nx"]),
        "grid_ny": int(grid["ny"]),
        "target_width_x": float(args.width_x),
        "target_height_y": float(args.height_y),
        "actual_width_x": float(bounds_max[0] - bounds_min[0]),
        "actual_height_y": float(bounds_max[1] - bounds_min[1]),
        "actual_thickness_z": float(bounds_max[2] - bounds_min[2]),
        "spacing_x": float(args.width_x / (int(grid["nx"]) - 1)),
        "spacing_y": float(args.height_y / (int(grid["ny"]) - 1)),
        "spring_stretch_stiffness": float(args.stretch_stiffness),
        "spring_bend_stiffness": float(args.bend_stiffness),
        "spring_shear_stiffness": float(args.shear_stiffness),
        "spring_damping": float(args.spring_damping),
        "particle_contact_offset": float(args.particle_contact_offset),
        "honest_note": "Synthetic clean rectangular towel PhysX cloth USD with rectangular grid topology and generated spring constraints.",
    }
    summary_path = args.out_dir / DEFAULT_SUMMARY_PATH.name
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)

    simulation_app.close()


if __name__ == "__main__":
    main()
