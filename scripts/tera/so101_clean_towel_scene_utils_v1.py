from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


TASK_ID = "LeIsaac-SO101-FoldCloth-BiArm-Direct-v0"
PYTHON_SH = "/workspace/isaaclab/_isaac_sim/python.sh"
OUT_DIR = Path("/workspace/leisaac/tera_checkpoints/so101_clean_towel_v1")
CLEAN_TOWEL_USD_PATH = Path(
    "/workspace/leisaac/tera_checkpoints/clean_towel_v2/clean_rect_towel_usd_v2.usda"
)
CLEAN_TOWEL_GRID_PATH = Path("/workspace/leisaac/tera_checkpoints/clean_towel_v2/clean_rect_towel_grid_v2.npz")

CLEAN_TOWEL_ROOT_NAME = "clean_towel"
CLEAN_TOWEL_MESH_SUFFIX = "cloth/sim_cloth/sim_cloth"
CLEAN_TOWEL_PARTICLE_SYSTEM_SUFFIX = "cloth/ParticleSystem"
DEFAULT_TOWEL_TRANSLATION = (-0.90, 8.42, 3.40)
DEFAULT_SUPPORT_SIZE = (1.0, 0.65, 0.012)
DEFAULT_SUPPORT_TOP_Z_OFFSET = -0.002

COMPILE_COMMAND = (
    f"{PYTHON_SH} -m py_compile "
    "scripts/tera/so101_clean_towel_scene_utils_v1.py "
    "scripts/tera/test_so101_clean_towel_scene_v1.py "
    "scripts/tera/run_so101_clean_towel_assisted_fold_v1.py"
)


def force_headless_no_cameras(args: Any) -> None:
    """Force a non-rendering Isaac launch, overriding environment defaults."""
    os.environ["HEADLESS"] = "1"
    os.environ["LIVESTREAM"] = "0"
    os.environ["ENABLE_CAMERAS"] = "0"
    if hasattr(args, "headless"):
        args.headless = True
    if hasattr(args, "livestream"):
        args.livestream = 0
    if hasattr(args, "enable_cameras"):
        args.enable_cameras = False


def command_run_string() -> str:
    return " ".join([PYTHON_SH, *sys.argv])


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if hasattr(value, "detach") and callable(value.detach):
        return jsonable(value.detach().cpu().numpy())
    return value


def write_json(path: Path, payload: dict[str, Any], print_payload: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_payload = jsonable(payload)
    with path.open("w", encoding="utf-8") as f:
        json.dump(safe_payload, f, indent=2)
    if print_payload:
        print(json.dumps(safe_payload, indent=2), flush=True)


def points_to_numpy(points: Any) -> np.ndarray:
    if hasattr(points, "detach") and callable(points.detach):
        points = points.detach().cpu().numpy()
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr.reshape(-1, arr.shape[-1])
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise RuntimeError(f"Expected particle positions with shape (N, 3), got {arr.shape}.")
    return arr


def size_metrics(points: Any) -> dict[str, Any]:
    arr = points_to_numpy(points)
    finite = arr[np.isfinite(arr).all(axis=1)]
    if finite.size == 0:
        raise RuntimeError("No finite clean towel particle positions were returned.")
    cmin = finite.min(axis=0)
    cmax = finite.max(axis=0)
    center = (cmin + cmax) * 0.5
    size = cmax - cmin
    return {
        "min": [float(x) for x in cmin],
        "max": [float(x) for x in cmax],
        "center": [float(x) for x in center],
        "center_z": float(center[2]),
        "size": [float(x) for x in size],
        "width_x": float(size[0]),
        "height_y": float(size[1]),
        "thickness_z": float(size[2]),
        "num_particles": int(arr.shape[0]),
        "num_finite": int(finite.shape[0]),
    }


def looks_rectangular(metrics: dict[str, Any], target_width: float, target_height: float, tolerance: float) -> bool:
    return (
        abs(float(metrics["width_x"]) - float(target_width)) < float(tolerance)
        and abs(float(metrics["height_y"]) - float(target_height)) < float(tolerance)
    )


def _install_fold_env_runtime_patches() -> None:
    import torch
    import leisaac.tasks.fold_cloth.direct.fold_cloth_bi_arm_env as fc

    def dummy_observations(self):
        return {"policy": torch.zeros((self.num_envs, 1), dtype=torch.float32, device=self.device)}

    def zero_rewards(self):
        return torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

    def never_success(self):
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    fc.FoldClothBiArmEnv._get_observations = dummy_observations
    fc.FoldClothBiArmEnv._get_rewards = zero_rewards
    fc.FoldClothBiArmEnv._check_success = never_success


def _install_clean_towel_scene_patch(
    clean_usd_path: Path,
    towel_translation: tuple[float, float, float],
    root_name: str = CLEAN_TOWEL_ROOT_NAME,
) -> None:
    import leisaac.tasks.fold_cloth.direct.fold_cloth_bi_arm_env as fc

    patch_key = "_tera_clean_towel_scene_patch_v1"
    if getattr(fc.FoldClothBiArmEnv, patch_key, False):
        return

    original_setup_scene = fc.FoldClothBiArmEnv._setup_scene

    def patched_setup_scene(self):
        original_setup_scene(self)

        from isaacsim.core.utils.stage import add_reference_to_stage, update_stage
        from pxr import Gf, UsdGeom, UsdPhysics

        clean_root_paths = []
        support_paths = []
        for env_path in self.scene.env_prim_paths:
            support_path = f"{env_path}/Scene/{root_name}_support"
            support = UsdGeom.Cube.Define(self.scene.stage, support_path)
            support.CreateSizeAttr(1.0)
            support_top_z = float(towel_translation[2]) + float(DEFAULT_SUPPORT_TOP_Z_OFFSET)
            support_center_z = support_top_z - float(DEFAULT_SUPPORT_SIZE[2]) * 0.5
            support_xform = UsdGeom.Xformable(support.GetPrim())
            support_xform.AddTranslateOp().Set(Gf.Vec3d(towel_translation[0], towel_translation[1], support_center_z))
            support_xform.AddScaleOp().Set(
                Gf.Vec3d(
                    float(DEFAULT_SUPPORT_SIZE[0]) * 0.5,
                    float(DEFAULT_SUPPORT_SIZE[1]) * 0.5,
                    float(DEFAULT_SUPPORT_SIZE[2]) * 0.5,
                )
            )
            UsdPhysics.CollisionAPI.Apply(support.GetPrim())
            support_paths.append(support_path)

            root_path = f"{env_path}/Scene/{root_name}"
            prim = add_reference_to_stage(str(clean_usd_path), root_path)
            UsdGeom.Xformable(prim).AddTranslateOp().Set(Gf.Vec3d(*towel_translation))
            clean_root_paths.append(root_path)
        self._tera_clean_towel_root_paths = clean_root_paths
        self._tera_clean_towel_support_paths = support_paths
        update_stage()

    fc.FoldClothBiArmEnv._setup_scene = patched_setup_scene
    setattr(fc.FoldClothBiArmEnv, patch_key, True)


def disable_env_cameras(env_cfg: Any) -> None:
    for name in ("left_wrist", "right_wrist", "top"):
        if hasattr(env_cfg.scene, name):
            setattr(env_cfg.scene, name, None)
            env_cfg.scene.__dict__.pop(name, None)
        if hasattr(env_cfg, "state_space") and isinstance(env_cfg.state_space, dict):
            env_cfg.state_space.pop(name, None)
        if hasattr(env_cfg, "observation_space") and isinstance(env_cfg.observation_space, dict):
            env_cfg.observation_space.pop(name, None)
    if hasattr(env_cfg, "cameras"):
        env_cfg.cameras = []
    if hasattr(env_cfg, "rerender_on_reset"):
        env_cfg.rerender_on_reset = False
    if hasattr(env_cfg, "wait_for_textures"):
        env_cfg.wait_for_textures = False


def make_env_cfg(args: Any, task: str, num_envs: int) -> Any:
    import leisaac.tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg(task, device=args.device, num_envs=num_envs)
    disable_env_cameras(env_cfg)
    if hasattr(env_cfg, "never_time_out"):
        env_cfg.never_time_out = True
    if hasattr(env_cfg, "manual_terminate"):
        env_cfg.manual_terminate = False
    if hasattr(env_cfg, "auto_terminate"):
        env_cfg.auto_terminate = False
    if hasattr(env_cfg, "return_success_status"):
        env_cfg.return_success_status = False
    if hasattr(env_cfg, "events"):
        env_cfg.events = None
    if hasattr(env_cfg, "recorders"):
        env_cfg.recorders = None
    if hasattr(env_cfg, "seed"):
        env_cfg.seed = 1234
    return env_cfg


@dataclass
class CleanTowelHandle:
    root_path: str
    mesh_path: str
    particle_system_path: str
    physics_scene_path: str
    cloth: Any
    particle_system: Any

    def points_tensor(self):
        return self.cloth._cloth_prim_view.get_world_positions()

    def points_numpy(self) -> np.ndarray:
        return points_to_numpy(self.points_tensor())

    def set_points(self, points: Any, device: str) -> None:
        import torch

        if hasattr(points, "detach") and callable(points.detach):
            tensor = points.to(device=device, dtype=torch.float32)
        else:
            tensor = torch.tensor(points, dtype=torch.float32, device=device)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 3 or tensor.shape[-1] != 3:
            raise RuntimeError(f"Expected set_points tensor with shape (1, N, 3), got {tuple(tensor.shape)}.")
        self.cloth._cloth_prim_view.set_world_positions(tensor.contiguous())


@dataclass
class StandaloneSo101CleanTowelScene:
    sim: Any
    left_arm: Any
    right_arm: Any
    clean_towel: CleanTowelHandle
    left_prim_path: str
    right_prim_path: str
    ground_prim_path: str
    light_prim_path: str
    device: str

    @property
    def action_shape(self) -> list[int]:
        return [1, 12]

    @property
    def action_dim(self) -> int:
        return 12

    def step_zero(self, steps: int = 1) -> None:
        for _ in range(int(steps)):
            self.left_arm.set_joint_position_target(self.left_arm.data.default_joint_pos.clone())
            self.right_arm.set_joint_position_target(self.right_arm.data.default_joint_pos.clone())
            self.left_arm.write_data_to_sim()
            self.right_arm.write_data_to_sim()
            self.sim.step(render=False)
            dt = self.sim.get_physics_dt()
            self.left_arm.update(dt)
            self.right_arm.update(dt)


def _physics_scene_path(base: Any) -> str:
    if hasattr(base, "sim") and hasattr(base.sim, "cfg") and hasattr(base.sim.cfg, "physics_prim_path"):
        return str(base.sim.cfg.physics_prim_path)
    return "/physicsScene"


def initialize_clean_towel_handle(base: Any, root_path: str) -> CleanTowelHandle:
    from isaacsim.core.prims import SingleClothPrim, SingleParticleSystem
    from isaacsim.core.simulation_manager import SimulationManager
    from isaacsim.core.utils.prims import is_prim_path_valid

    mesh_path = f"{root_path}/{CLEAN_TOWEL_MESH_SUFFIX}"
    particle_system_path = f"{root_path}/{CLEAN_TOWEL_PARTICLE_SYSTEM_SUFFIX}"
    if not is_prim_path_valid(root_path):
        raise RuntimeError(f"Clean towel root prim does not exist: {root_path}")
    if not is_prim_path_valid(mesh_path):
        raise RuntimeError(f"Clean towel mesh prim does not exist: {mesh_path}")
    if not is_prim_path_valid(particle_system_path):
        raise RuntimeError(f"Clean towel particle system prim does not exist: {particle_system_path}")

    physics_scene_path = _physics_scene_path(base)
    particle_system = SingleParticleSystem(
        prim_path=particle_system_path,
        simulation_owner=physics_scene_path,
    )
    particle_system.set_simulation_owner(physics_scene_path)
    cloth = SingleClothPrim(prim_path=mesh_path, particle_system=particle_system)
    cloth._cloth_prim_view.initialize(SimulationManager.get_physics_sim_view())
    base.sim.forward()
    return CleanTowelHandle(
        root_path=root_path,
        mesh_path=mesh_path,
        particle_system_path=particle_system_path,
        physics_scene_path=physics_scene_path,
        cloth=cloth,
        particle_system=particle_system,
    )


def initialize_clean_towel_handle_for_sim(sim: Any, root_path: str) -> CleanTowelHandle:
    from isaacsim.core.prims import SingleClothPrim, SingleParticleSystem
    from isaacsim.core.simulation_manager import SimulationManager
    from isaacsim.core.utils.prims import is_prim_path_valid

    mesh_path = f"{root_path}/{CLEAN_TOWEL_MESH_SUFFIX}"
    particle_system_path = f"{root_path}/{CLEAN_TOWEL_PARTICLE_SYSTEM_SUFFIX}"
    if not is_prim_path_valid(root_path):
        raise RuntimeError(f"Clean towel root prim does not exist: {root_path}")
    if not is_prim_path_valid(mesh_path):
        raise RuntimeError(f"Clean towel mesh prim does not exist: {mesh_path}")
    if not is_prim_path_valid(particle_system_path):
        raise RuntimeError(f"Clean towel particle system prim does not exist: {particle_system_path}")

    physics_scene_path = str(sim.cfg.physics_prim_path)
    particle_system = SingleParticleSystem(
        prim_path=particle_system_path,
        simulation_owner=physics_scene_path,
    )
    particle_system.set_simulation_owner(physics_scene_path)
    cloth = SingleClothPrim(prim_path=mesh_path, particle_system=particle_system)
    cloth._cloth_prim_view.initialize(SimulationManager.get_physics_sim_view())
    sim.forward()
    return CleanTowelHandle(
        root_path=root_path,
        mesh_path=mesh_path,
        particle_system_path=particle_system_path,
        physics_scene_path=physics_scene_path,
        cloth=cloth,
        particle_system=particle_system,
    )


def create_standalone_so101_clean_towel_scene(
    args: Any,
    clean_usd_path: Path = CLEAN_TOWEL_USD_PATH,
    towel_translation: tuple[float, float, float] = (0.0, 0.0, 0.02),
    pre_reset_spawn=None,
    left_base: tuple[float, float, float] | None = None,
    right_base: tuple[float, float, float] | None = None,
) -> StandaloneSo101CleanTowelScene:
    """Build the standalone SO-101 + clean-towel scene.

    ``pre_reset_spawn`` (optional, no-arg callable) is invoked AFTER the arms and
    towel are added but BEFORE the single ``sim.reset()``. Use it to spawn extra
    bodies (e.g. a contact proxy) so every rigid body is present when the GPU
    physics views are first initialized. Adding a rigid body AFTER reset and then
    re-resetting crashes the GPU particle+articulation solve on Isaac Sim 4.5;
    this hook preserves the proven single-reset build order (cf. V11 build_scene).
    """
    if not clean_usd_path.exists():
        raise FileNotFoundError(f"Missing clean towel USD: {clean_usd_path}")

    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation
    from isaaclab.sim import SimulationCfg, SimulationContext
    from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
    from isaacsim.core.utils.stage import add_reference_to_stage, create_new_stage, get_current_stage, update_stage
    from leisaac.assets.robots.lerobot import SO101_FOLLOWER_CFG
    from pxr import Gf, UsdGeom

    create_new_stage()
    stage = get_current_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.Xform.Define(stage, "/World/Scene")

    sim = SimulationContext(SimulationCfg(device=args.device, dt=1.0 / 60.0, render_interval=1))
    spawn_ground_plane(prim_path="/World/Ground", cfg=GroundPlaneCfg())
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    light_cfg.func("/World/Light", light_cfg)

    left_cfg = SO101_FOLLOWER_CFG.replace(prim_path="/World/Left_Robot")
    right_cfg = SO101_FOLLOWER_CFG.replace(prim_path="/World/Right_Robot")
    # Optional base repositioning (V13): the default ±0.75 m bases cannot reach a
    # centered stable towel; a V13 caller moves one base near a towel edge. Default
    # None keeps the proven V12 layout unchanged.
    left_cfg.init_state.pos = tuple(left_base) if left_base is not None else (-0.75, -0.55, 0.10)
    right_cfg.init_state.pos = tuple(right_base) if right_base is not None else (0.75, -0.55, 0.10)
    left_arm = Articulation(left_cfg)
    right_arm = Articulation(right_cfg)

    clean_root_path = f"/World/Scene/{CLEAN_TOWEL_ROOT_NAME}"
    reference_prim = add_reference_to_stage(str(clean_usd_path), clean_root_path)
    UsdGeom.Xformable(reference_prim).AddTranslateOp().Set(Gf.Vec3d(*towel_translation))
    # Spawn extra bodies (e.g. contact proxy) BEFORE the single reset (see docstring).
    if pre_reset_spawn is not None:
        pre_reset_spawn()
    update_stage()

    sim.reset()
    left_arm.update(sim.get_physics_dt())
    right_arm.update(sim.get_physics_dt())
    clean_towel = initialize_clean_towel_handle_for_sim(sim, clean_root_path)
    return StandaloneSo101CleanTowelScene(
        sim=sim,
        left_arm=left_arm,
        right_arm=right_arm,
        clean_towel=clean_towel,
        left_prim_path="/World/Left_Robot",
        right_prim_path="/World/Right_Robot",
        ground_prim_path="/World/Ground",
        light_prim_path="/World/Light",
        device=str(args.device),
    )


def move_original_garment_out_of_workspace(base: Any) -> bool:
    """Move the original FoldCloth garment particles away from the clean towel workspace."""
    try:
        original_cloth = getattr(base.scene, "particle_objects", {}).get("cloths")
        if original_cloth is None:
            return False
        positions = original_cloth.point_positions.clone()
        positions[..., 0] += 8.0
        positions[..., 2] -= 8.0
        for index, cloth_object in enumerate(original_cloth.cloth_objects):
            cloth_object._cloth_prim_view.set_world_positions(positions[index : index + 1].contiguous())
        base.sim.forward()
        return True
    except Exception:
        return False


def create_so101_clean_towel_env(
    args: Any,
    task: str = TASK_ID,
    num_envs: int = 1,
    clean_usd_path: Path = CLEAN_TOWEL_USD_PATH,
    towel_translation: tuple[float, float, float] = DEFAULT_TOWEL_TRANSLATION,
):
    if num_envs != 1:
        raise RuntimeError("This reversible smoke path currently supports exactly one environment.")
    if not clean_usd_path.exists():
        raise FileNotFoundError(f"Missing clean towel USD: {clean_usd_path}")

    import gymnasium as gym

    _install_fold_env_runtime_patches()
    _install_clean_towel_scene_patch(clean_usd_path, towel_translation)
    env_cfg = make_env_cfg(args, task=task, num_envs=num_envs)
    env = gym.make(task, cfg=env_cfg)
    base = env.unwrapped
    base.initialize()
    env.reset()
    original_garment_moved = move_original_garment_out_of_workspace(base)
    clean_root_paths = list(getattr(base, "_tera_clean_towel_root_paths", []))
    if not clean_root_paths:
        clean_root_paths = [f"{base.scene.env_prim_paths[0]}/Scene/{CLEAN_TOWEL_ROOT_NAME}"]
    clean_towel = initialize_clean_towel_handle(base, clean_root_paths[0])
    base._tera_original_garment_moved_out_of_workspace = original_garment_moved
    return env, base, clean_towel, env_cfg


def action_shape(env: Any, base: Any) -> list[int]:
    shape = getattr(getattr(env, "action_space", None), "shape", None)
    if shape:
        return [int(x) for x in shape]
    cfg_space = getattr(base.cfg, "action_space", None)
    if isinstance(cfg_space, int):
        return [int(cfg_space)]
    return []


def action_dim(env: Any, base: Any) -> int:
    shape = action_shape(env, base)
    if shape:
        total = 1
        for dim in shape:
            total *= int(dim)
        return int(total)
    return 0


def standalone_robot_scene_info(scene: StandaloneSo101CleanTowelScene) -> dict[str, Any]:
    from isaacsim.core.utils.prims import is_prim_path_valid

    left_joint_names = list(getattr(scene.left_arm.data, "joint_names", []) or [])
    right_joint_names = list(getattr(scene.right_arm.data, "joint_names", []) or [])
    return {
        "left_arm_exists": bool(is_prim_path_valid(scene.left_prim_path)),
        "right_arm_exists": bool(is_prim_path_valid(scene.right_prim_path)),
        "left_robot_prim_paths": [scene.left_prim_path] if is_prim_path_valid(scene.left_prim_path) else [],
        "right_robot_prim_paths": [scene.right_prim_path] if is_prim_path_valid(scene.right_prim_path) else [],
        "left_joint_names": left_joint_names,
        "right_joint_names": right_joint_names,
        "left_num_joints": int(len(left_joint_names)),
        "right_num_joints": int(len(right_joint_names)),
        "action_shape": scene.action_shape,
        "action_dim": int(scene.action_dim),
        "action_shape_is_12": bool(scene.action_dim == 12),
        "left_right_so101_control_exists": bool(len(left_joint_names) == 6 and len(right_joint_names) == 6),
        "scene_construction": "standalone_isaaclab_simulation_context",
    }


def robot_scene_info(env: Any, base: Any) -> dict[str, Any]:
    from isaaclab.sim import find_matching_prim_paths

    left_arm = base.scene["left_arm"] if "left_arm" in base.scene.articulations else None
    right_arm = base.scene["right_arm"] if "right_arm" in base.scene.articulations else None
    left_joint_names = list(getattr(getattr(left_arm, "data", None), "joint_names", []) or [])
    right_joint_names = list(getattr(getattr(right_arm, "data", None), "joint_names", []) or [])
    shape = action_shape(env, base)
    dim = action_dim(env, base)
    left_prims = []
    right_prims = []
    for env_path in base.scene.env_prim_paths:
        left_prims.extend(find_matching_prim_paths(f"{env_path}/Left_Robot"))
        right_prims.extend(find_matching_prim_paths(f"{env_path}/Right_Robot"))
    return {
        "left_arm_exists": left_arm is not None and bool(left_prims),
        "right_arm_exists": right_arm is not None and bool(right_prims),
        "left_robot_prim_paths": left_prims,
        "right_robot_prim_paths": right_prims,
        "left_joint_names": left_joint_names,
        "right_joint_names": right_joint_names,
        "left_num_joints": int(len(left_joint_names)),
        "right_num_joints": int(len(right_joint_names)),
        "action_shape": shape,
        "action_dim": int(dim),
        "action_shape_is_12": bool(dim == 12),
        "left_right_so101_control_exists": bool(
            left_arm is not None and right_arm is not None and len(left_joint_names) == 6 and len(right_joint_names) == 6
        ),
    }


def status_payload(
    smoke_summary: dict[str, Any] | None,
    assisted_summary: dict[str, Any] | None,
    commands_run: list[str],
) -> dict[str, Any]:
    smoke_ok = bool(smoke_summary and smoke_summary.get("status") == "SO101_CLEAN_TOWEL_SCENE_V1_TEST_OK")
    assisted_ok = bool(
        assisted_summary and assisted_summary.get("status") == "SO101_CLEAN_TOWEL_ASSISTED_FOLD_V1_OK"
    )
    scene_ready = bool(smoke_ok and assisted_ok)
    what_worked = []
    what_failed = []
    if smoke_ok:
        what_worked.append("SO-101 left/right arms, 12-D action control, and clean towel v2 particles loaded together.")
    else:
        what_failed.append("SO-101 + clean towel scene smoke test did not pass or was not found.")
    if assisted_ok:
        what_worked.append("Assisted particle-space fold reduced clean towel width below the requested threshold.")
    else:
        what_failed.append("Assisted fold did not pass or was not run.")

    return {
        "status": "SO101_CLEAN_TOWEL_V1_READY" if scene_ready else "SO101_CLEAN_TOWEL_V1_NOT_READY",
        "ready_for_v4_training": scene_ready,
        "what_worked": what_worked,
        "what_failed": what_failed,
        "paths": {
            "clean_towel_usd_v2": str(CLEAN_TOWEL_USD_PATH),
            "smoke_script": "/workspace/leisaac/scripts/tera/test_so101_clean_towel_scene_v1.py",
            "assisted_fold_script": "/workspace/leisaac/scripts/tera/run_so101_clean_towel_assisted_fold_v1.py",
            "helper_script": "/workspace/leisaac/scripts/tera/so101_clean_towel_scene_utils_v1.py",
            "output_dir": str(OUT_DIR),
            "smoke_summary": str(OUT_DIR / "test_so101_clean_towel_scene_v1_summary.json"),
            "assisted_summary": str(OUT_DIR / "assisted_fold_summary.json"),
            "status_json": str(OUT_DIR / "STATUS.json"),
        },
        "commands_run": commands_run,
        "smoke_summary_status": None if smoke_summary is None else smoke_summary.get("status"),
        "assisted_summary_status": None if assisted_summary is None else assisted_summary.get("status"),
        "scene_ready_basis": {
            "smoke_passed": smoke_ok,
            "assisted_fold_passed": assisted_ok,
            "assisted_note": "The fold primitive directly assists the real clean towel particle state inside the SO-101 scene; it is not a learned policy.",
        },
    }


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
