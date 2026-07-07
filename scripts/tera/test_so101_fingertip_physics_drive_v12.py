"""V12 SO-101 fingertip physics-drive test.

Ports the V11 fix to the SO-101 setting. In the REAL SO-101 scene (leisaac arm +
clean towel), a jaw-scale fingertip PROXY collider is swept along the towel edge
(the SO-101 jaw's intended fold path) and driven two ways:
- teleport            : USD `set_prim_translation` (the V8 SO-101-proxy method)
- kinematic_velocity  : physics `RigidPrim.set_world_poses` (the V11 fix)

This reuses the V11 drive + valid/ballistic classification (via a scene wrapper)
and V8's proxy spawn, so the ONLY thing that changes between trials is how the
SO-101 fingertip proxy is driven. It directly shows whether the V11 mechanism
transfers to the SO-101 fingertip: teleport should fail, physics should move the
cloth.

Note: the SO-101 arm is loaded and present; the fingertip proxy is the contact
element (as established in V8, because the raw jaw geometry under-contacts). The
V6 reach search output is unavailable on Modal, so the proxy follows the
geometry-derived fold path rather than a live-searched arm joint trajectory; this
is stated honestly in the JSON. No particle teleporting counts as success.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Install RL import stubs BEFORE any IsaacLab/SimulationApp import: on Isaac Sim
# 4.5.0 the isaaclab_rl extension auto-loads and imports rl_games/rsl_rl/sb3/skrl
# (not installed -> fatal). This import-only shim makes those imports succeed so
# SimulationApp starts. RL_STUBS_FOR_ISAACLAB_IMPORT_ONLY; never used for training.
import isaaclab_rl_stubs_v12  # noqa: F401,E402  (side-effect: installs meta_path stubs)

from isaaclab.app import AppLauncher

import clean_towel_contact_v11_common as c11
import test_clean_towel_primitive_contact_v8 as v8
from clean_towel_v4_common import jaw_positions_from_scene
from so101_clean_towel_scene_utils_v1 import (
    CLEAN_TOWEL_USD_PATH,
    command_run_string,
    create_standalone_so101_clean_towel_scene,
    force_headless_no_cameras,
    initialize_clean_towel_handle_for_sim,
    size_metrics,
    write_json,
)

V12_DIR = Path("/workspace/leisaac/tera_checkpoints/clean_towel_v12")
OUT_JSON = V12_DIR / "so101_fingertip_physics_drive_v12.json"
PROXY_PATH = "/World/V12ProxyFingertip"

# Stable resting position for the towel: flat and CENTERED, away from the SO-101
# arms. The V6-"recommended" arm-relative center (~(0.20, -0.55)) sits at the
# arms' own y-row (both arms are at y=-0.55) and interpenetrates their colliders
# at init, which hard-crashes the GPU particle+articulation solve (a native crash
# with no Python traceback). The V6 reach data that would justify that placement
# is not on Modal (it falls back to that harmful default). Because the physics-
# driven fingertip PROXY does the contact and derives its sweep path from the
# towel's OWN settled edge (default_path_centers(start_points)), a centered towel
# keeps the test valid while the real SO-101 arms remain present in the scene.
STABLE_TOWEL_TRANSLATION = (0.0, 0.0, 0.02)


def _mark(msg: str) -> None:
    """Flushed progress marker so a native (non-Python) crash is localizable."""
    print(f"V12_MARK: {msg}", flush=True)


@dataclass
class _ProxyScene:
    """Adapts the SO-101 scene so V11's run_drive_trial can drive the proxy."""
    sim: Any
    clean_towel: Any
    primitive_root_path: str
    device: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V12 SO-101 fingertip physics-drive test.")
    parser.add_argument("--usd_path", type=Path, default=CLEAN_TOWEL_USD_PATH)
    parser.add_argument("--out_dir", type=Path, default=V12_DIR)
    parser.add_argument("--edge", type=str, default="right")
    parser.add_argument("--edge_band", type=float, default=0.035)
    parser.add_argument("--settle_initial_steps", type=int, default=20)
    # Proxy / path geometry. Default is the V11-PROVEN contact scale (radius 0.04,
    # penetration 0.06): a minimal jaw-scale point (radius 0.02) ports the drive
    # mechanism (real particle-velocity spike vs an inert teleport) but under-drags
    # the stiff towel (edge ~2.4 mm << 20 mm), so it does not fold. The 0.04 proxy
    # represents the SO-101 gripper CONTACT PATCH (not a literal fingertip point)
    # and matches the primitive that produced a valid fold in V11.
    parser.add_argument("--primitive_radius", type=float, default=0.04)
    parser.add_argument("--primitive_height", type=float, default=0.012)
    parser.add_argument("--penetration_depth", type=float, default=0.06)
    parser.add_argument("--drag_distance", type=float, default=0.28)
    parser.add_argument("--drag_height_delta", type=float, default=0.0)
    parser.add_argument("--outside_margin", type=float, default=0.05)
    # Primitive (jaw-scale proxy) + cloth contact params consumed by
    # v8.default_primitive_params / v8.apply_contact_params. These MUST all exist
    # on `args` or default_primitive_params raises AttributeError (v8 defaults).
    parser.add_argument("--primitive_contact_offset", type=float, default=0.012)
    parser.add_argument("--primitive_rest_offset", type=float, default=0.0)
    parser.add_argument("--cloth_contact_offset", type=float, default=0.005)
    parser.add_argument("--cloth_rest_offset", type=float, default=0.003)
    parser.add_argument("--particle_radius", type=float, default=0.005)
    parser.add_argument("--static_friction", type=float, default=1.2)
    parser.add_argument("--dynamic_friction", type=float, default=1.0)
    parser.add_argument("--approach_steps", type=int, default=50)
    parser.add_argument("--hold_contact_steps", type=int, default=12)
    parser.add_argument("--drag_steps", type=int, default=80)
    parser.add_argument("--settle_after_steps", type=int, default=20)
    parser.add_argument("--settle_before_steps", type=int, default=4)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    force_headless_no_cameras(args)
    return args


def main() -> None:
    args = parse_args()
    V12_DIR.mkdir(parents=True, exist_ok=True)
    app = AppLauncher(args).app  # noqa: F841

    summary: dict[str, Any] = {
        "status": "V12_SO101_FINGERTIP_PHYSICS_DRIVE_STARTED",
        "command_run": command_run_string(),
        "diagnostic_only": True,
        "contact_element": "so101_gripper_contact_patch_proxy_v11scale_r0.04_in_real_so101_scene",
        "v6_reach_search_available_on_modal": False,
        "trials": [],
    }
    try:
        summary["towel_translation"] = list(STABLE_TOWEL_TRANSLATION)
        summary["towel_placement_note"] = (
            "centered stable rest (V6 arm-relative center omitted: not on Modal and "
            "interpenetrates the SO-101 arms -> GPU solver crash)"
        )

        # Spawn the jaw-scale fingertip PROXY BEFORE the scene's single sim.reset()
        # via the pre_reset_spawn hook. Spawning it after the reset and re-resetting
        # crashes the GPU particle+articulation solve on Isaac 4.5; the single-reset
        # order matches the proven V11 build. The proxy's rigid view is created
        # lazily inside run_drive_trial (make_rigid_view), so no second reset.
        proxy_holder: dict[str, Any] = {}

        def _spawn_proxy() -> None:
            proxy_holder["info"] = v8.spawn_kinematic_sphere(
                PROXY_PATH, radius=float(args.primitive_radius),
                contact_offset=float(args.primitive_contact_offset), rest_offset=float(args.primitive_rest_offset),
                static_friction=1.6, dynamic_friction=1.2, translation=(1.25, 0.0, 0.30),
            )

        _mark("creating SO-101 scene (real arms + centered towel + proxy pre-reset)")
        scene = create_standalone_so101_clean_towel_scene(
            args, clean_usd_path=args.usd_path, towel_translation=STABLE_TOWEL_TRANSLATION,
            pre_reset_spawn=_spawn_proxy,
        )
        _mark("scene created (single reset, proxy present); settling")
        scene.step_zero(int(args.settle_initial_steps))
        _mark("settled")

        summary["proxy_info"] = proxy_holder.get("info")
        jaws, jaw_info = jaw_positions_from_scene(scene)
        summary["so101_jaw_positions"] = [float(x) for x in jaws]
        summary["so101_jaw_info"] = jaw_info

        start_points = scene.clean_towel.points_numpy().astype(np.float32)
        summary["start_metrics"] = size_metrics(start_points)

        # apply cloth contact params (reuse V8 tuning) via the primitive path
        proxy_scene = _ProxyScene(sim=scene.sim, clean_towel=scene.clean_towel,
                                  primitive_root_path=PROXY_PATH, device=scene.device)
        params = v8.default_primitive_params(args)
        try:
            summary["contact_param_application"] = v8.apply_contact_params(proxy_scene, params)
        except Exception as exc:  # noqa: BLE001
            summary["contact_param_application_error"] = repr(exc)

        edge = str(args.edge)
        radius = float(args.primitive_radius)
        eb = float(args.edge_band)
        configs = [
            ("so101_proxy_teleport", "teleport", 140),
            ("so101_proxy_kinematic_velocity_slow", "kinematic_velocity", 140),
            ("so101_proxy_kinematic_velocity_fast", "kinematic_velocity", 40),
        ]
        for name, mode, nsteps in configs:
            _mark(f"trial start: {name} ({mode}, {nsteps} steps)")
            centers = c11.default_path_centers(start_points, args, num_steps=nsteps, edge=edge)
            trial = c11.run_drive_trial(proxy_scene, start_points, centers, edge, eb, radius, drive_mode=mode)
            trial["trial_name"] = name
            summary["trials"].append(trial)
            _mark(f"trial done: {name} valid={trial.get('valid_controlled_contact')} "
                  f"edge={trial.get('edge_displacement_m'):.4f} width={trial.get('width_before'):.3f}->{trial.get('best_width'):.3f}")

        summary["any_valid_controlled_contact"] = bool(
            any(t.get("valid_controlled_contact") for t in summary["trials"])
        )
        summary["status"] = "V12_SO101_FINGERTIP_PHYSICS_DRIVE_DONE"
        write_json(OUT_JSON, summary, print_payload=True)
    except Exception as exc:  # noqa: BLE001
        import traceback

        summary["status"] = "V12_SO101_FINGERTIP_PHYSICS_DRIVE_FAILED"
        summary["exception_type"] = exc.__class__.__name__
        summary["exception"] = repr(exc)
        summary["traceback_tail"] = traceback.format_exc()[-6000:]
        write_json(OUT_JSON, summary, print_payload=True)
    finally:
        sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == "__main__":
    main()
