"""Isolated per-container Modal runner for V14 SO-101 EDGE-CAPTURE fold
(Isaac Sim 4.5.0 + IsaacLab v2.0.2).

Builds on the proven V12/V13-isolated environment (RL import stubs + h5py + one
SimulationApp per container + single-reset build order via pre_reset_spawn). V14
turns the V13 arm-contact MECHANISM into a real fold by changing the ROBOT-SIDE
contact geometry + motion (wider bar/disk patch, real press, outside-hook,
gripper pinch) — all still driven by physics-resolved articulation with a
link-attached patch (NOT a free proxy / teleport / particle write).

Each geometry config is its OWN container (patch geometry + friction + cloth are
build-time, so a geometry sweep = parallel containers, matching the isolation
model). State is shared only via the terafold-artifacts volume.

Stages:
  edge_capture(config)  -> search_so101_edge_capture_v14 (press+drag + outside-hook)
  outside_hook(config)  -> test_so101_outside_hook_drag_v14 (focused fine sweep)
  pinch(config)         -> test_so101_pinch_lift_drag_v14 (two-patch gripper pinch)
  make_soft_towel(...)  -> create_clean_rect_towel_usd_v2 (soft variant, to volume)
  demo_collection(...)  -> run_so101_arm_fold_demo_v14 (real joint-action demos)
  summarize()           -> summarize_so101_fold_readiness_v14 (CPU)

Final artifact: /artifacts/contact_v14/contact_v14_fold_summary.json
"""

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import modal

app = modal.App("terafold-modal-isaac45-so101-v14")
volume = modal.Volume.from_name("terafold-artifacts")

ISAAC_PYTHON = "/isaac-sim/python.sh"
ISAACLAB_BRANCH = "v2.0.2"
ISAACLAB_ROOT = "/opt/IsaacLab"
LEISAAC_ASSETS_ROOT = "/opt/leisaac_assets"
LEISAAC_SRC = "/opt/leisaac/source/leisaac"
SO101_USD_URL = "https://github.com/LightwheelAI/leisaac/releases/download/v0.1.0/so101_follower.usd"

TOWEL_DIR = Path("/workspace/leisaac/tera_checkpoints/clean_towel_v2")
V13_DIR = Path("/workspace/leisaac/tera_checkpoints/clean_towel_v13")
V14_DIR = Path("/workspace/leisaac/tera_checkpoints/clean_towel_v14")

ISO_DIR = Path("/artifacts/contact_v14")
V12_ASSETS = Path("/artifacts/contact_v12_isolated/assets")     # proven towel USD
V13_ARTIFACTS = Path("/artifacts/contact_v13")                   # reach solution
SOFT_TOWEL_VOL = ISO_DIR / "assets" / "soft_rect_towel_v14.usda"

_EXCLUDE_EXTS = {"isaaclab_rl", "isaaclab_mimic"}
KIT_USER_CFGS = [
    "/isaac-sim/kit/data/Kit/Isaac-Sim/4.5/user.config.json",
    "/root/.local/share/ov/data/Kit/Isaac-Sim/4.5/user.config.json",
]
KIT_LOG_DIR = Path("/isaac-sim/kit/logs/Kit/Isaac-Sim/4.5")

SEARCH_SCRIPT = "scripts/tera/search_so101_edge_capture_v14.py"
HOOK_SCRIPT = "scripts/tera/test_so101_outside_hook_drag_v14.py"
PINCH_SCRIPT = "scripts/tera/test_so101_pinch_lift_drag_v14.py"
DEMO_SCRIPT = "scripts/tera/run_so101_arm_fold_demo_v14.py"
EVAL_SCRIPT = "scripts/tera/eval_so101_robot_contact_policy_v2.py"
SUMMARY_SCRIPT = "scripts/tera/summarize_so101_fold_readiness_v14.py"
CREATE_TOWEL = "scripts/tera/create_clean_rect_towel_usd_v2.py"
CLEAN_TOWEL_USD = TOWEL_DIR / "clean_rect_towel_usd_v2.usda"

# geometry configs for the edge-capture search (each its own container).
# Diagnostic (control run): the jaw's LOCAL Z axis is ~world-vertical
# ([0.065, 0.192, 0.979]), so a cylinder with axis=Z presents a flat HORIZONTAL
# disk facing down — the wide flat contact that can span the whole edge (incl.
# corners at y=+/-0.19) and buckle-fold the right band. Big disks are prioritized;
# a horizontal capsule (axis Y/X lie ~in-plane) is kept as a bar alternative.
EDGE_CONFIGS = {
    "sphere_r05_ctrl": ["--patch_shape", "sphere", "--patch_radius", "0.05"],
    # Attempt-1 lesson: disks (r<=0.14) keep a CLEAN baseline and do NOT deflect the
    # arm (jaw stays ~z=0.05); they only exploded on aggressive/fast press-drag. So
    # here disks use a LARGER contact_offset (softer contact) + the search's gentler,
    # slower press-drag. r<=0.14 keeps touch reliable and avoids arm deflection.
    "disk_z_r10": ["--patch_shape", "cylinder", "--patch_radius", "0.10", "--patch_length", "0.03", "--patch_axis", "Z", "--patch_contact_offset", "0.03"],
    "disk_z_r12": ["--patch_shape", "cylinder", "--patch_radius", "0.12", "--patch_length", "0.03", "--patch_axis", "Z", "--patch_contact_offset", "0.035"],
    "disk_z_r14": ["--patch_shape", "cylinder", "--patch_radius", "0.14", "--patch_length", "0.03", "--patch_axis", "Z", "--patch_contact_offset", "0.04"],
    "sphere_r10": ["--patch_shape", "sphere", "--patch_radius", "0.10", "--patch_contact_offset", "0.03"],
    # kept for reference / soft-cloth diagnostics (clean-baseline bar orientation)
    "capsule_y_l24": ["--patch_shape", "capsule", "--patch_radius", "0.03", "--patch_length", "0.24", "--patch_axis", "Y"],
}

image = (
    modal.Image.from_registry("nvcr.io/nvidia/isaac-sim:4.5.0", add_python="3.10")
    .dockerfile_commands("ENTRYPOINT []")
    .apt_install("git", "tar", "curl")
    .run_commands(f"{ISAAC_PYTHON} -m pip install --no-cache-dir flatdict toml")
    .run_commands(
        f"git clone --depth 1 --branch {ISAACLAB_BRANCH} "
        f"https://github.com/isaac-sim/IsaacLab.git {ISAACLAB_ROOT}"
    )
    .run_commands(
        "git clone --depth 1 https://github.com/LightwheelAI/leisaac.git /opt/leisaac",
        f"mkdir -p {LEISAAC_ASSETS_ROOT}/robots",
        f"curl -fL -o {LEISAAC_ASSETS_ROOT}/robots/so101_follower.usd {SO101_USD_URL}",
    )
    .run_commands(f"{ISAAC_PYTHON} -m pip install --no-cache-dir 'gym==0.26.2' h5py")
    .add_local_file("/tmp/terafold_modal_src.tar.gz", "/opt/terafold_modal_src.tar.gz", copy=True)
    .env(
        {
            "ACCEPT_EULA": "Y",
            "PRIVACY_CONSENT": "Y",
            "OMNI_KIT_ACCEPT_EULA": "YES",
            "NVIDIA_VISIBLE_DEVICES": "all",
            "NVIDIA_DRIVER_CAPABILITIES": "all",
            "LEISAAC_ASSETS_ROOT": LEISAAC_ASSETS_ROOT,
        }
    )
)


def _extract_status(stdout: str):
    found = None
    for m in re.finditer(r"\{", stdout or ""):
        frag = stdout[m.start():]
        try:
            obj, _ = json.JSONDecoder().raw_decode(frag)
            if isinstance(obj, dict) and "status" in obj:
                found = obj
        except Exception:
            continue
    return found


def _unpack_source():
    work = Path("/tmp/work")
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    unpack = subprocess.run(
        ["tar", "-xzf", "/opt/terafold_modal_src.tar.gz", "-C", str(work)],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300,
    )
    if unpack.returncode != 0:
        raise RuntimeError(f"source unpack failed: {unpack.stderr[-2000:]}")
    return work


def _isaac_env(work: Path):
    source_root = Path(ISAACLAB_ROOT) / "source"
    source_paths = (
        [str(p) for p in source_root.iterdir() if p.is_dir() and p.name not in _EXCLUDE_EXTS]
        if source_root.exists() else []
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = ":".join(source_paths + [LEISAAC_SRC, str(work), env.get("PYTHONPATH", "")])
    env["LEISAAC_ASSETS_ROOT"] = LEISAAC_ASSETS_ROOT
    return env


def _persist_full_logs(log_tag: str, proc):
    logs = ISO_DIR / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / f"{log_tag}.stdout.txt").write_text(proc.stdout or "")
    (logs / f"{log_tag}.stderr.txt").write_text(proc.stderr or "")
    try:
        kit_logs = sorted(KIT_LOG_DIR.glob("kit_*.log"), key=lambda p: p.stat().st_mtime)
        if kit_logs:
            (logs / f"{log_tag}.kitlog.txt").write_text(kit_logs[-1].read_text(errors="replace")[-200000:])
    except Exception as exc:  # noqa: BLE001
        (logs / f"{log_tag}.kitlog_error.txt").write_text(repr(exc))


def _run_isaac_script(work: Path, env: dict, rel: str, log_tag: str, extra_args=None, timeout: int = 3300):
    for cfg in KIT_USER_CFGS:
        try:
            Path(cfg).unlink()
        except Exception:
            pass
    script_env = dict(env)
    home = "/tmp/kithome"
    Path(home).mkdir(parents=True, exist_ok=True)
    script_env["HOME"] = home
    cmd = [ISAAC_PYTHON, str(work / rel)] + list(extra_args or [])
    proc = subprocess.run(
        cmd, cwd=str(work), text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, env=script_env,
    )
    st = _extract_status(proc.stdout)
    _persist_full_logs(log_tag, proc)
    record = {
        "script": rel, "cmd_extra": list(extra_args or []),
        "returncode": proc.returncode, "internal_status": (st or {}).get("status"),
        "stdout_tail": (proc.stdout or "")[-12000:], "stderr_tail": (proc.stderr or "")[-8000:],
    }
    return record, st


def _stage_towel_from_volume(soft: bool = False):
    """Stage the towel USD into the container. Normal = proven V12 towel; soft =
    the V14 soft variant from the volume (overwrites CLEAN_TOWEL_USD locally)."""
    TOWEL_DIR.mkdir(parents=True, exist_ok=True)
    staged = []
    if V12_ASSETS.exists():
        for f in V12_ASSETS.iterdir():
            if f.is_file():
                (TOWEL_DIR / f.name).write_bytes(f.read_bytes())
                staged.append(f.name)
    if soft and SOFT_TOWEL_VOL.exists():
        CLEAN_TOWEL_USD.write_text(SOFT_TOWEL_VOL.read_text())
        staged.append(f"SOFT<-{SOFT_TOWEL_VOL.name}")
    return staged


def _stage_reach_json_in():
    V13_DIR.mkdir(parents=True, exist_ok=True)
    staged = []
    src = V13_ARTIFACTS / "so101_articulation_reach_v13.json"
    if src.exists():
        (V13_DIR / src.name).write_text(src.read_text())
        staged.append(src.name)
    return staged


def _persist_v14():
    ISO_DIR.mkdir(parents=True, exist_ok=True)
    persisted = []
    if V14_DIR.exists():
        for f in list(V14_DIR.glob("*.json")) + list(V14_DIR.glob("*.npz")):
            (ISO_DIR / f.name).write_bytes(f.read_bytes())
            persisted.append(f.name)
        demo_src = V14_DIR / "demos"
        if demo_src.exists():
            (ISO_DIR / "demos").mkdir(parents=True, exist_ok=True)
            for f in demo_src.iterdir():
                if f.is_file():
                    (ISO_DIR / "demos" / f.name).write_bytes(f.read_bytes())
                    persisted.append(f"demos/{f.name}")
    return persisted


def _stage_v14_in():
    """Copy prior V14 JSONs/npz (and the demos/ subdir) from the volume back into
    the container V14 dir so summarize / demo stages find them."""
    V14_DIR.mkdir(parents=True, exist_ok=True)
    staged = []
    if ISO_DIR.exists():
        for f in list(ISO_DIR.glob("*.json")) + list(ISO_DIR.glob("*.npz")):
            (V14_DIR / f.name).write_bytes(f.read_bytes())
            staged.append(f.name)
        demo_src = ISO_DIR / "demos"
        if demo_src.exists():
            (V14_DIR / "demos").mkdir(parents=True, exist_ok=True)
            for f in demo_src.iterdir():
                if f.is_file():
                    (V14_DIR / "demos" / f.name).write_bytes(f.read_bytes())
                    staged.append(f"demos/{f.name}")
    return staged


def _write_stage_record(name: str, record: dict):
    ISO_DIR.mkdir(parents=True, exist_ok=True)
    (ISO_DIR / f"stage_{name}.json").write_text(json.dumps(record, indent=2))
    volume.commit()
    print(json.dumps({k: v for k, v in record.items() if k not in ("stdout_tail", "stderr_tail")}, indent=2))


# --------------------------------------------------------------------------- #
@app.function(gpu="L40S", image=image, volumes={"/artifacts": volume}, timeout=3600)
def edge_capture(config_name: str, geom_args: list, soft: bool = False, right_base=None):
    work = _unpack_source()
    env = _isaac_env(work)
    staged = _stage_towel_from_volume(soft=soft)
    _stage_reach_json_in()
    extra = ["--config_name", config_name] + list(geom_args)
    if soft:
        extra += ["--soft_cloth"]
    if right_base:
        extra += ["--right_base", *[str(x) for x in right_base]]
    run, st = _run_isaac_script(work, env, SEARCH_SCRIPT, log_tag=f"edge_{config_name}", extra_args=extra)
    persisted = _persist_v14()
    bt = (st or {}).get("best_trial", {}) if isinstance(st, dict) else {}
    record = {
        "stage": f"edge_capture_{config_name}", "soft": soft, "staged_assets": staged, "run": run,
        "persisted": persisted, "internal_status": run["internal_status"],
        "arm_driven_fold_solved": bool((st or {}).get("arm_driven_fold_solved")),
        "best_trial": {k: bt.get(k) for k in ("name", "edge_displacement_m", "width_reduction_m",
                                              "actual_touch_distance_m", "particle_velocity_peak", "valid_controlled_contact")},
    }
    _write_stage_record(f"edge_{config_name}", record)
    return {"config": config_name, "returncode": run["returncode"], "internal_status": run["internal_status"],
            "arm_driven_fold_solved": record["arm_driven_fold_solved"], "best_trial": record["best_trial"],
            "stderr_tail": run["stderr_tail"][-2500:]}


@app.function(gpu="L40S", image=image, volumes={"/artifacts": volume}, timeout=3600)
def outside_hook(config_name: str, geom_args: list, soft: bool = False, right_base=None):
    work = _unpack_source()
    env = _isaac_env(work)
    _stage_towel_from_volume(soft=soft)
    _stage_reach_json_in()
    extra = ["--config_name", config_name] + list(geom_args)
    if soft:
        extra += ["--soft_cloth"]
    if right_base:
        extra += ["--right_base", *[str(x) for x in right_base]]
    run, st = _run_isaac_script(work, env, HOOK_SCRIPT, log_tag=f"hook_{config_name}", extra_args=extra)
    persisted = _persist_v14()
    bt = (st or {}).get("best_trial", {}) if isinstance(st, dict) else {}
    record = {"stage": f"outside_hook_{config_name}", "run": run, "persisted": persisted,
              "internal_status": run["internal_status"],
              "arm_driven_fold_solved": bool((st or {}).get("arm_driven_fold_solved")),
              "best_trial": {k: bt.get(k) for k in ("name", "edge_displacement_m", "width_reduction_m", "valid_controlled_contact")}}
    _write_stage_record(f"hook_{config_name}", record)
    return {"config": config_name, "returncode": run["returncode"], "internal_status": run["internal_status"],
            "arm_driven_fold_solved": record["arm_driven_fold_solved"], "best_trial": record["best_trial"],
            "stderr_tail": run["stderr_tail"][-2500:]}


@app.function(gpu="L40S", image=image, volumes={"/artifacts": volume}, timeout=3600)
def pinch(config_name: str, geom_args: list, soft: bool = False, right_base=None):
    work = _unpack_source()
    env = _isaac_env(work)
    _stage_towel_from_volume(soft=soft)
    _stage_reach_json_in()
    extra = ["--config_name", config_name] + list(geom_args)
    if soft:
        extra += ["--soft_cloth"]
    if right_base:
        extra += ["--right_base", *[str(x) for x in right_base]]
    run, st = _run_isaac_script(work, env, PINCH_SCRIPT, log_tag=f"pinch_{config_name}", extra_args=extra)
    persisted = _persist_v14()
    bt = (st or {}).get("best_trial", {}) if isinstance(st, dict) else {}
    record = {"stage": f"pinch_{config_name}", "run": run, "persisted": persisted,
              "internal_status": run["internal_status"],
              "arm_driven_fold_solved": bool((st or {}).get("arm_driven_fold_solved")),
              "best_trial": {k: bt.get(k) for k in ("name", "edge_displacement_m", "width_reduction_m", "valid_controlled_contact")}}
    _write_stage_record(f"pinch_{config_name}", record)
    return {"config": config_name, "returncode": run["returncode"], "internal_status": run["internal_status"],
            "arm_driven_fold_solved": record["arm_driven_fold_solved"], "best_trial": record["best_trial"],
            "stderr_tail": run["stderr_tail"][-2500:]}


@app.function(gpu="L40S", image=image, volumes={"/artifacts": volume}, timeout=1800)
def make_soft_towel(stretch=300.0, bend=40.0, shear=60.0, damping=8.0):
    """Author a SOFT-stiffness towel USD (diagnostic variant) and copy to volume."""
    work = _unpack_source()
    env = _isaac_env(work)
    TOWEL_DIR.mkdir(parents=True, exist_ok=True)
    extra = [
        "--stretch_stiffness", str(stretch), "--bend_stiffness", str(bend),
        "--shear_stiffness", str(shear), "--spring_damping", str(damping),
    ]
    run, st = _run_isaac_script(work, env, CREATE_TOWEL, log_tag="make_soft_towel", extra_args=extra, timeout=1200)
    (ISO_DIR / "assets").mkdir(parents=True, exist_ok=True)
    ok = False
    if CLEAN_TOWEL_USD.exists():
        SOFT_TOWEL_VOL.write_text(CLEAN_TOWEL_USD.read_text())
        ok = True
    record = {"stage": "make_soft_towel", "run": run, "soft_towel_written": ok,
              "soft_towel_path": str(SOFT_TOWEL_VOL), "stretch": stretch, "bend": bend, "shear": shear}
    _write_stage_record("make_soft_towel", record)
    return {"soft_towel_written": ok, "returncode": run["returncode"], "internal_status": run["internal_status"]}


@app.function(gpu="L40S", image=image, volumes={"/artifacts": volume}, timeout=3600)
def demo_collection(actions_npz_name: str, geom_args: list, config_name: str, pinch_cfg: bool = False,
                    soft: bool = False, right_base=None, num_episodes: int = 12):
    work = _unpack_source()
    env = _isaac_env(work)
    _stage_towel_from_volume(soft=soft)
    _stage_v14_in()  # bring recorded actions npz into the container
    npz_path = V14_DIR / actions_npz_name
    extra = ["--actions_npz", str(npz_path), "--config_name", config_name,
             "--num_episodes", str(int(num_episodes))] + list(geom_args)
    if pinch_cfg:
        extra += ["--pinch"]
    if right_base:
        extra += ["--right_base", *[str(x) for x in right_base]]
    run, st = _run_isaac_script(work, env, DEMO_SCRIPT, log_tag="demo", extra_args=extra)
    persisted = _persist_v14()
    record = {"stage": "demo_collection", "run": run, "persisted": persisted,
              "internal_status": run["internal_status"],
              "robot_contact_data_ready": bool((st or {}).get("robot_contact_data_ready")),
              "total_samples": (st or {}).get("total_samples"),
              "num_valid_success": (st or {}).get("num_valid_success")}
    _write_stage_record("demo_collection", record)
    return {"returncode": run["returncode"], "internal_status": run["internal_status"],
            "robot_contact_data_ready": record["robot_contact_data_ready"],
            "total_samples": record["total_samples"], "stderr_tail": run["stderr_tail"][-2500:]}


@app.function(gpu="L40S", image=image, volumes={"/artifacts": volume}, timeout=3600)
def eval_policy(checkpoint_name: str, geom_args: list, config_name: str, pinch_cfg: bool = False,
                right_base=None, num_episodes: int = 6):
    """Closed-loop eval of the Training-v2 policy in a fresh Isaac scene."""
    work = _unpack_source()
    env = _isaac_env(work)
    _stage_towel_from_volume()
    ckpt = f"/artifacts/training_v2_robot_contact/{checkpoint_name}"
    extra = ["--checkpoint", ckpt, "--config_name", config_name,
             "--num_episodes", str(int(num_episodes))] + list(geom_args)
    if pinch_cfg:
        extra += ["--pinch"]
    if right_base:
        extra += ["--right_base", *[str(x) for x in right_base]]
    run, st = _run_isaac_script(work, env, EVAL_SCRIPT, log_tag="eval", extra_args=extra)
    eval_dir = Path("/workspace/leisaac/tera_checkpoints/eval_v2_robot_contact")
    out = Path("/artifacts/eval_v2_robot_contact")
    out.mkdir(parents=True, exist_ok=True)
    persisted = []
    if eval_dir.exists():
        for f in eval_dir.glob("*.json"):
            (out / f.name).write_text(f.read_text())
            persisted.append(f.name)
    volume.commit()
    record = {"stage": "eval_policy", "run": run, "persisted": persisted,
              "internal_status": run["internal_status"],
              "success_rate": (st or {}).get("success_rate"),
              "autonomous_fold_policy_validated": (st or {}).get("autonomous_fold_policy_validated"),
              "mean_width_before": (st or {}).get("mean_width_before"),
              "mean_width_after": (st or {}).get("mean_width_after")}
    _write_stage_record("eval_policy", record)
    return {"returncode": run["returncode"], "internal_status": run["internal_status"],
            "success_rate": record["success_rate"],
            "autonomous_fold_policy_validated": record["autonomous_fold_policy_validated"],
            "stderr_tail": run["stderr_tail"][-2500:]}


@app.function(image=image, volumes={"/artifacts": volume}, timeout=600)
def summarize():
    work = _unpack_source()
    _stage_v14_in()
    proc = subprocess.run(
        ["python3", str(work / SUMMARY_SCRIPT)], cwd=str(work), text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300,
    )
    summ_local = V14_DIR / "contact_v14_fold_summary.json"
    summary = None
    if summ_local.exists():
        summary = json.loads(summ_local.read_text())
        ISO_DIR.mkdir(parents=True, exist_ok=True)
        (ISO_DIR / "contact_v14_fold_summary.json").write_text(json.dumps(summary, indent=2))
    volume.commit()
    if summary:
        print(json.dumps({k: v for k, v in summary.items() if k != "all_trials"}, indent=2))
    else:
        print(json.dumps({"summarize_returncode": proc.returncode, "stderr_tail": (proc.stderr or "")[-2000:]}, indent=2))
    return {"summary": summary, "summarize_returncode": proc.returncode}


@app.local_entrypoint()
def main(stages: str = "edge_capture", configs: str = "all", soft: str = "", right_base: str = "",
         demo_config: str = "", demo_actions: str = "", demo_pinch: str = "", demo_geom: str = ""):
    rb = [float(x) for x in right_base.split(",")] if right_base else None
    want = [s.strip() for s in stages.split(",") if s.strip()]
    results = {}

    # soft towel MUST be authored before any soft edge_capture container stages it
    if "make_soft_towel" in want:
        results["make_soft_towel"] = make_soft_towel.remote()
        print(json.dumps(results["make_soft_towel"], indent=2))

    if "edge_capture" in want:
        names = list(EDGE_CONFIGS) if configs == "all" else [c.strip() for c in configs.split(",") if c.strip()]
        soft_names = [s.strip() for s in soft.split(",") if s.strip()]
        print(f">>> edge_capture normal={names} soft={soft_names}")
        handles = {}
        for n in names:
            if n in EDGE_CONFIGS:
                handles[n] = edge_capture.spawn(n, EDGE_CONFIGS[n], soft=False, right_base=rb)
        for n in soft_names:  # same geometry, on the soft towel, distinct output name
            if n in EDGE_CONFIGS:
                handles[f"{n}_soft"] = edge_capture.spawn(f"{n}_soft", EDGE_CONFIGS[n], soft=True, right_base=rb)
        results["edge_capture"] = {n: h.get() for n, h in handles.items()}
        print(json.dumps(results["edge_capture"], indent=2))

    if "outside_hook" in want:
        names = [c.strip() for c in configs.split(",") if c.strip()] or ["capsule_y_l34"]
        soft_names = set(s.strip() for s in soft.split(",") if s.strip())
        handles = {n: outside_hook.spawn(n, EDGE_CONFIGS.get(n, []), soft=(n in soft_names), right_base=rb) for n in names}
        results["outside_hook"] = {n: h.get() for n, h in handles.items()}
        print(json.dumps(results["outside_hook"], indent=2))

    if "pinch" in want:
        names = [c.strip() for c in configs.split(",") if c.strip()] or ["pinch_default"]
        handles = {n: pinch.spawn(n, [], soft=False, right_base=rb) for n in names}
        results["pinch"] = {n: h.get() for n, h in handles.items()}
        print(json.dumps(results["pinch"], indent=2))

    if "demo_collection" in want:
        geom = [g for g in demo_geom.split(" ") if g] if demo_geom else []
        results["demo_collection"] = demo_collection.remote(
            demo_actions, geom, demo_config or "best", pinch_cfg=bool(demo_pinch), right_base=rb)
        print(json.dumps(results["demo_collection"], indent=2))

    if "eval_policy" in want:
        geom = [g for g in demo_geom.split(" ") if g] if demo_geom else []
        results["eval_policy"] = eval_policy.remote(
            "policy_v2_robot_contact.pt", geom, demo_config or "best", pinch_cfg=bool(demo_pinch), right_base=rb)
        print(json.dumps(results["eval_policy"], indent=2))

    if "summarize" in want:
        results["summarize"] = summarize.remote()

    summ = (results.get("summarize") or {}).get("summary") or {}
    print(json.dumps({"V14_VERDICT": {
        "normal_cloth_success": summ.get("normal_cloth_success"),
        "soft_cloth_success": summ.get("soft_cloth_success"),
        "arm_driven_fold_solved": summ.get("arm_driven_fold_solved"),
        "best_config": summ.get("best_config"),
        "edge_displacement_m": summ.get("edge_displacement_m"),
        "width_before_m": summ.get("width_before_m"),
        "width_after_m": summ.get("width_after_m"),
        "robot_contact_data_ready": summ.get("robot_contact_data_ready"),
        "final_robot_contact_policy_training_ready": summ.get("final_robot_contact_policy_training_ready"),
        "remaining_blocker": summ.get("remaining_blocker"),
        "exact_next_step": summ.get("exact_next_step"),
    }}, indent=2))
