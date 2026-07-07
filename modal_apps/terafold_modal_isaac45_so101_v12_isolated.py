"""Isolated per-container Modal runner for the V12 SO-101 physics-drive contact
suite (Isaac Sim 4.5.0 + IsaacLab v2.0.2).

Why this exists (vs. terafold_modal_isaac45_so101_v12.py):
  V12 proved the real SO-101 arm scene builds on Modal, but the physics-drive
  test never executed: IsaacLab's `isaaclab_rl` extension auto-loads at
  SimulationApp startup and imports rl_games/rsl_rl/sb3/skrl (not installed ->
  fatal), and running several SimulationApp sessions in one container poisoned
  the kit user.config.json for later launches.

Two fixes, applied together:
  1. RL import stubs (scripts/tera/isaaclab_rl_stubs_v12.py, imported first by the
     patched V12 scripts) satisfy the isaaclab_rl imports without installing the
     frameworks (which would pull a conflicting torch/cu13).
  2. Container isolation: EACH Isaac stage runs in its OWN Modal function ->
     its OWN container -> at most ONE SimulationApp per process. Stages share
     state ONLY through the `terafold-artifacts` Volume (towel USD in / drive
     JSON out), never through container-local /workspace.

Stages (each its own container):
  0. make_towel        -> authors clean towel USD, publishes to Volume assets
  A. build_inspect     -> inspect_so101_drive_modes_v12  (build + drive audit)
  B+C. physics_drive   -> test_so101_fingertip_physics_drive_v12 (ONE scene runs
        the teleport baseline (B) AND the physics-driven set_world_poses trials
        (C), which is the honest same-scene apples-to-apples comparison)
  D. summarize         -> summarize_so101_contact_readiness_v12 (CPU, no Isaac)

Final artifact:
  /artifacts/contact_v12_isolated/contact_v12_isolated_summary.json
"""

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import modal

app = modal.App("terafold-modal-isaac45-so101-v12-isolated")
volume = modal.Volume.from_name("terafold-artifacts")

ISAAC_PYTHON = "/isaac-sim/python.sh"
ISAACLAB_BRANCH = "v2.0.2"
ISAACLAB_ROOT = "/opt/IsaacLab"
LEISAAC_ASSETS_ROOT = "/opt/leisaac_assets"
LEISAAC_SRC = "/opt/leisaac/source/leisaac"
SO101_USD_URL = "https://github.com/LightwheelAI/leisaac/releases/download/v0.1.0/so101_follower.usd"

# Container-local paths (per the repo scripts' hardcoded constants).
TOWEL_DIR = Path("/workspace/leisaac/tera_checkpoints/clean_towel_v2")
V12_DIR = Path("/workspace/leisaac/tera_checkpoints/clean_towel_v12")

# Volume paths (the ONLY cross-container channel).
ISO_DIR = Path("/artifacts/contact_v12_isolated")
ISO_ASSETS = ISO_DIR / "assets"

# RL extensions whose import we deliberately keep off PYTHONPATH (belt); the RL
# stubs are the suspenders that make them import-safe even if kit finds them.
_EXCLUDE_EXTS = {"isaaclab_rl", "isaaclab_mimic"}

# Kit user configs a prior leisaac SimulationApp could poison. In this isolated
# design each container runs ONE SimulationApp, so these start clean; we still
# unlink them + use a fresh HOME as cheap insurance.
KIT_USER_CFGS = [
    "/isaac-sim/kit/data/Kit/Isaac-Sim/4.5/user.config.json",
    "/root/.local/share/ov/data/Kit/Isaac-Sim/4.5/user.config.json",
]

CREATE_TOWEL = "scripts/tera/create_clean_rect_towel_usd_v2.py"
INSPECT_SCRIPT = "scripts/tera/inspect_so101_drive_modes_v12.py"
DRIVE_SCRIPT = "scripts/tera/test_so101_fingertip_physics_drive_v12.py"
SUMMARY_SCRIPT = "scripts/tera/summarize_so101_contact_readiness_v12.py"

image = (
    modal.Image.from_registry("nvcr.io/nvidia/isaac-sim:4.5.0", add_python="3.10")
    .dockerfile_commands("ENTRYPOINT []")
    .apt_install("git", "tar", "curl")
    .run_commands(f"{ISAAC_PYTHON} -m pip install --no-cache-dir flatdict toml")
    # Bake IsaacLab v2.0.2 into the image (clone once, not per stage/container).
    .run_commands(
        f"git clone --depth 1 --branch {ISAACLAB_BRANCH} "
        f"https://github.com/isaac-sim/IsaacLab.git {ISAACLAB_ROOT}"
    )
    # leisaac source + the SO-101 follower USD (real arm asset).
    .run_commands(
        "git clone --depth 1 https://github.com/LightwheelAI/leisaac.git /opt/leisaac",
        f"mkdir -p {LEISAAC_ASSETS_ROOT}/robots",
        f"curl -fL -o {LEISAAC_ASSETS_ROOT}/robots/so101_follower.usd {SO101_USD_URL}",
    )
    # Old `gym` (gym.spaces) may be referenced by the isaaclab_rl shim; install
    # just gym. We do NOT install rl_games/rsl_rl/sb3/skrl (they drag a
    # conflicting torch/cu13) -> the RL stubs cover those imports instead.
    #
    # h5py: transitive dep of `isaaclab.envs` (managers.recorder_manager ->
    # utils.datasets -> HDF5DatasetFileHandler). Without it, importing
    # isaaclab.envs raises ModuleNotFoundError, which (a) fails isaaclab_rl /
    # isaaclab_tasks extension startup and (b) leaves those extensions half-loaded
    # so the heavier contact-drive script hard-crashes during sim reset. h5py is a
    # light HDF5 binding (no torch), so it is safe alongside Isaac's torch 2.5.1.
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


# --------------------------------------------------------------------------- #
# helpers (run inside the container)
# --------------------------------------------------------------------------- #
def _extract_status(stdout: str):
    """Pull the last JSON object with a 'status' key out of a script's stdout."""
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
    # scripts/tera resolves via the script dir (sys.path[0]); we still add work
    # for parity with the prior app.
    env["PYTHONPATH"] = ":".join(source_paths + [LEISAAC_SRC, str(work), env.get("PYTHONPATH", "")])
    env["LEISAAC_ASSETS_ROOT"] = LEISAAC_ASSETS_ROOT
    return env


KIT_LOG_DIR = Path("/isaac-sim/kit/logs/Kit/Isaac-Sim/4.5")


def _persist_full_logs(log_tag: str, proc):
    """Write the FULL stdout/stderr + the newest kit C++ log to the Volume, so a
    hard (non-Python) crash is diagnosable without truncation."""
    logs = ISO_DIR / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / f"{log_tag}.stdout.txt").write_text(proc.stdout or "")
    (logs / f"{log_tag}.stderr.txt").write_text(proc.stderr or "")
    try:
        kit_logs = sorted(KIT_LOG_DIR.glob("kit_*.log"), key=lambda p: p.stat().st_mtime)
        if kit_logs:
            text = kit_logs[-1].read_text(errors="replace")
            (logs / f"{log_tag}.kitlog.txt").write_text(text[-200000:])
    except Exception as exc:  # noqa: BLE001
        (logs / f"{log_tag}.kitlog_error.txt").write_text(repr(exc))


def _run_isaac_script(work: Path, env: dict, rel: str, log_tag: str, timeout: int = 2400):
    """Run ONE script -> ONE SimulationApp in this container. Clean kit cfg first."""
    for cfg in KIT_USER_CFGS:
        try:
            Path(cfg).unlink()
        except Exception:
            pass
    script_env = dict(env)
    home = "/tmp/kithome"
    Path(home).mkdir(parents=True, exist_ok=True)
    script_env["HOME"] = home
    proc = subprocess.run(
        [ISAAC_PYTHON, str(work / rel)], cwd=str(work), text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, env=script_env,
    )
    st = _extract_status(proc.stdout)
    _persist_full_logs(log_tag, proc)
    record = {
        "script": rel,
        "returncode": proc.returncode,
        "internal_status": (st or {}).get("status"),
        "stdout_tail": (proc.stdout or "")[-12000:],
        "stderr_tail": (proc.stderr or "")[-8000:],
    }
    return record, st


def _stage_towel_from_volume():
    """Copy the towel USD (+ grid, +any aux) from the Volume into the container
    path the repo scripts expect. Cross-container sharing via Volume only."""
    TOWEL_DIR.mkdir(parents=True, exist_ok=True)
    staged = []
    if ISO_ASSETS.exists():
        for f in ISO_ASSETS.iterdir():
            if f.is_file():
                (TOWEL_DIR / f.name).write_bytes(f.read_bytes())
                staged.append(f.name)
    return staged


def _persist_v12_jsons():
    """Copy any V12_DIR/*.json the stage produced onto the Volume (drive audit,
    physics-drive results) so the summary stage can read them."""
    ISO_DIR.mkdir(parents=True, exist_ok=True)
    persisted = []
    if V12_DIR.exists():
        for f in V12_DIR.glob("*.json"):
            (ISO_DIR / f.name).write_text(f.read_text())
            persisted.append(f.name)
    return persisted


def _write_stage_record(name: str, record: dict):
    ISO_DIR.mkdir(parents=True, exist_ok=True)
    (ISO_DIR / f"stage_{name}.json").write_text(json.dumps(record, indent=2))
    volume.commit()
    trimmed = {k: v for k, v in record.items()
               if k not in ("stdout_tail", "stderr_tail")}
    print(json.dumps(trimmed, indent=2))


# --------------------------------------------------------------------------- #
# Stage 0: author the clean towel USD, publish to the Volume
# --------------------------------------------------------------------------- #
@app.function(gpu="L40S", image=image, volumes={"/artifacts": volume}, timeout=1800)
def make_towel():
    work = _unpack_source()
    env = _isaac_env(work)
    run, _ = _run_isaac_script(work, env, CREATE_TOWEL, log_tag="0_make_towel", timeout=1200)

    ISO_ASSETS.mkdir(parents=True, exist_ok=True)
    published = []
    if TOWEL_DIR.exists():
        for f in TOWEL_DIR.iterdir():
            if f.is_file():
                (ISO_ASSETS / f.name).write_bytes(f.read_bytes())
                published.append(f.name)

    record = {
        "stage": "0_make_towel",
        "run": run,
        "published_assets": published,
        "towel_usd_present": any(n.endswith((".usd", ".usda", ".usdc")) for n in published),
    }
    _write_stage_record("0_make_towel", record)
    return {k: v for k, v in record.items() if k != "run"} | {
        "returncode": run["returncode"], "internal_status": run["internal_status"]
    }


# --------------------------------------------------------------------------- #
# Stage A: build the real SO-101 scene + drive audit (inspect)
# --------------------------------------------------------------------------- #
@app.function(gpu="L40S", image=image, volumes={"/artifacts": volume}, timeout=3600)
def build_inspect():
    work = _unpack_source()
    env = _isaac_env(work)
    staged = _stage_towel_from_volume()
    run, st = _run_isaac_script(work, env, INSPECT_SCRIPT, log_tag="A_build_inspect")
    persisted = _persist_v12_jsons()
    record = {
        "stage": "A_build_inspect",
        "staged_assets": staged,
        "run": run,
        "persisted_jsons": persisted,
        "so101_scene_built": bool((st or {}).get("so101_scene_built")),
        "internal_status": run["internal_status"],
    }
    _write_stage_record("A_build_inspect", record)
    return {
        "stage": "A_build_inspect",
        "returncode": run["returncode"],
        "internal_status": run["internal_status"],
        "so101_scene_built": record["so101_scene_built"],
        "stderr_tail": run["stderr_tail"][-2500:],
    }


# --------------------------------------------------------------------------- #
# Stages B+C: teleport baseline + physics-driven set_world_poses trials
# (one scene, one SimulationApp -> honest same-scene comparison)
# --------------------------------------------------------------------------- #
@app.function(gpu="L40S", image=image, volumes={"/artifacts": volume}, timeout=3600)
def physics_drive():
    work = _unpack_source()
    env = _isaac_env(work)
    staged = _stage_towel_from_volume()
    run, st = _run_isaac_script(work, env, DRIVE_SCRIPT, log_tag="BC_physics_drive")
    persisted = _persist_v12_jsons()

    trials = (st or {}).get("trials", []) if isinstance(st, dict) else []
    record = {
        "stage": "BC_physics_drive",
        "staged_assets": staged,
        "run": run,
        "persisted_jsons": persisted,
        "internal_status": run["internal_status"],
        "num_trials": len(trials),
        "any_valid_controlled_contact": bool((st or {}).get("any_valid_controlled_contact")),
    }
    _write_stage_record("BC_physics_drive", record)
    return {
        "stage": "BC_physics_drive",
        "returncode": run["returncode"],
        "internal_status": run["internal_status"],
        "num_trials": len(trials),
        "any_valid_controlled_contact": record["any_valid_controlled_contact"],
        "stderr_tail": run["stderr_tail"][-2500:],
    }


# --------------------------------------------------------------------------- #
# Stage D: summarize (CPU only; no Isaac). Reads drive JSON from the Volume,
# writes the required contact_v12_isolated_summary.json back to the Volume.
# --------------------------------------------------------------------------- #
@app.function(image=image, volumes={"/artifacts": volume}, timeout=600)
def summarize():
    work = _unpack_source()

    # Stage the drive/inspect JSONs from the Volume into the container path the
    # summary script reads (V12_DIR).
    V12_DIR.mkdir(parents=True, exist_ok=True)
    staged_in = []
    for name in ("so101_fingertip_physics_drive_v12.json", "so101_drive_modes_v12.json"):
        src = ISO_DIR / name
        if src.exists():
            (V12_DIR / name).write_text(src.read_text())
            staged_in.append(name)

    proc = subprocess.run(
        ["python3", str(work / SUMMARY_SCRIPT)], cwd=str(work), text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300,
    )
    st = _extract_status(proc.stdout)

    # Publish the required final artifact + a consolidated run record.
    ISO_DIR.mkdir(parents=True, exist_ok=True)
    summ_local = V12_DIR / "contact_v12_summary.json"
    summary = None
    if summ_local.exists():
        summary = json.loads(summ_local.read_text())
        (ISO_DIR / "contact_v12_isolated_summary.json").write_text(json.dumps(summary, indent=2))

    stage_records = {}
    for f in sorted(ISO_DIR.glob("stage_*.json")):
        try:
            stage_records[f.stem] = json.loads(f.read_text())
        except Exception:
            pass
    consolidated = {
        "status": "TERAFOLD_MODAL_ISAAC45_SO101_V12_ISOLATED_DONE",
        "staged_inputs": staged_in,
        "summary_present": summary is not None,
        "contact_v12_isolated_summary": summary,
        "stage_records": {
            k: {kk: vv for kk, vv in v.items() if kk != "run"}
            for k, v in stage_records.items()
        },
    }
    (ISO_DIR / "modal_contact_v12_isolated_run.json").write_text(json.dumps(consolidated, indent=2))
    volume.commit()

    trimmed = dict(consolidated)
    trimmed["summarize_returncode"] = proc.returncode
    trimmed["summarize_status"] = (st or {}).get("status")
    print(json.dumps({k: v for k, v in trimmed.items() if k != "stage_records"}, indent=2))
    return {
        "summary": summary,
        "summarize_returncode": proc.returncode,
        "staged_inputs": staged_in,
    }


# --------------------------------------------------------------------------- #
# Orchestrator (runs locally; chains the stage containers in order)
# --------------------------------------------------------------------------- #
@app.local_entrypoint()
def main(stages: str = "make_towel,build_inspect,physics_drive,summarize"):
    want = [s.strip() for s in stages.split(",") if s.strip()]
    results = {}

    if "make_towel" in want:
        print(">>> stage 0: make_towel"); results["make_towel"] = make_towel.remote()
        print(json.dumps(results["make_towel"], indent=2))
    if "build_inspect" in want:
        print(">>> stage A: build_inspect"); results["build_inspect"] = build_inspect.remote()
        print(json.dumps(results["build_inspect"], indent=2))
    if "physics_drive" in want:
        print(">>> stages B+C: physics_drive"); results["physics_drive"] = physics_drive.remote()
        print(json.dumps(results["physics_drive"], indent=2))
    if "summarize" in want:
        print(">>> stage D: summarize"); results["summarize"] = summarize.remote()

    summary = (results.get("summarize") or {}).get("summary") or {}
    verdict = {
        "V12_ISOLATED_VERDICT": {
            "make_towel_rc": (results.get("make_towel") or {}).get("returncode"),
            "make_towel_status": (results.get("make_towel") or {}).get("internal_status"),
            "inspect_status": (results.get("build_inspect") or {}).get("internal_status"),
            "so101_scene_built": (results.get("build_inspect") or {}).get("so101_scene_built"),
            "physics_drive_status": (results.get("physics_drive") or {}).get("internal_status"),
            "physics_drive_num_trials": (results.get("physics_drive") or {}).get("num_trials"),
            "any_valid_controlled_contact": (results.get("physics_drive") or {}).get("any_valid_controlled_contact"),
            "so101_contact_transfer_solved": summary.get("so101_contact_transfer_solved"),
            "teleport_baseline_moved_towel": summary.get("teleport_baseline_moved_towel"),
            "best_edge_displacement_m": summary.get("edge_displacement_m"),
            "best_width_before": summary.get("width_before"),
            "best_width_after": summary.get("width_after"),
            "robot_contact_data_ready": summary.get("robot_contact_data_ready"),
            "remaining_blocker": summary.get("remaining_blocker"),
        }
    }
    print(json.dumps(verdict, indent=2))
