"""Modal runner for the V12 SO-101 physics-driven contact suite (Isaac Sim 4.5.0).

Adds the real SO-101 arm to the proven Isaac 4.5.0 image:
- clone LeIsaac (github.com/LightwheelAI/leisaac) for SO101_FOLLOWER_CFG,
- download so101_follower.usd (GitHub release v0.1.0) into LEISAAC_ASSETS_ROOT,
- run the V12 scripts and persist contact_v12_summary.json to the volume.

The script list is controlled by `args` (default = feasibility probe only:
create towel USD + inspect SO-101 drive modes). Pass a list of script relpaths
to run the full suite.
"""

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import modal

app = modal.App("terafold-modal-isaac45-so101-v12")
volume = modal.Volume.from_name("terafold-artifacts")

ISAAC_PYTHON = "/isaac-sim/python.sh"
ISAACLAB_BRANCH = "v2.0.2"
LEISAAC_ASSETS_ROOT = "/opt/leisaac_assets"
LEISAAC_SRC = "/opt/leisaac/source/leisaac"
SO101_USD_URL = "https://github.com/LightwheelAI/leisaac/releases/download/v0.1.0/so101_follower.usd"
V12_DIR = "/workspace/leisaac/tera_checkpoints/clean_towel_v12"

DEFAULT_SCRIPTS = [
    "scripts/tera/create_clean_rect_towel_usd_v2.py",
    "scripts/tera/inspect_so101_drive_modes_v12.py",
    "scripts/tera/test_so101_fingertip_physics_drive_v12.py",
    "scripts/tera/run_so101_contact_demo_collection_v12.py",
    "scripts/tera/summarize_so101_contact_readiness_v12.py",
]

image = (
    modal.Image.from_registry("nvcr.io/nvidia/isaac-sim:4.5.0", add_python="3.10")
    .dockerfile_commands("ENTRYPOINT []")
    .apt_install("git", "tar", "curl")
    .run_commands(f"{ISAAC_PYTHON} -m pip install --no-cache-dir flatdict toml")
    .run_commands(
        "git clone --depth 1 https://github.com/LightwheelAI/leisaac.git /opt/leisaac",
        f"mkdir -p {LEISAAC_ASSETS_ROOT}/robots",
        f"curl -fL -o {LEISAAC_ASSETS_ROOT}/robots/so101_follower.usd {SO101_USD_URL}",
        f"ls -la {LEISAAC_ASSETS_ROOT}/robots",
    )
    # Old `gym` (gym.spaces) is imported by the isaaclab_rl extension shim if it
    # auto-loads; install just gym (do NOT install rl_games/rsl_rl/sb3/skrl, which
    # drag in a conflicting torch build). We instead prevent isaaclab_rl autoload
    # by isolating each SO-101 script in its own container (fresh kit cache).
    .run_commands(f"{ISAAC_PYTHON} -m pip install --no-cache-dir 'gym==0.26.2'")
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
    for m in re.finditer(r"\{", stdout):
        frag = stdout[m.start():]
        try:
            obj, _ = json.JSONDecoder().raw_decode(frag)
            if isinstance(obj, dict) and "status" in obj:
                found = obj
        except Exception:
            continue
    return found


@app.function(gpu="L40S", image=image, volumes={"/artifacts": volume}, timeout=7200)
def run(scripts=None):
    scripts = scripts or DEFAULT_SCRIPTS
    result = {
        "status": "TERAFOLD_MODAL_ISAAC45_SO101_V12_STARTED",
        "isaac_sim_version_target": "4.5.0",
        "leisaac_assets_root": LEISAAC_ASSETS_ROOT,
        "so101_usd_present": Path(f"{LEISAAC_ASSETS_ROOT}/robots/so101_follower.usd").exists(),
        "repo_unpacked": False,
        "isaaclab_cloned": False,
        "script_runs": [],
        "contact_v12_summary": None,
        "so101_contact_transfer_solved": None,
        "final_robot_contact_policy_training_ready": None,
        "remaining_blockers": [],
    }

    work = Path("/tmp/terafold_repo_smoke")
    lab = Path("/tmp/IsaacLab")
    for p in (work, lab):
        if p.exists():
            shutil.rmtree(p)
    work.mkdir(parents=True, exist_ok=True)

    unpack = subprocess.run(
        ["tar", "-xzf", "/opt/terafold_modal_src.tar.gz", "-C", str(work)],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300,
    )
    if unpack.returncode != 0:
        result["status"] = "TERAFOLD_MODAL_ISAAC45_SO101_V12_UNPACK_FAILED"
        result["unpack_stderr_tail"] = unpack.stderr[-3000:]
        _write(result)
        return result
    result["repo_unpacked"] = True

    clone = subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", ISAACLAB_BRANCH,
         "https://github.com/isaac-sim/IsaacLab.git", str(lab)],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=900,
    )
    if clone.returncode != 0:
        result["status"] = "TERAFOLD_MODAL_ISAAC45_SO101_V12_CLONE_FAILED"
        result["clone_stderr_tail"] = clone.stderr[-3000:]
        _write(result)
        return result
    result["isaaclab_cloned"] = True

    source_root = lab / "source"
    # Exclude RL-only IsaacLab extensions from PYTHONPATH: when discoverable they
    # auto-load at SimulationApp startup and import rl_games/rsl_rl/sb3/skrl (+old
    # gym), which we do not need and are not installed -> fatal extension startup.
    _exclude = {"isaaclab_rl", "isaaclab_mimic"}
    source_paths = (
        [str(p) for p in source_root.iterdir() if p.is_dir() and p.name not in _exclude]
        if source_root.exists() else []
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = ":".join(source_paths + [LEISAAC_SRC, str(work), env.get("PYTHONPATH", "")])
    env["LEISAAC_ASSETS_ROOT"] = LEISAAC_ASSETS_ROOT
    result["leisaac_src_present"] = Path(LEISAAC_SRC).exists()

    # A leisaac-importing app poisons the kit user config (enables isaaclab_rl as
    # a required ext) so the NEXT app in this container fatally fails at startup.
    # Clear the kit user config + a fresh HOME before each script so every Isaac
    # launch gets a clean extension config.
    kit_user_cfgs = [
        "/isaac-sim/kit/data/Kit/Isaac-Sim/4.5/user.config.json",
        "/root/.local/share/ov/data/Kit/Isaac-Sim/4.5/user.config.json",
    ]
    for i, rel in enumerate(scripts):
        for cfg in kit_user_cfgs:
            try:
                Path(cfg).unlink()
            except Exception:
                pass
        script_env = dict(env)
        script_env["HOME"] = f"/tmp/kithome_{i}"
        Path(f"/tmp/kithome_{i}").mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(
            [ISAAC_PYTHON, str(work / rel)], cwd=str(work), text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=2400, env=script_env,
        )
        st = _extract_status(proc.stdout)
        result["script_runs"].append({
            "script": rel, "returncode": proc.returncode,
            "internal_status": (st or {}).get("status"),
            "stdout_tail": proc.stdout[-11000:], "stderr_tail": proc.stderr[-7000:],
        })

    # persist V12 artifacts (drive json + demos) to the volume for Training v2
    out_dir = Path("/artifacts/contact_v12")
    (out_dir / "demos").mkdir(parents=True, exist_ok=True)
    for src in (Path(V12_DIR) / "so101_fingertip_physics_drive_v12.json",
                Path(V12_DIR) / "so101_drive_modes_v12.json"):
        if src.exists():
            (out_dir / src.name).write_text(src.read_text())
    demo_src = Path(V12_DIR) / "demos"
    if demo_src.exists():
        for f in demo_src.iterdir():
            if f.is_file():
                (out_dir / "demos" / f.name).write_bytes(f.read_bytes())

    summ_path = Path(V12_DIR) / "contact_v12_summary.json"
    if summ_path.exists():
        try:
            summ = json.loads(summ_path.read_text())
            result["contact_v12_summary"] = summ
            result["so101_contact_transfer_solved"] = summ.get("so101_contact_transfer_solved")
            result["final_robot_contact_policy_training_ready"] = summ.get(
                "final_robot_contact_policy_training_ready"
            )
        except Exception as exc:  # noqa: BLE001
            result["remaining_blockers"].append(f"summary parse error: {exc!r}")

    result["status"] = "TERAFOLD_MODAL_ISAAC45_SO101_V12_DONE"
    _write(result)
    return result


def _write(result):
    out_dir = Path("/artifacts/contact_v12")
    out_dir.mkdir(parents=True, exist_ok=True)
    if result.get("contact_v12_summary") is not None:
        (out_dir / "contact_v12_summary.json").write_text(json.dumps(result["contact_v12_summary"], indent=2))
    (out_dir / "modal_contact_v12_run.json").write_text(json.dumps(result, indent=2))
    volume.commit()
    trimmed = {k: v for k, v in result.items() if k != "script_runs"}
    trimmed["script_runs_brief"] = [
        {"script": s["script"], "returncode": s["returncode"], "internal_status": s["internal_status"]}
        for s in result.get("script_runs", [])
    ]
    print(json.dumps(trimmed, indent=2))


@app.local_entrypoint()
def main(scripts: str = ""):
    script_list = [s for s in scripts.split(",") if s.strip()] or None
    print(run.remote(script_list))
