"""Isolated per-container Modal runner for the V17 DEEP-but-SLOW state-reactive
dataset (Isaac Sim 4.5.0 + IsaacLab v2.0.2).

Same proven isolated environment as V14/V16 (RL stubs + h5py + one SimulationApp
per container + single-reset pre_reset_spawn). Collects DEEP-but-SLOW real SO-101
articulation-driven fold demos (large inward drag over a long, slow duration) with
the V14 disk mechanism and the V16 state-reactive observation (NO phase clock),
then validates + summarizes on CPU.

Stages:
  demo_collection()  -> run_so101_deep_slow_demo_collection_v17 (GPU, Isaac)
  validate()         -> validate_so101_deep_slow_demos_v17 (CPU, numpy-only via ISAAC_PYTHON)
  summarize()        -> summarize_so101_deep_slow_dataset_v17 (CPU, numpy-only via ISAAC_PYTHON)

Final artifacts on the volume:
  /artifacts/contact_v17/demos/so101_deep_slow_demos_v17.npz
  /artifacts/contact_v17/demos/so101_deep_slow_demos_manifest_v17.json
  /artifacts/contact_v17/contact_v17_demo_dataset_summary.json
"""

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import modal

app = modal.App("terafold-modal-isaac45-so101-v17")
volume = modal.Volume.from_name("terafold-artifacts")

ISAAC_PYTHON = "/isaac-sim/python.sh"
ISAACLAB_BRANCH = "v2.0.2"
ISAACLAB_ROOT = "/opt/IsaacLab"
LEISAAC_ASSETS_ROOT = "/opt/leisaac_assets"
LEISAAC_SRC = "/opt/leisaac/source/leisaac"
SO101_USD_URL = "https://github.com/LightwheelAI/leisaac/releases/download/v0.1.0/so101_follower.usd"

TOWEL_DIR = Path("/workspace/leisaac/tera_checkpoints/clean_towel_v2")
V13_DIR = Path("/workspace/leisaac/tera_checkpoints/clean_towel_v13")
V17_DIR = Path("/workspace/leisaac/tera_checkpoints/clean_towel_v17")

ISO_DIR = Path("/artifacts/contact_v17")
V12_ASSETS = Path("/artifacts/contact_v12_isolated/assets")
V13_ARTIFACTS = Path("/artifacts/contact_v13")

_EXCLUDE_EXTS = {"isaaclab_rl", "isaaclab_mimic"}
KIT_USER_CFGS = [
    "/isaac-sim/kit/data/Kit/Isaac-Sim/4.5/user.config.json",
    "/root/.local/share/ov/data/Kit/Isaac-Sim/4.5/user.config.json",
]
KIT_LOG_DIR = Path("/isaac-sim/kit/logs/Kit/Isaac-Sim/4.5")

DEMO_SCRIPT = "scripts/tera/run_so101_deep_slow_demo_collection_v17.py"
VALIDATE_SCRIPT = "scripts/tera/validate_so101_deep_slow_demos_v17.py"
SUMMARIZE_SCRIPT = "scripts/tera/summarize_so101_deep_slow_dataset_v17.py"
DISK_GEOM = ["--patch_shape", "cylinder", "--patch_radius", "0.14", "--patch_length", "0.03",
             "--patch_axis", "Z", "--patch_contact_offset", "0.04"]

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


def _stage_towel_from_volume():
    TOWEL_DIR.mkdir(parents=True, exist_ok=True)
    staged = []
    if V12_ASSETS.exists():
        for f in V12_ASSETS.iterdir():
            if f.is_file():
                (TOWEL_DIR / f.name).write_bytes(f.read_bytes())
                staged.append(f.name)
    return staged


def _stage_reach_json_in():
    V13_DIR.mkdir(parents=True, exist_ok=True)
    staged = []
    src = V13_ARTIFACTS / "so101_articulation_reach_v13.json"
    if src.exists():
        (V13_DIR / src.name).write_text(src.read_text())
        staged.append(src.name)
    return staged


def _persist_v17():
    ISO_DIR.mkdir(parents=True, exist_ok=True)
    (ISO_DIR / "demos").mkdir(parents=True, exist_ok=True)
    persisted = []
    demo_src = V17_DIR / "demos"
    if demo_src.exists():
        for f in demo_src.iterdir():
            if f.is_file():
                (ISO_DIR / "demos" / f.name).write_bytes(f.read_bytes())
                persisted.append(f"demos/{f.name}")
    return persisted


def _write_stage_record(name: str, record: dict):
    ISO_DIR.mkdir(parents=True, exist_ok=True)
    (ISO_DIR / f"stage_{name}.json").write_text(json.dumps(record, indent=2))
    volume.commit()
    print(json.dumps({k: v for k, v in record.items() if k not in ("stdout_tail", "stderr_tail")}, indent=2))


@app.function(gpu="L40S", image=image, volumes={"/artifacts": volume}, timeout=3600)
def demo_collection(num_demos: int = 34, seed: int = 1717):
    work = _unpack_source()
    env = _isaac_env(work)
    staged = _stage_towel_from_volume()
    reach = _stage_reach_json_in()
    extra = ["--num_demos", str(int(num_demos)), "--seed", str(int(seed))] + DISK_GEOM
    run, st = _run_isaac_script(work, env, DEMO_SCRIPT, log_tag="demo_v17", extra_args=extra)
    persisted = _persist_v17()
    record = {"stage": "demo_collection_v17", "staged_assets": staged, "reach_staged": reach, "run": run,
              "persisted": persisted, "internal_status": run["internal_status"],
              "robot_contact_data_ready": bool((st or {}).get("robot_contact_data_ready")),
              "num_valid_success": (st or {}).get("num_valid_success"),
              "num_ballistic": (st or {}).get("num_ballistic"),
              "valid_mean_width_after": (st or {}).get("valid_mean_width_after"),
              "beats_v16_demo_mean": (st or {}).get("beats_v16_demo_mean"),
              "total_samples": (st or {}).get("total_samples")}
    _write_stage_record("demo_v17", record)
    return {"returncode": run["returncode"], "internal_status": run["internal_status"],
            "robot_contact_data_ready": record["robot_contact_data_ready"],
            "num_valid_success": record["num_valid_success"], "num_ballistic": record["num_ballistic"],
            "valid_mean_width_after": record["valid_mean_width_after"],
            "beats_v16_demo_mean": record["beats_v16_demo_mean"],
            "total_samples": record["total_samples"], "stderr_tail": run["stderr_tail"][-2500:]}


@app.function(image=image, volumes={"/artifacts": volume}, timeout=600)
def validate():
    work = _unpack_source()
    # ISAAC_PYTHON (system python3 on this image lacks numpy)
    proc = subprocess.run(
        [ISAAC_PYTHON, str(work / VALIDATE_SCRIPT)], cwd=str(work), text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300,
    )
    volume.commit()
    st = _extract_status(proc.stdout) or {}
    print((proc.stdout or "")[-4000:])
    if proc.returncode != 0:
        print((proc.stderr or "")[-3000:])
    return {"returncode": proc.returncode, "valid_dataset": st.get("valid_dataset"),
            "checks": st.get("checks"), "num_valid_success": st.get("num_valid_success")}


@app.function(image=image, volumes={"/artifacts": volume}, timeout=600)
def summarize():
    work = _unpack_source()
    proc = subprocess.run(
        [ISAAC_PYTHON, str(work / SUMMARIZE_SCRIPT)], cwd=str(work), text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300,
    )
    volume.commit()
    summ_path = ISO_DIR / "contact_v17_demo_dataset_summary.json"
    summary = json.loads(summ_path.read_text()) if summ_path.exists() else None
    if summary:
        print(json.dumps(summary, indent=2))
    else:
        print((proc.stdout or "")[-2000:]); print((proc.stderr or "")[-2000:])
    return {"returncode": proc.returncode, "summary": summary}


@app.local_entrypoint()
def main(stages: str = "demo_collection,validate,summarize", num_demos: int = 34, seed: int = 1717):
    want = [s.strip() for s in stages.split(",") if s.strip()]
    results = {}
    if "demo_collection" in want:
        results["demo_collection"] = demo_collection.remote(num_demos, seed)
        print(json.dumps(results["demo_collection"], indent=2))
    if "validate" in want:
        results["validate"] = validate.remote()
        print(json.dumps(results["validate"], indent=2))
    if "summarize" in want:
        results["summarize"] = summarize.remote()
    summ = (results.get("summarize") or {}).get("summary") or {}
    print(json.dumps({"V17_DATASET_VERDICT": {
        "num_valid_success": summ.get("num_valid_success"),
        "num_ballistic": summ.get("num_ballistic"),
        "num_nofold": summ.get("num_nofold"),
        "total_samples": summ.get("total_samples"),
        "valid_mean_width_after": summ.get("valid_mean_width_after"),
        "valid_min_width_after": summ.get("valid_min_width_after"),
        "valid_num_below_0_55": summ.get("valid_num_below_0_55"),
        "beats_v16_demo_mean": summ.get("beats_v16_demo_mean"),
        "valid_mean_width_after_below_0_55": summ.get("valid_mean_width_after_below_0_55"),
        "no_phase_clock": summ.get("no_phase_clock"),
        "robot_contact_data_ready": summ.get("robot_contact_data_ready"),
        "v17_dataset_ready_for_training_v4": summ.get("v17_dataset_ready_for_training_v4"),
        "exact_next_step": summ.get("exact_next_step"),
    }}, indent=2))
