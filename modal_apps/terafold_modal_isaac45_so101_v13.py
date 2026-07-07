"""Isolated per-container Modal runner for V13 SO-101 ARTICULATION-driven contact
(Isaac Sim 4.5.0 + IsaacLab v2.0.2).

Builds on the proven V12-isolated fixes (RL import stubs + h5py + one
SimulationApp per container + single-reset build order). V13 drives the REAL
SO-101 arm articulation (physics-resolved joint position targets), not a free
proxy, to reach and fold the towel, and records REAL (state, joint_action,
contact) demos.

Stages (each its own container; state shared only via the terafold-artifacts
volume; towel USD reused from the V12 assets already on the volume):
  A. reach_search       -> search_so101_articulation_reach_v13
  B. contact_fold       -> test_so101_articulation_contact_fold_v13
  C. demo_collection    -> run_so101_real_contact_demo_collection_v13
  D. summarize          -> summarize_so101_articulation_readiness_v13  (CPU)

Final artifact: /artifacts/contact_v13/contact_v13_articulation_summary.json
"""

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import modal

app = modal.App("terafold-modal-isaac45-so101-v13")
volume = modal.Volume.from_name("terafold-artifacts")

ISAAC_PYTHON = "/isaac-sim/python.sh"
ISAACLAB_BRANCH = "v2.0.2"
ISAACLAB_ROOT = "/opt/IsaacLab"
LEISAAC_ASSETS_ROOT = "/opt/leisaac_assets"
LEISAAC_SRC = "/opt/leisaac/source/leisaac"
SO101_USD_URL = "https://github.com/LightwheelAI/leisaac/releases/download/v0.1.0/so101_follower.usd"

TOWEL_DIR = Path("/workspace/leisaac/tera_checkpoints/clean_towel_v2")
V13_DIR = Path("/workspace/leisaac/tera_checkpoints/clean_towel_v13")

ISO_DIR = Path("/artifacts/contact_v13")
V12_ASSETS = Path("/artifacts/contact_v12_isolated/assets")  # reuse proven towel USD

_EXCLUDE_EXTS = {"isaaclab_rl", "isaaclab_mimic"}
KIT_USER_CFGS = [
    "/isaac-sim/kit/data/Kit/Isaac-Sim/4.5/user.config.json",
    "/root/.local/share/ov/data/Kit/Isaac-Sim/4.5/user.config.json",
]
KIT_LOG_DIR = Path("/isaac-sim/kit/logs/Kit/Isaac-Sim/4.5")

REACH_SCRIPT = "scripts/tera/search_so101_articulation_reach_v13.py"
FOLD_SCRIPT = "scripts/tera/test_so101_articulation_contact_fold_v13.py"
DEMO_SCRIPT = "scripts/tera/run_so101_real_contact_demo_collection_v13.py"
SUMMARY_SCRIPT = "scripts/tera/summarize_so101_articulation_readiness_v13.py"
CREATE_TOWEL = "scripts/tera/create_clean_rect_towel_usd_v2.py"

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
    # gym for isaaclab_rl shim; h5py so isaaclab.envs imports cleanly (see V12).
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


def _run_isaac_script(work: Path, env: dict, rel: str, log_tag: str, extra_args=None, timeout: int = 3000):
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
    """Reuse the proven V12 towel USD (deterministic) from the volume."""
    TOWEL_DIR.mkdir(parents=True, exist_ok=True)
    staged = []
    if V12_ASSETS.exists():
        for f in V12_ASSETS.iterdir():
            if f.is_file():
                (TOWEL_DIR / f.name).write_bytes(f.read_bytes())
                staged.append(f.name)
    return staged


def _persist_v13_jsons():
    ISO_DIR.mkdir(parents=True, exist_ok=True)
    persisted = []
    if V13_DIR.exists():
        for f in V13_DIR.glob("*.json"):
            (ISO_DIR / f.name).write_text(f.read_text())
            persisted.append(f.name)
        demo_src = V13_DIR / "demos"
        if demo_src.exists():
            (ISO_DIR / "demos").mkdir(parents=True, exist_ok=True)
            for f in demo_src.iterdir():
                if f.is_file():
                    (ISO_DIR / "demos" / f.name).write_bytes(f.read_bytes())
                    persisted.append(f"demos/{f.name}")
    return persisted


def _stage_v13_jsons_in():
    """Copy prior V13 JSONs from the volume back into the container V13 dir so a
    later stage (fold reads reach; demo reads fold; summary reads all) finds them."""
    V13_DIR.mkdir(parents=True, exist_ok=True)
    staged = []
    if ISO_DIR.exists():
        for f in ISO_DIR.glob("*.json"):
            (V13_DIR / f.name).write_text(f.read_text())
            staged.append(f.name)
    return staged


def _write_stage_record(name: str, record: dict):
    ISO_DIR.mkdir(parents=True, exist_ok=True)
    (ISO_DIR / f"stage_{name}.json").write_text(json.dumps(record, indent=2))
    volume.commit()
    print(json.dumps({k: v for k, v in record.items() if k not in ("stdout_tail", "stderr_tail")}, indent=2))


# --------------------------------------------------------------------------- #
# Stage A: articulation reach search
# --------------------------------------------------------------------------- #
@app.function(gpu="L40S", image=image, volumes={"/artifacts": volume}, timeout=3600)
def reach_search(right_base=None, num_candidates=640):
    work = _unpack_source()
    env = _isaac_env(work)
    staged = _stage_towel_from_volume()
    extra = ["--num_candidates_per_arm", str(int(num_candidates))]
    if right_base:
        extra += ["--right_base", *[str(x) for x in right_base]]
    run, st = _run_isaac_script(work, env, REACH_SCRIPT, log_tag="A_reach", extra_args=extra)
    persisted = _persist_v13_jsons()
    record = {
        "stage": "A_reach_search", "staged_assets": staged, "run": run, "persisted": persisted,
        "internal_status": run["internal_status"],
        "reachable_edge_found": bool((st or {}).get("reachable_edge_found")),
        "best_jaw_to_edge_center_distance_m": (st or {}).get("best_jaw_to_edge_center_distance_m"),
    }
    _write_stage_record("A_reach_search", record)
    return {
        "stage": "A_reach_search", "returncode": run["returncode"],
        "internal_status": run["internal_status"],
        "reachable_edge_found": record["reachable_edge_found"],
        "best_jaw_to_edge_center_distance_m": record["best_jaw_to_edge_center_distance_m"],
        "stderr_tail": run["stderr_tail"][-2500:],
    }


# --------------------------------------------------------------------------- #
# Stage B: articulation contact fold (real arm + jaw-attached contact patch)
# --------------------------------------------------------------------------- #
@app.function(gpu="L40S", image=image, volumes={"/artifacts": volume}, timeout=3600)
def contact_fold(right_base=None):
    work = _unpack_source()
    env = _isaac_env(work)
    staged = _stage_towel_from_volume()
    staged_json = _stage_v13_jsons_in()  # bring the reach JSON into the container
    extra = []
    if right_base:
        extra += ["--right_base", *[str(x) for x in right_base]]
    run, st = _run_isaac_script(work, env, FOLD_SCRIPT, log_tag="B_fold", extra_args=extra)
    persisted = _persist_v13_jsons()
    fold = (st or {}).get("fold", {}) if isinstance(st, dict) else {}
    record = {
        "stage": "B_contact_fold", "staged_assets": staged, "staged_json": staged_json,
        "run": run, "persisted": persisted, "internal_status": run["internal_status"],
        "articulation_contact_fold_solved": bool((st or {}).get("articulation_contact_fold_solved")),
        "fold_metrics": fold, "patch_attach": (st or {}).get("patch_attach"),
    }
    _write_stage_record("B_contact_fold", record)
    return {
        "stage": "B_contact_fold", "returncode": run["returncode"],
        "internal_status": run["internal_status"],
        "articulation_contact_fold_solved": record["articulation_contact_fold_solved"],
        "fold_metrics": fold, "patch_attach": (st or {}).get("patch_attach"),
        "stderr_tail": run["stderr_tail"][-2500:],
    }


# --------------------------------------------------------------------------- #
# Stage C: real robot-contact demo collection (only meaningful if fold solved)
# --------------------------------------------------------------------------- #
@app.function(gpu="L40S", image=image, volumes={"/artifacts": volume}, timeout=3600)
def demo_collection(right_base=None):
    work = _unpack_source()
    env = _isaac_env(work)
    staged = _stage_towel_from_volume()
    _stage_v13_jsons_in()
    # also stage the fold action trajectory npz back into the container
    src_npz = ISO_DIR / "demos" / "so101_articulation_fold_actions_v13.npz"
    alt_npz = ISO_DIR / "so101_articulation_fold_actions_v13.npz"
    V13_DIR.mkdir(parents=True, exist_ok=True)
    for cand in (alt_npz, src_npz):
        if cand.exists():
            (V13_DIR / "so101_articulation_fold_actions_v13.npz").write_bytes(cand.read_bytes())
            break
    extra = ["--right_base", *[str(x) for x in (right_base or [])]] if right_base else []
    run, st = _run_isaac_script(work, env, DEMO_SCRIPT, log_tag="C_demo", extra_args=extra)
    persisted = _persist_v13_jsons()
    record = {
        "stage": "C_demo_collection", "run": run, "persisted": persisted,
        "internal_status": run["internal_status"],
        "robot_contact_data_ready": bool((st or {}).get("robot_contact_data_ready")),
        "total_samples": (st or {}).get("total_samples"),
    }
    _write_stage_record("C_demo_collection", record)
    return {
        "stage": "C_demo_collection", "returncode": run["returncode"],
        "internal_status": run["internal_status"],
        "robot_contact_data_ready": record["robot_contact_data_ready"],
        "total_samples": record["total_samples"], "stderr_tail": run["stderr_tail"][-2500:],
    }


# --------------------------------------------------------------------------- #
# Stage D: summarize (CPU only; reads reach + fold + demo JSONs from the volume)
# --------------------------------------------------------------------------- #
@app.function(image=image, volumes={"/artifacts": volume}, timeout=600)
def summarize():
    work = _unpack_source()
    _stage_v13_jsons_in()
    # stage the demo manifest into the container demos dir
    demo_src = ISO_DIR / "demos" / "so101_real_contact_demos_manifest_v13.json"
    if demo_src.exists():
        (V13_DIR / "demos").mkdir(parents=True, exist_ok=True)
        (V13_DIR / "demos" / demo_src.name).write_text(demo_src.read_text())
    proc = subprocess.run(
        ["python3", str(work / SUMMARY_SCRIPT)], cwd=str(work), text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300,
    )
    st = _extract_status(proc.stdout)
    ISO_DIR.mkdir(parents=True, exist_ok=True)
    summ_local = V13_DIR / "contact_v13_articulation_summary.json"
    summary = None
    if summ_local.exists():
        summary = json.loads(summ_local.read_text())
        (ISO_DIR / "contact_v13_articulation_summary.json").write_text(json.dumps(summary, indent=2))
    stage_records = {}
    for f in sorted(ISO_DIR.glob("stage_*.json")):
        try:
            stage_records[f.stem] = {k: v for k, v in json.loads(f.read_text()).items() if k != "run"}
        except Exception:
            pass
    consolidated = {
        "status": "TERAFOLD_MODAL_ISAAC45_SO101_V13_DONE",
        "summary_present": summary is not None,
        "contact_v13_articulation_summary": summary, "stage_records": stage_records,
    }
    (ISO_DIR / "modal_contact_v13_run.json").write_text(json.dumps(consolidated, indent=2))
    volume.commit()
    print(json.dumps({k: v for k, v in consolidated.items() if k != "stage_records"}, indent=2))
    return {"summary": summary, "summarize_returncode": proc.returncode}


@app.local_entrypoint()
def main(stages: str = "reach_search", right_base: str = ""):
    want = [s.strip() for s in stages.split(",") if s.strip()]
    rb = [float(x) for x in right_base.split(",")] if right_base else None
    results = {}
    if "reach_search" in want:
        print(">>> stage A: reach_search")
        results["reach_search"] = reach_search.remote(right_base=rb)
        print(json.dumps(results["reach_search"], indent=2))
    if "contact_fold" in want:
        print(">>> stage B: contact_fold")
        results["contact_fold"] = contact_fold.remote(right_base=rb)
        print(json.dumps(results["contact_fold"], indent=2))
    if "demo_collection" in want:
        print(">>> stage C: demo_collection")
        results["demo_collection"] = demo_collection.remote(right_base=rb)
        print(json.dumps(results["demo_collection"], indent=2))
    if "summarize" in want:
        print(">>> stage D: summarize")
        results["summarize"] = summarize.remote()
    summ = (results.get("summarize") or {}).get("summary") or {}
    print(json.dumps({"V13_VERDICT": {
        "reach_status": (results.get("reach_search") or {}).get("internal_status"),
        "reachable_edge_found": (results.get("reach_search") or {}).get("reachable_edge_found"),
        "best_jaw_to_edge_center_distance_m": (results.get("reach_search") or {}).get("best_jaw_to_edge_center_distance_m"),
        "fold_status": (results.get("contact_fold") or {}).get("internal_status"),
        "articulation_contact_fold_solved": (results.get("contact_fold") or {}).get("articulation_contact_fold_solved"),
        "fold_metrics": (results.get("contact_fold") or {}).get("fold_metrics"),
        "robot_contact_data_ready": summ.get("robot_contact_data_ready"),
        "final_robot_contact_policy_training_ready": summ.get("final_robot_contact_policy_training_ready"),
        "remaining_blocker": summ.get("remaining_blocker"),
        "exact_next_step": summ.get("exact_next_step"),
    }}, indent=2))
