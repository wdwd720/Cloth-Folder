"""Modal Eval v4 — MULTI-SEED closed-loop eval of the DEEP-SLOW state-reactive policy.

Runs `eval_so101_state_reactive_multiseed_v4.py` inside the proven Isaac Sim 4.5.0 +
IsaacLab v2.0.2 container (one SimulationApp; the eval loops over seeds × EMA betas
internally). The Training-v4 checkpoint drives the SO-101 arm CLOSED-LOOP with the V16
state-reactive observation (NO clock), physics-resolved articulation, and the winning
V14 disk geometry. Reports the per-seed / per-beta distribution and the best beta.

Final artifact on the volume:
  /artifacts/eval_v4_deep_slow/eval_v4_deep_slow_summary.json
"""

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import modal

app = modal.App("terafold-policy-eval-v4-deep-slow")
volume = modal.Volume.from_name("terafold-artifacts")

ISAAC_PYTHON = "/isaac-sim/python.sh"
ISAACLAB_BRANCH = "v2.0.2"
ISAACLAB_ROOT = "/opt/IsaacLab"
LEISAAC_ASSETS_ROOT = "/opt/leisaac_assets"
LEISAAC_SRC = "/opt/leisaac/source/leisaac"
SO101_USD_URL = "https://github.com/LightwheelAI/leisaac/releases/download/v0.1.0/so101_follower.usd"

TOWEL_DIR = Path("/workspace/leisaac/tera_checkpoints/clean_towel_v2")
EVAL_DIR = Path("/workspace/leisaac/tera_checkpoints/eval_v4_deep_slow")
ISO_DIR = Path("/artifacts/eval_v4_deep_slow")
V12_ASSETS = Path("/artifacts/contact_v12_isolated/assets")
CKPT_PATH = "/artifacts/training_v4_deep_slow/policy_v4_deep_slow.pt"

_EXCLUDE_EXTS = {"isaaclab_rl", "isaaclab_mimic"}
KIT_USER_CFGS = [
    "/isaac-sim/kit/data/Kit/Isaac-Sim/4.5/user.config.json",
    "/root/.local/share/ov/data/Kit/Isaac-Sim/4.5/user.config.json",
]
KIT_LOG_DIR = Path("/isaac-sim/kit/logs/Kit/Isaac-Sim/4.5")

EVAL_SCRIPT = "scripts/tera/eval_so101_state_reactive_multiseed_v4.py"
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


@app.function(gpu="L40S", image=image, volumes={"/artifacts": volume}, timeout=3600)
def eval_policy(seeds: str = "3030,4040,5050", betas: str = "0.0,0.3,0.4,0.6",
                episodes_per_config: int = 10, rollout_steps: int = 240, time_budget_s: float = 3000.0):
    work = _unpack_source()
    env = _isaac_env(work)
    staged = _stage_towel_from_volume()
    if not Path(CKPT_PATH).exists():
        rec = {"stage": "eval_v4", "error": f"missing checkpoint {CKPT_PATH}", "staged_assets": staged}
        ISO_DIR.mkdir(parents=True, exist_ok=True)
        (ISO_DIR / "stage_eval_v4.json").write_text(json.dumps(rec, indent=2))
        volume.commit()
        return {"returncode": 1, "error": rec["error"]}
    extra = (["--checkpoint", CKPT_PATH,
              "--seeds", *[s for s in seeds.split(",") if s],
              "--betas", *[b for b in betas.split(",") if b],
              "--episodes_per_config", str(int(episodes_per_config)),
              "--rollout_steps", str(int(rollout_steps)),
              "--time_budget_s", str(float(time_budget_s))] + DISK_GEOM)
    run, st = _run_isaac_script(work, env, EVAL_SCRIPT, log_tag="eval_v4", extra_args=extra)

    ISO_DIR.mkdir(parents=True, exist_ok=True)
    persisted = []
    if EVAL_DIR.exists():
        for f in EVAL_DIR.glob("*.json"):
            (ISO_DIR / f.name).write_text(f.read_text())
            persisted.append(f.name)
    record = {
        "stage": "eval_v4", "staged_assets": staged, "persisted": persisted,
        "internal_status": run["internal_status"],
        "best_beta": (st or {}).get("best_beta"),
        "best_beta_mean_clean_success_rate": (st or {}).get("best_beta_mean_clean_success_rate"),
        "best_beta_mean_ballistic_rate": (st or {}).get("best_beta_mean_ballistic_rate"),
        "best_beta_max_ballistic_rate": (st or {}).get("best_beta_max_ballistic_rate"),
        "best_beta_mean_width_after": (st or {}).get("best_beta_mean_width_after"),
        "robust_autonomous_so101_folding_validated": (st or {}).get("robust_autonomous_so101_folding_validated"),
        "by_beta": (st or {}).get("by_beta"),
    }
    (ISO_DIR / "stage_eval_v4.json").write_text(json.dumps(record, indent=2))
    volume.commit()
    print(json.dumps(record, indent=2))
    return {"returncode": run["returncode"], "internal_status": run["internal_status"],
            "best_beta": record["best_beta"],
            "best_beta_mean_clean_success_rate": record["best_beta_mean_clean_success_rate"],
            "best_beta_max_ballistic_rate": record["best_beta_max_ballistic_rate"],
            "best_beta_mean_width_after": record["best_beta_mean_width_after"],
            "robust_autonomous_so101_folding_validated": record["robust_autonomous_so101_folding_validated"],
            "stderr_tail": run["stderr_tail"][-2500:]}


@app.local_entrypoint()
def main(seeds: str = "3030,4040,5050", betas: str = "0.0,0.3,0.4,0.6",
         episodes_per_config: int = 10, rollout_steps: int = 240):
    res = eval_policy.remote(seeds, betas, episodes_per_config, rollout_steps)
    print(json.dumps({"V4_EVAL_VERDICT": res}, indent=2))
