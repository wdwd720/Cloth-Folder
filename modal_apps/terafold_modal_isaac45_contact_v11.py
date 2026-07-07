"""Modal runner for the V11 contact-drive research suite (Isaac Sim 4.5.0).

Runs, in one L40S container:
  1. create_clean_rect_towel_usd_v2.py            (author the default towel USD)
  2. test_clean_towel_kinematic_velocity_contact_v11.py
  3. test_clean_towel_dynamic_rigid_contact_v11.py
  4. inspect_clean_towel_particle_mass_v11.py
  5. summarize_contact_readiness_v11.py           (aggregate -> contact_v11_summary.json)

Persists the aggregate to:
  /artifacts/contact_v11/contact_v11_summary.json
"""

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import modal

app = modal.App("terafold-modal-isaac45-contact-v11")
volume = modal.Volume.from_name("terafold-artifacts")

ISAAC_PYTHON = "/isaac-sim/python.sh"
ISAACLAB_BRANCH = "v2.0.2"
V11_DIR = "/workspace/leisaac/tera_checkpoints/clean_towel_v11"

image = (
    modal.Image.from_registry("nvcr.io/nvidia/isaac-sim:4.5.0", add_python="3.10")
    .dockerfile_commands("ENTRYPOINT []")
    .apt_install("git", "tar")
    .run_commands(f"{ISAAC_PYTHON} -m pip install --no-cache-dir flatdict toml")
    .add_local_file("/tmp/terafold_modal_src.tar.gz", "/opt/terafold_modal_src.tar.gz", copy=True)
    .env(
        {
            "ACCEPT_EULA": "Y",
            "PRIVACY_CONSENT": "Y",
            "OMNI_KIT_ACCEPT_EULA": "YES",
            "NVIDIA_VISIBLE_DEVICES": "all",
            "NVIDIA_DRIVER_CAPABILITIES": "all",
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
def run():
    result = {
        "status": "TERAFOLD_MODAL_ISAAC45_CONTACT_V11_STARTED",
        "isaac_sim_version_target": "4.5.0",
        "repo_unpacked": False,
        "isaaclab_cloned": False,
        "script_runs": [],
        "contact_v11_summary": None,
        "contact_transfer_solved": None,
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
        result["status"] = "TERAFOLD_MODAL_ISAAC45_CONTACT_V11_UNPACK_FAILED"
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
        result["status"] = "TERAFOLD_MODAL_ISAAC45_CONTACT_V11_CLONE_FAILED"
        result["clone_stderr_tail"] = clone.stderr[-3000:]
        _write(result)
        return result
    result["isaaclab_cloned"] = True

    source_root = lab / "source"
    source_paths = [str(p) for p in source_root.iterdir() if p.is_dir()] if source_root.exists() else []
    env = os.environ.copy()
    env["PYTHONPATH"] = ":".join(source_paths + [str(work), env.get("PYTHONPATH", "")])

    scripts = [
        "scripts/tera/create_clean_rect_towel_usd_v2.py",
        "scripts/tera/test_clean_towel_kinematic_velocity_contact_v11.py",
        "scripts/tera/test_clean_towel_dynamic_rigid_contact_v11.py",
        "scripts/tera/inspect_clean_towel_particle_mass_v11.py",
        "scripts/tera/summarize_contact_readiness_v11.py",
    ]
    for rel in scripts:
        proc = subprocess.run(
            [ISAAC_PYTHON, str(work / rel)], cwd=str(work), text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=2400, env=env,
        )
        st = _extract_status(proc.stdout)
        result["script_runs"].append({
            "script": rel, "returncode": proc.returncode,
            "internal_status": (st or {}).get("status"),
            "stdout_tail": proc.stdout[-9000:], "stderr_tail": proc.stderr[-6000:],
        })

    summ_path = Path(V11_DIR) / "contact_v11_summary.json"
    if summ_path.exists():
        try:
            summ = json.loads(summ_path.read_text())
            result["contact_v11_summary"] = summ
            result["contact_transfer_solved"] = summ.get("contact_transfer_solved")
            result["final_robot_contact_policy_training_ready"] = summ.get(
                "final_robot_contact_policy_training_ready"
            )
        except Exception as exc:  # noqa: BLE001
            result["remaining_blockers"].append(f"summary parse error: {exc!r}")
    else:
        result["remaining_blockers"].append("contact_v11_summary.json not written")

    result["status"] = "TERAFOLD_MODAL_ISAAC45_CONTACT_V11_DONE"
    _write(result)
    return result


def _write(result):
    out_dir = Path("/artifacts/contact_v11")
    out_dir.mkdir(parents=True, exist_ok=True)
    # persist both the aggregate summary and this run wrapper
    if result.get("contact_v11_summary") is not None:
        (out_dir / "contact_v11_summary.json").write_text(json.dumps(result["contact_v11_summary"], indent=2))
    (out_dir / "modal_contact_v11_run.json").write_text(json.dumps(result, indent=2))
    volume.commit()
    trimmed = {k: v for k, v in result.items() if k != "script_runs"}
    trimmed["script_runs_brief"] = [
        {"script": s["script"], "returncode": s["returncode"], "internal_status": s["internal_status"]}
        for s in result.get("script_runs", [])
    ]
    print(json.dumps(trimmed, indent=2))


@app.local_entrypoint()
def main():
    print(run.remote())
