"""Modal runner for the V10 corrected-mass clean-towel contact test (Isaac 4.5.0).

Runs scripts/tera/test_clean_towel_contact_corrected_mass_v10.py, which re-authors
the towel with a realistic total mass (fixing the per-particle density bug) + soft
springs, then runs the unchanged V8 contact trial and reports whether the collider
now moves the towel.

Result JSON: /artifacts/modal_isaac45_contact_v10/modal_isaac45_contact_v10_result.json
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import modal

app = modal.App("terafold-modal-isaac45-contact-v10")
volume = modal.Volume.from_name("terafold-artifacts")

ISAAC_PYTHON = "/isaac-sim/python.sh"
ISAACLAB_BRANCH = "v2.0.2"

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


@app.function(gpu="L40S", image=image, volumes={"/artifacts": volume}, timeout=7200)
def run():
    result = {
        "status": "TERAFOLD_MODAL_ISAAC45_CONTACT_V10_STARTED",
        "isaac_sim_version_target": "4.5.0",
        "repo_unpacked": False,
        "isaaclab_cloned": False,
        "v10_ran": False,
        "v10_returncode": None,
        "v10_summary": None,
        "corrected_mass_enables_contact_transfer": None,
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
        result["status"] = "TERAFOLD_MODAL_ISAAC45_CONTACT_V10_UNPACK_FAILED"
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
        result["status"] = "TERAFOLD_MODAL_ISAAC45_CONTACT_V10_CLONE_FAILED"
        result["clone_stderr_tail"] = clone.stderr[-3000:]
        _write(result)
        return result
    result["isaaclab_cloned"] = True

    source_root = lab / "source"
    source_paths = [str(p) for p in source_root.iterdir() if p.is_dir()] if source_root.exists() else []
    env = os.environ.copy()
    env["PYTHONPATH"] = ":".join(source_paths + [str(work), env.get("PYTHONPATH", "")])

    script = work / "scripts" / "tera" / "test_clean_towel_contact_corrected_mass_v10.py"
    proc = subprocess.run(
        [ISAAC_PYTHON, str(script)], cwd=str(work), text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=3000, env=env,
    )
    result["v10_ran"] = True
    result["v10_returncode"] = proc.returncode
    result["v10_stdout_tail"] = proc.stdout[-14000:]
    result["v10_stderr_tail"] = proc.stderr[-8000:]

    # V10 writes its own JSON under /workspace/leisaac/tera_checkpoints/clean_towel_v10/
    v10_json = Path("/workspace/leisaac/tera_checkpoints/clean_towel_v10/contact_corrected_mass_v10.json")
    if v10_json.exists():
        try:
            summ = json.loads(v10_json.read_text())
            result["v10_summary"] = summ
            result["corrected_mass_enables_contact_transfer"] = summ.get(
                "corrected_mass_enables_contact_transfer"
            )
        except Exception as exc:  # noqa: BLE001
            result["v10_summary_parse_error"] = repr(exc)
    else:
        result["remaining_blockers"].append("V10 did not write its summary json")

    result["status"] = "TERAFOLD_MODAL_ISAAC45_CONTACT_V10_DONE"
    _write(result)
    return result


def _write(result):
    out_dir = Path("/artifacts/modal_isaac45_contact_v10")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "modal_isaac45_contact_v10_result.json").write_text(json.dumps(result, indent=2))
    volume.commit()
    print(json.dumps({k: v for k, v in result.items() if k != "v10_stdout_tail"}, indent=2))


@app.local_entrypoint()
def main():
    print(run.remote())
