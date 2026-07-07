"""Modal IsaacLab repo smoke v4 for TeraFold.

v3 proved: repo unpack, py_compile, IsaacLab clone + `isaaclab.app.AppLauncher`
import, and the clean-towel USD create/test scripts all ran. It FAILED only
because `torch` was missing inside the Isaac Sim Python (`/isaac-sim/python.sh`),
so `test_clean_towel_primitive_contact_v8.py` raised ModuleNotFoundError: torch.

v4 fix: install torch into the Isaac Sim Python during the image build, add a
torch import probe, then run the actual Tera scripts.

Result JSON: /artifacts/modal_repo_isaac_smoke/modal_repo_isaac_smoke_v4_result.json
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import modal

app = modal.App("terafold-modal-repo-isaac-smoke-v4")
volume = modal.Volume.from_name("terafold-artifacts")

ISAAC_PYTHON = "/isaac-sim/python.sh"

image = (
    modal.Image.from_registry(
        "nvcr.io/nvidia/isaac-sim:6.0.1",
        add_python="3.11",
    )
    .dockerfile_commands("ENTRYPOINT []")
    .apt_install("git", "tar")
    # Install torch into the Isaac Sim Python (this is the v3 blocker fix).
    # Cached as its own layer so we do not rebuild the giant Isaac base.
    .run_commands(
        f"{ISAAC_PYTHON} -m pip install --no-cache-dir 'torch' 'numpy<2'"
    )
    .add_local_file(
        "/tmp/terafold_modal_src.tar.gz",
        "/opt/terafold_modal_src.tar.gz",
        copy=True,
    )
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


@app.function(
    gpu="L40S",
    image=image,
    volumes={"/artifacts": volume},
    timeout=7200,
)
def run():
    result = {
        "status": "TERAFOLD_MODAL_REPO_ISAAC_SMOKE_V4_STARTED",
        "repo_unpacked": False,
        "python_compile_ok": False,
        "isaaclab_cloned": False,
        "isaaclab_import_ok": False,
        "torch_import_ok": False,
        "torch_version": None,
        "torch_cuda_available": None,
        "isaac_script_results": [],
        "ready_to_migrate_brev_to_modal": False,
        "ready_to_start_modal_training_pipeline": False,
        "remaining_blockers": [],
    }

    work = Path("/tmp/terafold_repo_smoke")
    lab = Path("/tmp/IsaacLab")

    if work.exists():
        shutil.rmtree(work)
    if lab.exists():
        shutil.rmtree(lab)
    work.mkdir(parents=True, exist_ok=True)

    unpack = subprocess.run(
        ["tar", "-xzf", "/opt/terafold_modal_src.tar.gz", "-C", str(work)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=300,
    )
    result["unpack_returncode"] = unpack.returncode
    result["unpack_stderr_tail"] = unpack.stderr[-4000:]
    if unpack.returncode != 0:
        result["status"] = "TERAFOLD_MODAL_REPO_UNPACK_FAILED"
        result["remaining_blockers"].append("source tar unpack failed")
        _write_result(result)
        return result
    result["repo_unpacked"] = True

    scripts_dir = work / "scripts" / "tera"
    script_files = sorted(scripts_dir.glob("*.py")) if scripts_dir.exists() else []
    result["tera_scripts_found_count"] = len(script_files)

    compile_proc = subprocess.run(
        ["python", "-m", "py_compile", *[str(p) for p in script_files]],
        cwd=str(work),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=300,
    )
    result["python_compile_returncode"] = compile_proc.returncode
    result["python_compile_stderr_tail"] = compile_proc.stderr[-4000:]
    result["python_compile_ok"] = compile_proc.returncode == 0

    clone = subprocess.run(
        ["git", "clone", "--depth", "1", "https://github.com/isaac-sim/IsaacLab.git", str(lab)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=900,
    )
    result["isaaclab_clone_returncode"] = clone.returncode
    result["isaaclab_clone_stderr_tail"] = clone.stderr[-4000:]
    if clone.returncode != 0:
        result["status"] = "TERAFOLD_MODAL_ISAACLAB_CLONE_FAILED"
        result["remaining_blockers"].append("IsaacLab git clone failed")
        _write_result(result)
        return result
    result["isaaclab_cloned"] = True

    source_root = lab / "source"
    source_paths = [str(p) for p in source_root.iterdir() if p.is_dir()]
    env = os.environ.copy()
    env["PYTHONPATH"] = ":".join(source_paths + [str(work), env.get("PYTHONPATH", "")])

    result["isaac_python_exists"] = Path(ISAAC_PYTHON).exists()
    result["isaaclab_source_paths"] = source_paths

    # ---- torch import probe inside Isaac Sim Python ----
    torch_probe = subprocess.run(
        [
            ISAAC_PYTHON,
            "-c",
            "import torch, json; print('TORCH_PROBE ' + json.dumps({'v': torch.__version__, 'cuda': torch.cuda.is_available()}))",
        ],
        cwd=str(work),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=600,
        env=env,
    )
    result["torch_probe_returncode"] = torch_probe.returncode
    result["torch_probe_stdout_tail"] = torch_probe.stdout[-2000:]
    result["torch_probe_stderr_tail"] = torch_probe.stderr[-4000:]
    result["torch_import_ok"] = torch_probe.returncode == 0
    for line in torch_probe.stdout.splitlines():
        if line.startswith("TORCH_PROBE "):
            try:
                info = json.loads(line[len("TORCH_PROBE ") :])
                result["torch_version"] = info.get("v")
                result["torch_cuda_available"] = info.get("cuda")
            except Exception:
                pass
    if not result["torch_import_ok"]:
        result["remaining_blockers"].append("torch import failed inside Isaac Sim Python")

    # ---- IsaacLab import probe ----
    probe = subprocess.run(
        [ISAAC_PYTHON, "-c", "from isaaclab.app import AppLauncher; print('ISAACLAB_IMPORT_OK')"],
        cwd=str(work),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=900,
        env=env,
    )
    result["isaaclab_import_returncode"] = probe.returncode
    result["isaaclab_import_stdout_tail"] = probe.stdout[-2000:]
    result["isaaclab_import_stderr_tail"] = probe.stderr[-8000:]
    result["isaaclab_import_ok"] = probe.returncode == 0
    if not result["isaaclab_import_ok"]:
        result["remaining_blockers"].append("IsaacLab AppLauncher import failed")
        result["status"] = "TERAFOLD_MODAL_ISAACLAB_IMPORT_FAILED"
        _write_result(result)
        return result

    candidate_scripts = [
        "scripts/tera/create_clean_rect_towel_usd_v2.py",
        "scripts/tera/test_clean_rect_towel_usd_v2.py",
        "scripts/tera/test_clean_towel_primitive_contact_v8.py",
    ]

    for rel in candidate_scripts:
        script_path = work / rel
        item = {
            "script": rel,
            "exists": script_path.exists(),
            "ran": False,
            "returncode": None,
            "stdout_tail": "",
            "stderr_tail": "",
        }
        if not script_path.exists():
            result["isaac_script_results"].append(item)
            continue

        proc = subprocess.run(
            [ISAAC_PYTHON, str(script_path)],
            cwd=str(work),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=1800,
            env=env,
        )
        item["ran"] = True
        item["returncode"] = proc.returncode
        item["stdout_tail"] = proc.stdout[-10000:]
        item["stderr_tail"] = proc.stderr[-10000:]
        result["isaac_script_results"].append(item)
        if proc.returncode != 0:
            result["remaining_blockers"].append(f"{rel} exited {proc.returncode}")
            break

    any_script_ran_ok = any(
        x["ran"] and x["returncode"] == 0 for x in result["isaac_script_results"]
    )
    all_existing_scripts_ok = all(
        x["ran"] and x["returncode"] == 0
        for x in result["isaac_script_results"]
        if x["exists"]
    )

    result["ready_to_migrate_brev_to_modal"] = bool(
        result["repo_unpacked"]
        and result["python_compile_ok"]
        and result["torch_import_ok"]
        and result["isaaclab_import_ok"]
        and any_script_ran_ok
        and all_existing_scripts_ok
    )
    result["ready_to_start_modal_training_pipeline"] = bool(
        result["repo_unpacked"]
        and result["python_compile_ok"]
        and result["isaaclab_import_ok"]
    )
    result["status"] = (
        "TERAFOLD_MODAL_REPO_ISAAC_SMOKE_V4_OK"
        if result["ready_to_migrate_brev_to_modal"]
        else "TERAFOLD_MODAL_REPO_ISAAC_SMOKE_V4_PARTIAL_OR_FAILED"
    )

    _write_result(result)
    return result


def _write_result(result):
    out_dir = Path("/artifacts/modal_repo_isaac_smoke")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "modal_repo_isaac_smoke_v4_result.json"
    out_path.write_text(json.dumps(result, indent=2))
    volume.commit()
    print(json.dumps(result, indent=2))


@app.local_entrypoint()
def main():
    print(run.remote())
