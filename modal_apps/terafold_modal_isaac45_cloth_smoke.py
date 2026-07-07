"""Modal IsaacLab cloth smoke on Isaac Sim 4.5.0 for TeraFold.

WHY 4.5: The v4 smoke on Isaac Sim 6.0.1 proved the Modal Isaac infra works
(SimulationApp, `isaaclab.app.AppLauncher`, torch, L40S GPU), BUT the actual
Tera clean-towel scripts use `isaacsim.core.prims.SingleClothPrim` /
`PhysxSchema.PhysxParticleClothAPI` -- the PhysX particle-based cloth API.
Isaac Sim 6.0.1 REMOVED that API (NotImplementedError: "SingleClothPrim is no
longer available. Omniverse PhysX removed the deprecated particle-based cloth
features."). Brev ran an older Isaac Sim (the `isaacsim.core.prims` namespace
is 4.5). So to run the real cloth/contact scripts on Modal we must match that
version: Isaac Sim 4.5.0 + IsaacLab v2.0.2.

Result JSON: /artifacts/modal_isaac45_cloth_smoke/modal_isaac45_cloth_smoke_v1_result.json
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import modal

app = modal.App("terafold-modal-isaac45-cloth-smoke-v1")
volume = modal.Volume.from_name("terafold-artifacts")

ISAAC_PYTHON = "/isaac-sim/python.sh"
ISAACLAB_BRANCH = "v2.0.2"  # compatible with Isaac Sim 4.5

image = (
    modal.Image.from_registry(
        "nvcr.io/nvidia/isaac-sim:4.5.0",
        add_python="3.10",
    )
    .dockerfile_commands("ENTRYPOINT []")
    .apt_install("git", "tar")
    # IsaacLab python deps that our deeper script imports pull in (torch is
    # bundled in Isaac Sim 4.5, so we only add the small missing pure-python ones).
    .run_commands(
        f"{ISAAC_PYTHON} -m pip install --no-cache-dir flatdict toml"
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
        "status": "TERAFOLD_MODAL_ISAAC45_CLOTH_SMOKE_STARTED",
        "isaac_sim_version_target": "4.5.0",
        "isaaclab_branch": ISAACLAB_BRANCH,
        "repo_unpacked": False,
        "isaaclab_cloned": False,
        "torch_import_ok": False,
        "torch_version": None,
        "isaaclab_import_ok": False,
        "particle_cloth_api_available": False,
        "cloth_probe_stdout_tail": "",
        "cloth_probe_stderr_tail": "",
        "isaac_script_results": [],
        "particle_cloth_runs_on_modal": False,
        "contact_metrics": {},
        "ready_to_migrate_brev_cloth_to_modal": False,
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
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300,
    )
    result["unpack_returncode"] = unpack.returncode
    if unpack.returncode != 0:
        result["status"] = "TERAFOLD_MODAL_ISAAC45_UNPACK_FAILED"
        result["remaining_blockers"].append("source tar unpack failed")
        _write_result(result)
        return result
    result["repo_unpacked"] = True

    clone = subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", ISAACLAB_BRANCH,
         "https://github.com/isaac-sim/IsaacLab.git", str(lab)],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=900,
    )
    result["isaaclab_clone_returncode"] = clone.returncode
    result["isaaclab_clone_stderr_tail"] = clone.stderr[-3000:]
    if clone.returncode != 0:
        result["status"] = "TERAFOLD_MODAL_ISAAC45_ISAACLAB_CLONE_FAILED"
        result["remaining_blockers"].append(f"IsaacLab {ISAACLAB_BRANCH} clone failed")
        _write_result(result)
        return result
    result["isaaclab_cloned"] = True

    source_root = lab / "source"
    source_paths = [str(p) for p in source_root.iterdir() if p.is_dir()] if source_root.exists() else []
    env = os.environ.copy()
    env["PYTHONPATH"] = ":".join(source_paths + [str(work), env.get("PYTHONPATH", "")])
    result["isaaclab_source_paths"] = source_paths
    result["isaac_python_exists"] = Path(ISAAC_PYTHON).exists()

    # ---- torch probe ----
    tp = subprocess.run(
        [ISAAC_PYTHON, "-c",
         "import torch,json;print('TORCH '+json.dumps({'v':torch.__version__,'cuda':torch.cuda.is_available()}))"],
        cwd=str(work), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600, env=env,
    )
    result["torch_import_ok"] = tp.returncode == 0
    result["torch_probe_stderr_tail"] = tp.stderr[-2000:]
    for line in tp.stdout.splitlines():
        if line.startswith("TORCH "):
            try:
                result["torch_version"] = json.loads(line[6:]).get("v")
            except Exception:
                pass

    # ---- isaaclab import probe ----
    ip = subprocess.run(
        [ISAAC_PYTHON, "-c", "from isaaclab.app import AppLauncher;print('ISAACLAB_IMPORT_OK')"],
        cwd=str(work), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=900, env=env,
    )
    result["isaaclab_import_ok"] = ip.returncode == 0
    result["isaaclab_import_stderr_tail"] = ip.stderr[-6000:]
    if not result["isaaclab_import_ok"]:
        result["remaining_blockers"].append("IsaacLab AppLauncher import failed on 4.5")

    # ---- particle-cloth API availability probe (the crux) ----
    cloth_probe = subprocess.run(
        [ISAAC_PYTHON, "-c",
         (
             "from isaaclab.app import AppLauncher\n"
             "app=AppLauncher(headless=True).app\n"
             "ok=False;err=None\n"
             "try:\n"
             "    from isaacsim.core.prims import SingleClothPrim, SingleParticleSystem\n"
             "    ok=True\n"
             "except Exception as e:\n"
             "    err=repr(e)\n"
             "print('CLOTH_API '+('OK' if ok else 'MISSING'))\n"
             "print('CLOTH_ERR '+str(err))\n"
             "app.close()\n"
         )],
        cwd=str(work), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=1200, env=env,
    )
    result["cloth_probe_returncode"] = cloth_probe.returncode
    result["cloth_probe_stdout_tail"] = cloth_probe.stdout[-4000:]
    result["cloth_probe_stderr_tail"] = cloth_probe.stderr[-6000:]
    # NOTE: this standalone import probe is informational only and can
    # false-negative (import context differs from the real scripts). The
    # authoritative signal for "cloth runs on Modal" is whether
    # test_clean_rect_towel_usd_v2.py reports TEST_OK below.
    result["particle_cloth_api_available"] = "CLOTH_API OK" in cloth_probe.stdout

    # ---- run the actual cloth scripts (+ V9 gravity sanity probe) ----
    candidate_scripts = [
        "scripts/tera/create_clean_rect_towel_usd_v2.py",
        "scripts/tera/test_clean_rect_towel_usd_v2.py",
        "scripts/tera/test_clean_towel_primitive_contact_v8.py",
        "scripts/tera/inspect_clean_towel_particle_sim_v9.py",
    ]
    for rel in candidate_scripts:
        script_path = work / rel
        item = {"script": rel, "exists": script_path.exists(), "ran": False,
                "returncode": None, "internal_status": None,
                "stdout_tail": "", "stderr_tail": ""}
        if not script_path.exists():
            result["isaac_script_results"].append(item)
            continue
        proc = subprocess.run(
            [ISAAC_PYTHON, str(script_path)], cwd=str(work), text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=1800, env=env,
        )
        item["ran"] = True
        item["returncode"] = proc.returncode
        item["stdout_tail"] = proc.stdout[-12000:]
        item["stderr_tail"] = proc.stderr[-8000:]
        item["internal_status"] = _extract_status(proc.stdout)
        result["isaac_script_results"].append(item)
        # stop after the first script whose *internal* status shows failure
        if proc.returncode != 0 or (item["internal_status"] and "FAILED" in str(item["internal_status"].get("status", ""))):
            break

    # capture v8 contact metrics if present
    for s in result["isaac_script_results"]:
        if "v8" in s["script"] and s["internal_status"]:
            st = s["internal_status"]
            result["contact_metrics"] = {
                k: st.get(k) for k in (
                    "status", "actual_touch_achieved", "primitive_moves_towel",
                    "primitive_width_reduction_success", "selected_edge_particle_displacement",
                    "nearest_collider_to_towel_particle_m", "start_width", "best_width", "final_width",
                ) if k in st
            }

    # capture V9 gravity/root-cause result if present
    for s in result["isaac_script_results"]:
        if "v9" in s["script"] and s["internal_status"]:
            st = s["internal_status"]
            result["v9_particle_sim"] = {
                k: st.get(k) for k in (
                    "status", "num_particles", "cloth_simulates_under_gravity",
                    "final_max_particle_disp_m", "mean_z_drop_m", "root_cause_hypothesis",
                    "inverse_mass_probe", "gravity_trace",
                ) if k in st
            }

    def _script_ok(name):
        for s in result["isaac_script_results"]:
            if name in s["script"]:
                st = s.get("internal_status") or {}
                return bool(s["ran"] and s["returncode"] == 0 and "FAILED" not in str(st.get("status", "")))
        return False

    # authoritative: cloth runs on Modal iff the real cloth SIM test reports OK
    result["particle_cloth_runs_on_modal"] = bool(_script_ok("test_clean_rect_towel_usd_v2"))
    if not result["particle_cloth_runs_on_modal"]:
        result["remaining_blockers"].append("test_clean_rect_towel_usd_v2 did not report TEST_OK on 4.5")
    result["ready_to_migrate_brev_cloth_to_modal"] = bool(
        result["repo_unpacked"] and result["isaaclab_import_ok"]
        and result["particle_cloth_runs_on_modal"]
    )
    result["status"] = (
        "TERAFOLD_MODAL_ISAAC45_CLOTH_SMOKE_OK"
        if result["particle_cloth_runs_on_modal"]
        else "TERAFOLD_MODAL_ISAAC45_CLOTH_SMOKE_PARTIAL_OR_FAILED"
    )
    _write_result(result)
    return result


def _extract_status(stdout: str):
    import json as _json
    import re
    found = None
    for m in re.finditer(r"\{", stdout):
        frag = stdout[m.start():]
        try:
            obj, _ = _json.JSONDecoder().raw_decode(frag)
            if isinstance(obj, dict) and "status" in obj:
                found = obj
        except Exception:
            continue
    return found


def _write_result(result):
    out_dir = Path("/artifacts/modal_isaac45_cloth_smoke")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "modal_isaac45_cloth_smoke_v1_result.json"
    out_path.write_text(json.dumps(result, indent=2))
    volume.commit()
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("isaac_script_results",)}, indent=2))


@app.local_entrypoint()
def main():
    print(run.remote())
