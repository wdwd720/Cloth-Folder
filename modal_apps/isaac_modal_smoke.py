import json
import os
import subprocess
from pathlib import Path

import modal

app = modal.App("terafold-isaac-modal-smoke-v2")
volume = modal.Volume.from_name("terafold-artifacts")

image = (
    modal.Image.from_registry(
        "nvcr.io/nvidia/isaac-sim:6.0.1",
        add_python="3.11",
    )
    .dockerfile_commands("ENTRYPOINT []")
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
    timeout=1800,
)
def run():
    result = {
        "status": "TERAFOLD_ISAAC_MODAL_SMOKE_V2_STARTED",
        "experimental": True,
        "entrypoint_reset": True,
        "isaac_works_on_modal": False,
        "cwd": os.getcwd(),
        "env_accept_eula": os.environ.get("ACCEPT_EULA"),
    }

    inspect_cmd = ["bash", "-lc", "pwd; ls -lah /; find / -maxdepth 3 \\( -name python.sh -o -name isaac-sim.sh \\) 2>/dev/null | sort | head -50"]
    inspect = subprocess.run(inspect_cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    result["inspect_returncode"] = inspect.returncode
    result["inspect_stdout_tail"] = inspect.stdout[-8000:]
    result["inspect_stderr_tail"] = inspect.stderr[-4000:]

    candidates = [
        "/isaac-sim/python.sh",
        "/workspace/isaac-sim/python.sh",
        "/opt/isaac-sim/python.sh",
    ]
    existing = [p for p in candidates if Path(p).exists()]
    result["python_candidates_existing"] = existing

    if not existing:
        result["status"] = "TERAFOLD_ISAAC_MODAL_SMOKE_V2_NO_ISAAC_PYTHON"
        _write_result(result)
        return result

    smoke_py = Path("/tmp/isaac_smoke_inner.py")
    smoke_py.write_text(
        r'''
import json
import os
import sys
import traceback

out = {
    "python": sys.executable,
    "cwd": os.getcwd(),
    "simulation_app_import": None,
    "simulation_app_started": False,
    "simulation_app_update_ok": False,
    "error": None,
}

try:
    try:
        from isaacsim import SimulationApp
        out["simulation_app_import"] = "isaacsim.SimulationApp"
    except Exception:
        from omni.isaac.kit import SimulationApp
        out["simulation_app_import"] = "omni.isaac.kit.SimulationApp"

    app = SimulationApp({"headless": True})
    out["simulation_app_started"] = True
    app.update()
    out["simulation_app_update_ok"] = True
    app.close()
except Exception:
    out["error"] = traceback.format_exc()

print(json.dumps(out, indent=2))
sys.exit(0 if out["simulation_app_update_ok"] else 2)
'''
    )

    cmd = [existing[0], str(smoke_py)]
    result["command"] = cmd

    proc = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=os.environ.copy(),
        timeout=900,
    )

    result["returncode"] = proc.returncode
    result["stdout_tail"] = proc.stdout[-10000:]
    result["stderr_tail"] = proc.stderr[-10000:]
    result["isaac_works_on_modal"] = proc.returncode == 0
    result["status"] = "TERAFOLD_ISAAC_MODAL_SMOKE_V2_OK" if proc.returncode == 0 else "TERAFOLD_ISAAC_MODAL_SMOKE_V2_FAILED"

    _write_result(result)
    return result

def _write_result(result):
    out_dir = Path("/artifacts/isaac_modal_smoke")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "isaac_modal_smoke_v2_result.json"
    out_path.write_text(json.dumps(result, indent=2))
    volume.commit()
    print(json.dumps(result, indent=2))

@app.local_entrypoint()
def main():
    print(run.remote())
