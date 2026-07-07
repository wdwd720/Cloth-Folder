import json
import os
from pathlib import Path

import modal

app = modal.App("terafold-training-smoke")

volume = modal.Volume.from_name("terafold-artifacts")

image = (
    modal.Image.debian_slim()
    .pip_install("torch", "numpy")
)

@app.function(
    gpu="L40S",
    image=image,
    volumes={"/artifacts": volume},
    timeout=600,
)
def run():
    import torch
    import numpy as np

    result = {
        "status": "TERAFOLD_MODAL_TRAINING_SMOKE_OK",
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }

    x = torch.randn((2048, 2048), device="cuda")
    y = x @ x
    result["matmul_mean"] = float(y.mean().detach().cpu())

    out_dir = Path("/artifacts/modal_smoke")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "training_smoke_result.json"
    out_path.write_text(json.dumps(result, indent=2))

    volume.commit()

    print(json.dumps(result, indent=2))
    return result

@app.local_entrypoint()
def main():
    print(run.remote())
