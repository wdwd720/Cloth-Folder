"""Modal training pipeline v0 for TeraFold.

Runs a real PyTorch training smoke on a Modal L40S GPU using the repository's
own V4 compact-state policy building blocks (`build_training_dataset`,
`PolicyMLP`). No Isaac Sim is required: the dataset falls back to the
generated rectangular grid, so labels are explicitly SYNTHETIC and contain
NO robot-contact demonstrations.

This proves training infrastructure (GPU + volume + repo code path), NOT a
final robot-contact folding policy.

Result JSON: /artifacts/training_v0/training_v0_summary.json
Checkpoint : /artifacts/training_v0/clean_towel_policy_v0_smoke.pt  (volume only)
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import modal

app = modal.App("terafold-policy-training-v0")
volume = modal.Volume.from_name("terafold-artifacts")

image = (
    modal.Image.debian_slim()
    .pip_install("torch", "numpy")
    .add_local_file(
        "/tmp/terafold_modal_src.tar.gz",
        "/opt/terafold_modal_src.tar.gz",
        copy=True,
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
        "status": "TERAFOLD_MODAL_TRAINING_V0_STARTED",
        "synthetic_training_smoke": True,
        "not_robot_contact_training": True,
        "repo_unpacked": False,
        "cuda_available": None,
        "gpu_name": None,
        "dataset_source": None,
        "synthetic_or_real": "synthetic",
        "robot_contact_data_used": False,
        "training_steps": 0,
        "final_loss": None,
        "artifact_paths": [],
        "ready_for_real_training": False,
        "remaining_data_blockers": [],
    }

    out_dir = Path("/artifacts/training_v0")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "training_v0_summary.json"

    def _write():
        summary_path.write_text(json.dumps(result, indent=2))
        volume.commit()
        print(json.dumps(result, indent=2))

    # ---- torch / CUDA ----
    import numpy as np
    import torch

    result["cuda_available"] = bool(torch.cuda.is_available())
    result["gpu_name"] = (
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- unpack source ----
    work = Path("/tmp/terafold_repo")
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    unpack = subprocess.run(
        ["tar", "-xzf", "/opt/terafold_modal_src.tar.gz", "-C", str(work)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=300,
    )
    result["unpack_returncode"] = unpack.returncode
    result["unpack_stderr_tail"] = unpack.stderr[-2000:]
    if unpack.returncode != 0:
        result["status"] = "TERAFOLD_MODAL_TRAINING_V0_UNPACK_FAILED"
        _write()
        return result
    result["repo_unpacked"] = True

    tera_dir = work / "scripts" / "tera"
    sys.path.insert(0, str(tera_dir))

    try:
        from clean_towel_v4_common import (
            ACTION_NAMES,
            PolicyMLP,
            build_training_dataset,
        )
    except Exception as exc:  # noqa: BLE001
        import traceback

        result["status"] = "TERAFOLD_MODAL_TRAINING_V0_IMPORT_FAILED"
        result["import_error"] = repr(exc)
        result["import_traceback_tail"] = traceback.format_exc()[-4000:]
        _write()
        return result

    result["repo_import_ok"] = True

    # ---- build synthetic dataset (rect-grid fallback; no assisted data on Modal) ----
    dataset = build_training_dataset(fold_steps=80, settle_steps=20, augment_repeats=4)
    states = dataset["states"].astype(np.float32)
    actions = dataset["actions"].astype(np.float32)
    result["dataset_source"] = str(dataset["source_points"])
    result["robot_contact_data_used"] = bool(
        dataset.get("contains_robot_contact_demonstrations", False)
    )
    result["num_samples"] = int(states.shape[0])
    result["state_dim"] = int(states.shape[1])
    result["action_dim"] = int(actions.shape[1])

    if result["dataset_source"] != "generated_rect_grid_fallback":
        result["remaining_data_blockers"].append(
            f"dataset_source={result['dataset_source']} (unexpected on Modal; expected synthetic fallback)"
        )

    # ---- normalize ----
    state_mean = states.mean(axis=0)
    state_std = states.std(axis=0) + 1.0e-6
    action_mean = actions.mean(axis=0)
    action_std = actions.std(axis=0) + 1.0e-6
    states_n = (states - state_mean) / state_std
    actions_n = (actions - action_mean) / action_std

    torch.manual_seed(1234)
    np.random.seed(1234)
    x = torch.from_numpy(states_n).float().to(device)
    y = torch.from_numpy(actions_n).float().to(device)

    model = PolicyMLP(
        state_dim=states.shape[1], action_dim=actions.shape[1], hidden_dim=128
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3.0e-3, weight_decay=1.0e-4)
    loss_fn = torch.nn.MSELoss()

    steps = 300
    losses = []
    for _ in range(steps):
        pred = model(x)
        loss = loss_fn(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))

    result["training_steps"] = steps
    result["loss_first"] = float(losses[0])
    result["final_loss"] = float(losses[-1])

    with torch.no_grad():
        pred_actions = model(x).cpu().numpy() * action_std + action_mean
        result["final_action_mae"] = float(np.mean(np.abs(pred_actions - actions)))

    # ---- save small checkpoint to VOLUME ONLY (never committed to git) ----
    ckpt_path = out_dir / "clean_towel_policy_v0_smoke.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "state_dim": int(states.shape[1]),
            "action_dim": int(actions.shape[1]),
            "hidden_dim": 128,
            "state_mean": state_mean.astype(np.float32),
            "state_std": state_std.astype(np.float32),
            "action_mean": action_mean.astype(np.float32),
            "action_std": action_std.astype(np.float32),
            "action_names": list(ACTION_NAMES),
            "policy_basis": "modal_v0_synthetic_smoke_no_robot_contact",
            "contains_robot_contact_demonstrations": False,
            "synthetic_training_smoke": True,
        },
        ckpt_path,
    )
    result["artifact_paths"] = [str(summary_path), str(ckpt_path)]

    # ready-for-real-training = infra works; the honest data blocker remains
    result["ready_for_real_training"] = bool(
        result["cuda_available"]
        and result["repo_unpacked"]
        and result["final_loss"] is not None
        and result["final_loss"] < result["loss_first"]
    )
    result["remaining_data_blockers"].append(
        "No robot-contact demonstration dataset exists; V4 labels are synthetic. "
        "Real robot-contact policy training requires proven cloth/rigid contact transfer first."
    )
    result["status"] = "TERAFOLD_MODAL_TRAINING_V0_OK"
    _write()
    return result


@app.local_entrypoint()
def main():
    print(run.remote())
