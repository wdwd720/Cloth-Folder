"""Modal Training v2 (robot-contact) — GATED on valid V12 SO-101 demos.

Reads the V12 demo manifest from the Modal Volume. Trains a policy ONLY if the
demos carry REAL robot-joint-action contact labels
(`robot_contact_data_ready = true`). Otherwise it honestly SKIPS and records why
— it never fabricates robot-contact data.

Result JSON: /artifacts/training_v2_robot_contact/training_v2_robot_contact_summary.json
"""

import json
from pathlib import Path

import modal

app = modal.App("terafold-policy-training-v2-robot-contact")
volume = modal.Volume.from_name("terafold-artifacts")

image = modal.Image.debian_slim().pip_install("torch", "numpy")

MANIFEST_PATH = "/artifacts/contact_v12/demos/so101_contact_demos_manifest_v12.json"
TRAIN_STEPS = 3000


@app.function(gpu="L40S", image=image, volumes={"/artifacts": volume}, timeout=3600)
def run():
    import numpy as np
    import torch

    result = {
        "status": "TERAFOLD_MODAL_TRAINING_V2_STARTED",
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "robot_contact_data_used": False,
        "dataset_path": None,
        "num_episodes": None,
        "train_steps": 0,
        "initial_loss": None,
        "final_loss": None,
        "action_mae": None,
        "validation_metrics": None,
        "checkpoint_path": None,
        "final_robot_contact_policy_training_ready": False,
        "remaining_blockers": [],
    }
    out_dir = Path("/artifacts/training_v2_robot_contact")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "training_v2_robot_contact_summary.json"

    def _write():
        summary_path.write_text(json.dumps(result, indent=2))
        volume.commit()
        print(json.dumps({k: v for k, v in result.items() if k != "loss_curve"}, indent=2))

    manifest_path = Path(MANIFEST_PATH)
    manifest = None
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except Exception as exc:  # noqa: BLE001
            result["remaining_blockers"].append(f"manifest parse error: {exc!r}")

    # ---- GATE: only train on REAL robot-contact data ----
    if not manifest or not manifest.get("robot_contact_data_ready"):
        result["status"] = "TERAFOLD_MODAL_TRAINING_V2_SKIPPED"
        result["training_v2_skipped_reason"] = (
            "V12 robot-contact demos not ready"
            if manifest is None
            else manifest.get("remaining_blocker")
            or "V12 demos exist but robot_contact_data_ready=false (synthetic joint labels; "
               "true SO-101 joint-action contact demos require V6 reach + articulation fold)"
        )
        result["robot_contact_data_used"] = False
        result["num_episodes"] = None if manifest is None else manifest.get("num_episodes")
        result["dataset_path"] = None if manifest is None else manifest.get("dataset_path")
        result["remaining_blockers"].append(result["training_v2_skipped_reason"])
        _write()
        return result

    # ---- train on real robot-contact demos ----
    npz_path = Path(manifest.get("dataset_path", ""))
    if not npz_path.exists():
        result["status"] = "TERAFOLD_MODAL_TRAINING_V2_SKIPPED"
        result["training_v2_skipped_reason"] = f"dataset npz missing: {npz_path}"
        result["remaining_blockers"].append(result["training_v2_skipped_reason"])
        _write()
        return result

    data = np.load(npz_path, allow_pickle=True)
    states = data["states"].astype(np.float32)
    actions = data["actions"].astype(np.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    result["robot_contact_data_used"] = True
    result["dataset_path"] = str(npz_path)
    result["num_episodes"] = int(manifest.get("num_episodes", 0))

    s_mean, s_std = states.mean(0), states.std(0) + 1e-6
    a_mean, a_std = actions.mean(0), actions.std(0) + 1e-6
    rng = np.random.default_rng(1234)
    perm = rng.permutation(states.shape[0])
    n_val = max(1, int(0.15 * states.shape[0]))
    vi, ti = perm[:n_val], perm[n_val:]
    xt = torch.from_numpy(((states - s_mean) / s_std)[ti]).float().to(device)
    yt = torch.from_numpy(((actions - a_mean) / a_std)[ti]).float().to(device)
    xv = torch.from_numpy(((states - s_mean) / s_std)[vi]).float().to(device)
    yv = torch.from_numpy(((actions - a_mean) / a_std)[vi]).float().to(device)

    torch.manual_seed(1234)
    model = torch.nn.Sequential(
        torch.nn.Linear(states.shape[1], 256), torch.nn.ReLU(),
        torch.nn.Linear(256, 256), torch.nn.ReLU(),
        torch.nn.Linear(256, actions.shape[1]),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    loss_fn = torch.nn.MSELoss()
    for step in range(1, TRAIN_STEPS + 1):
        model.train()
        loss = loss_fn(model(xt), yt)
        opt.zero_grad(); loss.backward(); opt.step()
        if step == 1:
            result["initial_loss"] = float(loss.detach().cpu())
    result["train_steps"] = TRAIN_STEPS
    with torch.no_grad():
        result["final_loss"] = float(loss_fn(model(xt), yt).detach().cpu())
        result["validation_metrics"] = {"final_val_loss": float(loss_fn(model(xv), yv).detach().cpu())}
        pred = model(torch.from_numpy((states - s_mean) / s_std).float().to(device)).cpu().numpy() * a_std + a_mean
        result["action_mae"] = float(np.mean(np.abs(pred - actions)))

    ckpt = out_dir / "policy_v2_robot_contact.pt"
    torch.save({"model_state_dict": model.state_dict(), "robot_contact_data_used": True}, ckpt)
    result["checkpoint_path"] = str(ckpt)
    result["final_robot_contact_policy_training_ready"] = True
    result["status"] = "TERAFOLD_MODAL_TRAINING_V2_OK"
    _write()
    return result


@app.local_entrypoint()
def main():
    print(run.remote())
