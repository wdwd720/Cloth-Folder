"""Modal training pipeline v1 for TeraFold (longer, bounded, honest).

Extends training_v0: same repo V4 code path (`build_training_dataset` +
`PolicyMLP`) on a Modal L40S, but trains longer (5000 steps) with a held-out
validation split and periodic checkpoints. Data is still SYNTHETIC (generated
rect-grid fallback; no robot-contact demonstrations), so this remains training
INFRASTRUCTURE, not a final robot-contact policy.

Result JSON: /artifacts/training_v1/training_v1_summary.json
Checkpoints: /artifacts/training_v1/checkpoints/*.pt          (volume only)
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import modal

app = modal.App("terafold-policy-training-v1")
volume = modal.Volume.from_name("terafold-artifacts")

image = (
    modal.Image.debian_slim()
    .pip_install("torch", "numpy")
    .add_local_file("/tmp/terafold_modal_src.tar.gz", "/opt/terafold_modal_src.tar.gz", copy=True)
)

TRAIN_STEPS = 5000
CKPT_EVERY = 1000


@app.function(gpu="L40S", image=image, volumes={"/artifacts": volume}, timeout=3600)
def run():
    result = {
        "status": "TERAFOLD_MODAL_TRAINING_V1_STARTED",
        "synthetic_or_assisted": True,
        "robot_contact_data_used": False,
        "final_robot_contact_policy": False,
        "final_robot_contact_policy_training_ready": False,
        "cuda_available": None,
        "gpu_name": None,
        "dataset_source": None,
        "train_steps": 0,
        "initial_loss": None,
        "final_loss": None,
        "action_mae": None,
        "checkpoint_path": None,
        "checkpoint_paths": [],
        "ready_for_real_training": False,
        "remaining_data_blockers": [],
    }

    out_dir = Path("/artifacts/training_v1")
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "training_v1_summary.json"

    def _write():
        summary_path.write_text(json.dumps(result, indent=2))
        volume.commit()
        print(json.dumps({k: v for k, v in result.items() if k != "loss_curve"}, indent=2))

    import numpy as np
    import torch

    result["cuda_available"] = bool(torch.cuda.is_available())
    result["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    device = "cuda" if torch.cuda.is_available() else "cpu"

    work = Path("/tmp/terafold_repo")
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    unpack = subprocess.run(
        ["tar", "-xzf", "/opt/terafold_modal_src.tar.gz", "-C", str(work)],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300,
    )
    if unpack.returncode != 0:
        result["status"] = "TERAFOLD_MODAL_TRAINING_V1_UNPACK_FAILED"
        result["unpack_stderr_tail"] = unpack.stderr[-2000:]
        _write()
        return result
    sys.path.insert(0, str(work / "scripts" / "tera"))

    try:
        from clean_towel_v4_common import ACTION_NAMES, PolicyMLP, build_training_dataset
    except Exception as exc:  # noqa: BLE001
        import traceback
        result["status"] = "TERAFOLD_MODAL_TRAINING_V1_IMPORT_FAILED"
        result["import_error"] = repr(exc)
        result["import_traceback_tail"] = traceback.format_exc()[-3000:]
        _write()
        return result

    dataset = build_training_dataset(fold_steps=80, settle_steps=20, augment_repeats=8)
    states = dataset["states"].astype(np.float32)
    actions = dataset["actions"].astype(np.float32)
    result["dataset_source"] = str(dataset["source_points"])
    result["robot_contact_data_used"] = bool(dataset.get("contains_robot_contact_demonstrations", False))
    result["num_samples"] = int(states.shape[0])

    # normalize + train/val split
    rng = np.random.default_rng(1234)
    perm = rng.permutation(states.shape[0])
    n_val = max(1, int(0.15 * states.shape[0]))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    s_mean, s_std = states.mean(0), states.std(0) + 1e-6
    a_mean, a_std = actions.mean(0), actions.std(0) + 1e-6
    sn = (states - s_mean) / s_std
    an = (actions - a_mean) / a_std
    xt = torch.from_numpy(sn[train_idx]).float().to(device)
    yt = torch.from_numpy(an[train_idx]).float().to(device)
    xv = torch.from_numpy(sn[val_idx]).float().to(device)
    yv = torch.from_numpy(an[val_idx]).float().to(device)

    torch.manual_seed(1234)
    model = PolicyMLP(state_dim=states.shape[1], action_dim=actions.shape[1], hidden_dim=256).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=TRAIN_STEPS)
    loss_fn = torch.nn.MSELoss()

    loss_curve = []
    for step in range(1, TRAIN_STEPS + 1):
        model.train()
        pred = model(xt)
        loss = loss_fn(pred, yt)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step == 1:
            result["initial_loss"] = float(loss.detach().cpu())
        if step % 100 == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                vl = float(loss_fn(model(xv), yv).detach().cpu())
            loss_curve.append({"step": step, "train_loss": float(loss.detach().cpu()), "val_loss": vl})
        if step % CKPT_EVERY == 0:
            cp = ckpt_dir / f"clean_towel_policy_v1_step{step}.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "state_dim": int(states.shape[1]), "action_dim": int(actions.shape[1]),
                    "hidden_dim": 256, "step": step,
                    "state_mean": s_mean.astype(np.float32), "state_std": s_std.astype(np.float32),
                    "action_mean": a_mean.astype(np.float32), "action_std": a_std.astype(np.float32),
                    "action_names": list(ACTION_NAMES),
                    "policy_basis": "modal_v1_synthetic_longer_no_robot_contact",
                    "contains_robot_contact_demonstrations": False,
                    "synthetic_or_assisted": True,
                },
                cp,
            )
            result["checkpoint_paths"].append(str(cp))
            volume.commit()

    result["train_steps"] = TRAIN_STEPS
    result["loss_curve"] = loss_curve
    with torch.no_grad():
        result["final_loss"] = float(loss_fn(model(xt), yt).detach().cpu())
        result["final_val_loss"] = float(loss_fn(model(xv), yv).detach().cpu())
        pred_actions = model(torch.from_numpy(sn).float().to(device)).cpu().numpy() * a_std + a_mean
        result["action_mae"] = float(np.mean(np.abs(pred_actions - actions)))

    result["checkpoint_path"] = result["checkpoint_paths"][-1] if result["checkpoint_paths"] else None
    result["ready_for_real_training"] = bool(
        result["cuda_available"] and result["final_loss"] is not None
        and result["final_loss"] < result["initial_loss"]
    )
    # honest: infra ready, but final robot-contact policy is NOT authorized
    result["final_robot_contact_policy_training_ready"] = False
    result["remaining_data_blockers"].append(
        "Synthetic rect-grid labels only; no robot-contact demonstrations. "
        "Final robot-contact policy training requires proven cloth/rigid contact transfer (see contact_v11)."
    )
    result["status"] = "TERAFOLD_MODAL_TRAINING_V1_OK"
    _write()
    return result


@app.local_entrypoint()
def main():
    print(run.remote())
