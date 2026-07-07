"""Modal Training v2 (robot-contact) — GATED on valid REAL SO-101 arm-driven demos.

Reads the demo manifest from the Modal Volume (default: the V14 real-articulation
demos). Trains a behavior-cloning policy ONLY if the demos carry REAL
robot-joint-action contact labels (`robot_contact_data_ready = true`). Otherwise
it honestly SKIPS and records why — it NEVER fabricates robot-contact data and
never calls synthetic/proxy demos real robot-contact training.

Result JSON: /artifacts/training_v2_robot_contact/training_v2_robot_contact_summary.json
"""

import json
from pathlib import Path

import modal

app = modal.App("terafold-policy-training-v2-robot-contact")
volume = modal.Volume.from_name("terafold-artifacts")

image = modal.Image.debian_slim().pip_install("torch", "numpy")

# Default: V14 real-articulation arm-driven fold demos. A V13 manifest is a valid
# fallback with the same schema (real commanded joint targets as action labels).
DEFAULT_MANIFEST = "/artifacts/contact_v14/demos/so101_real_contact_demos_manifest_v14.json"
FALLBACK_MANIFESTS = [
    "/artifacts/contact_v13/demos/so101_real_contact_demos_manifest_v13.json",
]
TRAIN_STEPS = 3000


@app.function(gpu="L40S", image=image, volumes={"/artifacts": volume}, timeout=3600)
def run(manifest_path: str = DEFAULT_MANIFEST):
    import numpy as np
    import torch

    result = {
        "status": "TERAFOLD_MODAL_TRAINING_V2_STARTED",
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "manifest_path": manifest_path,
        "robot_contact_data_used": False,
        "dataset_path": None,
        "num_episodes": None,
        "train_steps": 0,
        "initial_loss": None,
        "final_loss": None,
        "action_mae": None,
        "validation_loss": None,
        "validation_action_mae": None,
        "checkpoint_path": None,
        "policy_architecture": None,
        "observation_schema": None,
        "action_schema": None,
        "ready_for_real_training": False,
        "final_robot_contact_policy_training_ready": False,
        "remaining_blockers": [],
        "exact_next_step": None,
    }
    out_dir = Path("/artifacts/training_v2_robot_contact")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "training_v2_robot_contact_summary.json"

    def _write():
        summary_path.write_text(json.dumps(result, indent=2))
        volume.commit()
        print(json.dumps({k: v for k, v in result.items() if k != "loss_curve"}, indent=2))

    # find a manifest (default V14; else fallbacks)
    candidates = [manifest_path] + [m for m in FALLBACK_MANIFESTS if m != manifest_path]
    manifest, chosen = None, None
    for cand in candidates:
        p = Path(cand)
        if p.exists():
            try:
                manifest = json.loads(p.read_text())
                chosen = cand
                break
            except Exception as exc:  # noqa: BLE001
                result["remaining_blockers"].append(f"manifest parse error ({cand}): {exc!r}")
    result["manifest_path"] = chosen or manifest_path

    # ---- GATE: only train on REAL robot-contact data ----
    if not manifest or not manifest.get("robot_contact_data_ready"):
        result["status"] = "TERAFOLD_MODAL_TRAINING_V2_SKIPPED"
        result["training_v2_skipped_reason"] = "real SO-101 arm-driven fold demos not ready"
        result["robot_contact_data_used"] = False
        result["num_episodes"] = None if manifest is None else manifest.get("num_episodes")
        result["dataset_path"] = None if manifest is None else manifest.get("dataset_path")
        result["final_robot_contact_policy_training_ready"] = False
        result["remaining_blockers"].append(result["training_v2_skipped_reason"])
        result["exact_next_step"] = (
            "Produce a VALID real SO-101 arm-driven fold (V14) and collect real (state, joint_action, "
            "contact) demos with robot_contact_data_ready=true, then re-run Training v2."
        )
        _write()
        return result

    # ---- train on real robot-contact demos ----
    # The manifest's dataset_path is CONTAINER-LOCAL (the demo-collection container's
    # scratch dir); here only the VOLUME copy exists, in the SAME dir as the manifest.
    raw_path = Path(manifest.get("dataset_path", ""))
    manifest_dir = Path(result["manifest_path"]).parent
    npz_path = None
    for cand in (manifest_dir / raw_path.name, raw_path):
        if cand.exists():
            npz_path = cand
            break
    if npz_path is None:
        result["status"] = "TERAFOLD_MODAL_TRAINING_V2_SKIPPED"
        result["training_v2_skipped_reason"] = f"dataset npz missing: tried {manifest_dir / raw_path.name} and {raw_path}"
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
    result["observation_schema"] = {"dim": int(states.shape[1]), "source": "compact_state(cloth metrics + real SO-101 jaw positions)"}
    result["action_schema"] = {"dim": int(actions.shape[1]), "source": "real_commanded_so101_joint_position_targets", "names": "left[6]+right[6]"}

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
    result["policy_architecture"] = "MLP[in->256->256->act], ReLU, AdamW(lr=2e-3, wd=1e-4), MSE behavior cloning"
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    loss_fn = torch.nn.MSELoss()
    for step in range(1, TRAIN_STEPS + 1):
        model.train()
        loss = loss_fn(model(xt), yt)
        opt.zero_grad(); loss.backward(); opt.step()
        if step == 1:
            result["initial_loss"] = float(loss.detach().cpu())
    result["train_steps"] = TRAIN_STEPS
    model.eval()
    with torch.no_grad():
        result["final_loss"] = float(loss_fn(model(xt), yt).detach().cpu())
        result["validation_loss"] = float(loss_fn(model(xv), yv).detach().cpu())
        pred_all = model(torch.from_numpy((states - s_mean) / s_std).float().to(device)).cpu().numpy() * a_std + a_mean
        result["action_mae"] = float(np.mean(np.abs(pred_all - actions)))
        pred_val = model(xv).cpu().numpy() * a_std + a_mean
        result["validation_action_mae"] = float(np.mean(np.abs(pred_val - actions[vi])))

    ckpt = out_dir / "policy_v2_robot_contact.pt"
    # Save normalization stats as plain Python lists (NOT numpy arrays): the eval
    # runs inside the Isaac Sim image (older numpy), and a numpy-2.x-pickled array
    # fails to unpickle there ("No module named 'numpy._core'").
    torch.save({"model_state_dict": model.state_dict(), "robot_contact_data_used": True,
                "state_mean": s_mean.tolist(), "state_std": s_std.tolist(),
                "action_mean": a_mean.tolist(), "action_std": a_std.tolist(),
                "obs_dim": int(states.shape[1]), "act_dim": int(actions.shape[1])}, ckpt)
    result["checkpoint_path"] = str(ckpt)
    result["ready_for_real_training"] = True
    result["final_robot_contact_policy_training_ready"] = True
    result["status"] = "TERAFOLD_MODAL_TRAINING_V2_OK"
    result["exact_next_step"] = "Evaluate the policy in a closed-loop Isaac rollout (eval v2) to test autonomous folding."
    _write()
    return result


@app.local_entrypoint()
def main(manifest_path: str = DEFAULT_MANIFEST):
    print(run.remote(manifest_path))
