"""Modal Training v3 (STATE-REACTIVE robot-contact) — GATED on valid V16 demos.

Trains a behavior-cloning policy on the V16 state-reactive demos
(`/artifacts/contact_v16/demos/...`). Two things separate v3 from v2:

  1. NO PHASE CLOCK. The V16 observation carries no progress/phase variable, so
     the policy must map the REAL robot+cloth state -> action (state-reactive),
     not imitate a scripted timeline. v3 refuses to run unless the manifest
     asserts `no_phase_clock = true` AND the state actually contains no clock
     feature (checked against the saved feature names).

  2. ANTI-BALLISTIC PENALTY. The V16 state embeds the PREVIOUS action, so v3 adds
     a smoothness / velocity penalty `w_smooth * ||pred_action - prev_action||^2`
     (the implied per-step joint velocity) plus weight-decay effort regularization.
     This directly discourages the large closed-loop jumps that made 3/6 v2 eval
     rollouts ballistic.

Trains ONLY if the demos carry REAL robot-joint-action contact labels
(`robot_contact_data_ready = true`); otherwise it honestly SKIPS. Never fabricates
robot-contact data.

Result: /artifacts/training_v3_state_reactive/training_v3_state_reactive_summary.json
Checkpoint: /artifacts/training_v3_state_reactive/policy_v3_state_reactive.pt
"""

import json
from pathlib import Path

import modal

app = modal.App("terafold-policy-training-v3-state-reactive")
volume = modal.Volume.from_name("terafold-artifacts")

image = modal.Image.debian_slim().pip_install("torch", "numpy")

DEFAULT_MANIFEST = "/artifacts/contact_v16/demos/so101_real_contact_demos_manifest_v16.json"
TRAIN_STEPS = 4000
W_SMOOTH = 0.5          # weight on the anti-ballistic smoothness/velocity penalty
WEIGHT_DECAY = 1e-4     # effort regularization
PREV_ACTION_SLICE = (21, 33)  # V16 state layout: jointpos[0:12] jaw[12:18] jawvel[18:21] prevact[21:33]
CLOCK_TOKENS = ("progress", "phase", "timestep", "time_step", "clock", "normalized_step")


@app.function(gpu="L40S", image=image, volumes={"/artifacts": volume}, timeout=3600)
def run(manifest_path: str = DEFAULT_MANIFEST):
    import numpy as np
    import torch

    result = {
        "status": "TERAFOLD_MODAL_TRAINING_V3_STARTED",
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "manifest_path": manifest_path,
        "robot_contact_data_used": False,
        "no_phase_clock": None,
        "dataset_path": None,
        "num_episodes": None,
        "train_steps": 0,
        "initial_loss": None,
        "final_loss": None,
        "validation_loss": None,
        "action_mae": None,
        "validation_action_mae": None,
        "smoothness_penalty": None,
        "smoothness_penalty_weight": W_SMOOTH,
        "checkpoint_path": None,
        "policy_architecture": None,
        "observation_schema": None,
        "action_schema": None,
        "final_robot_contact_policy_training_ready": False,
        "remaining_blockers": [],
        "exact_next_step": None,
    }
    out_dir = Path("/artifacts/training_v3_state_reactive")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "training_v3_state_reactive_summary.json"

    def _write():
        summary_path.write_text(json.dumps(result, indent=2))
        volume.commit()
        print(json.dumps(result, indent=2))

    mp = Path(manifest_path)
    manifest = None
    if mp.exists():
        try:
            manifest = json.loads(mp.read_text())
        except Exception as exc:  # noqa: BLE001
            result["remaining_blockers"].append(f"manifest parse error: {exc!r}")

    # ---- GATE 1: real robot-contact data ----
    if not manifest or not manifest.get("robot_contact_data_ready"):
        result["status"] = "TERAFOLD_MODAL_TRAINING_V3_SKIPPED"
        reason = "V16 real state-reactive fold demos not ready (robot_contact_data_ready != true)"
        result["training_v3_skipped_reason"] = reason
        result["num_episodes"] = None if manifest is None else manifest.get("num_valid_success")
        result["remaining_blockers"].append(reason)
        result["exact_next_step"] = "Collect >=20 valid V16 demos, then re-run Training v3."
        _write()
        return result

    # ---- GATE 2: no phase clock ----
    result["no_phase_clock"] = bool(manifest.get("no_phase_clock"))
    if not result["no_phase_clock"]:
        result["status"] = "TERAFOLD_MODAL_TRAINING_V3_SKIPPED"
        reason = "manifest does not assert no_phase_clock=true"
        result["training_v3_skipped_reason"] = reason
        result["remaining_blockers"].append(reason)
        _write()
        return result

    # locate npz (manifest dataset_path is container-local; use the volume copy)
    raw_path = Path(manifest.get("dataset_path", ""))
    npz_path = None
    for cand in (mp.parent / raw_path.name, raw_path, mp.parent / "so101_real_contact_demos_v16.npz"):
        if cand.exists():
            npz_path = cand
            break
    if npz_path is None:
        result["status"] = "TERAFOLD_MODAL_TRAINING_V3_SKIPPED"
        reason = f"dataset npz missing near {mp.parent}"
        result["training_v3_skipped_reason"] = reason
        result["remaining_blockers"].append(reason)
        _write()
        return result

    data = np.load(npz_path, allow_pickle=True)
    states = np.asarray(data["states"], np.float32)
    actions = np.asarray(data["actions"], np.float32)

    # ---- GATE 3: state actually carries no clock feature ----
    feature_names = [str(x) for x in list(data["feature_names"])] if "feature_names" in data else \
        [str(x) for x in manifest.get("state_feature_names", [])]
    clock_hits = [n for n in feature_names for t in CLOCK_TOKENS if t in n.lower()]
    if clock_hits:
        result["status"] = "TERAFOLD_MODAL_TRAINING_V3_SKIPPED"
        reason = f"state contains clock feature(s): {clock_hits[:6]}"
        result["training_v3_skipped_reason"] = reason
        result["remaining_blockers"].append(reason)
        _write()
        return result
    if states.ndim != 2 or states.shape[0] == 0:
        result["status"] = "TERAFOLD_MODAL_TRAINING_V3_SKIPPED"
        reason = f"empty/malformed states array shape={states.shape}"
        result["training_v3_skipped_reason"] = reason
        result["remaining_blockers"].append(reason)
        _write()
        return result

    device = "cuda" if torch.cuda.is_available() else "cpu"
    result["robot_contact_data_used"] = True
    result["dataset_path"] = str(npz_path)
    result["num_episodes"] = int(manifest.get("num_valid_success", 0))
    result["observation_schema"] = {
        "dim": int(states.shape[1]),
        "version": manifest.get("observation_schema_version"),
        "no_phase_clock": True,
        "source": "compact_state_reactive(joint_pos + jaw pose/vel + prev_action + cloth metrics + edge features)",
    }
    result["action_schema"] = {"dim": int(actions.shape[1]), "source": "real_commanded_so101_joint_position_targets"}

    s_mean, s_std = states.mean(0), states.std(0) + 1e-6
    a_mean, a_std = actions.mean(0), actions.std(0) + 1e-6
    rng = np.random.default_rng(1234)
    perm = rng.permutation(states.shape[0])
    n_val = max(1, int(0.15 * states.shape[0]))
    vi, ti = perm[:n_val], perm[n_val:]

    S = torch.from_numpy(states).float().to(device)
    A = torch.from_numpy(actions).float().to(device)
    sm = torch.from_numpy(s_mean).float().to(device); ss = torch.from_numpy(s_std).float().to(device)
    am = torch.from_numpy(a_mean).float().to(device); as_ = torch.from_numpy(a_std).float().to(device)
    Xn = (S - sm) / ss
    Yn = (A - am) / as_
    ti_t = torch.from_numpy(ti).long().to(device); vi_t = torch.from_numpy(vi).long().to(device)
    lo, hi = PREV_ACTION_SLICE
    prev_action_raw = S[:, lo:hi]  # raw prev action embedded in the state

    torch.manual_seed(1234)
    model = torch.nn.Sequential(
        torch.nn.Linear(states.shape[1], 256), torch.nn.ReLU(),
        torch.nn.Linear(256, 256), torch.nn.ReLU(),
        torch.nn.Linear(256, actions.shape[1]),
    ).to(device)
    result["policy_architecture"] = (
        "MLP[in->256->256->act], ReLU, AdamW(lr=2e-3, wd=1e-4), "
        f"loss = MSE(bc) + {W_SMOOTH}*mean||pred_action - prev_action||^2 (velocity/smoothness penalty)"
    )
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=WEIGHT_DECAY)
    mse = torch.nn.MSELoss()

    def losses(idx):
        xn = Xn[idx]; yn = Yn[idx]
        pred_n = model(xn)
        bc = mse(pred_n, yn)
        pred_raw = pred_n * as_ + am
        smooth = ((pred_raw - prev_action_raw[idx]) ** 2).sum(dim=1).mean()  # implied per-step joint velocity^2
        return bc, smooth

    for step in range(1, TRAIN_STEPS + 1):
        model.train()
        bc, smooth = losses(ti_t)
        loss = bc + W_SMOOTH * smooth
        opt.zero_grad(); loss.backward(); opt.step()
        if step == 1:
            result["initial_loss"] = float(loss.detach().cpu())
    result["train_steps"] = TRAIN_STEPS

    model.eval()
    with torch.no_grad():
        bc_t, smooth_t = losses(ti_t)
        bc_v, smooth_v = losses(vi_t)
        result["final_loss"] = float((bc_t + W_SMOOTH * smooth_t).detach().cpu())
        result["final_bc_loss"] = float(bc_t.detach().cpu())
        result["validation_loss"] = float((bc_v + W_SMOOTH * smooth_v).detach().cpu())
        result["validation_bc_loss"] = float(bc_v.detach().cpu())
        result["smoothness_penalty"] = float(smooth_t.detach().cpu())  # raw units (sum-sq joint delta)
        result["validation_smoothness_penalty"] = float(smooth_v.detach().cpu())
        pred_all = (model(Xn) * as_ + am).cpu().numpy()
        result["action_mae"] = float(np.mean(np.abs(pred_all - actions)))
        result["validation_action_mae"] = float(np.mean(np.abs(pred_all[vi] - actions[vi])))

    ckpt = out_dir / "policy_v3_state_reactive.pt"
    # save norm stats as plain lists (Isaac eval image has older numpy)
    torch.save({
        "model_state_dict": model.state_dict(),
        "robot_contact_data_used": True,
        "no_phase_clock": True,
        "state_mean": s_mean.tolist(), "state_std": s_std.tolist(),
        "action_mean": a_mean.tolist(), "action_std": a_std.tolist(),
        "obs_dim": int(states.shape[1]), "act_dim": int(actions.shape[1]),
        "prev_action_slice": list(PREV_ACTION_SLICE),
        "observation_schema_version": manifest.get("observation_schema_version"),
        "feature_names": feature_names,
    }, ckpt)
    result["checkpoint_path"] = str(ckpt)
    result["final_robot_contact_policy_training_ready"] = True
    result["status"] = "TERAFOLD_MODAL_TRAINING_V3_OK"
    result["exact_next_step"] = (
        "Run Eval v3 closed-loop (no clock, real articulation, reject ballistic) to test robust state-reactive folding."
    )
    _write()
    return result


@app.local_entrypoint()
def main(manifest_path: str = DEFAULT_MANIFEST):
    print(run.remote(manifest_path))
