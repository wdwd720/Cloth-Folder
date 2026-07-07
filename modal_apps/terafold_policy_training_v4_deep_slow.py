"""Modal Training v4 (DEEP-SLOW, no-clock) — GATED on valid V17 demos.

Trains a state-reactive behavior-cloning policy on the V17 DEEP-but-SLOW demos
(`/artifacts/contact_v17/demos/...`). Like v3 it uses NO phase clock, but it adds
STRONGER anti-ballistic regularization so the policy folds deep (from the V17 demos)
WITHOUT the closed-loop over-drive that made the v3 ballistic tail unstable:

  loss = MSE(bc)
       + W_SMOOTH  * mean||pred_action - prev_action||^2   (action smoothing / per-step velocity)
       + W_EFFORT  * mean||pred_action - a_mean||^2         (effort / magnitude penalty)

`prev_action` is embedded in the V16/V17 state (slice [21:33]); `a_mean` is the
dataset action mean. Triple-gated: real robot-contact data AND no_phase_clock=true
AND a clock-free feature scan. Never fabricates data.

Result: /artifacts/training_v4_deep_slow/training_v4_deep_slow_summary.json
Checkpoint: /artifacts/training_v4_deep_slow/policy_v4_deep_slow.pt
"""

import json
from pathlib import Path

import modal

app = modal.App("terafold-policy-training-v4-deep-slow")
volume = modal.Volume.from_name("terafold-artifacts")

image = modal.Image.debian_slim().pip_install("torch", "numpy")

DEFAULT_MANIFEST = "/artifacts/contact_v17/demos/so101_deep_slow_demos_manifest_v17.json"
TRAIN_STEPS = 5000
# Attempt-1 (W_SMOOTH=1.0 + W_EFFORT=0.05 toward a_mean) OVER-DAMPED the policy: closed-loop
# it folded erratically (clean 0.27) and any EMA filter killed folding entirely. The effort
# term (pull toward the trajectory-mean action) biased predictions off the sequential fold
# path. Attempt-2 reverts to v3's PROVEN recipe (smoothing only, W=0.5, no effort term) but on
# the DEEPER V17 demos — keep v3's ~0.75 clean rate while folding deeper.
W_SMOOTH = 0.5          # action smoothing / per-step velocity penalty (v3-proven level)
W_EFFORT = 0.0          # effort term disabled (biased predictions toward the trajectory mean)
WEIGHT_DECAY = 1e-4
PREV_ACTION_SLICE = (21, 33)  # V16/V17 state layout: prevact at [21:33]
CLOCK_TOKENS = ("progress", "phase", "timestep", "time_step", "clock", "normalized_step")


@app.function(gpu="L40S", image=image, volumes={"/artifacts": volume}, timeout=3600)
def run(manifest_path: str = DEFAULT_MANIFEST):
    import numpy as np
    import torch

    result = {
        "status": "TERAFOLD_MODAL_TRAINING_V4_STARTED",
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
        "effort_penalty": None,
        "smoothness_penalty_weight": W_SMOOTH,
        "effort_penalty_weight": W_EFFORT,
        "checkpoint_path": None,
        "policy_architecture": None,
        "observation_schema": None,
        "action_schema": None,
        "final_robot_contact_policy_training_ready": False,
        "remaining_blockers": [],
        "exact_next_step": None,
    }
    out_dir = Path("/artifacts/training_v4_deep_slow")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "training_v4_deep_slow_summary.json"

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

    if not manifest or not manifest.get("robot_contact_data_ready"):
        result["status"] = "TERAFOLD_MODAL_TRAINING_V4_SKIPPED"
        reason = "V17 deep-slow demos not ready (robot_contact_data_ready != true)"
        result["training_v4_skipped_reason"] = reason
        result["num_episodes"] = None if manifest is None else manifest.get("num_valid_success")
        result["remaining_blockers"].append(reason)
        result["exact_next_step"] = "Collect >=20 valid V17 deep-slow demos, then re-run Training v4."
        _write()
        return result

    result["no_phase_clock"] = bool(manifest.get("no_phase_clock"))
    if not result["no_phase_clock"]:
        result["status"] = "TERAFOLD_MODAL_TRAINING_V4_SKIPPED"
        reason = "manifest does not assert no_phase_clock=true"
        result["training_v4_skipped_reason"] = reason
        result["remaining_blockers"].append(reason)
        _write()
        return result

    raw_path = Path(manifest.get("dataset_path", ""))
    npz_path = None
    for cand in (mp.parent / raw_path.name, raw_path, mp.parent / "so101_deep_slow_demos_v17.npz"):
        if cand.exists():
            npz_path = cand
            break
    if npz_path is None:
        result["status"] = "TERAFOLD_MODAL_TRAINING_V4_SKIPPED"
        reason = f"dataset npz missing near {mp.parent}"
        result["training_v4_skipped_reason"] = reason
        result["remaining_blockers"].append(reason)
        _write()
        return result

    data = np.load(npz_path, allow_pickle=True)
    states = np.asarray(data["states"], np.float32)
    actions = np.asarray(data["actions"], np.float32)

    feature_names = [str(x) for x in list(data["feature_names"])] if "feature_names" in data else \
        [str(x) for x in manifest.get("state_feature_names", [])]
    clock_hits = [n for n in feature_names for t in CLOCK_TOKENS if t in n.lower()]
    if clock_hits or states.ndim != 2 or states.shape[0] == 0:
        result["status"] = "TERAFOLD_MODAL_TRAINING_V4_SKIPPED"
        reason = f"clock feature(s) {clock_hits[:4]} or empty states {states.shape}"
        result["training_v4_skipped_reason"] = reason
        result["remaining_blockers"].append(reason)
        _write()
        return result

    device = "cuda" if torch.cuda.is_available() else "cpu"
    result["robot_contact_data_used"] = True
    result["dataset_path"] = str(npz_path)
    result["num_episodes"] = int(manifest.get("num_valid_success", 0))
    result["observation_schema"] = {"dim": int(states.shape[1]), "version": manifest.get("observation_schema_version"),
                                    "no_phase_clock": True}
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
    prev_action_raw = S[:, lo:hi]

    torch.manual_seed(1234)
    model = torch.nn.Sequential(
        torch.nn.Linear(states.shape[1], 256), torch.nn.ReLU(),
        torch.nn.Linear(256, 256), torch.nn.ReLU(),
        torch.nn.Linear(256, actions.shape[1]),
    ).to(device)
    result["policy_architecture"] = (
        f"MLP[in->256->256->act], ReLU, AdamW(lr=2e-3, wd={WEIGHT_DECAY}), "
        f"loss = MSE(bc) + {W_SMOOTH}*mean||pred-prev_action||^2 + {W_EFFORT}*mean||pred-a_mean||^2"
    )
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=WEIGHT_DECAY)
    mse = torch.nn.MSELoss()

    def losses(idx):
        pred_n = model(Xn[idx])
        bc = mse(pred_n, Yn[idx])
        pred_raw = pred_n * as_ + am
        smooth = ((pred_raw - prev_action_raw[idx]) ** 2).sum(dim=1).mean()
        effort = ((pred_raw - am) ** 2).sum(dim=1).mean()
        return bc, smooth, effort

    for step in range(1, TRAIN_STEPS + 1):
        model.train()
        bc, smooth, effort = losses(ti_t)
        loss = bc + W_SMOOTH * smooth + W_EFFORT * effort
        opt.zero_grad(); loss.backward(); opt.step()
        if step == 1:
            result["initial_loss"] = float(loss.detach().cpu())
    result["train_steps"] = TRAIN_STEPS

    model.eval()
    with torch.no_grad():
        bct, smt, eft = losses(ti_t)
        bcv, smv, efv = losses(vi_t)
        result["final_loss"] = float((bct + W_SMOOTH * smt + W_EFFORT * eft).detach().cpu())
        result["final_bc_loss"] = float(bct.detach().cpu())
        result["validation_loss"] = float((bcv + W_SMOOTH * smv + W_EFFORT * efv).detach().cpu())
        result["validation_bc_loss"] = float(bcv.detach().cpu())
        result["smoothness_penalty"] = float(smt.detach().cpu())
        result["validation_smoothness_penalty"] = float(smv.detach().cpu())
        result["effort_penalty"] = float(eft.detach().cpu())
        pred_all = (model(Xn) * as_ + am).cpu().numpy()
        result["action_mae"] = float(np.mean(np.abs(pred_all - actions)))
        result["validation_action_mae"] = float(np.mean(np.abs(pred_all[vi] - actions[vi])))

    ckpt = out_dir / "policy_v4_deep_slow.pt"
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
    result["status"] = "TERAFOLD_MODAL_TRAINING_V4_OK"
    result["exact_next_step"] = "Run multi-seed Eval v4 (raw + EMA beta 0.3/0.4/0.6) to test robust deep folding."
    _write()
    return result


@app.local_entrypoint()
def main(manifest_path: str = DEFAULT_MANIFEST):
    print(run.remote(manifest_path))
