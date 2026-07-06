from __future__ import annotations

import argparse

import numpy as np
import torch

from clean_towel_v4_common import (
    ACTION_NAMES,
    DATASET_PATH,
    POLICY_PATH,
    TRAIN_SUMMARY_PATH,
    V1_ASSISTED_CAPTURE,
    V1_ASSISTED_SUMMARY,
    V1_ASSISTED_TRACE,
    V4_DIR,
    build_training_dataset,
    command_run_string,
    save_dataset,
    PolicyMLP,
)
from so101_clean_towel_scene_utils_v1 import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train V4 compact-state SO-101 policy for clean towel scene.")
    parser.add_argument("--out_dir", type=str, default=str(V4_DIR))
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=3.0e-3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--fold_steps", type=int, default=80)
    parser.add_argument("--settle_steps", type=int, default=20)
    parser.add_argument("--augment_repeats", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    V4_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(1234)
    np.random.seed(1234)

    dataset = build_training_dataset(
        fold_steps=args.fold_steps,
        settle_steps=args.settle_steps,
        augment_repeats=args.augment_repeats,
    )
    states = dataset["states"].astype(np.float32)
    actions = dataset["actions"].astype(np.float32)
    state_mean = states.mean(axis=0)
    state_std = states.std(axis=0) + 1.0e-6
    action_mean = actions.mean(axis=0)
    action_std = actions.std(axis=0) + 1.0e-6
    states_n = (states - state_mean) / state_std
    actions_n = (actions - action_mean) / action_std

    x = torch.from_numpy(states_n).float()
    y = torch.from_numpy(actions_n).float()
    model = PolicyMLP(state_dim=states.shape[1], action_dim=actions.shape[1], hidden_dim=args.hidden_dim)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1.0e-4)
    loss_fn = torch.nn.MSELoss()
    losses = []
    for _ in range(int(args.epochs)):
        pred = model(x)
        loss = loss_fn(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))

    with torch.no_grad():
        final_mse = float(loss_fn(model(x), y).detach().cpu())
        pred_actions = model(x).cpu().numpy() * action_std + action_mean
        action_mae = float(np.mean(np.abs(pred_actions - actions)))

    save_dataset(DATASET_PATH, dataset)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "state_dim": int(states.shape[1]),
            "action_dim": int(actions.shape[1]),
            "hidden_dim": int(args.hidden_dim),
            "state_mean": state_mean.astype(np.float32),
            "state_std": state_std.astype(np.float32),
            "action_mean": action_mean.astype(np.float32),
            "action_std": action_std.astype(np.float32),
            "feature_names": dataset["feature_names"].tolist(),
            "action_names": list(ACTION_NAMES),
            "policy_basis": "supervised_fit_to_synthetic_joint_targets_from_particle_space_assisted_fold_progress",
            "contains_robot_contact_demonstrations": False,
        },
        POLICY_PATH,
    )

    summary = {
        "status": "CLEAN_TOWEL_POLICY_V4_TRAIN_OK",
        "command_run": command_run_string(),
        "policy_path": str(POLICY_PATH),
        "dataset_path": str(DATASET_PATH),
        "v1_assisted_capture_exists": V1_ASSISTED_CAPTURE.exists(),
        "v1_assisted_trace_exists": V1_ASSISTED_TRACE.exists(),
        "v1_assisted_summary_exists": V1_ASSISTED_SUMMARY.exists(),
        "source_points": str(dataset["source_points"]),
        "num_samples": int(states.shape[0]),
        "state_dim": int(states.shape[1]),
        "action_dim": int(actions.shape[1]),
        "epochs": int(args.epochs),
        "final_train_mse_normalized": final_mse,
        "final_action_mae": action_mae,
        "loss_first": float(losses[0]),
        "loss_last": float(losses[-1]),
        "action_label_source": str(dataset["action_label_source"]),
        "contains_robot_contact_demonstrations": False,
        "honest_note": (
            "The V1 assisted capture has no robot action labels. "
            "This V4 policy is trained on synthetic SO-101 joint targets derived from assisted fold progress; it is not trained from robot-contact demonstrations."
        ),
    }
    write_json(TRAIN_SUMMARY_PATH, summary, print_payload=True)


if __name__ == "__main__":
    main()
