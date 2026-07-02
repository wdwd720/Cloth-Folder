# Data & LeRobot — logging now so the system can improve later

Build the data foundation **before** the robot works perfectly. Every real or
dry-run episode should be convertible into a structured episode and exported toward
a LeRobot-compatible dataset, so that once demonstrations exist you can train ACT
without re-plumbing. You do **not** need LeRobot installed to validate the internal
schema.

> **Growing the towel perception set?** Two perception-only data paths are
> documented separately (neither touches the hardware safety gates):
> external real images (Kaggle / Open Images) + Claude pseudo-labeling in
> [`TOWEL_PSEUDOLABELING.md`](TOWEL_PSEUDOLABELING.md), and OpenAI-generated
> "realism bridge" images in [`TOWEL_OPENAI_GENERATION.md`](TOWEL_OPENAI_GENERATION.md).

## What gets logged

Every motion session writes a JSONL event log to
`runs/real_motion_logs/<command>_<timestamp>.jsonl` via the command logger. Events
include the request args, a robot-config snapshot, the joint-map snapshot, the
perception output, the fold plan, each read, each write, each refusal/abort, and the
final status.

## The episode schema (`terafold.data.schema`)

An `EpisodeFrame` captures, per step (all fields optional except `timestamp`):

- identity: `timestamp`, `episode_id`, `task`, `robot_type`, `robot_config`,
  `port`, `protocol`, `baudrate`, `servo_ids`
- state/action (raw servo units): `joint_positions_raw`, `joint_targets_raw`,
  `joint_speeds_raw`, `action_raw`, `action_normalized`
- vision/calibration ids: `image_paths`, `camera_intrinsics_id`,
  `table_homography_id`, `robot_table_transform_id`
- perception: `perception`, `towel_corners`, `fold_line`, `grasp_point`,
  `place_point`, `fold_direction`
- plan/exec: `planned_trajectory`, `ghost_motion_plan`, `executed_command`,
  `readbacks`
- labels/safety: `safety_flags`, `dry_run`, `contact_enabled`, `success`,
  `failure_reason`, `operator_notes`

> **Action ordering is stable: `action_vector(frame)` orders by ascending servo
> id.** This is the contract that keeps datasets consistent across episodes and
> embodiments.

## Commands

```bash
# Summarize a motion log, an episode dir, or an episode JSON.
python3 -m terafold episode-summary --episode runs/real_motion_logs/<session>.jsonl

# Export recorded episodes to a LeRobot-compatible dataset (scaffold if LeRobot
# is not installed).
python3 -m terafold export-lerobot \
  --episodes data/episodes/fold_towel_half_demo \
  --out data/lerobot/terafold_custom_arm_v0
```

## LeRobot compatibility (degrade gracefully)

- If the `lerobot` package is importable, the export notes it and can be filled out
  more fully.
- If not, the exporter still writes a compatible folder scaffold:
  `meta/info.json` (fps, robot type, features describing `observation.state` and
  `action` as **joint-space vectors ordered by servo id**, plus image keys), the
  per-episode data (Parquet if `pandas`+`pyarrow` are available, else a JSONL
  fallback), and a README explaining the missing dependency.
- Whether calibration was present is recorded **explicitly** in the export, so a
  consumer never assumes calibrated coordinates that did not exist.

## Action representation

On the custom arm the default action space is **joint-space (raw units)** — it needs
no kinematics model and maps directly to `WritePosEx`. End-effector-space actions are
added only after a validated URDF/IK exists (which the custom arm does not have yet;
`robot-model-status` will tell you).

## Why this matters

Per the roadmap, the fastest path to a system that can *improve with data* is
LeRobot-compatible logging of observations, joint states, targets, images,
calibration metadata, and success/failure — followed by small-scale imitation
learning with **ACT first**, then a stronger sequential policy (Diffusion Policy)
once enough real demonstrations exist. The geometry + safety-gated primitives remain
the fallback controller.
