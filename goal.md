Build the first serious YAM + MolmoAct2 bimanual robot-learning foundation for TeraFold/Tera Robotics.

We are moving from a cheap-arm towel detector/planner stack to a real dual-YAM bimanual manipulation stack using:
- I2RT MolmoAct 2 Research Kit / dual YAM Standard arms
- allenai/MolmoAct2-BimanualYAM
- camera order: top, left, right
- norm_tag: yam_dual_molmoact2
- Rerun for logging
- Isaac Sim / MuJoCo later for digital twin and replay

This sprint is NOT about autonomous robot motion yet.
This sprint is ONLY:
1. asset/config validation
2. MolmoAct2 smoke testing
3. episode/data structure
4. Rerun logging scaffold
5. safe future hooks

Hard safety rules:
- Do NOT touch old cheap-arm robot safety files.
- Do NOT enable real robot movement.
- Do NOT send commands to YAM hardware.
- All hardware execution must stay disabled by default.
- Any future real-hardware command must require explicit dangerous flags.
- Shadow mode only.

Main objective:
Create a clean, tested YAM/MolmoAct2 foundation so I can run a cloud GPU smoke test first, then tomorrow use the real YAM setup only for safe camera/state logging and teleop demo collection.

Add these files/modules:

1. configs/yam_dual_reference.yaml

Include:
- robot: dual_yam_standard
- arms: left_yam, right_yam
- camera_order: [top, left, right]
- norm_tag: yam_dual_molmoact2
- action_mode: continuous
- control_mode: absolute_joint_pose
- autonomous_execution_enabled: false
- safe_speed_scale: 0.10
- max_joint_delta_rad placeholder
- max_gripper_delta placeholder
- workspace_bounds placeholder
- table dimensions placeholder
- camera intrinsics placeholders
- camera paths placeholders
- i2rt repo path placeholder
- molmoact2 model: allenai/MolmoAct2-BimanualYAM
- notes explaining that real execution is forbidden by default

2. terafold/yam/

Create:
- __init__.py
- config.py
- safety.py
- dataset.py
- molmoact2_smoke.py
- rerun_logger.py
- assets.py

Keep everything hardware-free for now.

3. CLI command: yam-check-assets

Command:
python3 -m terafold yam-check-assets \
  --config configs/yam_dual_reference.yaml

It should:
- confirm config exists and parses
- print camera order
- print norm_tag
- print model name
- check if I2RT repo exists if path is configured
- check if MolmoAct2 repo/checkpoint paths exist if configured
- check for URDF/MJCF files if paths are configured
- if missing, print exact next-step clone/download commands
- never require hardware
- never fail just because hardware is missing
- output clear PASS/WARN/FAIL summary

4. CLI command: yam-molmoact2-smoke-test

Command:
python3 -m terafold yam-molmoact2-smoke-test \
  --model allenai/MolmoAct2-BimanualYAM \
  --dtype bfloat16 \
  --out runs/yam_molmoact2_smoke/actions.json

It should:
- load the MolmoAct2-BimanualYAM model if dependencies are installed
- support --dtype bfloat16 and --dtype float32
- use dummy/sample top, left, right images if no images are provided
- allow optional:
  --top-image
  --left-image
  --right-image
- create dummy robot_state with the expected structure/shape
- call model inference or a clearly isolated adapter function
- save predicted actions JSON
- save metadata JSON including:
  model
  dtype
  camera_order
  norm_tag
  action_shape
  timestamp
  hardware_commanded: false
- never command robot hardware
- if dependencies are missing, fail gracefully with exact pip/install instructions

5. CLI command: yam-create-dummy-episode

Command:
python3 -m terafold yam-create-dummy-episode \
  --out data/yam_episodes/towel_fold_smoke/episode_000001 \
  --frames 10

It should create:
data/yam_episodes/towel_fold_smoke/episode_000001/
  metadata.json
  states.jsonl
  actions.jsonl
  cameras/top/
  cameras/left/
  cameras/right/

Dummy episode should include:
- timestamps
- dummy joint positions for left/right arms
- dummy gripper state
- dummy action vectors
- placeholder camera images
- task text: "fold the towel in half neatly"
- success: null
- hardware_commanded: false

6. CLI command: yam-shadow-policy

Command:
python3 -m terafold yam-shadow-policy \
  --episode data/yam_episodes/towel_fold_smoke/episode_000001 \
  --model allenai/MolmoAct2-BimanualYAM \
  --dtype bfloat16 \
  --out runs/yam_shadow_smoke

It should:
- load recorded episode frames/states
- run model prediction frame-by-frame or first-frame smoke mode
- save predicted actions
- compare predicted vs recorded dummy action if present
- log that no hardware was commanded
- optionally write Rerun logs if rerun is installed
- gracefully fallback if rerun is not installed

7. Rerun logging scaffold

Add helper that can log:
- top image
- left image
- right image
- robot state
- predicted action
- actual action if available
- task text
- safety status

If rerun is missing:
- do not crash
- print install hint
- still save JSON outputs

8. Safety clamps

Add pure functions:
- clamp_joint_delta
- clamp_gripper_delta
- validate_workspace_bounds
- assert_shadow_mode
- assert_no_hardware_execution

Tests should verify:
- large actions are clamped
- hardware execution defaults false
- shadow policy never commands hardware
- replay/execution functions cannot run without explicit future flags

9. Docs

Create:
docs/YAM_MOLMOACT2_QUICKSTART.md

Include:
- what this stack is for
- why sim alone is not enough for cloth
- why real teleop demos matter
- Mac setup
- RunPod/cloud GPU setup
- exact smoke-test commands
- exact tomorrow real-YAM checklist
- safety gates
- what not to do

Docs must say:
- Do not run autonomous contact motion tomorrow.
- Tomorrow is for safe setup check, cameras, state logging, teleop demos, and shadow policy only.
- Full cloth folding policy training requires real YAM demos, not only Isaac Sim.

10. Tests

Add tests with no hardware required:
- config loads
- yam-check-assets handles missing I2RT repo gracefully
- dummy episode writes correct structure
- safety clamp works
- smoke test can run with mocked MolmoAct2 model
- shadow policy saves predicted actions and never commands hardware
- rerun logger fallback works if rerun not installed
- CLI help includes new YAM commands

Run:
python3 -m pytest
python3 -m terafold --help

Important implementation constraints:
- Keep code small and clean.
- Do not create a giant brittle framework.
- Prefer pure Python + clear JSON outputs.
- No real robot SDK imports at module import time.
- Any optional heavy dependency must be lazy-imported.
- Missing dependencies should produce helpful messages, not stack traces.
- Never print API keys or secrets.
- Never assume Isaac Sim is installed locally.
- Isaac Sim integration can be documented/scaffolded later, not required in this sprint.

Final report must include:
1. files changed
2. tests passing
3. exact Mac commands to run
4. exact RunPod commands to run
5. what is still NOT implemented
6. tomorrow real-YAM checklist

Definition of done:
- I can run yam-check-assets locally.
- I can create a dummy YAM episode locally.
- I can run yam-shadow-policy locally with mocked/dummy data.
- On RunPod, I can run yam-molmoact2-smoke-test with bfloat16 and save actions.json.
- No real hardware movement is possible from this sprint.