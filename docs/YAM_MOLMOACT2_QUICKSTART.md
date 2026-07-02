# YAM + MolmoAct2 Quickstart (Shadow Mode)

TeraFold's bimanual robot-learning foundation for the I2RT MolmoAct2 Research
Kit: **dual YAM Standard arms**, cameras ordered **top, left, right**, model
**`allenai/MolmoAct2-BimanualYAM`**, normalization tag **`yam_dual_molmoact2`**.

**This sprint is shadow mode only.** There is no code path in this repo that
sends a command to YAM hardware. Config validation, model smoke tests, episode
structure, and Rerun logging — that's it.

## What this stack is for

The old stack drove a cheap 7-DOF bus-servo arm with a towel detector and a
scripted planner. That was the right way to learn the plumbing, but it tops out
fast: single arm, no compliant control, no learning-grade data. The YAM stack
replaces it with the setup serious cloth manipulation needs — two arms
(cloth folding is intrinsically bimanual), three calibrated cameras, and a
pretrained bimanual VLA (MolmoAct2) we can fine-tune on our own teleop demos.

## Why sim alone is not enough for cloth

Rigid-body sim transfers to reality reasonably well; **cloth does not**.
Simulated cloth (Isaac Sim / MuJoCo) gets bending stiffness, friction,
self-collision and crumpling dynamics wrong enough that a policy trained purely
in sim learns grasp points and fold trajectories that miss on real towels. Sim
is still valuable — as a digital twin for replay, collision checking and
visualization — but it cannot be the only data source for the folding policy.

## Why real teleop demos matter

MolmoAct2 is pretrained on real bimanual YAM data. To make it fold *our*
towels on *our* table, it needs fine-tuning demos from the real rig: real
lighting, real cloth, real gripper-cloth interaction, recorded in exactly the
(top, left, right) + joint-state format this package defines. Teleop demos are
also the safest data there is — a human hand is in the loop the whole time.
Episode collection tomorrow feeds directly into the `data/yam_episodes/`
structure that `yam-create-dummy-episode` prototypes today.

## Setup — Mac (local, no GPU)

```bash
cd Cloth-Folder
pip install -e '.[dev]'          # core deps only (numpy/yaml/typer/pydantic/scipy)
python3 -m pytest                # everything passes without torch/rerun
python3 -m terafold --help       # yam-* commands are listed at the bottom
```

Local shadow-mode dry run (no torch, no GPU, no hardware):

```bash
python3 -m terafold yam-check-assets --config configs/yam_dual_reference.yaml

python3 -m terafold yam-create-dummy-episode \
  --out data/yam_episodes/towel_fold_smoke/episode_000001 --frames 10

python3 -m terafold yam-molmoact2-smoke-test --mock \
  --out runs/yam_molmoact2_smoke/actions.json

python3 -m terafold yam-shadow-policy --mock \
  --episode data/yam_episodes/towel_fold_smoke/episode_000001 \
  --out runs/yam_shadow_smoke
```

Optional visual logging: `pip install -e '.[rerun]'`, then re-run the shadow
policy and open the saved file with `rerun runs/yam_shadow_smoke/shadow.rrd`.

## Setup — RunPod / cloud GPU

Pick a CUDA pod (A100/H100, or a 24 GB+ card; a 7B-class VLA in bfloat16 wants
roughly 16–20 GB). Then:

```bash
git clone <your-fork-of-this-repo> Cloth-Folder && cd Cloth-Folder
pip install -e '.[dev]'
pip install transformers accelerate einops pillow   # torch is preinstalled on CUDA pods
# (from scratch instead: pip install -e '.[molmoact]')

# real GPU smoke test — downloads the checkpoint from HF on first run
python3 -m terafold yam-molmoact2-smoke-test \
  --model allenai/MolmoAct2-BimanualYAM \
  --dtype bfloat16 \
  --out runs/yam_molmoact2_smoke/actions.json
```

Success looks like `Smoke test OK: action_shape=[...]` plus `actions.json` and
`metadata.json` (which records `hardware_commanded: false`). If the checkpoint's
remote code differs from the MolmoAct recipe, the command fails with a pointer
to the model card instead of a stack trace — adapt
`MolmoAct2Adapter.predict()` in `terafold/yam/molmoact2_smoke.py`.

Then shadow a (dummy or synced real) episode on the GPU:

```bash
python3 -m terafold yam-create-dummy-episode \
  --out data/yam_episodes/towel_fold_smoke/episode_000001 --frames 10
python3 -m terafold yam-shadow-policy \
  --episode data/yam_episodes/towel_fold_smoke/episode_000001 \
  --model allenai/MolmoAct2-BimanualYAM --dtype bfloat16 \
  --out runs/yam_shadow_smoke
```

## Tomorrow — real-YAM checklist (SAFE TASKS ONLY)

> **Do not run autonomous contact motion tomorrow.** Tomorrow is for safe setup
> check, cameras, state logging, teleop demos, and shadow policy **only**.
> Full cloth folding policy training requires real YAM demos, not only Isaac
> Sim.

1. **Power/E-stop walkthrough** before anything else: know where the kill
   switches are for both arms; clear the workspace.
2. **Clone the I2RT repo** on the rig machine and record its path:
   `git clone https://github.com/i2rt-robotics/i2rt.git ~/i2rt`, then set
   `paths.i2rt_repo` (and `paths.urdf`) in `configs/yam_dual_reference.yaml`.
3. **Re-run** `python3 -m terafold yam-check-assets` until the summary is PASS/WARN
   with the repo and URDF checks green.
4. **Cameras**: plug in top/left/right, fill in `cameras.*.path` in the config,
   verify each stream, and confirm the physical mounting matches the
   `[top, left, right]` order — MolmoAct2 is sensitive to camera order.
5. **State logging (arms passive/gravity-comp, no commands)**: capture joint
   states + camera frames into the `data/yam_episodes/` layout that
   `yam-create-dummy-episode` prototypes; verify with the episode loader.
6. **Teleop demos** using I2RT's own teleop tooling (human in the loop the
   whole time): record 5–10 towel-fold episodes with task text
   `"fold the towel in half neatly"`.
7. **Shadow the recorded episodes** (offline, zero hardware writes):
   `python3 -m terafold yam-shadow-policy --episode <recorded_dir> ...` and review the
   prediction-vs-teleop diffs in `comparison.json` / the Rerun log.
8. **Sync episodes off the rig** (cloud storage or the GPU box) the same day.

## Safety gates

- `autonomous_execution_enabled: false` in the config; loading a config with
  it flipped raises `ShadowModeViolation` before anything else runs.
- `terafold/yam/safety.py` has **no** SDK imports and hard-refuses execution:
  `assert_no_hardware_execution()` raises on any attempt this sprint, flags or
  not (`HARDWARE_EXECUTION_IMPLEMENTED = False` is a code constant, not config).
- Any *future* execution sprint must additionally require BOTH dangerous flags
  `--enable-yam-motion` and `--i-understand-this-moves-the-yam` — the same
  double-gate discipline as the old arm, whose safety files are untouched.
- Pure clamps (`clamp_joint_delta`, `clamp_gripper_delta`,
  `validate_workspace_bounds`) are ready for the future execution path and are
  tested against oversized actions now.
- Every output artifact records `hardware_commanded: false`.
- **Hub-code containment**: MolmoAct2 needs `trust_remote_code=True`, which
  executes code from the HF repo in-process. On the rig machine, always pin it —
  pass `--revision <commit-sha>` to `yam-molmoact2-smoke-test` /
  `yam-shadow-policy`, or point `--model` at a local snapshot directory
  (local paths are loaded with `local_files_only=True`, nothing is fetched).

## What NOT to do

- Do **not** run autonomous or contact motion tomorrow — no exceptions, not
  even "just one slow fold".
- Do **not** flip `autonomous_execution_enabled` or
  `HARDWARE_EXECUTION_IMPLEMENTED`; this sprint has no execution code behind
  them anyway.
- Do **not** modify the old cheap-arm safety files (`terafold/robot/safety_gate.py`,
  `safety_state.py`) — the YAM stack deliberately does not touch them.
- Do **not** reorder cameras or change `norm_tag: yam_dual_molmoact2` — the
  checkpoint's normalization statistics depend on both.
- Do **not** assume Isaac Sim numbers transfer to real cloth; the digital twin
  is for replay/visualization, not a substitute for real demos.
