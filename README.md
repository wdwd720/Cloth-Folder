# TeraFold

**A trainable, physically-grounded cloth-folding robot stack.**

TeraFold is a hybrid robot-learning system for folding cloth, towels, and
garments with one or two low-cost robot arms and a top-down camera. It is built
toward real hotel-room robotics — starting with the simplest useful skill and
growing from there.

The V0 task is **`fold_towel_half_right_to_left`**: a small rectangular towel
lies on a table, a top-down camera observes it, and the arm grasps the right
side, lifts, swings it over a vertical center crease, releases, and verifies the
result.

> ⚠️ **This is not a hard-coded demo.** Geometry and physics are used only as a
> *prior / scaffold*. Every fragile part (perception, grasp, place, lift, slip,
> success) is designed to be replaced or corrected by learned models trained on
> data. Hard-coded values exist to bootstrap learning, never as the final answer.

---

## 1. What TeraFold is

A layered stack where each layer is independently testable and learnable:

| Stage | Layer | What it does | Learnable? |
|------|-------|--------------|-----------|
| 0 | Synthetic data + keypoint detector | Render towels, learn to see corners/grasp/place | ✅ trained today |
| 1 | Perception | Predict cloth corners, fold line, grasp & place points | ✅ |
| 2 | Geometry + physics planner | Reflect across crease, lift/arc/place trajectory | prior (tunable) |
| 3 | Residual correction model | Learn dx/dy grasp & place, lift/arc/release deltas | ✅ |
| 4 | Human teleop demos → policy | Imitation learning from real episodes | ✅ |
| 5 | ACT / SmolVLA | Action-chunk policies from real robot data | ✅ |

## 2. Why cloth folding is hard

Cloth is **deformable and high-dimensional**. Unlike a rigid box, a towel has
near-infinite configurations, self-occludes, wrinkles, and slips. A fold that
works once may fail next time because the cloth started 2 cm to the left or had
a fold already in it. Pure scripts break the moment reality deviates from the
assumption. So we combine:

* **Geometry** — gives a correct *first guess* (reflect the grasp across the
  crease, lift to clear friction, arc over, place short to counter slip).
* **Learning** — corrects the guess from real data, and eventually replaces the
  fragile perception and grasp choices entirely.

## 3. Why hybrid geometry + learning

Pure learning needs enormous data; pure scripting is brittle. The hybrid lets us
**start today** (geometry + synthetic-trained perception) and **improve
continuously** (residual model → imitation policy → ACT/SmolVLA) as real demos
accumulate. The geometry also acts as a safety envelope and an interpretable
feature source for the learners.

## 4. Hardware setup

* **Arm (now):** LewanSoul / HiWonder LeArm 6-DOF (or similar). The adapter is a
  *safe shell only* — TeraFold does **not** ship an invented serial protocol.
  Real motion is blocked until you provide a protocol file.
* **Arm (future):** SO-100 / SO-101 / LeRobot-native arms; later **two arms**.
* **Camera:** Logitech C270 or any generic USB webcam, mounted top-down.
* **Computer:** Windows / macOS / Linux laptop. **No ROS required.**

## 5. Safety warnings

* **Everything physical defaults to DRY-RUN.** No motion happens unless you
  explicitly ask for it.
* Real movement requires **both** flags:
  `--enable-motion --i-understand-this-moves-hardware`
* Safety checks: workspace bounds, max speed/accel, min/max Z, forbidden zones,
  per-step distance, gripper/joint placeholders, command logging, **Ctrl+C
  emergency stop**, and a **`STOP_TERAFOLD` stop-file** watchdog. Slow mode is on
  by default.
* The LeArm adapter raises a clear error if you request real motion without a
  protocol implementation. **Never** point an arm at a person; keep an
  e-stop / power cut within reach.

## 6. Install

```bash
# Core (numpy-only Stage-0 pipeline works with just this):
pip install -e ".[dev]"

# Optional extras (install what you need):
pip install -e ".[ml]"       # torch + torchvision (train keypoint/success/residual, ACT/SmolVLA)
pip install -e ".[vision]"   # opencv + pillow (real webcam, JPEG)
pip install -e ".[lerobot]"  # lerobot + huggingface_hub + datasets
pip install -e ".[all]"      # everything
```

> The repo is intentionally usable with **numpy only**. A built-in pure-Python
> PNG codec means synthetic data, planning, mock rollout, recording, and
> LeRobot export all run before you install torch or OpenCV.

## 7. Run without hardware

```bash
pytest                                   # all Stage-0 logic is unit-tested
terafold dry-run-fold --task configs/task_fold_towel_half.yaml --robot mock --camera mock
```

`dry-run-fold` renders a synthetic towel (mock camera), perceives it, plans the
fold, and executes the trajectory on the **mock robot** with zero physical risk.

## 8. Generate synthetic data

```bash
terafold generate-synthetic --num 10000 --out data/synth_v0
terafold generate-synthetic --num 2000 --out data/synth_markers --markers   # marker-assisted
```

Randomizes cloth size, rotation, position, color, lighting, shadows, noise,
table texture, edge waviness, wrinkles, mild non-rectangular deformation, and
partial occlusion — with corner / edge / grasp / place / fold-line / mask labels.

## 9. Train the keypoint model

```bash
terafold train-keypoints --data data/synth_v0 --epochs 30 --out runs/keypoints_v0
```

A small U-Net predicts heatmaps for the 8 keypoints (4 corners, grasp, place,
2 fold-line endpoints). Requires `pip install -e ".[ml]"`.

## 10. Test the webcam

```bash
terafold camera-test --camera-index 0
```

## 11. Calibrate the homography (image → table)

```bash
terafold capture-calibration --camera-index 0 --out data/calib/table.jpg
terafold calibrate-homography --image data/calib/table.jpg --out data/calib/homography.json \
    --table-width 0.70 --table-height 0.50
```

Until calibrated, planning falls back to an **uncalibrated** metric estimate from
the cloth's expected size — usable for dry-run only (the planner flags it).

## 12. Plan a fold from the camera

```bash
terafold plan-fold --image data/calib/table.jpg --task configs/task_fold_towel_half.yaml \
    --checkpoint runs/keypoints_v0/model.pt --calibration data/calib/homography.json \
    --out runs/plan.json --viz runs/plan_overlay.png
```

## 13. Dry-run the fold

```bash
terafold dry-run-fold --task configs/task_fold_towel_half.yaml --robot mock --camera mock
```

### One-command demo (`demo-today`)

The fastest path to a real-world towel half-fold with one SO-101 arm, a top-down
webcam, and a towel with 4 colored corner markers (red/green/blue/yellow). It
opens the camera, detects the markers, computes + overlays the fold plan, runs a
safe dry-run, and records the episode:

```bash
# Safe dry-run (default). Falls back to a marker-rendering mock camera if no webcam:
terafold demo-today --camera-index 0 --mode markers --task configs/task_fold_towel_half.yaml --dry-run

# Dry-run against the SO-101 adapter (no lerobot needed):
terafold demo-today --camera-index 0 --mode markers --robot so101 --dry-run

# Real, slow execution — requires ALL of these (only when you're ready):
terafold demo-today --camera-index 0 --mode markers --robot so101 \
    --enable-motion --i-understand-this-moves-hardware
```

Safety is layered: slow speed + workspace bounds (from the task config), a
`STOP_TERAFOLD` stop-file and Ctrl+C emergency stop (watchdog), the trajectory is
validated against the workspace before any motion, and the run is saved as an
episode (frames, observations, actions, plan, safety, result) with an overlay
image. `--mode markers` uses the colored fiducials (falling back to the classical
detector if markers aren't all visible); `--mode model` uses the learned model
(`--checkpoint`); `--mode claude` uses Claude Vision (geometry only).

### Phone-camera / image / unknown-arm path (no robot camera needed)

For a non-SO-101 arm with no robot-mounted camera, the demo runs off a phone
webcam, a Continuity/USB camera, **or a single top-down photo**:

```bash
# 0. Find a working camera index (phone webcam / Continuity / USB):
terafold list-cameras --max-index 8 --out runs/camera_scan

# 1a. Take a photo from above and feed the file (markers / model / claude):
terafold demo-image --image towel.jpg --mode markers \
    --overlay-out runs/demo_image/overlay.png --save-json runs/demo_image/result.json
#   --mode model --checkpoint runs/keypoints_v0/model.pt
#   --mode claude   (needs:  export ANTHROPIC_API_KEY=... ; pip install anthropic)

# 1b. ...or go live from the chosen index:
terafold demo-today --camera-index 1 --mode markers --dry-run \
    --save-frame runs/demo_today/frame.png --overlay-out runs/demo_today/overlay.png \
    --save-json runs/demo_today/result.json --confidence-threshold 0.70

# 2. Map image pixels to table coordinates (required for real motion):
terafold calibrate-table-from-image --image calib.jpg --out data/calib/homography.json \
    --image-points "u1,v1;u2,v2;u3,v3;u4,v4" --table-points "0,0;0.3,0;0.3,0.3;0,0.3"

# 3. Identify the arm (NEVER sends motor commands) and fill in the worksheet:
terafold robot-scan --save-json runs/robot_scan/ports.json
terafold robot-info-template --out runs/robot_scan/robot_info_template.md

# 4. Export the trajectory to drive the arm via vendor software / a future adapter:
terafold export-trajectory --plan-json runs/demo_image/result.json \
    --out runs/demo_today/trajectory.csv
```

**Marker convention** (configurable with `--marker-map`): `red → top_left`,
`blue → top_right`, `green → bottom_left`, `orange/yellow → bottom_right`. HSV
blob detection reports exactly which colors are missing; if markers aren't all
found it does **not** pretend (`perception = marker_failed_fallback_classical`)
and real motion is refused. `--debug-markers` saves per-color masks.

**Real motion requires ALL of**: trusted perception above
`--confidence-threshold`, a calibrated homography, a known adapter with a
verified command backend, a workspace-valid trajectory, a completed dry-run
episode, and both `--enable-motion --i-understand-this-moves-hardware`. Any
missing requirement is listed and real motion is refused — overlay / JSON / plan
are still saved. (`GenericArmAdapter` is a dry-run-only safe shell for an
unidentified arm; it refuses real motion until a verified backend is wired.)

### Virtual SO-101 fold simulation

Demo the fold **virtually** — no physical arm powered on. A simulated end-effector
(and, with MuJoCo, an approximate 6-DOF arm) follows the planned trajectory and
renders a video. **SIMULATION ONLY — not calibrated to real hardware; no motor
commands are ever sent.**

```bash
# Plan from a photo (Claude / markers / model) and export the trajectory:
python3 -m terafold demo-image --image ~/Downloads/towel_demo_pic.png --mode claude \
    --overlay-out runs/demo_image/claude_overlay.png --save-json runs/demo_image/claude_result.json
python3 -m terafold export-trajectory --plan-json runs/demo_image/claude_result.json \
    --out runs/demo_image/trajectory.csv

# Install the sim extra (MuJoCo + video):
python3 -m pip install -e ".[sim]"

# Render the fold simulation:
python3 -m terafold sim-fold --plan-json runs/demo_image/claude_result.json --level ee-only \
    --out runs/sim/fold_ee.mp4
python3 -m terafold sim-fold --plan-json runs/demo_image/claude_result.json --level cloth-proxy \
    --out runs/sim/fold_cloth_proxy.mp4

# Virtual SO-101 arm (MuJoCo) with camera presets:
python3 -m terafold sim-fold --plan-json runs/demo_image/claude_result.json --level arm-ik \
    --view iso --out runs/sim/fold_arm_iso.mp4
python3 -m terafold sim-fold --plan-json runs/demo_image/claude_result.json --level arm-ik \
    --view top --out runs/sim/fold_arm_top.mp4
open runs/sim/fold_arm_iso.mp4
```

The MuJoCo scene uses a free camera **auto-fit** to the trajectory + table (with a
headlight + skybox so nothing is black) and draws debug objects: towel rectangle,
yellow fold crease, grasp/place markers, an EE trail, and an RGB frame triad.
`--view` presets: `iso` (default), `top`, `side`, `follow-ee`, `gripper`. If the
trajectory would fall outside the frame, a warning prints the camera position,
target, and bounds.

#### Realistic SO-101 arm (`--level so101-real`)

A much closer **5-DOF SO-101 approximation** (proportional link lengths, real
joint limits, a parallel-jaw gripper that opens/closes from the trajectory) with
distinct materials for links / gripper / towel / table / crease / markers and a
per-frame phase subtitle. TeraFold first looks for *official* SO-101 assets
(URDF/MJCF/meshes from the repo, `$TERAFOLD_SO101_MJCF`, or installed
LeRobot / MuJoCo Menagerie / `robot_descriptions`); if none are found it uses
this built-in approximation and says so.

```bash
# Single view:
python3 -m terafold sim-fold --plan-json runs/demo_image/claude_result.json \
    --level so101-real --view iso --out runs/sim/fold_so101_real_iso.mp4

# All four views -> fold_so101_real_{iso,top,side,gripper}.mp4:
python3 -m terafold sim-fold --plan-json runs/demo_image/claude_result.json \
    --level so101-real --all-views --out runs/sim/fold_so101_real.mp4

# Slow-motion review + exported PNG frames:
python3 -m terafold sim-fold --plan-json runs/demo_image/claude_result.json \
    --level so101-real --view side --slowmo --save-frames --out runs/sim/fold_so101_real_side.mp4
open runs/sim/fold_so101_real_iso.mp4
```

Flags: `--all-views` (iso/top/side/gripper), `--save-frames` (PNG export),
`--slowmo` (longer playback), `--waypoint-dots/--no-waypoint-dots` (planned
waypoint dots; on by default for `so101-real`). To use a real SO-101 model, point
`$TERAFOLD_SO101_MJCF` at an MJCF file and it will be picked up automatically.

#### Demo-quality folding (`--level cloth-physics-proxy`) — **best for demos**

The convincing fold: the SO-101 arm (driven by **Jacobian IK** so the gripper
actually reaches the cloth) grasps the towel's right edge, lifts, arcs over the
crease, and places it on the left. The towel is a **deforming grid mesh** (a
kinematic cloth proxy) — the grasped edge follows the gripper, the moving half
hinges over the crease with sag, cloth never goes below the table, and the folded
half rests on top with its **underside shaded differently**. Debug objects show
the active grasp point, gripper contact point, moving edge, crease, waypoint dots,
and an EE trail; a large **phase subtitle** (pregrasp→…→inspect) tracks progress;
and the video ends with a **success/fail evaluation** overlay (also written to the
meta JSON: `fold_visually_successful`, `final_edge_error_m`, `crossed_crease`,
`settled_on_target_side`).

```bash
# Best-looking single-view demo:
python3 -m terafold sim-fold --plan-json runs/demo_image/claude_result.json \
    --level cloth-physics-proxy --view demo --slowmo --out runs/sim/fold_best_demo.mp4

# 2x2 review grid (top / side / contact / demo):
python3 -m terafold sim-fold --plan-json runs/demo_image/claude_result.json \
    --level cloth-physics-proxy --view split --slowmo --out runs/sim/fold_review_grid.mp4

# Close-up on the gripper/cloth contact, slow:
python3 -m terafold sim-fold --plan-json runs/demo_image/claude_result.json \
    --level cloth-physics-proxy --view contact --slowmo --out runs/sim/fold_contact_slowmo.mp4
open runs/sim/fold_best_demo.mp4
```

Fold-judging camera presets: `--view demo` (best 3/4 angle), `contact` (gripper
close-up), `cloth` (cloth deformation), `split` (2x2 grid). An experimental
`--level mujoco-cloth` probes MuJoCo deformable flex; if it isn't practical it
prints *"MuJoCo deformable cloth unavailable or unstable; using
cloth-physics-proxy recommended."* and falls back to `cloth-physics-proxy` (no
faked success).

Levels:
- `ee-only` (default) — end-effector sphere follows the path; **always works**
  (2D top-down renderer, no MuJoCo needed). Saves `runs/sim/fold_ee.mp4`.
- `cloth-proxy` — the towel proxy visually folds along the crease as the gripper
  crosses the fold line.
- `arm-ik` — a simple 6-DOF arm follows via MuJoCo (needs `pip install -e ".[sim]"`;
  without it you get a clear install message).
- `so101-real` — the realistic 5-DOF SO-101 approximation (see above).
- `cloth-physics-proxy` — **best demo**: SO-101 arm (IK) + deforming cloth grid
  mesh + contact debug + success evaluation (see above).
- `mujoco-cloth` — experimental; falls back to `cloth-physics-proxy`.

Each run also writes `<out>.meta.json` (level, view, renderer, frames, fps, fold
direction, SO-101 asset report, `control: none (simulation only)`,
`motor_commands_sent: 0`). Use
`--use-mujoco` to force the 3D MuJoCo scene for any level. If MuJoCo is missing,
`ee-only`/`cloth-proxy` fall back to the 2D renderer so the demo still produces a
video.

## 13b. Physical bus-servo arm — safe real-motion workflow

For a custom **7-DOF bus-servo arm** on a **Waveshare Bus Servo Adapter (A)**
(DC 9–12.6V, USB-C serial). Everything defaults to **dry-run**. Real motion is
**double-gated** (`--enable-motion` + `--i-understand-this-moves-hardware`) **and**
requires a *confirmed* bus-servo protocol — and there is **no verified protocol in
the repo**, so TeraFold refuses every read/move with *"Waveshare bus servo
protocol not confirmed; refusing to move."* (it never guesses packet bytes). The
config (`configs/robots/physical_7dof_waveshare.yaml`) carries conservative
placeholder limits only.

**Safe to run first (read-only, never moves):**
```bash
python3 -m terafold robot-scan --save-json runs/robot_scan/ports.json
python3 -m terafold robot-probe --robot physical_7dof_waveshare --port auto --read-only
python3 -m terafold servo-scan --robot physical_7dof_waveshare --port auto --read-only   # refuses (no protocol)
python3 -m terafold robot-estop --robot physical_7dof_waveshare --port auto              # always safe
```

**Dry-run ghost fold (prints the air path ≥10 cm above the table, no motion):**
```bash
python3 -m terafold real-ghost-fold --robot physical_7dof_waveshare \
    --plan-json runs/demo_image/claude_result.json --height-clearance-m 0.10 --speed slow --dry-run
```

**Manual teach + replay (joint-space, no kinematics needed):**
```bash
python3 -m terafold teach-ghost-fold --robot physical_7dof_waveshare --out data/real_demos/ghost_fold_001.json
python3 -m terafold replay-joint-demo --robot physical_7dof_waveshare \
    --demo data/real_demos/ghost_fold_001.json --speed very_slow --dry-run
```

**Commands that *attempt* real hardware (currently refuse at the protocol gate):**
```bash
# Tiny one-servo nudge (moves one servo a few degrees, then returns):
python3 -m terafold servo-nudge --robot physical_7dof_waveshare --port auto \
    --servo-id 1 --delta-deg 3 --speed slow --enable-motion --i-understand-this-moves-hardware
# Real ghost fold / real replay: add --enable-motion --i-understand-this-moves-hardware
```
`servo-nudge` refuses deltas >5° unless `--dangerous-allow-larger-motion`, and
refuses to move blind unless `--allow-open-loop-nudge`. Image/table-based real
motion (`real-image-fold`) refuses without full table calibration. Every real
command logs to `runs/real_motion_logs/`. **To enable motion later:** confirm the
servo protocol and wire a verified `command_backend` into
`WaveshareBusServoAdapter` — only then does `protocol_confirmed` flip to True.

## 14. Record real demos

```bash
# Mock (safe, today):
terafold record-demo --task configs/task_fold_towel_half.yaml --robot mock --camera webcam

# Real hardware (only when ready, with both motion flags):
terafold record-demo --task configs/task_fold_towel_half.yaml --robot learm --camera webcam \
    --enable-motion --i-understand-this-moves-hardware
```

Episodes are saved under `data/episodes/<task>/episode_NNNNNN/` with frames,
observations, actions, keypoints, plan, safety log, and a result label.

## 15. Export to LeRobot

```bash
terafold export-lerobot --episodes data/episodes/fold_towel_half_v0 \
    --out data/lerobot/terafold_towel_half_v0
```

Writes a LeRobot-style dataset (`observation.images.top`, `observation.state`,
`action`, `timestamp`, `frame_index`, `episode_index`, `task`) and prints the
exact ACT/SmolVLA training commands.

## 16. Train ACT

```bash
terafold train-act --dataset data/lerobot/terafold_towel_half_v0 --out runs/act_v0
```

## 17. Train SmolVLA (later)

```bash
terafold train-smolvla --dataset data/lerobot/terafold_towel_half_v0 --out runs/smolvla_v0
```

> **On public datasets:** other-embodiment datasets (e.g. `lerobot/...fold...`)
> are useful for **visual pretraining, loader testing, success scoring, and
> representation learning** — *not* as drop-in policies. Their action/state space
> rarely matches our arm. Use `terafold inspect-hf-dataset --repo-id <id>` to
> check before assuming compatibility.

### Public dataset sources (streaming-first)

TeraFold ships a curated registry of public cloth/folding datasets and **never
downloads huge datasets by default** — everything streams or is bounded by an
explicit cap; a full download requires `--allow-large-download`.

```bash
terafold data-sources                                              # curated registry + how to use each
terafold inspect-hf-dataset --repo-id observabot/so101_cloth_folding1 --streaming   # metadata, no download
terafold sample-hf-dataset  --repo-id REPO --max-samples 500 --out data/public_samples/NAME
terafold cache-hf-subset    --repo-id REPO --max-episodes 20 --out data/cache/NAME
terafold import-hf-lerobot  --repo-id REPO --max-episodes 20 --out data/public/NAME
```

Each registry entry records modality, estimated size, embodiment (SO-101 / Aloha
/ Unitree H1 / human / image-only / synthetic), **direct policy compatibility
with our LeArm (yes/no/maybe)**, and recommended use (perception, fold-success
scoring, visual pretraining, ACT testing, bimanual reference — or *not for direct
rollout*). Sampled images are auto-converted into TeraFold's perception format.
If `datasets`/`huggingface_hub` are missing or you're offline, the commands fail
gracefully with the exact install command.

## 18. Evaluation metrics

`terafold score-fold --before before.jpg --after after.jpg` and
`terafold benchmark --episodes <dir>` report:

* **overlap_ratio** — IoU of the idealized vs observed folded shape
* **corner_error_m** — best-matched corner distance
* **edge_alignment_error** — how well the moving edge lands on the target edge
* **visible_area_ratio** — after / before footprint (~0.5 for a half fold)
* **wrinkle_score** — boundary/texture roughness (0 = flat)

A fold "succeeds" when all thresholds in the task config's `success:` block pass.

## 19. Roadmap

* **V0** — towel half-fold, one arm *(this release)*
* **V1** — towel two-fold
* **V2** — T-shirt side fold
* **V3** — full shirt fold
* **V4** — bimanual folding
* **V5** — hotel towel / linen manipulation
* **V6** — hotel housekeeping cart manipulation

---

## Project layout

```
terafold/
  cli.py                 # Typer CLI (all commands)
  config/                # pydantic schemas, YAML loading, validation
  math/                  # frames, geometry (reflection!), transforms, interpolation
  camera/                # base, opencv, mock, homography, calibration
  vision/                # synthetic data, masks, markers, keypoint + success models
  physics/               # cloth state, fold geometry, quasi-static heuristics, quality
  planning/              # fold task, geometric planner, trajectory, residual model
  robot/                 # base, mock, LeArm shell, SO-101, bimanual, safety
  data/                  # episode schema, recorder, replay, LeRobot export, HF inspect/stream, source registry
  learning/              # ACT / SmolVLA wrappers, safe rollout
  eval/                  # fold scoring, benchmarking
configs/                 # task / robot / camera / workspace YAML
scripts/                 # numbered runnable helpers (00..10)
tests/                   # pytest suite
```

## Design rules (enforced)

* Stage-0 runs with **numpy only**; torch/opencv/lerobot are lazy optional extras.
* Importing `terafold` never requires a heavy dependency.
* Physical motion is **off by default** and double-gated.
* No fabricated vendor protocols; no assuming public data controls our arm.
* Geometry is a prior — learned models correct it.

## License

Apache-2.0.
