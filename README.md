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
detector if markers aren't all visible); `--mode keypoints` uses the learned
model (`--checkpoint`).

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
