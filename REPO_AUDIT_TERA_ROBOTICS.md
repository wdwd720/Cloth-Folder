# TeraFold / Tera Robotics — Repository Audit

Generated: 2026-07-05 — read-only audit (branch `main`, HEAD `2a16901`). No source files were modified to produce this report; only the four `TERA_*` / `REPO_AUDIT_*` markdown files were created.

Companion reports: [`TERA_INTEGRATION_PLAN.md`](TERA_INTEGRATION_PLAN.md) (merging the Brev Isaac work), [`TERA_GIT_ARTIFACT_POLICY.md`](TERA_GIT_ARTIFACT_POLICY.md) (git hygiene / artifacts / secrets), [`TERA_NEXT_STEPS_V6.md`](TERA_NEXT_STEPS_V6.md) (the V6 reachability milestone).

**How this was produced:** 22 read-only recon agents statically analyzed every package, script, test, doc, dataset, and run artifact; their findings were cross-checked and are cited by `file:line` throughout. Some model-quality and history-count claims are self-reports that were spot-checked but not independently reproduced (marked where relevant). Nothing was executed — no pytest, no training, no simulation.

---

## Table of contents

1. [Executive summary](#1-executive-summary) — read this first
2. [Honesty ledger](#2-honesty-ledger-provenpartialunproven) — what is actually proven
3. [Repository overview](#3-repository-overview)
4. [Directory map](#4-directory-map)
5. [Robotics / Isaac / SO-101 / YAM inventory](#5-robotics--isaac--so-101--yam-inventory)
6. [Model & perception work](#6-model--perception-work)
7. [Training infrastructure](#7-training-infrastructure)
8. [Agents / automation / scripts](#8-agents--automation--scripts)
9. [Consolidated risks](#9-consolidated-risks)

---

## 1. Executive summary

**What this repo is.** TeraFold is an honest-by-design, mostly numpy cloth-folding robotics stack. Its public framing (README) is a "trainable, physically-grounded cloth-folding robot stack" aimed at hotel-linen manipulation. In ~18 commits over roughly 6 days (2026-06-25 → 2026-07-01) it passed through **five hardware/strategy generations**: LeArm 6-DOF → SO-101 (LeRobot) → a custom 7-DOF Waveshare bus-servo arm → a platform/towel-perception expansion → a dual-YAM + MolmoAct2 "shadow-mode" pivot (2026-07-01). The Brev Isaac work (dual SO-101 + PhysX cloth) is effectively a **sixth** direction that exists nowhere in this repo yet.

**What is proven to work (software, no hardware):** the numpy Stage-0 pipeline (synthetic towel → keypoints → geometric fold plan → mock/dry-run rollout → episode recording → LeRobot-shaped export); a serious geometric fold planner; dependency-free fold-quality metrics; a full towel-image dataset/labeling pipeline (synthetic + OpenAI-generated + Claude/OpenAI vision pseudo-labeling with human/auto approval gates); a laptop-side plan visualizer (numpy 2D + optional MuJoCo renderer); and a hardware-free dual-YAM shadow-mode scaffold that logs MolmoAct2-style predictions without ever commanding motors.

**What is NOT proven — and must not be claimed:**

- **No autonomous towel folding exists anywhere** — not locally, not on Brev. Locally the maximum physical event ever recorded is a single in-air "ghost" sweep of **one servo, three position writes, gripper open, no towel contact** (2026-06-26). On Brev, a *particle-space assisted* fold succeeded (width 0.68 → 0.339), but *pure robot-contact* folding **failed** (width stayed 0.68) because the jaw never got within ~0.406 m of the towel edge — a reachability/placement problem.
- **No real-world towel perception.** The trained "master" YOLO-pose detector saw **0 real photos** (2000 synthetic + 39 OpenAI-generated) and scores **avg keypoint confidence 0.0093 on a real photo** by its own saved eval. Zero real-image labels have ever been approved.
- **SO-101 is not a supported robot here** — only a dry-run adapter shell (`NotImplementedError` on all real paths), a config template, and an *approximate, uncalibrated* MuJoCo model. No URDF, no joint map.
- **The MolmoAct2 policy has never run for real locally** — only a mock adapter; the checkpoint API is unverified.
- **The local Isaac scene is visual-only** (rigid box towel, placeholder arms, dual-YAM), single commit, and there is no evidence it was ever built into a USD.

**What exists on Brev (from external context only — no local trace):** a verified rectangular PhysX cloth "towel v2" (69×39, 2691 particles); a dual-SO-101 scene (action_dim 12); a working particle-space assisted fold; a failed pure-contact fold with V5 diagnostics pointing at reachability (~0.406 m gap). **The Brev VM is a single point of failure with zero pointer, sync script, or backup in this repo.**

**The three things that must happen next (in order):**

1. **Fix git hygiene before touching anything** — an uncommitted `.gitignore` edit deleted the `/data/` rule, leaving **8.8 GB of `data/` untracked *and* un-ignored** (`git add -A` would try to commit gigabytes). Separately, `.git` has bloated to **1.5 GB, of which ~1.45 GB is unreachable garbage** reclaimable by `git gc`. See [`TERA_GIT_ARTIFACT_POLICY.md`](TERA_GIT_ARTIFACT_POLICY.md).
2. **Resolve the embodiment fork** — the whole local YAM stack is dual-YAM / action_dim 14; Brev is dual-SO-101 / action_dim 12; a third local LeRobot export is 13/7. Pick a canonical embodiment before merging any Brev code. See [`TERA_INTEGRATION_PLAN.md`](TERA_INTEGRATION_PLAN.md).
3. **Run V6 reachability diagnostics on Brev** — the blocker is geometry (arm base / towel placement / gripper reach), *not* model training. See [`TERA_NEXT_STEPS_V6.md`](TERA_NEXT_STEPS_V6.md).

**What should NOT be done yet:** no new policy training (nothing is reachable to fold); no real-hardware contact; no repo-wide refactor mid-integration; no updating the README to claim folding capability; no reuse of local sim "fold success" numbers as evidence (the local sim stretches the SO-101 arm up to 1.6× and its cloth cannot fail physically).

---

## 2. Honesty ledger (PROVEN/PARTIAL/UNPROVEN)

The single most important section for anyone summarizing this project. Verdicts are grounded in the actual code/artifacts, not docs or self-reports.

| # | Claim | Verdict | Evidence |
|---|---|---|---|
| A | Autonomous robot-contact towel folding exists (local or Brev) | **UNPROVEN / FALSE** | No contact code path is reachable locally (`capabilities.py:300-309` permanently locks levels 6-8; `real_image_ghost.py:145` hardcodes `contact_fold_allowed: False`). Brev pure-contact fold failed (width unchanged 0.68). |
| B | A real robot has moved | **PARTIAL (barely)** | Exactly one real session: `runs/real_motion_logs/real_image_ghost_fold_20260626_175358.jsonl` — 1 `protocol_confirm`, 3 `write_position` on servo ID 6 (1246→1166→1246), in-air, gripper open, no contact. All 62 later servo scans found 0 servos. ~225 of 228 "moved" events are mock (`port=/dev/null`). |
| C | Stage-0 sim is numpy-only, no physics engine | **PROVEN** | `terafold/sim/__init__.py` states it; `fold_sim.py` MJCF sets `gravity="0 0 0"`, EE is a mocap body, no actuators/contacts; MuJoCo is a renderer+IK tool only; `mujoco-cloth` unconditionally falls back (`fold_sim.py:1150-1157`). |
| D | YAM MolmoAct2 "shadow mode" never commands motors | **PROVEN** | `terafold/yam/safety.py:36` `HARDWARE_EXECUTION_IMPLEMENTED = False`; `shadow.py:80` `assert_no_hardware_execution(...)`; grep of `terafold/yam/*.py` finds no serial/socket/subprocess/SDK import; every output stamped `hardware_commanded: false`. |
| E | Trained towel-corner perception works on real images | **UNPROVEN / FALSE for the model** | `runs/towel_pose/real_IMG_0500_result.json` avg keypoint confidence **0.0093** on a real photo; `data/towel_master_pose_v0/manifest.json` shows `real_user: 0`. Only classical/marker/paid-VLM fallbacks touch real images, none validated. |
| F | Local Isaac code is executable on this Mac | **PARTIAL** | The *generator* runs (stdlib only) and the generated `build_isaac_scene.py` degrades gracefully without Isaac; but building the USD requires the Isaac Sim interpreter (container `isaac-sim:6.0.1`). No `.usd` exists anywhere locally. |
| G | Real hardware motion happened | **PARTIAL** | See row B — one in-air single-joint sweep, 2026-06-26. Nothing since. |
| H | The "double-gated motion" safety mechanism works | **PROVEN (but architecturally drifted)** | Two-flag gate (`safety.py:207-220`) + verified-protocol gate (`waveshare_sms_sts_backend.py:201-269`) both hold. But the advertised central `MotionGate`/`motion_core` chokepoint is **dead in production** (only tests call it); the real write path inlines its own checks (`real_image_ghost.py:247-268`). |
| I | "Today the system reaches Level 4" (README) | **PARTIAL / STALE** | The ladder needs a joint map at Level 3; **no `*_joint_map.yaml` exists** and `runs/calibration/` is empty, so a probe today reports < Level 3. The ghost fold was a one-time event, not a reproducible state. |
| J | Particle-space assist == robot-contact autonomy | **FALSE (do not conflate)** | Brev's success (0.68→0.339) moved *particles directly*; the robot-contact attempt with the same towel failed. These are different claims. |

---

## 3. Repository overview

**Purpose.** A cloth-folding robot stack ("fold a small towel in half, right-to-left" is task V0), built to be *trainable and physically grounded* rather than a hard-coded demo. README roadmap: V0 towel half-fold → V1 two-fold → V2/V3 shirts → V4 bimanual → V5 hotel linen → V6 housekeeping cart. `deep-research-report.md` frames towel folding as the "first technical wedge," not the product.

**Languages / frameworks.** Python only (211 of 241 tracked files are `.py`; Python 3.13). Core deps are deliberately minimal: numpy, scipy, pyyaml, typer, pydantic. Everything heavy is a **lazy optional extra** in `pyproject.toml`: `ml`/torch, `vision`, `lerobot`, `molmoact` (torch+transformers+accelerate), `yolo` (ultralytics), `sim` (mujoco), `rerun`, `claude` (anthropic), `openai`, `kaggle`, `video`. There is **no lockfile** (only `>=` ranges) and **no LICENSE file** despite Apache-2.0 declared in README/pyproject.

**Major modules** (all under `terafold/`):

| Package | Role | State |
|---|---|---|
| `cli.py` (128 KB) | 89 Typer subcommands spanning all four hardware generations | The real entry surface; sprawling |
| `robot/` | Adapters (mock/LeArm/SO-101/Waveshare/bimanual) + safety gates + ghost fold | Only Waveshare SMS/STS can move hardware |
| `physics/` | Pure-numpy cloth-state, fold geometry, fold-quality metrics | Heuristics + geometry, **no simulator** |
| `planning/` | Geometric pick-lift-arc-place fold planner + clamped residual model | Solid, self-contained, single-EE only |
| `sim/` | numpy 2D + optional MuJoCo **visualizer** (not physics) | Renders plans; masks reachability (1.6× stretch) |
| `towel/` (25 mods) | Real-image-first towel dataset + labeling + YOLO-pose pipeline | Well-engineered; perception only |
| `vision/` | Stage-0 8-keypoint U-Net + classical fallbacks + Claude labeler | Synthetic-trained; parallel to `towel/` |
| `calibration/`,`camera/`,`kinematics/`,`math/` | Homography, camera abstraction, DH FK / toy IK, transforms | Scaffolds; no validated arm model |
| `learning/`,`eval/`,`data/` | LeRobot command builders, fold/perception metrics, episode schemas | Trains nothing locally; metrics reusable |
| `yam/` | Dual-YAM config, MolmoAct2 adapter, shadow policy, Isaac scene generator | Hardware-free, shadow/log-only |

**What it currently supports (proven):** dry-run/mock folding pipeline, synthetic data generation, towel dataset labeling (paid VLM + gates), plan visualization videos, read-only servo scans, and the one-time in-air ghost sweep on the Waveshare arm. **What is merely planned:** ACT/SmolVLA training on real demos, SO-101 bring-up, calibrated hover, soft contact (90-day target), bimanual folding.

---

## 4. Directory map

```
Cloth-Folder/
├── terafold/                 SOURCE — the Python package (211 .py). Commit and keep.
│   ├── cli.py                89 Typer commands (4 hardware eras); 128 KB monolith.
│   ├── demo.py               demo-today/demo-image; real motion gate hardcoded off (has_verified_backend=False).
│   ├── perception.py         perception mode dispatch (markers/claude/model/classical).
│   ├── robot/ (23 .py)       adapters + safety.py/safety_gate.py/safety_state.py/capabilities.py/motion_core.py;
│   │                         waveshare_sms_sts_backend.py = ONLY real hardware backend; real_image_ghost.py = only real write path.
│   ├── physics/ (5)          cloth_state, fold_geometry, fold_quality (reusable IoU/edge metrics), friction, quasi_static.
│   ├── planning/ (7)         fold_plan, trajectory (10 phases), residual_model (±5cm clamped), towel_half_fold.
│   ├── sim/ (5)              cloth.py (kinematic proxy), fold_sim.py (renderer, 1.6× reach stretch), ghost_preview, so101.py (0.366 m reach).
│   ├── towel/ (25)           dataset/label/merge/yolo pipeline; claude+openai pseudolabel; multipass QC.
│   ├── vision/ (13)          synthetic_cloth, keypoint U-Net, classical mask/marker, claude_labeler, imageio (pure-PNG codec).
│   ├── calibration/,camera/,kinematics/,math/   calibration scaffolds; NO validated SO-101/YAM kinematics.
│   ├── learning/ (3)         train_act/smolvla WRAPPERS (print lerobot commands), rollout_policy.
│   ├── eval/ (7)             fold_metrics, perception_metrics, score_fold, benchmark, report — Brev-reusable.
│   ├── data/ (13)            3 episode schemas, 2 LeRobot exporters, hf_streaming, sources registry.
│   └── yam/ (8)              config (action_dim 14), safety (execution=False), dataset, molmoact2_smoke, shadow, isaac_scene.
├── scripts/                  SOURCE — 00–10 numbered pipeline (frozen V0, doc-orphaned); 09/10 are the only .sh.
├── configs/                  SOURCE (small) — YAML: mock/webcam/so101_template/yam_dual_reference + robots/physical_7dof_waveshare.
├── tests/ (~55 files)        SOURCE — ~595 hermetic mock/synthetic tests; zero touch hardware/GPU/network.
├── docs/ (11 .md)            NOTES — split across two strategy eras (see §5); YAM_ISAAC_SIM/SPRINT_REPORT/NEXT_90_DAYS matter most.
├── sim/yam_dual_towel_scene/ GENERATED (committed) — visual-only dual-YAM Isaac scene package (rigid towel). Regenerable.
├── README.md (37 KB)         NOTES — STALE: no YAM/MolmoAct2/Isaac mention; §13b/§13c self-contradiction; LeArm still "now".
├── goal.md, deep-research-report.md  NOTES — YAM sprint spec; pre-YAM strategy report.
├── pyproject.toml            SOURCE (config).
│
├── data/  (8.8 GB)           UNTRACKED and (after uncommitted .gitignore edit) UN-IGNORED. Mostly regenerable; 105 HEIC photos precious. DO NOT COMMIT.
├── runs/  (77 MB)            gitignored. 6 trained weight files + the one real-motion log + mock/test pollution. DO NOT COMMIT (but back up weights).
├── vendor/waveshare/ (35 MB) gitignored. Windows venv + scservo_sdk (real-hardware runtime dep) + local bring-up scripts. Keep untracked; rescue the SDK+scripts.
├── *.tar.gz  (~1.9 GB loose) gitignored (*.tar.gz). Deletable dups of data//git — except the raw-photo tarball (precious).
└── yolo11n-pose.pt (6 MB)    gitignored (*.pt). Stock Ultralytics COCO checkpoint; re-downloadable.
```

**Old / unused / generated to be aware of:** `runs/pose/runs/towel_pose/...` (double-nested by a cwd quirk); `data/towel_real_v0`, `raw_towel_images`, `*_candidates_*` (empty/no-op filter runs); `terafold.egg-info/` (build metadata, **not** tracked — good); `.pytest_cache/`, `.ruff_cache/`, `__pycache__/` (ignored). `sim/yam_dual_towel_scene/` is generator output committed to git (drift risk — see §5).

---

## 5. Robotics / Isaac / SO-101 / YAM inventory

### Isaac Sim (local footprint is tiny: 6 files, one commit)
- `terafold/yam/isaac_scene.py` — **source**; the scene *generator* (`_BUILD_SCRIPT` template + `isaac_cloud_commands()`). Never imports Isaac. **Integration: high** (the canonical pattern to fold Brev learnings into). **Conflicts with Brev: yes** (dual YAM, rigid towel).
- `sim/yam_dual_towel_scene/{build_isaac_scene.py,scene_config.json,README.md}` — **generated (committed)**; byte-identical to the template. `build_isaac_scene.py` is the only file with real Isaac API usage (`SimulationApp`, `pxr`, `UsdPhysics`), all lazy; without Isaac it exits 2. **Only `UsdPhysics.Scene.Define` — no PhysX/particle/cloth/articulation.** **Conflicts: yes.**
- `docs/YAM_ISAAC_SIM.md` — **notes**; "visual digital twin only… deformable cloth explicitly out of scope this sprint." This is exactly what Brev went beyond, for a different robot. **The word "PhysX" appears nowhere in the repo; no `.usd*` file exists on disk.**

### SO-101 (template/aspiration, not a supported robot)
- `terafold/robot/so101_adapter.py` — **source**; dry-run shell. Real `connect()/get_observation()/send_action()` all raise `NotImplementedError` (`:72-77,93-96,111-114`). **Integration: high** — the seam Brev SO-101 control would fill; today moves nothing.
- `terafold/demo.py:276,410` — **source**; `has_verified_backend=False` hardcoded → `demo-today --robot so101 --enable-motion` can never move hardware (tests assert refusal).
- `configs/robot_so101_template.yaml` — **config**; `port: null`, workspace 0.70×0.50×0.25 m (untested vs real reach).
- `terafold/sim/so101.py` — **source (sim)**; MJCF approximation, `nominal_reach() = 0.366 m`, "SIMULATION ONLY." **Independently corroborates the Brev V5 failure** (0.366 m < the 0.406 m jaw-to-edge gap ⇒ placement/reach problem). **No URDF/joint map exists anywhere.**
- `data/public_samples/so101_cloth_folding1_{100,1000}` — **dataset**; real SO-101 folding frames from HF `observabot/so101_cloth_folding1`, weak auto-labels; used to fine-tune `runs/keypoints_so101_ft_v0` (perception, not policy).

### YAM / MolmoAct2 (2026-07-01 pivot; hardware-free shadow mode)
- `terafold/yam/config.py` — **source**; `action_dim = 2×(6 joints + 1 gripper) = 14`, `norm_tag yam_dual_molmoact2`, model `allenai/MolmoAct2-BimanualYAM`. Refuses `autonomous_execution_enabled: true` at load. **Conflicts with Brev: yes** (Brev is action_dim 12).
- `terafold/yam/shadow.py` — **source**; THE shadow policy — replays an episode through the adapter and logs L2 diffs. **Only writes JSON/`.rrd`; no motor path exists behind the gates at all.**
- `terafold/yam/molmoact2_smoke.py` — **source**; adapter hedges between two unverified checkpoint APIs; `trust_remote_code=True` unpinned by default. Only the **mock** adapter has ever run (`runs/yam_shadow_smoke/metadata.json` shows `adapter.kind: "mock"` while the `model` field misleadingly names the real checkpoint).
- `configs/yam_dual_reference.yaml` — **config**; every physical number self-labeled PLACEHOLDER; all asset paths null.

### Towel / fold core (planner + metrics; no contact)
- `terafold/planning/fold_plan.py`, `trajectory.py` — **source**; geometric pick-lift-arc-place, 10 phases, table-frame waypoints, JSON-serializable. **Integration: high** — this is the piece Brev lacks (no planner on Brev). Needs a table→Isaac-world frame transform that does not exist yet.
- `terafold/physics/fold_quality.py`, `terafold/eval/fold_metrics.py` — **source**; pure-numpy IoU/edge/area/crossed-crease metrics on `(N,2)` corner arrays. **Directly reusable to score Isaac particle-space folds.** Note: **no width-reduction metric exists anywhere** (grep: zero hits) — the Brev headline metric has no local counterpart (a ~5-line addition, must be written).

### Rerun / visualization
- `terafold/yam/rerun_logger.py` + `shadow.py` — **source**; real Rerun integration, but **never actually run** (zero `.rrd` in repo; four Brev `.rrd` traces sit unread in `~/Downloads`). Every log call degrades to a warning by design. The working viz is numpy PNG overlays + optional MuJoCo mp4s. **Nothing local can visualize Isaac/PhysX results.**

### PyTorch / cloud
- See §7. Four tiny local training loops (all perception/plan-scoring, CPU-only, no MPS); **no folding-policy training code locally.** **Zero Modal/RunPod/Brev/ssh code** — cloud is docker-command *printers* + manual tarball handoffs.

---

## 6. Model & perception work

Every model and pipeline, with honest status:

| Asset | What it is | Status | Real-image capable? |
|---|---|---|---|
| `runs/towel_pose/yolo_towel_pose_master_v0/weights/best.pt` (5.6 MB) | YOLO11n-pose, 4-corner towel detector, 100 epochs on a cloud GPU | **Trained on 2000 synthetic + 39 OpenAI + 0 real.** 0.995 val mAP on synthetic-dominated split | **No** — 0.0093 avg keypoint confidence on a real photo |
| `runs/pose/.../yolo_towel_pose_synth_640_v1/weights/best.pt` | YOLO11n-pose, synthetic-only, 50 epochs local CPU (~38 h) | Baseline | No (synthetic only) |
| `runs/keypoints_v0/model.pt` (1.76 MB) | 8-keypoint heatmap U-Net, synthetic, 10 epochs | Superseded by YOLO track | Unvalidated on real |
| `runs/keypoints_so101_ft_v0/model.pt` (1.76 MB) | Same arch, 5-epoch fine-tune on weak-labeled SO-101 HF frames | Perception experiment | Unvalidated; weak labels mixed quality |
| Claude/OpenAI VLM labelers (`towel/claude_pseudolabel.py`, `openai_multipass.py`) | Paid vision pseudo-labelers with 6-stage QC + geometry gates | Work in principle; **0 real-image labels approved** ever | Yes (paid API), but corner/verify prompts hardcode a "beige/white striped towel" target |
| Classical (`vision/cloth_mask.py`, `marker_detector.py`) | numpy mask+min-area-rect; HSV corner-sticker detection | Dependency-free; used by the original ghost demo | Yes on cooperative images / tagged towels; accuracy unvalidated |
| `success_model.py`, `plan_critic.py`, `residual_model.py` | Tiny MLP / logreg scaffolds | **Self-referential** — trained to reproduce the geometric labels they consume | N/A — never cite as learned results |

**How this helps the clean-towel / SO-101 Isaac pipeline:** the labeling QC machinery and `towel/yolo_runtime.py` inference harness are reusable to label **Isaac-rendered** frames (Isaac gives exact ground-truth corners, which finally makes `eval/perception_metrics.py` meaningful). The 4-corner `towel/schema.py` is the sane label contract to standardize on. **But the detector cannot yet provide corner/edge detection for real camera data** — the "Layer C" real dataset the docs call "the thing that actually matters" is effectively empty (105 photos ingested, 0 usable labels).

---

## 7. Training infrastructure

**Local training loops (4, all tiny, all Mac-CPU):** keypoint U-Net (`vision/train_keypoints.py`, `cuda-if-available` else CPU — **no MPS**), success MLP, plan-residual MLP, plan-critic logreg. **No policy learning (ACT/SmolVLA/RL/BC) trains locally** — `learning/train_act_wrapper.py` / `train_smolvla_wrapper.py` only *build and print* `python -m lerobot.scripts.train ...` commands (subprocess only if `run=True` and lerobot is importable). lerobot itself is an optional extra with no evidence of being installed.

**Cloud infrastructure: none exists as code.** Zero Modal, zero RunPod SDK, zero Brev/ssh/rsync references anywhere (verified across tracked files + git log). The entire "cloud workflow" is: (a) `yam-print-isaac-cloud-commands` printing `docker` commands for `nvcr.io/nvidia/isaac-sim:6.0.1`; (b) `print-towel-training-command` printing a `yolo pose train` line; (c) the lerobot command printers; (d) manual `tar`/`scp` handoffs in docs. **The only cloud-GPU execution reflected locally is one YOLO towel-pose training** (`runs/towel_pose/.../args.yaml` shows `/workspace/...` paths, batch 64, 100 epochs) whose weights were copied back.

**What runs where (recommended):**

| Work | Mac (local) | Brev (Isaac GPU) | Cloud GPU (Modal/RunPod) |
|---|---|---|---|
| Numpy Stage-0, planning, metrics, dataset labeling | ✅ | — | — |
| YOLO towel-pose training | slow (CPU) | possible | ✅ (documented handoff) |
| Isaac scene build + PhysX cloth + reachability (V6) | ❌ (no Isaac) | ✅ | — |
| MolmoAct2 inference / shadow eval | mock only | — | ✅ (CUDA pod) |
| ACT/SmolVLA policy training (V7+) | ❌ | — | ✅ |

**Reusable for V4–V7:** `eval/fold_metrics.py` + `eval/perception_metrics.py` + `physics/fold_quality.py` (pure numpy, unit-agnostic — feed Isaac particle corners directly); the `data/recorder.py` episode format + `eval/benchmark.py` aggregator (add a `width_reduction` key); the `learning/rollout_policy.py` policy-agnostic loop as an Isaac rollout driver template. **Liabilities:** three incompatible episode schemas (7/13 mock, 14 YAM, 12 Brev) and two divergent LeRobot exporters whose output is an *unverified* scaffold (never load-tested as a real `LeRobotDataset`).

---

## 8. Agents / automation / scripts

- **`.claude/settings.local.json`** — a single Bash allow-rule (a yaml+typer import check). No secrets. `.claude/scheduled_tasks.lock` is a stale session lock.
- **No CI** — no `.github/` workflows, no Makefile.
- **Shell scripts** — only `scripts/09_train_act.sh` (prints the lerobot command; `--run` launches locally) and `scripts/10_rollout_safe.sh` (triple-gated local rollout; actually re-plans classically, does **not** roll out a trained policy). The numbered `scripts/00-10` are thin subprocess wrappers around 11 of the ~89 CLI commands — frozen at V0, doc-orphaned, and drifted (e.g. `07→08` default chaining is broken).
- **No email / contact / outreach automation exists** — searched explicitly; there is none. Everything automation-adjacent is robotics-related.
- **AI-agent tooling in-code:** Claude/OpenAI are used only as *vision labeling* backends (`terafold/towel/*pseudolabel*`, `terafold/vision/claude_labeler.py`), reading keys from env only. No Codex/agent framework.

---

## 9. Consolidated risks

**Blocking / high severity**
1. **`data/` (8.8 GB) is untracked AND un-ignored** after the uncommitted `.gitignore` edit removed `/data/`. One `git add -A` stages gigabytes. → [`TERA_GIT_ARTIFACT_POLICY.md`](TERA_GIT_ARTIFACT_POLICY.md).
2. **Embodiment fork (14 vs 12 vs 13/7)** will silently break episode round-tripping and shadow comparisons at integration. → [`TERA_INTEGRATION_PLAN.md`](TERA_INTEGRATION_PLAN.md).
3. **Brev VM is a single point of failure** with no pointer/sync/backup in the repo — including the only PhysX cloth work and the four `.rrd` traces stranded in `~/Downloads`.
4. **Only copies of critical assets are unbacked-up:** 105 raw HEIC photos (`data/raw_towel_real_v1/`, precious), 6 trained weight files (gitignored `runs/`), and the Waveshare bring-up scripts + `scservo_sdk` (gitignored `vendor/`). A `rm -rf runs/`/`vendor/` or disk loss erases them.

**Honesty hazards (dishonest-claim traps)**
5. Directory/file names oversell: `real_motion_logs/` (99% mock), `fold_so101_real_*.mp4` (kinematic animation), `fold_visually_successful: true` (a metric that cannot fail physically), "master" model (0 real images). The local sim stretches the SO-101 arm up to 1.6× (`fold_sim.py:981`) to fake reachability — the exact failure Brev hit.
6. README is stale/self-contradictory (no YAM/Isaac; §13b "no verified protocol" vs §13c "protocol confirmed"; LeArm still "now"); docs `DEEP_RESEARCH_DIGEST.md` / `NEXT_90_DAYS.md` still recommend the abandoned SO-101/ACT roadmap.

**Medium / cleanup**
7. Dead central-safety-gate architecture (`safety_gate.py`/`motion_core.py` unused in prod) invites future executors to bypass it.
8. `corner_error_m` stores **pixels** when uncalibrated (proven: 165.05 in a recorded episode) — recorded episode metrics are not physical accuracy.
9. `trust_remote_code=True` unpinned by default in the MolmoAct2 load path.
10. No LICENSE file despite Apache-2.0 declaration; no dependency lockfile.

**Reassuring (verified true):** no secrets are committed or in history (keys read from env only); the motion double-gate, YAM shadow gate, and dataset approval gates genuinely hold; git *history* is clean (largest blob ever = 128 KB `cli.py`).
