# TeraFold — Brev Isaac Integration Plan

Generated: 2026-07-05 — read-only audit. Companion to [`REPO_AUDIT_TERA_ROBOTICS.md`](REPO_AUDIT_TERA_ROBOTICS.md), [`TERA_GIT_ARTIFACT_POLICY.md`](TERA_GIT_ARTIFACT_POLICY.md), [`TERA_NEXT_STEPS_V6.md`](TERA_NEXT_STEPS_V6.md).

**Goal:** merge the Brev NVIDIA-VM Isaac work (dual SO-101 + PhysX cloth "towel v2") into this repo cleanly, without falsifying the repo's own honest safety/scope documentation and without the embodiment fork silently corrupting datasets.

**The Brev files to be integrated** (currently only on the VM — none exist in this repo; verified by grep, `brev`/`so101`-in-Isaac/`physx`/`2691` = 0 hits):
```
scripts/tera/create_clean_rect_towel_usd_v2.py        tera_checkpoints/clean_towel_v2/STATUS.json
scripts/tera/test_clean_rect_towel_usd_v2.py          tera_checkpoints/so101_clean_towel_v1/STATUS.json
scripts/tera/visualize_clean_towel_usd_v2.py          tera_checkpoints/clean_towel_v4/STATUS.json
scripts/tera/so101_clean_towel_scene_utils_v1.py      tera_checkpoints/clean_towel_v5_contact/STATUS.json
scripts/tera/test_so101_clean_towel_scene_v1.py
scripts/tera/run_so101_clean_towel_assisted_fold_v1.py
scripts/tera/clean_towel_v4_common.py                 train_clean_towel_policy_v4.py
scripts/tera/eval_clean_towel_policy_v4_no_assist.py  eval_clean_towel_policy_v4_with_stop.py
scripts/tera/run_clean_towel_v4_assist_curriculum.py
scripts/tera/clean_towel_v5_contact_common.py         test_clean_towel_gripper_contact_sweep_v5.py
scripts/tera/run_clean_towel_robot_contact_fold_best_v5.py
scripts/tera/run_clean_towel_gripper_attachment_fold_v5.py
```

---

## 1. The one decision that governs everything: embodiment

Before any file moves, decide the canonical embodiment. This is not cosmetic — it determines dataset schemas, config shape, the norm tag, and whether the local YAM stack is kept, retargeted, or frozen.

| | Local repo (YAM stack) | Brev (verified) | Local LeRobot export |
|---|---|---|---|
| Robot | dual I2RT YAM | **dual SO-101** | single arm (mock) |
| Action dim | **14** (2×(6+1)) | **12** | **7** |
| State dim | 14 | (unknown layout) | 13 |
| Cameras | top/left/right | (unknown) | top only |
| Norm tag | `yam_dual_molmoact2` | n/a for SO-101 | n/a |
| Cloth | rigid 0.40×0.30 box | PhysX 69×39, 2691 particles, 0.68 m | n/a |

**Recommendation: adopt dual SO-101 (action_dim 12) as the canonical *cloth-folding* embodiment**, because it is the only one with verified physics behind it, and freeze the YAM/MolmoAct2 shadow stack as a parallel, clearly-labeled research lane (it does not need to move for V6/V7). The alternative — retargeting Brev's SO-101 work onto YAM assets — throws away the one thing that is proven. Record this decision in a short `docs/EMBODIMENT_DECISION.md` so the stale `deep-research-report.md`/`NEXT_90_DAYS.md` (SO-101+ACT) and the `goal.md`/YAM docs (dual YAM) stop silently contradicting each other.

---

## 2. Package-layout recommendation

**Do not land the flat `scripts/tera/*_vN.py` files as-is beside the package, and do not drop them into `sim/yam_dual_towel_scene/`** (that path is committed generator output for a *different* robot; regenerating clobbers hand edits, and the name would mix embodiments).

**Recommended: a hybrid — a real `terafold/isaac/` module for reusable code, plus a thin `scripts/tera/` for runnable entry points that import it.** This matches the repo's existing convention (importable logic lives in `terafold/<pkg>/`; `scripts/NN_*.py` are thin wrappers around it) and keeps the Isaac dependency import-guarded so the package still imports on a Mac.

```
terafold/isaac/                       NEW module (import-guarded; heavy Isaac imports lazy inside functions)
├── __init__.py                       "requires the [isaac] extra; nothing here imports Isaac at module top"
├── cloth_towel.py                    <- create_clean_rect_towel_usd_v2.py  (PhysX cloth builder; the reusable core)
├── so101_scene.py                    <- so101_clean_towel_scene_utils_v1.py (dual-SO-101 + towel scene assembly)
├── reachability.py                   NEW for V6 (see TERA_NEXT_STEPS_V6.md)
├── contact.py                        <- clean_towel_v5_contact_common.py    (grasp/attachment/contact helpers)
└── policy_v4.py                      <- clean_towel_v4_common.py            (policy train/eval shared code)

scripts/tera/                         NEW dir of thin runnable entry points (Isaac interpreter on Brev)
├── build_clean_towel_scene.py        <- run_so101_clean_towel_assisted_fold_v1.py + scene build
├── test_clean_towel_scene.py         <- test_*_v2/_v1.py (also add to tests/ where hermetic)
├── run_assisted_fold.py              <- run_so101_clean_towel_assisted_fold_v1.py
├── run_v4_train.py / run_v4_eval.py  <- train_/eval_clean_towel_policy_v4*.py
├── run_v5_contact_sweep.py           <- test_clean_towel_gripper_contact_sweep_v5.py
└── run_v6_reachability.py            NEW (see TERA_NEXT_STEPS_V6.md)

configs/isaac/so101_clean_towel.yaml  NEW canonical SO-101 scene/config (action_dim 12), mirroring yam_dual_reference.yaml's shape
```

**Why a module, not flat scripts:** the visualize/test/utils/common files are genuinely reusable logic (scene assembly, cloth params, metrics) that other scripts and tests import — that belongs in `terafold/isaac/`, versioned by git, not duplicated across `*_v2/_v4/_v5` scripts. The `_vN` suffixes are experiment bookkeeping that git history should carry instead; collapse them into one evolving module and let commits track the versions.

**Mac / Brev / Cloud portability:** put all Isaac imports (`isaacsim`, `omni.*`, `pxr`) **inside functions**, never at module top, and add an `[isaac]` note in `pyproject.toml` (no pinned Isaac dependency — it ships in the NGC container). `terafold/isaac/__init__.py` must import cleanly on a Mac (so `import terafold` and the test suite still work); calling a build function without Isaac should print the friendly message and exit 2, exactly as the existing `build_isaac_scene.py` already does. This keeps the three environments working: Mac = numpy/dev + import-guarded stubs, Brev = Isaac present, Cloud GPU = Modal/RunPod for policy training.

---

## 3. Per-file conflict matrix

Kind: **source** (reusable logic → `terafold/isaac/`), **script** (runnable → `scripts/tera/`), **artifact** (evidence → not code). Conflict = collides with existing local code/convention.

| Brev file | Recommended repo path | Kind | Conflicts with | Severity |
|---|---|---|---|---|
| `create_clean_rect_towel_usd_v2.py` | `terafold/isaac/cloth_towel.py` | source | Local towel is a **rigid box** (`build_isaac_scene.py:107`), no PhysX. New capability, not a merge. | Low (additive) |
| `test_clean_rect_towel_usd_v2.py` | `tests/test_isaac_cloth_towel.py` (hermetic parts) + `scripts/tera/` | script/test | Must not require Isaac in CI (mirror `test_yam_isaac_scene.py`'s no-Isaac contract). | Low |
| `visualize_clean_towel_usd_v2.py` | `scripts/tera/visualize_clean_towel.py` | script | Overlaps `terafold/sim/fold_sim.py` viz **in role only** (different stack). | Low |
| `so101_clean_towel_scene_utils_v1.py` | `terafold/isaac/so101_scene.py` | source | **Duplicates the *role* of `terafold/yam/isaac_scene.py`** (both "build dual-arm+towel Isaac scene"), different robot. Decide one generator per embodiment. | **High** |
| `test_so101_clean_towel_scene_v1.py` | `tests/test_isaac_so101_scene.py` | script/test | — | Low |
| `run_so101_clean_towel_assisted_fold_v1.py` | `scripts/tera/run_assisted_fold.py` | script | Its "assisted fold" moves *particles*, not the robot — name/scope must stay explicit vs `terafold/planning` (which plans robot motion). | Medium (honesty) |
| `clean_towel_v4_common.py` | `terafold/isaac/policy_v4.py` | source | — | Low |
| `train_clean_towel_policy_v4.py` | `scripts/tera/run_v4_train.py` | script | Overlaps `terafold/learning/train_act_wrapper.py` intent; but this trains a real policy in Isaac vs the local command-printer. | Medium |
| `eval_clean_towel_policy_v4_no_assist.py` | `scripts/tera/run_v4_eval.py` | script | Overlaps `terafold/eval/benchmark.py`; **reuse local `fold_metrics`/`fold_quality`** rather than re-implementing width metrics. | Medium |
| `eval_clean_towel_policy_v4_with_stop.py` | `scripts/tera/run_v4_eval_stop.py` | script | Same as above. | Medium |
| `run_clean_towel_v4_assist_curriculum.py` | `scripts/tera/run_v4_curriculum.py` | script | — | Low |
| `clean_towel_v5_contact_common.py` | `terafold/isaac/contact.py` | source | Local contact is **structurally locked** (`capabilities.py:300-309`); Brev contact code has no local gate to pass through. Wire it behind a real V6-gated unlock, not a bypass. | **High** |
| `test_clean_towel_gripper_contact_sweep_v5.py` | `scripts/tera/run_v5_contact_sweep.py` | script | — | Low |
| `run_clean_towel_robot_contact_fold_best_v5.py` | `scripts/tera/run_v5_contact_fold.py` | script | This is the run that **failed** (jaw ≥0.406 m from edge). Keep, but do not re-run before V6 fixes placement. | Medium |
| `run_clean_towel_gripper_attachment_fold_v5.py` | `scripts/tera/run_v5_attachment_fold.py` | script | "Attachment" = a particle-attachment shortcut, not autonomous grasping — label explicitly. | Medium (honesty) |
| `tera_checkpoints/*/STATUS.json` (×4) | `docs/brev_status/<name>_STATUS.json` (tracked evidence) | **artifact** | `tera_checkpoints/` does not exist locally; **do not** create a new gitignored checkpoint tree that repeats the `runs/` backup problem. Commit the small STATUS.json evidence; keep large weights off-repo. | Low |

**Path-collision check (done):** `scripts/tera/` and `tera_checkpoints/` do **not** exist locally — no direct path collisions. All conflicts above are *functional/convention* collisions, chiefly the two "High" rows: two Isaac scene generators for two embodiments, and Brev contact code that has no local safety gate to land behind.

---

## 4. Checkpoints / datasets / model outputs — going forward

- **STATUS.json convention (keep it):** the Brev `STATUS.json` files are exactly the right pattern — small, machine-readable, greppable evidence. Commit them under `docs/brev_status/` (tracked). They are the *only* in-repo trace of the Brev work; without them the VM stays a single point of failure.
- **Weights stay off git.** Isaac policy checkpoints, YOLO weights, and MolmoAct2 snapshots do **not** belong in git (see [`TERA_GIT_ARTIFACT_POLICY.md`](TERA_GIT_ARTIFACT_POLICY.md)). Store them on the Brev VM + one off-repo backup (HF model repo, S3/GCS, or a synced drive); reference them by path/URL + sha256 in the STATUS.json.
- **Datasets: one schema, chosen deliberately.** Standardize episodes on the SO-101 (action_dim 12) layout, extend `terafold/data/episode_schema.py` (recorder contract #1) to multi-camera, and **write Isaac rollouts into that format** so `eval/benchmark.py` + `score_fold` work for free. Add a `width_reduction` metric key (it does not exist yet). Do **not** merge the 14-dim YAM dummy episodes or the 7-dim mock LeRobot export into SO-101 training data — the zero-/forward-filled actions in the exporters would poison behavior cloning.
- **What syncs locally vs stays on Brev:** sync back the small stuff (STATUS.json, metrics CSVs, `.rrd` traces, a few sample frames) into git or a tracked `docs/brev_status/`; keep the heavy stuff (USD scenes, full episode video, policy weights) on the VM + backup.

---

## 5. Migration checklist (ordered; every step read-only-safe until you act)

Each stage has a verification step. Do **not** proceed past a failed verification.

1. **Freeze the repo state.** Back up `data/raw_towel_real_v1/` (105 HEIC), `runs/` weights (6 files, ~20 MB), and `vendor/waveshare/.../{scservo_sdk, safe_*.py}` off-repo *before anything else*. Verify: checksums match. (Detail in [`TERA_GIT_ARTIFACT_POLICY.md`](TERA_GIT_ARTIFACT_POLICY.md).)
2. **Fix git hygiene.** Restore a `/data/` ignore rule; `git gc --prune=now` to reclaim ~1.45 GB of unreachable objects. Verify: `git status` shows no `?? data/`; `du -sh .git` drops to a few MB.
3. **Record the embodiment decision.** Write `docs/EMBODIMENT_DECISION.md` (SO-101 canonical for folding; YAM shadow lane frozen). Verify: it supersedes the conflicting claims in README/`deep-research-report.md`/`NEXT_90_DAYS.md`/`YAM_ISAAC_SIM.md` with explicit "as of 2026-07-05" notes.
4. **Create the skeleton, empty.** Add `terafold/isaac/__init__.py` (import-guarded), `configs/isaac/so101_clean_towel.yaml`. Verify: `import terafold` and the existing test suite still pass on the Mac (no Isaac needed).
5. **Land the pure/reusable code first** (`cloth_towel.py`, `so101_scene.py`, `contact.py`, `policy_v4.py`) with all Isaac imports moved inside functions. Verify on Mac: modules import; a no-Isaac call exits 2 with the friendly message (mirror `test_yam_isaac_scene.py`).
6. **Land the runnable scripts** in `scripts/tera/`, importing the module. Verify on Brev: scene builds; `test_*` pass inside the Isaac interpreter.
7. **Reconcile metrics.** Route Brev eval through `terafold/eval/fold_metrics.py` + `fold_quality.py`; add the `width_reduction` key. Verify: local metrics reproduce the Brev width numbers (0.68→0.339 assisted) on the same particle data.
8. **Commit the STATUS.json evidence** under `docs/brev_status/`. Verify: the four milestones (v2 towel, so101 scene, v4, v5-contact) are now discoverable from the repo alone.
9. **Do V6 next, not policy work.** See [`TERA_NEXT_STEPS_V6.md`](TERA_NEXT_STEPS_V6.md).

---

## 6. What could go wrong

- **Silent embodiment mixing** — landing Brev SO-101 code under the `yam` name/config, or evaluating it with `configs/yam_dual_reference.yaml` (action_dim 14), produces shape errors or, worse, silently mis-normalized data. *Mitigation: step 3 + a distinct `configs/isaac/` config, never relax the tested `action_dim==14` asserts.*
- **Two Isaac scene generators drift** — keeping both `terafold/yam/isaac_scene.py` and a new SO-101 generator invites divergence. *Mitigation: one generator per embodiment, both driven by config; add a CI check that regenerates and diffs the committed scene package.*
- **Contact code bypasses the safety architecture** — the local `MotionGate`/`motion_core` chokepoint is already dead in production; Brev contact code arriving without a gate widens that gap. *Mitigation: wire Brev contact behind a real, V6-informed unlock, and revive `MotionGate` as the single chokepoint (or delete it, but don't leave it as decorative).*
- **Unverified LeRobot export breaks V7 training** — neither local exporter produces a load-verified `LeRobotDataset`. *Mitigation: budget a conversion/validation step before any ACT/SmolVLA run.*
- **Re-running V5 contact before V6** — repeats the 0.406 m failure and wastes GPU. *Mitigation: gate all contact runs on a passing V6 reachability result.*
- **MolmoAct2 checkpoint API mismatch** — the adapter hedges between two unverified recipes and loads hub code `trust_remote_code=True` unpinned. *Mitigation: pin `--revision`, verify the checkpoint exists, before any real (non-mock) run.*
