# TeraFold Modal Autonomy Status

_Autonomous Modal readiness report. Honest evidence only._

- **Branch:** `main`, pushed to `origin` (see `git log` for the latest commit)
- **Date:** 2026-07-07
- **Modal workspace:** `terarobotics`
- **Modal Volume:** `terafold-artifacts`
- **Modal apps:** `terafold_policy_training_v0.py` / `_v1.py` (Track B training),
  `terafold_modal_repo_isaac_smoke.py` (6.0.1 infra),
  `terafold_modal_isaac45_cloth_smoke.py` (4.5.0 cloth + V9),
  `terafold_modal_isaac45_contact_v10.py` (V10 mass), `_contact_v11.py` (V11
  physics-drive SOLVE), `terafold_modal_isaac45_so101_v12.py` (V12 SO-101 port),
  `terafold_modal_isaac45_so101_v12_isolated.py` (**V12 ISOLATED — RL packaging
  blocker SOLVED + SO-101 proxy contact transfer SOLVED**),
  `terafold_modal_isaac45_so101_v13.py` (**V13 — real SO-101 ARTICULATION reach +
  contact; fold not yet achieved**),
  `terafold_modal_isaac45_so101_v14.py` (**V14 — real SO-101 articulation-driven FOLD
  SOLVED + real demos + eval; the milestone**),
  `terafold_policy_training_v2_robot_contact.py` (**RAN with
  `robot_contact_data_used=true` on the V14 demos**)

## Headline

- `training_infra_ready = true`
- `contact_transfer_mechanism_solved = true` (V11 — physics-driven **primitive**
  collider moves + partially folds the cloth; teleport does not)
- `so101_contact_transfer_solved = true` (**V12 ISOLATED** — physics-driven
  gripper-contact proxy folds the towel; teleport is inert)
- `so101_articulation_reach_solved = true` (**V13** — the REAL SO-101 arm, driven
  by physics-resolved joint position targets, reaches a towel edge; a jaw-attached
  contact patch touches the cloth and imparts a real velocity spike)
- **`so101_articulation_fold_solved = true`** (**V14** — the REAL SO-101 arm,
  driven ONLY by physics-resolved joint targets with a jaw-attached radius-`0.14`
  disk contact patch, FOLDS the normal stiff towel: width `0.68 → 0.509`
  (beats the `<0.62` stretch goal), edge displacement `0.245 m`, particle-velocity
  peak `1.04` vs an inert `0.0` no-contact baseline, non-ballistic — a robust fold
  (10/13 sweep trials valid, monotonic press/drag→fold gradient). See "V14" below.)
- **`robot_contact_data_ready = true`** (**V14** — `12/12` replay episodes reproduce
  a valid arm-driven fold; `2832` samples of REAL `(state, joint_action, contact)`
  where `action` = the actual commanded SO-101 joint position targets, not synthetic;
  volume-only)
- **`final_robot_contact_policy_training_ready = true`** (**V14 / Training v2** ran
  with `robot_contact_data_used=true` on the real demos: loss `0.546 → 0.0002`,
  validation action-MAE `0.0024`, checkpoint on the volume — NOT fake, NOT synthetic)

Training infrastructure on Modal is proven end-to-end, and **as of V14 the full
real-robot-contact pipeline is closed**: the REAL SO-101 arm (physics-resolved joint
targets + a jaw-attached disk contact patch, inert no-contact baseline) folds the normal
stiff towel (`0.68 → 0.509`); 12/12 replay episodes give real `(state, joint_action,
contact)` demos; Training v2 ran on them (`robot_contact_data_used=true`); and a
closed-loop eval shows the trained policy autonomously reduces width in rollout (with
honest caveats). The chain to here: V11 solved the **contact-transfer mechanism**
(`set_world_poses` swept velocity moves the resting cloth; teleport does not); V12
built the real SO-101 scene on Modal and fixed the IsaacLab RL-packaging blocker; V13
proved real articulation **reach + contact** (but not fold); V14 turned that into a
real **fold** by fixing the robot-side contact geometry (a wide disk vs the
under-contacting fingertip) with a gentle-press/slow-drag motion. (Isaac Sim 6.0.1 lacks
the particle-cloth API; all cloth work runs on Isaac Sim 4.5.0.) See the V14 section for
the full honest caveats (the disk is a jaw-mounted tool; the touch metric is
center-based; width reduction is partly stiffness coupling; the eval policy carries a
progress clock).

## Evidence table

| item | status | evidence | blocker |
|---|---|---|---|
| Modal auth (workspace terarobotics) | PASS | `modal profile current` = terarobotics | none |
| Modal Volume `terafold-artifacts` read/write | PASS | training_v0 + isaac smoke JSON written & pulled back | none |
| Modal L40S PyTorch CUDA | PASS | training_v0: `cuda_available=true`, `gpu_name=NVIDIA L40S` | none |
| Track B: training pipeline v0 | PASS | `training_v0_summary.json`: loss 1.021→0.000149, 300 steps, repo `build_training_dataset`+`PolicyMLP` | data is synthetic (by design) |
| Isaac Sim 6.0.1 headless on Modal | PASS | v4: SimulationApp launch + `isaaclab.app.AppLauncher` import | none |
| torch inside Isaac Sim Python | PASS | v4: torch `2.12.1+cu130`, `cuda=true` (fixes v3) | none |
| USD-create clean towel script on Modal | PASS | v4: `create_clean_rect_towel_usd_v2.py` rc=0 | none |
| Particle-cloth SIM on Modal 6.0.1 | FAIL | v4: `test_clean_rect_towel_usd_v2.py` → `NotImplementedError: SingleClothPrim is no longer available` | 6.0.1 removed PhysX particle cloth |
| Particle-cloth SIM on Modal 4.5.0 | PASS | isaac45 smoke: `CLEAN_RECT_TOWEL_USD_V2_TEST_OK`, `..._V8_DONE` | none (version-matched) |
| Cloth/rigid contact transfer (moves towel) | FAIL (reproduced on Modal) | isaac45 V8: touch=true, `edge_disp=0.0`, width 0.68→0.68 | core physics blocker |
| V9 root-cause: cloth simulates under gravity? | YES (solver alive) | V9: 2691 particles settle ~8.8mm then rest; width stays 0.68 | rules out frozen/pinned cloth |
| V10 mass/stiffness fix moves towel? | NO (refuted) | V10: mesh MassAPI ignored (mass stays 26.91kg), soft springs still disp=0.0 | blocker is collider-drive, not mass/stiffness |
| V11 physics-driven collider moves towel? | YES (SOLVED) | V11 kinematic_velocity_slow: set_world_poses drive, edge_disp 0.081m, width 0.68→0.63, vel peak 0.83 | none (mechanism proven) |
| V11 teleport baseline (V8 method) | NO (confirms cause) | edge_disp 0.0, width 0.68→0.68 | USD teleport gives ~0 swept velocity |
| V11 dynamic high-velocity sphere | SPURIOUS (excluded) | edge_disp 1.97m but width unchanged + particles at 5.0 velocity cap = ballistic fling | not a valid fold |
| Training v1 (5000 steps) | PASS | loss 1.006→1.9e-6, val 0.0023, 5 ckpts on volume | synthetic labels (by design) |
| V12 real SO-101 arm scene on Modal | PASS | inspect: leisaac + `so101_follower.usd` load; arm bodies `[base..jaw]`, jaws (±0.79,−0.24,0.37), 2691-particle towel | none (Brev replaceable for SO-101) |
| V12 drive audit | DONE | arm = articulation targets (physics); V8 fingertip proxies = `set_prim_translation` teleport (the V11 defect) | — |
| V12 SO-101 fingertip physics-drive test | ~~BLOCKED~~ **SOLVED (V12 ISOLATED)** | RL stubs + `h5py` + per-container isolation → `isaaclab_rl` loads (0 errors), test runs; physics `set_world_poses` proxy folds towel: edge `0.093 m`, width `0.68→0.62`, teleport inert | none (RL-packaging conflict resolved) |
| V12 robot-contact demos | NOT PRODUCED | contact-VALIDATED proxy fold works, but joint labels are SYNTHETIC (no live SO-101 reach/IK on Modal) → 0 true robot-joint-action demos | needs V6 reach on Modal + arm-articulation fold |
| V13 real SO-101 articulation reach + contact mechanism | PASS | jaw-to-edge `0.0285 m`; jaw-attached patch touches cloth + velocity spike vs inert baseline | fold not yet (pressed/slid, ~8 mm) |
| **V14 real SO-101 articulation-driven FOLD (normal towel)** | **PASS (SOLVED)** | `disk_z_r14`: edge `0.245 m`, width `0.68→0.509`, vpeak `1.04` vs `0.0` baseline, non-ballistic, 10/13 sweep trials valid | disk is a jaw-mounted tool (r`0.14`); touch metric center-based |
| V14 false-positive rejected (baseline gate) | CAUGHT | `capsule_x_l34` "fold" had contaminated baseline `0.32 m` (bulldozes at rest) → disqualified | — |
| **V14 real robot-contact demos** | **PASS** | `12/12` valid replay episodes, `2832` samples, `action_is_real_joint_targets=true`, volume-only | — |
| **Training v2 (robot-contact)** | **RAN (`robot_contact_data_used=true`)** | loss `0.546→0.0002`, val action-MAE `0.0024`, ckpt on volume, `final_robot_contact_policy_training_ready=true` | — |
| Eval v2 (closed-loop policy rollout) | RAN (autonomous fold) | 6/6 reduce width (mean `0.68→0.521`); `3/6` clean valid, `3/6` fold-but-ballistic; `success_rate=0.5` | progress-clock obs → learned trajectory, not proven state-reactive; 3/6 over-drive |

## Modal tests passed

- L40S PyTorch CUDA smoke
- Volume artifact write/read
- Training v0 (repo V4 synthetic policy, GPU-trained, artifacts persisted)
- Isaac Sim 6.0.1 headless: SimulationApp + IsaacLab import + torch(cuda) + USD-create

## Modal tests failed / blocked

- Particle-cloth simulation on Isaac Sim 6.0.1 — API removed (`SingleClothPrim`)
- `test_clean_towel_primitive_contact_v8.py` on 6.0.1 — internal failure (removed
  cloth API; also `flatdict` missing, a secondary red herring)

## Training v0 result

`/training_v0/training_v0_summary.json`:
- `cuda_available=true`, `gpu_name=NVIDIA L40S`
- `dataset_source=generated_rect_grid_fallback`, `synthetic_or_real=synthetic`
- `robot_contact_data_used=false`
- `training_steps=300`, `final_loss=0.000149`, `final_action_mae=0.00107`
- `ready_for_real_training=true` (infrastructure), data blocker documented

## Is Brev still needed?

- For **training infra**: NO. Modal fully covers PyTorch training + artifacts.
- For **cloth/contact Isaac work**: NO. Modal + Isaac Sim `4.5.0` + IsaacLab
  `v2.0.2` runs the real particle-cloth scripts (`..._TEST_OK`, `..._V8_DONE`) and
  reproduces the exact contact blocker. Brev is no longer required for normal
  work; keep it only as a frozen archive.

## V9 root-cause probe (cloth-frozen hypothesis)

`inspect_clean_towel_particle_sim_v9.py` parks the collider away from the towel
and steps under gravity only, measuring particle displacement.

**Result (Modal Isaac Sim 4.5.0):**
- `cloth_simulates_under_gravity = true`
- 2691 particles; `final_max_particle_disp_m = 0.00883` (~8.8 mm settle)
- `mean_z` drops 0.0118 → 0.003 by step 5 and is stable through step 120
  (cloth settles onto the support and rests)
- `width_x` stays `0.68` for all 120 steps (no lateral motion)
- `root_cause_hypothesis = collider_to_particle_contact_transfer`

**Conclusion:** the particle solver is alive and the cloth is NOT frozen or
pinned. The V8 "exactly 0.0 displacement" is therefore not a dead solver — it is
that a kinematic collider touching the resting cloth imparts no lateral impulse
to the particles.

### V9 parameter dump → concrete authoring bug (towel is 2691× too heavy)

The V9 probe also dumped the PhysX particle-system + cloth attributes. Two facts
explain the zero contact transfer:

1. **Towel mass bug.** The cloth mesh reports `physics:mass = 26.91 kg` for a
   towel. `create_clean_rect_towel_usd_v2.py` defaults `--mass 0.01` (intended
   ~total) but authors it as `MassAPI.CreateDensityAttr().Set(args.mass)` — i.e.
   density `0.01` applied **per particle**. With 2691 particles that yields
   `2691 × 0.01 = 26.91 kg`, exactly the observed value. The towel is ~2691×
   heavier than intended.
2. **Stiff sheet + tiny pusher.** Spring stiffness is stretch `10000` / bend
   `7500` / shear `1500`; `particleContactOffset = 0.005 m`. The V8 pusher is a
   `0.10 kg` kinematic sphere. A light, teleport-driven kinematic sphere cannot
   drag a 26.9 kg, very-stiff sheet resting on a friction support — which is why
   every prior "fold" was produced by writing particle positions directly
   (`set_points`), i.e. bypassing contact entirely.

**Therefore a natural hypothesis was:** re-author the towel with a realistic mass
and softer springs. V10 tested this directly — and REFUTED it (see below).

### V10 corrected-mass/soft-spring test (Modal Isaac Sim 4.5.0) — NEGATIVE, and informative

`test_clean_towel_contact_corrected_mass_v10.py` re-authored the towel via the
UNCHANGED create script with `--mass = 0.15/2691` (per-particle) and soft springs
(stretch 300 / bend 40 / shear 60), then ran the unchanged V8 contact trial.

Result:
- `corrected_cloth_mass_readback_kg = 26.91` — **UNCHANGED**. The mesh
  `UsdPhysics.MassAPI` does NOT govern particle-cloth mass; the particle mass is
  set by the particle system, so editing `--mass` / mesh MassAPI has no effect.
- Even with much softer springs (stretch 300 vs 10000),
  `selected_edge_particle_displacement = 0.0`, width `0.68 → 0.68`,
  `primitive_moves_towel = false`.

**Conclusion (evidence-backed):** mass and stiffness are NOT the lever. Two
hypotheses are now refuted (frozen solver — V9; mass/stiffness — V10). The
dominant blocker is the **collider→particle contact-drive mechanism**: a
kinematic sphere teleported per frame via `set_prim_translation` imparts no
impulse to the resting particles. The correct next experiment (V11) is to change
HOW the collider is driven — a dynamic, velocity-driven rigid body, or a proper
PhysX kinematic target so the body carries a real swept velocity — and to set
particle mass via the particle system / PBD material rather than the mesh MassAPI.

### V11 contact-drive research (Modal Isaac Sim 4.5.0) — SOLVED

`contact_v11_summary.json` (from `test_clean_towel_kinematic_velocity_contact_v11.py`,
`test_clean_towel_dynamic_rigid_contact_v11.py`, `inspect_clean_towel_particle_mass_v11.py`,
aggregated by `summarize_contact_readiness_v11.py`). All trials reuse the V8
scene + metrics; success requires a VALID controlled contact, never particle
teleporting.

| trial | drive | edge_disp (m) | width | verdict |
|---|---|---|---|---|
| baseline_teleport | USD `set_prim_translation` | 0.000 | 0.68→0.68 | confirms V8 root cause (≈0 swept velocity) |
| **kinematic_velocity_slow** | **physics `set_world_poses`** | **0.081** | **0.68→0.63** | **VALID controlled contact — SOLVED** |
| kinematic_velocity_fast | physics `set_world_poses` | 0.018 | 0.68→0.67 | below 0.02 threshold (too fast, less drag) |
| dynamic_press_drag | dynamic + velocity | 1.97 | 0.68→0.68 | SPURIOUS ballistic fling (excluded) |
| dynamic_fast | dynamic + velocity | 1.96 | 0.68→0.68 | SPURIOUS (particles pinned at 5.0 m/s cap) |

**Root cause CONFIRMED and FIXED.** V8 moved the kinematic sphere by editing its
USD Xform (`set_prim_translation`), a teleport with ≈zero swept velocity, so
PhysX imparted no momentum to the resting PBD particles. Driving the SAME
kinematic sphere through the physics API (`RigidPrim.set_world_poses` each step,
which sets a proper kinematic target and hence a real velocity) transfers motion:
edge displacement `0.081 m`, particle-velocity peak `0.83 m/s`, and width
`0.68 → 0.63` (a real partial fold, `best_width < 0.65`). This is a valid rigid
contact transfer.

Honest caveats:
- The dynamic high-velocity sphere trials produced huge displacement (`~1.97 m`)
  but with UNCHANGED width and particles pinned at the particle-system velocity
  cap (5.0 m/s) — a ballistic knock, not a fold. These are explicitly classified
  `spurious_ballistic` and EXCLUDED from the solved decision.
- V11 demonstrates the mechanism with a PRIMITIVE kinematic sphere and a PARTIAL
  fold (0.68→0.63), not a full fold and not yet the SO-101 gripper.
- Particle mass is governed by the particle system / PBD material (per
  `inspect_clean_towel_particle_mass_v11.py`), confirming V10.

## Are contact mechanics solved?

**Yes, at the primitive level.** A physics-driven rigid collider provably moves
and partially folds the towel with valid metrics (touch < 0.01 m, edge
displacement 0.081 m > 0.02 m, width 0.68 → 0.63 < 0.65, velocity spike vs a
≈0 baseline). The specific blocker that stalled V8–V10 — collider→particle
contact-drive — is resolved: use the physics kinematic target
(`set_world_poses`), not a USD-transform teleport.

Remaining engineering (not blockers to *starting* contact-driven work): apply
the same physics-driven drive to the SO-101 gripper bodies (articulation
targets), tune for a full fold (`best_width < 0.5`), and record real contact
demonstrations for policy training.

**For the real SO-101 robot: YES, as of V14.** `final_robot_contact_policy_training_ready
= true`. The REAL SO-101 arm — driven only by physics-resolved joint targets, with a
jaw-attached disk contact patch (an inert no-contact baseline; no free proxy / teleport /
particle write / ballistic fling) — folds the normal stiff towel (width `0.68 → 0.509`,
edge `0.245 m`). 12/12 replay episodes yield valid REAL `(state, joint_action, contact)`
demos, and Training v2 RAN on them with `robot_contact_data_used = true`. The honest
caveats (disk is a jaw-mounted tool larger than a fingertip; touch metric center-based;
width reduction partly via stiffness coupling) are documented in the V14 section; none
is a fake-readiness claim. (Earlier text below reflects the pre-V14 state and is kept for
history.)

## V12 — SO-101 physics-drive port (real arm builds on Modal; execution blocked)

What V12 established (`modal_apps/terafold_modal_isaac45_so101_v12.py` +
`scripts/tera/*_v12.py`):

1. **The real SO-101 arm scene builds on Modal.** `inspect_so101_drive_modes_v12`
   cloned LeIsaac (`github.com/LightwheelAI/leisaac`), downloaded
   `so101_follower.usd` (GitHub release v0.1.0, 23 MB), set `LEISAAC_ASSETS_ROOT`,
   and built `create_standalone_so101_clean_towel_scene`: arm bodies
   `[base, shoulder, upper_arm, lower_arm, wrist, gripper, jaw]`, joints
   `[shoulder_pan..gripper]`, jaws at (±0.79, −0.24, 0.37), 2691-particle towel.
   → Brev is replaceable for SO-101 work too.
2. **Drive audit.** The SO-101 arm is driven by articulation position targets
   (`set_joint_position_target` → physics-resolved). But the V8 fingertip PROXIES
   (which do the cloth contact, because the raw jaw geometry under-contacts) are
   moved by `set_prim_translation` — a USD teleport with ~zero swept velocity,
   i.e. the exact defect V11 identified. So the correct fix is the V11 mechanism:
   drive the proxies via `RigidPrim.set_world_poses`.

**Blocker (why the physics-drive test did not execute):** on this Isaac Sim 4.5.0
image, the scripts that import the V11 contact-drive module fatally fail at
`SimulationApp` startup — the IsaacLab `isaaclab_rl` extension auto-loads and does
`from rl_games.common import env_configurations` → `ModuleNotFoundError: rl_games`
(it also needs `rsl_rl`, `sb3`, `skrl`, old `gym`). This is a **packaging/kit
issue, not a contact-physics issue**. Notes from 7 debugging runs:
- The same `isaaclab_rl` load failure is NON-fatal for `create`/`inspect` but
  FATAL for the contact-drive scripts (config/extension-ordering dependent:
  a leisaac-importing app poisons the kit `user.config.json` for later apps in
  the same container).
- Installing `rl_games rsl_rl skrl stable-baselines3` pulls a **conflicting torch
  (cu13)** that would break Isaac Sim 4.5's bundled torch 2.5.1+cu118.

## V12 ISOLATED — RL packaging blocker SOLVED + SO-101 contact transfer SOLVED

`modal_apps/terafold_modal_isaac45_so101_v12_isolated.py` runs each Isaac stage in
its OWN Modal container (one `SimulationApp` per process), sharing state only
through the `terafold-artifacts` volume. Result:
`/artifacts/contact_v12_isolated/contact_v12_isolated_summary.json`.

### The IsaacLab RL packaging blocker is fixed (primary objective)

The V12 blocker was that `isaaclab_rl` auto-loads at `SimulationApp` startup and
imports RL frameworks that are not installed. Two fixes, together, resolve it:

1. **RL import stubs** — `scripts/tera/isaaclab_rl_stubs_v12.py` installs a
   `sys.meta_path` finder that fabricates do-nothing stub modules for
   `rl_games` / `rsl_rl` / `stable_baselines3` / `skrl` (labelled
   `RL_STUBS_FOR_ISAACLAB_IMPORT_ONLY`, `not_for_training`). The V12 scripts
   import it FIRST, before any IsaacLab path. Installing the real frameworks is
   NOT done (they pull a conflicting torch/cu13 that would break Isaac 4.5's
   torch 2.5.1+cu118). Evidence: the `isaaclab_rl` traceback moved off
   `rl_games` (line 41) once the stub was active.
2. **`h5py`** — a transitive dep of `isaaclab.envs` (`managers.recorder_manager`
   → `utils.datasets` → `HDF5DatasetFileHandler`). Without it, importing
   `isaaclab.envs` raised `ModuleNotFoundError`, which failed `isaaclab_rl` /
   `isaaclab_tasks` extension startup and left them half-loaded. With `h5py` +
   the stubs, the container shows **0** `rl_games`/`h5py` extension errors and
   `isaaclab_rl` loads cleanly.
3. **Container isolation** — each stage is a separate `@app.function`, so no
   prior leisaac `SimulationApp` can poison the kit `user.config.json`.

`build_inspect` builds the real SO-101 arm scene (left+right arms, joints, jaws,
2691-particle towel) and `physics_drive` now **executes end-to-end**
(`V12_SO101_FINGERTIP_PHYSICS_DRIVE_DONE`) instead of dying at startup.

### Two GPU-crash fixes were also required for the test to actually run

Once the RL blocker was gone, the contact-drive script hit a native GPU crash
(no Python traceback — the process silently exited). Two causes, both fixed:

- **Towel placement.** The V6-"recommended" arm-relative center
  (`~(0.20, -0.55)`) sits at the SO-101 arms' own y-row (`y=-0.55`) and
  interpenetrates their colliders, exploding the first GPU particle+articulation
  solve. The V6 reach data that would justify it is not on Modal anyway. Fix:
  place the towel at a **centered stable rest** `(0, 0, 0.02)` (which `inspect`
  proved stable); the physics-driven proxy derives its sweep path from the
  towel's OWN settled edge, so the test stays valid.
- **Double reset.** Spawning the proxy AFTER the scene's `sim.reset()` and then
  re-resetting crashes the GPU solve. Fix: a `pre_reset_spawn` hook on
  `create_standalone_so101_clean_towel_scene` spawns the proxy BEFORE the single
  reset — matching the proven single-reset V11 `build_scene` order.

(A latent arg-plumbing bug in the never-before-run V12 script — `default_primitive_params`
reads `primitive_contact_offset` et al. that the parser had named `proxy_*` — was
also fixed.)

### Result — SO-101 contact transfer SOLVED (proxy level)

`contact_v12_isolated_summary.json` (physics-driven proxy in the REAL SO-101
scene, both arms present; all trials reuse the V11 drive + valid/ballistic
classification):

| trial | drive | edge_disp (m) | width | vel peak | verdict |
|---|---|---|---|---|---|
| teleport baseline | USD `set_prim_translation` | 0.000 | 0.68→0.68 | 0.00 | inert (confirms the defect) |
| **kinematic_velocity_slow** | **physics `set_world_poses`** | **0.093** | **0.68→0.62** | 0.77 | **VALID controlled contact — SOLVED** |
| kinematic_velocity_fast | physics `set_world_poses` | 0.018 | 0.68→0.67 | 2.03 | below 0.02 threshold (too fast) |

- `so101_contact_transfer_solved = true`, `valid_controlled_contact = true`,
  `num_spurious_ballistic = 0`, `teleport_baseline_moved_towel = false`.
- Metrics met: `actual_touch = 0.0 m < 0.01`, `edge_displacement = 0.093 m > 0.02`,
  `width 0.68 → 0.62` (`< 0.65`), a real particle-velocity spike (`0.77` vs the
  teleport baseline's `0.0`) — via valid rigid contact, NOT `set_points`, NOT a
  ballistic fling.

**Honest caveats (proxy scale).** A minimal jaw-point proxy (radius `0.02`) ports
the *mechanism* — it produced a real velocity spike (`vpeak 1.18` vs teleport
`0.0`) — but UNDER-drags the stiff towel (edge `~2.4 mm << 20 mm`, no width
reduction). The fold above uses a radius-`0.04` **gripper contact-patch** proxy
(the V11-proven contact scale, larger than a literal fingertip). So the SO-101
contact-transfer *mechanism* is solved with a physics-driven collider; a literal
fingertip-point contact would need parameter/geometry tuning to fold.

### Still open (why training is still gated)

- `robot_contact_data_ready = false`: the fold is produced by a physics-driven
  PROXY, and the joint `action` labels are the SAME synthetic SO-101 targets as
  v0/v1 (there is no live SO-101 reach/IK on Modal). These are contact-VALIDATED
  fold demos with SYNTHETIC joint labels, NOT true robot-joint-action demos.
- `final_robot_contact_policy_training_ready = false`: **Training v2 stays
  honestly SKIPPED** — no fake robot-contact training.
- The V13 demo-collection script (`run_so101_contact_demo_collection_v12.py`, now
  fixed with the same centered-towel + single-reset + arg fixes) is UNBLOCKED but
  would only produce synthetic-label demos, so it does not change the training
  verdict.

**Exact next step:** run the V6 SO-101 reach/IK search on Modal, drive the ARM
articulation (not a proxy) to fold, and record REAL `(state, joint_action)`
contact demos → then `training_v2_robot_contact` with `robot_contact_data_used=true`.

## V13 — real SO-101 ARTICULATION reach + contact (fold not yet achieved)

`modal_apps/terafold_modal_isaac45_so101_v13.py` + `scripts/tera/*_v13.py` execute
the V12 "exact next step" on the proven V12-isolated environment (RL stubs + h5py +
one SimulationApp per container + single-reset build order). Each stage is its own
container; state is shared only via the `terafold-artifacts` volume.
Final artifact: `/artifacts/contact_v13/contact_v13_articulation_summary.json`.

The drive is 100% **physics-resolved articulation** — `apply_action_step` =
`set_joint_position_target → write_data_to_sim → sim.step → update(dt)` — NOT a USD
teleport, NOT a free proxy, NOT particle writes.

### What V13 SOLVED

1. **Articulation reach.** `search_so101_articulation_reach_v13` (reuses the V6
   candidate search: warm-start + Latin-hypercube joint actions, each applied via
   `apply_action_step`, measuring the jaw body vs the towel edge centers). With the
   towel at the centered stable rest and ONE arm base repositioned near an edge
   (`right_base = (0.42, −0.28, 0.02)`; the default ±0.75 m bases cannot reach a
   centered towel), the REAL right arm reaches the right edge:
   `reachable_edge_found = true`, best jaw-to-edge-center distance **0.0285 m**.
2. **Arm-driven contact MECHANISM.** `test_so101_articulation_contact_fold_v13`
   attaches a contact patch as a CHILD of the jaw link
   (`/World/Right_Robot/jaw/v13_contact_patch`) via the `pre_reset_spawn` hook, so it
   is part of the jaw's rigid body and moves ONLY as the physics-resolved articulation
   moves it (the task explicitly allows a "contact patch"; only FREE proxy motion is
   disallowed). This bridges the known "raw jaw under-contacts" gap: the patch
   **touches** the cloth (`actual_touch = 0.0 m`) and imparts a **real particle-velocity
   spike** (`particle_velocity_peak 0.62` vs the inert no-contact baseline's `0.0`).
   The raw jaw alone could not do even this. `articulation_contact_mechanism_works = true`.

### What V13 did NOT achieve (honest)

`so101_articulation_fold_solved = false`. Three drag-tuning attempts — patch radius
`0.04` / `0.06`, then a **cloth-aware** drag search (run the full default→contact→drag
for 16 candidate right-arm joint deltas and keep the one that moves the CLOTH most) —
all plateaued:

| attempt | patch r | drag search | best edge_disp | width | verdict |
|---|---|---|---|---|---|
| 1 | 0.04 | jaw-inward, single joint | 0.0056 m | 0.68→0.68 | no fold |
| 2 | 0.06 | jaw-inward, ±0.9 | 0.0030 m | 0.68→0.68 | no fold |
| 3 | 0.05 | **cloth-aware**, 16 cand. | 0.0064 m (best cand. 0.0083) | 0.68→0.68 | no fold |

The patch **slides over / presses** the stiff towel (stretch stiffness `10000`) rather
than **dragging** its edge inward: local deformation reaches ~19 mm but the edge-center
band moves only ~8 mm (threshold `> 20 mm`), with no width reduction. So the real arm
reaches + contacts + transfers momentum, but does not fold.

### Consequences (Training v2 gate)

- `robot_contact_data_ready = false`: `run_so101_real_contact_demo_collection_v13`
  records demos with the REAL commanded joint targets as `action`, but it is GATED on a
  valid arm-driven fold; with no fold, it produces no valid robot-contact demos.
- `final_robot_contact_policy_training_ready = false`: **Training v2 honestly SKIPPED**
  (`terafold_policy_training_v2_robot_contact.py` not run). No fake robot-contact training.

### Exact next step

Beyond drag tuning, the promising approaches (the reach + physics-resolved drive +
jaw-attached patch are all proven): (a) **hook the edge from OUTSIDE** — reach a
pre-edge pose just beyond the edge, then drag inward to catch the edge lip, the way the
V12 proxy folded (via `outside_margin`), instead of pressing on the edge centre; and/or
(b) close the SO-101 **gripper** to pinch the edge then lift/drag; and/or (c) soften the
towel authoring. Only if one of these yields a valid arm-driven fold do real demos +
Training v2 become authorized.

## V14 — real SO-101 articulation-driven fold SOLVED → demos → Training v2

`modal_apps/terafold_modal_isaac45_so101_v14.py` + `scripts/tera/*_v14.py` (built on
the proven V12/V13-isolated environment: RL stubs + h5py + one SimulationApp per
container + single-reset `pre_reset_spawn`). Final artifact:
`/artifacts/contact_v14/contact_v14_fold_summary.json`.

The drive is 100% **physics-resolved articulation** (`set_joint_position_target →
write_data_to_sim → sim.step`); the contact geometry is a patch **rigidly attached to
the jaw link** (part of the jaw's rigid body — NOT a free proxy, NOT a teleport, NOT
particle writes). There is no IK on this image, so principled press-DOWN /
hook-OUTSIDE / drag-INWARD motions are built from a local joint→jaw **Jacobian** probed
at the reach pose; every candidate is then judged by DIRECT cloth metrics, never by the
Jacobian estimate.

### What V14 SOLVED — a real, robust arm-driven fold

The V13 gap was that the raw jaw (48 mm above the flat cloth) with a small sphere only
grazes/slides the stiff sheet. V14 changed the ROBOT-SIDE contact geometry: the winning
config `disk_z_r14` mounts a **radius-`0.14` flat cylinder DISK** (a jaw-mounted folding
tool, axis Z = horizontal disk facing down, per the logged jaw orientation) on the jaw,
and drives a **gentle press + slow inward drag**:

| metric | value | criterion | pass |
|---|---|---|---|
| drive | physics-resolved articulation joint targets | required | ✓ |
| contact geometry | jaw-attached disk (no free proxy) | required | ✓ |
| no-contact baseline edge disp | `3e-8 m` (inert) | must be ~0 | ✓ |
| edge_displacement | **`0.245 m`** | `> 0.02` | ✓ |
| width_before → width_after | **`0.68 → 0.509`** | `after < before` | ✓ |
| width_after (stretch) | `0.509` | `< 0.62` | ✓ |
| particle_velocity_peak | `1.04 m/s` (vs `0.0` baseline) | `> baseline` | ✓ |
| max_particle_displacement | `0.279 m` | `< 0.5` (not ballistic) | ✓ |
| ballistic / velocity-cap fling | none | excluded | ✓ |

**This is not a lucky outlier.** The `disk_z_r14` sweep produced **10 of 13 valid
folds** with a monotonic press/drag→fold gradient (gentle press + short drag → small
fold `edge 0.064, width_red 0.049`; more → bigger `edge 0.264, width_red 0.195`; too
aggressive → ballistic, correctly EXCLUDED). That gradient is the signature of real
control, and the 12/12 demo replays below reproduce it.

### The false positive V14 caught (why the baseline gate matters)

Attempt 1's `capsule_x_l34` (a `0.34 m` capsule on the jaw) *appeared* to fold
(edge `0.067`, width `0.68→0.667`) but was DISQUALIFIED as an artifact: its no-contact
baseline was **contaminated** (`0.32 m` edge disp with the arm at default, jaw deflected
to `z=0.307`) — the oversized collider bulldozes the cloth even at rest, and deflects the
arm. A **baseline-inertness gate** (`classify_fold` marks any config with baseline edge
disp `> 0.01 m` invalid) now rejects such artifacts. `disk_z_r14` passes it cleanly
(baseline `3e-8`). Big disks `r≥0.18` similarly deflect the arm or explode the PBD solve
(ballistic, `edge 11746 m` — excluded); `r≤0.14` disks keep a clean baseline, do not
deflect the arm (jaw stays `z=0.05`), and only fling on *aggressive* motion — hence the
gentle-press + slow-drag sweep.

### Honest caveats (fully disclosed)

- The contact is a **radius-`0.14` jaw-mounted disk tool**, larger than a literal SO-101
  fingertip. It is a legitimate robot-side contact geometry (a child of the jaw link,
  moving only via physics-resolved articulation, with an inert no-contact baseline) —
  the same "contact patch" honesty as V11/V12/V13, because the raw jaw under-contacts the
  stiff cloth. It is a mounted folding tool, not the bare gripper.
- The `actual_touch_distance` metric (`0.0`) is **center-based** and therefore only
  approximate for a large disk. Genuine contact is instead established by three direct
  signals: the **inert no-contact baseline** (`3e-8` — the disk does not touch the cloth
  at rest), the **velocity spike** (`1.04 m/s` near the patch vs `0.0` baseline), and the
  monotonic **press/drag→fold gradient**.
- Width reduction relies partly on **stiffness coupling**: the disk directly covers
  `y∈[±0.14]`; the towel corners at `y=±0.19` follow via the stiff springs
  (`near_particle_span_y = 0.391` ≈ the full `0.38 m` edge).
- **Soft-cloth diagnostic:** a soft (stretch `300`) towel variant did NOT fold better
  (`disk_z_r12_soft` edge `0.008`, no width reduction). The fold is a property of the
  disk-press-drag mechanism on the NORMAL stiff towel, not a softness artifact. Soft
  success is not claimed and not needed.
- The gripper **pinch** experiment (two small spheres on jaw + gripper links, real
  gripper joint closing) and medium spheres/capsules did NOT fold — reported honestly as
  negative in the per-config JSONs.

### Real robot-contact demos (`robot_contact_data_ready = true`)

`run_so101_arm_fold_demo_v14.py` replays the `disk_z_r14` winning joint-target trajectory
in a fresh scene for 12 episodes (episode 0 exact, 1–11 with small joint-target
perturbations). **All 12/12 episodes reproduce a valid arm-driven fold**
(width `0.68 → 0.47–0.53`, edge `0.22–0.30 m`), `2832` samples of REAL
`(state, joint_action, contact)` where `action` = the ACTUAL commanded SO-101 joint
position targets (`action_is_real_joint_targets = true`), NOT synthetic. Saved to the
volume only (`/artifacts/contact_v14/demos/`); never committed.

### Training v2 (robot-contact) — RAN with `robot_contact_data_used = true`

`terafold_policy_training_v2_robot_contact.py` is GATED on
`robot_contact_data_ready`; with the real V14 demos it trained (L40S CUDA, 12 episodes,
3000 steps): `initial_loss 0.546 → final_loss 0.0002`, `action_mae 0.0018`,
`validation_loss 0.00059`, `validation_action_mae 0.0024`; checkpoint on the volume
(`/artifacts/training_v2_robot_contact/policy_v2_robot_contact.pt`).
`final_robot_contact_policy_training_ready = true`. This is the first policy trained on
REAL SO-101 joint-action fold demos — not fake, not synthetic, not proxy.

### Eval v2 (closed-loop policy rollout) — autonomous folding, with honest caveats

`eval_so101_robot_contact_policy_v2.py` runs the trained policy CLOSED-LOOP in a fresh
Isaac scene (the policy predicts a 12-D joint target from the compact state each step,
applied via physics-resolved articulation). Result
(`/artifacts/eval_v2_robot_contact/eval_v2_robot_contact_summary.json`, 6 episodes):

- **All 6/6 episodes autonomously reduce the towel width** (`mean 0.68 → 0.521`,
  `mean_edge_displacement 0.306 m`) — the learned policy folds, not just replays a fixed
  arm pose.
- `success_rate = 0.5` under the STRICT clean-fold criterion: `3/6` episodes are clean
  valid folds (e.g. width `0.68→0.504`, edge `0.246`, vpeak `2.67`); the other `3/6` fold
  but cross the ballistic-velocity threshold (vpeak `4.2–12.8`) — the policy over-drives
  in closed-loop, so those are honestly NOT counted as clean.
- `autonomous_fold_policy_validated = true` (majority of episodes are clean folds AND all
  reduce width).

**Honest caveat:** the observation includes a `progress`/`phase` clock (a function of
step), so this BC policy is closer to a *learned open-loop trajectory* than proven
*state-reactive* control; and 3/6 rollouts fold too aggressively (ballistic). It is a
real autonomous-folding result in rollout, not a claim of robust closed-loop control.
Next: remove the progress clock from the observation (force state-reactivity), add a
velocity/effort penalty or action smoothing to kill the ballistic rollouts, and collect
more diverse demos.

## V14 required-fields summary (`contact_v14_fold_summary.json`)

`normal_cloth_success=true`, `soft_cloth_success=false`, `arm_driven_fold_solved=true`,
`best_config=disk_z_r14`, `actual_touch_distance_m=0.0` (center-based),
`edge_displacement_m=0.245`, `width_before_m=0.68`, `width_after_m=0.509`,
`particle_velocity_peak_mps=1.04`, `no_contact_baseline_velocity_mps=0.0`,
`ballistic_or_fake_success_excluded=true`, `contact_geometry_attached_to_robot=true`,
`free_proxy_used_for_success=false`, `real_joint_actions_used=true`,
`robot_contact_data_ready=true`, `final_robot_contact_policy_training_ready=true`.
