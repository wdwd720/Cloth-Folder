# TeraFold Modal Autonomy Status

_Autonomous Modal readiness report. Honest evidence only._

- **Branch:** `main`, pushed to `origin` (see `git log` for the latest commit)
- **Date:** 2026-07-06
- **Modal workspace:** `terarobotics`
- **Modal Volume:** `terafold-artifacts`
- **Modal apps:** `terafold_policy_training_v0.py` / `_v1.py` (Track B training),
  `terafold_modal_repo_isaac_smoke.py` (6.0.1 infra),
  `terafold_modal_isaac45_cloth_smoke.py` (4.5.0 cloth + V9),
  `terafold_modal_isaac45_contact_v10.py` (V10 mass), `_contact_v11.py` (V11
  physics-drive SOLVE), `terafold_modal_isaac45_so101_v12.py` (V12 SO-101 port),
  `terafold_policy_training_v2_robot_contact.py` (gated, skipped)

## Headline

- `training_infra_ready = true`
- `contact_transfer_mechanism_solved = true` (V11 — physics-driven **primitive**
  collider moves + partially folds the cloth; teleport does not)
- `so101_contact_transfer_solved = false` (V12 — the physics-drive test on the
  real SO-101 scene did not execute; blocked by an IsaacLab RL-extension conflict)
- `robot_contact_data_ready = false` (no real robot-joint-action contact demos)
- `final_robot_contact_policy_training_ready = false` (Training v2 honestly
  SKIPPED; no fake robot-contact training)

Training infrastructure on Modal is proven end-to-end. The **contact-transfer
mechanism is solved** (V11: `RigidPrim.set_world_poses` gives the kinematic body
a real swept velocity → moves the resting cloth, edge `0.081 m`, width
`0.68 → 0.63`; a USD-transform teleport does not). V12 confirmed the real SO-101
arm scene **builds on Modal** (leisaac + downloaded `so101_follower.usd`) and that
the same physics-drive fix is the correct port (the V8 SO-101 proxies were
teleported — the exact V11 defect). But the V12 physics-drive execution is
**blocked** by an IsaacLab packaging issue (see below), so no real robot-contact
demonstrations were produced and Training v2 was honestly skipped. (Isaac Sim
6.0.1 lacks the particle-cloth API; all cloth work runs on Isaac Sim 4.5.0.)

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
| V12 SO-101 fingertip physics-drive test | BLOCKED | contact-drive scripts fatally fail at SimulationApp startup: `isaaclab_rl` ext → `ModuleNotFoundError: rl_games` | IsaacLab RL-extension packaging conflict |
| V12 robot-contact demos | NOT PRODUCED | physics-drive test blocked → 0 demos | gated on V12 execution |
| Training v2 (robot-contact) | SKIPPED (honest) | `training_v2_robot_contact_summary.json`: `robot_contact_data_used=false`, reason "V12 robot-contact demos not ready" | no valid robot-contact data |
| Final robot-contact policy training | ALLOWED at mechanism level | V11 valid rigid contact transfer with metrics | remaining: SO-101 gripper drive + full fold + real demos |

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

**For the real SO-101 robot: NO.** `final_robot_contact_policy_training_ready =
false`. V12 could not produce valid robot-contact demonstrations (the SO-101
physics-drive test is blocked, see below), so Training v2 was honestly SKIPPED
(`robot_contact_data_used = false`). No fake robot-contact training was run.
The physics-drive MECHANISM is proven (V11, primitive collider), but that is not
the same as a trainable SO-101 robot-contact policy.

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

## Exact next command/job to run

Resolve the `isaaclab_rl` conflict, then run the V11 mechanism on the SO-101
fingertip proxy. Recommended, in order of cleanliness:
1. Provide lightweight **stub packages** for `rl_games`/`rsl_rl`/`stable_baselines3`/
   `skrl` (a `sys.meta_path` finder returning dummy modules) so `isaaclab_rl`
   loads without pulling torch — install it at the top of each contact-drive
   script before `AppLauncher`.
2. And/or **isolate each SO-101 Isaac script in its own Modal container** (fresh
   kit config) — share state (towel USD, JSONs, demos) via the `terafold-artifacts`
   volume instead of container-local `/workspace`.
3. Then re-run `terafold_modal_isaac45_so101_v12.py`; expect
   `test_so101_fingertip_physics_drive_v12.py` to show teleport-fails vs
   physics-`set_world_poses`-works on the SO-101 fingertip proxy.
4. If it moves the cloth, extend to a full arm-articulation fold (needs the V6
   reach solution, also re-runnable on Modal) to generate REAL (state, joint_action)
   contact demos → `training_v2_robot_contact` with `robot_contact_data_used=true`.

Success metrics (met by V11 at the primitive level, to be reproduced on SO-101):
`actual_touch < 0.01 m`, `edge_displacement > 0.02 m`, `best_width < 0.65`,
particle-velocity spike — via valid rigid contact, NEVER by writing particle
positions (`set_points`), and NEVER counting a ballistic fling.
