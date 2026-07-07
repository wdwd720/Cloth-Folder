# TeraFold Modal Autonomy Status

_Autonomous Modal readiness report. Honest evidence only._

- **Git commit:** `dd6ac9a` (branch `main`, pushed to `origin`)
- **Date:** 2026-07-06
- **Modal workspace:** `terarobotics`
- **Modal Volume:** `terafold-artifacts`

## Headline

- `training_infra_ready = true`
- `final_robot_contact_policy_training_ready = false`

Training infrastructure on Modal is proven end-to-end (GPU + volume + the
repo's own training code path). A final robot-contact folding policy is **not**
authorized: cloth/rigid contact transfer is still unproven, and we additionally
found that the current Isaac Sim on Modal (6.0.1) has removed the particle-cloth
API the Tera stack depends on.

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
| Final robot-contact policy training | NOT ALLOWED | contact mechanics unproven | gated on contact transfer |

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

**Therefore the contact blocker is now concrete and fixable-in-principle:** the
towel must be re-authored with a realistic total mass (fix the density/mass bug)
and softer springs before any rigid-contact fold can be expected. V10 tests this
directly.

## Are contact mechanics solved?

No. V8 evidence (from Brev): collider actual touch achieved (<0.01 m) but towel
did not move meaningfully. This is the central unsolved blocker and it gates any
final robot-contact policy training.

## Is final robot-contact training allowed?

No. `final_robot_contact_policy_training_ready = false`. Only synthetic/assisted
training-infra work is authorized until contact transfer is proven with metrics
(edge displacement > 0.02 m, best_width < 0.65, valid rigid/kinematic contact).

## Exact next command/job to run

Cloth on Modal is proven and the blocker is localized to collider→particle
contact transfer. The next experiment is V10: does a **dynamic, velocity-driven**
rigid pusher (instead of a teleported kinematic sphere) transfer lateral motion
to the resting cloth?

```
# Re-confirm the current proven state (cloth runs; V9 root-cause):
python3 -m modal run modal_apps/terafold_modal_isaac45_cloth_smoke.py
python3 -m modal volume get terafold-artifacts \
  /modal_isaac45_cloth_smoke/modal_isaac45_cloth_smoke_v1_result.json ./r.json
cat r.json   # inspect contact_metrics + v9_particle_sim

# Then build + run V10 (dynamic rigid pusher) on the same Isaac 4.5.0 image.
```

Success metrics for V10 contact transfer: `edge_displacement > 0.02 m`,
`best_width < 0.65`, particle velocities change near contact, via a valid
rigid/kinematic contact (not by directly writing particle positions).
